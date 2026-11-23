"""Model 3 (pre-date PC network): faithful reproduction plus switchable fixes.

Every switch in `Config` corresponds to one problem found in pc-mle/pc_mle_analysis.py:

  leak        the post-date column sincere_o is excluded ("sinsere_o" typo in their denylist)
  forced      the top-5 mutual-information features are forced to be parents of match
  recode      field is a 260-state free-text column and expected_num_interested_in_me is
              78% missing (encoded as a 19-state categorical); recode field with the
              18-category field_cd, drop the 78%-missing column, drop wave (an event id),
              keep d_age (the one genuine pre-date difference column)
  mi_correct  mutual information is biased upward by cardinality; subtract the
              Miller-Madow term (r_x - 1)(r_y - 1) / 2N before ranking
  smooth      MLE -> BDeu (unseen parent configurations otherwise get 0/0 -> uniform,
              seen-once configurations get probability 0 or 1)
"""
from __future__ import annotations

import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402

import sys
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, str(SCRIPT_DIR))
import pc_mle_analysis as P  # noqa: E402  (their module, unchanged)

DATA = str(DATA) + "/"
TARGET = "match"


# ----------------------------------------------------------------------------- data
def load() -> pd.DataFrame:
    df = P.load_dataset(DATA + "speeddating.csv")
    raw = pd.read_csv(DATA + "annavictoria_data.csv", encoding="latin1")
    assert len(df) == len(raw)
    assert (df["wave"].astype(float).values == raw["wave"].values).all()
    assert (pd.to_numeric(df["decision"]).values == raw["dec"].values).all()
    assert (pd.to_numeric(df["decision_o"]).values == raw["dec_o"].values).all()
    for c in ["iid", "pid", "field_cd"]:
        df["_" + c] = raw[c].values
    return df


@dataclass(frozen=True)
class Config:
    name: str
    leak: bool = False        # True = sincere_o still in the candidate pool (original bug)
    forced: bool = True       # True = top-5 MI features forced as parents of match
    recode: bool = False
    mi_correct: bool = False
    smooth: float = 0.0       # 0 = MLE (pgmpy semantics); >0 = BDeu equivalent sample size
    max_features: int = 10
    max_cond_vars: int = 3
    alpha: float = 0.05


ORIGINAL = Config("original", leak=True)


def candidates(df: pd.DataFrame, cfg: Config) -> list[str]:
    cols = P.feature_columns(df, "pre_date")           # their denylist (with the typo)
    cols = [c for c in cols if not c.startswith("_")]
    if not cfg.leak:
        cols = [c for c in cols if c != "sincere_o"]
    if cfg.recode:
        cols = [c for c in cols if c not in ("field", "expected_num_interested_in_me", "wave")]
        cols += ["_field_cd", "d_age"]
    return cols


# ----------------------------------------------------------------------------- selection
def mi_rank(train: pd.DataFrame, y: pd.Series, correct: bool) -> list[str]:
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(train, y, discrete_features=True, random_state=P.RANDOM_STATE)
    if correct:
        n = len(y)
        r = np.array([train[c].nunique() for c in train.columns])
        mi = mi - (r - 1) * (y.nunique() - 1) / (2 * n)
    ranked = sorted(zip(train.columns, mi), key=lambda kv: (-float(kv[1]), kv[0]))
    return [c for c, _ in ranked]


# ----------------------------------------------------------------------------- network
class Net:
    """Discrete BN with CPTs as dense arrays; prediction by the Markov-blanket formula."""

    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.parents = {v: [a for a, b in edges if b == v] for v in self.nodes}

    def fit(self, data: pd.DataFrame, ess: float):
        """Sparse CPTs: only parent configurations seen in training are stored.

        Unseen configurations are uniform under both estimators (MLE as in pgmpy's
        DiscreteMLE; BDeu because all its pseudo-counts are equal), so nothing is lost.
        """
        self.ess = ess
        self.states = {v: np.sort(data[v].unique()) for v in self.nodes}
        self.card = {v: len(s) for v, s in self.states.items()}
        idx = {v: np.searchsorted(self.states[v], data[v].to_numpy()) for v in self.nodes}
        self.tab = {}
        for v in self.nodes:
            pa, r = self.parents[v], self.card[v]
            key = self._key(idx, pa)
            keys, inv = np.unique(key, return_inverse=True)
            counts = np.zeros((len(keys), r))
            np.add.at(counts, (inv, idx[v]), 1)
            if ess > 0:   # BDeu: alpha = ess / (q * r) per cell
                q = float(np.prod([float(self.card[u]) for u in pa])) if pa else 1.0
                a = ess / (q * r)
                prob = (counts + a) / (counts.sum(1, keepdims=True) + a * r)
            else:
                prob = counts / counts.sum(1, keepdims=True)
            self.tab[v] = (keys, prob)
        return self

    def _key(self, idx, pa):
        """Mixed-radix code of a parent configuration (Python ints avoid overflow)."""
        n = len(next(iter(idx.values())))
        if not pa:
            return np.zeros(n, dtype=object)
        key = np.zeros(n, dtype=object)
        for u in pa:
            key = key * self.card[u] + idx[u].astype(object)
        return key

    def _index(self, data, v):
        """State index of each row; -1 for a state never seen in training."""
        x = data[v].to_numpy()
        i = np.searchsorted(self.states[v], x).clip(0, self.card[v] - 1)
        return np.where(self.states[v][i] == x, i, -1)

    def _lookup(self, data, v, override=None):
        """(row of the stored table or -1, state index or -1) for every data row."""
        pa = self.parents[v]
        ids = {u: (np.full(len(data), override[1]) if override and u == override[0]
                   else self._index(data, u)) for u in pa + [v]}
        bad = np.zeros(len(data), bool)
        for u in pa:
            bad |= ids[u] < 0
        key = self._key({u: np.where(ids[u] < 0, 0, ids[u]) for u in pa} or
                        {"_": np.zeros(len(data), int)}, pa)
        keys, _ = self.tab[v]
        pos = np.searchsorted(keys, key) if len(keys) else np.zeros(len(data), int)
        pos = np.clip(pos, 0, len(keys) - 1)
        found = (keys[pos] == key) & ~bad
        return np.where(found, pos, -1), ids[v]

    def _prob(self, data, v, override=None):
        row, st = self._lookup(data, v, override)
        _, prob = self.tab[v]
        ok = (row >= 0) & (st >= 0)
        p = np.full(len(data), 1.0 / self.card[v])
        p[ok] = prob[row[ok], st[ok]]
        return p

    def predict(self, data, target=TARGET, fallback=None):
        children = [c for c in self.nodes if target in self.parents[c]]
        k = int(np.where(self.states[target] == 1)[0][0])
        terms = []
        for s in range(self.card[target]):
            w = self._prob(data.assign(**{target: self.states[target][s]}), target)
            for c in children:
                w = w * self._prob(data, c, override=(target, s))
            terms.append(w)
        terms = np.array(terms)
        z = terms.sum(0)
        p = np.divide(terms[k], z, out=np.full(len(data), np.nan), where=z > 0)
        nonfinite = int(np.isnan(p).sum())
        if fallback is not None:
            p = np.where(np.isnan(p), fallback, p)
        return p, nonfinite

    def unseen_rows(self, data, target=TARGET):
        """Rows whose parent configuration for `target` never occurred in training."""
        if not self.parents[target]:
            return np.zeros(len(data), bool)
        row, _ = self._lookup(data, target)
        return row < 0


def learn_structure(train: pd.DataFrame, cfg: Config, forced_parents):
    from pgmpy.estimators import PC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dag = PC(train).estimate(variant="stable", ci_test="chi_square",
                                 significance_level=cfg.alpha,
                                 max_cond_vars=cfg.max_cond_vars, return_type="dag",
                                 show_progress=False)
    edges = list(dag.edges())
    learn_structure.dropped = 0
    if cfg.forced:
        import networkx as nx
        g = nx.DiGraph(edges)
        g.add_nodes_from(train.columns)
        forced = {(p, TARGET) for p in forced_parents}
        dropped = 0
        for p in forced_parents:
            if g.has_edge(TARGET, p):          # PC oriented it the other way
                g.remove_edge(TARGET, p)
                dropped += 1
            g.add_edge(p, TARGET)
            # pgmpy's add_edge refuses a cycle (the original script stops there);
            # we drop one learned (non-forced) edge on the cycle instead.
            while not nx.is_directed_acyclic_graph(g):
                cyc = nx.find_cycle(g)
                a, b = next(e[:2] for e in cyc if e[:2] not in forced)
                g.remove_edge(a, b)
                dropped += 1
        learn_structure.dropped = dropped
        edges = list(g.edges())
    return edges


# ----------------------------------------------------------------------------- one fit
def encode(cfg: Config, tr_raw: pd.DataFrame, te_raw: pd.DataFrame):
    """Their preprocessing (median imputation, 5 quantile bins, categorical codes)."""
    pre = P.DiscretePreprocessor().fit(tr_raw)
    return pre.transform(tr_raw), pre.transform(te_raw)


def fit_predict(df, cfg: Config, tr_idx, te_idx):
    cols = candidates(df, cfg)
    y = P._encode_target(df[TARGET])
    tr_raw, te_raw = df.iloc[tr_idx][cols], df.iloc[te_idx][cols]
    tr, te = encode(cfg, tr_raw, te_raw)
    ranked = mi_rank(tr, y.iloc[tr_idx], cfg.mi_correct)
    sel = ranked[: cfg.max_features]
    trn, ten = tr[sel].copy(), te[sel].copy()
    trn[TARGET], ten[TARGET] = y.iloc[tr_idx].to_numpy(), y.iloc[te_idx].to_numpy()
    edges = learn_structure(trn, cfg, sel[:5])
    net = Net(trn.columns, edges).fit(trn, cfg.smooth)
    prior = float(trn[TARGET].mean())
    p_tr, _ = net.predict(trn, fallback=prior)
    p_te, nonfin = net.predict(ten, fallback=prior)
    return dict(sel=sel, edges=edges, net=net, p_tr=p_tr, p_te=p_te,
                y_tr=trn[TARGET].to_numpy(), y_te=ten[TARGET].to_numpy(),
                nonfinite=nonfin, unseen=net.unseen_rows(ten).mean(),
                parents=net.parents[TARGET], dropped=learn_structure.dropped,
                n_configs=int(np.prod([net.card[p] for p in net.parents[TARGET]] or [1])))


def tune_threshold(y, p):
    """Their rule: the training-set probability that maximises training F1."""
    best_t, best_f = 0.5, -1
    for t in np.unique(np.concatenate(([0.0, 1.0], p))):
        yhat = (p >= t).astype(int)
        if yhat.min() == yhat.max():
            continue
        f = f1_score(y, yhat, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def metrics(y, p, t):
    pc = p   # sklearn clips at machine epsilon, as in their evaluate_model
    yhat = (p >= t).astype(int)
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                log_loss=log_loss(y, pc, labels=[0, 1]), brier=brier_score_loss(y, p),
                f1=f1_score(y, yhat, zero_division=0), acc=float((yhat == y).mean()))


# ----------------------------------------------------------------------------- splits
def their_split(df):
    y = P._encode_target(df[TARGET])
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=P.TEST_SIZE, stratify=y, random_state=P.RANDOM_STATE)
    return tr, te


def wave_folds(df, k=5):
    """5 folds grouped by wave: no person and no mirrored row is ever on both sides."""
    waves = df["wave"].astype(float).to_numpy()
    return list(GroupKFold(n_splits=k).split(df, groups=waves))

"""Candidate repaired structures for the pre-date network, all on the same fold-wise
10 variables (cardinality-corrected MI, training rows only) with BDeu CPTs.

  nb        match -> every selected variable
  tan       nb + a Chow-Liu tree over the variables given match (Friedman et al. 1997)
  pc_nb     PC's edges among the variables (edges touching match dropped) + match -> every
            variable: keeps Model 3's PC step, but stops PC from deciding match's family
  usage: variants.py <name> [<name> ...]   (results cached per name/split)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, itertools, warnings; sys.path.insert(0, str(HERE))
from dataclasses import replace
from pathlib import Path
import numpy as np, networkx as nx
import lib

df = lib.load()
cfg = replace(lib.ORIGINAL, name="fixed", leak=False, recode=True, mi_correct=True,
              forced=False, smooth=5.0)
y_all = lib.P._encode_target(df[lib.TARGET])
out = Path(str(OUT) + "/variants"); out.mkdir(parents=True, exist_ok=True)


def cmi(d, a, b, c="match"):
    """I(a; b | c) in nats from counts."""
    tab = d.groupby([c, a, b]).size()
    n = tab.sum(); tot = 0.0
    for cv, sub in tab.groupby(level=0):
        sub = sub.droplevel(0); nc = sub.sum()
        pa = sub.groupby(level=0).sum() / nc; pb = sub.groupby(level=1).sum() / nc
        for (x, z), cnt in sub.items():
            pxz = cnt / nc
            tot += (nc / n) * pxz * np.log(pxz / (pa[x] * pb[z]))
    return tot


def structure(kind, trn, sel):
    nb = [("match", v) for v in sel]
    if kind == "nb":
        return nb
    if kind == "tan":
        g = nx.Graph()
        for a, b in itertools.combinations(sel, 2):
            g.add_edge(a, b, weight=cmi(trn, a, b))
        tree = nx.maximum_spanning_tree(g)
        return nb + list(nx.bfs_edges(tree, sel[0]))          # orient away from the root
    if kind == "pc_nb":
        edges = lib.learn_structure(trn, cfg, [])
        return nb + [(a, b) for a, b in edges if "match" not in (a, b)]
    raise ValueError(kind)


def run(kind, k=10, ess=5.0):
    tag = f"{kind}_k{k}_e{ess:g}"
    res = {}
    for split in ["their", "wave"]:
        f = out / f"{tag}_{split}.json"
        if f.exists():
            res[split] = json.load(open(f)); continue
        folds = [lib.their_split(df)] if split == "their" else lib.wave_folds(df)
        P, Y, rec = [], [], []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            cols = lib.candidates(df, cfg)
            tr, te = lib.encode(cfg, df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
            ytr, yte = y_all.iloc[tr_idx].to_numpy(), y_all.iloc[te_idx].to_numpy()
            sel = lib.mi_rank(tr, y_all.iloc[tr_idx], True)[:k]
            trn, ten = tr[sel].assign(match=ytr), te[sel].assign(match=yte)
            edges = structure(kind, trn, sel)
            net = lib.Net(trn.columns, edges).fit(trn, ess)
            p_tr, _ = net.predict(trn, fallback=ytr.mean())
            p, _ = net.predict(ten, fallback=ytr.mean())
            t = lib.tune_threshold(ytr, p_tr)
            m = lib.metrics(yte, p, t)
            rec.append(dict(fold=fi, sel=sel, edges=[list(e) for e in edges], threshold=t,
                            **{a: float(b) for a, b in m.items()}))
            P.append(p); Y.append(yte)
        P, Y = np.concatenate(P), np.concatenate(Y)
        pm = lib.metrics(Y, P, 0.5)
        res[split] = dict(kind=kind, k=k, ess=ess, folds=rec, pooled_auc=float(pm["auc"]),
                          pooled_log_loss=float(pm["log_loss"]), pooled_brier=float(pm["brier"]),
                          fold_auc=[r["auc"] for r in rec], fold_f1=[r["f1"] for r in rec])
        json.dump(res[split], open(f, "w"), indent=1)
        np.save(out / f"{tag}_{split}_p.npy", np.vstack([P, Y]))
    print(tag, {s: (round(r["pooled_auc"], 4), round(r["pooled_log_loss"], 4)) for s, r in res.items()},
          "wave fold AUCs", np.round(res["wave"]["fold_auc"], 3), flush=True)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        kind, *rest = arg.split(":")
        k = int(rest[0]) if rest else 10
        ess = float(rest[1]) if len(rest) > 1 else 5.0
        run(kind, k, ess)

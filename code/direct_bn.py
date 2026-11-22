"""Direct-edge Bayesian network (plan Step 3).

Each decision gets, as parents, the ratings made by the person who decides:

    participant's view of partner  ->  decision
    partner's view of participant  ->  decision_o
    P(match) = P(decision) * P(decision_o)

CPTs are fit with pgmpy's BayesianEstimator (BDeu prior). Parent sets and the
prior strength are chosen on a validation split carved out of TRAIN (by
participant), never on the test set.

Run from code/:
    python direct_bn.py                 # participant split (default)
    python direct_bn.py --split row     # same rows as the original notebook
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from data_prep import PARTNER_VIEW, SELF_VIEW, prepare
from evaluate import baselines, evaluate, save_results, tune_threshold

# Starting parents (the five used in the scratch test).
START = {
    "decision": ["like", "guess_prob_liked", "attractive_partner", "funny_partner",
                 "shared_interests_partner"],
    "decision_o": ["like_o", "prob_o", "attractive_o", "funny_o", "shared_interests_o"],
}
# Extra candidates tried one at a time (same viewpoint, plus pair-level context).
CANDIDATES = {
    "decision": [c for c in SELF_VIEW if c not in START["decision"]] + ["samerace", "met", "interests_correlate"],
    "decision_o": [c for c in PARTNER_VIEW if c not in START["decision_o"]] + ["samerace", "met", "interests_correlate"],
}
MAX_PARENTS = 5
ESS_GRID = [1, 5, 10, 20, 50]


def fit(data, parents, state_names, ess):
    """Fit a BN where `target` has `parents` (one family per decision)."""
    edges = [(p, t) for t, ps in parents.items() for p in ps]
    nodes = sorted({n for e in edges for n in e})
    model = DiscreteBayesianNetwork(edges)
    est = BayesianEstimator(model, data[nodes], state_names={n: state_names[n] for n in nodes})
    model.add_cpds(*est.get_parameters(prior_type="BDeu", equivalent_sample_size=ess))
    assert model.check_model()
    return model


def predict(model, data, target):
    """P(target = 1 | its parents), read straight from the CPT (all parents are observed)."""
    cpd = model.get_cpds(target)
    idx = tuple(data[v].to_numpy() for v in cpd.variables[1:])
    return cpd.values[1][idx]


def val_score(train, val, target, parents, state_names, ess):
    m = fit(train, {target: parents}, state_names, ess)
    return log_loss(val[target], np.clip(predict(m, val, target), 1e-6, 1 - 1e-6))


def select(train, val, target, state_names, ess=10, log=print):
    """Greedy: drop parents that do not help, then add candidates that do (lower val log loss)."""
    parents = list(START[target])
    best = val_score(train, val, target, parents, state_names, ess)
    log(f"  {target}: start {parents}  val log loss {best:.4f}")
    improved = True
    while improved and len(parents) > 1:  # backward
        improved = False
        for p in list(parents):
            s = val_score(train, val, target, [q for q in parents if q != p], state_names, ess)
            if s < best - 1e-4:
                best, parents, improved = s, [q for q in parents if q != p], True
                log(f"    drop {p:26s} -> {best:.4f}")
                break
    improved = True
    while improved and len(parents) < MAX_PARENTS:  # forward
        improved = False
        scores = {c: val_score(train, val, target, parents + [c], state_names, ess)
                  for c in CANDIDATES[target] if c not in parents}
        c, s = min(scores.items(), key=lambda kv: kv[1])
        if s < best - 1e-4:
            best, parents, improved = s, parents + [c], True
            log(f"    add  {c:26s} -> {best:.4f}")
    # prior strength
    ess_scores = {e: val_score(train, val, target, parents, state_names, e) for e in ESS_GRID}
    ess_best = min(ess_scores, key=ess_scores.get)
    log(f"  {target}: final {parents}  ESS={ess_best}  val log loss {ess_scores[ess_best]:.4f}")
    return parents, ess_best


def main(split):
    train, test, state_names, _ = prepare(split=split)

    # Validation split inside train, by participant.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr_i, va_i = next(gss.split(train, groups=train["iid"]))
    tr, va = train.iloc[tr_i], train.iloc[va_i]

    print(f"== split: {split}   train {len(train)}  test {len(test)}  (selection: {len(tr)} / {len(va)})")
    chosen = {}
    for target in ["decision", "decision_o"]:
        chosen[target] = select(tr, va, target, state_names)

    rows = []
    for name, cfg in [("direct_bn_start", {t: (START[t], 10) for t in START}),
                      ("direct_bn_selected", chosen)]:
        p_tr, p_te = {}, {}
        for target, (parents, ess) in cfg.items():
            m = fit(train, {target: parents}, state_names, ess)  # refit on all of train
            p_tr[target], p_te[target] = predict(m, train, target), predict(m, test, target)
        p_tr["match"] = p_tr["decision"] * p_tr["decision_o"]
        p_te["match"] = p_te["decision"] * p_te["decision_o"]
        for target in ["decision", "decision_o", "match"]:
            t = tune_threshold(train[target], p_tr[target])
            rows.append(evaluate(name, target, test[target], p_te[target], t, split))
    for target in ["decision", "decision_o", "match"]:
        rows.append(baselines(train[target], test[target], target, split))

    table = save_results(rows)
    out = Path(__file__).resolve().parent.parent / "results" / f"direct_bn_parents_{split}.json"
    out.write_text(json.dumps({t: {"parents": p, "ess": e} for t, (p, e) in chosen.items()}, indent=2))

    cols = ["model", "target", "AUC", "log_loss", "threshold", "precision", "recall", "F1", "accuracy", "always_no_acc"]
    print(table[table.split == split][cols].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="participant", choices=["participant", "pair", "row"])
    main(ap.parse_args().split)

"""Reference predictors for the repaired pre-date model, same splits and preprocessing.

  constant   training match rate for everyone
  nb10       naive Bayes (match -> each feature) on the same 10 variables, BDeu ESS 5
  lr_all     L2 logistic regression on one-hot codes of every pre-date candidate
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json; sys.path.insert(0, str(HERE))
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import lib

df = lib.load()
cfg = replace(lib.ORIGINAL, name="fixed", leak=False, recode=True, mi_correct=True,
              forced=False, smooth=5.0)
y_all = lib.P._encode_target(df[lib.TARGET])
out = Path(str(OUT) + "/ablation")
res = {}
for split in ["their", "wave"]:
    folds = [lib.their_split(df)] if split == "their" else lib.wave_folds(df)
    preds = {k: [] for k in ["constant", "nb10", "lr_all"]}
    Y = []
    for tr_idx, te_idx in folds:
        cols = lib.candidates(df, cfg)
        tr, te = lib.encode(cfg, df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
        ytr, yte = y_all.iloc[tr_idx].to_numpy(), y_all.iloc[te_idx].to_numpy()
        preds["constant"].append(np.full(len(te_idx), ytr.mean()))
        sel = lib.mi_rank(tr, y_all.iloc[tr_idx], True)[:10]
        trn, ten = tr[sel].assign(match=ytr), te[sel].assign(match=yte)
        nb = lib.Net(trn.columns, [("match", v) for v in sel]).fit(trn, 5.0)
        preds["nb10"].append(nb.predict(ten, fallback=ytr.mean())[0])
        enc = OneHotEncoder(handle_unknown="ignore").fit(tr.astype(str))
        lr = LogisticRegression(C=0.05, max_iter=5000).fit(enc.transform(tr.astype(str)), ytr)
        preds["lr_all"].append(lr.predict_proba(enc.transform(te.astype(str)))[:, 1])
        Y.append(yte)
    Y = np.concatenate(Y)
    for k, v in preds.items():
        p = np.concatenate(v)
        m = lib.metrics(Y, p, 0.5)
        res[f"{k}_{split}"] = dict(auc=float(m["auc"]), log_loss=float(m["log_loss"]),
                                   brier=float(m["brier"]))
        np.save(out / f"ref_{k}_{split}_p.npy", np.vstack([p, Y]))
        print(k, split, {a: round(b, 4) for a, b in res[f"{k}_{split}"].items()}, flush=True)
json.dump(res, open(out / "references.json", "w"), indent=1)

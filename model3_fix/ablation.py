"""Apply the Model 3 fixes one at a time; evaluate on their split and on 5 wave-grouped folds."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, time; sys.path.insert(0, str(HERE))
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np, lib

df = lib.load()
STEPS = [
    lib.ORIGINAL,
    replace(lib.ORIGINAL, name="no_leak", leak=False),
    replace(lib.ORIGINAL, name="recode", leak=False, recode=True, mi_correct=True),
    replace(lib.ORIGINAL, name="pc_parents", leak=False, recode=True, mi_correct=True, forced=False),
    replace(lib.ORIGINAL, name="bdeu", leak=False, recode=True, mi_correct=True, forced=False, smooth=5.0),
]
out = Path(str(OUT) + "/ablation"); out.mkdir(parents=True, exist_ok=True)
only = sys.argv[1:] or [c.name for c in STEPS]
for cfg in STEPS:
    if cfg.name not in only:
        continue
    for split in ["their", "wave"]:
        f = out / f"{cfg.name}_{split}.json"
        if f.exists():
            continue
        t0 = time.time()
        folds = [lib.their_split(df)] if split == "their" else lib.wave_folds(df)
        P, Y, rec = [], [], []
        for k, (tr, te) in enumerate(folds):
            r = lib.fit_predict(df, cfg, tr, te)
            t = lib.tune_threshold(r["y_tr"], r["p_tr"])
            m = lib.metrics(r["y_te"], r["p_te"], t)
            pte = r["p_te"]
            rec.append(dict(fold=k, sel=r["sel"], parents=r["parents"], n_configs=r["n_configs"],
                            edges=[list(e) for e in r["edges"]], threshold=t, unseen=float(r["unseen"]),
                            frac01=float(np.mean((pte == 0) | (pte == 1))), nonfinite=r["nonfinite"], dropped=r["dropped"],
                            train_auc=float(lib.metrics(r["y_tr"], r["p_tr"], t)["auc"]), **m))
            P.append(pte); Y.append(r["y_te"])
            print(cfg.name, split, k, {x: round(float(m[x]), 4) for x in ("auc", "log_loss", "f1")},
                  "parents", r["parents"], "unseen", round(float(r["unseen"]), 3), flush=True)
        P, Y = np.concatenate(P), np.concatenate(Y)
        # pooled metrics across folds (threshold per fold already applied in fold records)
        pooled = lib.metrics(Y, P, 0.5)
        json.dump(dict(config=asdict(cfg), split=split, folds=rec,
                       pooled_auc=float(pooled["auc"]), pooled_log_loss=float(pooled["log_loss"]),
                       pooled_brier=float(pooled["brier"]), seconds=round(time.time() - t0, 1)),
                  open(f, "w"), indent=1)
        np.save(out / f"{cfg.name}_{split}_p.npy", np.vstack([P, Y]))

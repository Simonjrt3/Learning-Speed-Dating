import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json; sys.path.insert(0, str(HERE))
import numpy as np, lib
df = lib.load()
tr, te = lib.their_split(df)
r = lib.fit_predict(df, lib.ORIGINAL, tr, te)
t = lib.tune_threshold(r["y_tr"], r["p_tr"])
m = lib.metrics(r["y_te"], r["p_te"], t)
orig = json.load(open(str(OUT) + "/orig_f10.json"))
print("selected same:", r["sel"] == orig["selected"])
print("edges same:", sorted(map(list, r["edges"])) == sorted(orig["edges"]))
print("threshold", t, "vs", orig["threshold"])
print("auc", m["auc"], "vs", orig["metrics"]["roc_auc"], "| logloss", m["log_loss"], "vs", orig["metrics"]["log_loss"], "| f1", m["f1"], "vs", orig["metrics"]["f1"])
p = r["p_te"]
print("test rows: unseen config", r["unseen"].round(3), " p==0.5", np.mean(p == 0.5).round(3), " p in {0,1}", np.mean((p == 0) | (p == 1)).round(3))
ptr = r["p_tr"]; print("train rows p in {0,1}", round(float(np.mean((ptr == 0) | (ptr == 1))), 3), "train auc", round(float(lib.metrics(r["y_tr"], ptr, t)["auc"]), 3))

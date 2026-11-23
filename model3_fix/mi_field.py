"""Mutual information with match on the team's training rows: how much of field's comes from its
260 states? (Plug-in MI, Miller-Madow corrected MI, and field's MI with shuffled labels.)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json; sys.path.insert(0, str(HERE))
import numpy as np, lib
from sklearn.feature_selection import mutual_info_classif
df = lib.load(); tr, te = lib.their_split(df)
cols = lib.candidates(df, lib.ORIGINAL); y = lib.P._encode_target(df["match"])
a, _ = lib.encode(lib.ORIGINAL, df.iloc[tr][cols], df.iloc[te][cols])
mi = mutual_info_classif(a, y.iloc[tr], discrete_features=True, random_state=42)
r = np.array([a[c].nunique() for c in a.columns]); n = len(tr)
corr = mi - (r - 1) / (2 * n)
res = {a.columns[i]: dict(states=int(r[i]), mi=float(mi[i]), corrected=float(corr[i]),
                          rank_corrected=int((corr > corr[i]).sum() + 1)) for i in np.argsort(-mi)[:6]}
rng = np.random.default_rng(0)
null = [mutual_info_classif(a[["field"]], rng.permutation(y.iloc[tr].to_numpy()), discrete_features=True,
                            random_state=42)[0] for _ in range(20)]
res["field_null_mean"] = float(np.mean(null))
json.dump(res, open(str(OUT) + "/mi_field.json", "w"), indent=1)
print({k: (round(v["mi"] * 1000, 2) if isinstance(v, dict) else round(v * 1000, 2)) for k, v in res.items()})

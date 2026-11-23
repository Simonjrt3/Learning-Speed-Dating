"""Committed network, team's split: for how many test rows did the parent configuration of match
occur in training, and how often only in rows of the same participant?"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json; sys.path.insert(0, str(HERE))
from collections import defaultdict
import numpy as np, lib
df = lib.load(); tr, te = lib.their_split(df)
r = lib.fit_predict(df, lib.ORIGINAL, tr, te)
pa = r["parents"]; cols = lib.candidates(df, lib.ORIGINAL)
a, b = lib.encode(lib.ORIGINAL, df.iloc[tr][cols], df.iloc[te][cols])
owners = defaultdict(set)
for k, i in zip(map(tuple, a[pa].to_numpy()), df["_iid"].to_numpy()[tr]):
    owners[k].add(i)
kt, it = list(map(tuple, b[pa].to_numpy())), df["_iid"].to_numpy()[te]
seen = np.array([k in owners for k in kt])
same_only = np.array([k in owners and owners[k] == {i} for k, i in zip(kt, it)])
incl = np.array([k in owners and i in owners[k] for k, i in zip(kt, it)])
out = dict(seen=float(seen.mean()), same_only=float(same_only.mean()), incl_same=float(incl.mean()))
json.dump(out, open(str(OUT) + "/lookup.json", "w")); print(out)

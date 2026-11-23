"""Reproduce Model 3 exactly as committed, and record diagnostics."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, time, warnings
sys.path.insert(0, str(SCRIPT_DIR))
import numpy as np, pandas as pd
import pc_mle_analysis as P

data = sys.argv[1]; maxf = int(sys.argv[2]); tag = sys.argv[3]
t0 = time.time()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    res = P.run_experiment(data, mode="pre_date", max_features=maxf)
    msgs = [str(x.message) for x in w if "impossible" in str(x.message)]
m = res.model
cpd = m.get_cpds("match")
out = {
    "selected": list(res.selected_features),
    "match_parents": list(cpd.variables[1:]),
    "match_parent_configs": int(np.prod(cpd.cardinality[1:])),
    "markov_blanket": sorted(m.get_markov_blanket("match")),
    "edges": [list(e) for e in res.edges],
    "threshold": res.threshold,
    "metrics": res.metrics,
    "fallback_warnings": msgs,
    "seconds": round(time.time() - t0, 1),
}
json.dump(out, open(f"{OUT}/orig_{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "edges"}, indent=1))

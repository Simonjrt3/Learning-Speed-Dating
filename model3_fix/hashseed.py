"""Run the committed Model 3 script (their split, their features) under several hash seeds."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import json, os, subprocess, sys
res = []
for seed in ["1", "2", "3", "4", "5", "6"]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    p = subprocess.run([sys.executable, "-P", str(HERE / "run_orig.py"),
                        str(DATA / "speeddating.csv"), "10", f"seed{seed}"],
                       env=env, capture_output=True, text=True)
    if p.returncode != 0:
        err = [l for l in p.stderr.splitlines() if "Error" in l][-1:]
        res.append(dict(seed=seed, ok=False, error=err)); print(seed, "FAILED", err, flush=True); continue
    d = json.load(open(f"{OUT}/orig_seed{seed}.json"))
    res.append(dict(seed=seed, ok=True, edges=sorted(map(tuple, d["edges"])), auc=d["metrics"]["roc_auc"],
                    f1=d["metrics"]["f1"], parents=sorted(d["match_parents"])))
    print(seed, "auc", round(d["metrics"]["roc_auc"], 4), "n_edges", len(d["edges"]), flush=True)
ok = [r for r in res if r["ok"]]
summary = dict(seeds=len(res), failed=sum(not r["ok"] for r in res),
               distinct_edge_sets=len({tuple(r["edges"]) for r in ok}),
               aucs=sorted({round(r["auc"], 4) for r in ok}), runs=res)
json.dump(summary, open(str(OUT) + "/hashseed.json", "w"), indent=1, default=list)
print({k: v for k, v in summary.items() if k != "runs"})

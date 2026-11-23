import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import json, os, subprocess, sys
res = []
for seed in ["0", "1", "2", "3", "4", "5"]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    subprocess.run([sys.executable, "-P", str(HERE / "run_fixed_typo.py"),
                    str(DATA / "speeddating.csv"), f"seed{seed}"], env=env,
                   capture_output=True, text=True)
    d = json.load(open(f"{OUT}/fixedtypo_seed{seed}.json")); d["seed"] = seed; res.append(d)
    print(seed, d.get("ok"), d.get("auc"), d.get("f1"), flush=True)
ok = [r for r in res if r["ok"]]
s = dict(seeds=len(res), crashed=sum(not r["ok"] for r in res), aucs=sorted({round(r["auc"], 4) for r in ok}),
         f1s=sorted({round(r["f1"], 4) for r in ok}), distinct_edge_sets=len({json.dumps(r["edges"]) for r in ok}))
json.dump(dict(summary=s, runs=res), open(str(OUT) + "/hashseed_fixed.json", "w"), indent=1)
print(s)

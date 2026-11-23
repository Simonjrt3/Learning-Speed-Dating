"""Their script with only the column name corrected (sincere_o excluded), one hash seed per call."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, warnings; sys.path.insert(0, str(SCRIPT_DIR))
warnings.simplefilter("ignore")
import pc_mle_analysis as P
P._POST_DATE_BASE_COLUMNS = P._POST_DATE_BASE_COLUMNS | {"sincere_o"}
tag = sys.argv[2]
try:
    r = P.run_experiment(sys.argv[1], mode="pre_date", max_features=10)
    out = dict(ok=True, auc=r.metrics["roc_auc"], f1=r.metrics["f1"], edges=sorted(map(list, r.edges)),
               selected=list(r.selected_features))
except ValueError as e:
    out = dict(ok=False, error=str(e))
json.dump(out, open(f"{OUT}/fixedtypo_{tag}.json", "w"), indent=1)
print(tag, out.get("auc"), out.get("f1"), out.get("error", ""))

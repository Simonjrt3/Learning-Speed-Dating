"""Independent checks of the headline numbers.
1. Their own script with only the column name corrected -> README numbers (AUC 0.5917, F1 0.3041).
2. Our naive Bayes probabilities vs pgmpy BayesianEstimator(BDeu, ESS 5) + VariableElimination.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import HERE, REPO, DATA, SCRIPT_DIR, OUT  # noqa: E402
import sys, json, warnings; sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")
import numpy as np, lib
P = lib.P
out = {}
from dataclasses import replace
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
df = lib.load()
cfg = replace(lib.ORIGINAL, leak=False, recode=True, mi_correct=True, forced=False, smooth=5.0)
y = P._encode_target(df["match"])
tr_idx, te_idx = lib.wave_folds(df)[0]
cols = lib.candidates(df, cfg)
tr, te = lib.encode(cfg, df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
sel = lib.mi_rank(tr, y.iloc[tr_idx], True)[:10]
trn, ten = tr[sel].assign(match=y.iloc[tr_idx].to_numpy()), te[sel].assign(match=y.iloc[te_idx].to_numpy())
ours = lib.Net(trn.columns, [("match", v) for v in sel]).fit(trn, 5.0).predict(ten)[0]
m = DiscreteBayesianNetwork([("match", v) for v in sel])
m.add_cpds(*BayesianEstimator(m, trn).get_parameters(prior_type="BDeu", equivalent_sample_size=5))
ve = VariableElimination(m)
rows = ten[sel].iloc[:400]
ok = [all(r[v] in m.get_cpds(v).state_names[v] for v in sel) for _, r in rows.iterrows()]
theirs = [ve.query(["match"], evidence={v: int(r[v]) for v in sel}, show_progress=False).get_value(match=1)
          for (_, r), k in zip(rows.iterrows(), ok) if k]
diff = np.max(np.abs(np.array(theirs) - ours[:400][np.array(ok)]))
print("naive Bayes vs pgmpy on", int(np.sum(ok)), "test rows: max |diff| = %.2e" % diff)
out.update(nb_vs_pgmpy_rows=int(np.sum(ok)), nb_vs_pgmpy_maxdiff=float(diff))
json.dump(out, open(str(OUT) + "/verify.json", "w"), indent=1)

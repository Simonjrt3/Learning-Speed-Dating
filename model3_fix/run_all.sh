#!/usr/bin/env bash
# Reproduce every number in "Repairing the pre-date model (Model 3)".
# Run from this folder, placed in the team repository next to pc-mle/ (or set LSD_REPO).
# out/ ships with our results; delete it first to recompute everything (about 45 min on a laptop CPU).
set -euo pipefail
export PYTHONHASHSEED=0      # fixed so that runs are reproducible
PY="python3 -P"
$PY run_orig.py "$(cd .. && pwd)/data/speeddating.csv" 10 f10   # the committed script, unchanged
$PY check_repro.py           # our harness reproduces it exactly
$PY ablation.py              # fixes 1-4, row split and 5 folds by wave
$PY variants.py nb tan pc_nb nb:5 nb:20 nb:10:1 nb:10:20         # fix 5 + alternatives + sensitivity
$PY references.py            # constant, naive Bayes, logistic regression
$PY mi_field.py              # field's mutual information with shuffled labels
$PY lookup.py                # parent configurations seen in training, by participant
$PY hashseed.py              # committed script under six hash seeds
$PY hashseed_fixed.py        # same, with only the column name corrected
$PY check_cycle.py           # pgmpy refuses the forced edges on two new-event folds
$PY verify.py                # naive Bayes vs pgmpy BDeu + variable elimination
$PY summarize.py             # -> out/numbers.json (with wave-bootstrap intervals)
$PY fig_model3.py            # -> out/fig_model3.pdf
$PY render_model3.py section_model3.tex appendix_model3.tex

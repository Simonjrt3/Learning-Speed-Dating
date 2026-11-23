# Model 3 audit and repair

## What the fixes are (`lib.Config`)
| switch       | problem in the committed script                                                    |
|--------------|-------------------------------------------------------------------------------------|
| `leak`       | post-date `sincere_o` survives the filter (it lists `sinsere_o`)                     |
| `recode`     | `field` has 260 free-text values -> 18-category `field_cd`; drop 79%-missing `expected_num_interested_in_me` and the `wave` id; keep `d_age` |
| `mi_correct` | mutual information favours many-valued variables -> Miller-Madow correction          |
| `forced`     | top-5 variables forced as parents of `match` (494,000 parent configurations)         |
| `smooth`     | MLE -> BDeu prior                                                                    |
| fix 5        | `variants.py nb`: `match` -> every selected variable (naive Bayes)                   |

Evaluation: the team's stratified 85/15 row split, and 5 folds grouped by wave (no participant
and no date on both sides). Intervals resample the 21 waves.

## Files
`run_orig.py` (committed script, unchanged), `check_repro.py`, `ablation.py`, `variants.py`,
`references.py`, `mi_field.py`, `lookup.py`, `hashseed.py`, `hashseed_fixed.py`,
`run_fixed_typo.py`, `check_cycle.py`, `verify.py`, `summarize.py` (-> `out/numbers.json`),
`fig_model3.py` (+ `plotstyle.py`), `render_model3.py` (fills the two `*_template.tex` files).

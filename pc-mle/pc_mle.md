# PC/MLE Results

## Setup

- Dataset: `speeddating.csv`, 8,378 rows
- Target: `match`
- Split: 85% training / 15% test
- Random seed: 42
- PC limit: 20 selected variables

## All-data benchmark

This includes post-date variables.
Because `match` is `decision AND decision_o`, this is an upper-bound benchmark
with deterministic target information.

| Metric | Result |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 | 1.0000 |
| ROC-AUC | 1.0000 |
| Confusion matrix | TP=207, TN=1,050, FP=0, FN=0 |

## Pre-date model

This excludes post-date decisions and ratings and uses 10 selected variables:

`field`, `expected_num_matches`, `expected_num_interested_in_me`,
`shared_interests_important`, `pref_o_shared_interests`, `age_o`, `clubbing`,
`age`, `race_o`, and `race`.

| Metric | Result |
|---|---:|
| Accuracy | 0.6905 |
| Precision | 0.2415 |
| Recall | 0.4106 |
| F1 | 0.3041 |
| ROC-AUC | 0.5917 |
| Confusion matrix | TP=85, TN=783, FP=267, FN=122 |

The pre-date model predicts matches but has substantial false positives and
false negatives.

## Metric meanings

- **Accuracy:** Fraction of test rows classified correctly.
- **Precision:** `TP / (TP + FP)`; fraction of predicted matches that are real matches.
- **Recall:** `TP / (TP + FN)`; fraction of real matches found.
- **F1:** Balance between precision and recall.
- **ROC-AUC:** Ability to rank matches above non-matches across thresholds.

TP, TN, FP, and FN mean true positive, true negative, false positive, and false
negative.

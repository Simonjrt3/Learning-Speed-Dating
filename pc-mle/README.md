# PC/MLE Speed-Dating Analysis

This folder contains a PC structure-learning and MLE parameter-estimation
experiment for the `match` target. It uses an 85% training split and a 15%
test split.

## Setup

From the repository root:

```bash
python -m pip install pandas numpy scikit-learn pgmpy
```

Run the script from this folder:

```bash
cd pc-mle
python pc_mle_analysis.py --data ../speeddating.csv --mode both
```

The `both` mode runs the all-data benchmark and then the pre-date model.

Run one experiment:

```bash
python pc_mle_analysis.py --data ../speeddating.csv --mode all --max-features 20
python pc_mle_analysis.py --data ../speeddating.csv --mode pre_date --max-features 10
```

## Feature sets

### All-data columns

The all-data benchmark uses these 20 columns after feature selection:

- `like`
- `funny_partner`
- `funny_o`
- `d_like`
- `field`
- `attractive_partner`
- `attractive_o`
- `shared_interests_o`
- `shared_interests_partner`
- `guess_prob_liked`
- `d_funny_partner`
- `d_funny_o`
- `d_attractive_partner`
- `d_shared_interests_partner`
- `d_shared_interests_o`
- `d_attractive_o`
- `d_guess_prob_liked`
- `intelligence_o`
- `decision`
- `decision_o`

`decision` and `decision_o` are retained because `match` is determined by
`decision AND decision_o`. This experiment is an upper-bound benchmark, not a
real pre-date predictor.

### Pre-date columns

The completed 10-column pre-date experiment uses:

- `field`
- `expected_num_matches`
- `expected_num_interested_in_me`
- `shared_interests_important`
- `pref_o_shared_interests`
- `age_o`
- `clubbing`
- `age`
- `race_o`
- `race`

The pre-date feature set excludes post-date decisions, ratings, and derived
`d_*` variables.

## Results

All-data: accuracy, precision, recall, and F1 are `1.0000`.

Pre-date:

- Accuracy: `0.6905`
- Precision: `0.2415`
- Recall: `0.4106`
- F1: `0.3041`
- ROC-AUC: `0.5917`

See [pc_mle.md](pc_mle.md) for the confusion matrices.

## Options

- `--mode both`: Run all-data and pre-date experiments.
- `--mode all`: Run only the all-data benchmark.
- `--mode pre_date`: Run only the pre-date model.
- `--max-features`: Maximum number of selected variables. The default is 20.
- `--max-cond-vars`: Maximum conditioning-set size for PC. The default is 3.

PC searches can take several minutes. The 20-column pre-date run was
memory-intensive on this machine, so the recorded pre-date result uses 10
columns.

# Plan: stronger date-outcome predictions

**Goal:** predict `decision`, `decision_o` and `match` (= `decision AND decision_o`) from the
ratings recorded on each date. Use one shared data pipeline and one test set for every model,
so the numbers can be compared.

## Where we are

Steps 1–3 are done (`code/data_prep.py`, `code/evaluate.py`, `code/direct_bn.py`).
All numbers are in `results/metrics.csv`. The match cutoff is tuned on training data.

**Same test rows as the notebook** (random 85/15 row split, seed 42, 1,257 rows):

| Model | decision AUC | decision_o AUC | match AUC | match log loss | match F1 |
|---|---|---|---|---|---|
| EM, 3 hidden nodes (notebook) | 0.76 | 0.58 | 0.72 | 0.422 | 0.45 |
| Direct-edge BN, selected parents | 0.79 | 0.80 | **0.81** | **0.373** | **0.49** |
| Always the training yes-rate | 0.50 | 0.50 | 0.50 | 0.470 | 0.00 |

**Participant split** (no participant in both train and test, 1,334 rows; this is the split to report):

| Model | decision AUC | decision_o AUC | match AUC | match log loss | match F1 |
|---|---|---|---|---|---|
| Direct-edge BN, 5 starting parents | 0.81 | 0.81 | 0.81 | 0.390 | 0.50 |
| Direct-edge BN, selected parents | 0.79 | **0.82** | **0.82** | **0.370** | **0.51** |
| Always the training yes-rate | 0.50 | 0.50 | 0.50 | 0.475 | 0.00 |

The notebook's "match F1 = 0" came from its 0.5 cutoff. With a cutoff tuned on training data, the EM model reaches F1 0.45.
Its real weaknesses are `decision_o` (AUC 0.58) and calibration (match log loss 0.42 vs 0.37).

Pre-date PC + MLE model (`pc-mle/`, its own split): match AUC 0.59, F1 0.30.

**Why the EM model was weak:**
1. Both decisions see the data only through 3 hidden nodes with 3 states each, 27 combinations in total.
2. EM spends its effort explaining the 57 input columns, not the 2 decisions.
3. One hidden node mixes "how I rated them" with "how they rated me".
4. Duplicate columns count the same information twice (`like` and its binned copy `d_like`, `age` and `d_age`, and so on).
5. A 0.5 cutoff is far too high for a target that is "yes" only 16.5% of the time.

---

## Step 1: Shared data pipeline (`code/data_prep.py`)

Every model imports this, instead of each one preprocessing differently.

- [x] **Load and join.** Load `data/speeddating.csv` and add `iid`, `pid`, `like_o` and `prob_o` from
      `data/annavictoria_data.csv`. The two files line up row for row (checked: wave, both decisions,
      `attractive_o` and `like` agree in 100% of rows). Add an `assert` on the alignment.
- [x] **Decode values.** Strip the `b'...'` encoding, and treat both empty cells and `b'?'` as missing.
- [x] **Keep raw columns only.** Drop all `d_*` columns; we bin the raw values ourselves.
      `d_*` also files missing values under the lowest bin: `d_like` puts the 240 rows with no `like`
      score into `[0-5]`.
- [x] **Never use these as features:** `match` (it is the label), `has_null`, `wave`.
- [x] **Bins** for 1–10 ratings: `≤5`, `6–8`, `9–10`, plus a separate `missing` state. That is 4 states.
      For `guess_prob_liked` and `prob_o`, check the scale first.
- [ ] **Waves 6–9 scale fix.** *(Not done; the direct-edge model doesn't use these columns.)* Only needed if the stated-importance (`*_important`) variables are used:
      add the scale as a variable, or drop those waves.
- [x] **Train/test split by participant.** Use `GroupShuffleSplit` on `iid`, 85/15, seed 42.
      Save the test indices to `data/test_ids.csv` so every model uses exactly the same test rows.
      In the scratch test, splitting by participant or by date pair scored about the same as a random
      row split (match AUC 0.81–0.82), so this is for honesty rather than a big score change.

**Done when:** `prepare()` returns `train, test` DataFrames with integer-coded states and a state-name map,
and both the notebook and `pc-mle/` use it.
*(The module is done. Moving the notebook and `pc-mle/` onto it is part of Steps 5 and 6.)*

## Step 2: Shared evaluation (`code/evaluate.py`)

- [x] **Inputs:** `evaluate(y_true, p, threshold)` scores one target. Call it for `decision`,
      `decision_o` and `match`.
- [x] **Metrics:** AUC, log loss, Brier score, precision, recall, F1, and the confusion matrix.
- [x] **Baselines:** always predict "no"; always predict the training "yes" rate.
- [x] **Cutoff:** choose the threshold that maximizes F1 **on the training data only**, then apply it to the test set.
- [x] **Output:** append one row per model to `results/metrics.csv`. The report table is built from this file.

## Step 3: Direct-edge Bayesian network (main fix)

Each decision gets the ratings made by the person who decides as its parents:

```
like, attractive_partner, funny_partner, shared_interests_partner, guess_prob_liked  ->  decision
like_o, attractive_o, funny_o, shared_interests_o, prob_o                            ->  decision_o
P(match) = P(decision) * P(decision_o)
```

- [x] **Build** it in pgmpy and fit with `BayesianEstimator` (BDeu prior, equivalent sample size about 5–10).
      Five 4-state parents give 4^5 = 1,024 parent combinations, and many will have few or no rows, so a
      prior is required.
- [x] **Remove one parent at a time** and keep the parents whose removal lowers validation AUC.
      Fewer parents usually means steadier probabilities.
- [x] **Try adding** *(none of them helped)* `sincere_*`, `intelligence_*`, `ambition_*`, `samerace` and `interests_correlate`,
      one at a time.
- [x] **Target**, from the scratch test: match AUC ≥ 0.80, match F1 ≥ 0.45.
      As a reference, logistic regression on the same ratings reaches match AUC of about 0.85.

**Step 3 findings (participant split):**
- **Selected parents:** greedy selection on a validation split taken from train kept three parents per decision:
  - `decision` ← `attractive_partner`, `funny_partner`, `shared_interests_partner`
  - `decision_o` ← `like_o`, `attractive_o`, `funny_o`
- **Fewer parents worked better.** Dropping `like` / `guess_prob_liked` improved validation log loss, because with
  5 states per parent a 5-parent table has 3,125 cells and many are nearly empty. No extra candidate helped:
  sincere, intelligence, ambition, samerace, met and interests_correlate were all tried.
- **The selection is not stable.** On the row split, `decision_o` kept `attractive_o`, `funny_o` and
  `shared_interests_o` instead. Treat the exact parent list as one reasonable choice, not a finding.
- **Accuracy:** the F1-tuned cutoff (about 0.3) gives match accuracy 0.79, just under always-no (0.82).
  The F1 cutoff trades accuracy for catching about 60% of the matches. Report AUC, log loss and F1, not accuracy.

## Step 4: Check that multiplying the two decisions is OK

`P(match) = P(decision) × P(decision_o)` assumes the two decisions are independent once their ratings are known.

- [ ] **Calibration check:** compare predicted against observed match rates in 10 bins of predicted probability.
- [ ] **If matches are under-predicted:** add a shared hidden node "chemistry" (2–3 states) as a parent of both
      decisions, fit it with EM, and compare match log loss. This also gives the report a principled reason
      to use a hidden variable.

## Step 5: Rebuild the hidden-variable model (keeps the EM part of the project)

- [ ] **Separate hidden node per viewpoint:** `my_impression` gets the participant's ratings of the partner
      as children; `their_impression` gets the partner's ratings of the participant.
- [ ] **Edges:** `my_impression → decision` and `their_impression → decision_o`.
      Also keep the direct edges from Step 3 for the 2–3 strongest ratings.
- [ ] **Inputs:** drop the duplicate `d_*` columns. Leave out demographics and hobbies unless the
      one-at-a-time test shows they help.
- [ ] **Hidden states:** try 2, 3 and 4.
- [ ] **Speed:** with about 15 variables, EM should take minutes rather than 24.
- [ ] **Compare** with Step 3 in `results/metrics.csv`.

## Step 6: Structure learning with PC (`pc-mle/`)

- [ ] **Shared setup:** switch to `data_prep.py` and `evaluate.py`, and predict `decision` and `decision_o`
      rather than `match` directly. This covers the existing TODO for Alex, and it also removes the leaky
      "all-data" run, which scored 1.00 because the decisions were features.
- [ ] **Run PC** on the Step 3 feature set. Report which edges it finds into `decision` and `decision_o`,
      and compare them with the hand-built edges.
- [ ] **Forced parents:** remove the forced `target_parents`, or say clearly in the report that they were forced.
- [ ] **Pre-date run:** rerun with pre-date variables only, dropping `expected_num_interested_in_me`, which is
      78% missing.
- [ ] **README:** fix the stale path (`../speeddating.csv` → `../data/speeddating.csv`).

## Step 7: Report

- [ ] **Section 2:** match it to the final pipeline (bins, missing state, participant split, the `like_o` join).
- [ ] **Results:** one table from `results/metrics.csv`, with every model and the always-no baseline.
- [ ] **Discussion:** the hidden-node bottleneck as a lesson, PC edges compared with hand-built edges,
      and pre-date versus after-date predictability.

---

## Order and who can work in parallel

1. Steps 1 and 2 come first, because everything else depends on them.
2. Then Steps 3, 5 and 6 can run in parallel.
3. Step 4 comes after Step 3.
4. Step 7 comes last.

## Risks

- **Sparse tables:** too many parents per decision means many parent combinations have few rows.
  Keep 5 or fewer parents and use the BDeu prior.
- **Scale differences:** waves differ in session size and preference scale. Check the per-wave AUC on the test set.
- **Same person on both sides:** each date appears twice, once from each side. The participant split keeps a
  participant's own rows together, but their partner's rows can still land in training.
  Splitting by date pair is the stricter alternative.

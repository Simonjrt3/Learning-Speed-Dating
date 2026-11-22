"""Shared data pipeline for every speed-dating model (plan Step 1).

Usage (from code/):
    from data_prep import prepare
    train, test, state_names, labels = prepare()

`state_names` maps column -> list of integer states (pass it to pgmpy);
`labels` maps column -> human-readable name of each state.
`train` / `test` hold one integer-coded column per feature, plus the 0/1 labels
`decision`, `decision_o`, `match`, and the ids `iid`, `pid`, `wave`
(ids are for bookkeeping, never use them as features).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LABELS = ["decision", "decision_o", "match"]
IDS = ["iid", "pid", "wave"]
MISSING = "missing"

# Ratings on a 0-10 scale -> low [0-5], mid [6-8], high [9-10], plus missing.
RATING_EDGES = [-np.inf, 5.5, 8.5, np.inf]
RATING_LABELS = ["0-5", "6-8", "9-10"]
# 100-point importance allocations -> same cut points as the dataset's d_* columns.
IMPORTANCE_EDGES = [-np.inf, 15.5, 20.5, np.inf]
IMPORTANCE_LABELS = ["0-15", "16-20", "21-100"]

# Participant's ratings of the partner (made by the person who gives `decision`).
SELF_VIEW = [
    "like", "guess_prob_liked", "attractive_partner", "sincere_partner",
    "intelligence_partner", "funny_partner", "ambition_partner",
    "shared_interests_partner",
]
# Partner's ratings of the participant (made by the person who gives `decision_o`).
# like_o and prob_o come from the raw annavictoria file.
PARTNER_VIEW = [
    "like_o", "prob_o", "attractive_o", "sincere_o", "intelligence_o",
    "funny_o", "ambitous_o", "shared_interests_o",
]
SELF_RATINGS = ["attractive", "sincere", "intelligence", "funny", "ambition"]
OTHER_RATINGS = [
    "expected_happy_with_sd_people", "importance_same_race",
    "importance_same_religion",
] + ["sports", "tvsports", "exercise", "dining", "museums", "art", "hiking",
     "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts",
     "music", "shopping", "yoga"]
RATING_COLS = SELF_VIEW + PARTNER_VIEW + SELF_RATINGS + OTHER_RATINGS
IMPORTANCE_COLS = [
    "attractive_important", "sincere_important", "intellicence_important",
    "funny_important", "ambition_important", "shared_interests_important",
    "pref_o_attractive", "pref_o_sincere", "pref_o_intelligence",
    "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests",
]
# Numeric columns without a fixed scale: 3 quantile bins learned on train.
QUANTILE_COLS = ["age", "age_o", "d_age", "interests_correlate",
                 "expected_num_matches"]
CATEGORICAL_COLS = ["gender", "race", "race_o", "samerace", "met", "field"]

# Never used: label leak, constant flag, 78% missing, binned duplicates (d_*).
EXCLUDED = ["match", "has_null", "expected_num_interested_in_me"]


def _decode(value):
    """b'female' -> 'female'; b'?' and '?' -> NaN."""
    if isinstance(value, str):
        if value.startswith("b'") and value.endswith("'"):
            value = value[2:-1]
        if value in ("?", ""):
            return np.nan
    return value


def load_raw() -> pd.DataFrame:
    """Load speeddating.csv, decode it, and join ids + like_o/prob_o from the raw file."""
    df = pd.read_csv(DATA_DIR / "speeddating.csv")
    raw = pd.read_csv(DATA_DIR / "annavictoria_data.csv", encoding="latin1")

    for col in df.select_dtypes(exclude="number").columns:  # object or pandas str dtype
        df[col] = df[col].map(_decode)

    # The two files are the same 8,378 dates in the same order. Check before joining.
    assert len(df) == len(raw)
    assert (df["wave"].values == raw["wave"].values).all()
    assert (df["decision"].astype(int).values == raw["dec"].values).all()
    assert (df["decision_o"].astype(int).values == raw["dec_o"].values).all()
    assert np.allclose(df["like"], raw["like"], equal_nan=True)

    for col in ["iid", "pid", "like_o", "prob_o"]:
        df[col] = raw[col].values

    for col in LABELS:
        df[col] = df[col].astype(int)
    # met should be 0/1; a handful of rows hold 3-8 (data-entry errors), treat any >0 as "met before".
    df["met"] = df["met"].where(df["met"].isna(), (df["met"] > 0).astype(float))
    df["samerace"] = df["samerace"].astype("Int64").astype(str).replace("<NA>", np.nan)

    # Keep raw columns only: d_* are binned copies (and put missing values in the lowest bin).
    keep = [c for c in df.columns if not c.startswith("d_") or c == "d_age"]
    return df[keep]


def _bin_fixed(s, edges, labels):
    return pd.cut(s, edges, labels=labels).astype(object).where(s.notna(), MISSING)


def _bin_quantile(train_s, s):
    edges = np.unique(np.nanquantile(train_s, [0, 1 / 3, 2 / 3, 1]))
    edges[0], edges[-1] = -np.inf, np.inf
    labels = [f"q{i}" for i in range(len(edges) - 1)]
    return pd.cut(s, edges, labels=labels).astype(object).where(s.notna(), MISSING)


def split_ids(df, split="participant", test_size=0.15, seed=42):
    """Return (train_index, test_index).

    participant: all rows of a participant (iid) on one side.
    pair:        both rows of a date (A,B) and (B,A) on one side.
    row:         random rows (same as the original notebook split).
    """
    idx = np.arange(len(df))
    if split == "row":
        return train_test_split(idx, test_size=test_size, random_state=seed)
    if split == "participant":
        groups = df["iid"]
    elif split == "pair":
        groups = [f"{min(a, b)}-{max(a, b)}" for a, b in zip(df["iid"], df["pid"].fillna(-1))]
    else:
        raise ValueError(split)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=groups))


def prepare(split="participant", test_size=0.15, seed=42, save_test_ids=True):
    """Load, split, and discretize. All learned state (quantiles, categories) comes from train only."""
    df = load_raw()
    tr_idx, te_idx = split_ids(df, split, test_size, seed)
    if save_test_ids:
        pd.DataFrame({"row": te_idx}).to_csv(DATA_DIR / f"test_ids_{split}.csv", index=False)

    out = pd.DataFrame(index=df.index)
    for c in RATING_COLS:
        out[c] = _bin_fixed(df[c], RATING_EDGES, RATING_LABELS)
    for c in IMPORTANCE_COLS:
        out[c] = _bin_fixed(df[c], IMPORTANCE_EDGES, IMPORTANCE_LABELS)
    for c in QUANTILE_COLS:
        out[c] = _bin_quantile(df[c].iloc[tr_idx], df[c])
    for c in CATEGORICAL_COLS:
        out[c] = df[c].astype(object).where(df[c].notna(), MISSING).astype(str)

    # Integer-code every feature. Categories seen only in test map to "unknown".
    state_names, labels = {}, {}
    for c in out.columns:
        cats = sorted(set(out[c].iloc[tr_idx]) - {MISSING}) + [MISSING, "unknown"]
        coded = pd.Categorical(out[c].where(out[c].isin(cats), "unknown"), categories=cats)
        out[c] = coded.codes.astype(int)
        state_names[c] = list(range(len(cats)))
        labels[c] = cats
    for c in LABELS:
        out[c] = df[c].astype(int)
        state_names[c] = [0, 1]
        labels[c] = ["no", "yes"]
    for c in IDS:
        out[c] = df[c]

    return out.iloc[tr_idx].copy(), out.iloc[te_idx].copy(), state_names, labels


def wave_scale_note():
    """Waves 6-9 report importances as 1-10 ratings rescaled to sum to 100 (see report Sec. 2).
    The direct-edge model does not use IMPORTANCE_COLS; if you do, add `wave in 6..9` as a variable."""
    return {6, 7, 8, 9}


if __name__ == "__main__":
    for s in ["participant", "row"]:
        tr, te, sn, _ = prepare(split=s)
        print(f"{s:12s} train {tr.shape}  test {te.shape}  "
              f"test participants in train: {len(set(te.iid) & set(tr.iid))}  "
              f"test match rate: {te.match.mean():.3f}")

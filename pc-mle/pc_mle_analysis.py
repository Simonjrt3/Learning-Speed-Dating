"""Learn and evaluate a discrete Bayesian network for speed-dating matches.

This module is intentionally separate from ``speed_dating.ipynb``.  It provides
one reproducible pipeline for two experiments:

``all``
    Uses every predictor except ``match``.  This is an upper-bound benchmark,
    not a prospective predictor, because it includes post-date responses.

``pre_date``
    Removes date outcomes, partner ratings, and derived ``d_*`` response
    columns.  This estimates whether a match can be predicted before the date.
    This is called leakage-safe because no information collected after the
    prediction point is allowed to influence the prediction.

The pipeline uses an 85/15 stratified train/test split.  Numeric variables are
median-imputed and discretized on the training partition only; categorical
variables are ordinal-encoded with an explicit unknown category.  The PC
algorithm then learns a discrete graph from the training data, and MLE fits
conditional probability tables for that graph.

Install the optional Bayesian-network dependency in the notebook's active
kernel before running this module:

    %pip install pgmpy

Example:

    python pc_mle_analysis.py --data speeddating.csv --mode both

The ``all`` experiment is useful for measuring the information contained in
post-date responses, but its score must not be presented as a deployable
pre-date prediction result.

Pipeline stages:

Step 1: Load and validate the dataset.
Step 2: Define the all-data or pre-date feature set.
Step 3: Create the stratified 85/15 train/test split.
Step 4: Impute, discretize, and encode features using training data only.
Step 5: For each input column, measure how useful its values are for
distinguishing ``match = 1`` from ``match = 0`` using training rows only.
Keep the top ``max_features`` columns (20 by default). This is an exploratory
runtime limit; pass ``--max-features 121`` to request all predictors in
``all`` mode.
Step 6: Learn the graph structure with the PC algorithm.
Step 7: Estimate CPDs (conditional probability distributions) with MLE.
Step 8: Infer match probabilities and evaluate held-out predictions.

Prediction leakage occurs when a model uses information that would not be
available when the prediction is supposed to be made.  For example, using
``decision`` to predict ``match`` would leak the outcome because ``match`` is
defined using both participants' decisions.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

TARGET = "match"
RANDOM_STATE = 42
TEST_SIZE = 0.15
Mode = Literal["all", "pre_date"]

_BYTE_VALUE_PATTERN = re.compile(r"^b['\"](.*)['\"]$")
_RANGE_PATTERN = re.compile(r"^b['\"]\[(\d+)-(\d+)\]['\"]$")

# These are variables collected during or after the date.  ``d_*`` columns are
# excluded in pre-date mode because they are derived discrepancy/response
# fields in this version of the dataset.
_POST_DATE_BASE_COLUMNS = frozenset(
    {
        "attractive_o",
        "sinsere_o",
        "intelligence_o",
        "funny_o",
        "ambitous_o",
        "shared_interests_o",
        "attractive_partner",
        "sincere_partner",
        "intelligence_partner",
        "funny_partner",
        "ambition_partner",
        "shared_interests_partner",
        "like",
        "guess_prob_liked",
        "met",
        "decision",
        "decision_o",
    }
)


@dataclass(frozen=True)
class ExperimentResult:
    """Results and learned artifacts for one feature setting."""

    mode: Mode
    selected_features: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    metrics: dict[str, float]
    threshold: float
    model: object


def _decode_value(value: object) -> object:
    """Normalize byte-string representations and range strings."""
    if not isinstance(value, str):
        return value

    range_match = _RANGE_PATTERN.fullmatch(value)
    if range_match:
        low, high = map(int, range_match.groups())
        return (low + high) / 2

    byte_match = _BYTE_VALUE_PATTERN.fullmatch(value)
    return byte_match.group(1) if byte_match else value


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw speed-dating dataset."""
    frame = pd.read_csv(path).map(_decode_value)
    required = {TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def post_date_columns(columns: Sequence[str]) -> set[str]:
    """Return columns excluded from the prospective pre-date experiment."""
    return {
        column
        for column in columns
        if column in _POST_DATE_BASE_COLUMNS or column.startswith("d_")
    }


def feature_columns(
    frame: pd.DataFrame,
    mode: Mode,
    *,
    include_wave: bool = True,
) -> list[str]:
    """Build a documented predictor list for the requested experiment."""
    if mode not in {"all", "pre_date"}:
        raise ValueError(f"Unsupported mode {mode!r}; expected 'all' or 'pre_date'.")

    excluded = {TARGET, "has_null"}
    if not include_wave:
        excluded.add("wave")
    if mode == "pre_date":
        excluded.update(post_date_columns(frame.columns))

    return [column for column in frame.columns if column not in excluded]


class DiscretePreprocessor:
    """Fit leakage-safe imputation and discretization for Bayesian networks."""

    def __init__(self, numeric_bins: int = 5) -> None:
        if numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2.")
        self.numeric_bins = numeric_bins
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.numeric_medians: pd.Series | None = None
        self.numeric_edges: dict[str, np.ndarray] = {}
        self.binary_columns: set[str] = set()
        self.category_maps: dict[str, dict[object, int]] = {}

    def fit(self, frame: pd.DataFrame) -> "DiscretePreprocessor":
        """Learn all transformations from training data only."""
        normalized = frame.map(_decode_value)
        numeric = normalized.apply(pd.to_numeric, errors="coerce")
        self.numeric_columns = [
            column
            for column in normalized.columns
            if numeric[column].notna().mean() >= 0.8
        ]
        self.categorical_columns = [
            column for column in normalized.columns if column not in self.numeric_columns
        ]
        self.numeric_medians = numeric[self.numeric_columns].median()

        for column in self.numeric_columns:
            values = numeric[column].fillna(self.numeric_medians[column]).to_numpy()
            unique_values = np.unique(values)
            if set(unique_values.tolist()).issubset({0, 1}) and len(unique_values) <= 2:
                self.binary_columns.add(column)
            quantiles = np.linspace(0, 1, self.numeric_bins + 1)
            edges = np.unique(np.quantile(values, quantiles))
            self.numeric_edges[column] = edges

        for column in self.categorical_columns:
            values = normalized[column].fillna("__MISSING__")
            categories = sorted(values.astype(str).unique())
            self.category_maps[column] = {value: index for index, value in enumerate(categories)}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted training-only state."""
        if self.numeric_medians is None:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        normalized = frame.map(_decode_value)
        encoded_columns: dict[str, np.ndarray] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(normalized[column], errors="coerce")
            values = values.fillna(self.numeric_medians[column]).to_numpy()
            edges = self.numeric_edges[column]
            if column in self.binary_columns:
                encoded = (values >= 0.5).astype(np.int64)
            elif len(edges) <= 1:
                encoded = np.zeros(len(values), dtype=np.int64)
            else:
                encoded = np.digitize(values, edges[1:-1], right=False)
            encoded_columns[column] = encoded.astype(np.int64)

        for column in self.categorical_columns:
            mapping = self.category_maps[column]
            values = normalized[column].fillna("__MISSING__").astype(str)
            # Unknown test-time categories receive a dedicated state.
            unknown_state = len(mapping)
            encoded_columns[column] = (
                values.map(mapping).fillna(unknown_state).astype(np.int64).to_numpy()
            )

        return pd.DataFrame(
            encoded_columns,
            index=frame.index,
            dtype=np.int64,
        )[self.numeric_columns + self.categorical_columns]


def _encode_target(series: pd.Series) -> pd.Series:
    """Convert match values to binary integers and reject malformed targets."""
    encoded = pd.to_numeric(series, errors="coerce")
    if encoded.isna().any() or not encoded.isin([0, 1]).all():
        raise ValueError("The match target must contain only binary 0/1 values.")
    return encoded.astype(np.int64)


def select_features(
    training: pd.DataFrame,
    target: pd.Series,
    *,
    max_features: int,
) -> list[str]:
    """Select informative variables using training data only.

    PC has exponential worst-case conditioning-set growth.  A documented cap
    makes the first experiment reproducible and computationally tractable.
    """
    if max_features < 1:
        raise ValueError("max_features must be positive.")
    if len(training.columns) <= max_features:
        return list(training.columns)

    scores = mutual_info_classif(
        training,
        target,
        discrete_features=True,
        random_state=RANDOM_STATE,
    )
    ranked = sorted(
        zip(training.columns, scores),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [column for column, _ in ranked[:max_features]]


def learn_and_fit_network(
    training: pd.DataFrame,
    *,
    alpha: float = 0.05,
    max_cond_vars: int | None = 3,
    target_parents: Sequence[str] = (),
) -> object:
    """Run discrete PC and fit CPDs with MLE using pgmpy.

    The returned model is a ``DiscreteBayesianNetwork``.  The PC result is
    requested as a DAG so that MLE has a directed parent set for every node.
    """
    try:
        from pgmpy.estimators import PC
        from pgmpy.models import DiscreteBayesianNetwork
        from pgmpy.parameter_estimator import DiscreteMLE
    except ImportError as exc:
        raise ImportError(
            "This step requires pgmpy. Install it in the active notebook "
            "kernel with `%pip install pgmpy`."
        ) from exc

    estimator = PC(training)
    # PC removes edges when a chi-square test finds conditional independence.
    # Its natural result is a CPDAG; pgmpy gives us a DAG orientation here so
    # that each variable has a parent set for parameter estimation.
    dag = estimator.estimate(
        variant="stable",
        ci_test="chi_square",
        significance_level=alpha,
        max_cond_vars=max_cond_vars,
        return_type="dag",
    )
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(training.columns)
    model.add_edges_from(dag.edges())
    # A user-requested domain constraint can supplement PC.  For the all-data
    # benchmark, match is deterministically decision AND decision_o.
    for parent in target_parents:
        if parent in training.columns:
            model.add_edge(parent, TARGET)
    # MLE counts each variable's observed states within every parent
    # configuration to estimate the network's conditional probability tables.
    model.fit(training, estimator=DiscreteMLE())
    return model


def posterior_probabilities(
    model: object,
    data: pd.DataFrame,
    *,
    target: str = TARGET,
    fallback_probability: float | None = None,
) -> tuple[np.ndarray, int]:
    """Return match probabilities and count impossible MLE evidence rows."""
    try:
        from pgmpy.inference import VariableElimination
    except ImportError as exc:
        raise ImportError(
            "Evaluation requires pgmpy. Install it with `%pip install pgmpy`."
        ) from exc

    inference = VariableElimination(model)
    # Variables outside the target's Markov blanket cannot change its
    # posterior.  Excluding them also avoids conditioning on unrelated
    # combinations that were never observed by an unsmoothed MLE model.
    markov_blanket = set(model.get_markov_blanket(target))
    evidence_columns = [
        column
        for column in data.columns
        if column != target and column in markov_blanket
    ]
    probabilities: list[float] = []
    non_finite_count = 0
    for row in data[evidence_columns].to_dict(orient="records"):
        posterior = inference.query([target], evidence=row, show_progress=False)
        states = list(posterior.state_names[target])
        probability = float(posterior.values[states.index(1)])
        if not np.isfinite(probability):
            if fallback_probability is None:
                raise ValueError(
                    "MLE produced a non-finite posterior for a held-out evidence "
                    "configuration. Provide fallback_probability or use smoothing."
                )
            non_finite_count += 1
            probability = fallback_probability
        probabilities.append(probability)

    if non_finite_count:
        warnings.warn(
            f"{non_finite_count} held-out rows had impossible MLE evidence "
            "configurations; the training-set target prior was used explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.asarray(probabilities), non_finite_count


def tune_threshold(
    model: object,
    training_data: pd.DataFrame,
    *,
    target: str = TARGET,
    fallback_probability: float | None = None,
) -> float:
    """Tune F1 threshold using training data only.

    This adds cost because it performs one posterior pass over the training
    rows. It is kept separate from test evaluation, so test outcomes cannot
    influence the threshold.
    """
    probabilities, _ = posterior_probabilities(
        model,
        training_data,
        target=target,
        fallback_probability=fallback_probability,
    )
    y_true = training_data[target].to_numpy()
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 1.0]),
                probabilities,
            )
        )
    )

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        y_predicted = (probabilities >= threshold).astype(np.int64)
        if y_predicted.min() == y_predicted.max():
            continue
        score = f1_score(y_true, y_predicted, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(
    model: object,
    test_data: pd.DataFrame,
    *,
    target: str = TARGET,
    fallback_probability: float | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate posterior predictions and report impossible MLE evidence.

    Plain MLE can produce zero-probability CPT entries.  If a held-out
    configuration has no valid posterior, the evaluator requires an explicit
    training-set prior fallback and reports how often it was needed.
    """
    y_probability, non_finite_count = posterior_probabilities(
        model,
        test_data,
        target=target,
        fallback_probability=fallback_probability,
    )

    y_true = test_data[target].to_numpy()
    y_predicted = (y_probability >= threshold).astype(np.int64)
    true_positives = int(np.sum((y_true == 1) & (y_predicted == 1)))
    true_negatives = int(np.sum((y_true == 0) & (y_predicted == 0)))
    false_positives = int(np.sum((y_true == 0) & (y_predicted == 1)))
    false_negatives = int(np.sum((y_true == 1) & (y_predicted == 0)))
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision": float(precision_score(y_true, y_predicted, zero_division=0)),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "pr_auc": float(average_precision_score(y_true, y_probability)),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "non_finite_posteriors": float(non_finite_count),
        "true_positives": float(true_positives),
        "true_negatives": float(true_negatives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
    }


def run_experiment(
    data_path: str | Path,
    *,
    mode: Mode,
    max_features: int = 20,
    alpha: float = 0.05,
    max_cond_vars: int | None = 3,
) -> ExperimentResult:
    """Run one complete 85/15 split, PC, MLE, and evaluation experiment."""
    # Step 1: Load and validate the raw dataset and binary target.
    raw = load_dataset(data_path)

    # Step 2: Define either the all-data benchmark or the leakage-safe
    # pre-date feature set. Leakage-safe means that post-date responses are
    # excluded because they are unavailable before the date begins.
    predictors = feature_columns(raw, mode)
    target = _encode_target(raw[TARGET])

    # Step 3: Create a reproducible, stratified 85/15 split.
    # Stratification preserves the approximately 16.5% match rate in both
    # partitions, while fitting all learned preprocessing state on train only
    # prevents information from the test set leaking into the experiment.
    train_raw, test_raw, y_train, y_test = train_test_split(
        raw[predictors],
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    # Step 4: Fit preprocessing on train only, then apply the same mappings to
    # the test data. Bayesian-network structure learning requires discrete
    # integer-valued states.
    preprocessor = DiscretePreprocessor().fit(train_raw)
    train_encoded = preprocessor.transform(train_raw)
    test_encoded = preprocessor.transform(test_raw)

    # Step 5: Use only training rows to score how well each input column
    # separates matches from non-matches. Keep the top max_features columns;
    # otherwise PC would need to test relationships among all 121 inputs.
    selected = select_features(
        train_encoded,
        y_train,
        max_features=max_features,
    )
    # The all-data benchmark is meant to measure the information explicitly
    # available after the date. Keep the two variables that define match
    # whenever they fit within the selected-feature cap.
    if mode == "all":
        retained = [
            column
            for column in selected
            if column not in {"decision", "decision_o"}
        ]
        for required_column in ("decision", "decision_o"):
            if required_column in train_encoded.columns:
                selected.append(required_column)
        selected = retained[: max_features - 2] + [
            column
            for column in ("decision", "decision_o")
            if column in train_encoded.columns and column in selected
        ]

    train_network_data = train_encoded[selected].copy()
    test_network_data = test_encoded[selected].copy()
    train_network_data[TARGET] = y_train.to_numpy()
    test_network_data[TARGET] = y_test.to_numpy()

    # Step 6: Learn the dependency graph with PC.
    model = learn_and_fit_network(
        train_network_data,
        alpha=alpha,
        max_cond_vars=max_cond_vars,
        target_parents=(
            ("decision", "decision_o")
            if mode == "all"
            else tuple(selected[:5])
        ),
    )

    # Step 7: learn_and_fit_network estimates the model's CPDs with MLE.
    # Choose the binary cutoff using training predictions rather than assuming
    # a 0.5 cutoff is appropriate for the 16.5% match class.
    threshold = tune_threshold(
        model,
        train_network_data,
        fallback_probability=float(y_train.mean()),
    )

    # Step 8: use those CPDs and the training-tuned cutoff for held-out
    # inference and evaluation.
    metrics = evaluate_model(
        model,
        test_network_data,
        fallback_probability=float(y_train.mean()),
        threshold=threshold,
    )
    edges = tuple(sorted((str(source), str(destination)) for source, destination in model.edges()))
    return ExperimentResult(mode, tuple(selected), edges, metrics, threshold, model)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("speeddating.csv"))
    parser.add_argument(
        "--mode",
        choices=("both", "all", "pre_date"),
        default="both",
        help="Run both feature settings, or select one setting.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20,
        help="Maximum predictors for PC; use 121 for all predictors in all mode.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-cond-vars", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run one or both experiments and print reproducible summary output."""
    args = _parse_args()
    modes: tuple[Mode, ...] = (
        ("all", "pre_date") if args.mode == "both" else (args.mode,)
    )
    for index, mode in enumerate(modes):
        if index:
            print("\n" + "=" * 80 + "\n")
        result = run_experiment(
            args.data,
            mode=mode,
            max_features=args.max_features,
            alpha=args.alpha,
            max_cond_vars=args.max_cond_vars,
        )
        print(f"mode: {result.mode}")
        print(f"training-tuned threshold: {result.threshold:.6f}")
        print(f"selected features ({len(result.selected_features)}):")
        print(", ".join(result.selected_features))
        print(f"learned edges ({len(result.edges)}):")
        for source, destination in result.edges:
            print(f"  {source} -> {destination}")
        print("held-out metrics:")
        for name, value in result.metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()

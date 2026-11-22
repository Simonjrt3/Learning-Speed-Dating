"""Shared evaluation for every speed-dating model (plan Step 2).

Usage:
    from evaluate import tune_threshold, evaluate, save_results
    t = tune_threshold(y_train, p_train)          # cutoff chosen on TRAIN only
    row = evaluate("my_model", "match", y_test, p_test, t, split="participant")
    save_results([row, ...])                      # writes results/metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)

RESULTS = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"


def tune_threshold(y_train, p_train, grid=np.linspace(0.05, 0.95, 91)):
    """Cutoff that maximizes F1 on the training data (never look at test labels)."""
    y_train, p_train = np.asarray(y_train), np.asarray(p_train)
    return float(max(grid, key=lambda t: f1_score(y_train, p_train >= t, zero_division=0)))


def evaluate(model, target, y, p, threshold=0.5, split=""):
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "model": model, "split": split, "target": target, "n_test": len(y),
        "yes_rate": round(y.mean(), 4),
        "AUC": round(roc_auc_score(y, p), 4),
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy_score(y, yhat), 4),
        "always_no_acc": round(1 - y.mean(), 4),
        "precision": round(precision_score(y, yhat, zero_division=0), 4),
        "recall": round(recall_score(y, yhat, zero_division=0), 4),
        "F1": round(f1_score(y, yhat, zero_division=0), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def baselines(y_train, y_test, target, split=""):
    """Predict the training yes-rate for everyone (AUC 0.5; accuracy = always-no)."""
    p = np.full(len(y_test), np.mean(y_train))
    return evaluate("baseline_train_rate", target, y_test, p, 0.5, split)


def save_results(rows, path=RESULTS):
    """Insert/replace rows keyed by (model, split, target) so reruns do not duplicate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        key = ["model", "split", "target"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new

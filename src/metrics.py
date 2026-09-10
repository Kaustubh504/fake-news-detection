"""
Evaluation and result recording.

CONTRACT — agreed across all four approaches.

Every approach writes rows in the same shape, to its own CSV. `report.py`
concatenates them into the final comparison table. Separate files per
approach avoid merge conflicts when several people run experiments.

POSITIVE CLASS
--------------
Precision, recall and FPR are computed with FAKE (label 0) as the positive
class, because detecting fabrication is the task:

    precision = of the articles we called fake, how many were fake
    recall    = of the fake articles, how many we caught
    fpr       = real articles wrongly flagged

macro_f1 averages both classes equally and is the PRIMARY metric throughout.
Accuracy is recorded but never relied on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import LABEL_FAKE

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Every row written to results/ has exactly these fields.
RESULT_FIELDS = [
    "approach",    # "A", "B", "C", "D"
    "system",      # "baseline", "ceiling", ...
    "scope",       # "overall", or a topic name for per-topic rows
    "seed",
    "n_train",     # training rows this model actually saw
    "n_test",
    "macro_f1",    # PRIMARY
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "roc_auc",
    "mcc",
]

_METRIC_KEYS = ["macro_f1", "accuracy", "precision", "recall", "fpr", "roc_auc", "mcc"]


def evaluate(
    y_true, y_pred, y_prob_fake=None
) -> dict[str, float]:
    """Score one set of predictions.

    Parameters
    ----------
    y_true, y_pred
        Binary arrays: 0 = fake, 1 = real.
    y_prob_fake
        Predicted probability of the FAKE class (label 0). Needed for
        roc_auc; if omitted, roc_auc comes back as NaN rather than raising.
        Note the class: with sklearn's `predict_proba`, that is column 0
        when `clf.classes_` is [0, 1].

    Returns
    -------
    dict
        macro_f1, accuracy, precision, recall, fpr, roc_auc, mcc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    scores = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
    }

    # FPR = real articles wrongly flagged as fake.
    cm = confusion_matrix(y_true, y_pred, labels=[LABEL_FAKE, 1])
    _, _, fp_real, tn_real = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    denom = fp_real + tn_real
    scores["fpr"] = float(fp_real / denom) if denom else float("nan")

    if y_prob_fake is not None and len(set(y_true)) > 1:
        # score_of_fake vs. y_true==FAKE
        scores["roc_auc"] = float(
            roc_auc_score((y_true == LABEL_FAKE).astype(int), np.asarray(y_prob_fake))
        )
    else:
        scores["roc_auc"] = float("nan")

    return scores


def result_row(
    approach: str,
    system: str,
    scope: str,
    seed: int,
    n_train: int,
    n_test: int,
    scores: dict[str, float],
) -> dict:
    """Assemble one RESULT_FIELDS row. Missing metrics become NaN."""
    row = {
        "approach": approach,
        "system": system,
        "scope": scope,
        "seed": int(seed),
        "n_train": int(n_train),
        "n_test": int(n_test),
    }
    for key in _METRIC_KEYS:
        row[key] = float(scores.get(key, float("nan")))
    return {k: row[k] for k in RESULT_FIELDS}


def append_result(row: dict, approach: str) -> Path:
    """Append a row to results/approach_<a>.csv, creating it if needed.

    One file per approach, deliberately: a single shared results.csv would
    conflict on every push once more than one person is running experiments.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"approach_{approach.lower()}.csv"
    frame = pd.DataFrame([{k: row.get(k) for k in RESULT_FIELDS}])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def append_results(rows: list[dict], approach: str) -> Path:
    """Append several rows at once."""
    path = None
    for row in rows:
        path = append_result(row, approach)
    return path


def load_all_results() -> pd.DataFrame:
    """Concatenate every results/approach_*.csv into one frame.

    This is what the final comparison table is built from.
    """
    files = sorted(RESULTS_DIR.glob("approach_*.csv"))
    if not files:
        return pd.DataFrame(columns=RESULT_FIELDS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summary_table(scope: str = "overall") -> pd.DataFrame:
    """Mean and std of macro-F1 per (approach, system), across seeds.

    The headline table: where each approach lands between the Approach A
    floor and ceiling.
    """
    df = load_all_results()
    if df.empty:
        return df
    df = df[df["scope"] == scope]
    return (
        df.groupby(["approach", "system"])["macro_f1"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .round(4)
    )

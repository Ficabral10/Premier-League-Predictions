"""Evaluation metrics for 3-way (Home / Draw / Away) probabilistic forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean RPS. Standard for football because H > D > A is an ordered outcome:
    predicting a draw when the home side wins is "less wrong" than predicting an away win.
    0 is perfect; lower is better.
    """
    y_true = np.asarray(y_true)
    n_classes = proba.shape[1]
    onehot = np.eye(n_classes)[y_true]
    cum_p = np.cumsum(proba, axis=1)[:, :-1]
    cum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (n_classes - 1)))


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.eye(proba.shape[1])[np.asarray(y_true)]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    proba = np.clip(proba, 1e-9, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y_true, proba.argmax(axis=1))),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "brier": brier_score(y_true, proba),
        "rps": ranked_probability_score(y_true, proba),
    }

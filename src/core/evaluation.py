"""Evaluation metrics for forecasting models."""

import numpy as np

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error (SMAPE).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        SMAPE value between 0 and 2 (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero
    denom[denom == 0] = 1e-6
    return np.mean(np.abs(y_true - y_pred) / denom)
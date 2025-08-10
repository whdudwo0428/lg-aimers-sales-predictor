"""
Evaluation routines for forecasting models.

This module exposes helper functions to compute the Weighted SMAPE on
Numpy arrays as well as to summarise errors by item and horizon.  The
functions mirror the behaviour expected by the competition and can be
used to validate models locally.  Unlike the PyTorch loss defined in
:mod:`src.core.loss`, these functions operate on Numpy arrays and are
intended for offline analysis rather than training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Tuple


def compute_weight_vector(item_names: Iterable[str]) -> np.ndarray:
    """Return a per‑item weight vector for Weighted SMAPE.

    Items whose name contains ``Damha`` or ``Miracia`` receive a weight
    of 2.0; all others receive a weight of 1.0.
    """
    weights = []
    for name in item_names:
        if ("Damha" in name) or ("Miracia" in name):
            weights.append(2.0)
        else:
            weights.append(1.0)
    return np.array(weights, dtype=np.float32)


def weighted_smape(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    """Compute the Weighted SMAPE between prediction and target arrays.

    Parameters
    ----------
    pred : np.ndarray
        Array of shape ``(num_windows, horizon, num_items)`` containing
        forecasts.
    target : np.ndarray
        Array of the same shape containing ground truth values.
    weights : np.ndarray
        1‑D array of length ``num_items`` with per‑item weights.

    Returns
    -------
    float
        Weighted SMAPE averaged over all items and horizons.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    diff = np.abs(pred_flat - target_flat)
    denom = np.abs(pred_flat) + np.abs(target_flat)
    # Avoid division by zero by replacing zeros in the denominator with
    # ones (their contributions will be masked out anyway).
    denom[denom == 0] = 1.0
    smape = 2.0 * diff / denom
    # Mask out positions where the target is zero
    mask = target_flat != 0
    smape[~mask] = np.nan
    # Compute per‑item mean ignoring NaNs
    smape_per_item = np.nanmean(smape, axis=0)
    smape_per_item = np.nan_to_num(smape_per_item, nan=0.0)
    weighted = (smape_per_item * weights).sum() / weights.sum()
    return float(weighted)


def summary_by_item(pred: np.ndarray, target: np.ndarray, item_names: Iterable[str]) -> pd.DataFrame:
    """Return MAE and SMAPE for each item.

    The returned DataFrame has one row per item with columns
    ``item_name``, ``mae`` and ``smape``.  SMAPE is unweighted on a
    per‑item basis.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    mae = np.mean(np.abs(pred_flat - target_flat), axis=0)
    # Compute itemwise smape ignoring zeros
    diff = np.abs(pred_flat - target_flat)
    denom = np.abs(pred_flat) + np.abs(target_flat)
    denom[denom == 0] = 1.0
    smape = 2.0 * diff / denom
    mask = target_flat != 0
    smape[~mask] = np.nan
    smape_per_item = np.nanmean(smape, axis=0)
    smape_per_item = np.nan_to_num(smape_per_item, nan=0.0)
    return pd.DataFrame({
        "item_name": list(item_names),
        "mae": mae,
        "smape": smape_per_item,
    })


def summary_by_horizon(pred: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    """Return MAE grouped by forecast horizon.

    The returned DataFrame has columns ``day`` (1–H), ``mae`` and
    ``smape``; both metrics are averaged over all items and windows.
    """
    # Compute MAE and SMAPE along items axis
    mae_by_h = np.mean(np.abs(pred - target), axis=2).mean(axis=0)
    diff = np.abs(pred - target)
    denom = np.abs(pred) + np.abs(target)
    denom[denom == 0] = 1.0
    smape = 2.0 * diff / denom
    mask = target != 0
    smape[~mask] = np.nan
    smape_by_h = np.nanmean(smape, axis=(0, 2))
    smape_by_h = np.nan_to_num(smape_by_h, nan=0.0)
    days = [f"Day {i+1}" for i in range(pred.shape[1])]
    return pd.DataFrame({
        "day": days,
        "mae": mae_by_h,
        "smape": smape_by_h,
    })


__all__ = [
    "compute_weight_vector",
    "weighted_smape",
    "summary_by_item",
    "summary_by_horizon",
]
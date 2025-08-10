"""
Loss functions for time‑series forecasting.

This module defines a PyTorch function for computing the Weighted
Symmetric Mean Absolute Percentage Error (Weighted SMAPE).  The
competition uses this metric both for training and for leaderboard
evaluation.  A vector of item weights is passed to emphasise specific
series; items whose name contains the substrings ``Damha`` or
``Miracia`` receive twice the weight of other items.

The loss is differentiable and can be used directly during model
training.  It ignores all instances where the actual value is zero
because dividing by zero would otherwise dominate the loss.  When all
actual values for an item are zero the contribution of that item to
the loss is zero.
"""

from __future__ import annotations

# We use typing.Any to annotate arbitrary array‑like inputs to the
# weighted_smape_loss function.  Importing here avoids a NameError at
# runtime when type hints are evaluated.  The implementation remains
# runtime agnostic and will work with both numpy arrays and torch
# tensors when available.
from typing import Any

try:
    import torch  # type: ignore
    from torch import Tensor  # type: ignore
except ImportError:
    # Torch is unavailable in this environment.  Define a dummy Tensor type for
    # type checking and fall back to numpy in the implementation.  Calls to
    # weighted_smape_loss will handle torch==None gracefully.
    torch = None  # type: ignore
    Tensor = None  # type: ignore


def weighted_smape_loss(pred: Any, target: Any, weights: Any, eps: float = 1e-8) -> Any:
    """Compute the Weighted SMAPE between predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        Forecasts with shape ``(B, H, N)`` where ``B`` is the batch
        size, ``H`` the forecast horizon and ``N`` the number of
        series.  Predictions are expected to be real numbers.
    target : torch.Tensor
        Ground truth values with shape ``(B, H, N)``.  Must be on the
        same device and have the same dtype as ``pred``.
    weights : torch.Tensor
        1‑D tensor of length ``N`` containing the per‑item weights.
        Typically the vector consists of ones with two for series
        containing the substrings ``Damha`` or ``Miracia``.
    eps : float, default 1e-8
        Small constant added to the denominator to avoid division by
        zero when both prediction and target are very close to zero.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the weighted SMAPE averaged over all
        non–zero target entries.
    """
    # If torch is available and inputs are torch tensors, use the
    # torch implementation; otherwise fall back to a numpy version.
    if torch is not None and isinstance(pred, torch.Tensor):
        B, H, N = pred.shape
        pred_flat = pred.reshape(-1, N)
        target_flat = target.reshape(-1, N)
        nonzero_mask = target_flat != 0
        diff = torch.abs(pred_flat - target_flat)
        denom = torch.abs(pred_flat) + torch.abs(target_flat) + eps
        smape = torch.zeros_like(diff)
        valid = nonzero_mask
        smape[valid] = 2.0 * diff[valid] / denom[valid]
        sum_per_item = smape.sum(dim=0)
        count_per_item = valid.sum(dim=0)
        smape_per_item = torch.where(
            count_per_item > 0,
            sum_per_item / count_per_item.clamp(min=1),
            torch.zeros_like(sum_per_item),
        )
        weighted_smape = (smape_per_item * weights).sum() / weights.sum()
        return weighted_smape
    else:
        import numpy as np  # type: ignore
        pred_np = np.asarray(pred)
        target_np = np.asarray(target)
        weights_np = np.asarray(weights)
        B, H, N = pred_np.shape
        pred_flat = pred_np.reshape(-1, N)
        target_flat = target_np.reshape(-1, N)
        diff = np.abs(pred_flat - target_flat)
        denom = np.abs(pred_flat) + np.abs(target_flat) + eps
        smape = np.zeros_like(diff)
        valid = target_flat != 0
        smape[valid] = 2.0 * diff[valid] / denom[valid]
        sum_per_item = smape.sum(axis=0)
        count_per_item = valid.sum(axis=0)
        smape_per_item = np.where(
            count_per_item > 0,
            sum_per_item / np.maximum(count_per_item, 1),
            0.0,
        )
        weighted_smape = float((smape_per_item * weights_np).sum() / weights_np.sum())
        return weighted_smape


__all__ = ["weighted_smape_loss"]
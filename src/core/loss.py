"""Common loss functions used in the LG Aimers forecasting pipeline."""

import torch

def weighted_smape_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Compute weighted symmetric mean absolute percentage error (SMAPE).

    SMAPE is defined as

    .. math::

       \frac{1}{N} \sum_{i=1}^N \frac{|y_i - \hat y_i|}{(|y_i| + |\hat y_i|)/2}

    where :math:`y_i` is the true value and :math:`\hat y_i` is the
    prediction.  This implementation allows optional per‑sample
    weights which, if provided, must broadcast to the shape of the
    error tensor.

    Args:
        pred: Predicted values (batch_size, horizon).
        target: True values with the same shape as ``pred``.
        weights: Optional weights for each sample in the batch.

    Returns:
        Scalar tensor representing the mean weighted SMAPE over the batch.
    """
    denom = (torch.abs(target) + torch.abs(pred)).clamp(min=1e-6) / 2.0
    smape = torch.abs(target - pred) / denom
    if weights is not None:
        smape = smape * weights
    return smape.mean()
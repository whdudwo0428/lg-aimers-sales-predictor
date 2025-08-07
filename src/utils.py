"""
General utility functions used throughout the LG Aimers forecasting project.

This module contains functionality for reproducibility, loss computation and
metrics.  It also defines a simple evaluation helper that can be used to
compare different models on a hold‑out validation set using the weighted
SMAPE metric.  When running on a machine without a GPU, PyTorch will
automatically fall back to CPU.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
try:
    import torch  # type: ignore
except ImportError:
    # When torch is unavailable we create a minimal namespace with only the
    # pieces used in this module.  This allows the rest of the code to
    # import utils without failing.  Note that training deep learning models
    # will not work without PyTorch installed.
    class _DummyTensor:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for WeightedSMAPELoss")

    class _DummyModule:
        class Module:
            pass
        Tensor = _DummyTensor

        @staticmethod
        def tensor(*args, **kwargs):
            raise ImportError("PyTorch is required for WeightedSMAPELoss")
        @staticmethod
        def manual_seed(seed):
            pass
        class cuda:
            @staticmethod
            def is_available():
                return False
            @staticmethod
            def manual_seed_all(seed):
                pass
        class device:
            def __init__(self, dev):
                self.dev = dev

    torch = _DummyModule()  # type: ignore


def set_seed(seed: int = 42) -> None:
    """Set the random seed for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed: int
        Seed value used to initialise random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if hasattr(torch, 'nn'):
    class WeightedSMAPELoss(torch.nn.Module):
        """PyTorch implementation of a weighted symmetric mean absolute percentage error.

        This loss follows the same formulation as the baseline LSTM: each item has
        an associated weight (which may be >1 for more important items).  The
        calculation is fully vectorised and differentiable, making it suitable for
        training neural networks.
        """

        def __init__(self, weights_map: Dict[str, float], classes: Iterable[str], device: torch.device | None = None) -> None:
            super().__init__()
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            weight_list = [float(weights_map.get(cls, 1.0)) for cls in classes]
            tensor = torch.tensor(weight_list, dtype=torch.float32, device=self.device)
            self.register_buffer("weight_tensor", tensor)

        def forward(self, pred: torch.Tensor, true: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
            eps = 1e-6
            diff = torch.abs(pred - true)
            denom = torch.abs(pred) + torch.abs(true) + eps
            smape = 2 * diff / denom
            weights = self.weight_tensor[item_ids].unsqueeze(1)
            weighted = smape * weights
            return weighted.mean()
else:
    class WeightedSMAPELoss:
        """Dummy WeightedSMAPELoss placeholder when PyTorch is unavailable."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use WeightedSMAPELoss")


def weighted_smape_np(pred: np.ndarray, true: np.ndarray, item_ids: np.ndarray, weights_map: Dict[int, float]) -> float:
    """Compute weighted SMAPE on NumPy arrays.

    Parameters
    ----------
    pred: np.ndarray of shape (n_samples, horizon)
        Predicted values.
    true: np.ndarray of shape (n_samples, horizon)
        Ground truth values.
    item_ids: np.ndarray of shape (n_samples,)
        Integer identifiers mapping each sample to an item weight.
    weights_map: dict[int, float]
        Mapping from integer item_id to a weight.  Items not in the map
        default to weight 1.0.

    Returns
    -------
    float
        Weighted SMAPE value.
    """
    eps = 1e-6
    diff = np.abs(pred - true)
    denom = np.abs(pred) + np.abs(true) + eps
    smape = 2 * diff / denom
    weights = np.vectorize(lambda i: weights_map.get(int(i), 1.0))(item_ids)
    weighted_smape = smape.mean(axis=1) * weights
    return float(weighted_smape.mean())


def evaluate_model(model, X: np.ndarray, y: np.ndarray, ids: np.ndarray, weights_map: Dict[int, float], device: torch.device | None = None) -> float:
    """Evaluate a PyTorch model on a validation set using weighted SMAPE.

    The provided model is expected to accept a batch of inputs ``X`` and an
    optional ``item_ids`` tensor and return predictions of shape
    ``(batch_size, horizon)``.  This helper will move the data to the
    appropriate device, disable gradient calculation and compute the weighted
    SMAPE metric.
    """
    if X.size == 0:
        return float("inf")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        preds = model(Xt, ids_t)
        # ensure shape matches (batch,horizon)
        preds_np = preds.detach().cpu().numpy()
        true_np = yt.detach().cpu().numpy()
        ids_np = ids_t.detach().cpu().numpy()
    return weighted_smape_np(preds_np, true_np, ids_np, weights_map)
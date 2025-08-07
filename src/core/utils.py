"""Miscellaneous helper functions for the forecasting pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

def seed_everything(seed: int) -> None:
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_to_submission_format(predictions: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Ensure predictions conform to the sample submission structure.

    The competition typically expects a file with the same index/order as
    ``sample_submission``.  This function aligns the provided
    predictions to the sample index and fills any missing values with
    zeros.

    Args:
        predictions: DataFrame with a ``store_item``/``id`` column and a
            ``sales`` column containing the predictions.
        sample_submission: The original sample submission used to
            define the required index ordering.

    Returns:
        DataFrame ready to be saved as ``submission.csv``.
    """
    preds = predictions.set_index("store_item")["sales"]
    base = sample_submission.copy()
    # Some competitions use ``id`` rather than ``store_item``; handle both
    key_col = "store_item" if "store_item" in base.columns else "id"
    base["sales"] = base[key_col].map(preds).fillna(0.0)
    return base


@dataclass
class ModelConfigBase:
    """Abstract base class for model configuration.

    Subclasses should define the hyperparameters relevant to a
    specific model.  The base class provides JSON serialisation via
    ``to_dict``.
    """
    seed: int = 42
    # Input/output lengths – shared across models
    input_length: int = 28  # e.g. 4 weeks of history
    forecast_horizon: int = 7  # predict one week ahead
    # Data splitting
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    # Training details
    batch_size: int = 32
    max_epochs: int = 10
    learning_rate: float = 1e-3

    # Hardware
    accelerator: str | None = None  # e.g. 'gpu'
    devices: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
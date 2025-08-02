"""
Base model classes for the LG Aimers demand forecasting challenge.

Each forecasting model implemented under the ``models`` package should
inherit from ``BaseModel`` and implement the ``fit`` and ``predict``
methods.  The base class provides a unified interface and common
behaviour for handling training and prediction.  Models are free to
override or extend this behaviour as needed.

The goal of defining a base class is to simplify experimentation
across a diverse set of algorithms by standardising method names and
parameter handling.  You can instantiate a model, call ``fit`` with
``X_train`` and ``y_train``, then call ``predict`` with new feature
matrices to obtain forecasts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on the provided training data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target matrix of shape (n_samples, horizon).  The horizon
            dimension can be 1 for single step forecasts or greater
            than 1 for multi‑output regression.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict future values given the feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples, horizon).
        """
        raise NotImplementedError


def build_model(model_name: str, **params: Any) -> BaseModel:
    """Factory method to construct a model by name.

    Parameters
    ----------
    model_name : str
        Name of the model to build.  Valid names correspond to the
        Python modules within the ``models`` package (excluding
        ``base`` and ``common``).
    params : dict
        Hyperparameters specific to the chosen model.  These will be
        passed to the model constructor.

    Returns
    -------
    BaseModel
        An instance of a subclass of ``BaseModel``.
    """
    name = model_name.lower()
    if name == "tft":
        from .tft import TFTModel
        return TFTModel(**params)
    elif name == "nbeats":
        from .nbeats import NBeatsModel
        return NBeatsModel(**params)
    elif name == "dlinear":
        from .dlinear import DLinearModel
        return DLinearModel(**params)
    elif name == "autoformer":
        from .autoformer import AutoformerModel
        return AutoformerModel(**params)
    elif name == "fedformer":
        from .fedformer import FEDformerModel
        return FEDformerModel(**params)
    elif name == "patchtst":
        from .patchtst import PatchTSTModel
        return PatchTSTModel(**params)
    elif name == "deepar":
        from .deepar import DeepARModel
        return DeepARModel(**params)
    elif name == "gbt":
        from .gbt import GBTModel
        return GBTModel(**params)
    elif name == "sliding_transformer":
        from .sliding_transformer import SlidingTransformerModel
        return SlidingTransformerModel(**params)
    elif name == "hybrid":
        from .hybrid import HybridModel
        return HybridModel(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

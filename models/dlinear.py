"""
Deterministic linear model for multi‑step forecasting.

DLinear refers to simple linear regression applied to historical
windowed features.  Although far less expressive than deep models,
linear models can serve as strong baselines and often capture
dominant trends effectively.  This implementation uses scikit‑learn's
``Ridge`` regression with ``MultiOutputRegressor`` to support
predicting multiple horizons simultaneously.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseModel


class DLinearModel(BaseModel):
    """Simple linear regression baseline."""

    def __init__(self, horizon: int = 7, alpha: float = 1.0, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        alpha = params.get("alpha", alpha)
        base = Ridge(alpha=alpha)
        self.model = MultiOutputRegressor(base)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

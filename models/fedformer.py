"""
FEDformer approximated with Random Forests.

FEDformer (frequency enhanced decomposed transformer) introduces
frequency attention mechanisms into the transformer architecture for
long term forecasting.  In this simplified offline implementation we
use random forest regressors to capture non‑linear interactions
between the engineered features.  A separate forest is trained for
each forecast horizon.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


class FEDformerModel(BaseModel):
    """Random forest based approximation of FEDformer."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        default_params = {
            "n_estimators": 200,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params
        self.models: List[RandomForestRegressor] = [
            RandomForestRegressor(**self.params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for t in range(self.horizon):
            self.models[t].fit(X, y[:, t])

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for t in range(self.horizon):
            p = self.models[t].predict(X)
            preds.append(p.reshape(-1, 1))
        return np.hstack(preds)

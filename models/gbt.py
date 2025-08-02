"""
Gradient Boosted Tree model for demand forecasting.

This model uses the XGBoost library to train a separate gradient
boosted regressor for each forecast horizon.  XGBoost is a powerful
ensemble method capable of capturing complex non‑linear patterns.

Tuneable hyperparameters include number of estimators, learning
rate, maximum depth and subsample fraction.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from xgboost import XGBRegressor

from .base import BaseModel


class GBTModel(BaseModel):
    """XGBoost based gradient boosted tree model."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params
        self.models: List[XGBRegressor] = [
            XGBRegressor(**self.params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for t in range(self.horizon):
            self.models[t].fit(X, y[:, t])

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for t in range(self.horizon):
            preds.append(self.models[t].predict(X).reshape(-1, 1))
        return np.hstack(preds)

"""
Autoformer approximated with LightGBM.

Autoformer is a transformer architecture tailored for long sequence
forecasting.  Without access to transformer libraries, we substitute a
powerful gradient boosting model (LightGBM) and train one regressor
per forecast horizon.  LightGBM can efficiently capture non‑linear
relationships among features and is well suited to tabular data.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from lightgbm import LGBMRegressor

from .base import BaseModel


class AutoformerModel(BaseModel):
    """LightGBM based approximation of the Autoformer."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "objective": "regression_l1",
            "verbosity": -1,
        }
        default_params.update(params)
        self.params = default_params
        # One model per horizon step
        self.models: List[LGBMRegressor] = [
            LGBMRegressor(**self.params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for t in range(self.horizon):
            target = y[:, t]
            self.models[t].fit(X, target)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for t in range(self.horizon):
            pred = self.models[t].predict(X)
            preds.append(pred.reshape(-1, 1))
        return np.hstack(preds)

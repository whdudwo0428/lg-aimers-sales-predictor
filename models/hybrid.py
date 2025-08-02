"""
Hybrid model combining gradient boosting and neural residual correction.

This hybrid approach first trains a LightGBM regressor on the raw
features to obtain a baseline forecast.  The residuals (errors) on the
training set are then modelled using a lightweight multi‑layer
perceptron (MLP).  The final prediction is computed by summing the
baseline and residual predictions.  This can help capture complex
patterns not fully learned by the boosting model.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

from .base import BaseModel


class HybridModel(BaseModel):
    """LightGBM + MLP hybrid model."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        # Base model parameters
        base_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 31,
            "objective": "regression_l1",
            "verbosity": -1,
        }
        base_params.update(params.get("base_params", {}))
        self.base_models: List[LGBMRegressor] = [
            LGBMRegressor(**base_params) for _ in range(self.horizon)
        ]
        # Residual model parameters
        res_params = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
        }
        res_params.update(params.get("res_params", {}))
        self.res_models: List[MLPRegressor] = [
            MLPRegressor(**res_params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Fit base models and compute residuals
        baseline_preds = np.zeros_like(y)
        for t in range(self.horizon):
            self.base_models[t].fit(X, y[:, t])
            baseline_preds[:, t] = self.base_models[t].predict(X)
        residuals = y - baseline_preds
        # Fit residual models
        for t in range(self.horizon):
            self.res_models[t].fit(X, residuals[:, t])

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_pred = []
        res_pred = []
        for t in range(self.horizon):
            base = self.base_models[t].predict(X)
            res = self.res_models[t].predict(X)
            base_pred.append(base.reshape(-1, 1))
            res_pred.append(res.reshape(-1, 1))
        base_pred = np.hstack(base_pred)
        res_pred = np.hstack(res_pred)
        return base_pred + res_pred

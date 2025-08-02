"""
Approximate implementation of a Temporal Fusion Transformer (TFT) using
gradient boosting on tabular features.

The true Temporal Fusion Transformer is a deep learning model that
leverages recurrent and attention mechanisms to capture both
long‑term dependencies and variable selection.  In this offline
environment where GPU acceleration and deep learning frameworks are
unavailable, we approximate the TFT with a powerful gradient boosted
decision tree model (CatBoost) trained on a rich set of engineered
features.

This class trains one independent CatBoost regressor per forecast
horizon.  Each regressor learns to map the flattened 28‑day history
(plus date and categorical features) to a single target day.  During
prediction, the models are executed for each horizon and their
predictions concatenated.

Hyperparameters accepted by this model include:

* ``iterations``: number of boosting iterations (trees)
* ``learning_rate``: shrinkage parameter controlling learning rate
* ``depth``: maximum depth of each tree
* ``loss_function``: set to ``MAE`` by default for robustness

Refer to the CatBoost documentation for further tunable parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from catboost import CatBoostRegressor

from .base import BaseModel


class TFTModel(BaseModel):
    """Gradient boosted regression model approximating a TFT."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        # Default parameters for CatBoost
        default_params = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MAE",
            "verbose": False,
        }
        # Merge user params
        default_params.update(params)
        self.params = default_params
        # We maintain one model per horizon
        self.models: List[CatBoostRegressor] = [
            CatBoostRegressor(**self.params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # y has shape (n_samples, horizon)
        for t in range(self.horizon):
            target = y[:, t]
            self.models[t].fit(X, target)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for t in range(self.horizon):
            p = self.models[t].predict(X)
            preds.append(p.reshape(-1, 1))
        return np.hstack(preds)

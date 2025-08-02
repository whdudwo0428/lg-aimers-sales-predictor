"""
Sliding Transformer ensemble approximated with enriched LightGBM.

The notion behind a sliding transformer ensemble is to capture local
trends and seasonality via attention‑like mechanisms applied on
sliding windows.  In this simplified implementation we enrich the
existing tabular features with first‑order differences of the past
sales history.  These additional features help the gradient boosting
model learn change dynamics between consecutive days.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from lightgbm import LGBMRegressor

from .base import BaseModel


class SlidingTransformerModel(BaseModel):
    """LightGBM model with lag difference features."""

    def __init__(self, horizon: int = 7, lookback: int = 28, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        self.lookback = lookback
        default_params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 64,
            "objective": "regression_l1",
            "verbosity": -1,
        }
        default_params.update(params)
        self.params = default_params
        self.models: List[LGBMRegressor] = [
            LGBMRegressor(**self.params) for _ in range(self.horizon)
        ]

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Append first differences of past sales to the feature matrix.

        The past sales history is assumed to occupy the first
        ``lookback`` positions of each row in ``X``.  The resulting
        augmented matrix has shape (n_samples, n_features + lookback-1).
        """
        if X.shape[1] <= self.lookback:
            return X
        past = X[:, :self.lookback]
        diff = np.diff(past, axis=1)
        return np.hstack([X, diff])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_aug = self._augment_features(X)
        for t in range(self.horizon):
            self.models[t].fit(X_aug, y[:, t])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = self._augment_features(X)
        preds = []
        for t in range(self.horizon):
            preds.append(self.models[t].predict(X_aug).reshape(-1, 1))
        return np.hstack(preds)

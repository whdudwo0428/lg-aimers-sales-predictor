"""
PatchTST approximated with Extra Trees Regressors.

PatchTST (patching transformer for time series) processes input
sequences in non‑overlapping patches.  To capture a similar spirit in
this environment we employ extremely randomized trees (ExtraTrees) to
reduce variance and improve generalisation.  One regressor is
constructed per forecast horizon.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .base import BaseModel


class PatchTSTModel(BaseModel):
    """ExtraTrees based approximation of PatchTST."""

    def __init__(self, horizon: int = 7, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        default_params = {
            "n_estimators": 300,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params
        self.models: List[ExtraTreesRegressor] = [
            ExtraTreesRegressor(**self.params) for _ in range(self.horizon)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for t in range(self.horizon):
            self.models[t].fit(X, y[:, t])

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for t in range(self.horizon):
            preds.append(self.models[t].predict(X).reshape(-1, 1))
        return np.hstack(preds)

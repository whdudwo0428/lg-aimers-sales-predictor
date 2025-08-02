"""
Simplified N‑BEATS model implemented using a multi‑layer perceptron.

N‑BEATS is a deep neural architecture for univariate and multivariate
time series forecasting.  It stacks multiple fully connected blocks
with backcasting and forecasting branches to capture trend and
seasonality.  In this environment without PyTorch, we approximate
N‑BEATS by training a feedforward neural network on flattened
historical windows and predicting multiple future steps in one shot.

The implementation uses scikit‑learn's ``MLPRegressor`` wrapped in
``MultiOutputRegressor`` to support multi‑step forecasting.  The
hidden layer sizes and other hyperparameters can be tuned via the
constructor.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseModel


class NBeatsModel(BaseModel):
    """Approximate N‑BEATS using a multi‑layer perceptron."""

    def __init__(self, horizon: int = 7, hidden_layer_sizes: tuple = (128, 64), max_iter: int = 300, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        self.hidden_layer_sizes = params.get("hidden_layer_sizes", hidden_layer_sizes)
        self.max_iter = params.get("max_iter", max_iter)
        # Base estimator
        estimator = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            learning_rate=params.get("learning_rate", "adaptive"),
            learning_rate_init=params.get("learning_rate_init", 0.001),
            max_iter=self.max_iter,
            random_state=params.get("random_state", 42),
            early_stopping=params.get("early_stopping", True),
            verbose=False,
        )
        # Wrap into multi output regressor
        self.model = MultiOutputRegressor(estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

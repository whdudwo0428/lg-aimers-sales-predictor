"""
Simplified DeepAR model implemented via univariate ARIMA per item.

DeepAR is a probabilistic forecasting method based on autoregressive
recurrent neural networks.  Due to the absence of a deep learning
framework, we approximate DeepAR using classical time series modelling.
For each item, a seasonal ARIMA model is fitted on its entire sales
history.  The fitted model is then used to forecast the specified
horizon.  This implementation ignores exogenous features and treats
each item independently.

Fitting ARIMA models for all items can be computationally expensive,
especially with auto_arima.  To mitigate this, you can set
``max_p``, ``max_q``, ``seasonal`` and other hyperparameters via the
constructor.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
"""
DeepAR approximation based on ARIMA models.

This module tries to import the optional dependency ``pmdarima`` when the module
is first loaded.  On some platforms and Python versions, binary wheels for
``pmdarima`` may not be available or may be incompatible with the installed
NumPy version (for example, NumPy 2.0+).  In such cases the import will fail
with a binary compatibility error.  To avoid crashing the entire experiment
pipeline when DeepAR is not being used, this module traps the import error
and degrades gracefully.

If ``pmdarima`` cannot be imported, the ``DeepARModel`` will still be
instantiable.  It will bypass ARIMA fitting and instead rely on a simple
naive forecast for each item (repeating the last observed value) when
``fit_series`` is called.  A warning is emitted to inform the user that
ARIMA-based modelling is not available.
"""

try:
    import pmdarima as pm  # type: ignore
except Exception as e:  # pragma: no cover - pmdarima might not be installed or might be incompatible
    pm = None  # type: ignore
    import warnings
    warnings.warn(
        "pmdarima could not be imported. DeepAR will fall back to a naive forecast.\n"
        f"Import error: {e}",
        RuntimeWarning,
    )

from .base import BaseModel


class DeepARModel(BaseModel):
    """ARIMA based approximation of DeepAR."""

    def __init__(self, horizon: int = 7, max_p: int = 2, max_q: int = 2, seasonal: bool = False, **params: Any) -> None:
        super().__init__(**params)
        self.horizon = horizon
        self.max_p = params.get("max_p", max_p)
        self.max_q = params.get("max_q", max_q)
        self.seasonal = params.get("seasonal", seasonal)
        # Models per item
        self.models: Dict[str, Any] = {}
        # Precomputed forecasts per item
        self.forecasts: Dict[str, np.ndarray] = {}

    def fit_series(self, series_dict: Dict[str, pd.Series]) -> None:
        """Fit an ARIMA model for each item.

        Parameters
        ----------
        series_dict : dict
            Mapping from item name to a pandas Series of sales history
            indexed by date.  Missing dates should be filled with zero.
        """
        for item, series in series_dict.items():
            # Convert to numpy array
            values = series.values.astype(float)
            model = None
            # Only attempt to fit an ARIMA model if pmdarima is available
            if pm is not None:
                try:
                    model = pm.auto_arima(
                        values,
                        start_p=1,
                        start_q=1,
                        max_p=self.max_p,
                        max_q=self.max_q,
                        seasonal=self.seasonal,
                        error_action="ignore",
                        suppress_warnings=True,
                        maxiter=25,
                    )
                except Exception:
                    # Fallback to naive model if auto_arima fails
                    model = None
            # Save the model (could be None)
            self.models[item] = model
            # Precompute forecast: use model if available, else naive forecast
            if model is not None:
                try:
                    forecast = model.predict(self.horizon)
                except Exception:
                    forecast = np.repeat(values[-1] if len(values) > 0 else 0.0, self.horizon)
            else:
                # naive: repeat last observed value across horizon
                last_value = values[-1] if len(values) > 0 else 0.0
                forecast = np.repeat(last_value, self.horizon)
            self.forecasts[item] = np.array(forecast)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # DeepAR does not use sliding window X or y; fitting requires full series.
        # This method should be called after calling fit_series() externally.
        pass

    def predict(self, X: np.ndarray, item_ids: List[str]) -> np.ndarray:
        """Predict using precomputed ARIMA forecasts.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (ignored).
        item_ids : list of str
            Ordered list of item names corresponding to rows in X.

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, horizon).
        """
        preds = []
        for item in item_ids:
            pred = self.forecasts.get(item)
            if pred is None:
                # Unknown item: output zeros
                pred = np.zeros(self.horizon)
            preds.append(pred.reshape(1, -1))
        return np.vstack(preds)

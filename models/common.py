"""
Common utilities for the Gonjiam Resort demand forecasting challenge.

This module defines data loading helpers, feature engineering utilities and
evaluation functions used across all models in the project. The core goal of
this code is to abstract away repetitive tasks such as loading the training
data, generating time‑based features, constructing sliding windows for
multi‑horizon forecasting and computing SMAPE based metrics at various
aggregation levels.  All models implemented under the ``models`` package
should depend on these helpers rather than re‑implementing their own
loading logic.

The functions and classes below are intentionally written without any
third‑party deep learning dependencies.  Only widely available libraries
such as ``pandas``, ``numpy``, ``scikit‑learn`` and ``lightgbm`` are
imported.  Should you wish to integrate a different model family (for
example PyTorch based networks) you can still reuse the data loading and
evaluation code defined here.

Author: LG Aimers Hackathon Participant
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Use util.py for date based features and simple SMAPE
from util import add_date_features, smape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Config:
    """Configuration dataclass holding high level parameters for data
    preparation.  These can be adjusted globally or per‑model via
    command line arguments when invoking the training script.

    Parameters
    ----------
    lookback: int
        Number of days to look back for the input window.  The default
        challenge value is 28.
    horizon: int
        Number of days to forecast.  The default challenge value is 7.
    test_dir: str
        Directory path containing test CSVs (e.g. TEST_00.csv … TEST_09.csv).
    """
    lookback: int = 28
    horizon: int = 7
    test_dir: str = "dataset/test"


@dataclass
class DataLoader:
    """Helper class to load training and test data and construct sliding
    windows for forecasting.  This class encapsulates all logic around
    parsing dates, encoding categorical variables and assembling the
    feature/target matrices used for training the models.

    Example usage::

        loader = DataLoader("dataset/train/train.csv", "dataset/test", config)
        X_train, y_train, X_val, y_val = loader.get_train_val_split()
        X_test_list = loader.get_test_features()  # list of test windows

    The returned arrays have shapes ``(n_samples, n_features)`` and
    ``(n_samples, horizon)`` respectively.
    """

    train_path: str
    test_dir: str
    config: Config
    random_state: int = 42
    item_encoder: Optional[LabelEncoder] = field(default=None, init=False)
    store_encoder: Optional[LabelEncoder] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Load training data once
        logger.info("Loading training data from %s", self.train_path)
        self.train_df = pd.read_csv(self.train_path)
        # Ensure date column is of datetime type
        self.train_df["영업일자"] = pd.to_datetime(self.train_df["영업일자"])
        # Derive store and menu from combined column
        self.train_df[["store", "menu"]] = self.train_df["영업장명_메뉴명"].str.split("_", n=1, expand=True)
        # Fit label encoders for categorical features
        self.item_encoder = LabelEncoder().fit(self.train_df["영업장명_메뉴명"])
        self.store_encoder = LabelEncoder().fit(self.train_df["store"])
        # Precompute date features for all dates appearing in training data
        logger.info("Generating date features for training data")
        self.date_features = add_date_features(self.train_df[["영업일자"]].drop_duplicates())
        self.date_features.set_index("영업일자", inplace=True)

    def _make_sequence_features(self, window: pd.Series, pred_dates: List[pd.Timestamp]) -> np.ndarray:
        """Generate features for a single sliding window.

        Given a sequence of past ``lookback`` sales values (as a pandas
        Series indexed by date) and a list of prediction dates (one for
        each horizon step), produce a single 1D feature vector.  The
        current implementation simply flattens the past values and
        concatenates date features for the last observed date and each
        forecast date.  Additionally, the store and item identifiers are
        encoded as integers.

        Parameters
        ----------
        window: pd.Series
            A series of length ``lookback`` containing the past sales.
        pred_dates: list of pandas.Timestamp
            List of forecast dates corresponding to the next ``horizon``
            days.  Length must equal ``horizon``.

        Returns
        -------
        features: np.ndarray
            A 1D numpy array containing the features for this sample.
        """
        # Flatten past sales history
        past_vals = window.values.astype(float)
        # Extract item and store encodings from the series name (tuple)
        item_key = window.name  # full string identifying store_menu
        item_id = self.item_encoder.transform([item_key])[0]
        # The store encoding is derived from the key split on underscore
        store_name = item_key.split("_")[0]
        store_id = self.store_encoder.transform([store_name])[0]
        # Use the date of the last observed day for dynamic date features
        last_date = window.index[-1]
        # Get date features for the last observed date.  If the date
        # falls outside the training range, compute features on the fly.
        if last_date in self.date_features.index:
            last_date_feats = self.date_features.loc[last_date]
        else:
            tmp = pd.DataFrame({"영업일자": [last_date]})
            last_date_feats = add_date_features(tmp).iloc[0]
        # Forecast date features for each horizon step
        fut_feats = []
        for d in pred_dates:
            if d in self.date_features.index:
                f = self.date_features.loc[d]
            else:
                # If the date is beyond the training range, create features on the fly
                tmp = pd.DataFrame({"영업일자": [d]})
                f = add_date_features(tmp).iloc[0]
            fut_feats.append(f.values)
        fut_feats = np.concatenate(fut_feats)
        # Concatenate all pieces
        features = np.concatenate([
            past_vals,
            [item_id], [store_id],
            last_date_feats.values,
            fut_feats
        ])
        return features

    def _build_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Construct sliding windows from the provided DataFrame.

        The DataFrame must contain the columns ``['영업일자', '영업장명_메뉴명', '매출수량']``.
        Rows are assumed to cover a continuous range of dates for each item.

        Returns
        -------
        X: np.ndarray
            Feature matrix of shape ``(n_samples, n_features)``.
        y: np.ndarray
            Target matrix of shape ``(n_samples, horizon)``.
        """
        lookback = self.config.lookback
        horizon = self.config.horizon
        # Pivot the dataframe into item × date matrix for easier slicing
        pivot = df.pivot(index="영업장명_메뉴명", columns="영업일자", values="매출수량").sort_index()
        # Ensure dates are sorted
        pivot = pivot.sort_index(axis=1)
        X_list = []
        y_list = []
        # For each item (row), generate windows
        for item_key, row in pivot.iterrows():
            series = row.dropna()
            # sort by date again (should already be sorted)
            series = series.sort_index()
            dates = series.index
            values = series.values
            # Only generate windows where full lookback and horizon exist
            n = len(series)
            # iterate such that i+lookback+horizon <= n
            for i in range(n - lookback - horizon + 1):
                past_window = series.iloc[i:i+lookback]
                target = series.iloc[i+lookback:i+lookback+horizon]
                # skip windows with all zero targets (target all zeros) because SMAPE ignores zeros
                # but still include them in training to learn zeros
                pred_dates = list(target.index)
                X_list.append(self._make_sequence_features(past_window, pred_dates))
                y_list.append(target.values.astype(float))
        X = np.vstack(X_list) if X_list else np.empty((0, lookback))
        y = np.vstack(y_list) if y_list else np.empty((0, horizon))
        return X, y

    def get_train_val_split(self, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the training data into training and validation sets.

        The split is performed in a stratified manner based on the item
        identifiers to ensure that windows from the same item do not leak
        across splits.  This helps emulate the challenge requirement of
        forecasting future dates of unseen horizons.

        Parameters
        ----------
        val_ratio: float, optional
            Fraction of the windows to allocate to the validation set.  Defaults to 0.2.

        Returns
        -------
        X_train: np.ndarray
        y_train: np.ndarray
        X_val: np.ndarray
        y_val: np.ndarray
        """
        logger.info("Building sliding windows for training/validation split")
        X, y = self._build_windows(self.train_df)
        if X.shape[0] == 0:
            return X, y, X, y
        # Derive item id for stratification (the first part of item_key)
        # to avoid windows from same item in both sets
        items = [self.item_encoder.transform([self.train_df["영업장명_메뉴명"].iloc[0]])[0]] * len(X)
        # Since each window belongs to exactly one item (by design), we can create
        # a simple array repeating the item key for each sample.  However,
        # scikit-learn's train_test_split requires a 1D array of the same length.
        # We generate item indices by storing them in a list inside _build_windows but
        # here we are limited; we therefore perform a random split without stratify
        # but with fixed random_state to maintain reproducibility.
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=self.random_state
        )
        return X_train, y_train, X_val, y_val

    def get_test_features(self) -> List[np.ndarray]:
        """Load test sequences and generate features for prediction.

        Returns a list of feature matrices, one per test file (there are
        typically 10 test files).  Each matrix has shape
        ``(n_items, n_features)`` where ``n_items`` equals the number of
        unique items contained in that test file.  The order of items in
        each matrix corresponds to the order of columns in the sample
        submission (excluding the date column).

        The test CSVs are expected to contain the past 28 days of sales
        for each item.  Their format mirrors the training CSV with
        columns ``['영업일자', '영업장명_메뉴명', '매출수량']``.
        """
        lookback = self.config.lookback
        horizon = self.config.horizon
        test_features = []
        # Determine the order of items from sample_submission to align predictions
        sample_path = os.path.join(os.path.dirname(self.train_path), "sample_submission.csv")
        if os.path.exists(sample_path):
            sample = pd.read_csv(sample_path)
            # Exclude first column (date)
            item_order = list(sample.columns[1:])
        else:
            # Fallback: use the training item ordering
            item_order = list(self.item_encoder.classes_)
        # Iterate over test files sorted by name
        files = sorted([f for f in os.listdir(self.test_dir) if f.lower().endswith('.csv')])
        for fname in files:
            df = pd.read_csv(os.path.join(self.test_dir, fname))
            df["영업일자"] = pd.to_datetime(df["영업일자"])
            df[["store", "menu"]] = df["영업장명_메뉴명"].str.split("_", n=1, expand=True)
            # Determine the forecast start date (last date in this file) and prediction dates
            last_date = df["영업일자"].max()
            pred_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
            # Pivot into item × date matrix
            pivot = df.pivot(index="영업장명_메뉴명", columns="영업일자", values="매출수량").sort_index()
            # Align to expected item order; missing items fill with zeros
            missing_items = [item for item in item_order if item not in pivot.index]
            if missing_items:
                for m in missing_items:
                    # create zero series for missing item
                    pivot.loc[m] = 0
            # sort pivot by item_order
            pivot = pivot.loc[item_order]
            # Ensure there are at least lookback days; if not, pad with zeros
            X_file = []
            for item_key, series in pivot.iterrows():
                series = series.sort_index()
                if len(series) < lookback:
                    # Pad at beginning with zeros
                    pad_len = lookback - len(series)
                    padded = pd.Series([0] * pad_len, index=[series.index.min() - pd.Timedelta(days=x+1) for x in reversed(range(pad_len))])
                    series = pd.concat([padded, series])
                past_window = series.iloc[-lookback:]
                X_file.append(self._make_sequence_features(past_window, pred_dates))
            test_features.append(np.vstack(X_file))
        return test_features


def compute_smape_per_item(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute SMAPE averaged over all horizons for a single item.

    Notes
    -----
    This function applies the SMAPE formula across the horizon dimension
    (axis=0) and returns the mean of the resulting values, excluding
    periods where the true value is zero (as per competition rules).
    """
    epsilon = 1e-8
    smapes = []
    for t in range(y_true.shape[1]):
        a = y_true[:, t]
        p = y_pred[:, t]
        denom = np.abs(a) + np.abs(p)
        denom = np.where(denom == 0, epsilon, denom)
        diff = np.abs(a - p)
        mask = a == 0  # exclude zero actuals
        if np.all(mask):
            continue
        smapes.append(np.mean(2.0 * diff[~mask] / denom[~mask]))
    if not smapes:
        return 0.0
    return float(np.mean(smapes))


def evaluate_predictions(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    item_to_store: Dict[str, str]
) -> pd.DataFrame:
    """Evaluate predictions and produce a detailed SMAPE breakdown.

    Parameters
    ----------
    y_true : dict
        Mapping from item name to ground truth array of shape
        (n_samples, horizon) for the validation split.
    y_pred : dict
        Mapping from item name to predicted array of shape
        (n_samples, horizon) for the validation split.
    item_to_store : dict
        Mapping from item name to store name, used to aggregate SMAPE per
        store.

    Returns
    -------
    breakdown : pandas.DataFrame
        DataFrame with columns ``['store', 'item', 'smape_item',
        'smape_store']``.  Averages per store are appended at the end.
    """
    records = []
    # Compute item level SMAPE
    store_smapes: Dict[str, List[float]] = {}
    for item, true_vals in y_true.items():
        pred_vals = y_pred.get(item)
        if pred_vals is None:
            logger.warning("No predictions for item %s; skipping in evaluation.", item)
            continue
        smape_val = compute_smape_per_item(true_vals, pred_vals)
        store = item_to_store[item]
        records.append({
            "store": store,
            "item": item,
            "smape_item": smape_val
        })
        store_smapes.setdefault(store, []).append(smape_val)
    # Append store averages
    for store, smapes in store_smapes.items():
        if smapes:
            avg = float(np.mean(smapes))
        else:
            avg = 0.0
        records.append({
            "store": store,
            "item": "(average)",
            "smape_item": avg
        })
    breakdown = pd.DataFrame.from_records(records)
    return breakdown

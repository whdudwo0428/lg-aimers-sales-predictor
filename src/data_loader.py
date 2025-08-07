"""
Data loading and preprocessing utilities for the LG Aimers hackathon time‑series
forecasting project.

This module encapsulates all dataset related logic used across the different
models.  It loads the provided CSV files, performs simple feature engineering
and returns train/validation sequences ready for consumption by deep learning
architectures such as PatchTST, TimesFM and Autoformer.  The code is heavily
influenced by the original LSTM baseline provided in ``lstm_custom.py`` but
extracted into a reusable API.

The implementation assumes the following folder structure relative to the
project root::

    dataset/
        train/train.csv
        test/TEST_00.csv … TEST_09.csv
        sample_submission.csv

When a full dataset is not available (for instance during local testing), the
functions defined here will still operate on empty DataFrames and provide
graceful fallbacks.
"""

from __future__ import annotations

import os
import glob
import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DataConfig:
    """Configuration for data loading and windowing."""
    lookback: int = 28  # number of historical days used as input
    horizon: int = 7    # forecast horizon (days)
    stride: int = 1     # sliding window stride

    train_path: str = os.path.join("dataset", "train", "train.csv")
    test_dir: str = os.path.join("dataset", "test")
    submission_path: str = os.path.join("dataset", "sample_submission.csv")

    def resolve_paths(self, root: str) -> Tuple[str, str, str]:
        """Return absolute paths for the train, test and submission files."""
        return (
            os.path.join(root, self.train_path),
            os.path.join(root, self.test_dir),
            os.path.join(root, self.submission_path),
        )


def load_raw_data(cfg: DataConfig, root: str) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Load the train DataFrame, list of test file paths and sample submission.

    Parameters
    ----------
    cfg: DataConfig
        Configuration containing relative dataset locations.
    root: str
        Absolute path to the project root.

    Returns
    -------
    train_df: pd.DataFrame
        The raw training data.  Expected columns include ``영업일자`` (date),
        ``영업장명_메뉴명`` (item identifier) and ``매출수량`` (target).
    test_files: List[str]
        Sorted list of absolute test CSV file paths.
    sample_submission: pd.DataFrame
        DataFrame representing the provided sample submission.
    """
    train_path, test_dir, submission_path = cfg.resolve_paths(root)
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
    else:
        # If the data is missing create an empty DataFrame with the expected
        # schema.  This allows code that depends on the presence of these
        # columns to run without error and is useful for unit testing.
        train_df = pd.DataFrame(columns=["영업일자", "영업장명_메뉴명", "매출수량"])
    # test files may not exist when running locally; we return an empty list
    if os.path.exists(test_dir):
        test_files = sorted(glob.glob(os.path.join(test_dir, "TEST_*.csv")))
    else:
        test_files = []
    if os.path.exists(submission_path):
        sample_submission = pd.read_csv(submission_path)
    else:
        sample_submission = pd.DataFrame()
    return train_df, test_files, sample_submission


def _make_calendar_features(dates: pd.Series) -> np.ndarray:
    """Generate calendar based features for an array of dates.

    This replicates the feature engineering used in the baseline LSTM:
    weekday, month and day of year are converted to sine/cosine cycles; flags
    for weekends and Korean holidays are also included.  The set of holidays
    covers the years 2023–2025; if a wider range is required it should be
    extended accordingly.

    Parameters
    ----------
    dates: pd.Series
        A series of datetime64 values.

    Returns
    -------
    numpy.ndarray
        An array of shape (len(dates), 10) containing the engineered features.
    """
    ds = pd.to_datetime(dates)
    dow = ds.dt.dayofweek.values
    month = ds.dt.month.values
    doy = ds.dt.dayofyear.values
    is_wknd = (dow >= 5).astype(int)
    # List of known Korean holidays; extend if necessary
    korean_holidays = {
        _dt.date(2023, 1, 1), _dt.date(2023, 1, 21), _dt.date(2023, 1, 22), _dt.date(2023, 1, 23),
        _dt.date(2023, 1, 24), _dt.date(2023, 3, 1), _dt.date(2023, 5, 5), _dt.date(2023, 5, 27),
        _dt.date(2023, 6, 6), _dt.date(2023, 8, 15), _dt.date(2023, 9, 28), _dt.date(2023, 9, 29),
        _dt.date(2023, 9, 30), _dt.date(2023, 10, 2), _dt.date(2023, 10, 3), _dt.date(2023, 10, 9),
        _dt.date(2023, 12, 25), _dt.date(2024, 1, 1), _dt.date(2024, 2, 9), _dt.date(2024, 2, 10),
        _dt.date(2024, 2, 11), _dt.date(2024, 2, 12), _dt.date(2024, 3, 1), _dt.date(2024, 4, 10),
        _dt.date(2024, 5, 5), _dt.date(2024, 5, 6), _dt.date(2024, 5, 15), _dt.date(2024, 6, 6),
        _dt.date(2024, 8, 15), _dt.date(2024, 9, 16), _dt.date(2024, 9, 17), _dt.date(2024, 9, 18),
        _dt.date(2024, 10, 3), _dt.date(2024, 10, 9), _dt.date(2024, 12, 25), _dt.date(2025, 1, 1),
        _dt.date(2025, 1, 28), _dt.date(2025, 1, 29), _dt.date(2025, 1, 30), _dt.date(2025, 3, 3),
        _dt.date(2025, 5, 5), _dt.date(2025, 5, 6), _dt.date(2025, 6, 6), _dt.date(2025, 8, 15),
        _dt.date(2025, 10, 3), _dt.date(2025, 10, 6), _dt.date(2025, 10, 7), _dt.date(2025, 10, 8),
        _dt.date(2025, 10, 9), _dt.date(2025, 12, 25),
    }
    is_hol = ds.dt.date.isin(korean_holidays).astype(int)
    # Trigonometric encoding for periodic features
    features = np.stack(
        [
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(4 * np.pi * dow / 7), np.cos(4 * np.pi * dow / 7),
            np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
            np.sin(2 * np.pi * (doy - 1) / 365), np.cos(2 * np.pi * (doy - 1) / 365),
            is_wknd, is_hol,
        ],
        axis=1,
    )
    return features


def _create_lag_and_rolling_features(values: pd.Series) -> pd.DataFrame:
    """Create lagged and rolling features for a univariate series.

    Five lags (1, 7, 14, 21, 28 days) are computed as well as the 7‑day
    rolling mean and variance.  Missing values at the beginning of the series
    are forward filled with the earliest available observations.
    """
    df = pd.DataFrame({"value": values})
    for lag in (1, 7, 14, 21, 28):
        df[f"lag_{lag}"] = df["value"].shift(lag)
    df["roll_7"] = df["value"].shift(1).rolling(7, min_periods=1).mean()
    df["roll_var_7"] = df["value"].shift(1).rolling(7, min_periods=1).var().fillna(0)
    # forward fill initial NaNs to avoid dropping too much data
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df


def encode_items(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encode the categorical ``영업장명_메뉴명`` column and add ``item_id``.

    Returns the transformed DataFrame and the fitted encoder.  If the input
    DataFrame is empty the encoder will still be created but no encoding
    performed.
    """
    le = LabelEncoder()
    if not train_df.empty:
        train_df = train_df.copy()
        train_df["item_id"] = le.fit_transform(train_df["영업장명_메뉴명"])
    else:
        # fit on an empty list to initialise classes_ attribute
        le.fit([])
        train_df = train_df.copy()
        train_df["item_id"] = []
    return train_df, le


def generate_windows(train_df: pd.DataFrame, le: LabelEncoder, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding windows for all items in the training set.

    Parameters
    ----------
    train_df: pd.DataFrame
        DataFrame with columns ``영업일자``, ``매출수량`` and ``item_id``.
    le: LabelEncoder
        Encoder mapping item names to integer ids.
    cfg: DataConfig
        Windowing configuration specifying lookback, horizon and stride.

    Returns
    -------
    X: np.ndarray
        3‑D array of shape (num_windows, lookback, feature_dim).
    y: np.ndarray
        2‑D array of shape (num_windows, horizon) containing targets.
    item_ids: np.ndarray
        1‑D array of shape (num_windows,) specifying the item id for each window.
    """
    if train_df.empty:
        return np.empty((0, cfg.lookback, 0)), np.empty((0, cfg.horizon)), np.empty((0,), dtype=int)
    feature_list = []
    target_list = []
    id_list = []
    # group by item to maintain temporal ordering within each series
    for item_id, group in train_df.groupby("item_id"):
        # sort by date
        group = group.sort_values("영업일자").reset_index(drop=True)
        values = group["매출수량"].astype(float)
        dates = pd.to_datetime(group["영업일자"])
        # create base DataFrame with value and calendar features
        lag_roll = _create_lag_and_rolling_features(values)
        cal = _make_calendar_features(dates)
        item_wd = (item_id * 7 + dates.dt.dayofweek.values).reshape(-1, 1)
        full_features = np.hstack([values.values.reshape(-1, 1), cal, lag_roll.iloc[:, 1:].values, item_wd])
        # drop initial rows where lag features are not valid
        valid_start = 0
        total_len = len(full_features)
        # slide window
        for start in range(0, total_len - cfg.lookback - cfg.horizon + 1, cfg.stride):
            end = start + cfg.lookback
            window = full_features[start:end]
            target = values.iloc[end:end + cfg.horizon].values
            feature_list.append(window)
            target_list.append(target)
            id_list.append(item_id)
    X = np.array(feature_list)
    y = np.array(target_list)
    ids = np.array(id_list, dtype=int)
    return X, y, ids


def standardize_train(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Standardize the first column of X and the target y.

    Since the calendar and lag features are already bounded between −1 and 1
    (via sine/cosine encoding and rolling statistics), we only scale the
    ``value`` column and the target values.  Both are individually fitted
    scalers and returned for later inverse transformation.
    """
    if X.size == 0:
        return X, y, StandardScaler(), StandardScaler()
    sx = StandardScaler()
    sy = StandardScaler()
    # scale the first feature (original series value)
    X_scaled = X.copy()
    first_feat = X[:, :, 0]
    X_scaled[:, :, 0] = sx.fit_transform(first_feat)
    # scale target
    y_scaled = sy.fit_transform(y)
    return X_scaled, y_scaled, sx, sy


def prepare_datasets(cfg: DataConfig, root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler, LabelEncoder]:
    """End‑to‑end dataset preparation.

    This helper loads the raw data, encodes items, generates windows and
    standardizes the features.  It returns the arrays ready for model
    consumption along with the fitted scalers and label encoder.
    """
    train_df, _, _ = load_raw_data(cfg, root)
    train_df, le = encode_items(train_df)
    X, y, ids = generate_windows(train_df, le, cfg)
    X_scaled, y_scaled, sx, sy = standardize_train(X, y)
    return X_scaled, y_scaled, ids, sx, sy, le
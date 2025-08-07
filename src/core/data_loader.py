"""Data loading and basic preprocessing routines.

The functions defined in this module provide a thin abstraction over
``pandas.read_csv`` to load the training and test data supplied in
the LG Aimers forecasting challenge.  They normalise column names,
parse dates, ensure data types are consistent and perform some
simple cleaning such as dropping duplicate records and filling
missing values.

These helpers intentionally do not perform any feature engineering –
that is handled in ``core.feature_engineer``.  The idea is to keep
data ingestion and low‑level cleaning in one place so it can be
reused across different models.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import pandas as pd


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename canonical columns and coerce types.

    The original dataset uses Korean column names (e.g. ``영업일자`` for
    date).  This helper renames them to English identifiers
    (``date``, ``store_item``, ``sales``) and converts the ``date`` column
    to a ``datetime64[ns]`` dtype.  Duplicate rows are dropped and
    missing sales values are replaced with zeros.

    Args:
        df: Raw dataframe as loaded from CSV.

    Returns:
        Normalised dataframe.
    """
    col_map = {
        "영업일자": "date",
        "영업장명_메뉴명": "store_item",
        "매출수량": "sales",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sales" in df.columns:
        # Some CSVs may include commas as thousand separators; remove them
        df["sales"] = pd.to_numeric(df["sales"].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0)
    # Drop duplicate records based on date and item
    if {"date", "store_item"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["date", "store_item"])
    # Fill any remaining NaNs
    return df.fillna(0.0)


def load_train(train_dir: str) -> pd.DataFrame:
    """Load and preprocess the training CSV.

    Args:
        train_dir: Path to the folder containing ``train.csv``.

    Returns:
        DataFrame with columns ``date``, ``store_item`` and ``sales``.
    """
    csv_path = os.path.join(train_dir, "train.csv")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = _normalise_columns(df)
    return df


def load_test(test_dir: str) -> List[pd.DataFrame]:
    """Load all test CSVs from a directory.

    The challenge distributes test files as ``TEST_00.csv`` through
    ``TEST_09.csv``.  This helper returns a list of DataFrames, one
    per file, with normalised columns.

    Args:
        test_dir: Path to the folder containing ``TEST_*.csv`` files.

    Returns:
        List of DataFrames in arbitrary order.
    """
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(".csv")]
    dfs: List[pd.DataFrame] = []
    for fname in sorted(files):
        path = os.path.join(test_dir, fname)
        df = pd.read_csv(path, encoding="utf-8")
        df = _normalise_columns(df)
        dfs.append(df)
    return dfs


def load_sample_submission(path: str) -> pd.DataFrame:
    """Load the sample submission CSV.

    This file defines the submission format expected by the
    competition platform.  It typically contains columns
    ``id``/``store_item`` and ``sales`` or similar.  The loader
    normalises column names but does not alter the content.
    """
    df = pd.read_csv(path, encoding="utf-8")
    df = _normalise_columns(df)
    return df
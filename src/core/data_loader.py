"""
Lightweight data loading utilities for the LG Aimers forecasting task.

This module centralises the reading of raw CSV files for both the
training set and the multiple test batches.  Separating IO logic into
its own file simplifies unit testing and decouples data access from
the rest of the pipeline.  All functions are pure and return pandas
DataFrames without performing any pivoting or feature engineering.

Usage
-----
Import the convenience functions and call them with explicit paths::

    from src.core.data_loader import read_train_data, read_test_data
    train_df = read_train_data("dataset/train/train.csv")
    test_dfs = read_test_data("dataset/test")

These functions do not enforce any particular schema beyond requiring
the presence of the columns ``영업일자``, ``영업장명_메뉴명`` and
``매출수량``.  They simply read and return the contents of the CSV
files.  Any further processing (pivoting, feature engineering,
windowing) is handled by :mod:`src.core.data_module`.
"""

from __future__ import annotations

import os
import glob
import pandas as pd
from typing import Dict, List


def read_train_data(path: str) -> pd.DataFrame:
    """Read the training CSV file into a DataFrame.

    Parameters
    ----------
    path : str
        Full path to the ``train.csv`` file.  Must include the
        directory and filename.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the entire training set.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_csv(path)
    return df


def read_test_data(directory: str) -> Dict[str, pd.DataFrame]:
    """Read all test batch CSV files into a mapping of name to DataFrame.

    Parameters
    ----------
    directory : str
        Path to the directory containing test batch files.  All files
        matching the pattern ``TEST_*.csv`` will be read.

    Returns
    -------
    dict
        A dictionary mapping the base filename (e.g. ``TEST_00``) to
        the corresponding DataFrame.  The dictionary is sorted by
        filename to provide deterministic ordering.
    """
    pattern = os.path.join(directory, "TEST_*.csv")
    files: List[str] = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No test files found in {directory}")
    data: Dict[str, pd.DataFrame] = {}
    for file in files:
        key = os.path.splitext(os.path.basename(file))[0]
        data[key] = pd.read_csv(file)
    return data


__all__ = ["read_train_data", "read_test_data"]
"""Feature engineering helpers for time‑series forecasting.

This module exposes a small set of functions that operate on
DataFrames containing ``date`` and ``sales`` columns to add useful
features for sequence modelling.  These include cyclical encodings
for calendar variables (weekday, month, day of year), lagged values
and moving averages.

Note that feature engineering is kept distinct from data loading to
facilitate reuse across different models and experiments.  The
functions here work with ``pandas`` objects and return new DataFrames
with additional columns.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Iterable, List


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with cyclic date features.

    Adds sine and cosine transforms of the weekday, month and day
    within year in order to capture seasonality.  Also flags
    weekends and Korean public holidays if ``holidays.py`` defines
    them.

    Args:
        df: DataFrame expected to contain a ``date`` column of
            type ``datetime64[ns]``.

    Returns:
        DataFrame with new columns appended.  The original input is
        not modified in place.
    """
    res = df.copy()
    if "date" not in res.columns:
        raise ValueError("DataFrame must contain a 'date' column for date features")
    # Convert to pandas datetime just in case
    res["date"] = pd.to_datetime(res["date"], errors="coerce")
    # Extract calendar components
    res["weekday"] = res["date"].dt.weekday
    res["month"] = res["date"].dt.month
    res["day_of_year"] = res["date"].dt.dayofyear
    # Sine/cosine encodings
    res["weekday_sin"] = np.sin(2 * np.pi * res["weekday"] / 7)
    res["weekday_cos"] = np.cos(2 * np.pi * res["weekday"] / 7)
    res["month_sin"] = np.sin(2 * np.pi * res["month"] / 12)
    res["month_cos"] = np.cos(2 * np.pi * res["month"] / 12)
    res["doy_sin"] = np.sin(2 * np.pi * res["day_of_year"] / 365)
    res["doy_cos"] = np.cos(2 * np.pi * res["day_of_year"] / 365)
    # Weekend flag
    res["is_weekend"] = (res["weekday"] >= 5).astype(int)
    # Holiday flag – optional
    try:
        from .holidays import KOREAN_HOLIDAYS
        res["is_holiday"] = res["date"].dt.strftime("%Y-%m-%d").isin(KOREAN_HOLIDAYS).astype(int)
    except Exception:
        # If holiday list is not available just set zero
        res["is_holiday"] = 0
    return res


def add_lag_features(df: pd.DataFrame, periods: Iterable[int], group_col: str = "store_item") -> pd.DataFrame:
    """Add lagged sales features to a DataFrame.

    For each integer ``p`` in ``periods`` a new column named
    ``lag_p`` is created containing the value of ``sales`` shifted
    backwards by ``p`` days.  The lags are computed per group (e.g.
    store or item) if ``group_col`` is provided.

    Args:
        df: DataFrame with at least ``sales`` and ``date`` columns.
        periods: Iterable of lag offsets in days.
        group_col: Column to group by when computing lags.

    Returns:
        Copy of ``df`` with additional lag columns.
    """
    res = df.copy()
    if group_col not in res.columns:
        raise ValueError(f"Group column '{group_col}' must exist in DataFrame")
    for p in periods:
        col_name = f"lag_{p}"
        res[col_name] = res.groupby(group_col)["sales"].shift(p)
    return res


def add_moving_average_features(df: pd.DataFrame, windows: Iterable[int], group_col: str = "store_item") -> pd.DataFrame:
    """Add moving average sales features.

    For each integer ``w`` in ``windows`` a new column named
    ``ma_w`` is created containing the rolling mean of ``sales`` over
    the previous ``w`` days, computed per group.

    Args:
        df: DataFrame with at least ``sales`` and ``date`` columns.
        windows: Iterable of window sizes.
        group_col: Column used to group data before computing the
            rolling statistics.

    Returns:
        Copy of ``df`` with additional moving average columns.
    """
    res = df.copy()
    if group_col not in res.columns:
        raise ValueError(f"Group column '{group_col}' must exist in DataFrame")
    for w in windows:
        col_name = f"ma_{w}"
        res[col_name] = res.groupby(group_col)["sales"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
    return res
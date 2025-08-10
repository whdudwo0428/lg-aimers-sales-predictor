"""
Feature engineering utilities for time‑series forecasting.

This module defines reusable components that augment raw pivot tables
with calendar‑based features, lagged values and moving averages.  The
engineering is deliberately simple and deterministic: it does not
require fitting on the training data and can therefore be applied
consistently to both training and inference datasets without risk of
data leakage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import datetime
from typing import Iterable, Optional

from .holidays import KOREAN_HOLIDAYS


def add_date_features(df: pd.DataFrame, date_col: str = "영업일자") -> pd.DataFrame:
    """Augment a DataFrame with calendar‑based features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least a date column.
    date_col : str, default "영업일자"
        Name of the column representing the business date.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the date column converted to a
        ``datetime64`` dtype and additional calendar features such as
        year, month, day of week, week number, weekend flag and
        holiday flags.
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    dt_series = result[date_col]

    result["year"] = dt_series.dt.year
    result["month"] = dt_series.dt.month
    result["day"] = dt_series.dt.day
    result["dayofweek"] = dt_series.dt.dayofweek  # Monday=0
    result["weekofyear"] = dt_series.dt.isocalendar().week.astype(int)
    result["is_weekend"] = result["dayofweek"].isin([5, 6]).astype(int)

    # Holiday flags
    date_list = dt_series.dt.date
    result["is_holiday"] = date_list.isin(KOREAN_HOLIDAYS).astype(int)

    # Before/after holiday flags
    holiday_set = set(KOREAN_HOLIDAYS)
    before_holiday = {d - datetime.timedelta(days=1) for d in holiday_set}
    after_holiday = {d + datetime.timedelta(days=1) for d in holiday_set}
    result["is_holiday_eve"] = date_list.isin(before_holiday).astype(int)
    result["is_holiday_after"] = date_list.isin(after_holiday).astype(int)

    # Quarter and day of year
    result["quarter"] = dt_series.dt.quarter
    result["dayofyear"] = dt_series.dt.day_of_year

    # Season indicator: Winter=4, Spring=1, Summer=2, Autumn=3.  This
    # encoding preserves ordinal relationships across seasons but
    # deliberately reorders to capture the ski season as a contiguous
    # block (1: Spring, 2: Summer, 3: Autumn, 4: Winter).  Adjust
    # values if you desire a different mapping.
    month = result["month"]
    result["season"] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [1, 2, 3],
        default=4,
    ).astype(int)

    # Ski season flag (December through February)
    result["ski_season"] = month.isin([12, 1, 2]).astype(int)

    return result


class FeatureEngineer:
    """Generate lag and moving average features for a pivoted DataFrame.

    The :class:`TimeSeriesDataModule` orchestrates feature engineering by
    combining the raw pivot table with the outputs of a
    :class:`FeatureEngineer` instance.  Only lag features and
    moving–average features are supported; no fitting is performed on
    the data, thereby avoiding data leakage into the validation and
    test splits.
    """

    def __init__(self, lag_periods: Optional[Iterable[int]] = None, ma_windows: Optional[Iterable[int]] = None) -> None:
        self.lag_periods = list(lag_periods) if lag_periods is not None else []
        self.ma_windows = list(ma_windows) if ma_windows is not None else []

    def _create_lag_features(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Return a list of DataFrames containing lagged values for each column.

        Each DataFrame in the returned list has the same index as the input
        and contains one column per original series, with the suffix
        ``_lag_{lag}`` appended to the column name.  If no lag periods are
        specified an empty list is returned.
        """
        if not self.lag_periods:
            return []
        features = []
        for lag in self.lag_periods:
            shifted = df.shift(lag).rename(columns=lambda c: f"{c}_lag_{lag}")
            features.append(shifted)
        return features

    def _create_ma_features(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Return a list of DataFrames containing rolling mean values.

        Each DataFrame in the returned list has the same index as the input
        and contains one column per original series, with the suffix
        ``_ma_{window}`` appended to the column name.  Rolling means are
        computed with a minimum window of 1 so that the early part of the
        series is not discarded.  If no windows are specified an empty
        list is returned.
        """
        if not self.ma_windows:
            return []
        features = []
        for window in self.ma_windows:
            rolled = df.rolling(window=window, min_periods=1).mean().rename(
                columns=lambda c: f"{c}_ma_{window}"
            )
            features.append(rolled)
        return features

    def transform(self, pivot_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate lag and moving–average features for a pivoted table.

        Parameters
        ----------
        pivot_df : pandas.DataFrame
            A DataFrame indexed by date with one column per item.  The
            index must be monotonically increasing and contiguous; gaps
            will propagate through the lagged features.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing all engineered features, or ``None``
            if no lags and moving average windows have been defined.
        """
        additional = []
        additional.extend(self._create_lag_features(pivot_df))
        additional.extend(self._create_ma_features(pivot_df))
        if not additional:
            return None
        return pd.concat(additional, axis=1)


__all__ = ["add_date_features", "FeatureEngineer"]
"""
Utility functions for feature engineering and evaluation in the
Gonjiam resort demand forecasting challenge.

This module provides date-based feature augmentation, Korean holiday
definitions for 2023–2025, simple SMAPE calculation, and helper
functions that are used throughout the modelling pipeline.
"""
import datetime
from typing import Iterable, List

import numpy as np
import pandas as pd

# Predefined set of major Korean public holidays for 2023–2025.  These dates
# reflect official public holidays and some substitution days that fall
# within the training and evaluation horizon (2023-01-01 through
# 2025-05-24).  Only holidays up to May 2025 are included, as the
# test period ends in May 2025.
KOREAN_HOLIDAYS: List[datetime.date] = [
    # 2023 holidays
    datetime.date(2023, 1, 1),  # New Year's Day
    datetime.date(2023, 1, 21), datetime.date(2023, 1, 22), datetime.date(2023, 1, 23),
    datetime.date(2023, 1, 24),  # Seollal (Korean New Year) + substitute holiday
    datetime.date(2023, 3, 1),   # Independence Movement Day
    datetime.date(2023, 5, 5),   # Children's Day
    datetime.date(2023, 5, 27),  # Buddha's Birthday
    datetime.date(2023, 6, 6),   # Memorial Day
    datetime.date(2023, 8, 15),  # Liberation Day
    datetime.date(2023, 9, 28), datetime.date(2023, 9, 29), datetime.date(2023, 9, 30),
    datetime.date(2023, 10, 2),  # Chuseok holidays & substitute
    datetime.date(2023, 10, 3),  # National Foundation Day
    datetime.date(2023, 10, 9),  # Hangul Proclamation Day
    datetime.date(2023, 12, 25), # Christmas Day
    # 2024 holidays
    datetime.date(2024, 1, 1),
    datetime.date(2024, 2, 9), datetime.date(2024, 2, 10), datetime.date(2024, 2, 11),
    datetime.date(2024, 2, 12),  # Seollal & substitute holiday
    datetime.date(2024, 3, 1),   # Independence Movement Day
    datetime.date(2024, 4, 10),  # National Assembly Election Day
    datetime.date(2024, 5, 5), datetime.date(2024, 5, 6),  # Children's Day + substitute
    datetime.date(2024, 5, 15),  # Buddha's Birthday
    datetime.date(2024, 6, 6),   # Memorial Day
    datetime.date(2024, 8, 15),  # Liberation Day
    datetime.date(2024, 9, 16), datetime.date(2024, 9, 17), datetime.date(2024, 9, 18),
    datetime.date(2024, 10, 3),  # National Foundation Day
    datetime.date(2024, 10, 9),  # Hangul Proclamation Day
    datetime.date(2024, 12, 25), # Christmas Day
    # 2025 holidays (until May)
    datetime.date(2025, 1, 1),   # New Year's Day
    datetime.date(2025, 1, 28), datetime.date(2025, 1, 29), datetime.date(2025, 1, 30),  # Seollal Holiday
    datetime.date(2025, 3, 3),   # March 1st substitute
    datetime.date(2025, 5, 5),   # Children's Day
    datetime.date(2025, 5, 6),   # Buddha's Birthday substitute
    datetime.date(2025, 6, 6),   # Memorial Day
    datetime.date(2025, 8, 15),  # Liberation Day
    datetime.date(2025, 10, 3),  # National Foundation Day
    datetime.date(2025, 10, 6), datetime.date(2025, 10, 7), datetime.date(2025, 10, 8),  # Chuseok Holiday
    datetime.date(2025, 10, 9),  # Hangul Day
    datetime.date(2025, 12, 25), # Christmas
]
# 연휴일자에 따라 가중치 추가

def add_date_features(df: pd.DataFrame, date_col: str = '영업일자') -> pd.DataFrame:
    """Augment the DataFrame with additional calendar-based features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the date column.
    date_col : str, optional
        Name of the date column to process. Defaults to '영업일자'.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with added feature columns.
    """
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    dt_series = result[date_col]
    result['year'] = dt_series.dt.year
    result['month'] = dt_series.dt.month
    result['day'] = dt_series.dt.day
    result['dayofweek'] = dt_series.dt.dayofweek  # Monday=0
    result['weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
    result['is_weekend'] = result['dayofweek'].isin([5, 6]).astype(int)
    # Holiday flags
    date_list = dt_series.dt.date
    result['is_holiday'] = date_list.isin(KOREAN_HOLIDAYS).astype(int)
    # Before/after holiday flags
    holiday_set = set(KOREAN_HOLIDAYS)
    before_holiday = set(d - datetime.timedelta(days=1) for d in holiday_set)
    after_holiday = set(d + datetime.timedelta(days=1) for d in holiday_set)
    result['is_holiday_eve'] = date_list.isin(before_holiday).astype(int)
    result['is_holiday_after'] = date_list.isin(after_holiday).astype(int)
    # Quarter and day of year
    result['quarter'] = dt_series.dt.quarter
    result['dayofyear'] = dt_series.dt.dayofyear
    # Season (1: Spring, 2: Summer, 3: Autumn, 4: Winter)
    month = result['month']
    result['season'] = np.select(
        [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [1, 2, 3],
        default=4
    )
    # Ski season (1 if date falls in ski season; December–February)
    result['ski_season'] = month.isin([12, 1, 2]).astype(int)
    return result
    
def add_store_menu_features(df: pd.DataFrame, col_name: str = '영업장명_메뉴명') -> pd.DataFrame:
    """
    '영업장명_메뉴명' 컬럼을 '_'로 분리해
    '영업장명'과 '메뉴명' 컬럼을 생성합니다.
    """
    result = df.copy()
    split_cols = result[col_name].str.split('_', n=1, expand=True)
    result['영업장명'] = split_cols[0]
    result['메뉴명'] = split_cols[1]
    return result

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        SMAPE value expressed as a percentage (0–100 range).
    """
    denom = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero by adding a small epsilon to denom
    epsilon = 1e-8
    denom = np.where(denom == 0, epsilon, denom)
    diff = np.abs(y_true - y_pred)
    smape_values = 2.0 * diff / denom
    # Exclude cases where both y_true and y_pred are zero from averaging
    mask = (y_true == 0) & (y_pred == 0)
    if np.all(mask):
        return 0.0
    return float(np.mean(smape_values[~mask]))

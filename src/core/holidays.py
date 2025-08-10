"""
Fixed list of Korean public holidays used for feature engineering.

The holiday dates defined here cover the period from January 2023
through December 2025.  The list is intentionally conservative—it
contains the official public holidays as well as substitute days where
applicable.  During feature engineering a number of additional flags
are derived from this list (for example, whether a date is the eve
before a holiday or immediately follows one).  If your forecasting
horizon extends beyond the dates covered here you should extend the
list accordingly; however, be mindful that using future holidays
constitutes external data and may violate competition rules.
"""

from __future__ import annotations

import datetime
from typing import List

KOREAN_HOLIDAYS: List[datetime.date] = [
    # 2023
    datetime.date(2023, 1, 1),
    datetime.date(2023, 1, 21), datetime.date(2023, 1, 22), datetime.date(2023, 1, 23),
    datetime.date(2023, 1, 24),  # Seollal and substitute holiday
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

    # 2024
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

    # 2025 (through the end of the year)
    datetime.date(2025, 1, 1),   # New Year's Day
    datetime.date(2025, 1, 28), datetime.date(2025, 1, 29), datetime.date(2025, 1, 30),  # Seollal holiday
    datetime.date(2025, 3, 3),   # March 1st substitute
    datetime.date(2025, 5, 5),   # Children's Day
    datetime.date(2025, 5, 6),   # Buddha's Birthday substitute
    datetime.date(2025, 6, 6),   # Memorial Day
    datetime.date(2025, 8, 15),  # Liberation Day
    datetime.date(2025, 10, 3),  # National Foundation Day
    datetime.date(2025, 10, 6), datetime.date(2025, 10, 7), datetime.date(2025, 10, 8),  # Chuseok holiday
    datetime.date(2025, 10, 9),  # Hangul Day
    datetime.date(2025, 12, 25), # Christmas Day
]

__all__ = ["KOREAN_HOLIDAYS"]
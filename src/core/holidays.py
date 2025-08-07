"""Static list of Korean public holidays for feature engineering.

The list below contains the dates of major Korean public holidays for
the years 2023–2025.  If your training data spans multiple years you
should update this list accordingly or consider using the :mod:`holidays`
Python package to compute holidays on the fly.
"""

from datetime import date as _date

# A static set of Korean public holidays.  This list is used by
# ``feature_engineer.add_date_features`` to flag holiday periods.  When
# extending the training data beyond 2025 you should expand this list
# accordingly.  The use of ``_date`` avoids polluting the module
# namespace and makes the declarations more concise.
KOREAN_HOLIDAYS = {
    # 2023 holidays
    _date(2023, 1, 1),  # New Year's Day
    _date(2023, 1, 21), _date(2023, 1, 22), _date(2023, 1, 23),
    _date(2023, 1, 24),  # Seollal and substitute holiday
    _date(2023, 3, 1),   # Independence Movement Day
    _date(2023, 5, 5),   # Children's Day
    _date(2023, 5, 27),  # Buddha's Birthday
    _date(2023, 6, 6),   # Memorial Day
    _date(2023, 8, 15),  # Liberation Day
    _date(2023, 9, 28), _date(2023, 9, 29), _date(2023, 9, 30),
    _date(2023, 10, 2),  # Chuseok holidays & substitute
    _date(2023, 10, 3),  # National Foundation Day
    _date(2023, 10, 9),  # Hangul Proclamation Day
    _date(2023, 12, 25), # Christmas Day
    # 2024 holidays
    _date(2024, 1, 1),
    _date(2024, 2, 9), _date(2024, 2, 10), _date(2024, 2, 11), _date(2024, 2, 12),
    _date(2024, 3, 1),
    _date(2024, 4, 10),  # National Assembly Election Day
    _date(2024, 5, 5), _date(2024, 5, 6),  # Children's Day + substitute
    _date(2024, 5, 15),
    _date(2024, 6, 6),
    _date(2024, 8, 15),
    _date(2024, 9, 16), _date(2024, 9, 17), _date(2024, 9, 18),
    _date(2024, 10, 3),
    _date(2024, 10, 9),
    _date(2024, 12, 25),
    # 2025 holidays (up to end of year)
    _date(2025, 1, 1),
    _date(2025, 1, 28), _date(2025, 1, 29), _date(2025, 1, 30),
    _date(2025, 3, 3),
    _date(2025, 5, 5), _date(2025, 5, 6),
    _date(2025, 6, 6),
    _date(2025, 8, 15),
    _date(2025, 10, 3),
    _date(2025, 10, 6), _date(2025, 10, 7), _date(2025, 10, 8),
    _date(2025, 10, 9),
    _date(2025, 12, 25),
}
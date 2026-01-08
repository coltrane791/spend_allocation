# src/split.py

"""
Dataset splitting utilities, small and explicit. The default split is time-based, which is the
appropriate CV primitive for a date-indexed marketing dataset.
"""

# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

@dataclass(frozen=True)
class TimeHoldoutSplit:
    train_ad_spend: pd.DataFrame
    test_ad_spend: pd.DataFrame
    train_macro: pd.DataFrame
    test_macro: pd.DataFrame
    split_date: pd.Timestamp
    n_train_days: int
    n_test_days: int

def time_holdout_split(
    *,
    ad_spend_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_col: str = "date",
    test_size_days: int = 30,
) -> TimeHoldoutSplit:
    """
    Time-based holdout split (recommended for time-indexed panels).

    Uses the last `test_size_days` unique dates as TEST; everything before is TRAIN.

    Returns a TimeHoldoutSplit bundle with train/test versions of:
      - ad_spend_df (panel)
      - macro_df (daily environment)
    """
    if date_col not in ad_spend_df.columns:
        raise KeyError(f"ad_spend_df missing date_col='{date_col}'")
    if date_col not in macro_df.columns:
        raise KeyError(f"macro_df missing date_col='{date_col}'")

    ad = ad_spend_df.copy()
    mc = macro_df.copy()

    # Ensure datetime
    ad[date_col] = pd.to_datetime(ad[date_col])
    mc[date_col] = pd.to_datetime(mc[date_col])

    # Unique sorted dates based on macro_df (authoritative daily calendar)
    dates = mc[date_col].drop_duplicates().sort_values().to_list()
    if len(dates) <= test_size_days:
        raise ValueError(
            f"Not enough days to split: n_days={len(dates)} <= test_size_days={test_size_days}"
        )

    split_date = pd.Timestamp(dates[-test_size_days])

    train_macro = mc.loc[mc[date_col] < split_date].copy()
    test_macro = mc.loc[mc[date_col] >= split_date].copy()

    train_ad = ad.loc[ad[date_col] < split_date].copy()
    test_ad = ad.loc[ad[date_col] >= split_date].copy()

    n_train_days = int(train_macro[date_col].nunique())
    n_test_days = int(test_macro[date_col].nunique())

    return TimeHoldoutSplit(
        train_ad_spend=train_ad,
        test_ad_spend=test_ad,
        train_macro=train_macro,
        test_macro=test_macro,
        split_date=split_date,
        n_train_days=n_train_days,
        n_test_days=n_test_days,
    )

def split_summary_df(split: TimeHoldoutSplit) -> pd.DataFrame:
    """Small helper for exporting split metadata."""
    return pd.DataFrame(
        [
            {
                "split_date": split.split_date,
                "n_train_days": split.n_train_days,
                "n_test_days": split.n_test_days,
                "train_rows": len(split.train_ad_spend),
                "test_rows": len(split.test_ad_spend),
            }
        ]
    )

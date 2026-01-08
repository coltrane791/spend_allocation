# src/utils.py

from __future__ import annotations
from typing import Sequence

import pandas as pd

def infer_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first candidate column that exists in df.

    This is intentionally tiny and shared across modules to avoid copy/paste.
    """
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(
        f"Could not infer column. Tried {list(candidates)}; found {sorted(cols)}"
    )

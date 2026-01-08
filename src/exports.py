# src/exports.py

"""
Outputs/exports files.
"""

# Import libraries and modules
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

# Helpers
def _resolve_out_path(out_path: Path, *, overwrite: bool, make_unique: bool) -> Path:
    out_path = Path(out_path)

    if overwrite:
        return out_path

    if not out_path.exists():
        return out_path

    if not make_unique:
        raise FileExistsError(f"Output file already exists: {out_path}")

    ## Create a unique filename: "name (1).ext", "name (2).ext", ...
    stem = out_path.stem
    suffix = out_path.suffix
    parent = out_path.parent

    i = 1
    while True:
        candidate = parent / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1

# Save csv or single-tab Excel
def save_df(
    df: pd.DataFrame,
    out_path: Path,
    *,
    index: bool = False,
    overwrite: bool = False,
    make_unique: bool = True,
) -> Path:
    """
    Save a single DataFrame to CSV or Excel based on file suffix.

    Returns the actual path written (useful when make_unique=True).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path = _resolve_out_path(out_path, overwrite=overwrite, make_unique=make_unique)

    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=index)
        return out_path

    if suffix in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=index)
        return out_path

    raise ValueError(f"Unsupported output format: {suffix}. Use .csv or .xlsx/.xls")

# Save multi-tab Excel
def save_workbook(
    sheets: Mapping[str, pd.DataFrame],
    out_path: Path,
    *,
    index: bool = False,
    engine: Optional[str] = "openpyxl",
    overwrite: bool = False,
    make_unique: bool = True,
) -> Path:
    """
    Save multiple DataFrames into one Excel workbook.

    Returns the actual path written (useful when make_unique=True).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("save_workbook requires an Excel path ending in .xlsx or .xls")

    out_path = _resolve_out_path(out_path, overwrite=overwrite, make_unique=make_unique)

    def _clean_sheet_name(name: str) -> str:
        bad = [":", "\\", "/", "?", "*", "[", "]"]
        for ch in bad:
            name = name.replace(ch, " ")
        name = name.strip()
        return name[:31] if len(name) > 31 else (name or "Sheet1")

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        for sheet_name, df in sheets.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Sheet '{sheet_name}' is not a DataFrame (got {type(df)}).")
            df.to_excel(writer, sheet_name=_clean_sheet_name(str(sheet_name)), index=index)

    return out_path

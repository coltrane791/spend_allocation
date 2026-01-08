# src/marg_grid.py

"""
Marginal grid generation helpers (estimator-agnostic).

Design intent
-------------
This module should NOT assume any particular response-curve functional form.
Estimator-specific logic (e.g., MMM saturation curve, analytic derivatives,
scenario multipliers) belongs in the estimator modules under src/estimators/.

What lives here:
- Spend-grid construction utilities shared across estimators.
- Small helpers to derive spend grids from an existing curve grid.

What does NOT live here:
- expected_conversions / d_expected_conversions_ds / ResponseParams
- any alpha/beta/gamma/delta assumptions
"""

# Import libraries and modules
from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import COL_ARM_ID, COL_SPEND

# Standardized spend grid
def build_spend_grid_by_arm(
    min_spend: Dict[str, float],
    max_spend: Dict[str, float],
    *,
    arms: Optional[Iterable[str]] = None,
    n_points: int = 25,
) -> Dict[str, np.ndarray]:
    """Construct a per-arm spend grid.

    Parameters
    ----------
    min_spend / max_spend:
        Dicts keyed by arm_id.
    arms:
        Optional explicit list/iterable of arms to build grids for. If omitted,
        uses the union of keys in min_spend and max_spend.
    n_points:
        Number of grid points per arm.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping arm_id -> spend grid (numpy array, ascending).

    Notes
    -----
    - Includes endpoints (min and max) via linspace.
    - Validates hi >= lo for each arm.
    """
    if n_points <= 1:
        raise ValueError("n_points must be >= 2")

    if arms is None:
        arms = sorted(set(min_spend.keys()) | set(max_spend.keys()))

    out: Dict[str, np.ndarray] = {}
    for a in arms:
        lo = float(min_spend.get(a, 0.0))
        hi = float(max_spend.get(a, lo))
        if hi < lo:
            raise ValueError(f"Invalid bounds for {a}: max({hi}) < min({lo})")
        out[a] = np.linspace(lo, hi, int(n_points), dtype=float)
    return out

## Old version - not used
def spend_grid_by_arm_from_curve_grid(
    curve_grid_df: pd.DataFrame,
    *,
    arm_col: str = COL_ARM_ID,
    spend_col: str = COL_SPEND,
) -> Dict[str, np.ndarray]:
    """
    Derive spend_grid_by_arm from an existing curve grid DataFrame.
    """
    if arm_col not in curve_grid_df.columns or spend_col not in curve_grid_df.columns:
        raise KeyError(
            f"curve_grid_df must include columns {arm_col!r} and {spend_col!r}. "
            f"Found: {sorted(curve_grid_df.columns)}"
        )

    out: Dict[str, np.ndarray] = {}
    for arm_id, g in curve_grid_df.groupby(arm_col, sort=True):
        out[str(arm_id)] = np.asarray(sorted(g[spend_col].astype(float).unique()), dtype=float)
    return out

# Enforce monotonicity
def _pava_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression (non-decreasing).
    Returns y_hat minimizing sum_i w_i (y_hat_i - y_i)^2 subject to y_hat non-decreasing.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return y

    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if w.size != n:
            raise ValueError(f"w has size {w.size} but y has size {n}")
        w = np.maximum(w, 0.0)

    # Block representation: each block has (value, weight, length)
    vals = y.tolist()
    wgts = w.tolist()
    lens = [1] * n

    i = 0
    while i < len(vals) - 1:
        # Violation for non-decreasing: vals[i] > vals[i+1]
        if vals[i] > vals[i + 1]:
            tot_w = wgts[i] + wgts[i + 1]
            if tot_w <= 0:
                # both zero weight: average the values equally
                new_val = 0.5 * (vals[i] + vals[i + 1])
            else:
                new_val = (vals[i] * wgts[i] + vals[i + 1] * wgts[i + 1]) / tot_w

            vals[i] = float(new_val)
            wgts[i] = float(tot_w)
            lens[i] = int(lens[i] + lens[i + 1])

            del vals[i + 1]
            del wgts[i + 1]
            del lens[i + 1]

            # Step back to re-check previous adjacency
            i = max(i - 1, 0)
        else:
            i += 1

    y_hat = np.repeat(np.asarray(vals, dtype=float), np.asarray(lens, dtype=int))
    if y_hat.size != n:
        raise RuntimeError("PAVA expansion produced wrong length output.")
    return y_hat

def project_monotone_marginal(
    curve_grid_df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("env_id", "arm_id", "estimator"),
    spend_col: str = "spend",
    dcol: str = "d_conv_d$",
    weight_col: Optional[str] = None,
    nonneg: bool = True,
    keep_raw: bool = True,
    raw_suffix: str = "_init",
) -> pd.DataFrame:
    """
    Enforce diminishing returns by projecting d_conv_d$ to be non-increasing in spend,
    separately within each group.

    Implementation: isotonic regression with a non-increasing constraint via PAVA on (-dcol).

    Parameters
    ----------
    curve_grid_df : DataFrame
        Must contain group_cols + spend_col + dcol.
    group_cols : Sequence[str]
        Keys defining separate curves (e.g., ("env_id","arm_id") for ensemble, or also
        include "estimator" for multi-estimator grids).
    weight_col : Optional[str]
        If provided, used as isotonic weights (e.g., n_eff_data or n_eff_fit).
    nonneg : bool
        If True, clips projected marginal to >= 0.
    keep_raw : bool
        If True, preserves original dcol as f"{dcol}{raw_suffix}".
    """
    df = curve_grid_df.copy()

    needed = set(group_cols) | {spend_col, dcol}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise KeyError(f"curve_grid_df missing required columns: {missing}")

    if keep_raw and f"{dcol}{raw_suffix}" not in df.columns:
        df[f"{dcol}{raw_suffix}"] = df[dcol]

    # Sort within groups by spend before projecting
    df = df.sort_values(list(group_cols) + [spend_col]).reset_index(drop=True)

    out_vals = np.empty(len(df), dtype=float)

    for _, g in df.groupby(list(group_cols), sort=False):
        idx = g.index.to_numpy()
        y = g[dcol].to_numpy(dtype=float)

        if not np.all(np.isfinite(y)):
            raise ValueError(f"Non-finite values found in {dcol} for group={tuple(g.iloc[0][list(group_cols)])}")

        w = None
        if weight_col is not None:
            if weight_col not in g.columns:
                raise KeyError(f"weight_col={weight_col} not in curve_grid_df columns")
            w = g[weight_col].to_numpy(dtype=float)
            if not np.all(np.isfinite(w)):
                raise ValueError(f"Non-finite values found in {weight_col} for group={tuple(g.iloc[0][list(group_cols)])}")

        # Non-increasing in spend == isotonic non-decreasing on (-y)
        y_hat = -_pava_increasing(-y, w=w)

        if nonneg:
            y_hat = np.maximum(y_hat, 0.0)

        out_vals[idx] = y_hat

    df[dcol] = out_vals

    # Reorder columns to put d_conv_d$ before d_conv_d$_init
    if keep_raw:
        cols = df.columns.tolist()
        idx_dcol = cols.index(dcol)
        idx_raw = cols.index(f"{dcol}{raw_suffix}")
        cols[idx_dcol], cols[idx_raw] = cols[idx_raw], cols[idx_dcol]
        df = df[cols]

    return df

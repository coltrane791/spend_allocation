# src/estimators/estimator_dd1.py

"""
Local-slope / shape-constrained response-curve estimator.

Implements a nonparametric estimator of the marginal response curve d_conv_d$
per arm using binning + isotonic regression (PAVA) to enforce diminishing returns.

Public API
----------
local_slope_curve_grid(...):
    Fits the local-slope artifact per arm and returns a standardized marginal grid.

Notes
-----
- (S, M) are currently carried through as metadata only; the estimator can be
  extended later to condition on these environment variables.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import infer_col

def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted isotonic regression (non-decreasing) via PAVA.
    Returns y_hat with y_hat[0] <= y_hat[1] <= ... <= y_hat[n-1].
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or y.shape != w.shape:
        raise ValueError("y and w must be 1D arrays of same shape.")

    n = len(y)
    if n == 0:
        return y.copy()

    ## Use Python lists so we can merge/delete blocks
    vals = list(y.tolist())
    weights = list(w.tolist())

    ## Track original index spans for each block
    starts = list(range(n))
    ends = list(range(n))

    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1]:
            i += 1
            continue

        ### Merge blocks i and i+1
        new_w = weights[i] + weights[i + 1]
        new_v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / max(new_w, 1e-12)

        vals[i] = float(new_v)
        weights[i] = float(new_w)
        ends[i] = ends[i + 1]

        ### Delete block i+1
        del vals[i + 1]
        del weights[i + 1]
        del starts[i + 1]
        del ends[i + 1]

        ### Step back to re-check monotonicity with previous block
        if i > 0:
            i -= 1

    ## Expand block values back to full length
    y_hat = np.empty(n, dtype=float)
    for v, s, e in zip(vals, starts, ends):
        y_hat[s : e + 1] = v

    return y_hat

def _pava_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted isotonic regression enforcing non-increasing sequence:
      y[0] >= y[1] >= ... >= y[n-1]
    Implemented by applying increasing isotonic to -y.
    """
    return -_pava_increasing(-np.asarray(y, dtype=float), np.asarray(w, dtype=float))

def _fit_local_concave_marginal(
    s_obs: np.ndarray,
    y_obs: np.ndarray,
    *,
    n_bins: int = 12,
    min_bins: int = 3,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a concave, nondecreasing response shape using binned means + isotonic smoothing:

    Steps:
      1) bin spend into quantile bins
      2) compute mean spend and mean conversions per bin
      3) isotonic regression (increasing) on bin means => smooth, monotone level curve
      4) compute slopes between adjacent bins
      5) enforce diminishing returns: slopes non-increasing (decreasing isotonic)
    Returns:
      x_knots (len K)  : increasing spend knot locations
      slopes  (len K-1): piecewise-constant marginal d_conv/d$ between knots
    """
    s_obs = np.asarray(s_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    ## Keep finite values; require nonnegative spend for this estimator
    m = np.isfinite(s_obs) & np.isfinite(y_obs) & (s_obs >= 0)
    s_obs = s_obs[m]
    y_obs = y_obs[m]

    if len(s_obs) < 10:
        # Not enough data; fall back to a tiny slope
        return np.array([0.0, 1.0], dtype=float), np.array([0.0], dtype=float)

    ## Bin spend into quantiles; duplicates='drop' handles low-variance spend
    try:
        bins = pd.qcut(s_obs, q=int(n_bins), duplicates="drop")
    except ValueError:
        # If qcut cannot form bins (e.g., all spends equal), fall back
        total_spend = float(np.sum(s_obs))
        slope = float(np.sum(y_obs) / max(total_spend, eps))
        return np.array([0.0, max(float(np.max(s_obs)), 1.0)], dtype=float), np.array([max(slope, 0.0)], dtype=float)

    dfb = pd.DataFrame({"spend": s_obs, "conv": y_obs, "bin": bins})
    agg = (
        dfb.groupby("bin", observed=True)
        .agg(mean_spend=("spend", "mean"), mean_conv=("conv", "mean"), n=("conv", "size"))
        .reset_index(drop=True)
        .sort_values("mean_spend")
    )

    ## Collapse any duplicate mean_spend (rare but possible)
    agg = (
        agg.groupby("mean_spend", as_index=False)
        .agg(mean_conv=("mean_conv", "mean"), n=("n", "sum"))
        .sort_values("mean_spend")
        .reset_index(drop=True)
    )

    if len(agg) < min_bins:
        total_spend = float(np.sum(s_obs))
        slope = float(np.sum(y_obs) / max(total_spend, eps))
        xmax = max(float(np.max(s_obs)), 1.0)
        return np.array([0.0, xmax], dtype=float), np.array([max(slope, 0.0)], dtype=float)

    x = agg["mean_spend"].to_numpy(dtype=float)
    y = agg["mean_conv"].to_numpy(dtype=float)
    w = agg["n"].to_numpy(dtype=float)

    # ---------------------------------------------------------------------
    # Anchor the level curve at (0, 0) to avoid global level-shift downward.
    # This is appropriate for "incremental conversions" where spend=0 implies 0.
    # Only add if the first knot is meaningfully above 0.
    # ---------------------------------------------------------------------
    if float(x[0]) > eps:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[0.0], y])

        # Give the anchor a moderate weight so it enforces the intercept without
        # completely dominating the fit.
        w0 = float(max(np.max(w), 10.0))
        w = np.concatenate([[w0], w])

    # Enforce monotone increasing level curve on binned means (now anchored)
    y_iso = _pava_increasing(y, w)
    y_iso = np.maximum(y_iso, 0.0)

    # Slopes between adjacent knot points
    dx = np.diff(x)
    dx = np.maximum(dx, eps)
    slopes_raw = np.diff(y_iso) / dx
    slopes_raw = np.maximum(slopes_raw, 0.0)

    # Enforce diminishing returns: slopes non-increasing
    w_seg = 0.5 * (w[:-1] + w[1:])
    w_seg = np.maximum(w_seg, 1.0)

    slopes = _pava_decreasing(slopes_raw, w_seg)
    slopes = np.maximum(slopes, 0.0)

    return x, slopes

def local_slope_curve_grid(
    *,
    estimator: str,
    env_id: str,
    ad_spend_df: pd.DataFrame,
    min_spend: Dict[str, float],
    max_spend: Dict[str, float],
    S: float,
    M: float,
    arm_col: str = "arm_id",
    spend_col: Optional[str] = None,
    conv_col: Optional[str] = None,
    n_points: int = 25,
    n_bins: int = 12,
    spend_grid_by_arm: Optional[Dict[str, np.ndarray]] = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Local-slope estimator producing a thin curve grid for ONE environment (S, M).
    Environment fields are carried as metadata; this estimator does not model S/M.

    Output columns:
      estimator, env_id, arm_id, spend, seasonality, macro_index,
      exp_conversions, d_conv_d$
    """
    
    spend_col = spend_col or infer_col(ad_spend_df, ["spend", "daily_spend", "cost"])
    conv_col = conv_col or infer_col(ad_spend_df, ["conversions", "conversions_realized", "realized_conversions", "conv", "C"])

    S = float(S)
    M = float(M)

    out_frames: list[pd.DataFrame] = []

    for arm_id, g in ad_spend_df.groupby(arm_col, sort=True):
        lo = float(min_spend.get(arm_id, 0.0))
        hi = float(max_spend.get(arm_id, lo))
        if hi < lo:
            raise ValueError(f"Invalid bounds for {arm_id}: max({hi}) < min({lo})")

        if spend_grid_by_arm is not None:
            spends = np.asarray(spend_grid_by_arm[arm_id], dtype=float)
        else:
            spends = np.linspace(lo, hi, int(n_points), dtype=float)

        s_obs = g[spend_col].to_numpy(dtype=float)
        y_obs = g[conv_col].to_numpy(dtype=float)

        x_knots, slopes = _fit_local_concave_marginal(
            s_obs, y_obs, n_bins=n_bins, eps=eps
        )

        ## Evaluate piecewise-constant marginal on the spend grid
        ## slopes apply between x_knots[j] and x_knots[j+1]
        if len(slopes) == 0:
            dlam = np.zeros_like(spends)
        else:
            # Find segment index for each spend: j such that x[j] <= s < x[j+1]
            # For s < x[0], use j=0; for s >= x[-1], use j=len(slopes)-1
            j = np.searchsorted(x_knots[1:], spends, side="right")
            j = np.clip(j, 0, len(slopes) - 1)
            dlam = slopes[j]

        dlam = np.maximum(dlam, 0.0)

        # Construct exp_conversions by integrating the marginal curve from 0.
        if len(slopes) == 0:
            lam = np.zeros_like(spends, dtype=float)
        else:
            xk = np.asarray(x_knots, dtype=float)
            sl = np.asarray(slopes, dtype=float)

            # cumulative integral at knots: lam(xk[i]) = ∫_0^{xk[i]} dlam(t) dt
            dxk = np.diff(xk)
            dxk = np.maximum(dxk, eps)
            cum_at_knots = np.concatenate([[0.0], np.cumsum(sl * dxk)])

            s_pos = np.maximum(spends, 0.0)

            # segment index k such that xk[k] <= s < xk[k+1]
            k = np.searchsorted(xk[1:], s_pos, side="right")
            k = np.clip(k, 0, len(sl) - 1)

            lam = cum_at_knots[k] + sl[k] * (s_pos - xk[k])

        lam = np.maximum(lam, eps)

        df_arm = pd.DataFrame(
            {
                "estimator": estimator,
                "env_id": env_id,
                arm_col: arm_id,
                "spend": spends,
                "seasonality": S,
                "macro_index": M,
                "exp_conversions": lam,
                "d_conv_d$": dlam,
            }
        )
        out_frames.append(df_arm)

    out = pd.concat(out_frames, ignore_index=True)
    out = out.sort_values([arm_col, "spend"]).reset_index(drop=True)
    return out

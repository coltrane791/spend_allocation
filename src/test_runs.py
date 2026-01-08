# src/test_runs.py

"""
Collect aggregates from test runs
"""

# Import libraries and modules
from __future__ import annotations

import numpy as np
import pandas as pd

# Create daily total from daily arm-level data
def build_day_total_row(
    day_compare_df: pd.DataFrame,
    *,
    total_label: str = "Total",
    day: str | pd.Timestamp | None = None,
    env_id: str | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Build a 1-row TOTAL record from a single-day day_compare_df.

    Key behavior:
      - Output schema == input schema (same columns, same order)
      - Additive columns are summed (spend, conversions, profits, expected values, deltas, most fc_error_*).
      - Ratio columns are computed at portfolio level (vpc_actual, exp_vpc).
      - Marginal columns (d_*) are set to NaN in the Total row (not meaningfully aggregatable).
      - Bool flags (e.g., clamp flags) are aggregated as ANY-true, except conv_is_zero which is recomputed.
    """

    if day_compare_df.empty:
        raise ValueError("day_compare_df is empty")

    cols = list(day_compare_df.columns)

    df = day_compare_df.copy()
    if "arm_id" in df.columns:
        df = df.loc[df["arm_id"].ne(total_label)].copy()

    # Initialize output with NaNs, then fill selectively
    out: dict[str, object] = {c: np.nan for c in cols}

    # --- Carry-through identifiers / constants ---
    if "arm_id" in cols:
        out["arm_id"] = total_label

    # Prefer explicit args, otherwise take from df if present
    if "date_eval" in cols:
        out["date_eval"] = day if day is not None else df["date_eval"].iloc[0]
    if "env_id" in cols:
        out["env_id"] = env_id if env_id is not None else df["env_id"].iloc[0]

    # Constants within-day (take first non-null)
    first_cols = [
        "budget",
        "tau",
        "nu_conv",
        "allow_unspent",
        "capacity_conversions",
    ]
    for c in first_cols:
        if c in cols:
            out[c] = df[c].dropna().iloc[0] if df[c].notna().any() else np.nan

    # --- Bool flags: aggregate as ANY-true by default ---
    # (we'll recompute conv_is_zero explicitly later if present)
    for c in cols:
        if pd.api.types.is_bool_dtype(df[c]) or df[c].dtype == bool:
            out[c] = bool(df[c].fillna(False).any())

    # --- Additive sums (coerce numeric robustly) ---
    def _sum_if_present(c: str) -> None:
        if c in cols:
            out[c] = float(pd.to_numeric(df[c], errors="coerce").sum())

    # Common additive columns (expand safely as your schema evolves)
    sum_cols = [
        "spend_min",
        "spend_max",
        "spend_actual",
        "spend_clamped_actual",
        "spend_opt",
        "spend_clamped_opt",
        "conversions_actual",
        "funded_loans_actual",
        "profit_gross_actual",
        "profit_net_actual",
        "exp_conversions_actual",
        "exp_profit_net_actual",
        "exp_conversions_opt",
        "exp_profit_net_opt",
        "delta_spend",
        "delta_exp_conversions",
        "delta_exp_profit_net",
        "fc_error_conv",
        "fc_error_np",
        "fc_error_np_conv",
        "fc_error_np_vpc",
    ]
    for c in sum_cols:
        _sum_if_present(c)

    # --- Portfolio-level ratio metrics ---
    # vpc_actual = total gross / total conversions
    if "vpc_actual" in cols and "profit_gross_actual" in cols and "conversions_actual" in cols:
        gross = float(out.get("profit_gross_actual", np.nan))
        conv = float(out.get("conversions_actual", np.nan))
        out["vpc_actual"] = gross / max(conv, eps) if np.isfinite(gross) and np.isfinite(conv) else np.nan

    # exp_vpc = weighted average of arm exp_vpc by exp_conversions_actual
    # (fallback to simple mean if weights degenerate / missing)
    if "exp_vpc" in cols:
        if "exp_conversions_actual" in cols and "exp_conversions_actual" in df.columns:
            w = pd.to_numeric(df["exp_conversions_actual"], errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(df["exp_vpc"], errors="coerce").to_numpy(dtype=float)
            wsum = float(np.nansum(w))
            out["exp_vpc"] = float(np.nansum(w * v) / wsum) if wsum > eps else float(np.nanmean(v))
        else:
            out["exp_vpc"] = float(np.nanmean(pd.to_numeric(df["exp_vpc"], errors="coerce")))

    # fc_error_vpc is NOT additive; define at portfolio level if present
    if "fc_error_vpc" in cols and ("exp_vpc" in cols) and ("vpc_actual" in cols):
        ev = out.get("exp_vpc", np.nan)
        av = out.get("vpc_actual", np.nan)
        out["fc_error_vpc"] = float(ev - av) if np.isfinite(ev) and np.isfinite(av) else np.nan

    # --- If deltas weren’t present, compute them from totals (consistent fallback) ---
    if "delta_spend" in cols and (not np.isfinite(out.get("delta_spend", np.nan))):
        if "spend_opt" in cols and "spend_actual" in cols:
            out["delta_spend"] = float(out["spend_opt"] - out["spend_actual"])

    if "delta_exp_conversions" in cols and (not np.isfinite(out.get("delta_exp_conversions", np.nan))):
        if "exp_conversions_opt" in cols and "exp_conversions_actual" in cols:
            out["delta_exp_conversions"] = float(out["exp_conversions_opt"] - out["exp_conversions_actual"])

    if "delta_exp_profit_net" in cols and (not np.isfinite(out.get("delta_exp_profit_net", np.nan))):
        if "exp_profit_net_opt" in cols and "exp_profit_net_actual" in cols:
            out["delta_exp_profit_net"] = float(out["exp_profit_net_opt"] - out["exp_profit_net_actual"])

    # --- Marginal columns (d_*) are not aggregated: leave NaN intentionally ---
    # (If you *really* want something here later, we can add explicit portfolio definitions.)

    # --- conv_is_zero flag: recompute from portfolio conversions if present ---
    if "conv_is_zero" in cols and "conversions_actual" in cols:
        conv = float(out.get("conversions_actual", 0.0))
        out["conv_is_zero"] = bool(np.isfinite(conv) and conv <= 0.0)

    # Return in original column order
    return pd.DataFrame([[out[c] for c in cols]], columns=cols)

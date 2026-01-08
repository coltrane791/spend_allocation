# src/diagnostics.py

"""
Cross-cutting diagnostics.  Guiding principle is that diagnostics should consume standardized 
artifacts (e.g.,curve grids, spend point tables) rather than re-implementing estimator math.
"""
# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Iterable, Sequence, Optional, Callable, Tuple

import numpy as np
import pandas as pd

from .checks import assert_curve_grid_contract, assert_grids_joinable
from .contracts import (
    COL_ARM_ID,
    COL_ENV_ID,
    COL_ESTIMATOR,
    COL_SPEND,
    COL_EXP_CONV,
    COL_DCONV,
)
from .estimators.estimator_rc1 import ResponseParams, expected_conversions

# Helpers
def _as_econ_table(econ_df: pd.DataFrame, *, arm_col: str) -> pd.DataFrame:
    """Normalize econ_df to columns [arm_id, p_fund, margin]."""
    if arm_col in econ_df.columns:
        econ = econ_df[[arm_col, "p_fund", "margin"]].copy()
    else:
        # allow index to be arm_id
        econ = econ_df[["p_fund", "margin"]].copy()
        econ = econ.reset_index().rename(columns={"index": arm_col})
    return econ

def _interp_1d(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Linear interpolation with clamping to domain endpoints."""
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(y[0])
    # assume x sorted
    if x0 <= x[0]:
        return float(y[0])
    if x0 >= x[-1]:
        return float(y[-1])
    return float(np.interp(x0, x, y))

def _infer_singleton_label(df: pd.DataFrame, estimator_col: str, default: str) -> str:
    if estimator_col not in df.columns:
        return default
    vals = pd.Series(df[estimator_col].dropna().unique()).astype(str).tolist()
    if len(vals) == 1:
        return vals[0]
    return default

def _contract_check(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    try:
        assert_curve_grid_contract(df, name=name)
        return pd.DataFrame({"check": ["curve_grid_contract"], "ok": [True], "rows": [len(df)]})
    except AssertionError as e:
        return pd.DataFrame({"check": ["curve_grid_contract"], "ok": [False], "error": [str(e)], "rows": [len(df)]})

def _joinable_check(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    try:
        assert_grids_joinable(
            a,
            b,
            arm_col=arm_col,
            env_id_col=env_id_col,
            spend_col=spend_col,
            atol=atol,
            rtol=rtol,
        )
        return pd.DataFrame({"check": ["grids_joinable"], "ok": [True]})
    except AssertionError as e:
        return pd.DataFrame({"check": ["grids_joinable"], "ok": [False], "error": [str(e)]})

def _grid_support_summary(
    df: pd.DataFrame,
    *,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
) -> pd.DataFrame:
    g = df.groupby([env_id_col, arm_col], sort=True)
    out = g.agg(
        n_points=(spend_col, "size"),
        spend_min=(spend_col, "min"),
        spend_max=(spend_col, "max"),
    ).reset_index()
    return out.sort_values([env_id_col, arm_col]).reset_index(drop=True)

def _grid_overview(
    df: pd.DataFrame,
    *,
    grid_name: str,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
    dcol: str,
    zero_atol: float,
) -> pd.DataFrame:
    g = (
        df.groupby([env_id_col, arm_col], sort=True)
        .agg(
            n_points=(spend_col, "size"),
            spend_min=(spend_col, "min"),
            spend_max=(spend_col, "max"),
            d_min=(dcol, "min"),
            d_max=(dcol, "max"),
            frac_zero=(dcol, lambda x: float(np.mean(np.isclose(x.to_numpy(dtype=float), 0.0, atol=zero_atol)))),
            n_unique_d=(dcol, lambda x: int(pd.Series(x).nunique(dropna=True))),
        )
        .reset_index()
    )
    g.insert(0, "grid", grid_name)
    return g.sort_values([env_id_col, "n_points", arm_col], ascending=[True, False, True]).reset_index(drop=True)

def _grid_monotonicity_summary(
    df: pd.DataFrame,
    *,
    dcol: str,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
    increase_tol: float,
) -> pd.DataFrame:
    rows = []
    for (env_id, arm), g in df.sort_values([env_id_col, arm_col, spend_col]).groupby([env_id_col, arm_col], sort=True):
        y = g[dcol].to_numpy(dtype=float)
        dy = np.diff(y)
        if dy.size == 0:
            rows.append(
                {
                    env_id_col: env_id,
                    arm_col: arm,
                    "n_increases": 0,
                    "max_step": 0.0,
                    "max_increase": 0.0,
                }
            )
            continue

        max_step = float(np.max(dy))  # can be negative if always decreasing
        n_increases = int(np.sum(dy > increase_tol))
        max_increase = float(np.max(dy[dy > increase_tol])) if n_increases else 0.0

        rows.append(
            {
                env_id_col: env_id,
                arm_col: arm,
                "n_increases": n_increases,
                "max_step": max_step,
                "max_increase": max_increase,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["n_increases", "max_increase"], ascending=[False, False])
        .reset_index(drop=True)
    )

def _grid_shape_summary(
    df: pd.DataFrame,
    *,
    grid_name: str,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
    dcol: str,
    zero_atol: float,
) -> pd.DataFrame:
    rows = []
    for (env, arm), g in df.groupby([env_id_col, arm_col], sort=True):
        gg = g.sort_values(spend_col)
        spends = gg[spend_col].to_numpy(dtype=float)
        d = gg[dcol].to_numpy(dtype=float)

        is_zero = np.isclose(d, 0.0, atol=zero_atol)
        frac_zero = float(np.mean(is_zero))

        first_zero_spend = np.nan
        if np.any(is_zero):
            first_zero_spend = float(spends[np.where(is_zero)[0][0]])

        mid_idx = int(len(spends) // 2)

        rows.append(
            {
                "grid": grid_name,
                env_id_col: env,
                arm_col: arm,
                "frac_zero": frac_zero,
                "first_zero_spend": first_zero_spend,
                "d_at_min_spend": float(d[0]),
                "d_at_mid_spend": float(d[mid_idx]),
                "d_at_max_spend": float(d[-1]),
                "d_min": float(np.min(d)),
                "d_max": float(np.max(d)),
                "n_unique_d": int(pd.Series(d).nunique(dropna=True)),
            }
        )

    out = pd.DataFrame(rows).sort_values(["frac_zero", "d_max"], ascending=[False, False]).reset_index(drop=True)
    return out

def _pairwise_diff_summary(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
    dcol: str,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    keys = [env_id_col, arm_col, spend_col]
    merged = a[keys + [dcol]].merge(
        b[keys + [dcol]],
        on=keys,
        how="inner",
        suffixes=(f"_{label_a}", f"_{label_b}"),
        validate="one_to_one",
    )

    da = merged[f"{dcol}_{label_a}"].to_numpy(dtype=float)
    db = merged[f"{dcol}_{label_b}"].to_numpy(dtype=float)

    merged["abs_diff"] = np.abs(da - db)
    merged["sq_diff"] = (da - db) ** 2

    out = (
        merged.groupby([env_id_col, arm_col], sort=True)
        .agg(
            mean_abs_diff=("abs_diff", "mean"),
            max_abs_diff=("abs_diff", "max"),
            rmse_diff=("sq_diff", lambda x: float(np.sqrt(np.mean(x.to_numpy(dtype=float))))),
        )
        .reset_index()
        .sort_values(["mean_abs_diff", "rmse_diff"], ascending=False)
        .reset_index(drop=True)
    )

    out.insert(0, "comparison", f"{label_a}_vs_{label_b}")
    return out

def _grid_env_scaling_summary(
    df: pd.DataFrame,
    *,
    dcol: str,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
) -> pd.DataFrame:
    rows = []
    for arm, g_arm in df.groupby(arm_col, sort=True):
        piv = g_arm.pivot_table(index=spend_col, columns=env_id_col, values=dcol, aggfunc="first")
        if piv.shape[1] <= 1:
            continue
        denom = piv.min(axis=1).replace(0.0, np.nan)
        ratio = piv.max(axis=1) / denom
        rows.append(
            {
                arm_col: arm,
                "ratio_p50": float(np.nanmedian(ratio.to_numpy(dtype=float))),
                "ratio_p95": float(np.nanpercentile(ratio.to_numpy(dtype=float), 95)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[arm_col, "ratio_p50", "ratio_p95"])
    return pd.DataFrame(rows).sort_values("ratio_p95", ascending=False).reset_index(drop=True)

def _grid_level_vs_slope_summary(
    df: pd.DataFrame,
    *,
    exp_conv_col: str,
    dcol: str,
    arm_col: str,
    env_id_col: str,
    spend_col: str,
) -> pd.DataFrame:
    rows = []
    for (env_id, arm), g in df.sort_values([env_id_col, arm_col, spend_col]).groupby([env_id_col, arm_col], sort=True):
        lam = g[exp_conv_col].to_numpy(dtype=float)
        dlam = g[dcol].to_numpy(dtype=float)
        lam_decreases = int(np.sum(np.diff(lam) < 0))
        dlam_negative = int(np.sum(dlam < 0))
        rows.append(
            {
                env_id_col: env_id,
                arm_col: arm,
                "lam_decreases": lam_decreases,
                "dlam_negative": dlam_negative,
            }
        )
    return pd.DataFrame(rows).sort_values(["lam_decreases", "dlam_negative"], ascending=[False, False]).reset_index(drop=True)

def _require_cols(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")

def _nearest_grid_point(spend_grid: np.ndarray, s: float) -> Tuple[float, float]:
    """
    Returns (nearest_spend, abs_diff).
    """
    if spend_grid.size == 0:
        return (np.nan, np.nan)
    idx = int(np.argmin(np.abs(spend_grid - float(s))))
    nearest = float(spend_grid[idx])
    return nearest, float(abs(nearest - float(s)))

# Marginal grid
def marginal_grids_diagnostics(
    grids: Dict[str, pd.DataFrame],
    *,
    arm_col: str = COL_ARM_ID,
    env_id_col: str = COL_ENV_ID,
    spend_col: str = COL_SPEND,
    exp_conv_col: str = COL_EXP_CONV,
    dcol: str = COL_DCONV,
    estimator_col: str = COL_ESTIMATOR,
    # joinability tolerance for spend grid equality
    atol: float = 1e-9,
    rtol: float = 0.0,
    # for interpreting “increase” in monotonicity
    increase_tol: float = 1e-12,
    # for “zero-ish” checks (shape summary)
    zero_atol: float = 1e-12,
    # reference grid name; if None, uses first key in dict
    reference: Optional[str] = None,
    # include all pairwise diff summaries (can get large as N grows)
    include_pairwise_diffs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Produce Excel-friendly diagnostics tables for N marginal curve grids (N>=2).

    `grids` is a mapping: grid_name -> curve_grid_df

    Assumes grids are intended to be joinable on (env_id, arm_id, spend).
    Returns dict[sheet_name -> DataFrame] for exports.save_workbook(...).
    """
    if not isinstance(grids, dict) or len(grids) < 2:
        raise ValueError("marginal_grids_diagnostics requires a dict of >=2 grids: {name: df}")

    names = list(grids.keys())
    ref_name = reference or names[0]
    if ref_name not in grids:
        raise KeyError(f"reference='{ref_name}' not found in grids keys={names}")

    tabs: Dict[str, pd.DataFrame] = {}

    # ---- Contract checks (combined) ----
    contract_rows = []
    for name, df in grids.items():
        cdf = _contract_check(df, name=name)
        cdf.insert(0, "grid", name)
        contract_rows.append(cdf)
    tabs["contract"] = pd.concat(contract_rows, ignore_index=True)

    # ---- Joinability vs reference (and optionally pairwise) ----
    join_rows = []
    ref_df = grids[ref_name]
    for name, df in grids.items():
        if name == ref_name:
            continue
        jdf = _joinable_check(
            ref_df,
            df,
            arm_col=arm_col,
            env_id_col=env_id_col,
            spend_col=spend_col,
            atol=atol,
            rtol=rtol,
        )
        jdf.insert(0, "grid_a", ref_name)
        jdf.insert(1, "grid_b", name)
        join_rows.append(jdf)
    tabs["joinable_vs_ref"] = pd.concat(join_rows, ignore_index=True) if join_rows else pd.DataFrame(
        {"grid_a": [ref_name], "grid_b": [ref_name], "check": ["grids_joinable"], "ok": [True]}
    )

    if include_pairwise_diffs:
        join_pair_rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = names[i], names[j]
                jdf = _joinable_check(
                    grids[a_name],
                    grids[b_name],
                    arm_col=arm_col,
                    env_id_col=env_id_col,
                    spend_col=spend_col,
                    atol=atol,
                    rtol=rtol,
                )
                jdf.insert(0, "grid_a", a_name)
                jdf.insert(1, "grid_b", b_name)
                join_pair_rows.append(jdf)
        tabs["joinable_pairwise"] = pd.concat(join_pair_rows, ignore_index=True) if join_pair_rows else pd.DataFrame()

    # ---- Overview / Support / Monotonicity / Shape (combined across grids) ----
    overview_rows = []
    support_rows = []
    mono_rows = []
    shape_rows = []
    envscale_rows = []
    lvlslope_rows = []

    for name, df in grids.items():
        overview_rows.append(
            _grid_overview(
                df,
                grid_name=name,
                arm_col=arm_col,
                env_id_col=env_id_col,
                spend_col=spend_col,
                dcol=dcol,
                zero_atol=zero_atol,
            )
        )

        s = _grid_support_summary(df, arm_col=arm_col, env_id_col=env_id_col, spend_col=spend_col)
        s.insert(0, "grid", name)
        support_rows.append(s)

        m = _grid_monotonicity_summary(
            df,
            dcol=dcol,
            arm_col=arm_col,
            env_id_col=env_id_col,
            spend_col=spend_col,
            increase_tol=increase_tol,
        )
        m.insert(0, "grid", name)
        mono_rows.append(m)

        shape_rows.append(
            _grid_shape_summary(
                df,
                grid_name=name,
                arm_col=arm_col,
                env_id_col=env_id_col,
                spend_col=spend_col,
                dcol=dcol,
                zero_atol=zero_atol,
            )
        )

        e = _grid_env_scaling_summary(df, dcol=dcol, arm_col=arm_col, env_id_col=env_id_col, spend_col=spend_col)
        if not e.empty:
            e.insert(0, "grid", name)
            envscale_rows.append(e)

        if exp_conv_col in df.columns:
            l = _grid_level_vs_slope_summary(
                df,
                exp_conv_col=exp_conv_col,
                dcol=dcol,
                arm_col=arm_col,
                env_id_col=env_id_col,
                spend_col=spend_col,
            )
            l.insert(0, "grid", name)
            lvlslope_rows.append(l)

    tabs["overview"] = pd.concat(overview_rows, ignore_index=True) if overview_rows else pd.DataFrame()
    tabs["support"] = pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame()
    tabs["monotonicity"] = pd.concat(mono_rows, ignore_index=True) if mono_rows else pd.DataFrame()
    tabs["shape_summary"] = pd.concat(shape_rows, ignore_index=True) if shape_rows else pd.DataFrame()
    tabs["env_scaling"] = pd.concat(envscale_rows, ignore_index=True) if envscale_rows else pd.DataFrame(
        columns=["grid", arm_col, "ratio_p50", "ratio_p95"]
    )
    tabs["level_vs_slope"] = pd.concat(lvlslope_rows, ignore_index=True) if lvlslope_rows else pd.DataFrame(
        columns=["grid", env_id_col, arm_col, "lam_decreases", "dlam_negative"]
    )

    # ---- Disagreement summaries ----
    # vs reference
    diff_vs_ref = []
    for name, df in grids.items():
        if name == ref_name:
            continue
        diff_vs_ref.append(
            _pairwise_diff_summary(
                grids[ref_name],
                df,
                arm_col=arm_col,
                env_id_col=env_id_col,
                spend_col=spend_col,
                dcol=dcol,
                label_a=ref_name,
                label_b=name,
            )
        )
    tabs["diff_vs_ref"] = pd.concat(diff_vs_ref, ignore_index=True) if diff_vs_ref else pd.DataFrame()

    # all pairwise
    if include_pairwise_diffs:
        pair_rows = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name, b_name = names[i], names[j]
                pair_rows.append(
                    _pairwise_diff_summary(
                        grids[a_name],
                        grids[b_name],
                        arm_col=arm_col,
                        env_id_col=env_id_col,
                        spend_col=spend_col,
                        dcol=dcol,
                        label_a=a_name,
                        label_b=b_name,
                    )
                )
        tabs["diff_pairwise"] = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

    return tabs

def build_spend_support_tables(
    ad_spend_df: pd.DataFrame,
    *,
    spend_grid_by_arm: dict[str, np.ndarray],
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    pcts: tuple[float, float, float] = (0.05, 0.50, 0.95),
    conf_local_df: Optional[pd.DataFrame] = None,
    estimator_col: str = "estimator",
    n_eff_col_candidates: Sequence[str] = ("n_eff", "n_eff_fit", "n_eff_data", "n_eff_x", "n_eff_y"),
    k_first: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Build spend-support diagnostics to explain boundary behavior.

    Returns a dict of DataFrames:
      - support_by_arm: min/p05/p50/p95/max + how grid overlaps observed spends
      - neff_firstk (optional): effective sample size at first k grid points (long form)
    """

    need = {arm_col, spend_col}
    missing = sorted(need - set(ad_spend_df.columns))
    if missing:
        raise KeyError(f"ad_spend_df missing required columns: {missing}")

    # --- per-arm observed spend distribution + overlap with grid ---
    rows = []
    for arm_id, g in ad_spend_df.groupby(arm_col, sort=True):
        if arm_id not in spend_grid_by_arm:
            raise KeyError(f"spend_grid_by_arm missing arm_id={arm_id}")

        x = pd.to_numeric(g[spend_col], errors="coerce").dropna().to_numpy(dtype=float)
        grid = np.asarray(spend_grid_by_arm[arm_id], dtype=float)

        if x.size == 0:
            rows.append(
                {
                    arm_col: arm_id,
                    "n_obs": 0,
                    "min_obs_spend": np.nan,
                    "p05_obs_spend": np.nan,
                    "p50_obs_spend": np.nan,
                    "p95_obs_spend": np.nan,
                    "max_obs_spend": np.nan,
                    "grid_min": float(np.nanmin(grid)),
                    "grid_max": float(np.nanmax(grid)),
                    "n_obs_le_grid_min": 0,
                    "n_obs_ge_grid_max": 0,
                }
            )
            continue

        p05, p50, p95 = (float(np.quantile(x, q)) for q in pcts)
        grid_min = float(np.min(grid))
        grid_max = float(np.max(grid))

        rows.append(
            {
                arm_col: arm_id,
                "n_obs": int(x.size),
                "min_obs_spend": float(np.min(x)),
                "p05_obs_spend": p05,
                "p50_obs_spend": p50,
                "p95_obs_spend": p95,
                "max_obs_spend": float(np.max(x)),
                "grid_min": grid_min,
                "grid_max": grid_max,
                "n_obs_le_grid_min": int(np.sum(x <= grid_min)),
                "n_obs_ge_grid_max": int(np.sum(x >= grid_max)),
            }
        )

    support_by_arm = pd.DataFrame(rows).sort_values(arm_col).reset_index(drop=True)

    out = {"support_by_arm": support_by_arm}

    # --- optional: n_eff at first k grid points (long form) ---
    if conf_local_df is not None:
        found_neff_cols = [c for c in n_eff_col_candidates if c in conf_local_df.columns]

        if found_neff_cols:
            base_cols = {arm_col, "spend"}
            has_estimator = estimator_col in conf_local_df.columns

            for neff_col in found_neff_cols:
                use_cols = [arm_col, "spend", neff_col] + ([estimator_col] if has_estimator else [])
                df = conf_local_df[use_cols].copy()

                df["spend"] = pd.to_numeric(df["spend"], errors="coerce")
                df[neff_col] = pd.to_numeric(df[neff_col], errors="coerce")
                df = df.dropna(subset=["spend", neff_col])

                sort_cols = [arm_col, "spend"] + ([estimator_col] if has_estimator else [])
                if has_estimator:
                    df = df.sort_values([arm_col, estimator_col, "spend"])
                    df["grid_rank"] = df.groupby([arm_col, estimator_col], sort=True).cumcount()
                else:
                    df = df.sort_values([arm_col, "spend"])
                    df["grid_rank"] = df.groupby([arm_col], sort=True).cumcount()

                firstk = df.loc[df["grid_rank"] < int(k_first)].copy()
                out[f"neff_firstk_{neff_col}"] = firstk.reset_index(drop=True)

    return out

# Allocation
@dataclass(frozen=True)
class AllocDiagCols:
    arm_col: str = "arm_id"
    spend_min_col: str = "spend_min"
    spend_opt_col: str = "spend_opt"
    spend_max_col: str = "spend_max"

    # objective marginal (the thing you optimize on)
    dprofit_col: str = "d_profit_net_d$"

    # level outputs (optional but strongly recommended)
    exp_profit_col: str = "exp_profit_net"
    exp_conv_col: str = "exp_conversions"
    dconv_col: str = "d_conv_d$"
    value_per_conv_col: str = "value_per_conv"

    # global duals (often repeated per row)
    tau_col: str = "tau"
    nu_col: str = "nu_conv"

    # optional precomputed adjusted marginal
    dprofit_adj_col: str = "d_profit_net_d$_adj"

## Optimization point
def allocation_opt_diagnostics(
    *,
    alloc_df: pd.DataFrame,
    profit_grid_df: pd.DataFrame,
    env_id: str,
    cols: AllocDiagCols = AllocDiagCols(),
    allow_unspent: Optional[bool] = None,
    budget: Optional[float] = None,
    capacity_conversions: Optional[float] = None,
    atol_bounds: float = 1e-6,
    kkt_tol: float = 1e-6,
) -> Dict[str, pd.DataFrame]:
    """
    Consolidated optimized-point-only diagnostics.

    Assumptions:
      - alloc_df does NOT include a Total row.
      - alloc_df contains spend_min/spend_opt/spend_max and d_profit_net_d$.
      - profit_grid_df contains a spend grid per arm for the same env_id.

    Returns:
      dict[sheet_name] = DataFrame
    """
    ## Contract checks
    _require_cols(
        alloc_df,
        [
            cols.arm_col,
            cols.spend_min_col,
            cols.spend_opt_col,
            cols.spend_max_col,
            cols.dprofit_col,
            cols.tau_col,
        ],
        name="alloc_df",
    )
    _require_cols(
        profit_grid_df,
        [cols.arm_col, "env_id", "spend"],
        name="profit_grid_df",
    )

    ## Create objects
    alloc = alloc_df.copy()
    grid = profit_grid_df.loc[profit_grid_df["env_id"] == env_id].copy()

    ## Captrue misnaming
    if cols.dconv_col not in alloc.columns and "d_conv_ds" in alloc.columns:
        alloc[cols.dconv_col] = alloc["d_conv_ds"]
    if cols.dconv_col not in grid.columns and "d_conv_ds" in grid.columns:
        grid[cols.dconv_col] = grid["d_conv_ds"]

    ## Optimization Summary
    ## Purpose: Provide a one-row “executive summary” of the optimized solution for the 
    ## selected env_id, including global dual values (tau/nu), totals, and slack versus 
    ## any supplied constraints (budget/capacity). This is the fastest sanity check that 
    ## the optimizer ran in the intended regime (budget binding vs unspent allowed; capacity 
    ## binding vs slack).
    ## Output: Single row with global fields: env_id, tau, nu_conv, optional allow_unspent, 
    ## budget, capacity_conversions, plus totals total_spend, total_exp_profit_net, 
    ## total_exp_conversions, and optional slack measures unspent and cap_slack.

    ### globals: tau / nu
    tau_star = float(pd.to_numeric(alloc[cols.tau_col], errors="coerce").dropna().iloc[0])
    nu_star = 0.0
    if cols.nu_col in alloc.columns:
        nu_star = float(pd.to_numeric(alloc[cols.nu_col], errors="coerce").dropna().iloc[0])

    ### totals
    total_spend = float(pd.to_numeric(alloc[cols.spend_opt_col], errors="coerce").sum())

    total_profit = float(pd.to_numeric(alloc[cols.exp_profit_col], errors="coerce").sum()) if cols.exp_profit_col in alloc.columns else np.nan
    total_conv = float(pd.to_numeric(alloc[cols.exp_conv_col], errors="coerce").sum()) if cols.exp_conv_col in alloc.columns else np.nan

    ### constraints
    unspent = np.nan
    if budget is not None:
        unspent = float(budget) - total_spend

    cap_slack = np.nan
    if capacity_conversions is not None and np.isfinite(total_conv):
        cap_slack = float(capacity_conversions) - float(total_conv)

    opt_summary = pd.DataFrame(
        [
            {
                "env_id": env_id,
                "tau": tau_star,
                "nu_conv": nu_star,
                "allow_unspent": allow_unspent,
                "budget": budget,
                "capacity_conversions": capacity_conversions,
                "total_spend": total_spend,
                "total_exp_profit_net": total_profit,
                "total_exp_conversions": total_conv,
                "unspent": unspent,
                "cap_slack": cap_slack,
            }
        ]
    )

    # Vertical orientation
    opt_summary_v = (
        opt_summary
        .T
        .reset_index()
        .rename(columns={"index": "metric", 0: "value"})
    )

    ## Value Components
    ## Purpose: Provide an interpretable decomposition of “why” each arm received spend by showing 
    ## the key economic and marginal signals at the chosen point. This is primarily for human 
    ## review and narrative: it ties together spend, marginal profit, adjusted marginal profit (if 
    ## capacity is active), and optional level outputs.
    ## Output: One row per arm containing spend_opt, d_profit_net_d$, d_profit_net_d$_adj, tau, 
    ## and—when present in alloc_df—level columns like exp_profit_net, exp_conversions, d_conv_d$, 
    ## and value_per_conv. Sorted by spend_opt descending.

    value_cols = [cols.arm_col, cols.spend_opt_col, cols.dprofit_col, cols.dprofit_adj_col, cols.tau_col]
    for c in [cols.exp_profit_col, cols.exp_conv_col, cols.dconv_col, cols.value_per_conv_col]:
        if c in alloc.columns:
            value_cols.append(c)

    value_components = alloc[value_cols].copy()
    value_components = value_components.sort_values(cols.spend_opt_col, ascending=False).reset_index(drop=True)

    ## Bounds Slack
    ## Purpose: Explain whether the solution is being driven by the min/max spend constraints. 
    ## This is the “constraint activity” view: which arms are pinned at bounds, and how much 
    ## room exists to move before hitting a bound.
    ## Output: One row per arm with spend_min, spend_opt, spend_max, boolean flags at_min/at_max, 
    ## and slack distances slack_from_min and slack_to_max. Sorted by spend_opt descending

    a = alloc[[cols.arm_col, cols.spend_min_col, cols.spend_opt_col, cols.spend_max_col]].copy()
    a[cols.spend_min_col] = pd.to_numeric(a[cols.spend_min_col], errors="coerce")
    a[cols.spend_opt_col] = pd.to_numeric(a[cols.spend_opt_col], errors="coerce")
    a[cols.spend_max_col] = pd.to_numeric(a[cols.spend_max_col], errors="coerce")

    a["at_min"] = np.isclose(a[cols.spend_opt_col], a[cols.spend_min_col], atol=atol_bounds)
    a["at_max"] = np.isclose(a[cols.spend_opt_col], a[cols.spend_max_col], atol=atol_bounds)
    a["slack_from_min"] = a[cols.spend_opt_col] - a[cols.spend_min_col]
    a["slack_to_max"] = a[cols.spend_max_col] - a[cols.spend_opt_col]

    ## KKT Per ARM
    ## Purpose: Validate the optimizer’s first-order optimality logic at the arm level
    ## (water-filling/KKT). This checks whether each arm is behaving like an interior solution 
    ## (adjusted marginal approximately equals tau) or a bound solution (inequality relative 
    ## to tau), and surfaces violations.
    ## Output: One row per arm including allocation fields and derived KKT diagnostics. Contains 
    ## computed adjusted marginal d_profit_net_d$_adj (either provided or computed as 
    ## d_profit_net_d$ - nu_conv * d_conv_d$), plus flags at_min, at_max, slack measures, 
    ## stationarity_gap (d_profit_net_d$_adj - tau), abs_gap, and kkt_ok. Sorted by spend_opt 
    ## descending.

    kkt = alloc.copy()

    ### adjusted marginal under cap: d_profit_adj = d_profit - nu * d_conv
    if cols.dprofit_adj_col in kkt.columns:
        kkt[cols.dprofit_adj_col] = pd.to_numeric(kkt[cols.dprofit_adj_col], errors="coerce")
    else:
        if cols.dconv_col in kkt.columns:
            kkt[cols.dprofit_adj_col] = (
                pd.to_numeric(kkt[cols.dprofit_col], errors="coerce")
                - nu_star * pd.to_numeric(kkt[cols.dconv_col], errors="coerce")
            )
        else:
            # If no d_conv_d$ is present, adjusted == unadjusted (cap logic can’t be checked per-arm)
            kkt[cols.dprofit_adj_col] = pd.to_numeric(kkt[cols.dprofit_col], errors="coerce")

    kkt = kkt.merge(
        a[[cols.arm_col, "at_min", "at_max", "slack_from_min", "slack_to_max"]],
        on=cols.arm_col,
        how="left",
        validate="one_to_one",
    )

    kkt["stationarity_gap"] = pd.to_numeric(kkt[cols.dprofit_adj_col], errors="coerce") - tau_star
    kkt["abs_gap"] = kkt["stationarity_gap"].abs()

    ### add binding flags (needed for interior-point checks)
    atol = 1e-6  # or use an existing atol parameter if one exists
    kkt["at_min"] = np.isclose(kkt["spend_opt"], kkt["spend_min"], atol=atol)
    kkt["at_max"] = np.isclose(kkt["spend_opt"], kkt["spend_min"], atol=atol)

    ### KKT logic by bound status
    interior = (~kkt["at_min"]) & (~kkt["at_max"])
    ok_interior = interior & (kkt["abs_gap"] <= kkt_tol)
    ok_at_min = kkt["at_min"] & (kkt["stationarity_gap"] <= kkt_tol)
    ok_at_max = kkt["at_max"] & (kkt["stationarity_gap"] >= -kkt_tol)
    kkt["kkt_ok"] = ok_interior | ok_at_min | ok_at_max
    
    ## KKT Violations
    ## Purpose: Surface only the arms that fail the KKT checks, so can focus debugging effort on 
    ## the exceptions. This is the “action list” tab.
    ## Output: Subset of kkt_per_arm where kkt_ok == False, with columns sufficient to diagnose 
    ## the failure: arm_id, spend_opt, at_min, at_max, d_profit_net_d$, d_profit_net_d$_adj, tau, 
    ## stationarity_gap, abs_gap. Sorted by abs_gap descending.

    kkt_violations = (
        kkt.loc[~kkt["kkt_ok"], [cols.arm_col, cols.spend_opt_col, "at_min", "at_max", cols.dprofit_col, cols.dprofit_adj_col, cols.tau_col, "stationarity_gap", "abs_gap"]]
        .sort_values("abs_gap", ascending=False)
        .reset_index(drop=True)
    )

    ## Grid Alignment
    ## Purpose: Confirm that the chosen spend_opt values are compatible with the discrete spend 
    ## grid used by the allocator (or, if not exactly on-grid, quantify how far off-grid they are). 
    ## This is especially useful for debugging when spend grids change or if the allocator is 
    ## modified to allow interpolation.
    ## Output: One row per arm with spend_opt, nearest_grid_spend, and abs_diff (absolute distance 
    ## between the chosen spend and the closest grid point). Sorted by abs_diff descending (largest 
    ## misalignments first).

    _require_cols(grid, ["spend"], name="profit_grid_df[env]")
    grid_align_rows = []
    for arm, sopt in a[[cols.arm_col, cols.spend_opt_col]].itertuples(index=False):
        g = grid.loc[grid[cols.arm_col] == arm, "spend"].to_numpy(dtype=float)
        nearest, diff = _nearest_grid_point(g, float(sopt))
        grid_align_rows.append(
            {
                cols.arm_col: arm,
                "spend_opt": float(sopt),
                "nearest_grid_spend": nearest,
                "abs_diff": diff,
            }
        )
    grid_alignment = pd.DataFrame(grid_align_rows).sort_values("abs_diff", ascending=False).reset_index(drop=True)

    # Return tab bundle
    tabs: Dict[str, pd.DataFrame] = {
        "opt_summary": opt_summary_v,
        "value_components": value_components,
        "bounds_slack": a.sort_values(cols.spend_opt_col, ascending=False).reset_index(drop=True),
        "kkt_per_arm": kkt.sort_values(cols.spend_opt_col, ascending=False).reset_index(drop=True),
        "kkt_violations": kkt_violations,
        "grid_alignment": grid_alignment,
    }
    return tabs

## Optimization point scenario analysis (not used)
def budget_sweep(
    allocator_fn: Callable[..., pd.DataFrame],
    *,
    marg_profit_grid_df: pd.DataFrame,
    env_id: str,
    budgets: list[float],
    allow_unspent: bool,
    capacity_conversions: Optional[float] = None,
    p: AllocDiagCols = AllocDiagCols(),
) -> pd.DataFrame:
    """
    Re-run allocator across a list of budgets; track tau and totals.

    allocator_fn must accept:
      (marg_profit_grid_df, env_id=..., budget=..., allow_unspent=..., capacity_conversions=...)
    and return alloc_df with spend_opt, exp_profit_net, exp_conversions, tau, nu_conv where available.
    """
    rows = []
    for b in budgets:
        alloct = allocator_fn(
            marg_profit_grid_df,
            env_id=env_id,
            budget=float(b),
            allow_unspent=allow_unspent,
            capacity_conversions=capacity_conversions,
        )
        alloc = alloct.iloc[:-1].copy()

        row = {
            "env_id": env_id,
            "budget": float(b),
            "total_spend": float(alloc[p.spend_opt_col].sum()),
        }
        if p.exp_profit_col in alloc.columns:
            row["total_exp_profit_net"] = float(alloc[p.exp_profit_col].sum())
        if p.exp_conv_col in alloc.columns:
            row["total_exp_conversions"] = float(alloc[p.exp_conv_col].sum())
        if p.tau_col in alloc.columns:
            row["tau"] = float(np.nanmedian(alloc[p.tau_col].to_numpy(dtype=float)))
        if p.nu_col in alloc.columns:
            row["nu_conv"] = float(np.nanmedian(alloc[p.nu_col].to_numpy(dtype=float)))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)

def cap_sweep(
    allocator_fn: Callable[..., pd.DataFrame],
    *,
    marg_profit_grid_df: pd.DataFrame,
    env_id: str,
    budget: float,
    caps: list[float],
    allow_unspent: bool,
    p: AllocDiagCols = AllocDiagCols(),
) -> pd.DataFrame:
    """
    Re-run allocator across a list of conversion capacity values; track nu and totals.
    """
    rows = []
    for cap in caps:
        try:
            alloct = allocator_fn(
                marg_profit_grid_df,
                env_id=env_id,
                budget=float(budget),
                allow_unspent=allow_unspent,
                capacity_conversions=float(cap),
            )
            alloc = alloct.iloc[:-1].copy()
            
            row = {
                "env_id": env_id,
                "budget": float(budget),
                "capacity_conversions": float(cap),
                "status": "ok",
                "message": "",
                "total_spend": float(alloc["spend_opt"].sum()),
                "total_exp_conversions": float(alloc["exp_conversions"].sum()) if "exp_conversions" in alloc.columns else np.nan,
                "total_exp_profit_net": float(alloc["exp_profit_net"].sum()) if "exp_profit_net" in alloc.columns else np.nan,
                "tau": float(np.nanmedian(alloc["tau"].to_numpy(dtype=float))) if "tau" in alloc.columns else np.nan,
                "nu_conv": float(np.nanmedian(alloc["nu_conv"].to_numpy(dtype=float))) if "nu_conv" in alloc.columns else np.nan,
            }
        except ValueError as e:
            # Most commonly infeasible due to cap < min_conv implied by min spends
            row = {
                "env_id": env_id,
                "budget": float(budget),
                "capacity_conversions": float(cap),
                "status": "infeasible",
                "message": str(e),
                "total_spend": np.nan,
                "total_exp_conversions": np.nan,
                "total_exp_profit_net": np.nan,
                "tau": np.nan,
                "nu_conv": np.nan,
            }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("capacity_conversions").reset_index(drop=True)

## Comparison
def eval_policy_on_profit_grid(
    profit_grid_df: pd.DataFrame,
    *,
    env_id: str,
    spend_by_arm: pd.Series,      # index arm_id, values spend
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    cols_to_interp: tuple[str, ...] = (
        "exp_conversions",
        "exp_profit_net",
        "d_conv_d$",
        "d_profit_net_d$",
    ),
) -> pd.DataFrame:
    """
    Interpolate grid columns at a given spend for each arm.
    Returns one row per arm_id.
    """
    g = profit_grid_df.loc[profit_grid_df["env_id"].eq(env_id)].copy()
    if g.empty:
        raise ValueError(f"profit_grid_df has no rows for env_id={env_id}")

    out_rows = []
    for arm_id, s in spend_by_arm.items():
        ga = g.loc[g[arm_col].eq(arm_id)].sort_values(spend_col)
        if ga.empty:
            raise KeyError(f"profit_grid_df missing arm_id={arm_id} for env_id={env_id}")

        x = ga[spend_col].to_numpy(dtype=float)
        s = float(s)

        # If spend is outside grid, clamp (or raise if you prefer)
        s_eval = float(np.clip(s, float(x.min()), float(x.max())))

        row = {arm_col: arm_id, "spend_eval": s, "spend_eval_clamped": s_eval}
        for c in cols_to_interp:
            y = ga[c].to_numpy(dtype=float)
            row[c] = float(np.interp(s_eval, x, y))
        out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values(arm_col).reset_index(drop=True)

def allocation_comp_diagnostics(
    day_compare_fin_df: pd.DataFrame,
    *,
    env_id: Optional[str] = None,
    arm_col: str = "arm_id",
    date_col: str = "date_eval",
    # spend + clamped spend
    spend_actual_col: str = "spend_actual",
    spend_opt_col: str = "spend_opt",
    spend_clamped_actual_col: str = "spend_clamped_actual",
    spend_clamped_opt_col: str = "spend_clamped_opt",
    # actual outcomes
    conv_actual_col: str = "conversions_actual",
    profit_actual_col: str = "profit_net_actual",
    # expected (grid-evaluated) outcomes at actual and at opt
    exp_conv_actual_col: str = "exp_conversions_actual",
    exp_profit_actual_col: str = "exp_profit_net_actual",
    exp_conv_opt_col: str = "exp_conversions_opt",
    exp_profit_opt_col: str = "exp_profit_net_opt",
    # optional precomputed columns (will be created if missing)
    fc_err_conv_col: str = "fc_error_conv",
    fc_err_vpc_col: str = "fc_error_vpc",
    fc_err_profit_col: str = "fc_error_np",
    delta_spend_col: str = "delta_spend",
    delta_exp_conv_col: str = "delta_exp_conversions",
    delta_exp_profit_col: str = "delta_exp_profit_net",
    # optional flags
    conv_is_zero_col: str = "flag_conv_act_zero",
    is_clamped_actual_col: str = "flag_clamped_actual",
    is_clamped_opt_col: str = "flag_clamped_opt",
    # numeric tolerances
    atol: float = 1e-9,
) -> Dict[str, pd.DataFrame]:
    """
    Build single-day diagnostics comparing model-optimal allocation vs actual outcomes and vs
    model-expected outcomes under actual spend (forecast error).

    Returns:
        Dict[str, pd.DataFrame] where keys are sheet/tab names.
    """
    df = day_compare_fin_df.copy()

    ## Validate minimum contract
    required = [
        arm_col,
        spend_actual_col,
        spend_opt_col,
        spend_clamped_actual_col,
        spend_clamped_opt_col,
        conv_actual_col,
        profit_actual_col,
        exp_conv_actual_col,
        exp_profit_actual_col,
        exp_conv_opt_col,
        exp_profit_opt_col,
    ]
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise KeyError(f"day_compare_fin_df missing required columns: {missing}")

    ## One row per arm
    if df.duplicated(subset=[arm_col]).any():
        dupes = df.loc[df.duplicated(subset=[arm_col], keep=False), arm_col].tolist()
        raise ValueError(f"day_compare_fin_df has duplicate {arm_col} rows: {dupes[:10]}")

    ## Ensure numeric types for core numeric columns (robust to strings)
    num_cols = [
        spend_actual_col, spend_opt_col, spend_clamped_actual_col, spend_clamped_opt_col,
        conv_actual_col, profit_actual_col,
        exp_conv_actual_col, exp_profit_actual_col,
        exp_conv_opt_col, exp_profit_opt_col,
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ## Derive standard columns if absent
    ### Forecast errors at ACTUAL spend
    if fc_err_conv_col not in df.columns:
        df[fc_err_conv_col] = df[exp_conv_actual_col] - df[conv_actual_col]
    if fc_err_profit_col not in df.columns:
        df[fc_err_profit_col] = df[exp_profit_actual_col] - df[profit_actual_col]

    ### Expected deltas: OPT vs ACTUAL (both expected)
    if delta_exp_conv_col not in df.columns:
        df[delta_exp_conv_col] = df[exp_conv_opt_col] - df[exp_conv_actual_col]
    if delta_exp_profit_col not in df.columns:
        df[delta_exp_profit_col] = df[exp_profit_opt_col] - df[exp_profit_actual_col]

    ### Spend delta (unclamped) — keep your convention
    if delta_spend_col not in df.columns:
        df[delta_spend_col] = df[spend_opt_col] - df[spend_actual_col]

    ### Also compute clamped spend delta for diagnostics (does not replace your delta_spend)
    df["delta_spend_clamped"] = df[spend_clamped_opt_col] - df[spend_clamped_actual_col]

    ### conv_is_zero flag if not present
    if conv_is_zero_col not in df.columns:
        df[conv_is_zero_col] = df[conv_actual_col].fillna(0.0).astype(float).le(0.0)

    ### clamping flags if not present
    if is_clamped_actual_col not in df.columns:
        df[is_clamped_actual_col] = ~np.isclose(
            df[spend_actual_col].to_numpy(dtype=float),
            df[spend_clamped_actual_col].to_numpy(dtype=float),
            atol=atol,
            rtol=0.0,
        )
    if is_clamped_opt_col not in df.columns:
        df[is_clamped_opt_col] = ~np.isclose(
            df[spend_opt_col].to_numpy(dtype=float),
            df[spend_clamped_opt_col].to_numpy(dtype=float),
            atol=atol,
            rtol=0.0,
        )

    ## Percent errors (safe denominators)
    df["fc_error_conversions_pct"] = np.where(
        np.abs(df[conv_actual_col].to_numpy(dtype=float)) > atol,
        df[fc_err_conv_col].to_numpy(dtype=float) / df[conv_actual_col].to_numpy(dtype=float),
        np.nan,
    )
    df["fc_error_profit_net_pct"] = np.where(
        np.abs(df[profit_actual_col].to_numpy(dtype=float)) > atol,
        df[fc_err_profit_col].to_numpy(dtype=float) / df[profit_actual_col].to_numpy(dtype=float),
        np.nan,
    )

    df["delta_exp_conversions_pct"] = np.where(
        np.abs(df[exp_conv_actual_col].to_numpy(dtype=float)) > atol,
        df[delta_exp_conv_col].to_numpy(dtype=float) / df[exp_conv_actual_col].to_numpy(dtype=float),
        np.nan,
    )
    df["delta_exp_profit_net_pct"] = np.where(
        np.abs(df[exp_profit_actual_col].to_numpy(dtype=float)) > atol,
        df[delta_exp_profit_col].to_numpy(dtype=float) / df[exp_profit_actual_col].to_numpy(dtype=float),
        np.nan,
    )

    ## Absolute error columns (useful for ranking)
    df["fc_abs_error_conversions"] = np.abs(df[fc_err_conv_col].to_numpy(dtype=float))
    df["fc_abs_error_profit_net"] = np.abs(df[fc_err_profit_col].to_numpy(dtype=float))

    ## Implied expected value of reallocating spend (clamped basis is usually more meaningful)
    denom = df["delta_spend_clamped"].to_numpy(dtype=float)
    num = df[delta_exp_profit_col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["delta_exp_profit_per$_clamped"] = np.where(np.abs(denom) > atol, num / denom, np.nan)

    ## Build diagnostic tabs
    tabs: Dict[str, pd.DataFrame] = {}

    # --- TAB: comp_summary ---
    # Purpose:
    #   Provide a single-page summary of: (a) actual outcomes, (b) forecast accuracy under actual spend,
    #   and (c) expected lift from switching from actual spend to model-opt spend (both evaluated on the grid).
    #
    # Output:
    #   One-row table with totals + a few normalized error metrics (WAPE-style) and key counts (clamping/zero-conv).
    total_spend_actual = float(df[spend_actual_col].sum())
    total_spend_opt = float(df[spend_opt_col].sum())
    total_spend_clamped_actual = float(df[spend_clamped_actual_col].sum())
    total_spend_clamped_opt = float(df[spend_clamped_opt_col].sum())

    total_conv_actual = float(df[conv_actual_col].sum())
    total_profit_actual = float(df[profit_actual_col].sum())

    total_exp_conv_actual = float(df[exp_conv_actual_col].sum())
    total_exp_profit_actual = float(df[exp_profit_actual_col].sum())

    total_exp_conv_opt = float(df[exp_conv_opt_col].sum())
    total_exp_profit_opt = float(df[exp_profit_opt_col].sum())

    # WAPE-like metrics
    sum_abs_fc_conv = float(df["fc_abs_error_conversions"].sum())
    sum_abs_fc_profit = float(df["fc_abs_error_profit_net"].sum())
    wape_conv = (sum_abs_fc_conv / total_conv_actual) if abs(total_conv_actual) > atol else np.nan
    wape_profit = (sum_abs_fc_profit / (np.abs(df[profit_actual_col]).sum() + atol))

    # ---- Spend-weighted forecast errors (overall) ----
    w = df[spend_clamped_actual_col].to_numpy(dtype=float)
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))

    e_conv = df[fc_err_conv_col].to_numpy(dtype=float)
    e_prof = df[fc_err_profit_col].to_numpy(dtype=float)

    abs_e_conv = np.abs(e_conv)
    abs_e_prof = np.abs(e_prof)

    if w_sum > atol:
        wmae_conv_spend = float(np.sum(w * abs_e_conv) / w_sum)
        wrmse_conv_spend = float(np.sqrt(np.sum(w * (e_conv**2)) / w_sum))

        wmae_profit_spend = float(np.sum(w * abs_e_prof) / w_sum)
        wrmse_profit_spend = float(np.sqrt(np.sum(w * (e_prof**2)) / w_sum))
    else:
        wmae_conv_spend = np.nan
        wrmse_conv_spend = np.nan
        wmae_profit_spend = np.nan
        wrmse_profit_spend = np.nan

    comp_summary = pd.DataFrame(
        [
            {
                "env_id": env_id,
                "date": df[date_col].iloc[0] if date_col in df.columns else None,
                "n_arms": int(df.shape[0]),
                "total_spend_actual": total_spend_actual,
                "total_spend_opt": total_spend_opt,
                "total_spend_clamped_actual": total_spend_clamped_actual,
                "total_spend_clamped_opt": total_spend_clamped_opt,
                "total_conversions_actual": total_conv_actual,
                "total_profit_net_actual": total_profit_actual,
                "total_exp_conversions_actual": total_exp_conv_actual,
                "total_exp_profit_net_actual": total_exp_profit_actual,
                "total_exp_conversions_opt": total_exp_conv_opt,
                "total_exp_profit_net_opt": total_exp_profit_opt,
                "total_fc_error_conversions": float(df[fc_err_conv_col].sum()),
                "total_fc_error_profit_net": float(df[fc_err_profit_col].sum()),
                "sum_abs_fc_error_conversions": sum_abs_fc_conv,
                "sum_abs_fc_error_profit_net": sum_abs_fc_profit,
                "wape_conversions": wape_conv,
                "wape_profit_net": wape_profit,
                "total_delta_exp_conversions": float(df[delta_exp_conv_col].sum()),
                "total_delta_exp_profit_net": float(df[delta_exp_profit_col].sum()),
                "wmae_conversions_spend": wmae_conv_spend,
                "wrmse_conversions_spend": wrmse_conv_spend,
                "wmae_profit_net_spend": wmae_profit_spend,
                "wrmse_profit_net_spend": wrmse_profit_spend,
                "n_zero_conv_arms": int(df[conv_is_zero_col].sum()),
                "n_clamped_actual_arms": int(df[is_clamped_actual_col].sum()),
                "n_clamped_opt_arms": int(df[is_clamped_opt_col].sum()),
            }
        ]
    )

    # Vertical orientation
    comp_summary_v = (
        comp_summary
        .T
        .reset_index()
        .rename(columns={"index": "metric", 0: "value"})
    )

    tabs["comp_summary"] = comp_summary_v

    # --- TAB: forecast_error_weighted ---
    # Purpose:
    #   Decision-weighted (spend-weighted) decomposition of forecast error under actual spend.
    #   Highlights arms where forecast error is large AND where spend share is large.
    #
    # Output:
    #   Per-arm table with spend weights, absolute errors, and contribution shares.
    total_w = float(df[spend_clamped_actual_col].sum())
    df["_w_spend"] = np.where(total_w > atol, df[spend_clamped_actual_col] / total_w, np.nan)

    df["_abs_fc_profit"] = np.abs(df[fc_err_profit_col].to_numpy(dtype=float))
    df["_abs_fc_conv"] = np.abs(df[fc_err_conv_col].to_numpy(dtype=float))

    sum_abs_profit = float(df["_abs_fc_profit"].sum())
    sum_abs_conv = float(df["_abs_fc_conv"].sum())

    df["_abs_fc_profit_share"] = np.where(sum_abs_profit > atol, df["_abs_fc_profit"] / sum_abs_profit, np.nan)
    df["_abs_fc_conv_share"] = np.where(sum_abs_conv > atol, df["_abs_fc_conv"] / sum_abs_conv, np.nan)

    # Weighted contributions
    df["_w_abs_fc_profit"] = df["_w_spend"] * df["_abs_fc_profit"]
    df["_w_abs_fc_conv"] = df["_w_spend"] * df["_abs_fc_conv"]

    cols = [
        arm_col,
        spend_actual_col,
        spend_clamped_actual_col,
        "_w_spend",
        fc_err_profit_col,
        "_abs_fc_profit",
        "_abs_fc_profit_share",
        "_w_abs_fc_profit",
        fc_err_conv_col,
        "_abs_fc_conv",
        "_abs_fc_conv_share",
        "_w_abs_fc_conv",
        is_clamped_actual_col,
        conv_is_zero_col,
    ]
    cols = [c for c in cols if c in df.columns]

    tabs["forecast_error_weighted"] = (
        df[cols]
        .sort_values(["_w_abs_fc_profit", "_abs_fc_profit"], ascending=False)
        .reset_index(drop=True)
    )

    # --- TAB: per_arm_core ---
    # Purpose:
    #   Per-arm “single source of truth” comparison that supports review:
    #   actual outcomes, model-expected outcomes under actual spend, model-expected outcomes under opt spend,
    #   plus forecast errors and expected deltas.
    #
    # Output:
    #   One row per arm, with the most decision-relevant fields, sortable by expected lift or by error magnitude.
    per_arm_cols = [
        arm_col,
        spend_actual_col,
        spend_clamped_actual_col,
        spend_opt_col,
        spend_clamped_opt_col,
        is_clamped_actual_col,
        is_clamped_opt_col,
        conv_actual_col,
        exp_conv_actual_col,
        exp_conv_opt_col,
        fc_err_conv_col,
        "fc_error_conversions_pct",
        profit_actual_col,
        exp_profit_actual_col,
        exp_profit_opt_col,
        fc_err_profit_col,
        "fc_error_profit_net_pct",
        delta_spend_col,
        "delta_spend_clamped",
        delta_exp_conv_col,
        "delta_exp_conversions_pct",
        delta_exp_profit_col,
        "delta_exp_profit_net_pct",
        "delta_exp_profit_per$_clamped",
        conv_is_zero_col,
    ]
    per_arm_cols = [c for c in per_arm_cols if c in df.columns]
    per_arm_core = (
        df[per_arm_cols]
        .sort_values(delta_exp_profit_col, ascending=False)
        .reset_index(drop=True)
    )
    tabs["per_arm_core"] = per_arm_core

    # --- TAB: forecast_error_rank ---
    # Purpose:
    #   Identify which arms drive forecast error under actual spend (this is the diagnostic that tells you
    #   where the model’s level prediction is miscalibrated).
    #
    # Output:
    #   Per-arm ranking table by absolute profit forecast error (primary) with conversions error alongside.
    ferr_cols = [
        arm_col,
        spend_actual_col,
        spend_clamped_actual_col,
        conv_actual_col,
        exp_conv_actual_col,
        fc_err_conv_col,
        "fc_abs_error_conversions",
        profit_actual_col,
        exp_profit_actual_col,
        fc_err_profit_col,
        "fc_abs_error_profit_net",
        is_clamped_actual_col,
        conv_is_zero_col,
    ]
    ferr_cols = [c for c in ferr_cols if c in df.columns]
    forecast_error_rank = (
        df[ferr_cols]
        .sort_values("fc_abs_error_profit_net", ascending=False)
        .reset_index(drop=True)
    )
    tabs["forecast_error_rank"] = forecast_error_rank

    # --- TAB: reco_impact ---
    # Purpose:
    #   Explain the *model’s recommendation* in expected-value terms:
    #   how much spend shifts by arm and the expected profit/conversion lift implied by the grid.
    #
    # Output:
    #   Per-arm table sorted by expected profit lift, including an “expected profit per $ shifted (clamped)”.
    rcols = [
        arm_col,
        spend_actual_col,
        spend_opt_col,
        spend_clamped_actual_col,
        spend_clamped_opt_col,
        delta_spend_col,
        "delta_spend_clamped",
        delta_exp_profit_col,
        "delta_exp_profit_net_pct",
        delta_exp_conv_col,
        "delta_exp_conversions_pct",
        "delta_exp_profit_per$_clamped",
        is_clamped_actual_col,
        is_clamped_opt_col,
    ]
    rcols = [c for c in rcols if c in df.columns]
    reco_impact = (
        df[rcols]
        .sort_values(delta_exp_profit_col, ascending=False)
        .reset_index(drop=True)
    )
    tabs["reco_impact"] = reco_impact

    # --- TAB: clamping ---
    # Purpose:
    #   Detect where comparisons are less reliable because spends were clamped to grid bounds.
    #   Clamping can affect both (a) forecast evaluation under actual spend and (b) evaluation of the opt policy.
    #
    # Output:
    #   Only arms where clamping occurred (actual or opt). Includes raw spend, clamped spend, and key expected outputs.
    clamp_mask = df[is_clamped_actual_col].astype(bool) | df[is_clamped_opt_col].astype(bool)
    clamp_cols = [
        arm_col,
        spend_actual_col,
        spend_clamped_actual_col,
        spend_opt_col,
        spend_clamped_opt_col,
        is_clamped_actual_col,
        is_clamped_opt_col,
        exp_conv_actual_col,
        exp_profit_actual_col,
        exp_conv_opt_col,
        exp_profit_opt_col,
        delta_exp_profit_col,
        delta_exp_conv_col,
    ]
    clamp_cols = [c for c in clamp_cols if c in df.columns]
    clamping = (
        df.loc[clamp_mask, clamp_cols]
        .sort_values([is_clamped_opt_col, is_clamped_actual_col, arm_col], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    tabs["clamping"] = clamping

    # --- TAB: sanity_checks ---
    # Purpose:
    #   Provide explicit “math checks” so a reviewer can confirm internal consistency:
    #   fc_error == exp(actual_spend) - actual, delta_exp == exp(opt) - exp(actual), etc.
    #
    # Output:
    #   Small table of max absolute deviations for key identities (should be ~0 up to tolerance).
    chk = {}

    chk["max_abs_fc_conv_identity"] = float(
        np.nanmax(
            np.abs(
                df[fc_err_conv_col].to_numpy(dtype=float)
                - (df[exp_conv_actual_col].to_numpy(dtype=float) - df[conv_actual_col].to_numpy(dtype=float))
            )
        )
    )
    chk["max_abs_fc_profit_identity"] = float(
        np.nanmax(
            np.abs(
                df[fc_err_profit_col].to_numpy(dtype=float)
                - (df[exp_profit_actual_col].to_numpy(dtype=float) - df[profit_actual_col].to_numpy(dtype=float))
            )
        )
    )
    chk["max_abs_delta_exp_conv_identity"] = float(
        np.nanmax(
            np.abs(
                df[delta_exp_conv_col].to_numpy(dtype=float)
                - (df[exp_conv_opt_col].to_numpy(dtype=float) - df[exp_conv_actual_col].to_numpy(dtype=float))
            )
        )
    )
    chk["max_abs_delta_exp_profit_identity"] = float(
        np.nanmax(
            np.abs(
                df[delta_exp_profit_col].to_numpy(dtype=float)
                - (df[exp_profit_opt_col].to_numpy(dtype=float) - df[exp_profit_actual_col].to_numpy(dtype=float))
            )
        )
    )

    sanity_checks = pd.DataFrame(
        [{"check": k, "value": v, "pass_at_tol": (v <= 1e-6)} for k, v in chk.items()]
    )
    tabs["sanity_checks"] = sanity_checks

    return tabs

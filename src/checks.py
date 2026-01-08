# src/checks.py

"""
Cross-cutting assertions used across the pipeline.

Intentionally lightweight and designed to fail fast with actionable
error messages. Estimator-neutral unless explicitly labeled.
"""

# Import libraries and modules
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    CURVE_GRID_KEY_COLS,
    CURVE_GRID_JOIN_COLS,
    CURVE_GRID_REQUIRED_COLS,
    SPEND_POINTS_REQUIRED_COLS,
    PROFIT_GRID_REQUIRED_COLS,
    ALLOC_OUT_REQUIRED_COLS,
)

# Data checks
def assert_has_cols(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{name} missing columns: {missing}. Found: {list(df.columns)}")

def assert_unique_col(df: pd.DataFrame, col: str, name: str) -> None:
    if col not in df.columns:
        raise AssertionError(f"{name} missing column: {col}")
    if not df[col].is_unique:
        dupes = df.loc[df[col].duplicated(), col].head(10).tolist()
        raise AssertionError(f"{name}.{col} is not unique. Example duplicates: {dupes}")

def assert_unique_row(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    arm_col: str = "arm_id",
    name: str = "df",
) -> None:
    subset = [date_col, arm_col]
    n_dupes = int(df.duplicated(subset=subset).sum())
    if n_dupes != 0:
        # NOTE: we intentionally include keys in the message for clarity.
        raise AssertionError(f"{name} has {n_dupes} duplicate rows on keys={subset}")

def assert_nonneg(df: pd.DataFrame, col: str, name: str) -> None:
    if col not in df.columns:
        raise AssertionError(f"{name} missing column: {col}")
    if (df[col] < 0).any():
        bad = df.loc[df[col] < 0, col].head(10).tolist()
        raise AssertionError(f"{name}.{col} has negative values. Examples: {bad}")

# Curve-grid checks (estimator-neutral)
def assert_curve_grid_contract(
    df: pd.DataFrame,
    *,
    name: str = "curve_grid",
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    dcol: str = "d_conv_d$",
    # Optional metadata columns (validated only if present)
    seasonality_col: str = "seasonality",
    macro_col: str = "macro_index",
    # Optional ensemble-only columns (validated only if present)
    weight_sum_col: str = "weight_sum",
    n_estimators_col: str = "n_estimators",
    # Numeric tolerances
    atol: float = 1e-9,
    weight_atol: float = 1e-6,
    require_nonnegative_dcol: bool = True,
    require_strictly_increasing_spend: bool = True,
    show: int = 10,
) -> None:
    """
    Validate the standardized marginal curve grid contract.

    Applies to both:
      - Single-estimator grids (rc1_mmm, rc2_local_slope, ...)
      - Ensemble grid (estimator='ensemble'), which may also include weight_sum, n_estimators.

    Required columns:
      estimator, env_id, arm_id, spend, d_conv_d$

    Core checks:
      - required columns exist
      - key columns non-null
      - spend and dcol are numeric + finite
      - uniqueness of (estimator, env_id, arm_id, spend)
      - spend strictly increasing within each (estimator, env_id, arm_id) group (optional)

    Optional checks (only if columns exist):
      - seasonality/macro are constant within env_id (informational sanity)
      - weight_sum ~ 1.0 (ensemble)
      - n_estimators >= 1 and integer-like (ensemble)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{name}] Expected a pandas DataFrame, got {type(df)}")

    required = [estimator_col, env_id_col, arm_col, spend_col, dcol]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{name}] Curve grid missing required columns: {missing}. Found: {list(df.columns)}")

    # Non-null keys
    key_cols = [estimator_col, env_id_col, arm_col]
    if df[key_cols].isna().any().any():
        bad = df.loc[df[key_cols].isna().any(axis=1), key_cols].head(int(show))
        raise AssertionError(f"[{name}] Nulls found in key columns {key_cols}. Sample:\n{bad.to_string(index=False)}")

    # spend numeric + finite
    spend = pd.to_numeric(df[spend_col], errors="coerce")
    if spend.isna().any():
        bad = df.loc[spend.isna(), [estimator_col, env_id_col, arm_col, spend_col]].head(int(show))
        raise AssertionError(f"[{name}] Non-numeric spend values found in '{spend_col}'. Sample:\n{bad.to_string(index=False)}")
    if not np.isfinite(spend.to_numpy(dtype=float)).all():
        bad = df.loc[~np.isfinite(spend.to_numpy(dtype=float)), [estimator_col, env_id_col, arm_col, spend_col]].head(int(show))
        raise AssertionError(f"[{name}] Non-finite spend values found in '{spend_col}'. Sample:\n{bad.to_string(index=False)}")

    # dcol numeric + finite
    dvals = pd.to_numeric(df[dcol], errors="coerce")
    if dvals.isna().any():
        bad = df.loc[dvals.isna(), [estimator_col, env_id_col, arm_col, spend_col, dcol]].head(int(show))
        raise AssertionError(f"[{name}] Non-numeric values found in '{dcol}'. Sample:\n{bad.to_string(index=False)}")
    if not np.isfinite(dvals.to_numpy(dtype=float)).all():
        bad = df.loc[~np.isfinite(dvals.to_numpy(dtype=float)), [estimator_col, env_id_col, arm_col, spend_col, dcol]].head(int(show))
        raise AssertionError(f"[{name}] Non-finite values found in '{dcol}'. Sample:\n{bad.to_string(index=False)}")

    if require_nonnegative_dcol:
        neg = dvals.to_numpy(dtype=float) < -atol
        if bool(np.any(neg)):
            bad = df.loc[neg, [estimator_col, env_id_col, arm_col, spend_col, dcol]].head(int(show))
            raise AssertionError(f"[{name}] Negative '{dcol}' values found (beyond atol={atol}). Sample:\n{bad.to_string(index=False)}")

    # Uniqueness of (estimator, env_id, arm_id, spend)
    join_key = [estimator_col, env_id_col, arm_col, spend_col]
    if df.duplicated(subset=join_key).any():
        dup = df.loc[df.duplicated(subset=join_key, keep=False), join_key].head(int(show))
        raise AssertionError(f"[{name}] Duplicate rows found for key {join_key}. Sample:\n{dup.to_string(index=False)}")

    # Spend grid strictly increasing within each (estimator, env_id, arm)
    if require_strictly_increasing_spend:
        # Sort within groups and check diffs > 0
        tmp = df[[estimator_col, env_id_col, arm_col, spend_col]].copy()
        tmp[spend_col] = spend.to_numpy(dtype=float)
        tmp = tmp.sort_values([estimator_col, env_id_col, arm_col, spend_col])

        def _bad_group(g: pd.DataFrame) -> bool:
            x = g[spend_col].to_numpy(dtype=float)
            if len(x) < 2:
                return False
            return bool(np.any(np.diff(x) <= atol))

        bad_keys = []
        for (est, env, arm), g in tmp.groupby([estimator_col, env_id_col, arm_col], sort=True):
            if _bad_group(g):
                bad_keys.append((est, env, arm))

        if bad_keys:
            sample = bad_keys[: int(show)]
            raise AssertionError(
                "Spend grid is not strictly increasing (or has repeats within atol) for some groups. "
                f"[{name}] Sample groups: {sample}"
            )

    # Optional: seasonality/macro constant within env_id
    for col in [seasonality_col, macro_col]:
        if col in df.columns:
            # within each env_id, values should be constant (or all null)
            grp = df[[env_id_col, col]].dropna()
            if not grp.empty:
                spread = grp.groupby(env_id_col, sort=True)[col].agg(lambda s: float(s.max() - s.min()))
                bad_env = spread.loc[spread > 1e-12].head(int(show))
                if len(bad_env) > 0:
                    raise AssertionError(
                        f"[{name}] Column '{col}' varies within env_id (expected constant per env). Sample spreads:\n"
                        f"{bad_env.to_string()}"
                    )

    # Optional: ensemble-only fields
    if weight_sum_col in df.columns:
        wsum = pd.to_numeric(df[weight_sum_col], errors="coerce")
        if wsum.isna().any():
            bad = df.loc[wsum.isna(), [estimator_col, env_id_col, arm_col, spend_col, weight_sum_col]].head(int(show))
            raise AssertionError(f"[{name}] Non-numeric '{weight_sum_col}' values found. Sample:\n{bad.to_string(index=False)}")
        bad = ~np.isclose(wsum.to_numpy(dtype=float), 1.0, atol=weight_atol, rtol=0.0)
        if bool(np.any(bad)):
            samp = df.loc[bad, [estimator_col, env_id_col, arm_col, spend_col, weight_sum_col]].head(int(show))
            raise AssertionError(
                f"[{name}]'{weight_sum_col}' not close to 1.0 (atol={weight_atol}) for some rows. Sample:\n{samp.to_string(index=False)}"
            )

    if n_estimators_col in df.columns:
        n = pd.to_numeric(df[n_estimators_col], errors="coerce")
        if n.isna().any():
            bad = df.loc[n.isna(), [estimator_col, env_id_col, arm_col, spend_col, n_estimators_col]].head(int(show))
            raise AssertionError(f"[{name}] Non-numeric '{n_estimators_col}' values found. Sample:\n{bad.to_string(index=False)}")
        if (n.to_numpy(dtype=float) < 1.0 - atol).any():
            bad = df.loc[n.to_numpy(dtype=float) < 1.0 - atol, [estimator_col, env_id_col, arm_col, spend_col, n_estimators_col]].head(int(show))
            raise AssertionError(f"[{name}] '{n_estimators_col}' has values < 1. Sample:\n{bad.to_string(index=False)}")
        # integer-like check
        if (np.abs(n.to_numpy(dtype=float) - np.round(n.to_numpy(dtype=float))) > 1e-6).any():
            bad = df.loc[np.abs(n.to_numpy(dtype=float) - np.round(n.to_numpy(dtype=float))) > 1e-6,
                         [estimator_col, env_id_col, arm_col, spend_col, n_estimators_col]].head(int(show))
            raise AssertionError(f"[{name}] '{n_estimators_col}' is not integer-like. Sample:\n{bad.to_string(index=False)}")

def assert_grids_joinable(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    join_cols: Optional[Sequence[str]] = None,
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    require_same_env: bool = True,
    atol: float = 1e-9,
    rtol: float = 0.0,
) -> None:
    """
    Ensure two curve grids can be joined/compared on a common spend support. This is 
    the key precondition for ensemble construction.
    """
    join_cols = list(join_cols) if join_cols is not None else list(CURVE_GRID_JOIN_COLS)
    if spend_col not in join_cols:
        raise ValueError(f"join_cols must include spend_col={spend_col!r}. Got join_cols={join_cols}")

    assert_curve_grid_contract(a, name="grid_a")
    assert_curve_grid_contract(b, name="grid_b")

    if require_same_env:
        env_a = set(a[env_id_col].unique().tolist())
        env_b = set(b[env_id_col].unique().tolist())
        if env_a != env_b:
            raise AssertionError(f"env_id mismatch: grid_a={sorted(env_a)} vs grid_b={sorted(env_b)}")

    # Compare (env, arm) support sets
    keys_a = a[[env_id_col, arm_col]].drop_duplicates()
    keys_b = b[[env_id_col, arm_col]].drop_duplicates()

    set_a = set(map(tuple, keys_a.to_numpy()))
    set_b = set(map(tuple, keys_b.to_numpy()))
    if set_a != set_b:
        only_a = list(set_a - set_b)[:10]
        only_b = list(set_b - set_a)[:10]
        raise AssertionError(
            f"(env, arm) support mismatch.\nOnly in grid_a (sample): {only_a}\nOnly in grid_b (sample): {only_b}"
        )

    # For each (env, arm), compare spend grids (float-tolerant)
    for env_id, arm_id in set_a:
        sa = a.loc[(a[env_id_col] == env_id) & (a[arm_col] == arm_id), spend_col].to_numpy(dtype=float)
        sb = b.loc[(b[env_id_col] == env_id) & (b[arm_col] == arm_id), spend_col].to_numpy(dtype=float)

        sa = np.sort(sa)
        sb = np.sort(sb)

        if sa.shape != sb.shape:
            raise AssertionError(
                f"Spend grid length mismatch for (env_id={env_id}, arm_id={arm_id}): "
                f"grid_a has {sa.shape[0]} points, grid_b has {sb.shape[0]} points."
            )

        if not np.allclose(sa, sb, atol=atol, rtol=rtol):
            diffs = sa - sb
            idx = int(np.argmax(np.abs(diffs)))
            raise AssertionError(
                f"Spend grid values differ for (env_id={env_id}, arm_id={arm_id}). "
                f"Max abs diff={float(np.max(np.abs(diffs)))} at index {idx} "
                f"(grid_a={float(sa[idx])}, grid_b={float(sb[idx])})."
            )

# Confidence and Weights check
def assert_unique_row_conf(df: pd.DataFrame, key_cols: Sequence[str], name: str) -> None:
    n_dupes = df.duplicated(subset=key_cols).sum()
    if n_dupes:
        dupes = df.loc[df.duplicated(subset=key_cols, keep=False), key_cols]
        raise AssertionError(f"{name} has {n_dupes} duplicate key rows. Example:\n{dupes.head(10)}")

def assert_weights_sum_to_one(
    weights_df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("env_id", "arm_id"),
    weight_col: str = "weight",
    atol: float = 1e-6,
    rtol: float = 0.0,
    show: int = 10,
) -> None:
    """
    Assert that weights sum to 1 within each group (default: (env_id, arm_id)).

    Raises AssertionError with a small sample of offending groups.
    """
    missing = [c for c in (*group_cols, weight_col) if c not in weights_df.columns]
    if missing:
        raise KeyError(f"weights_df missing required columns: {missing}")

    sums = (
        weights_df.groupby(list(group_cols), sort=True)[weight_col]
        .sum()
        .reset_index(name="weight_sum")
    )

    ok = np.isclose(sums["weight_sum"].to_numpy(dtype=float), 1.0, atol=atol, rtol=rtol)
    if not bool(np.all(ok)):
        bad = sums.loc[~ok].head(int(show))
        raise AssertionError(
            f"Weights do not sum to 1 within groups {group_cols} (atol={atol}, rtol={rtol}). "
            f"Sample offending groups:\n{bad.to_string(index=False)}"
        )

# Downstream value
def assert_downstream_val(dv_df: pd.DataFrame) -> None:
    if dv_df.duplicated(subset=["arm_id"]).any():
        dupes = dv_df.loc[dv_df.duplicated(subset=["arm_id"], keep=False), "arm_id"].tolist()
        raise ValueError(f"ds_val_df has duplicate arm_id rows: {dupes}")

    dv_df[["p_fund", "margin"]] = dv_df[["p_fund", "margin"]].astype(float)

    if dv_df[["p_fund", "margin"]].isna().any().any():
        raise ValueError("ds_val_df has missing p_fund or margin values")

# Profit grid
def assert_profit_grid_contract(df: pd.DataFrame, *, name: str = "profit_grid") -> None:
    missing = [c for c in PROFIT_GRID_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")

    # basic uniqueness: (env_id, arm_id, spend)
    n_dupes = df.duplicated(subset=["env_id", "arm_id", "spend"]).sum()
    if n_dupes:
        ex = df.loc[df.duplicated(subset=["env_id", "arm_id", "spend"], keep=False), ["env_id","arm_id","spend"]]
        raise AssertionError(f"{name} has {n_dupes} duplicate (env_id, arm_id, spend) rows. Example:\n{ex.head(10)}")

# Spend bounds
def assert_spend_bounds_feasible(min_spend: Dict[str, float], max_spend: Dict[str, float], budget: float, tol: float = 1e-6) -> None:
    min_total = float(sum(min_spend.values()))
    max_total = float(sum(max_spend.values()))
    if min_total - budget > tol:
        raise AssertionError(f"Infeasible bounds: sum(min_spend)={min_total:.2f} > budget={budget:.2f}")
    if budget - max_total > tol:
        raise AssertionError(f"Infeasible bounds: sum(max_spend)={max_total:.2f} < budget={budget:.2f}")

# Allocation output
def assert_alloc_output(alloc_df: pd.DataFrame, budget: float, allow_unspent: bool, name: str = "alloc_df") -> None:
    ## Missing column
    missing = [c for c in ALLOC_OUT_REQUIRED_COLS if c not in alloc_df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")

    ## Duplicate row
    if alloc_df["arm_id"].duplicated().any():
        dupes = alloc_df.loc[alloc_df["arm_id"].duplicated(keep=False), ["arm_id"]]
        raise AssertionError(f"{name} has duplicate arm_id rows. Example:\n{dupes.head(10)}")
        
    ## Budget check
    if not allow_unspent and not np.isclose(float(alloc_df["spend_opt"].sum()), budget, rtol=0, atol=1e-3):
        raise AssertionError("Allocated spend does not match budget (within tolerance)")

    ## Bounds respected
    for _, r in alloc_df.iterrows():
        a = str(r["arm_id"])
        s = float(r["spend_opt"])
        if s < float(r["spend_min"]) - 1e-6 or s > float(r["spend_max"]) + 1e-6:
            raise AssertionError(f"Spend for {a} violates bounds: {s} not in [{r["spend_min"]}, {r["spend_max"]}]")

    ## Warn (don't fail) if any arm has negative expected profit under forced-spend regime
    neg = alloc_df[alloc_df["exp_profit_net"] < 0]
    if len(neg) > 0:
        print("\nWARNING: Some arms have negative expected profit at the optimum.")
        print(neg[["arm_id", "spend_opt", "exp_profit_net"]].to_string(index=False))
        print("This can happen if min_spend constraints force spend, or if allow_unspent=False.")

# Spend points dataframe
def assert_spend_df(spend_points_df: pd.DataFrame) -> None:
    """Validate a spend-points table used for operating-point diagnostics."""
    assert_has_cols(spend_points_df, SPEND_POINTS_REQUIRED_COLS, name="spend_points_df")

    if spend_points_df["arm_id"].duplicated().any():
        raise ValueError("spend_points_df has duplicate arm_id rows")

    if spend_points_df[["spend_actual", "spend_opt"]].isna().any().any():
        missing = spend_points_df.loc[
            spend_points_df[["spend_actual", "spend_opt"]].isna().any(axis=1), "arm_id"
        ].tolist()
        raise ValueError(f"Missing spend_actual or spend_opt for arms: {missing}")

# src/allocator.py

"""
Spend allocation routine.
"""

# Import libraries and modules
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Helpers
@dataclass(frozen=True)
class _ArmGrid:
    arm_id: str
    spend: np.ndarray          # increasing
    marg: np.ndarray           # marginal net profit per $; assumed non-increasing in spend
    exp_conv: Optional[np.ndarray] = None
    exp_profit: Optional[np.ndarray] = None

def _interp_1d(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    # Safe linear interpolation (clamps to endpoints)
    return float(np.interp(float(xq), x.astype(float), y.astype(float)))

def _spend_at_threshold(g: _ArmGrid, tau: float) -> float:
    """
    For non-increasing marg(spend), return spend s such that marg(s) ~= tau
    using piecewise-linear inversion. Clamps to [min_spend, max_spend].
    """
    s = g.spend
    m = g.marg

    # Edge clamps
    if tau >= m[0]:
        return float(s[0])
    if tau <= m[-1]:
        return float(s[-1])

    # Find segment where m[i] >= tau >= m[i+1]
    # m is decreasing (or flat), so we can search linearly (small grids) or via argmax.
    for i in range(len(m) - 1):
        if m[i] >= tau >= m[i + 1]:
            # Linear interpolation on (m,s). Handle flat segment.
            if abs(m[i] - m[i + 1]) < 1e-15:
                return float(s[i + 1])
            w = (m[i] - tau) / (m[i] - m[i + 1])  # in [0,1]
            return float(s[i] + w * (s[i + 1] - s[i]))

    # Fallback (should not happen if monotone + clamps above)
    return float(s[-1])

# Format output
def _format_kkt_allocation_output(
    *,
    alloc_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    env_id: str,
    tau_star: float,
    nu_conv: float,
    arm_col: str = "arm_id",
    spend_min_col: str = "spend_min",
    spend_opt_col: str = "spend_opt",
    spend_max_col: str = "spend_max",
    # canonical marginal net profit column name
    d_profit_net_col: str = "d_profit_net_d$",
    # optional legacy names (if allocator still emits them)
    legacy_marg_col: str = "marg_opt",
    legacy_marg_adj_col: str = "marg_adj_opt",
    # grid column names (used only for interpolation at spend_opt)
    spend_col_in_grid: str = "spend",
    exp_profit_col_in_grid: str = "exp_profit_net",
    exp_conv_col_in_grid: str = "exp_conversions",
    d_conv_col_in_grid: str = "d_conv_d$",
    value_per_conv_col_in_grid: str = "value_per_conv",
) -> pd.DataFrame:
    """
    Format KKT allocation output to a standardized schema.

    Conventions:
      - The marginal objective is always named `d_profit_net_d$`.
      - Do NOT rely on any adjusted marginal column for the primary output.
        (If present, we keep it as an optional extra at the end.)

    Compute level columns (exp_conversions, exp_profit_net, d_conv_ds, value_per_conv)
    by interpolation from grid_df at spend_opt (no nearest-grid snapping).
    """

    # ---- Validate alloc_df core columns ----
    need_alloc = {arm_col, spend_min_col, spend_opt_col, spend_max_col}
    missing_alloc = sorted(need_alloc - set(alloc_df.columns))
    if missing_alloc:
        raise KeyError(f"alloc_df missing required columns: {missing_alloc}")

    # Find which marginal net-profit column to use from alloc_df
    if d_profit_net_col in alloc_df.columns:
        marg_src_col = d_profit_net_col
    elif legacy_marg_col in alloc_df.columns:
        marg_src_col = legacy_marg_col
    else:
        raise KeyError(
            f"alloc_df must contain '{d_profit_net_col}' (preferred) or '{legacy_marg_col}' (legacy). "
            f"Found columns={sorted(alloc_df.columns)}"
        )

    # Optional: keep adjusted marginal if present (but do not use it to define d_profit_net_d$)
    adj_src_col = None
    if legacy_marg_adj_col in alloc_df.columns:
        adj_src_col = legacy_marg_adj_col

    # ---- Validate grid_df ----
    need_grid = {
        "env_id",
        arm_col,
        spend_col_in_grid,
        exp_profit_col_in_grid,
        exp_conv_col_in_grid,
        d_conv_col_in_grid,
        value_per_conv_col_in_grid,
    }
    missing_grid = sorted(need_grid - set(grid_df.columns))
    if missing_grid:
        raise KeyError(f"grid_df missing required columns: {missing_grid}")

    grid_env = grid_df.loc[grid_df["env_id"] == env_id].copy()
    if grid_env.empty:
        raise ValueError(f"grid_df has no rows for env_id={env_id!r}")

    # ---- Build per-arm interpolation tables ----
    grid_by_arm: dict[str, dict[str, np.ndarray]] = {}
    for a, g in grid_env.groupby(arm_col, sort=True):
        gg = g.sort_values(spend_col_in_grid)
        grid_by_arm[a] = {
            "spend": gg[spend_col_in_grid].to_numpy(dtype=float),
            "exp_profit_net": gg[exp_profit_col_in_grid].to_numpy(dtype=float),
            "exp_conversions": gg[exp_conv_col_in_grid].to_numpy(dtype=float),
            "d_conv_d$": gg[d_conv_col_in_grid].to_numpy(dtype=float),
            "value_per_conv": gg[value_per_conv_col_in_grid].to_numpy(dtype=float),
        }

    def _interp(a: str, x: float, key: str) -> float:
        arr = grid_by_arm[a]
        xp = arr["spend"]
        fp = arr[key]
        return float(np.interp(float(x), xp, fp))  # clamps to endpoints outside range

    # ---- Assemble formatted output rows ----
    out_rows = []
    for _, r in alloc_df.iterrows():
        a = r[arm_col]
        if a not in grid_by_arm:
            raise KeyError(f"grid_df missing {arm_col}={a!r} for env_id={env_id!r}")

        s_opt = float(r[spend_opt_col])

        exp_profit_net = _interp(a, s_opt, "exp_profit_net")
        exp_conversions = _interp(a, s_opt, "exp_conversions")
        d_conv_ds = _interp(a, s_opt, "d_conv_d$")
        value_per_conv = _interp(a, s_opt, "value_per_conv")

        d_profit_net = float(r[marg_src_col])

        row_out = {
            arm_col: a,
            "spend_min": float(r[spend_min_col]),
            "spend_opt": s_opt,
            "spend_max": float(r[spend_max_col]),
            "tau": float(tau_star),
            "d_profit_net_d$": d_profit_net,   # canonical output name
            "exp_profit_net": exp_profit_net,
            "exp_conversions": exp_conversions,
            "d_conv_d$": d_conv_ds,
            "value_per_conv": value_per_conv,
        }

        # Optional extras (kept only if present)
        if adj_src_col is not None:
            row_out["marg_adj_opt"] = float(r[adj_src_col])
        row_out["nu_conv"] = float(nu_conv)

        out_rows.append(row_out)

    out = (
        pd.DataFrame(out_rows)
        .sort_values("spend_opt", ascending=False)
        .reset_index(drop=True)
    )

    # ---- Total row (requested) ----
    total = {
        arm_col: "TOTAL",
        "spend_min": float(out["spend_min"].sum()),
        "spend_opt": float(out["spend_opt"].sum()),
        "spend_max": float(out["spend_max"].sum()),
        "tau": float(tau_star),
        "d_profit_net_d$": np.nan,
        "exp_profit_net": float(out["exp_profit_net"].sum()),
        "exp_conversions": float(out["exp_conversions"].sum()),
        "d_conv_d$": np.nan,
        "value_per_conv": np.nan,
    }
    if "nu_conv" in out.columns:
        total["nu_conv"] = float(nu_conv)
    if "marg_adj_opt" in out.columns:
        total["marg_adj_opt"] = np.nan

    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

    # Column order
    base_cols = [
        "arm_id",
        "spend_min",
        "spend_opt",
        "spend_max",
        "tau",
        "d_profit_net_d$",
        "exp_profit_net",
        "exp_conversions",
        "d_conv_d$",
        "value_per_conv",
    ]
    extras = []
    if "nu_conv" in out.columns:
        extras.append("nu_conv")
    if "marg_adj_opt" in out.columns:
        out = out.rename(columns={"marg_adj_opt": "d_profit_net_d$_adj"})
        extras.append("d_profit_net_d$_adj")

    out = out[base_cols + extras]

    return out

# Main routine
def allocate_budget_kkt(
    profit_grid_df: pd.DataFrame,
    *,
    budget: float,
    env_id: Optional[str] = None,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    marg_col: str = "d_profit_net_d$",
    exp_conv_col: str = "exp_conversions",
    d_conv_col: str = "d_conv_d$",
    exp_profit_col: str = "exp_profit_net",
    val_col: str = "value_per_conv",
    allow_unspent: bool = False,
    min_spend: Optional[Dict[str, float]] = None,
    max_spend: Optional[Dict[str, float]] = None,
    capacity_conversions: Optional[float] = None,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Water-filling / KKT allocator on a marginal net-profit grid, with optional global
    conversion-capacity constraint.

    If capacity_conversions is provided, solves:
      maximize sum_i Profit_i(s_i)
      s.t.  sum_i s_i  == budget   (or <= if allow_unspent=True)
            min_i <= s_i <= max_i
            sum_i E[Conv_i(s_i)] <= capacity_conversions

    Implementation:
      - Inner bisection on tau (spend shadow price) using adjusted marginals.
      - Outer bisection on nu (conversion shadow price) to satisfy capacity.
        Adjusted marginal: marg_adj(s) = marg_profit(s) - nu * d_conv_ds(s)
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")

    need = {arm_col, spend_col, marg_col}
    missing = sorted(need - set(profit_grid_df.columns))
    if missing:
        raise KeyError(f"profit_grid_df missing required columns: {missing}")

    df = profit_grid_df.copy()
    if env_id is not None:
        if "env_id" not in df.columns:
            raise KeyError("env_id was provided but profit_grid_df has no 'env_id' column")
        df = df[df["env_id"] == env_id].copy()

    if df.empty:
        raise ValueError("profit_grid_df is empty after env_id filtering")

    min_spend = min_spend or {}
    max_spend = max_spend or {}

    if capacity_conversions is not None:
        cap = float(capacity_conversions)
        if cap < 0:
            raise ValueError("capacity_conversions must be non-negative")
        # Need levels + slopes of conversions to enforce capacity via KKT
        for col in (exp_conv_col, d_conv_col):
            if col not in df.columns:
                raise KeyError(f"capacity_conversions requires column '{col}' in profit_grid_df")

    def _finalize(out_df: pd.DataFrame, *, tau_star: float, nu_conv: float, total_spend: float, total_conv: float | None = None) -> pd.DataFrame:
        # preserve attrs you already set
        out_df.attrs["tau_star"] = float(tau_star)
        out_df.attrs["nu_conv"] = float(nu_conv)
        out_df.attrs["total_spend"] = float(total_spend)
        out_df.attrs["budget"] = float(budget)
        out_df.attrs["allow_unspent"] = bool(allow_unspent)
        if total_conv is not None:
            out_df.attrs["total_conversions"] = float(total_conv)
        if capacity_conversions is not None:
            out_df.attrs["capacity_conversions"] = float(capacity_conversions)

        # IMPORTANT: formatting requires an env_id to filter the grid
        if env_id is None:
            # if grid contains env_id and only one env is present, infer it
            if "env_id" in profit_grid_df.columns:
                uniq = pd.unique(profit_grid_df["env_id"])
                if len(uniq) == 1:
                    _env = str(uniq[0])
                else:
                    raise ValueError("allocate_budget_kkt: env_id is None but profit_grid_df has multiple env_id values; pass env_id.")
            else:
                raise ValueError("allocate_budget_kkt: env_id is required for formatted output (grid has no env_id column to infer from).")
        else:
            _env = str(env_id)

        formatted = _format_kkt_allocation_output(
            alloc_df=out_df,
            grid_df=profit_grid_df,
            env_id=_env,
            tau_star=tau_star,
            nu_conv=nu_conv,
            arm_col=arm_col,
            spend_min_col="spend_min",
            spend_opt_col="spend_opt",
            spend_max_col="spend_max",
            d_profit_net_col=marg_col,
            spend_col_in_grid=spend_col,
            exp_profit_col_in_grid=exp_profit_col,
            exp_conv_col_in_grid=exp_conv_col,
            d_conv_col_in_grid=d_conv_col,
            value_per_conv_col_in_grid=val_col,
        )
        formatted.attrs.update(out_df.attrs)
        return formatted

    def _trim_to_bounds(s: np.ndarray, y: np.ndarray, smin: float, smax: float) -> Tuple[np.ndarray, np.ndarray]:
        # Keep interior points
        keep = (s >= smin) & (s <= smax)
        s2 = s[keep]
        y2 = y[keep]

        # Ensure endpoints exist by interpolation if needed
        if s2[0] > smin + 1e-12:
            y_smin = _interp_1d(s, y, smin)
            s2 = np.insert(s2, 0, smin)
            y2 = np.insert(y2, 0, y_smin)
        if s2[-1] < smax - 1e-12:
            y_smax = _interp_1d(s, y, smax)
            s2 = np.append(s2, smax)
            y2 = np.append(y2, y_smax)

        return s2, y2

    # Precompute per-arm base grids + keep full df for interpolation of level outputs
    base_grids: list[_ArmGrid] = []
    gfull_by_arm: dict[str, pd.DataFrame] = {}

    for arm_id, g in df.groupby(arm_col, sort=True):
        arm_id = str(arm_id)
        g = g.sort_values(spend_col)
        s = g[spend_col].to_numpy(dtype=float)

        if len(s) < 2:
            raise ValueError(f"Arm {arm_id} has <2 grid points; cannot allocate.")
        if not np.all(np.diff(s) > 0):
            raise ValueError(f"Arm {arm_id} spend grid must be strictly increasing.")

        # bounds
        smin = max(float(s[0]), float(min_spend.get(arm_id, s[0])))
        smax = min(float(s[-1]), float(max_spend.get(arm_id, s[-1])))
        if smax < smin:
            raise ValueError(f"Invalid bounds for {arm_id}: max({smax}) < min({smin})")

        m = g[marg_col].to_numpy(dtype=float)
        s2, m2 = _trim_to_bounds(s, m, smin, smax)

        exp_conv = None
        exp_profit = None
        dconv2 = None

        if exp_conv_col in g.columns:
            exp_conv = g[exp_conv_col].to_numpy(dtype=float)
        if exp_profit_col in g.columns:
            exp_profit = g[exp_profit_col].to_numpy(dtype=float)

        if capacity_conversions is not None:
            dconv = g[d_conv_col].to_numpy(dtype=float)
            _, dconv2 = _trim_to_bounds(s, dconv, smin, smax)

        base_grids.append(_ArmGrid(arm_id=arm_id, spend=s2, marg=m2, exp_conv=exp_conv, exp_profit=exp_profit))
        gfull_by_arm[arm_id] = g

        # stash dconv2 on the object (lightweight; keeps backward compatibility)
        if capacity_conversions is not None:
            object.__setattr__(base_grids[-1], "dconv2", dconv2)  # type: ignore[attr-defined]

    # Feasibility on budget
    sum_mins = float(sum(gr.spend[0] for gr in base_grids))
    sum_maxs = float(sum(gr.spend[-1] for gr in base_grids))
    if budget < sum_mins - 1e-9:
        raise ValueError(f"Infeasible: budget ({budget}) < sum(min_spend) ({sum_mins}).")
    if budget > sum_maxs + 1e-9 and not allow_unspent:
        raise ValueError(f"Infeasible: budget ({budget}) > sum(max_spend) ({sum_maxs}) with allow_unspent=False.")

    # Feasibility on capacity (at mins, you cannot go lower)
    if capacity_conversions is not None:
        min_conv = 0.0
        for gr in base_grids:
            gfull = gfull_by_arm[gr.arm_id]
            min_conv += _interp_1d(
                gfull[spend_col].to_numpy(dtype=float),
                gfull[exp_conv_col].to_numpy(dtype=float),
                float(gr.spend[0]),
            )
        if min_conv > cap + 1e-6:
            raise ValueError(
                f"Infeasible: sum expected conversions at min_spend ({min_conv:.6g}) exceeds cap ({cap:.6g})."
            )

    def _solve_given_nu(nu: float) -> Tuple[pd.DataFrame, float, float, float]:
        # Build adjusted grids for this nu
        adj_grids: list[_ArmGrid] = []
        for gr in base_grids:
            if capacity_conversions is None:
                marg_adj = gr.marg
            else:
                dconv2 = getattr(gr, "dconv2")  # type: ignore[attr-defined]
                marg_adj = gr.marg - float(nu) * np.asarray(dconv2, dtype=float)
            adj_grids.append(_ArmGrid(arm_id=gr.arm_id, spend=gr.spend, marg=marg_adj))

        def total_spend(tau: float) -> float:
            return float(sum(_spend_at_threshold(gr, tau) for gr in adj_grids))

        # Find tau*
        if allow_unspent:
            spend_at_zero = total_spend(0.0)
            if spend_at_zero <= budget + tol:
                tau_star = 0.0
            else:
                tau_lo = 0.0
                tau_hi = max(float(gr.marg[0]) for gr in adj_grids) + 1.0
                for _ in range(max_iter):
                    tau_mid = 0.5 * (tau_lo + tau_hi)
                    tmid = total_spend(tau_mid)
                    if abs(tmid - budget) <= tol:
                        tau_lo = tau_hi = tau_mid
                        break
                    if tmid > budget:
                        tau_lo = tau_mid
                    else:
                        tau_hi = tau_mid
                tau_star = 0.5 * (tau_lo + tau_hi)
        else:
            tau_hi = max(float(gr.marg[0]) for gr in adj_grids) + 1.0
            tau_lo = min(float(gr.marg[-1]) for gr in adj_grids) - 1.0
            for _ in range(max_iter):
                tau_mid = 0.5 * (tau_lo + tau_hi)
                tmid = total_spend(tau_mid)
                if abs(tmid - budget) <= tol:
                    tau_lo = tau_hi = tau_mid
                    break
                if tmid > budget:
                    tau_lo = tau_mid
                else:
                    tau_hi = tau_mid
            tau_star = 0.5 * (tau_lo + tau_hi)

        # Build output + compute totals
        out_rows = []
        total_conv = 0.0
        total_sp = 0.0

        for gr_base, gr_adj in zip(base_grids, adj_grids):
            s_opt = _spend_at_threshold(gr_adj, tau_star)
            total_sp += float(s_opt)

            gfull = gfull_by_arm[gr_base.arm_id].sort_values(spend_col)

            row = {
                arm_col: gr_base.arm_id,
                "spend_min": float(gr_base.spend[0]),
                "spend_opt": float(s_opt),
                "spend_max": float(gr_base.spend[-1]),
                "tau": float(tau_star),
                "nu_conv": float(nu),
                # marginal net profit at optimum (original, not adjusted)
                "marg_opt": _interp_1d(gr_base.spend, gr_base.marg, s_opt),
                # adjusted marginal (should be ~tau at interior optima)
                "marg_adj_opt": _interp_1d(gr_adj.spend, gr_adj.marg, s_opt),
            }

            if exp_conv_col in gfull.columns:
                conv = _interp_1d(
                    gfull[spend_col].to_numpy(dtype=float),
                    gfull[exp_conv_col].to_numpy(dtype=float),
                    s_opt,
                )
                row["exp_conversions"] = conv
                total_conv += float(conv)

            if exp_profit_col in gfull.columns:
                row["exp_profit_net"] = _interp_1d(
                    gfull[spend_col].to_numpy(dtype=float),
                    gfull[exp_profit_col].to_numpy(dtype=float),
                    s_opt,
                )

            out_rows.append(row)

        out_df = pd.DataFrame(out_rows).sort_values("spend_opt", ascending=False).reset_index(drop=True)
        return out_df, float(tau_star), float(total_sp), float(total_conv)

    # No capacity: just solve once (nu = 0)
    if capacity_conversions is None:
        out_df, tau_star, total_sp, _ = _solve_given_nu(0.0)
        out_df.attrs["tau_star"] = tau_star
        out_df.attrs["nu_conv"] = 0.0
        out_df.attrs["total_spend"] = total_sp
        out_df.attrs["budget"] = float(budget)
        out_df.attrs["allow_unspent"] = bool(allow_unspent)
        return _finalize(out_df, tau_star=tau_star, nu_conv=0.0, total_spend=total_sp)

    # With capacity: outer bisection on nu
    out0, tau0, sp0, conv0 = _solve_given_nu(0.0)
    if conv0 <= cap + 1e-6:
        out0.attrs["tau_star"] = tau0
        out0.attrs["nu_conv"] = 0.0
        out0.attrs["total_spend"] = sp0
        out0.attrs["total_conversions"] = conv0
        out0.attrs["capacity_conversions"] = cap
        out0.attrs["allow_unspent"] = bool(allow_unspent)
        return _finalize(out0, tau_star=tau0, nu_conv=0.0, total_spend=sp0, total_conv=conv0)

    # Find nu_hi s.t. conv(nu_hi) <= cap (or conclude infeasible for equality-budget case)
    nu_lo = 0.0
    nu_hi = 1.0
    out_hi = None
    for _ in range(40):
        out_tmp, tau_tmp, sp_tmp, conv_tmp = _solve_given_nu(nu_hi)
        if conv_tmp <= cap + 1e-6:
            out_hi = (out_tmp, tau_tmp, sp_tmp, conv_tmp)
            break
        nu_hi *= 2.0

    if out_hi is None:
        raise ValueError(
            "Could not satisfy capacity constraint by increasing nu. "
            "If allow_unspent=False, this can occur when the cap is too tight given the required budget/min/max. "
            "Try allow_unspent=True or relax the cap/budget/bounds."
        )

    # Bisection to hit the cap (within tolerance)
    best = out_hi
    for _ in range(max_iter):
        nu_mid = 0.5 * (nu_lo + nu_hi)
        out_mid, tau_mid, sp_mid, conv_mid = _solve_given_nu(nu_mid)
        best = (out_mid, tau_mid, sp_mid, conv_mid)

        if abs(conv_mid - cap) <= 1e-4:
            nu_lo = nu_hi = nu_mid
            break

        if conv_mid > cap:
            nu_lo = nu_mid
        else:
            nu_hi = nu_mid

    out_df, tau_star, total_sp, total_conv = best
    out_df.attrs["tau_star"] = float(tau_star)
    out_df.attrs["nu_conv"] = float(out_df["nu_conv"].iloc[0]) if not out_df.empty else float("nan")
    out_df.attrs["total_spend"] = float(total_sp)
    out_df.attrs["total_conversions"] = float(total_conv)
    out_df.attrs["capacity_conversions"] = float(cap)
    out_df.attrs["budget"] = float(budget)
    out_df.attrs["allow_unspent"] = bool(allow_unspent)

    return _finalize(out_df, tau_star=tau_star, nu_conv=float(out_df["nu_conv"].iloc[0]) if not out_df.empty else float("nan"),
        total_spend=total_sp, total_conv=total_conv)

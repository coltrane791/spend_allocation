# src/estimators/estimator_rc1.py

"""
Estimator RC1: Negative-exponential saturation response curve per arm.

Functional form: 
    lambda(s; S, M) = alpha * (1 - exp(-beta*s)) * (1 + gamma*S) * (1 + delta*M)

Derivative:
    d lambda / d s = alpha * beta * exp(-beta*s) * (1 + gamma*S) * (1 + delta*M)

This module includes:
- The structural response function (expected_conversions)
- Its analytic derivative w.r.t. spend (d_expected_conversions_ds)
- Parameter fitting per arm (fit_mmm_per_arm)
- rc1-specific marginal grid generation (marg_curve_grid)
- rc1-specific scenario multipliers (scenario_multipliers)

Plan mapping:
- Step 3 (Fit): fit_mmm_per_arm
- Step 5 (Marginal grids): marg_curve_grid
"""

# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
from scipy.optimize import least_squares

import numpy as np
import pandas as pd

from ..utils import infer_col

# Core response-curve definitions
@dataclass(frozen=True)
class ResponseParams:
    alpha: float
    beta: float
    gamma: float
    delta: float

def expected_conversions(
    spend: np.ndarray,
    seasonality: np.ndarray,
    macro: np.ndarray,
    p: ResponseParams,
    eps: float = 1e-9,
) -> np.ndarray:
    """Expected conversions under the MMM functional form.

    Matches the spec functional form:

        C = alpha*(1 - exp(-beta*s))*(1 + gamma*S)*(1 + delta*M)

    with numerical guards to keep lambda positive.
    """
    spend = np.asarray(spend, dtype=float)
    seasonality = np.asarray(seasonality, dtype=float)
    macro = np.asarray(macro, dtype=float)

    sat = 1.0 - np.exp(-np.maximum(p.beta, eps) * np.maximum(spend, 0.0))
    mult_s = 1.0 + p.gamma * seasonality
    mult_m = 1.0 + p.delta * macro

    lam = np.maximum(p.alpha, eps) * sat * mult_s * mult_m
    return np.maximum(lam, eps)

def d_expected_conversions_ds(
    spend: float,
    S: float,
    M: float,
    p: ResponseParams,
    eps: float = 1e-9,
) -> float:
    """Analytic derivative d lambda / d spend.

    If lambda(s) = alpha*(1 - exp(-beta*s))*(1 + gamma*S)*(1 + delta*M),
    then:

        d lambda / d s = alpha*beta*exp(-beta*s)*(1 + gamma*S)*(1 + delta*M)

    Notes
    -----
    - spend is clamped at 0 for numerical stability.
    - beta/alpha are clamped to eps to avoid zeros.
    """
    beta = max(float(p.beta), eps)
    alpha = max(float(p.alpha), eps)
    mult = (1.0 + float(p.gamma) * float(S)) * (1.0 + float(p.delta) * float(M))
    return float(alpha * beta * np.exp(-beta * max(float(spend), 0.0)) * mult)

# Fitting routine
def fit_arm_response_curve(
    df_arm: pd.DataFrame,
    spend_col: str,
    conv_col: str,
    seasonality_col: str,
    macro_col: str,
) -> Tuple[ResponseParams, Dict[str, float]]:
    """Fit a single arm's response parameters.

    Robust Poisson-ish fitting via variance-stabilized residuals:

        r = sqrt(y + 0.5) - sqrt(lambda + 0.5)

    Minimizes sum(r^2) with bounds to keep parameters sane.
    """
    s = df_arm[spend_col].to_numpy(dtype=float)
    y = df_arm[conv_col].to_numpy(dtype=float)
    S = df_arm[seasonality_col].to_numpy(dtype=float)
    M = df_arm[macro_col].to_numpy(dtype=float)

    # Initial guesses
    alpha0 = max(1.0, float(np.percentile(y, 95) + 1.0))
    beta0 = 1.0 / max(1.0, float(np.mean(s) + 1e-6))
    gamma0 = 0.0
    delta0 = 0.0
    x0 = np.array([alpha0, beta0, gamma0, delta0], dtype=float)

    # Bounds (conservative; can be widened later)
    # alpha: positive scale; beta: positive; gamma/delta: moderate range
    lb = np.array([1e-6, 1e-12, -2.0, -2.0], dtype=float)
    ub = np.array([1e9, 1.0, 2.0, 2.0], dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        p = ResponseParams(alpha=float(x[0]), beta=float(x[1]), gamma=float(x[2]), delta=float(x[3]))
        lam = expected_conversions(s, S, M, p)
        return np.sqrt(y + 0.5) - np.sqrt(lam + 0.5)

    res = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")

    p_hat = ResponseParams(
        alpha=float(res.x[0]),
        beta=float(res.x[1]),
        gamma=float(res.x[2]),
        delta=float(res.x[3]),
    )

    lam_hat = expected_conversions(s, S, M, p_hat)
    rmse = float(np.sqrt(np.mean((y - lam_hat) ** 2)))
    mae = float(np.mean(np.abs(y - lam_hat)))

    diagnostics = {
        "rmse": rmse,
        "mae": mae,
        "n_obs": float(len(df_arm)),
        "success": float(bool(res.success)),
        "cost": float(res.cost),
    }
    return p_hat, diagnostics

def fit_per_arm_rc1(
    ad_spend_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    date_col: str = "date",
    arm_col: str = "arm_id",
    spend_col: Optional[str] = None,
    conversions_col: Optional[str] = None,
    seasonality_col: str = "seasonality",
    macro_col: str = "macro_index",
) -> pd.DataFrame:
    """Fit MMM response curves per arm.

    Returns params_df with one row per arm and fitted alpha/beta/gamma/delta + diagnostics.
    """
    spend_col = spend_col or infer_col(ad_spend_df, ["spend", "daily_spend", "cost"])
    conversions_col = conversions_col or infer_col(
        ad_spend_df,
        ["conversions", "conversions_realized", "realized_conversions", "conv", "C"],
    )

    df = ad_spend_df.merge(
        macro_df[[date_col, seasonality_col, macro_col]],
        on=date_col,
        how="left",
        validate="many_to_one",
    )

    out_rows = []
    for arm, g in df.groupby(arm_col, sort=True):
        p_hat, diag = fit_arm_response_curve(
            g,
            spend_col=spend_col,
            conv_col=conversions_col,
            seasonality_col=seasonality_col,
            macro_col=macro_col,
        )
        out_rows.append(
            {
                arm_col: arm,
                "alpha_hat": p_hat.alpha,
                "beta_hat": p_hat.beta,
                "gamma_hat": p_hat.gamma,
                "delta_hat": p_hat.delta,
                **diag,
            }
        )

    return pd.DataFrame(out_rows).sort_values(arm_col).reset_index(drop=True)

# Environment scenario multpliers (not used)
def environ_multipliers(
    environs_df: pd.DataFrame,
    params_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    env_id_col: str = "env_id",
    seasonality_col: str = "seasonality",
    macro_col: str = "macro_index",
    gamma_col: Optional[str] = None,
    delta_col: Optional[str] = None,
) -> pd.DataFrame:
    """Compute rc1-specific multiplicative factors (1+gamma*S)(1+delta*M).

    Returns long table with one row per (scenario_id, arm_id).

    This is useful because the rc1 derivative and exp_conversions scale by the
    same multiplicative constant across environments (S,M).
    """
    for c in [env_id_col, seasonality_col, macro_col]:
        if c not in environs_df.columns:
            raise KeyError(f"environs_df missing required column: {c}")

    gamma_col = gamma_col or infer_col(params_df, ["gamma", "gamma_hat"])
    delta_col = delta_col or infer_col(params_df, ["delta", "delta_hat"])

    ptab = params_df[[arm_col, gamma_col, delta_col]].copy()
    if ptab.duplicated(subset=[arm_col]).any():
        dupes = ptab.loc[ptab.duplicated(subset=[arm_col], keep=False), arm_col].tolist()
        raise ValueError(f"params_df has duplicate {arm_col} rows; duplicates={dupes}")

    ptab = ptab.set_index(arm_col).rename(columns={gamma_col: "gamma", delta_col: "delta"})

    rows = []
    for _, env in environs_df.iterrows():
        env_id = env[env_id_col]
        S = float(env[seasonality_col])
        M = float(env[macro_col])

        for arm_id in ptab.index:
            gamma = float(ptab.loc[arm_id, "gamma"])
            delta = float(ptab.loc[arm_id, "delta"])
            k = (1.0 + gamma * S) * (1.0 + delta * M)
            rows.append(
                {
                    env_id_col: env_id,
                    seasonality_col: S,
                    macro_col: M,
                    arm_col: str(arm_id),
                    "mult_factor": float(k),
                }
            )

    return pd.DataFrame(rows)

# Marginal curve construction
def marg_curve_grid_rc1(
    *,
    estimator: str,
    env_id: str,
    params_df: pd.DataFrame,
    min_spend: Dict[str, float],
    max_spend: Dict[str, float],
    S: float,
    M: float,
    arm_col: str = "arm_id",
    n_points: int = 25,
    spend_grid_by_arm: Optional[Dict[str, np.ndarray]] = None,
    alpha_col: Optional[str] = None,
    beta_col: Optional[str] = None,
    gamma_col: Optional[str] = None,
    delta_col: Optional[str] = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Thin marginal-value curve grid for ONE environment (S, M), focused on d_conv_d$.

    One row per (arm_id, spend_grid_point).

    Output columns (contract + some optional extras):
      estimator, env_id, arm_id, spend, seasonality, macro_index,
      exp_conversions, d_conv_d$

    Notes
    -----
    - Uses the MMM functional form for exp_conversions.
    - Uses d_expected_conversions_ds for d_conv_d$ (consistency with other code).
    - Does NOT include economics columns (p_fund/margin) by design.
    """

    # Infer param columns if not specified
    alpha_col = alpha_col or infer_col(params_df, ["alpha", "alpha_hat"])
    beta_col = beta_col or infer_col(params_df, ["beta", "beta_hat"])
    gamma_col = gamma_col or infer_col(params_df, ["gamma", "gamma_hat"])
    delta_col = delta_col or infer_col(params_df, ["delta", "delta_hat"])

    ptab = params_df[[arm_col, alpha_col, beta_col, gamma_col, delta_col]].copy()
    if ptab.duplicated(subset=[arm_col]).any():
        dupes = ptab.loc[ptab.duplicated(subset=[arm_col], keep=False), arm_col].tolist()
        raise ValueError(f"params_df has duplicate {arm_col} rows; duplicates={dupes}")

    ptab = ptab.set_index(arm_col).rename(
        columns={alpha_col: "alpha", beta_col: "beta", gamma_col: "gamma", delta_col: "delta"}
    )

    S = float(S)
    M = float(M)

    out_frames: list[pd.DataFrame] = []

    for arm_id in ptab.index:
        lo = float(min_spend.get(str(arm_id), 0.0))
        hi = float(max_spend.get(str(arm_id), lo))
        if hi < lo:
            raise ValueError(f"Invalid bounds for {arm_id}: max({hi}) < min({lo})")

        if spend_grid_by_arm is not None:
            spends = np.asarray(spend_grid_by_arm[str(arm_id)], dtype=float)
        else:
            spends = np.linspace(lo, hi, int(n_points), dtype=float)

        pr = ptab.loc[arm_id]
        p = ResponseParams(
            alpha=float(pr["alpha"]),
            beta=float(pr["beta"]),
            gamma=float(pr["gamma"]),
            delta=float(pr["delta"]),
        )

        # exp_conversions: vectorized
        lam = expected_conversions(
            spend=spends,
            seasonality=np.full_like(spends, S, dtype=float),
            macro=np.full_like(spends, M, dtype=float),
            p=p,
            eps=eps,
        )

        # d_conv_d$: use the scalar derivative for consistency
        dlam = np.array([d_expected_conversions_ds(float(s), S, M, p, eps=eps) for s in spends], dtype=float)

        df_arm = pd.DataFrame(
            {
                "estimator": estimator,
                "env_id": env_id,
                arm_col: str(arm_id),
                "spend": spends.astype(float),
                "seasonality": S,
                "macro_index": M,
                "exp_conversions": lam.astype(float),
                "d_conv_d$": dlam.astype(float),
            }
        )

        out_frames.append(df_arm)

    out = pd.concat(out_frames, ignore_index=True)
    out = out.sort_values([arm_col, "spend"]).reset_index(drop=True)
    return out

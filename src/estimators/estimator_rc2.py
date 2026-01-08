# src/estimators/estimator_rc2.py

"""
Estimator RC2: Michaelis–Menten (hyperbolic) response curve per arm.

Functional form:
    lambda(s; S, M) = alpha * s / (s + beta) * (1 + gamma*S) * (1 + delta*M)

Derivative:
    d lambda / d s = alpha * beta / (s + beta)^2 * (1 + gamma*S) * (1 + delta*M)
"""

# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import curve_fit
from typing import Optional, Dict, Sequence
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
_EPS = 1e-9

import numpy as np
import pandas as pd

from ..utils import infer_col

# Core response-curve definitions
@dataclass
class ResponseParamsRC2:
    alpha: float
    beta: float
    gamma: float
    delta: float

def expected_conversions_rc2(spend: float | np.ndarray, S: float, M: float, p: ResponseParamsRC2) -> np.ndarray:
    """Expected conversions under the RC2 (Michaelis–Menten) model."""
    spend = np.asarray(spend, dtype=float)
    mult = (1.0 + p.gamma * S) * (1.0 + p.delta * M)
    lam = p.alpha * (spend / (spend + np.maximum(p.beta, 1e-9))) * mult
    return np.maximum(lam, 0.0)

def d_expected_conversions_ds_rc2(
    spend: float | np.ndarray,
    S: float,
    M: float,
    p: ResponseParamsRC2,
    eps: float = 1e-9,
) -> np.ndarray:
    """Derivative of expected conversions wrt spend for RC2 model."""
    spend = np.asarray(spend, dtype=float)
    spend_pos = np.maximum(spend, 0.0)

    alpha = max(float(p.alpha), eps)
    beta = max(float(p.beta), eps)

    mult = (1.0 + float(p.gamma) * float(S)) * (1.0 + float(p.delta) * float(M))

    lam_prime = alpha * beta / np.power(spend_pos + beta, 2) * mult
    return np.maximum(lam_prime, 0.0)

# Fitting routine
def _mm_curve(xdata, alpha, beta, gamma, delta):
    """
    xdata is a tuple: (spend, S, M), all same length arrays.
    """
    spend, S, M = xdata
    spend = np.asarray(spend, dtype=float)
    S = np.asarray(S, dtype=float)
    M = np.asarray(M, dtype=float)

    beta_eff = np.maximum(beta, _EPS)
    mult = (1.0 + gamma * S) * (1.0 + delta * M)
    return alpha * (spend / (spend + beta_eff)) * mult

def fit_per_arm_rc2(
    ad_spend_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    date_col: str = "date",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    seasonality_col: str = "seasonality",
    macro_col: str = "macro_index",
) -> pd.DataFrame:
    """
    Fit the RC2 Michaelis–Menten response curve for each arm separately.
    Uses non-linear least squares (scipy.curve_fit).
    """
    # Join seasonality/macro onto spend panel
    panel = ad_spend_df.merge(
        macro_df[[date_col, seasonality_col, macro_col]],
        on=date_col,
        how="left",
        validate="many_to_one",
    ).copy()

    results = []

    for arm, grp in panel.groupby(arm_col):
        spend = grp[spend_col].to_numpy(dtype=float)
        conv = grp[conv_col].to_numpy(dtype=float)
        S = grp[seasonality_col].to_numpy(dtype=float)
        M = grp[macro_col].to_numpy(dtype=float)

        # Initial guesses
        a0 = float(np.maximum(conv.max(), 1e-3))
        b0 = float(np.maximum(spend.mean(), 1e-3))
        g0, d0 = 0.0, 0.0

        # Bounds: keep things sane
        lb = [1e-6, 1e-6, -2.0, -2.0]
        ub = [1e9,  1e9,   2.0,  2.0]

        xdata = (spend, S, M)

        try:
            # You can ignore the covariance warning since we don't use pcov,
            # but the real fix is that gamma/delta are now actually identified.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(
                    _mm_curve,
                    xdata,
                    conv,
                    p0=[a0, b0, g0, d0],
                    bounds=(lb, ub),
                    maxfev=20000,
                )
            alpha, beta, gamma, delta = map(float, popt)

        except Exception:
            alpha, beta, gamma, delta = np.nan, np.nan, np.nan, np.nan

        results.append(
            {
                arm_col: arm,
                "alpha_hat": alpha,
                "beta_hat": beta,
                "gamma_hat": gamma,
                "delta_hat": delta,
            }
        )

    return pd.DataFrame(results)

# Marginal curve construction
def marg_curve_grid_rc2(
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
    """
    RC2 marginal-value curve grid for ONE environment (S, M), focused on d_conv_d$.

    One row per (arm_id, spend_grid_point).

    Output columns (matches curve-grid contract):
      estimator, env_id, arm_id, spend, seasonality, macro_index,
      exp_conversions, d_conv_d$
    """

    def _infer_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
        return infer_col(df, list(candidates))

    # Infer param columns if not specified
    alpha_col = alpha_col or _infer_col(params_df, ["alpha", "alpha_hat"])
    beta_col  = beta_col  or _infer_col(params_df, ["beta", "beta_hat"])
    gamma_col = gamma_col or _infer_col(params_df, ["gamma", "gamma_hat"])
    delta_col = delta_col or _infer_col(params_df, ["delta", "delta_hat"])

    # Param table: one row per arm_id
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
        arm_key = str(arm_id)

        lo = float(min_spend.get(arm_key, min_spend.get(arm_id, 0.0)))
        hi = float(max_spend.get(arm_key, max_spend.get(arm_id, lo)))
        if hi < lo:
            raise ValueError(f"Invalid bounds for {arm_id}: max({hi}) < min({lo})")

        if spend_grid_by_arm is not None:
            spends = np.asarray(spend_grid_by_arm[arm_key], dtype=float) if arm_key in spend_grid_by_arm else np.asarray(
                spend_grid_by_arm[arm_id], dtype=float
            )
        else:
            spends = np.linspace(lo, hi, int(n_points), dtype=float)

        # Params
        alpha = float(ptab.loc[arm_id, "alpha"])
        beta  = float(ptab.loc[arm_id, "beta"])
        gamma = float(ptab.loc[arm_id, "gamma"])
        delta = float(ptab.loc[arm_id, "delta"])

        p = ResponseParamsRC2(alpha=alpha, beta=beta, gamma=gamma, delta=delta)

        # Response + derivative (vectorized)
        lam = expected_conversions_rc2(spends, S, M, p)
        dlam = d_expected_conversions_ds_rc2(spends, S, M, p, eps=eps)

        df_arm = pd.DataFrame(
            {
                "estimator": estimator,
                "env_id": env_id,
                arm_col: arm_key,
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

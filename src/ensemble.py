# src/ensemble.py

"""
Accept confidence measures, compute weights and construct ensemble grid.
"""

# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# Weights
## Per-arm confidence
def make_weights(
    conf_df: pd.DataFrame,
    *,
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    conf_fit_col: str = "conf_fit",
    conf_data_col: str = "conf_data",
    conf_stability_col: str = "conf_stability",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Build per-(env, arm) weights across estimators.

    Expected input: one row per (estimator, env_id, arm_id) with confidence components.
    Output: same keys + weight, weight_raw
    """
    df = conf_df.copy()

    # default missing components to 1.0 (so you can stage them in later)
    if conf_data_col not in df.columns:
        df[conf_data_col] = 1.0
    if conf_stability_col not in df.columns:
        df[conf_stability_col] = 1.0

    for c in [conf_fit_col, conf_data_col, conf_stability_col]:
        df[c] = df[c].astype(float).fillna(0.0)

    df["weight_raw"] = df[conf_fit_col] * df[conf_data_col] * df[conf_stability_col]
    df["weight_raw"] = np.maximum(df["weight_raw"].to_numpy(dtype=float), 0.0)

    gcols = [env_id_col, arm_col]
    denom = df.groupby(gcols, sort=True)["weight_raw"].transform("sum").astype(float)
    df["weight"] = np.where(denom > eps, df["weight_raw"] / denom, 0.0)

    # Optional: if denom==0 (all confidence 0), fall back to uniform weights
    mask = denom <= eps
    if mask.any():
        n_est = df.groupby(gcols, sort=True)[estimator_col].transform("count").astype(float)
        df.loc[mask, "weight"] = 1.0 / np.maximum(n_est[mask], 1.0)

    out_cols = [estimator_col, env_id_col, arm_col, conf_fit_col, conf_data_col, conf_stability_col, "weight_raw", "weight"]
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].sort_values([env_id_col, arm_col, "weight"], ascending=[True, True, False]).reset_index(drop=True)

## Spend level confidence
def make_weights_local(
    conf_local_df: pd.DataFrame,
    *,
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conf_fit_col: str = "conf_fit_local",
    conf_data_col: str = "conf_data_local",
    conf_stability_col: str = "conf_stability_local",
    weight_col: str = "weight",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Build spend-local weights across estimators.

    Expected input: one row per (env_id, estimator, arm_id, spend) with confidence components.
    Output: same keys + weight_raw, weight.

    Normalization is done within each (env_id, arm_id, spend) group.
    """

    df = conf_local_df.copy()

    # Required key columns
    req = {estimator_col, env_id_col, arm_col, spend_col}
    missing = sorted(req - set(df.columns))
    if missing:
        raise KeyError(f"conf_local_df missing required columns: {missing}")

    # Default missing components to 1.0 so you can stage them in later
    if conf_data_col not in df.columns:
        df[conf_data_col] = 1.0
    if conf_stability_col not in df.columns:
        df[conf_stability_col] = 1.0
    if conf_fit_col not in df.columns:
        raise KeyError(f"conf_local_df missing required column: {conf_fit_col}")

    # Ensure numeric, fill missing with 0 (conservative)
    for c in [conf_fit_col, conf_data_col, conf_stability_col]:
        df[c] = df[c].astype(float).fillna(0.0)

    # Raw weight = product of components
    w_raw = (
        df[conf_fit_col].to_numpy(dtype=float)
        * df[conf_data_col].to_numpy(dtype=float)
        * df[conf_stability_col].to_numpy(dtype=float)
    )
    w_raw = np.maximum(w_raw, 0.0)
    df["weight_raw"] = w_raw

    # Normalize within (env, arm, spend)
    gcols = [env_id_col, arm_col, spend_col]
    denom = df.groupby(gcols, sort=True)["weight_raw"].transform("sum").astype(float)

    df[weight_col] = np.where(denom > eps, df["weight_raw"] / denom, 0.0)

    # If denom==0 (all confidence 0), fall back to uniform weights in that group
    mask = denom <= eps
    if mask.any():
        n_est = df.groupby(gcols, sort=True)[estimator_col].transform("count").astype(float)
        df.loc[mask, weight_col] = 1.0 / np.maximum(n_est[mask], 1.0)

    out_cols = [
        estimator_col,
        env_id_col,
        arm_col,
        spend_col,
        conf_fit_col,
        conf_data_col,
        conf_stability_col,
        "weight_raw",
        weight_col,
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    return (
        df[out_cols]
        .sort_values([env_id_col, arm_col, spend_col, weight_col], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

# Ensemble grid
def build_ensemble_grid(
    grids: list[pd.DataFrame],
    weights_df: pd.DataFrame,
    *,
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    dcol: str = "d_conv_d$",
    weight_col: str = "weight",
    out_estimator: str = "ensemble",
    carry_cols: tuple[str, ...] = ("seasonality", "macro_index"),
    atol: float = 1e-6,
) -> pd.DataFrame:
    """
    Weighted combine of marginal grids.

    - Assumes grids are joinable on (env_id, arm_id, spend).
    - Supports weights that are either:
        (estimator, env_id, arm_id)
      or
        (estimator, env_id, arm_id, spend)  [spend-varying weights]

    Returns one row per (env_id, arm_id, spend) with ensemble d_conv_d$.
    Also carries through selected constant columns (e.g., seasonality, macro_index) if present.
    """

    if not grids:
        raise ValueError("build_ensemble_grid: grids list is empty")

    df = pd.concat(grids, ignore_index=True).copy()

    # --- Required columns in grids ---
    req_grid = {estimator_col, env_id_col, arm_col, spend_col, dcol}
    missing = req_grid - set(df.columns)
    if missing:
        raise KeyError(f"Grid(s) missing required columns: {sorted(missing)}")

    # --- Determine merge keys (support spend-varying weights later) ---
    merge_keys = [estimator_col, env_id_col, arm_col]
    if spend_col in weights_df.columns:
        merge_keys = merge_keys + [spend_col]

    # --- Required columns in weights ---
    req_w = set(merge_keys + [weight_col])
    missing_w = req_w - set(weights_df.columns)
    if missing_w:
        raise KeyError(f"weights_df missing required columns: {sorted(missing_w)}")

    wsub = weights_df[merge_keys + [weight_col]].copy()

    # Attach weights
    df = df.merge(
        wsub,
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    if df[weight_col].isna().any():
        miss = df.loc[df[weight_col].isna(), merge_keys].drop_duplicates()
        raise ValueError(f"Missing weights for some key rows (showing unique keys):\n{miss}")

    # Numeric types
    df[dcol] = df[dcol].astype(float)
    df[weight_col] = df[weight_col].astype(float)

    # Weighted sum per (env, arm, spend)
    gcols = [env_id_col, arm_col, spend_col]
    df["_wd"] = df[dcol] * df[weight_col]

    out = (
        df.groupby(gcols, sort=True, as_index=False)
        .agg(
            **{
                dcol: ("_wd", "sum"),
                "weight_sum": (weight_col, "sum"),
                "n_estimators": (estimator_col, "nunique"),
            }
        )
    )

    # Validate weight normalization (should be ~1 per (env, arm, spend))
    bad = out.loc[~np.isclose(out["weight_sum"].astype(float), 1.0, atol=atol)]
    if len(bad) > 0:
        sample = bad.head(10)
        raise ValueError(
            "Weights do not sum to 1 for some (env, arm, spend) groups. "
            f"First rows:\n{sample}"
        )

    # Carry through constant columns (if present)
    carry_cols_present = [c for c in carry_cols if c in df.columns]
    if carry_cols_present:
        carry = (
            df.groupby(gcols, sort=True)[carry_cols_present]
            .first()
            .reset_index()
        )
        out = out.merge(carry, on=gcols, how="left", validate="one_to_one")

    out[estimator_col] = out_estimator

    # Column order
    base_cols = [estimator_col, env_id_col, arm_col, spend_col]
    meta_cols = carry_cols_present + ["n_estimators", "weight_sum"]
    out = out[base_cols + meta_cols + [dcol]]

    return out.sort_values([env_id_col, arm_col, spend_col]).reset_index(drop=True)

## Add expected conversions
def _interp1(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """1D linear interpolation with edge clamping."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return float(y[0])
    # np.interp clamps outside range
    return float(np.interp(float(x0), x, y))

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (u ** 2))

def _effective_n(w: np.ndarray, eps: float = 1e-12) -> float:
    """Effective sample size from weights vector."""
    w = np.asarray(w, dtype=float)
    sw = float(np.sum(w))
    sw2 = float(np.sum(w ** 2))
    if sw2 <= eps:
        return 0.0
    return (sw * sw) / sw2

def _default_bandwidth_from_grid(grid: np.ndarray, bandwidth_mult: float = 2.5, eps: float = 1e-9) -> float:
    """Default bandwidth ~ (median grid step) * mult."""
    g = np.asarray(grid, dtype=float)
    if g.size < 2:
        return 1.0
    step = np.median(np.diff(np.sort(g)))
    return float(max(step * float(bandwidth_mult), eps))

def _cumulative_trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral with y evaluated on x. Returns array same length as x."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    if x.size == 1:
        return np.array([0.0], dtype=float)
    dx = np.diff(x)
    area = 0.5 * (y[:-1] + y[1:]) * dx
    return np.concatenate([[0.0], np.cumsum(area)])

@dataclass(frozen=True)
class AnchorConfig:
    env_ref_id: str = "base_last"          # interpret empirical anchor as living in this env
    kernel_bandwidth_mult: float = 2.5     # ~2–3 grid steps
    n0_anchor: float = 10.0                # gating scale for empirical-vs-model blend
    eps: float = 1e-9

def add_exp_conversions_to_ensemble_grid(
    ensemble_grid_df: pd.DataFrame,
    *,
    baseline_spend: pd.Series,                 # index=arm_id, values=spend (s*)
    ad_spend_train_df: pd.DataFrame,           # needs: arm_id, spend, conversions
    spend_grid_by_arm: dict[str, np.ndarray],  # for bandwidth + sanity
    level_grids_for_anchor: Iterable[pd.DataFrame],  # e.g. [marg_grid_rc1_df, marg_grid_rc2_df]
    weights_df: Optional[pd.DataFrame] = None,       # local weights preferred; per-arm ok too
    response_curve_estimators: Optional[list[str]] = None,  # estimator names to use in model-anchor
    cfg: AnchorConfig = AnchorConfig(),
    # column names
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    exp_conv_col: str = "exp_conversions",
    dcol: str = "d_conv_d$",
    weight_col: str = "weight",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build exp_conversions for the ENSEMBLE grid by integrating ensemble d_conv_d$ and anchoring at baseline spend.

    Anchor per (env_id, arm_id):
      anchor_data_raw(arm) = kernel mean of observed conversions near baseline spend on TRAIN
      anchor_model(env, arm) = weighted average of exp_conversions from parametric grids at baseline spend
      anchor_data_env = anchor_data_raw * (anchor_model(env,arm) / anchor_model(env_ref,arm))   [env scaling]
      gate = 1 - exp(-n_eff / n0_anchor)
      anchor = gate*anchor_data_env + (1-gate)*anchor_model(env,arm)

    Returns:
      (ensemble_grid_with_levels_df, anchors_df)
    """
    df = ensemble_grid_df.copy()

    # basic checks
    need = {env_id_col, arm_col, spend_col, dcol}
    missing = sorted(need - set(df.columns))
    if missing:
        raise KeyError(f"ensemble_grid_df missing required columns: {missing}")

    if not isinstance(baseline_spend, pd.Series):
        raise TypeError("baseline_spend must be a pandas Series indexed by arm_id.")

    if baseline_spend.index.name != arm_col:
        # not fatal, but we enforce for clarity
        baseline_spend = baseline_spend.copy()
        baseline_spend.index.name = arm_col

    # ---- Build model anchor table from provided level grids (rc estimators) ----
    all_level = pd.concat(list(level_grids_for_anchor), ignore_index=True).copy()
    need_level = {estimator_col, env_id_col, arm_col, spend_col, exp_conv_col}
    missing_level = sorted(need_level - set(all_level.columns))
    if missing_level:
        raise KeyError(f"level_grids_for_anchor missing required columns: {missing_level}")

    if response_curve_estimators is not None:
        all_level = all_level[all_level[estimator_col].isin(response_curve_estimators)].copy()

    if all_level.empty:
        raise ValueError("level_grids_for_anchor filtered to empty. Provide rc grids (with exp_conversions).")

    # We will interpolate exp_conversions and (optionally) weights at the baseline spend per arm.
    # Support both local weights (with spend) and per-arm weights (no spend).
    has_local_weights = weights_df is not None and spend_col in weights_df.columns

    # Pre-split weights for speed
    if weights_df is not None:
        wdf = weights_df.copy()
        w_need = {estimator_col, env_id_col, arm_col, weight_col}
        w_missing = sorted(w_need - set(wdf.columns))
        if w_missing:
            raise KeyError(f"weights_df missing required columns: {w_missing}")
    else:
        wdf = None

    # ---- Empirical anchor raw per arm (treated as env_ref) ----
    train_need = {arm_col, spend_col, conv_col}
    train_missing = sorted(train_need - set(ad_spend_train_df.columns))
    if train_missing:
        raise KeyError(f"ad_spend_train_df missing required columns: {train_missing}")

    anchor_rows = []

    # build quick lookups
    level_groups = {(e, env, arm): g for (e, env, arm), g in all_level.groupby([estimator_col, env_id_col, arm_col], sort=False)}
    if wdf is not None:
        if has_local_weights:
            w_groups = {(e, env, arm): g for (e, env, arm), g in wdf.groupby([estimator_col, env_id_col, arm_col], sort=False)}
        else:
            w_groups = {(e, env, arm): g for (e, env, arm), g in wdf.groupby([estimator_col, env_id_col, arm_col], sort=False)}
    else:
        w_groups = {}

    # collect env ids present
    envs = sorted(df[env_id_col].unique().tolist())

    for arm_id, g_arm in ad_spend_train_df.groupby(arm_col, sort=True):
        arm_id = str(arm_id)
        if arm_id not in spend_grid_by_arm:
            raise KeyError(f"spend_grid_by_arm missing arm_id={arm_id}")
        if arm_id not in baseline_spend.index:
            raise KeyError(f"baseline_spend missing arm_id={arm_id}")

        s_star = float(baseline_spend.loc[arm_id])

        # Empirical kernel anchor (raw, env-agnostic)
        x = g_arm[spend_col].to_numpy(dtype=float)
        y = g_arm[conv_col].to_numpy(dtype=float)

        grid = np.asarray(spend_grid_by_arm[arm_id], dtype=float)
        bw = _default_bandwidth_from_grid(grid, bandwidth_mult=cfg.kernel_bandwidth_mult, eps=cfg.eps)

        u = (x - s_star) / bw
        w = _gaussian_kernel(u)

        sw = float(np.sum(w))
        if sw <= cfg.eps:
            y_data_raw = float(np.mean(y)) if y.size else 0.0
        else:
            y_data_raw = float(np.sum(w * y) / sw)

        n_eff = _effective_n(w, eps=cfg.eps)

        # gating weight for empirical-vs-model anchor
        gate = 1.0 - np.exp(-n_eff / float(cfg.n0_anchor))

        # Model anchors per env
        model_anchor_by_env = {}
        for env_id in envs:
            # build model anchor as weighted avg over estimators at s_star
            preds = []
            wts = []
            for est in sorted(all_level[estimator_col].unique().tolist()):
                key = (est, env_id, arm_id)
                if key not in level_groups:
                    continue
                gg = level_groups[key].sort_values(spend_col)
                xg = gg[spend_col].to_numpy(dtype=float)
                yg = gg[exp_conv_col].to_numpy(dtype=float)
                pred = _interp1(xg, yg, s_star)

                if wdf is None:
                    wt = 1.0
                else:
                    wkey = (est, env_id, arm_id)
                    if wkey not in w_groups:
                        continue
                    wg = w_groups[wkey]
                    if has_local_weights:
                        wg = wg.sort_values(spend_col)
                        wx = wg[spend_col].to_numpy(dtype=float)
                        wy = wg[weight_col].to_numpy(dtype=float)
                        wt = _interp1(wx, wy, s_star)
                    else:
                        # constant per arm/env/est
                        wt = float(wg[weight_col].iloc[0])

                if np.isfinite(pred) and np.isfinite(wt) and wt >= 0.0:
                    preds.append(pred)
                    wts.append(wt)

            if len(preds) == 0:
                model_anchor_by_env[env_id] = float("nan")
            else:
                ww = np.asarray(wts, dtype=float)
                denom = float(np.sum(ww))
                if denom <= cfg.eps:
                    model_anchor_by_env[env_id] = float(np.mean(preds))
                else:
                    model_anchor_by_env[env_id] = float(np.sum(ww * np.asarray(preds, dtype=float)) / denom)

        # env scaling for empirical anchor
        ref = cfg.env_ref_id
        model_ref = model_anchor_by_env.get(ref, float("nan"))

        for env_id in envs:
            model_env = model_anchor_by_env.get(env_id, float("nan"))

            if np.isfinite(model_env) and np.isfinite(model_ref) and abs(model_ref) > cfg.eps:
                scale = model_env / model_ref
            else:
                scale = 1.0

            y_data_env = y_data_raw * scale

            # final blended anchor (fallback to model if data weak)
            if np.isfinite(model_env):
                anchor = gate * y_data_env + (1.0 - gate) * model_env
            else:
                anchor = y_data_env  # last resort

            anchor_rows.append(
                {
                    env_id_col: env_id,
                    arm_col: arm_id,
                    "spend_anchor": s_star,
                    "anchor_data_raw": y_data_raw,
                    "anchor_data_env": y_data_env,
                    "n_eff_anchor": n_eff,
                    "gate_anchor": gate,
                    "anchor_model": model_env,
                    "anchor_final": anchor,
                    "bandwidth": bw,
                    "env_ref_id": ref,
                }
            )

    anchors_df = pd.DataFrame(anchor_rows)

    # ---- Integrate ensemble slope and shift to match anchor_final at s_star ----
    out_frames = []
    for (env_id, arm_id), g in df.sort_values([env_id_col, arm_col, spend_col]).groupby([env_id_col, arm_col], sort=True):
        arm_id = str(arm_id)
        s_star = float(baseline_spend.loc[arm_id])

        gg = g.sort_values(spend_col).copy()
        xg = gg[spend_col].to_numpy(dtype=float)
        dlam = gg[dcol].to_numpy(dtype=float)

        if xg.size == 0:
            continue

        if s_star < float(np.min(xg)) - 1e-9 or s_star > float(np.max(xg)) + 1e-9:
            raise ValueError(
                f"Baseline spend s*={s_star} for arm={arm_id} is outside spend grid range "
                f"[{float(np.min(xg))}, {float(np.max(xg))}]"
            )

        # integrate slope to get unanchored levels (up to an additive constant)
        lam0 = _cumulative_trapz(dlam, xg)

        # anchor lookup
        arow = anchors_df[(anchors_df[env_id_col] == env_id) & (anchors_df[arm_col] == arm_id)]
        if arow.empty:
            raise KeyError(f"Missing anchor row for env_id={env_id}, arm_id={arm_id}")
        anchor = float(arow["anchor_final"].iloc[0])

        # compute unanchored level at s* (interpolate)
        lam_at_s = _interp1(xg, lam0, s_star)

        # shift
        offset = anchor - lam_at_s
        lam = lam0 + offset
        lam = np.maximum(lam, 0.0)  # expected conversions cannot be negative

        gg[exp_conv_col] = lam
        out_frames.append(gg)

    out_df = pd.concat(out_frames, ignore_index=True).sort_values([env_id_col, arm_col, spend_col]).reset_index(drop=True)
    return out_df, anchors_df

# Marginal profit grid
def build_marginal_profit_grid(
    *,
    conv_grid_df: pd.DataFrame,
    ds_val_df: pd.DataFrame,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    d_conv_col: str = "d_conv_d$",
    exp_conv_col: str = "exp_conversions",
    p_fund_col: str = "p_fund",
    margin_col: str = "margin",
    keep_ds_cols: bool = False,
) -> pd.DataFrame:
    """
    Convert a marginal *conversion* grid into a marginal *profit* grid by merging
    downstream value and applying:

      value_per_conv = p_fund * margin

      d_profit_gross_per$ = value_per_conv * d_conv_d$
      d_profit_net_per$   = d_profit_gross_per$ - 1

      exp_profit_gross = value_per_conv * exp_conversions
      exp_profit_net   = exp_profit_gross - spend

    Expects conv_grid_df to already contain the spend grid (including any min/max
    bounds used when constructing it).
    """
    # --- Validate inputs ---
    need_conv = {arm_col, spend_col, d_conv_col, exp_conv_col}
    missing_conv = sorted(need_conv - set(conv_grid_df.columns))
    if missing_conv:
        raise KeyError(f"conv_grid_df missing required columns: {missing_conv}")

    need_ds = {arm_col, p_fund_col, margin_col}
    missing_ds = sorted(need_ds - set(ds_val_df.columns))
    if missing_ds:
        raise KeyError(f"ds_val_df missing required columns: {missing_ds}")

    ds = ds_val_df[[arm_col, p_fund_col, margin_col]].copy()
    if ds.duplicated(subset=[arm_col]).any():
        dupes = ds.loc[ds.duplicated(subset=[arm_col], keep=False), arm_col].tolist()
        raise ValueError(f"ds_val_df has duplicate {arm_col} rows; duplicates={dupes}")

    if ds[[p_fund_col, margin_col]].isna().any().any():
        bad = ds.loc[ds[[p_fund_col, margin_col]].isna().any(axis=1), arm_col].tolist()
        raise ValueError(f"ds_val_df has missing {p_fund_col}/{margin_col} for arms: {bad}")

    # --- Compute value per conversion ---
    ds[p_fund_col] = ds[p_fund_col].astype(float)
    ds[margin_col] = ds[margin_col].astype(float)
    ds["value_per_conv"] = ds[p_fund_col] * ds[margin_col]

    # --- Merge and compute profit-grid fields ---
    out = conv_grid_df.merge(
        ds[[arm_col, "value_per_conv"] + ([p_fund_col, margin_col] if keep_ds_cols else [])],
        on=arm_col,
        how="left",
        validate="many_to_one",
    ).copy()

    if out["value_per_conv"].isna().any():
        bad = out.loc[out["value_per_conv"].isna(), arm_col].dropna().unique().tolist()
        raise ValueError(f"Missing downstream value for arms after merge: {bad}")

    out[d_conv_col] = out[d_conv_col].astype(float)
    out[exp_conv_col] = out[exp_conv_col].astype(float)
    out[spend_col] = out[spend_col].astype(float)

    out["d_profit_gross_d$"] = out[d_conv_col] * out["value_per_conv"]
    out["exp_profit_gross"] = out[exp_conv_col] * out["value_per_conv"]
    
    out["d_profit_net_d$"] = out["d_profit_gross_d$"] - 1.0
    out["exp_profit_net"] = out["exp_profit_gross"] - out[spend_col]

    return out

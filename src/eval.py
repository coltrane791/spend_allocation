# src/eval.py

"""
Evaluates model/estimator performance, in two phases.  The first constututes 'standard' 
evaluation of a predictive model.  The second is explitly focused on generating confidence 
measures to be used for ensemble constuction.
"""
# Import libraries and modules
from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.estimators.estimator_rc1 import ResponseParams, expected_conversions
from src.estimators.estimator_rc2 import ResponseParamsRC2, expected_conversions_rc2

# Standard evaluation
## rc1 parameters (not used)
def comp_params(pf_df: pd.DataFrame, pt_df: pd.DataFrame) -> None:
    ### Validation: fitted vs true params (synthetic only)
    compare = pf_df.merge(
        pt_df[["arm_id", "alpha", "beta", "gamma", "delta"]],
        on="arm_id",
        how="left",
        validate="one_to_one",
    )

    compare["alpha_err"] = compare["alpha_hat"] - compare["alpha"]
    compare["beta_err"]  = compare["beta_hat"]  - compare["beta"]
    compare["gamma_err"] = compare["gamma_hat"] - compare["gamma"]
    compare["delta_err"] = compare["delta_hat"] - compare["delta"]

    print("\nFitted vs True (summary):")
    print(compare[["alpha_err", "beta_err", "gamma_err", "delta_err"]].describe().to_string())

    ### Worst parameter recovery (synthetic truth check)
    compare["abs_beta_err"]  = compare["beta_err"].abs()
    compare["abs_alpha_err"] = compare["alpha_err"].abs()
    compare["abs_gamma_err"] = compare["gamma_err"].abs()
    compare["abs_delta_err"] = compare["delta_err"].abs()

    print("\nWorst arms by |beta_err|:")
    print(compare.sort_values("abs_beta_err", ascending=False)[
        ["arm_id", "beta_hat", "beta", "beta_err", "rmse", "mae"]
    ].to_string(index=False))

## Response curves
def arm_pred_perform_rc1(p_df: pd.DataFrame, as_df: pd.DataFrame, m_df: pd.DataFrame) -> pd.DataFrame:
    ### Build a fitted-params lookup for prediction
    params_lookup = (
        p_df.set_index("arm_id")[["alpha_hat", "beta_hat", "gamma_hat", "delta_hat"]]
        .to_dict("index")
    )

    ### Join macro onto the panel
    panel = as_df.merge(
        m_df[["date", "seasonality", "macro_index"]],
        on="date",
        how="left",
        validate="many_to_one",
    ).copy()

    ### Compute expected conversions and residuals per row
    def predict_row(r):
        p = ResponseParams(
            alpha=float(params_lookup[r["arm_id"]]["alpha_hat"]),
            beta=float(params_lookup[r["arm_id"]]["beta_hat"]),
            gamma=float(params_lookup[r["arm_id"]]["gamma_hat"]),
            delta=float(params_lookup[r["arm_id"]]["delta_hat"]),
        )
        lam = expected_conversions(
            spend=float(r["spend"]),
            seasonality=float(r["seasonality"]),
            macro=float(r["macro_index"]),
            p=p,
        )
        return float(np.atleast_1d(lam)[0])

    panel["pred_conversions"] = panel.apply(predict_row, axis=1)
    panel["resid"] = panel["conversions"] - panel["pred_conversions"]

    ### Performance measures per arm
    arm_diag = (
    panel.groupby("arm_id", sort=True)
    .agg(
        n_obs=("conversions", "size"),
        mean_conv=("conversions", "mean"),
        mean_pred=("pred_conversions", "mean"),
        rmse=("resid", lambda x: float(np.sqrt(np.mean(x**2)))),
        mae=("resid", lambda x: float(np.mean(np.abs(x)))),
    )
    .reset_index()
    .sort_values("rmse", ascending=False)
    .reset_index(drop=True)
    )

    return arm_diag

def arm_pred_perform_rc2(
    p_df: pd.DataFrame,
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
    Evaluate RC2 predictions vs actual conversions (out-of-sample panel).
    Returns same schema as arm_pred_perform_rc1.
    """
    params_lookup = p_df.set_index(arm_col)[["alpha_hat", "beta_hat", "gamma_hat", "delta_hat"]].to_dict("index")

    panel = ad_spend_df.merge(
        macro_df[[date_col, seasonality_col, macro_col]],
        on=date_col,
        how="left",
        validate="many_to_one",
    ).copy()

    def predict_row(r):
        p = ResponseParamsRC2(
            alpha=float(params_lookup[r[arm_col]]["alpha_hat"]),
            beta=float(params_lookup[r[arm_col]]["beta_hat"]),
            gamma=float(params_lookup[r[arm_col]]["gamma_hat"]),
            delta=float(params_lookup[r[arm_col]]["delta_hat"]),
        )
        lam = expected_conversions_rc2(
            spend=float(r[spend_col]),
            S=float(r[seasonality_col]),
            M=float(r[macro_col]),
            p=p,
        )
        return float(np.atleast_1d(lam)[0])

    panel["pred_conversions"] = panel.apply(predict_row, axis=1)
    panel["resid"] = panel[conv_col] - panel["pred_conversions"]

    arm_diag = (
        panel.groupby(arm_col)
        .agg(
            n_obs=(conv_col, "size"),
            mean_conv=(conv_col, "mean"),
            mean_pred=("pred_conversions", "mean"),
            rmse=("resid", lambda x: float(np.sqrt(np.mean(x**2)))),
            mae=("resid", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    return arm_diag

## Data driven local slope
def eval_curve_grid_on_panel(
    curve_grid_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    grid_spend_col: str = "spend",
    grid_pred_col: str = "exp_conversions",
) -> pd.DataFrame:
    """
    Evaluate a curve-grid predictor on a panel by interpolation.

    Intended primarily for estimators like RC2 (local slope) that do not model
    date-varying S/M at the row level. For each arm:
      pred(conv | spend) = interp(spend, grid_spend, grid_exp_conversions)

    Returns per-arm summary: mean_conv, mean_pred, rmse, mae, n_obs.
    """
    
    need_grid = {arm_col, grid_spend_col, grid_pred_col}
    need_panel = {arm_col, spend_col, conv_col}
    missing_g = sorted(need_grid - set(curve_grid_df.columns))
    missing_p = sorted(need_panel - set(panel_df.columns))
    if missing_g:
        raise KeyError(f"curve_grid_df missing required columns: {missing_g}")
    if missing_p:
        raise KeyError(f"panel_df missing required columns: {missing_p}")

    rows = []
    for arm_id, g_panel in panel_df.groupby(arm_col, sort=True):
        g_grid = curve_grid_df.loc[curve_grid_df[arm_col] == arm_id].copy()
        if g_grid.empty:
            raise ValueError(f"No grid rows found for arm_id={arm_id}")

        xs = g_grid[grid_spend_col].to_numpy(dtype=float)
        ys = g_grid[grid_pred_col].to_numpy(dtype=float)

        # Sort for interpolation safety
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        spend = g_panel[spend_col].to_numpy(dtype=float)
        y_true = g_panel[conv_col].to_numpy(dtype=float)

        # Clamp to grid support then interpolate
        spend_clip = np.clip(spend, xs[0], xs[-1])
        y_pred = np.interp(spend_clip, xs, ys)

        resid = y_true - y_pred
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae = float(np.mean(np.abs(resid)))

        rows.append(
            {
                arm_col: arm_id,
                "n_obs": float(len(g_panel)),
                "mean_conv": float(np.mean(y_true)),
                "mean_pred": float(np.mean(y_pred)),
                "rmse": rmse,
                "mae": mae,
            }
        )

    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)

# Confidence measures
## Per-arm
### Confidence fit from RMSE
def conf_fit_from_rmse(
    perf_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    rmse_col: str = "rmse",
    mean_col: str = "mean_conv",
    estimator: str | None = None,
    env_id: str | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    First-pass confidence score derived from out-of-sample RMSE.

    conf_fit in (0, 1], higher is better.
    Uses normalized error: rmse / (mean_conv + eps)
      conf_fit = 1 / (1 + rmse_norm)
    """
    df = perf_df[[arm_col, rmse_col, mean_col]].copy()
    rmse_norm = df[rmse_col].to_numpy(dtype=float) / (df[mean_col].to_numpy(dtype=float) + eps)
    df["rmse_norm"] = rmse_norm
    df["conf_fit"] = 1.0 / (1.0 + rmse_norm)

    if estimator is not None:
        df["estimator"] = str(estimator)
    if env_id is not None:
        df["env_id"] = str(env_id)

    # put identifiers first
    cols = [c for c in ["estimator", "env_id", arm_col, rmse_col, mean_col, "rmse_norm", "conf_fit"] if c in df.columns]
    return df[cols]

### Confidence data from training data
def conf_data_from_train(
    ad_spend_train_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    n0: float = 60.0,     # half-saturation scale
) -> pd.DataFrame:
    """
    Per-arm data support confidence in [0,1].
    First pass: saturating function of training observation count per arm.
    """
    g = ad_spend_train_df.groupby(arm_col, sort=True)
    out = g.agg(
        n_obs_train=(spend_col, "size"),
        spend_min_train=(spend_col, "min"),
        spend_max_train=(spend_col, "max"),
        n_unique_spend_train=(spend_col, lambda x: int(pd.Series(x).nunique())),
    ).reset_index()

    out["conf_data"] = 1.0 - np.exp(-out["n_obs_train"].astype(float) / float(n0))
    return out[[arm_col, "n_obs_train", "n_unique_spend_train", "spend_min_train", "spend_max_train", "conf_data"]]

### Confidence stability
def make_expanding_time_folds(
    *,
    ad_spend_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_col: str = "date",
    test_size_days: int = 30,
    n_folds: int = 4,
    step_days: int = 30,
    gap_days: int = 0,
    min_train_days: int = 60,
) -> list[dict]:
    """
    Builds expanding-window time folds ending near the end of the sample.
    Each fold:
      train = up to (test_start - gap)
      test  = [test_start, test_end]
    """
    # Normalize/ensure datetime
    ad = ad_spend_df.copy()
    ma = macro_df.copy()
    ad[date_col] = pd.to_datetime(ad[date_col])
    ma[date_col] = pd.to_datetime(ma[date_col])

    all_days = pd.Index(ad[date_col].unique()).sort_values()
    if len(all_days) < (min_train_days + test_size_days):
        raise ValueError("Not enough days for requested folds/min_train_days/test_size_days.")

    folds: list[dict] = []
    last_day = all_days[-1]

    for k in range(n_folds):
        # Walk backwards in time by step_days per fold
        test_end = last_day - pd.Timedelta(days=k * step_days)
        test_start = test_end - pd.Timedelta(days=test_size_days - 1)

        # Train ends before test starts (optionally with a gap)
        train_end = test_start - pd.Timedelta(days=gap_days + 1)

        # Enforce minimum train length
        train_days = all_days[all_days <= train_end]
        if len(train_days) < min_train_days:
            # Stop creating additional folds once we can't satisfy min train size
            break

        fold_id = f"fold_{k+1:02d}"

        ad_train = ad[ad[date_col] <= train_end].copy()
        ad_test  = ad[(ad[date_col] >= test_start) & (ad[date_col] <= test_end)].copy()

        ma_train = ma[ma[date_col] <= train_end].copy()
        ma_test  = ma[(ma[date_col] >= test_start) & (ma[date_col] <= test_end)].copy()

        folds.append(
            {
                "fold_id": fold_id,
                "ad_spend_train": ad_train,
                "ad_spend_test": ad_test,
                "macro_train": ma_train,
                "macro_test": ma_test,
            }
        )

    if not folds:
        raise ValueError("No folds created; relax n_folds/step_days/min_train_days.")
    return folds

def build_perf_folds(
    *,
    folds: list[dict],
    estimators: dict,
    # Shared grid (optional but recommended for joinability)
    spend_grid_by_arm: dict[str, np.ndarray],
    # Bounds + identifiers
    min_spend: dict[str, float],
    max_spend: dict[str, float],
    arm_col: str = "arm_id",
) -> pd.DataFrame:
    """
    Returns perf_folds_df with rows per (estimator, fold_id, arm_id).

    `estimators` is a dict keyed by estimator name, each value a dict with:
      - name
      - kind: "panel_model" or "curve_grid"
      - fit:  callable
      - eval: callable
      - fit_kwargs: optional dict of extra kwargs to pass to fit

    Expected behaviors:
      kind="panel_model":
        params = fit(ad_train, macro_train, arm_col=...)
        perf   = eval(params, ad_test, macro_test)

      kind="curve_grid":
        grid   = fit(estimator=..., env_id=..., ad_spend_df=ad_train, min_spend=..., max_spend=...,
                     S=..., M=..., arm_col=..., n_points=..., spend_grid_by_arm=..., **fit_kwargs)
        perf   = eval(curve_grid_df=grid, panel_df=ad_test)
    """
    out: list[pd.DataFrame] = []

    # We assume spend_grid_by_arm is joinable: all arms share same point count
    n_points = int(len(next(iter(spend_grid_by_arm.values()))))

    for f in folds:
        fold_id = f["fold_id"]
        ad_tr = f["ad_spend_train"]
        ad_te = f["ad_spend_test"]
        ma_tr = f["macro_train"]
        ma_te = f["macro_test"]

        # Choose an environment for any curve-grid estimators per fold (simple, stable)
        S_env = float(ma_te["seasonality"].mean())
        M_env = float(ma_te["macro_index"].mean())

        for est_id, spec in estimators.items():
            name = spec.get("name", None)
            kind = spec.get("kind", None)
            fit_fn = spec.get("fit", None)
            eval_fn = spec.get("eval", None)
            fit_kwargs = spec.get("fit_kwargs", {}) or {}

            if kind is None or fit_fn is None or eval_fn is None:
                raise ValueError(
                    f"Estimator spec for '{name}' must include kind, fit, eval. Got: {spec}"
                )

            if kind == "panel_data":
                params = fit_fn(ad_tr, ma_tr, arm_col=arm_col)
                perf = eval_fn(params, ad_te, ma_te).copy()

                # Normalize to common schema
                perf = perf.rename(columns={"rmse_panel": "rmse", "mae_panel": "mae"})

            elif kind == "curve_grid":
                grid = fit_fn(
                    estimator=name,
                    env_id=f"{fold_id}_test_mean",
                    ad_spend_df=ad_tr,
                    min_spend=min_spend,
                    max_spend=max_spend,
                    S=S_env,
                    M=M_env,
                    arm_col=arm_col,
                    n_points=n_points,
                    spend_grid_by_arm=spend_grid_by_arm,
                    **fit_kwargs,
                )
                perf = eval_fn(curve_grid_df=grid, panel_df=ad_te).copy()

            else:
                raise ValueError(f"Unknown estimator kind='{kind}' for '{name}'")

            # Enforce required columns, then add identifiers
            required = [arm_col, "n_obs", "mean_conv", "mean_pred", "rmse", "mae"]
            missing = [c for c in required if c not in perf.columns]
            if missing:
                raise KeyError(f"perf for '{name}' fold '{fold_id}' missing columns: {missing}")

            perf.insert(0, "fold_id", fold_id)
            perf.insert(0, "estimator", name)

            out.append(perf[["estimator", "fold_id", arm_col, "n_obs", "mean_conv", "mean_pred", "rmse", "mae"]])

    return pd.concat(out, ignore_index=True)

def conf_stability_from_perf_folds(
    perf_folds_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Per (estimator, arm): stability based on variability of RMSE across folds.

      rmse_cv = std(rmse) / (mean(rmse)+eps)
      conf_stability = 1 / (1 + rmse_cv)

    Returns: estimator, arm_id, rmse_mean, rmse_std, rmse_cv, conf_stability
    """
    g = (
        perf_folds_df.groupby(["estimator", arm_col], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
    )
    g["rmse_std"] = g["rmse_std"].fillna(0.0)
    g["rmse_cv"] = g["rmse_std"] / (g["rmse_mean"] + eps)
    g["conf_stability"] = 1.0 / (1.0 + g["rmse_cv"])
    return g

## Spend level
### Helpers
def _kernel(u: np.ndarray, kernel: str = "gaussian") -> np.ndarray:
    """
    u is scaled distance: (x - x0) / bandwidth
    returns nonnegative weights
    """
    k = (kernel or "gaussian").lower()
    if k == "gaussian":
        return np.exp(-0.5 * u * u)
    if k == "epanechnikov":
        # support |u|<=1
        return np.maximum(0.0, 1.0 - u * u)
    if k == "triangular":
        return np.maximum(0.0, 1.0 - np.abs(u))
    raise ValueError(f"Unknown kernel='{kernel}'. Use gaussian|epanechnikov|triangular.")

def _effective_n(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Kish effective sample size per column:
      n_eff = (sum w)^2 / sum(w^2)
    w: shape (n_obs, n_grid)
    returns: shape (n_grid,)
    """
    sw = np.sum(w, axis=0)
    sw2 = np.sum(w * w, axis=0)
    return (sw * sw) / (sw2 + eps)

def _default_bandwidth_from_grid(grid: np.ndarray, bandwidth_mult: float = 2.5, eps: float = 1e-9) -> float:
    """
    Default bandwidth based on grid spacing (robust).
    """
    g = np.asarray(grid, dtype=float)
    if g.size < 2:
        return 1.0
    diffs = np.diff(np.sort(g))
    step = float(np.median(diffs)) if diffs.size else 1.0
    step = max(step, eps)
    return float(bandwidth_mult * step)

### Confidence fit from RMSE
def panel_preds_from_curve_grid(
    curve_grid_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    grid_spend_col: str = "spend",
    grid_pred_col: str = "exp_conversions",
) -> pd.DataFrame:
    """
    Row-level predictions + residuals for any curve-grid estimator (dd1/rc2/etc) via interpolation.
    Returns columns: arm_id, spend, conversions, pred_conversions, resid
    """
    need_grid = {arm_col, grid_spend_col, grid_pred_col}
    need_panel = {arm_col, spend_col, conv_col}
    missing_g = sorted(need_grid - set(curve_grid_df.columns))
    missing_p = sorted(need_panel - set(panel_df.columns))
    if missing_g:
        raise KeyError(f"curve_grid_df missing required columns: {missing_g}")
    if missing_p:
        raise KeyError(f"panel_df missing required columns: {missing_p}")

    out_parts: list[pd.DataFrame] = []

    for arm_id, g_panel in panel_df.groupby(arm_col, sort=True):
        g_grid = curve_grid_df.loc[curve_grid_df[arm_col] == arm_id].copy()
        if g_grid.empty:
            raise ValueError(f"No grid rows found for arm_id={arm_id}")

        xs = g_grid[grid_spend_col].to_numpy(dtype=float)
        ys = g_grid[grid_pred_col].to_numpy(dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        spend = g_panel[spend_col].to_numpy(dtype=float)
        y_true = g_panel[conv_col].to_numpy(dtype=float)

        spend_clip = np.clip(spend, xs[0], xs[-1])
        y_pred = np.interp(spend_clip, xs, ys)

        part = g_panel[[arm_col, spend_col, conv_col]].copy()
        part["pred_conversions"] = y_pred
        part["resid"] = y_true - y_pred
        out_parts.append(part)

    out = pd.concat(out_parts, ignore_index=True)
    return out[[arm_col, spend_col, conv_col, "pred_conversions", "resid"]].copy()

#### Function below not used.  Estimates conversion prediction error on basis of model parameters
#### rather than grid interpolation.  Only use if marginal grid not available, which will never be.
def panel_preds_rc1(
    params_df: pd.DataFrame,
    ad_spend_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    date_col: str = "date",
    spend_col: str = "spend",
    conv_col: str = "conversions",
) -> pd.DataFrame:
    """
    Row-level predictions + residuals for RC1 (panel_data estimator).
    Returns columns: arm_id, date, spend, conversions, pred_conversions, resid
    """
    params_lookup = (
        params_df.set_index(arm_col)[["alpha_hat", "beta_hat", "gamma_hat", "delta_hat"]]
        .to_dict("index")
    )

    panel = ad_spend_df.merge(
        macro_df[[date_col, "seasonality", "macro_index"]],
        on=date_col,
        how="left",
        validate="many_to_one",
    ).copy()

    def _predict_row(r):
        p = ResponseParams(
            alpha=float(params_lookup[r[arm_col]]["alpha_hat"]),
            beta=float(params_lookup[r[arm_col]]["beta_hat"]),
            gamma=float(params_lookup[r[arm_col]]["gamma_hat"]),
            delta=float(params_lookup[r[arm_col]]["delta_hat"]),
        )
        lam = expected_conversions(
            spend=float(r[spend_col]),
            seasonality=float(r["seasonality"]),
            macro=float(r["macro_index"]),
            p=p,
        )
        return float(np.atleast_1d(lam)[0])

    panel["pred_conversions"] = panel.apply(_predict_row, axis=1)
    panel["resid"] = panel[conv_col].astype(float) - panel["pred_conversions"].astype(float)

    return panel[[arm_col, date_col, spend_col, conv_col, "pred_conversions", "resid"]].copy()

def conf_fit_local_from_panel_preds(
    panel_preds_df: pd.DataFrame,
    *,
    spend_grid_by_arm: dict[str, np.ndarray],
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    resid_col: str = "resid",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    bandwidth_mult: float = 2.5,
    estimator: str | None = None,
    env_id: str | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Spend-local conf_fit derived from kernel-weighted RMSE around each grid spend point.

    Returns one row per (arm_id, spend) with:
      rmse_local, mean_conv_local, rmse_norm_local, conf_fit_local, plus n_eff/sum_w.
    """
    need = {arm_col, spend_col, conv_col, resid_col}
    missing = sorted(need - set(panel_preds_df.columns))
    if missing:
        raise KeyError(f"panel_preds_df missing required columns: {missing}")

    rows: list[pd.DataFrame] = []

    for arm_id, g in panel_preds_df.groupby(arm_col, sort=True):
        if arm_id not in spend_grid_by_arm:
            raise KeyError(f"spend_grid_by_arm missing arm_id={arm_id}")

        x = g[spend_col].to_numpy(dtype=float)
        y = g[conv_col].to_numpy(dtype=float)
        r = g[resid_col].to_numpy(dtype=float)
        r2 = r * r

        grid = np.asarray(spend_grid_by_arm[arm_id], dtype=float)

        bw = float(bandwidth) if bandwidth is not None else _default_bandwidth_from_grid(grid, bandwidth_mult=bandwidth_mult)
        bw = max(bw, eps)

        u = (x[:, None] - grid[None, :]) / bw
        w = _kernel(u, kernel=kernel)

        sw = np.sum(w, axis=0)
        sw_safe = np.maximum(sw, eps)

        n_eff = _effective_n(w)
        mean_conv_local = np.sum(w * y[:, None], axis=0) / sw_safe
        rmse_local = np.sqrt(np.sum(w * r2[:, None], axis=0) / sw_safe)

        rmse_norm_local = rmse_local / (np.maximum(mean_conv_local, 0.0) + eps)
        conf_fit_local = 1.0 / (1.0 + rmse_norm_local)

        df_arm = pd.DataFrame(
            {
                arm_col: arm_id,
                "spend": grid,
                "bandwidth": bw,
                "kernel": kernel,
                "sum_w": sw,
                "n_eff": n_eff,
                "mean_conv_local": mean_conv_local,
                "rmse_local": rmse_local,
                "rmse_norm_local": rmse_norm_local,
                "conf_fit_local": conf_fit_local,
            }
        )
        rows.append(df_arm)

    out = pd.concat(rows, ignore_index=True)

    if estimator is not None:
        out.insert(0, "estimator", str(estimator))
    if env_id is not None:
        out.insert(0, "env_id", str(env_id))

    sort_cols = [c for c in ["env_id", "estimator", arm_col, "spend"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out

### Confidence data
#### Function below not used.  Does not differentiate values by estimator - i.e., based just on
#### how much data is available, not how data is used by estimator.
def conf_data_local_from_train(
    ad_spend_train_df: pd.DataFrame,
    *,
    spend_grid_by_arm: dict[str, np.ndarray],
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    bandwidth_mult: float = 2.5,
    n0_local: float = 10.0,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Spend-local data support confidence in [0,1], one row per (arm_id, spend).

    Uses kernel-weighted effective sample size around each grid spend point:
      n_eff(s0) = (sum_t w_t)^2 / sum_t w_t^2
      conf_data_local(s0) = 1 - exp(- n_eff(s0) / n0_local)
    """
    need = {arm_col, spend_col}
    missing = sorted(need - set(ad_spend_train_df.columns))
    if missing:
        raise KeyError(f"ad_spend_train_df missing required columns: {missing}")

    rows: list[pd.DataFrame] = []

    for arm_id, g in ad_spend_train_df.groupby(arm_col, sort=True):
        if arm_id not in spend_grid_by_arm:
            raise KeyError(f"spend_grid_by_arm missing arm_id={arm_id}")

        x = g[spend_col].to_numpy(dtype=float)
        grid = np.asarray(spend_grid_by_arm[arm_id], dtype=float)

        bw = float(bandwidth) if bandwidth is not None else _default_bandwidth_from_grid(grid, bandwidth_mult=bandwidth_mult)
        bw = max(bw, eps)

        # weights matrix: (n_obs, n_grid)
        u = (x[:, None] - grid[None, :]) / bw
        w = _kernel(u, kernel=kernel)

        sw = np.sum(w, axis=0)
        n_eff = _effective_n(w)

        conf = 1.0 - np.exp(-n_eff / float(n0_local))

        df_arm = pd.DataFrame(
            {
                arm_col: arm_id,
                "spend": grid,
                "bandwidth": bw,
                "kernel": kernel,
                "sum_w": sw,
                "n_eff": n_eff,
                "conf_data_local": conf,
            }
        )
        rows.append(df_arm)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values([arm_col, "spend"]).reset_index(drop=True)

def conf_data_local_by_estimator_from_train(
    ad_spend_train_df: pd.DataFrame,
    *,
    spend_grid_by_arm: dict[str, np.ndarray],
    estimators: dict,  # your estimator registry dict
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    bandwidth_mult: float = 2.5,
    bandwidth_mult_by_estimator: Optional[Dict[str, float]] = None,
    n0_local: float = 10.0,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Estimator-specific spend-local data support confidence in [0,1].

    Output: one row per (estimator, arm_id, spend).

    Differentiation comes from estimator-specific bandwidth multipliers:
      bw_est = bw_base * bandwidth_mult_by_estimator.get(estimator, 1.0)

    This is a principled first-pass proxy for "how local is the estimator's learning signal".
    """
    need = {arm_col, spend_col}
    missing = sorted(need - set(ad_spend_train_df.columns))
    if missing:
        raise KeyError(f"ad_spend_train_df missing required columns: {missing}")

    # Default multipliers: dd1 more local, parametric RCs less local (borrow strength)
    # You can tune later.
    default_mults = {
        "dd1": 1.0,   # most local
        "rc1": 2.0,   # less local
        "rc2": 2.0,   # less local
    }
    mults = dict(default_mults)
    if bandwidth_mult_by_estimator:
        mults.update(bandwidth_mult_by_estimator)

    # Determine which estimators are "in scope" for local data support.
    # Use the estimator keys (e.g., "dd1", "rc1", "rc2") from your registry.
    estimator_ids = list(estimators.keys())
    if not estimator_ids:
        raise ValueError("estimators is empty")

    rows: list[pd.DataFrame] = []

    for arm_id, g in ad_spend_train_df.groupby(arm_col, sort=True):
        if arm_id not in spend_grid_by_arm:
            raise KeyError(f"spend_grid_by_arm missing arm_id={arm_id}")

        x = g[spend_col].to_numpy(dtype=float)
        grid = np.asarray(spend_grid_by_arm[arm_id], dtype=float)

        # Base bandwidth derived from the grid (shared), then scaled per estimator.
        bw_base = float(bandwidth) if bandwidth is not None else _default_bandwidth_from_grid(
            grid, bandwidth_mult=bandwidth_mult
        )
        bw_base = max(bw_base, eps)

        # Precompute u and kernel weights once per *bw*? bw differs by estimator.
        # We'll compute per estimator (still vectorized and fast).

        for est_id in estimator_ids:
            bw = bw_base * float(mults.get(est_id, 1.0))
            bw = max(bw, eps)

            # weights matrix: (n_obs, n_grid)
            u = (x[:, None] - grid[None, :]) / bw
            w = _kernel(u, kernel=kernel)

            sw = np.sum(w, axis=0)
            n_eff = _effective_n(w)

            conf = 1.0 - np.exp(-n_eff / float(n0_local))

            df_arm = pd.DataFrame(
                {
                    "estimator": estimators[est_id]["name"],
                    arm_col: arm_id,
                    "spend": grid,
                    "kernel": kernel,
                    "bandwidth": bw,
                    "bandwidth_mult_est": float(mults.get(est_id, 1.0)),
                    "sum_w": sw,
                    "n_eff": n_eff,
                    "conf_data_local": conf,
                }
            )
            rows.append(df_arm)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["estimator", arm_col, "spend"]).reset_index(drop=True)

### Stability
def build_local_fit_folds(
    *,
    folds: list[dict],
    estimators: dict,
    spend_grid_by_arm: dict[str, np.ndarray],
    min_spend: dict[str, float],
    max_spend: dict[str, float],
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    conv_col: str = "conversions",
    grid_spend_col: str = "spend",
    grid_pred_col: str = "exp_conversions",
    resid_col: str = "resid",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    bandwidth_mult: float = 2.5,
    bandwidth_mult_by_estimator: Optional[dict[str, float]] = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Build a fold-by-fold *local fit* table (RMSE-local) for stability.

    Output rows: (env_id, estimator, fold_id, arm_id, spend) with:
      rmse_local, rmse_norm_local, conf_fit_local, mean_conv_local, n_eff, sum_w, bandwidth, kernel

    Notes
    -----
    - Uses grid interpolation predictions for ALL estimators (uniform downstream artifact).
    - For panel_data estimators (e.g., rc1), the estimator spec MUST include a 'grid' callable
      that converts fitted params into a curve grid.
    """
    if not folds:
        raise ValueError("folds is empty")

    if not estimators:
        raise ValueError("estimators is empty")

    if not spend_grid_by_arm:
        raise ValueError("spend_grid_by_arm is empty")

    n_points = int(len(next(iter(spend_grid_by_arm.values()))))
    mults = bandwidth_mult_by_estimator or {}

    out_parts: list[pd.DataFrame] = []

    for f in folds:
        fold_id = f["fold_id"]
        ad_tr = f["ad_spend_train"]
        ad_te = f["ad_spend_test"]
        ma_tr = f["macro_train"]
        ma_te = f["macro_test"]

        # One stable env for this fold’s grid-evaluations
        S_env = float(ma_te["seasonality"].mean())
        M_env = float(ma_te["macro_index"].mean())
        env_id = f"{fold_id}_test_mean"

        for est_key, spec in estimators.items():
            est_name = str(spec.get("name", est_key))
            kind = spec.get("kind", None)
            fit_fn = spec.get("fit", None)
            grid_fn = spec.get("grid", None)  # required for panel_data
            fit_kwargs = spec.get("fit_kwargs", {}) or {}
            grid_kwargs = spec.get("grid_kwargs", {}) or {}

            if kind is None or fit_fn is None:
                raise ValueError(f"Estimator '{est_name}' missing kind/fit. Spec={spec}")

            # Build a curve grid for this estimator on this fold
            if kind == "curve_grid":
                grid = fit_fn(
                    estimator=est_name,
                    env_id=env_id,
                    ad_spend_df=ad_tr,
                    min_spend=min_spend,
                    max_spend=max_spend,
                    S=S_env,
                    M=M_env,
                    arm_col=arm_col,
                    n_points=n_points,
                    spend_grid_by_arm=spend_grid_by_arm,
                    **fit_kwargs,
                )

            elif kind == "panel_data":
                if grid_fn is None:
                    raise ValueError(
                        f"Estimator '{est_name}' is kind='panel_data' but has no 'grid' builder. "
                        "Add spec['grid']=<callable> to convert params -> curve grid."
                    )
                params = fit_fn(ad_tr, ma_tr, arm_col=arm_col)

                grid = grid_fn(
                    estimator=est_name,
                    env_id=env_id,
                    params_df=params,
                    min_spend=min_spend,
                    max_spend=max_spend,
                    S=S_env,
                    M=M_env,
                    arm_col=arm_col,
                    n_points=n_points,
                    spend_grid_by_arm=spend_grid_by_arm,
                    **grid_kwargs,
                )

            else:
                raise ValueError(f"Unknown estimator kind='{kind}' for '{est_name}'")

            # Row-level preds on this fold’s TEST panel via interpolation
            panel_preds = panel_preds_from_curve_grid(
                curve_grid_df=grid,
                panel_df=ad_te,
                arm_col=arm_col,
                spend_col=spend_col,
                conv_col=conv_col,
                grid_spend_col=grid_spend_col,
                grid_pred_col=grid_pred_col,
            )

            # Estimator-specific locality knob (optional)
            bw_mult_est = float(mults.get(est_key, 1.0))
            bw_mult = float(bandwidth_mult) * bw_mult_est

            # Local RMSE at each spend-grid point
            local_fit = conf_fit_local_from_panel_preds(
                panel_preds_df=panel_preds,
                spend_grid_by_arm=spend_grid_by_arm,
                arm_col=arm_col,
                spend_col=spend_col,
                conv_col=conv_col,
                resid_col=resid_col,
                kernel=kernel,
                bandwidth=bandwidth,
                bandwidth_mult=bw_mult,
                estimator=est_name,
                env_id=env_id,
                eps=eps,
            )

            # Insert fold_id after env_id/estimator if present
            insert_at = 0
            if "env_id" in local_fit.columns:
                insert_at += 1
            if "estimator" in local_fit.columns:
                insert_at += 1
            local_fit.insert(insert_at, "fold_id", fold_id)

            keep = [
                c for c in [
                    "env_id", "estimator", "fold_id",
                    arm_col, "spend",
                    "rmse_local", "rmse_norm_local", "conf_fit_local",
                    "mean_conv_local", "n_eff", "sum_w",
                    "bandwidth", "kernel",
                ]
                if c in local_fit.columns
            ]
            out_parts.append(local_fit[keep])

    out = pd.concat(out_parts, ignore_index=True)

    sort_cols = [c for c in ["estimator", "fold_id", arm_col, "spend"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)

def conf_stability_local_from_fit_folds(
    local_fit_folds_df: pd.DataFrame,
    *,
    arm_col: str = "arm_id",
    spend_col: str = "spend",
    rmse_local_col: str = "rmse_local",
    fold_id_col: str = "fold_id",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Spend-local stability: variability of local RMSE across folds.

      rmse_cv_local = std(rmse_local) / (mean(rmse_local) + eps)
      conf_stability_local = 1 / (1 + rmse_cv_local)

    Output rows: (estimator, arm_id, spend) with:
      rmse_local_mean, rmse_local_std, rmse_local_cv, n_folds, conf_stability_local
    """
    need = {"estimator", arm_col, spend_col, rmse_local_col, fold_id_col}
    missing = sorted(need - set(local_fit_folds_df.columns))
    if missing:
        raise KeyError(f"local_fit_folds_df missing required columns: {missing}")

    g = (
        local_fit_folds_df
        .groupby(["estimator", arm_col, spend_col], as_index=False)
        .agg(
            rmse_local_mean=(rmse_local_col, "mean"),
            rmse_local_std=(rmse_local_col, "std"),
            n_folds=(fold_id_col, "nunique"),
        )
    )

    g["rmse_local_std"] = g["rmse_local_std"].fillna(0.0)
    g["rmse_local_cv"] = g["rmse_local_std"] / (g["rmse_local_mean"] + eps)
    g["conf_stability_local"] = 1.0 / (1.0 + g["rmse_local_cv"])

    return g.sort_values(["estimator", arm_col, spend_col]).reset_index(drop=True)

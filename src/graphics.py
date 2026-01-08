# src/graphics.py

"""
Create graphical displays of forecast errors over time and marginal curves by arm.
"""

# Import libraries and modules
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Mapping
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Forecast errors
def rolling_rmse_by_arm(
    daily_comp_arm_df: pd.DataFrame,
    *,
    err_col: str,
    window: int,
    arm_col: str = "arm_id",
    date_col: str = "date_eval",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute trailing rolling RMSE of `err_col` per arm over `window` days.

    Returns columns: [arm_id, date_eval, err_col, rmse_roll]
    """
    if min_periods is None:
        # conservative default: require full window for stable RMSE
        min_periods = window

    need = {arm_col, date_col, err_col}
    missing = sorted(need - set(daily_comp_arm_df.columns))
    if missing:
        raise KeyError(f"daily_comp_arm_df missing required columns: {missing}")

    df = daily_comp_arm_df[[arm_col, date_col, err_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[err_col] = pd.to_numeric(df[err_col], errors="coerce")

    df = df.sort_values([arm_col, date_col]).reset_index(drop=True)

    se = df[err_col].to_numpy(dtype=float) ** 2
    df["_se"] = se

    df["rmse_roll"] = (
        df.groupby(arm_col, sort=True)["_se"]
        .transform(lambda s: np.sqrt(s.rolling(window=window, min_periods=min_periods).mean()))
    )

    df = df.drop(columns=["_se"])
    return df

def plot_rolling_rmse_per_arm(
    rmse_df: pd.DataFrame,
    *,
    out_dir: Path,
    title: str,
    metric_label: str,
    arm_col: str = "arm_id",
    date_col: str = "date_eval",
    rmse_col: str = "rmse_roll",
    write_pngs: bool = True,
    png_dpi: int = 160,
    write_pdf: bool = True,
    pdf_name: str = "rolling_rmse.pdf",
) -> dict[str, Path]:
    """
    Plot rolling RMSE per arm.
    Writes:
      - multipage PDF (one page per arm) if write_pdf=True
      - per-arm PNGs if write_pngs=True

    Returns dict of written artifact paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    png_dir = out_dir / "png"
    if write_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / pdf_name
    pdf = PdfPages(pdf_path) if write_pdf else None

    try:
        for arm_id, g in rmse_df.groupby(arm_col, sort=True):
            g = g.sort_values(date_col)

            fig, ax = plt.subplots(figsize=(9.5, 4.8))
            ax.plot(g[date_col], g[rmse_col])

            ax.set_title(f"{title}\narm_id={arm_id}")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"Rolling RMSE ({metric_label})")

            # Light grid for readability
            ax.grid(True, alpha=0.25)
            fig.autofmt_xdate()

            if write_pngs:
                png_path = png_dir / f"{arm_id}.png"
                fig.savefig(png_path, dpi=png_dpi)
                written[f"png:{arm_id}"] = png_path

            if pdf is not None:
                pdf.savefig(fig)

            plt.close(fig)

    finally:
        if pdf is not None:
            pdf.close()
            written["pdf"] = pdf_path

    return written

def forecast_errors(
    daily_comp_arm_df: pd.DataFrame,
    *,
    out_dir: Path,
    window: int,
    arm_col: str = "arm_id",
    date_col: str = "date_eval",
    conv_err_col: str = "fc_error_conv",
    vpc_err_col: str = "fc_error_vpc",
    min_periods: Optional[int] = None,
    write_pngs: bool = True,
    write_pdf: bool = True,
) -> dict[str, dict[str, Path]]:
    """
    Convenience wrapper: generate BOTH sets of rolling RMSE plots:
      1) conversions error
      2) value-per-conversion error

    Returns a dict with keys {"conv", "vpc"} mapping to the written artifacts dicts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Conversions
    rmse_conv = rolling_rmse_by_arm(
        daily_comp_arm_df,
        err_col=conv_err_col,
        window=window,
        arm_col=arm_col,
        date_col=date_col,
        min_periods=min_periods,
    )
    conv_written = plot_rolling_rmse_per_arm(
        rmse_conv,
        out_dir=out_dir / f"rolling_rmse_conversions_w{window}",
        title=f"Forecast quality: Conversions (rolling RMSE, window={window})",
        metric_label="conversions",
        arm_col=arm_col,
        date_col=date_col,
        rmse_col="rmse_roll",
        write_pngs=write_pngs,
        write_pdf=write_pdf,
        pdf_name=f"rolling_rmse_conversions_w{window}.pdf",
    )

    # 2) Value-per-conversion
    rmse_vpc = rolling_rmse_by_arm(
        daily_comp_arm_df,
        err_col=vpc_err_col,
        window=window,
        arm_col=arm_col,
        date_col=date_col,
        min_periods=min_periods,
    )
    vpc_written = plot_rolling_rmse_per_arm(
        rmse_vpc,
        out_dir=out_dir / f"rolling_rmse_vpc_w{window}",
        title=f"Forecast quality: Value-per-conv (rolling RMSE, window={window})",
        metric_label="value-per-conv",
        arm_col=arm_col,
        date_col=date_col,
        rmse_col="rmse_roll",
        write_pngs=write_pngs,
        write_pdf=write_pdf,
        pdf_name=f"rolling_rmse_vpc_w{window}.pdf",
    )

    return {"conv": conv_written, "vpc": vpc_written}

# Marginal curves
@dataclass(frozen=True)
class GridPlotContract:
    estimator_col: str = "estimator"
    env_id_col: str = "env_id"
    arm_col: str = "arm_id"
    spend_col: str = "spend"
    y_col: str = "d_conv_d$"
    seasonality_col: str = "seasonality"
    macro_col: str = "macro_index"

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _infer_env_meta(
    df_env: pd.DataFrame,
    contract: GridPlotContract,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if contract.seasonality_col in df_env.columns:
        s = df_env[contract.seasonality_col].dropna()
        if len(s) > 0:
            out["seasonality"] = float(s.iloc[0])
    if contract.macro_col in df_env.columns:
        m = df_env[contract.macro_col].dropna()
        if len(m) > 0:
            out["macro_index"] = float(m.iloc[0])
    return out

def _weights_for_arm_env(
    weights_df: pd.DataFrame,
    *,
    env_id: str,
    arm_id: str,
    estimator_col: str = "estimator",
    env_id_col: str = "env_id",
    arm_col: str = "arm_id",
    weight_col: str = "weight",
) -> dict[str, float]:
    if weights_df is None:
        return {}
    req = {estimator_col, env_id_col, arm_col, weight_col}
    if not req.issubset(set(weights_df.columns)):
        return {}
    w = weights_df.loc[
        (weights_df[env_id_col].astype(str) == str(env_id))
        & (weights_df[arm_col].astype(str) == str(arm_id)),
        [estimator_col, weight_col],
    ].copy()
    if w.empty:
        return {}
    return {str(r[estimator_col]): float(r[weight_col]) for _, r in w.iterrows()}

def plot_arm_marginal_curves(
    plot_df: pd.DataFrame,
    *,
    env_id: str,
    arm_id: str,
    contract: GridPlotContract = GridPlotContract(),
    estimator_order: Optional[Sequence[str]] = None,
    spend_points_df: Optional[pd.DataFrame] = None,
    marker_cols: Optional[Mapping[str, str]] = None,
    weights_df: Optional[pd.DataFrame] = None,
    # NEW:
    spend_obs_df: Optional[pd.DataFrame] = None,
    spend_obs_col: str = "spend",
    show_rug: bool = True,
    rug_alpha: float = 0.35,
    rug_markersize: float = 10.0,
    title_prefix: str = "Marginal conversions per $",
    figsize: tuple[float, float] = (8.5, 5.0),
) -> plt.Figure:
    """
    Build a single overlay plot for one (env_id, arm_id).
    Markers are optional: pass spend_points_df + marker_cols to draw vertical lines.
    Weights are optional: pass weights_df to append weights in the legend.
    """
    c = contract
    required = {c.estimator_col, c.env_id_col, c.arm_col, c.spend_col, c.y_col}
    missing = [x for x in required if x not in plot_df.columns]
    if missing:
        raise KeyError(f"plot_df missing required columns: {missing}")

    df = plot_df.loc[
        (plot_df[c.env_id_col].astype(str) == str(env_id))
        & (plot_df[c.arm_col].astype(str) == str(arm_id))
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for env_id={env_id}, arm_id={arm_id}")

    # Standardize numeric types
    df[c.spend_col] = pd.to_numeric(df[c.spend_col], errors="coerce")
    df[c.y_col] = pd.to_numeric(df[c.y_col], errors="coerce")
    df = df.dropna(subset=[c.spend_col, c.y_col])

    env_meta = _infer_env_meta(df, c)

    # Determine estimator ordering
    ests = sorted(df[c.estimator_col].astype(str).unique().tolist())
    if estimator_order is not None:
        # keep only those present, in requested order, plus any remaining
        requested = [e for e in estimator_order if e in ests]
        remaining = [e for e in ests if e not in requested]
        ests = requested + remaining

    # Optional weights for labeling
    wmap = _weights_for_arm_env(
        weights_df,
        env_id=str(env_id),
        arm_id=str(arm_id),
        estimator_col=c.estimator_col,
        env_id_col=c.env_id_col,
        arm_col=c.arm_col,
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    for est in ests:
        g = df.loc[df[c.estimator_col].astype(str) == est].sort_values(c.spend_col)
        label = est
        if est in wmap:
            label = f"{est} (w={wmap[est]:.3f})"
        ax.plot(g[c.spend_col].to_numpy(dtype=float), g[c.y_col].to_numpy(dtype=float), label=label)

    # --- Support visualization: rug of observed spends (optional) ---
    if show_rug and spend_obs_df is not None:
        if c.arm_col not in spend_obs_df.columns:
            raise KeyError(f"spend_obs_df must contain column '{c.arm_col}'")
        if spend_obs_col not in spend_obs_df.columns:
            raise KeyError(f"spend_obs_df must contain spend column '{spend_obs_col}'")

        x_obs = spend_obs_df.loc[
            spend_obs_df[c.arm_col].astype(str) == str(arm_id),
            spend_obs_col,
        ]
        x_obs = pd.to_numeric(x_obs, errors="coerce").dropna().to_numpy(dtype=float)

        if x_obs.size > 0:
            # Place rug slightly above the bottom y-limit so it’s visible
            y0, y1 = ax.get_ylim()
            y_rug = y0 + 0.03 * (y1 - y0) if y1 > y0 else y0

            ax.plot(
                x_obs,
                np.full_like(x_obs, y_rug),
                "|",
                alpha=rug_alpha,
                markersize=rug_markersize,
            )

    # Optional markers (baseline/optimum/etc.)
    marker_cols = dict(marker_cols) if marker_cols is not None else {
        "actual": "spend_actual",
        "optimum": "spend_opt",
    }

    if spend_points_df is not None:
        if c.arm_col not in spend_points_df.columns:
            raise KeyError(f"spend_points_df must contain column '{c.arm_col}'")

        sp_row = spend_points_df.loc[
            spend_points_df[c.arm_col].astype(str) == str(arm_id)
        ]
        if len(sp_row) > 0:
            sp_row = sp_row.iloc[0]
            for marker_label, col in marker_cols.items():
                if col in spend_points_df.columns and pd.notna(sp_row.get(col, np.nan)):
                    x = float(sp_row[col])
                    ax.axvline(x, linestyle="--")
                    # annotate lightly near top
                    y_top = ax.get_ylim()[1]
                    ax.text(x, y_top, f" {marker_label}", rotation=90, va="top", ha="left")

    # Titles / labels
    subtitle_bits = [f"env_id={env_id}", f"arm_id={arm_id}"]
    if "seasonality" in env_meta:
        subtitle_bits.append(f"S={env_meta['seasonality']:.3f}")
    if "macro_index" in env_meta:
        subtitle_bits.append(f"M={env_meta['macro_index']:.3f}")

    ax.set_title(f"{title_prefix}\n" + " | ".join(subtitle_bits))
    ax.set_xlabel("Spend")
    ax.set_ylabel(c.y_col)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    return fig

def export_env_curve_pngs_and_pdf(
    plot_df: pd.DataFrame,
    *,
    out_dir: Path,
    env_id: str,
    contract: GridPlotContract = GridPlotContract(),
    arms: Optional[Sequence[str]] = None,
    estimator_order: Optional[Sequence[str]] = None,
    spend_points_df: Optional[pd.DataFrame] = None,
    marker_cols: Optional[Mapping[str, str]] = None,
    weights_df: Optional[pd.DataFrame] = None,
    spend_obs_df: Optional[pd.DataFrame] = None,
    spend_obs_col: str = "spend",
    show_rug: bool = True,
    write_pngs: bool = True,
    write_pdf: bool = True,
    png_dpi: int = 160,
) -> dict[str, Path]:
    """
    Export per-arm PNGs and a single multi-page PDF for one env_id.
    Returns paths written: {"png_dir": ..., "pdf_path": ...} (subset based on flags).
    """
    c = contract
    out_dir = Path(out_dir)

    df_env = plot_df.loc[plot_df[c.env_id_col].astype(str) == str(env_id)].copy()
    if df_env.empty:
        raise ValueError(f"No rows found for env_id={env_id}")

    all_arms = sorted(df_env[c.arm_col].astype(str).unique().tolist())
    if arms is None:
        arms_to_plot = all_arms
    else:
        arms_to_plot = [a for a in arms if str(a) in set(all_arms)]
        if not arms_to_plot:
            raise ValueError(f"No requested arms found in plot_df for env_id={env_id}. Requested={list(arms)}")

    written: dict[str, Path] = {}

    # PNG directory
    png_dir = out_dir / f"env_id={env_id}" / "png"
    if write_pngs:
        png_dir.mkdir(parents=True, exist_ok=True)
        written["png_dir"] = png_dir

    # PDF path
    pdf_path = out_dir / f"env_id={env_id}" / f"marginal_curves_env_id={env_id}.pdf"
    if write_pdf:
        _ensure_dir(pdf_path)
        written["pdf_path"] = pdf_path

    pdf = PdfPages(pdf_path) if write_pdf else None

    try:
        for arm_id in arms_to_plot:
            fig = plot_arm_marginal_curves(
                plot_df,
                env_id=str(env_id),
                arm_id=str(arm_id),
                contract=c,
                estimator_order=estimator_order,
                spend_points_df=spend_points_df,
                marker_cols=marker_cols,
                weights_df=weights_df,
                spend_obs_df=spend_obs_df,
                spend_obs_col=spend_obs_col,
                show_rug=show_rug,
            )

            if write_pngs:
                png_path = png_dir / f"arm_id={arm_id}.png"
                fig.savefig(png_path, dpi=png_dpi)
            if write_pdf and pdf is not None:
                pdf.savefig(fig)

            plt.close(fig)

    finally:
        if pdf is not None:
            pdf.close()

    return written


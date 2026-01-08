# notebooks/07_spend_allocation.py

"""
Functions as 'control panel' in relation to other modules.
"""

# Import libraries and modules
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from datetime import date
today_str = date.today().strftime("%m-%d-%y")

import pandas as pd
import numpy as np

from src.dgp import generate_synthetic_spend_world
from src.split import time_holdout_split, split_summary_df
from src.estimators.estimator_dd1 import local_slope_curve_grid
from src.estimators.estimator_rc1 import fit_per_arm_rc1, marg_curve_grid_rc1
from src.estimators.estimator_rc2 import fit_per_arm_rc2, marg_curve_grid_rc2
from src.marg_grid import build_spend_grid_by_arm, project_monotone_marginal
from src.eval import (
    arm_pred_perform_rc1,
    arm_pred_perform_rc2,
    eval_curve_grid_on_panel,
    conf_fit_from_rmse,
    conf_data_from_train,
    make_expanding_time_folds,
    build_perf_folds,
    conf_stability_from_perf_folds,
    panel_preds_from_curve_grid,
    conf_fit_local_from_panel_preds,
    conf_data_local_by_estimator_from_train,
    build_local_fit_folds,
    conf_stability_local_from_fit_folds,
)
from src.ensemble import (
    make_weights, 
    make_weights_local, 
    build_ensemble_grid, 
    AnchorConfig, 
    add_exp_conversions_to_ensemble_grid, 
    build_marginal_profit_grid,
)
from src.allocator import allocate_budget_kkt
from src.test_runs import build_day_total_row
from src.graphics import forecast_errors, export_env_curve_pngs_and_pdf
from src.exports import save_df, save_workbook
from src.checks import (
    assert_has_cols, 
    assert_unique_col, 
    assert_unique_row,
    assert_nonneg, 
    assert_curve_grid_contract, 
    assert_grids_joinable,
    assert_unique_row_conf,
    assert_weights_sum_to_one,
    assert_downstream_val,
    assert_profit_grid_contract,
    assert_spend_bounds_feasible,
    assert_alloc_output,
    assert_spend_df,
)
from src.diagnostics import (
    marginal_grids_diagnostics,
    build_spend_support_tables,
    allocation_opt_diagnostics,
    budget_sweep,
    cap_sweep,
    eval_policy_on_profit_grid,
    allocation_comp_diagnostics,
)

# Parameters
## Directory
out_dir = PROJECT_ROOT / "outputs"
out_dir_g = out_dir / "graphics"

## Select output
Export = {
    "data_generation": False,
    "model_fitting": False,
    "marginal_grids": False,
    "confidence": False,
    "ensemble": False,
    "allocation": False,
    "test_runs": False,
    "marg_plots": False,
    "fc_error": False,
}

# Define the main function to execute the notebook steps
def main():
    # Generate synthetic data
    arms_df, macro_df, ad_spend_df = generate_synthetic_spend_world(n_days=180, random_state=42)

    ## Check synthetic data
    assert_has_cols(arms_df, ["arm_id", "p_fund", "margin", "alpha", "beta", "gamma", "delta"], "arms_df")
    assert_has_cols(macro_df, ["date", "seasonality", "macro_index"], "macro_df")
    assert_has_cols(ad_spend_df, ["date", "arm_id", "spend", "conversions"], "ad_spend_df")
    assert_unique_col(arms_df, "arm_id", "arms_df")
    assert_unique_col(macro_df, "date", "macro_df")
    assert_unique_row(ad_spend_df)
    assert_nonneg(ad_spend_df, "spend", "ad_spend_df")
    assert_nonneg(ad_spend_df, "conversions", "ad_spend_df")

    # Split data into train/test
    split = time_holdout_split(ad_spend_df=ad_spend_df, macro_df=macro_df, test_size_days=30)
    ad_spend_train_df = split.train_ad_spend
    ad_spend_test_df = split.test_ad_spend
    macro_train_df = split.train_macro
    macro_test_df = split.test_macro
    split_df = split_summary_df(split)

    # Specify estimators
    estimators = {
    "dd1": {
        "name": "dd1_ls",
        "kind": "curve_grid",
        "fit": local_slope_curve_grid,
        "eval": eval_curve_grid_on_panel,
        "fit_kwargs": {"n_bins": 12},  # optional; only used by rc1
    },
    "rc1": {
        "name": "rc1_nes",
        "kind": "panel_data",
        "fit": fit_per_arm_rc1,
        "eval": arm_pred_perform_rc1,
        "grid": marg_curve_grid_rc1,
    },
    "rc2": {
        "name": "rc2_mmh",
        "kind": "panel_data",
        "fit": fit_per_arm_rc2,
        "eval": arm_pred_perform_rc2,
        "grid": marg_curve_grid_rc2,
    },
    }

    # Fit response curves on training
    params_rc1_df = fit_per_arm_rc1(ad_spend_train_df, macro_train_df, arm_col="arm_id")
    params_rc2_df = fit_per_arm_rc2(ad_spend_train_df, macro_train_df, arm_col="arm_id")

    # Evaluate response curves on test
    # comp_params(params_rc1_df, arms_df)
    perf_rc1_df = arm_pred_perform_rc1(params_rc1_df, ad_spend_test_df, macro_test_df)
    perf_rc2_df = arm_pred_perform_rc2(params_rc2_df, ad_spend_test_df, macro_test_df)

    # Establish settings for grid construction
    ## Environment
    ### Scenarios
    S_test = float(macro_test_df["seasonality"].mean())
    M_test = float(macro_test_df["macro_index"].mean())
    s_sd = float(macro_test_df["seasonality"].std())
    m_sd = float(macro_test_df["macro_index"].std())

    environs_df = pd.DataFrame(
        [
            {"env_id": "test_mean", "seasonality": S_test, "macro_index": M_test},
            {"env_id": "season_high", "seasonality": S_test + s_sd, "macro_index": M_test},
            {"env_id": "season_low", "seasonality": S_test - s_sd, "macro_index": M_test},
            {"env_id": "macro_high", "seasonality": S_test, "macro_index": M_test + m_sd},
            {"env_id": "macro_low", "seasonality": S_test, "macro_index": M_test - m_sd},
        ]
    )

    ### Select environment
    env_id = str(environs_df.loc[0, "env_id"])
    S = float(environs_df.loc[0, "seasonality"])
    M = float(environs_df.loc[0, "macro_index"])
    
    ## Spend 
    ### Bounds per-arm
    mean_spend_test = ad_spend_test_df.groupby("arm_id")["spend"].mean().to_dict()
    min_spend_test = ad_spend_test_df.groupby("arm_id")["spend"].min().to_dict()
    max_spend_test = ad_spend_test_df.groupby("arm_id")["spend"].max().to_dict()
    min_spend = {a: 0.5 * v for a, v in min_spend_test.items()}
    max_spend = {a: 2 * v for a, v in max_spend_test.items()}

    ## Shared spend grid for all estimators
    n_points = 25
    arms = sorted(ad_spend_df["arm_id"].unique())
    spend_grid_by_arm = build_spend_grid_by_arm(
        min_spend=min_spend,
        max_spend=max_spend,
        arms=arms,
        n_points=n_points,
    )

    # Compute marginal curves as grids
    marg_grid_dd1_df = local_slope_curve_grid(
    estimator=estimators["dd1"]["name"],
    env_id=env_id,
    ad_spend_df=ad_spend_train_df,
    min_spend=min_spend,
    max_spend=max_spend,
    S=S,
    M=M,
    arm_col="arm_id",
    n_points=n_points,
    n_bins=12,
    spend_grid_by_arm=spend_grid_by_arm,
    )
    assert_curve_grid_contract(marg_grid_dd1_df)
        
    marg_grid_rc1_df = marg_curve_grid_rc1(
        estimator=estimators["rc1"]["name"],
        env_id=env_id,
        params_df=params_rc1_df,     # alpha_hat/beta_hat/gamma_hat/delta_hat
        min_spend=min_spend,
        max_spend=max_spend,
        S=S,
        M=M,
        arm_col="arm_id",
        n_points=n_points,
        spend_grid_by_arm=spend_grid_by_arm,
    )
    assert_curve_grid_contract(marg_grid_rc1_df)

    marg_grid_rc2_df = marg_curve_grid_rc2(
        estimator=estimators["rc2"]["name"],
        env_id=env_id,
        params_df=params_rc2_df,     # alpha_hat/beta_hat/gamma_hat/delta_hat
        min_spend=min_spend,
        max_spend=max_spend,
        S=S,
        M=M,
        arm_col="arm_id",
        n_points=n_points,
        spend_grid_by_arm=spend_grid_by_arm,
    )
    assert_curve_grid_contract(marg_grid_rc2_df)

    ## Check marginal grid conformity
    assert_grids_joinable(marg_grid_dd1_df, marg_grid_rc1_df, require_same_env=True)
    assert_grids_joinable(marg_grid_dd1_df, marg_grid_rc2_df, require_same_env=True)

    ## Marginal grid diagnostics
    tabs_diag_marg_grid = marginal_grids_diagnostics(
        {"dd1": marg_grid_dd1_df, "rc1": marg_grid_rc1_df, "rc2": marg_grid_rc2_df},
        reference="dd1",              # optional; defaults to first key
        include_pairwise_diffs=True,  # set False if the workbook gets too large
    )

    # Evaluate dd1 holdout performance (TEST) via curve-grid interpolation
    perf_dd1_df = eval_curve_grid_on_panel(
        curve_grid_df=marg_grid_dd1_df,
        panel_df=ad_spend_test_df,
    )
    
    # Confidence
    ## Per-arm
    ### Fit
    conf_fit_dd1_df = conf_fit_from_rmse(
        perf_dd1_df,
        estimator=estimators["dd1"]["name"],
        env_id=env_id,
    )
    conf_fit_rc1_df = conf_fit_from_rmse(
        perf_rc1_df,
        estimator=estimators["rc1"]["name"],
        env_id=env_id,
    )
    conf_fit_rc2_df = conf_fit_from_rmse(
        perf_rc2_df,
        estimator=estimators["rc2"]["name"],
        env_id=env_id,
    )
    conf_fit_df = pd.concat([conf_fit_dd1_df, conf_fit_rc1_df, conf_fit_rc2_df], ignore_index=True)

    ### Data
    conf_data_df = conf_data_from_train(ad_spend_train_df)

    ### Stability
    folds = make_expanding_time_folds(
        ad_spend_df=ad_spend_train_df,
        macro_df=macro_train_df,
        test_size_days=30,
        n_folds=4,
        step_days=30,
        gap_days=0,
        min_train_days=60,
    )

    perf_folds_df = build_perf_folds(
        folds=folds,
        estimators=estimators,
        spend_grid_by_arm=spend_grid_by_arm,
        min_spend=min_spend,
        max_spend=max_spend,
        arm_col="arm_id",
    )

    conf_stability_df = conf_stability_from_perf_folds(
        perf_folds_df,
        arm_col="arm_id",
    )

    ### Combine components
    conf_df = (
        conf_fit_df
        .merge(conf_data_df[["arm_id", "conf_data"]], on="arm_id", how="left", validate="many_to_one")
        .merge(conf_stability_df[["estimator", "arm_id", "conf_stability"]], on=["estimator", "arm_id"], how="left", validate="one_to_one")
    )
    conf_df["conf_data"] = conf_df["conf_data"].fillna(1.0)
    conf_df["conf_stability"] = conf_df["conf_stability"].fillna(1.0)

    ## Spend level
    ### Fit
    panel_preds_test_dd1_df = panel_preds_from_curve_grid(marg_grid_dd1_df, ad_spend_test_df)
    conf_fit_local_dd1_df = conf_fit_local_from_panel_preds(
        panel_preds_test_dd1_df,
        spend_grid_by_arm=spend_grid_by_arm,
        estimator=estimators["dd1"]["name"],
        env_id=env_id,
    )
    panel_preds_test_rc1_df = panel_preds_from_curve_grid(marg_grid_rc1_df, ad_spend_test_df)
    conf_fit_local_rc1_df = conf_fit_local_from_panel_preds(
        panel_preds_test_rc1_df,
        spend_grid_by_arm=spend_grid_by_arm,
        estimator=estimators["rc1"]["name"],
        env_id=env_id,
    )
    panel_preds_test_rc2_df = panel_preds_from_curve_grid(marg_grid_rc2_df, ad_spend_test_df)
    conf_fit_local_rc2_df = conf_fit_local_from_panel_preds(
        panel_preds_test_rc2_df,
        spend_grid_by_arm=spend_grid_by_arm,
        estimator=estimators["rc2"]["name"],
        env_id=env_id,
    )
    conf_fit_local_df = pd.concat([conf_fit_local_dd1_df, conf_fit_local_rc1_df, conf_fit_local_rc2_df], ignore_index=True)
    assert_unique_row_conf(conf_fit_local_df, ["env_id", "estimator", "arm_id", "spend"], "conf_fit_local_df")
    conf_fit_local_df = conf_fit_local_df.sort_values(["env_id", "estimator", "arm_id", "spend"]).reset_index(drop=True)

    ### Data
    conf_data_local_df = conf_data_local_by_estimator_from_train(
        ad_spend_train_df,
        spend_grid_by_arm=spend_grid_by_arm,
        estimators=estimators,
        kernel="gaussian",
        bandwidth=None,
        bandwidth_mult=2.5,
        bandwidth_mult_by_estimator={"dd1": 1.0, "rc1": 2.0, "rc2": 2.0},
        n0_local=10.0,
    )

    ### Stability
    folds = make_expanding_time_folds(
        ad_spend_df=ad_spend_df,
        macro_df=macro_df,
        test_size_days=30,
        n_folds=4,
        step_days=30,
        gap_days=0,
        min_train_days=60,
    )

    local_fit_folds_df = build_local_fit_folds(
        folds=folds,
        estimators=estimators,
        spend_grid_by_arm=spend_grid_by_arm,
        min_spend=min_spend,
        max_spend=max_spend,
        kernel="gaussian",
        bandwidth_mult=2.5,
        # optional: more/less locality per estimator *key* (not name)
        bandwidth_mult_by_estimator={"dd1": 1.0, "rc1": 2.0, "rc2": 2.0},
    )

    conf_stability_local_df = conf_stability_local_from_fit_folds(local_fit_folds_df)

    ### Combine confidence components
    keys = ["estimator", "arm_id", "spend"]

    fit_keep = conf_fit_local_df.copy().rename(
        columns={
            "bandwidth": "bandwidth_fit",
            "kernel": "kernel_fit",
            "n_eff": "n_eff_fit",
            "sum_w": "sum_w_fit",
        }
    )

    data_keep = conf_data_local_df.copy().rename(
        columns={
            "bandwidth": "bandwidth_data",
            "kernel": "kernel_data",
            "n_eff": "n_eff_data",
            "sum_w": "sum_w_data",
        }
    )

    conf_local_df = (
        fit_keep
        .merge(data_keep, on=keys, how="left", validate="many_to_one")
        .merge(conf_stability_local_df, on=keys, how="left", validate="many_to_one")
    )

    ## Diagnostics (spend-leve)
    tabs_diag_data_support = build_spend_support_tables(
        ad_spend_df=ad_spend_train_df,
        spend_grid_by_arm=spend_grid_by_arm,
        conf_local_df=conf_local_df,   # optional but recommended if it contains n_eff
    )

    # Specify weights
    ## Per-arm
    weights_df = make_weights(conf_df)
    assert_weights_sum_to_one(weights_df, group_cols=("env_id", "arm_id"), weight_col="weight", atol=1e-6)

    ## Spend level
    weights_local_df = make_weights_local(conf_local_df)
    assert_weights_sum_to_one(weights_local_df, group_cols=("env_id", "arm_id","spend"), weight_col="weight", atol=1e-6)

    # Create ensemble grid
    ensemble_grid_init_df = build_ensemble_grid(
        grids=[marg_grid_dd1_df, marg_grid_rc1_df, marg_grid_rc2_df],
        weights_df=weights_local_df,
        out_estimator="ensemble_local",
    )
    assert_curve_grid_contract(ensemble_grid_init_df)

    ## Enforce monotonicity
    ensemble_grid_mono_df = project_monotone_marginal(
        ensemble_grid_init_df,
        group_cols=("env_id", "arm_id"),   # ensemble has one curve per (env, arm)
        dcol="d_conv_d$",
        spend_col="spend",
        weight_col=None,                  # optional: could use e.g. "n_eff_data" later
        nonneg=True,
        keep_raw=True,
    )
    assert_curve_grid_contract(ensemble_grid_mono_df)
    
    ## Diagnostics using init as reference
    tabs_diag_ens = marginal_grids_diagnostics(
        {"ensemble_init": ensemble_grid_init_df, "ensemble_mono": ensemble_grid_mono_df},
        reference="ensemble_init",
        include_pairwise_diffs=True,
    )

    ## Add expected conversions
    ### Baseline spend per arm, training mean, as the anchor spend s*
    baseline_spend = ad_spend_train_df.groupby("arm_id")["spend"].mean().copy()
    baseline_spend.index.name = "arm_id"

    ### Use weighted average of *response-curve* estimators for fallback anchor
    rc_estimators = [estimators["rc1"]["name"], estimators["rc2"]["name"]]

    ensemble_grid_fin_df, anchors_df = add_exp_conversions_to_ensemble_grid(
        ensemble_grid_mono_df,
        baseline_spend=baseline_spend,
        ad_spend_train_df=ad_spend_train_df,
        spend_grid_by_arm=spend_grid_by_arm,
        level_grids_for_anchor=[marg_grid_rc1_df, marg_grid_rc2_df],
        weights_df=weights_local_df,
        response_curve_estimators=rc_estimators,
        cfg=AnchorConfig(env_ref_id=env_id, kernel_bandwidth_mult=2.5, n0_anchor=10.0),
    )

    # Downstream value (placeholder for future prediction routines)
    ds_val_df = arms_df[["arm_id", "p_fund", "margin"]].copy()
    assert_downstream_val(ds_val_df)

    # Marginal value grid
    marg_profit_grid_df = build_marginal_profit_grid(
        conv_grid_df=ensemble_grid_fin_df,
        ds_val_df=ds_val_df,
    )
    assert_profit_grid_contract(marg_profit_grid_df)

    # Allocate daily spend
    ## Run timeframe
    test_dates = sorted(ad_spend_test_df["date"].unique())
    
    ## Grid input
    alloc_grid_df = marg_profit_grid_df[
        ["env_id", "arm_id", "spend", "value_per_conv", "d_conv_d$", "exp_conversions", "d_profit_gross_d$", "exp_profit_gross", "d_profit_net_d$", "exp_profit_net"]
    ].copy()

    ## Constraints
    allow_unspent=True
    capacity_conversions=None
    
    ## Run control
    num_dates = len(test_dates)
    cnt = 0

    ## Data capture
    daily_comp_arm = []
    daily_comp_total = []

    ## Daily run
    for date_t in test_dates:
        cnt=cnt+1
        print(f"counter: {cnt}")

        ## Actuals
        actuals_day_df = (
            ad_spend_test_df.loc[ad_spend_test_df["date"].eq(date_t), ["arm_id", "spend", "conversions", "funded_loans", "profit_gross", "profit_net"]]
            .rename(columns={"spend": "spend_actual", "conversions": "conversions_actual", "funded_loans": "funded_loans_actual", "profit_gross": "profit_gross_actual", "profit_net": "profit_net_actual"})
            .copy()
        )

        ## Budget
        spend_actual = actuals_day_df.set_index("arm_id")["spend_actual"]
        budget = float(spend_actual.sum())*1.2
        assert_spend_bounds_feasible(min_spend, max_spend, budget)
        
        ## Optimization routine
        alloc_df = allocate_budget_kkt(
            marg_profit_grid_df,
            budget=budget,
            env_id=env_id,
            arm_col="arm_id",
            spend_col="spend",
            marg_col="d_profit_net_d$",
            exp_conv_col="exp_conversions",
            d_conv_col="d_conv_d$",
            exp_profit_col="exp_profit_net",
            val_col="value_per_conv",
            allow_unspent=allow_unspent,
            min_spend=min_spend,
            max_spend=max_spend,
            capacity_conversions=capacity_conversions,
        )
        alloc_sum_df = alloc_df.iloc[:-1].copy()
        assert_alloc_output(alloc_sum_df, budget, allow_unspent)
        
        ## Diagnostics
        ### Optimim alone
        if cnt==num_dates:
            tabs_diag_alloc_opt = allocation_opt_diagnostics(
                alloc_df=alloc_sum_df,
                profit_grid_df=marg_profit_grid_df,
                env_id=env_id,
                allow_unspent=allow_unspent,
                budget=budget,
                capacity_conversions=capacity_conversions,
            )
            print(f"Computed first diagnostic")

        ### Compare performance
        spend_opt = alloc_sum_df.set_index("arm_id")["spend_opt"]

        eval_actuals_df = eval_policy_on_profit_grid(
            marg_profit_grid_df,
            env_id=env_id,
            spend_by_arm=spend_actual,
        ).rename(columns={
            "spend_eval": "spend_actual",
            "spend_eval_clamped": "spend_clamped_actual",
            "exp_conversions": "exp_conversions_actual",
            "exp_profit_net": "exp_profit_net_actual",
            "d_conv_d$": "d_conv_d$_actual",
            "d_profit_net_d$": "d_profit_net_d$_actual",
        })

        eval_opt_df = eval_policy_on_profit_grid(
            marg_profit_grid_df,
            env_id=env_id,
            spend_by_arm=spend_opt,
        ).rename(columns={
            "spend_eval": "spend_opt",
            "spend_eval_clamped": "spend_clamped_opt",
            "exp_conversions": "exp_conversions_opt",
            "exp_profit_net": "exp_profit_net_opt",
            "d_conv_d$": "d_conv_d$_opt",
            "d_profit_net_d$": "d_profit_net_d$_opt",
        })

        day_compare_df = (
            actuals_day_df
            .merge(eval_actuals_df[["arm_id", "spend_clamped_actual", "exp_conversions_actual", "exp_profit_net_actual", "d_conv_d$_actual", "d_profit_net_d$_actual"]], on="arm_id", how="left", validate="one_to_one")
            .merge(eval_opt_df, on="arm_id", how="left", validate="one_to_one")
            .merge(alloc_sum_df[["arm_id", "spend_min", "spend_max", "tau", "value_per_conv", "nu_conv", "d_profit_net_d$_adj"]], on="arm_id", how="left", validate="one_to_one")
        )

        day_compare_df = day_compare_df.rename(columns={"value_per_conv": "exp_vpc"})

        eps = 1e-9
        day_compare_df["date_eval"] = date_t
        day_compare_df["env_id"] = env_id
        day_compare_df["budget"] = budget
        day_compare_df["vpc_actual"] = day_compare_df["profit_gross_actual"] / (day_compare_df["conversions_actual"] + eps)
        day_compare_df.loc[day_compare_df["conversions_actual"].eq(0), "vpc_actual"] = np.nan
        day_compare_df["fc_error_conv"] = day_compare_df["exp_conversions_actual"] - day_compare_df["conversions_actual"]
        day_compare_df["fc_error_vpc"] = day_compare_df["exp_vpc"] - day_compare_df["vpc_actual"]
        day_compare_df["fc_error_np"] = day_compare_df["exp_profit_net_actual"] - day_compare_df["profit_net_actual"]
        day_compare_df["fc_error_np_conv"] = (day_compare_df["exp_conversions_actual"] * day_compare_df["vpc_actual"]) - day_compare_df["spend_actual"] - day_compare_df["profit_net_actual"]
        day_compare_df["fc_error_np_vpc"] = (day_compare_df["conversions_actual"] * day_compare_df["exp_vpc"]) - day_compare_df["spend_actual"] - day_compare_df["profit_net_actual"]
        day_compare_df["delta_spend"] = day_compare_df["spend_opt"] - day_compare_df["spend_actual"]
        day_compare_df["delta_spend_clamped"] = day_compare_df["spend_clamped_opt"] - day_compare_df["spend_clamped_actual"]
        day_compare_df["delta_exp_conversions"] = day_compare_df["exp_conversions_opt"] - day_compare_df["exp_conversions_actual"]
        day_compare_df["delta_exp_profit_net"] = day_compare_df["exp_profit_net_opt"] - day_compare_df["exp_profit_net_actual"]
        day_compare_df["flag_clamped_actual"] = (day_compare_df["spend_clamped_actual"] - day_compare_df["spend_actual"]).abs() > eps
        day_compare_df["flag_clamped_opt"] = (day_compare_df["spend_clamped_opt"] - day_compare_df["spend_opt"]).abs() > eps
        day_compare_df["flag_conv_act_zero"] = day_compare_df["conversions_actual"] < eps

        day_compare_fin_df = day_compare_df[
            ["date_eval",
            "env_id",
            "arm_id",
            "spend_min",
            "spend_max",
            "budget",
            "exp_vpc",
            "spend_actual",
            "conversions_actual",
            "funded_loans_actual",
            "profit_gross_actual",
            "vpc_actual",
            "profit_net_actual",
            "spend_clamped_actual",
            "exp_conversions_actual",
            "d_conv_d$_actual",
            "exp_profit_net_actual",
            "d_profit_net_d$_actual",
            "spend_opt",
            "spend_clamped_opt",
            "exp_conversions_opt",
            "d_conv_d$_opt",
            "exp_profit_net_opt",
            "d_profit_net_d$_opt",
            "d_profit_net_d$_adj",
            "tau",
            "nu_conv",
            "fc_error_conv",
            "fc_error_vpc",
            "fc_error_np",
            "fc_error_np_conv",
            "fc_error_np_vpc",
            "delta_spend",
            "delta_spend_clamped",
            "delta_exp_conversions",
            "delta_exp_profit_net",
            "flag_clamped_actual",
            "flag_clamped_opt",
            "flag_conv_act_zero",
            ]
        ].copy()
        daily_comp_arm.append(day_compare_fin_df)

        if cnt==num_dates:
            tabs_diag_alloc_comp = allocation_comp_diagnostics(day_compare_fin_df, env_id=env_id)
            print(f"Computed second diagnostic")

        daily_comp_total.append(build_day_total_row(day_compare_fin_df, day=date_t, env_id=env_id))

    daily_comp_arm_df = pd.concat(daily_comp_arm, ignore_index=True)
    daily_comp_total_df = pd.concat(daily_comp_total, ignore_index=True)

    # Graphics

    ## Forecast errors
    if Export["fc_error"]:
        g_fc = forecast_errors(
            daily_comp_arm_df,
            out_dir=out_dir_g,
            window=7,              # rolling window length
            min_periods=5,         # optional; use <window if you want earlier days filled
            write_pngs=True,
            write_pdf=True,
        )
        print(f"Wrote forecast error graphics to {out_dir_g}:")

    # Marginal curves
    if Export["marg_plots"]:
        spend_points_df = (pd.concat([spend_actual, spend_opt], axis=1,).reset_index())
        assert_spend_df(spend_points_df)

        plot_df = pd.concat(
            [marg_grid_dd1_df, marg_grid_rc1_df, marg_grid_rc2_df, ensemble_grid_fin_df],
            ignore_index=True,
        )

        plotted = export_env_curve_pngs_and_pdf(
            plot_df=plot_df,
            out_dir=out_dir_g,
            env_id=env_id,
            spend_points_df=spend_points_df,
            weights_df=weights_local_df,
            spend_obs_df=ad_spend_train_df,
            spend_obs_col="spend",
            show_rug=True,
        )
    
        print(f"Wrote marginal curve graphics to {out_dir_g}:")

    # Save outputs
    ## Data generation and split
    if Export["data_generation"]:
        out_path = out_dir / f"data_generation ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "arms": arms_df,
                "macro": macro_df,
                "ad_spend": ad_spend_df,
                "split_summary": split_df,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote data_generation workbook to: {actual_path}")

    ## Model fitting
    if Export["model_fitting"]:
        out_path = out_dir / f"model_fitting ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "model_perform_dd1": perf_dd1_df,
                "model_params_rc1": params_rc1_df,
                "model_perform_rc1": perf_rc1_df,
                "model_params_rc2": params_rc2_df,
                "model_perform_rc2": perf_rc2_df,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote model_fitting workbook to: {actual_path}")

    ## Marginal grids
    if Export["marginal_grids"]:
        out_path = out_dir / f"marginal_grids ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "scenarios": environs_df,
                "marg_grid_dd1": marg_grid_dd1_df,
                "marg_grid_rc1": marg_grid_rc1_df,
                "marg_grid_rc2": marg_grid_rc2_df,
                **tabs_diag_marg_grid,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote marginal_grids workbook to: {actual_path}")

    ## Confidence
    if Export["confidence"]:
        out_path = out_dir / f"confidence ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "fit_dd1": conf_fit_dd1_df,
                "fit_rc1": conf_fit_rc1_df,
                "fit_rc2": conf_fit_rc2_df,
                "fit_comb": conf_fit_df,
                "data": conf_data_df,
                "folds": perf_folds_df,
                "stability": conf_stability_df,
                "confidence": conf_df,
                "fit_loc_dd1":conf_fit_local_dd1_df,
                "fit_loc_rc1":conf_fit_local_rc1_df,
                "fit_loc_rc2": conf_fit_local_rc2_df,
                "fit_loc_comb": conf_fit_local_df,
                "data_loc": conf_data_local_df,
                "folds_loc": local_fit_folds_df,
                "stability_loc": conf_stability_local_df,
                "confidence_loc": conf_local_df,
                **tabs_diag_data_support,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote confidence workbook to: {actual_path}")

    ## Ensemble
    if Export["ensemble"]:
        out_path = out_dir / f"ensemble ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "weights": weights_df,
                "weights_loc": weights_local_df,
                "ensemble_grid_init": ensemble_grid_init_df,
                "ensemble_grid_mono": ensemble_grid_mono_df,
                **tabs_diag_ens,
                "ensemble_grid_fin": ensemble_grid_fin_df,
                "anchors": anchors_df,
                "downstream_vals": ds_val_df,
                "marg_profit_grid": marg_profit_grid_df,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote ensemble workbook to: {actual_path}")

    ## Allocation
    if Export["allocation"]:
        out_path = out_dir / f"allocation ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "input": alloc_grid_df,
                "allocation": alloc_df,
                **tabs_diag_alloc_opt,
                #"budget_sweep": budget_sweep_df,
                #"cap_sweep": cap_sweep_df,
                "actuals": actuals_day_df,
                "eval_actuals": eval_actuals_df,
                "eval_opt": eval_opt_df,
                "compare": day_compare_fin_df,
                **tabs_diag_alloc_comp
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote allocation workbook to: {actual_path}")

    ## Test run
    if Export["test_runs"]:
        out_path = out_dir / f"test_runs ({today_str}).xlsx"
        actual_path = save_workbook(
            sheets={
                "daily_comp_arm": daily_comp_arm_df,
                "daily_comp_total": daily_comp_total_df,
            },
            out_path=out_path,
            index=False,
        )
        print(f"Wrote test_runs workbook to: {actual_path}")

if __name__ == "__main__":
    main()


    # ## Optimization point scenario analyses
    # ### Budget constraint
    # budgets = [0.5 * budget, 0.75 * budget, budget, 1.25 * budget, 1.5 * budget]
    # budget_sweep_df = budget_sweep(
    #     allocate_budget_kkt,
    #     marg_profit_grid_df=marg_profit_grid_df,
    #     env_id=env_id,
    #     budgets=budgets,
    #     allow_unspent=allow_unspent,
    #     capacity_conversions=capacity_conversions,
    # )

    # ## Capacity constraint
    # min_conv = (
    #     marg_profit_grid_df.loc[marg_profit_grid_df["env_id"].eq(env_id)]
    #     .sort_values(["arm_id", "spend"], ascending=[True, True])
    #     .groupby("arm_id", as_index=False)
    #     .first()["exp_conversions"]
    #     .sum()
    # )
    # opt_conv = float(alloc_sum_df["exp_conversions"].sum())

    # # Choose caps in a feasible range (examples)
    # caps = [
    #     float(min_conv) * 1.00,
    #     float(min_conv) * 1.05,
    #     float(min_conv) * 1.10,
    #     (float(min_conv) + float(opt_conv)) / 2.0,
    #     float(opt_conv),
    # ]

    # cap_sweep_df = cap_sweep(
    #     allocate_budget_kkt,
    #     marg_profit_grid_df=marg_profit_grid_df,
    #     env_id=env_id,
    #     budget=budget,
    #     caps=caps,
    #     allow_unspent=allow_unspent,
    # )
        
    # Save csv
    # out_dir = PROJECT_ROOT / "outputs"
    # csv_path = out_dir / "truth_cmp_true_eval.csv"
    # save_df(truth_cmp, csv_path)
    # print(f"Saved CSV: {csv_path}")
    
    # Save Excel, single sheet
    # out_dir = PROJECT_ROOT / "outputs"
    # xlsx_path = out_dir / "truth_cmp_true_eval.xlsx"
    # save_df(truth_cmp, xlsx_path)
    # print(f"Saved Excel: {xlsx_path}")

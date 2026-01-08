# src/contracts.py

# Import libraries and modules
from __future__ import annotations
from typing import Final, Tuple

# Canonical column names (use everywhere to avoid drift)
COL_ESTIMATOR: Final[str] = "estimator"
COL_ENV_ID: Final[str] = "env_id"
COL_ARM_ID: Final[str] = "arm_id"
COL_SPEND: Final[str] = "spend"
COL_SEASONALITY: Final[str] = "seasonality"
COL_MACRO_INDEX: Final[str] = "macro_index"

COL_EXP_CONVERSIONS: Final[str] = "exp_conversions"
COL_DCONV_DDOLLAR: Final[str] = "d_conv_d$"

# Backward-compatible aliases (shorter names used in some modules)
COL_EXP_CONV: Final[str] = COL_EXP_CONVERSIONS
COL_DCONV: Final[str] = COL_DCONV_DDOLLAR

# 'Thin' marginal curve grid contract, estimator agnostic.
CURVE_GRID_REQUIRED_COLS: Final[Tuple[str, ...]] = (
    COL_ESTIMATOR,
    COL_ENV_ID,
    COL_ARM_ID,
    COL_SPEND,
    COL_SEASONALITY,
    COL_MACRO_INDEX,
    COL_EXP_CONVERSIONS,
    COL_DCONV_DDOLLAR,
)

# Uniqueness contract for curve grid output
CURVE_GRID_KEY_COLS: Final[Tuple[str, ...]] = (
    COL_ESTIMATOR,
    COL_ENV_ID,
    COL_ARM_ID,
    COL_SPEND,
)

# Join keys for comparing/combining multiple estimator grids
CURVE_GRID_JOIN_COLS: Final[Tuple[str, ...]] = (
    COL_ENV_ID,
    COL_ARM_ID,
    COL_SPEND,
)

# Convenience contract, baseline vs opt spends
SPEND_POINTS_REQUIRED_COLS: Final[Tuple[str, ...]] = (
    COL_ARM_ID,
    "spend_actual",
    "spend_opt",
)

# Mmarginal net profit grid used by allocator - one row per env_id, arm_id, spend.
PROFIT_GRID_REQUIRED_COLS = [
    "env_id",
    "arm_id",
    "spend",
    # primary optimization signal
    "d_profit_net_d$",
    # needed for capacity constraint + exp conversions/profit reporting
    "d_conv_d$",
    "exp_conversions",
    "exp_profit_net",
    # value metadata used to construct profit terms (may be constant within arm)
    "value_per_conv",
]

# Allocation output contract from allocate_budget_kkt (post-format).
ALLOC_OUT_REQUIRED_COLS = [
    "arm_id",
    "spend_min",
    "spend_opt",
    "spend_max",
    "tau",
    # base marginal net profit slope at optimum
    "d_profit_net_d$",
    # adjusted slope used for KKT checks when capacity constraint is active
    "d_profit_net_d$_adj",
    # reporting columns
    "exp_profit_net",
    "exp_conversions",
    "d_conv_d$",
    "value_per_conv",
]

# Optional but commonly present columns
ALLOC_OUT_OPTIONAL_COLS = [
    "nu_conv",          # shadow price on capacity constraint (scalar repeated)
    "at_min",
    "at_max",
    "slack_from_min",
    "slack_to_max",
]

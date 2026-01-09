# spend_allocation

## Overview
Optimizes marketing spend allocation to maximize estimated net profit subject to:
- campaign-level min/max spend constraints
- global budget constraints
- operating capacity constraints

For each campaign (referred to as an **‘arm’**), the project fits multiple alternative response models (estimators) for expected conversions (loan applications) and constructs marginal conversion curves in grid form. Model forecast quality is evaluated across three confidence components, which are combined into weights used to produce campaign-level ensemble marginal conversion grids.

Marginal conversion grids are converted to marginal net profit grids using exogenous downstream forecasts of lower-funnel conversion and transaction value. A KKT-based optimizer then solves for optimal global spend and allocation across campaigns.

## Repository structure
- `notebooks/`
  - `spend_allocation.py` — orchestration / end-to-end runner
- `src/` — operating routines
  - `contracts.py` — object structure and naming conventions
  - `dgp.py` — synthetic data generation
  - `split.py` — train/test splitting
  - `estimators/`
    - `estimator_dd1.py` — data-driven local slope model
    - `estimator_rc1.py` — negative-exponential saturation response curve
    - `estimator_rc2.py` — Michaelis–Menten response curve
  - `marg_grid.py` — marginal conversion grid construction
  - `eval.py` — forecast quality scoring (three confidence components)
  - `ensemble.py` — confidence → weights; ensemble conversion + net profit grids
  - `allocator.py` — KKT optimization for spend allocation
  - `test_runs.py` — 30-day test harness
  - `graphics.py` — marginal grid + forecast quality reporting artifacts
  - `exports.py` — data export artifacts
  - `checks.py` — structural validations
  - `diagnostics.py` — quality/performance diagnostics artifacts

## Exclusions
The public repository intentionally excludes:
- `data/`
- `docs/`
- `outputs/`

## Quickstart
```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
pip install -r requirements.txt
```

## Usage
```bash
python notebooks/spend_allocation.py
```

# project_knowledge_content

This document identifies and briefly describes all documents constituting the Knowledge base for this Project, organized within three sections: (i) code; (ii) diagnostics; and (iii) context and direction.

## Code
- `notebooks/`
  - `00_spend_allocation.py` — orchestration / end-to-end runner
- `src/` — operating routines
  - `01_contracts.py` — object structure and naming conventions
  - `02_utils.py` — helpers
  - `03_dgp.py` — synthetic data generation
  - `04_split.py` — train/test splitting
  - `estimators/`
    - `05a_estimator_dd1.py` — data-driven local slope model
    - `05b_estimator_rc1.py` — negative-exponential saturation response curve
    - `05c_estimator_rc2.py` — Michaelis–Menten response curve
  - `06_marg_grid.py` — marginal conversion grid construction
  - `07_eval.py` — forecast quality scoring (three confidence components)
  - `08_ensemble.py` — confidence → weights; ensemble conversion + net profit grids
  - `09_allocator.py` — KKT optimization for spend allocation
  - `10_test_runs.py` — 30-day test harness
  - `11_graphics.py` — marginal grid + forecast quality reporting artifacts
  - `12_exports.py` — data export artifacts
  - `13_checks.py` — structural validations
  - `14_diagnostics.py` — quality/performance diagnostics artifacts

## Diagnostics
Each run of the code generates the following three Excel files and two pdfs as output for diagnostic review (note: an arm corresponds to a marketing campaign):
- **marginal_grids.xlsx**: one tab for each set of conversion/net-profit marginal grids (set is comprised of all arms) generated (these remain unchanged throughout the 30-day run); last tab (ensemble_grid_net_profit) is input to the allocation routine
- **marginal_curves_env_id=test_mean.pdf**: set of graphics, one for each arm, displaying marginal conversion curves associated with each estimator, ensemble curve, actual and optimized spend points and, at bottom, amount of data support across the spend grid
- **diagnostics_last_day.xlsx**: diagnostics, one tab for each diagnostic routine, generated each day throughout the run; version provided is for last day of the run
- **diagnostics_30_days.xlsx**: high-level diagnostic encompassing the full 30-days of the run
- **rolling_rmse_conversions_w7.pdf**: set of graphics, one for each arm, displaying rolling seven-day moving average of root mean squared error for arm-level ensemble estimators

## Context and Direction
- `system_overview.md` — objective, business context, high-level workflow, run characteristic and output
- `repo_architecture.md` — code structure and functionality
- `optimization_formulation.md` — specifies optimization in general mathematical terms and provides guidance on interpreting results
- `diagnostic_interpretation.md` — maps diagnostic workbook tabs in diagnostics_last_day.xlsx to their intent and how to interpret them
- `decision_framework.md` — first cut at a guide for recommending business actions in response to run results

Last updated: 2026-01-11
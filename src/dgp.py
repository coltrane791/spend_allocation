# src/dgp.py

"""
Data generation process - construct synthetic data.
"""

# Import libraries and modules
import numpy as np
import pandas as pd

# Function to generate synthetic marketing spend world
def generate_synthetic_spend_world(n_days: int = 180, random_state: int = 42):
    """Generate a synthetic marketing world at daily x arm granularity.

    Returns
    -------
    arms : pd.DataFrame
        One row per arm (channel x campaign) with true underlying parameters.
    macro : pd.DataFrame
        One row per day with seasonality and macro indices.
    ad_spend : pd.DataFrame
        One row per day x arm with spend, conversions, loans, and profit.
    """
    rng = np.random.default_rng(random_state)

    ## Define arms (channel x campaign "arms" for allocation)
    arms = pd.DataFrame(
        [
            {"arm_id": "SEARCH_BRAND_A", "channel": "Search_Brand", "campaign": "Brand_A"},
            {"arm_id": "SEARCH_GENERIC_A", "channel": "Search_Generic", "campaign": "Generic_A"},
            {"arm_id": "SEARCH_GENERIC_B", "channel": "Search_Generic", "campaign": "Generic_B"},
            {"arm_id": "PAID_SOCIAL_A", "channel": "Paid_Social", "campaign": "Prospecting_A"},
            {"arm_id": "PAID_SOCIAL_B", "channel": "Paid_Social", "campaign": "Prospecting_B"},
            {"arm_id": "DISPLAY_RETARGET", "channel": "Display_Retargeting", "campaign": "Retarget"},
            {"arm_id": "EMAIL_A", "channel": "Email", "campaign": "Email_A"},
            {"arm_id": "AFFIL_AGG", "channel": "Affiliate", "campaign": "Aggregator"},
        ]
    )

    n_arms = len(arms)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    t = np.arange(n_days, dtype=float)

    ## Seasonality and macro indices (simple but non-trivial patterns)
    seasonality = 0.2 * np.sin(2 * np.pi * t / 30.0)  # ~monthly wave
    macro_index = 0.1 * np.sin(2 * np.pi * t / 90.0) + 0.05 * rng.normal(size=n_days)

    macro = pd.DataFrame(
        {
            "date": dates,
            "seasonality": seasonality,
            "macro_index": macro_index,
        }
    )

    ## Underlying "true" parameters for each arm
    alpha = rng.uniform(10, 40, size=n_arms)       # max conversions/day at high spend
    beta = rng.uniform(1e-4, 5e-4, size=n_arms)    # responsiveness / saturation
    gamma = rng.normal(0.2, 0.05, size=n_arms)     # sensitivity to seasonality
    delta = rng.normal(0.1, 0.05, size=n_arms)     # sensitivity to macro
    p_fund = rng.uniform(0.15, 0.35, size=n_arms)  # app->funded probability
    margin = rng.uniform(3000, 8000, size=n_arms)  # profit per funded loan (gross)

    arms = arms.assign(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        p_fund=p_fund,
        margin=margin,
    )

    ## Generate daily spend and outcomes
    rows = []
    for a_idx, arm in arms.iterrows():
        # Baseline spend pattern for this arm
        base_spend = rng.uniform(5000, 30000)
        # Smooth seasonal variation in spend
        spend_trend = base_spend * (1.0 + 0.3 * np.sin(2 * np.pi * t / 60.0))
        # Add noise and floor at zero
        daily_spend = spend_trend + rng.normal(0.0, base_spend * 0.1, size=n_days)
        daily_spend = np.clip(daily_spend, 0.0, None)

        for day_idx, date in enumerate(dates):
            s = float(daily_spend[day_idx])

            # Expected "conversions" (e.g., applications) from this arm/day
            c_exp = (
                alpha[a_idx]
                * (1.0 - np.exp(-beta[a_idx] * s))
                * (1.0 + gamma[a_idx] * seasonality[day_idx])
                * (1.0 + delta[a_idx] * macro_index[day_idx])
            )
            c_exp = max(c_exp, 0.0)

            conversions = rng.poisson(c_exp)
            funded_loans = rng.binomial(conversions, p_fund[a_idx])

            profit_gross = funded_loans * margin[a_idx]
            profit_net = profit_gross - s

            rows.append(
                {
                    "date": date,
                    "arm_id": arm["arm_id"],
                    "channel": arm["channel"],
                    "campaign": arm["campaign"],
                    "spend": s,
                    "expected_conversions": c_exp,
                    "conversions": conversions,
                    "funded_loans": funded_loans,
                    "profit_gross": profit_gross,
                    "profit_net": profit_net,
                }
            )

    ad_spend = pd.DataFrame(rows)
    return arms, macro, ad_spend

# Run indepenently for validation.
if __name__ == "__main__":
    arms_df, macro_df, ad_spend_df = generate_synthetic_spend_world()
    print("Arms (with true parameters):")
    print(arms_df)
    print(arms_df.columns.tolist())
    print("\nMacro index head:")
    print(macro_df.head())
    print("\nAd spend head:")
    print(ad_spend_df.head())
    print(ad_spend_df.columns.tolist())

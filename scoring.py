import pandas as pd
import numpy as np

class FundScorer:
    """Normalizes metrics and computes the final weighted score."""
    
    @staticmethod
    def calculate_scores(
        funds_data: pd.DataFrame,
        horizon: str = "long",
        aum_bonus_threshold: float = 25000.0,
        aum_bonus_points: float = 2.0,
    ) -> pd.DataFrame:
        """
        Hybrid Scoring: Combines relative percentile ranking with absolute performance floors.
        Horizon-Aware: Adjustable split between relative and absolute logic.
        Category-Aware: Quality floors adjust based on fund category (Equity vs Debt).
        """
        if funds_data.empty:
            return funds_data

        import logging
        logger = logging.getLogger("Scorer")
        df = funds_data.copy()

        # Defensive guardrails against pathological metrics from sparse/bad histories.
        sanity_bounds = {
            "cagr": (-0.95, 1.00),
            "alpha": (-0.80, 0.50),
            "sortino_ratio": (-10.0, 10.0),
            "sharpe_ratio": (-10.0, 10.0),
            "volatility": (0.0, 1.50),
            "max_drawdown": (-1.0, 0.0),
            "calmar_ratio": (-10.0, 10.0),
            "rolling_1y_positive_ratio": (0.0, 1.0),
            "rolling_cagr_consistency": (0.0, 1.0),
            "downside_deviation_stability": (0.0, 1.0),
            "return_variance_stability": (0.0, 1.0),
            "benchmark_outperformance_ratio": (0.0, 1.0),
            "excess_cagr_vs_benchmark": (-1.0, 1.0),
        }
        for metric, (low, high) in sanity_bounds.items():
            if metric in df.columns:
                df[metric] = df[metric].clip(lower=low, upper=high)

        # Category-specific caps for known implausible ranges in debt/hybrid schemes.
        if "broad_category" in df.columns:
            debt_like = df["broad_category"].isin(["Debt", "Hybrid"])
            if "cagr" in df.columns:
                df.loc[debt_like, "cagr"] = df.loc[debt_like, "cagr"].clip(upper=0.25)
            if "alpha" in df.columns:
                df.loc[debt_like, "alpha"] = df.loc[debt_like, "alpha"].clip(upper=0.20)
            if "sortino_ratio" in df.columns:
                df.loc[debt_like, "sortino_ratio"] = df.loc[debt_like, "sortino_ratio"].clip(upper=5.0)
        
        # --- PHASE 1: CATEGORY-AWARE QUALITY FLOORS ---
        initial_count = len(df)
        
        def apply_floors(row):
            cat = row.get('broad_category', 'Equity')
            # 1. Alpha Floor: Strict for Equity, relaxed for Debt (due to benchmark mismatch potential)
            if cat == 'Equity':
                if row['alpha'] < -0.08: return False # Must not significantly underperform Nifty
            else:
                if row['alpha'] < -0.15: return False # Relaxed for Debt
            
            # 2. Sortino Floor: Must be positive for all, but stricter for Equity
            sortino = row.get('sortino_ratio', 0)
            if cat == 'Equity' and sortino < 0.2: return False
            if cat == 'Debt' and sortino < 0.0: return False # Just needs to be better than cash
            
            # 3. Max Drawdown Limit (Negative notation: -0.15 = 15% loss)
            mdd = row.get('max_drawdown', 0)
            if cat == 'Debt' and mdd < -0.15: return False # Reject if loss > 15%
            if cat == 'Equity' and mdd < -0.65: return False # Reject if loss > 65%

            # 4. Consistency floor: fraction of rolling 1Y windows with positive return.
            consistency = row.get('rolling_1y_positive_ratio', 0.0)
            if cat == 'Equity' and consistency < 0.45: return False
            if cat in ['Debt', 'Hybrid'] and consistency < 0.50: return False
            
            return True

        df = df[df.apply(apply_floors, axis=1)]
        
        filtered_count = len(df)
        if filtered_count < initial_count:
            logger.info(f"Hybrid Floor: Filtered out {initial_count - filtered_count} funds below quality threshold.")

        if df.empty:
            logger.warning("All funds failed absolute quality floors! Falling back to best available.")
            df = funds_data.copy()

        # --- PHASE 2: HORIZON-AWARE SPLIT ---
        # Short: More Absolute (Floor protection), Long: More Relative (Alpha chasing)
        if horizon == "short":
            hybrid_split = 0.4 # 40% Relative, 60% Absolute
            weights = {
                'cagr': 0.08, 'alpha': 0.08, 'sortino_ratio': 0.15,
                'volatility': -0.25, 'max_drawdown': -0.20, 'calmar_ratio': 0.14,
                'rolling_1y_positive_ratio': 0.10, 'rolling_cagr_consistency': 0.05,
                'downside_deviation_stability': 0.05, 'return_variance_stability': 0.05,
                'benchmark_outperformance_ratio': 0.05
            }
        elif horizon == "medium":
            hybrid_split = 0.6 # 60% Relative, 40% Absolute
            weights = {
                'cagr': 0.22, 'alpha': 0.14, 'sortino_ratio': 0.18,
                'volatility': -0.12, 'max_drawdown': -0.14, 'calmar_ratio': 0.10,
                'rolling_1y_positive_ratio': 0.10, 'rolling_cagr_consistency': 0.06,
                'downside_deviation_stability': 0.06, 'return_variance_stability': 0.06,
                'benchmark_outperformance_ratio': 0.05, 'excess_cagr_vs_benchmark': 0.03
            }
        else: # long
            hybrid_split = 0.8 # 80% Relative, 20% Absolute
            weights = {
                'cagr': 0.34, 'alpha': 0.18, 'sortino_ratio': 0.16,
                'volatility': -0.05, 'max_drawdown': -0.10, 'calmar_ratio': 0.07,
                'rolling_1y_positive_ratio': 0.10, 'rolling_cagr_consistency': 0.08,
                'downside_deviation_stability': 0.08, 'return_variance_stability': 0.08,
                'benchmark_outperformance_ratio': 0.07, 'excess_cagr_vs_benchmark': 0.04
            }
        # Debt is scored with a more risk/stability-heavy profile so it is less equity-biased.
        debt_weights_by_horizon = {
            "short": {
                'cagr': 0.04, 'alpha': 0.03, 'sortino_ratio': 0.12,
                'volatility': -0.24, 'max_drawdown': -0.20, 'calmar_ratio': 0.09,
                'rolling_1y_positive_ratio': 0.14, 'rolling_cagr_consistency': 0.10,
                'downside_deviation_stability': 0.10, 'return_variance_stability': 0.10,
                'benchmark_outperformance_ratio': 0.04
            },
            "medium": {
                'cagr': 0.10, 'alpha': 0.05, 'sortino_ratio': 0.14,
                'volatility': -0.18, 'max_drawdown': -0.16, 'calmar_ratio': 0.09,
                'rolling_1y_positive_ratio': 0.12, 'rolling_cagr_consistency': 0.10,
                'downside_deviation_stability': 0.10, 'return_variance_stability': 0.10,
                'benchmark_outperformance_ratio': 0.04, 'excess_cagr_vs_benchmark': 0.02
            },
            "long": {
                'cagr': 0.12, 'alpha': 0.04, 'sortino_ratio': 0.12,
                'volatility': -0.15, 'max_drawdown': -0.15, 'calmar_ratio': 0.08,
                'rolling_1y_positive_ratio': 0.12, 'rolling_cagr_consistency': 0.10,
                'downside_deviation_stability': 0.10, 'return_variance_stability': 0.10,
                'benchmark_outperformance_ratio': 0.05, 'excess_cagr_vs_benchmark': 0.03
            },
        }

        # Relative Component
        def robust_normalize(series, ascending=True):
            if series.max() == series.min(): return series * 0.0 + 0.5
            lower, upper = series.quantile(0.05), series.quantile(0.95)
            clipped = series.clip(lower, upper)
            norm = (clipped - clipped.min()) / (clipped.max() - clipped.min()) if clipped.max() != clipped.min() else series * 0.0 + 0.5
            return norm if ascending else (1.0 - norm)

        # Relative scores are computed inside each broad category so debt/hybrid/equity
        # do not directly outrank each other on incomparable distributions.
        relative_score = pd.Series(0.0, index=df.index)
        if "broad_category" not in df.columns:
            df["broad_category"] = "Other"

        for cat, cat_group in df.groupby("broad_category"):
            cat_weights = debt_weights_by_horizon.get(horizon, debt_weights_by_horizon["long"]) if cat == "Debt" else weights
            group_score = pd.Series(0.0, index=cat_group.index)
            for metric, weight in cat_weights.items():
                if metric in df.columns:
                    norm_val = robust_normalize(cat_group[metric], ascending=(weight > 0))
                    group_score += abs(weight) * norm_val
            relative_score.loc[cat_group.index] = group_score

        # Absolute Component (Milestones) - Category Aware
        absolute_bonus = 0.0
        
        def calculate_bonus(row):
            bonus = 0.0
            cat = row.get('broad_category', 'Equity')
            cagr = row['cagr']
            alpha = row['alpha']
            vol = row.get('volatility', 0.5)
            
            # CAGR Milestones (Category-specific)
            if cat == 'Equity':
                bonus += 0.50 * (1.0 if cagr > 0.18 else (0.5 if cagr > 0.12 else 0.0))
            else: # Debt/Hybrid
                bonus += 0.50 * (1.0 if cagr > 0.10 else (0.5 if cagr > 0.07 else 0.0))
            
            # Alpha Milestones
            bonus += 0.30 * (1.0 if alpha > 0.05 else (0.5 if alpha > 0.02 else 0.0))
            
            # Volatility Milestone
            if cat == 'Equity':
                bonus += 0.20 * (1.0 if vol < 0.14 else (0.5 if vol < 0.20 else 0.0))
            else:
                bonus += 0.20 * (1.0 if vol < 0.05 else (0.5 if vol < 0.08 else 0.0))
            consistency = row.get('rolling_1y_positive_ratio', 0.0)
            bonus += 0.20 * (1.0 if consistency > 0.70 else (0.5 if consistency > 0.55 else 0.0))
            return bonus

        absolute_bonus = df.apply(calculate_bonus, axis=1)

        # Final weighted raw score before per-category scaling.
        df['raw_score'] = (relative_score * hybrid_split) + (absolute_bonus * (1 - hybrid_split))

        # Scale inside each broad category so funds compete against peers, not all funds.
        df['score_within_category'] = 50.0
        for _, cat_group in df.groupby('broad_category'):
            group_scores = cat_group['raw_score']
            if group_scores.max() != group_scores.min():
                scaled = (group_scores - group_scores.min()) / (group_scores.max() - group_scores.min()) * 100
            else:
                scaled = pd.Series(50.0, index=cat_group.index)
            df.loc[cat_group.index, 'score_within_category'] = scaled

        # Global selection score used only for mixed-category leaderboard ordering.
        category_multiplier = {"Equity": 1.00, "Hybrid": 0.92, "Debt": 0.82}
        df['score_base'] = df.apply(
            lambda row: row['score_within_category'] * category_multiplier.get(row.get('broad_category'), 0.90),
            axis=1
        )
        if "aum_cr" in df.columns:
            aum_series = pd.to_numeric(df["aum_cr"], errors="coerce")
            eligible = aum_series > float(aum_bonus_threshold)
            df["aum_bonus"] = 0.0
            if eligible.any():
                eligible_log = np.log1p(aum_series[eligible].clip(lower=0))
                if eligible_log.max() != eligible_log.min():
                    scaled = (eligible_log - eligible_log.min()) / (eligible_log.max() - eligible_log.min())
                else:
                    scaled = pd.Series(1.0, index=eligible_log.index)
                df.loc[eligible, "aum_bonus"] = np.clip(scaled * float(aum_bonus_points), 0.0, float(aum_bonus_points))
        else:
            df["aum_bonus"] = 0.0
        df['score_pre_decay'] = df['score_base'] + df['aum_bonus']

        # Progressive score decay for repeated AMCs in top ranks.
        scored_sorted = df.sort_values("score_pre_decay", ascending=False).copy()
        amc_seen = {}
        cat_seen = {}
        decayed_scores = []
        for _, row in scored_sorted.iterrows():
            amc = str(row.get("fund_house", "Unknown"))
            subcat = str(row.get("scheme_category", "Unknown"))
            amc_count = amc_seen.get(amc, 0)
            cat_count = cat_seen.get(subcat, 0)
            amc_decay = min(0.08 * amc_count, 0.32)
            cat_decay = min(0.04 * cat_count, 0.20)
            final_decay = max(0.0, 1.0 - amc_decay - cat_decay)
            decayed_scores.append(float(row["score_pre_decay"]) * final_decay)
            amc_seen[amc] = amc_count + 1
            cat_seen[subcat] = cat_count + 1
        scored_sorted["score"] = decayed_scores
        df = scored_sorted

        return df.sort_values('score', ascending=False)

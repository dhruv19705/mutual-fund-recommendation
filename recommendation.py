import pandas as pd
import logging

logger = logging.getLogger("Recommendation")

class RecommendationEngine:
    """Generates portfolio recommendations based on risk profiles."""

    RISK_PROFILES = {
        "low": {"Equity": 0.10, "Hybrid": 0.40, "Debt": 0.50},
        "medium": {"Equity": 0.50, "Hybrid": 0.30, "Debt": 0.20},
        "high": {"Equity": 0.80, "Hybrid": 0.15, "Debt": 0.05}
    }

    # Categories to exclude based on risk profile
    EXCLUDED_CATEGORIES = {
        "low": ["Small Cap", "Mid Cap", "Sectoral", "Thematic"],
        "medium": ["Sectoral", "Thematic"],
        "high": ["Liquid", "Overnight", "Money Market"]
    }

    AMC_CAP = 1.0 # Reserved for future hard-cap re-enable.
    SOFT_AMC_CAP = 1.0 # Reserved for future soft-cap re-enable.
    CATEGORY_CAP = 0.45 # No more than 45% in any single sub-category
    MIN_ALLOC_FLOOR = 0.01  # Drop allocations below 1%
    TARGET_TOTAL_ALLOCATION = 1.0

    @staticmethod
    def categorize_fund(category_str: str) -> str:
        """Maps SEBI/MFAPI category strings to broad types."""
        cat = str(category_str).lower()

        debt_keywords = [
            "debt", "bond", "gilt", "liquid", "money market", "overnight",
            "corporate bond", "credit risk", "banking and psu", "banking & psu",
            "floater", "floating rate", "dynamic bond", "medium duration",
            "medium to long duration", "long duration", "short duration",
            "ultra short", "low duration", "short term", "income", "fixed maturity"
        ]
        hybrid_keywords = [
            "hybrid", "balanced", "aggressive hybrid", "conservative hybrid",
            "dynamic asset allocation", "balanced advantage", "equity savings",
            "multi asset", "arbitrage"
        ]
        equity_keywords = [
            "equity", "large cap", "mid cap", "small cap", "large and mid cap",
            "multicap", "multi cap", "flexi cap", "focused", "value", "contra",
            "elss", "dividend yield", "thematic", "sectoral", "index", "index fund"
        ]

        # Debt first to avoid broad equity/hybrid keywords overshadowing debt labels.
        if any(x in cat for x in debt_keywords):
            return "Debt"
        if any(x in cat for x in hybrid_keywords):
            return "Hybrid"
        if any(x in cat for x in equity_keywords):
            return "Equity"
        return "Other"

    @staticmethod
    def recommend_portfolio(scored_funds: pd.DataFrame, risk_appetite: str, top_n: int = 3) -> dict:
        """
        Input: Scored funds with 'broad_category', 'score', and 'fund_house'
        Output: Dictionary with allocations and selected funds
        """
        risk = risk_appetite.lower()
        if risk not in RecommendationEngine.RISK_PROFILES:
            risk = "medium"

        allocation_plan = RecommendationEngine.RISK_PROFILES[risk]
        excluded = RecommendationEngine.EXCLUDED_CATEGORIES[risk]
        
        portfolio = []
        amc_exposure = {}
        category_exposure = {} # Track it like amc_exposure
        
        for category, target_weight in allocation_plan.items():
            if target_weight <= 0:
                continue

            # Filtering:
            # 1. Broad category match
            # 2. Risk-based exclusion
            cat_funds = scored_funds[scored_funds['broad_category'] == category]
            for ex_keyword in excluded:
                cat_funds = cat_funds[~cat_funds['scheme_category'].astype(str).str.contains(ex_keyword, case=False)]
            
            if cat_funds.empty:
                logger.warning(f"No suitable funds found for category: {category}")
                continue
            
            # Keep modest buffer for min-allocation and category constraints.
            effective_top_n = top_n + 2
            
            selected_in_cat = []
            cat_funds = cat_funds.sort_values('score', ascending=False)
            
            for _, fund in cat_funds.iterrows():
                if len(selected_in_cat) >= effective_top_n:
                    break
                    
                sub_cat = str(fund.get('scheme_category', ''))
                projected_fund_share = (target_weight / effective_top_n)

                if category_exposure.get(sub_cat, 0) + projected_fund_share > RecommendationEngine.CATEGORY_CAP:
                    logger.info(
                        "Skipping %s - Category cap reached for %s",
                        fund.get('scheme_name', fund.get('schemeName', 'Unknown')),
                        sub_cat,
                    )
                    continue

                selected_in_cat.append(fund.to_dict())
                category_exposure[sub_cat] = category_exposure.get(sub_cat, 0) + projected_fund_share

            if not selected_in_cat:
                logger.warning(f"Could not fulfill target for {category} due to category constraints.")
                continue

            # Score -> Weight mapping (explicit):
            # 1) Convert each score to non-negative score_power.
            # 2) Normalize score_power to score_share.
            # 3) Multiply by category target weight.
            score_floor = 0.0
            score_shift = min(f['score'] for f in selected_in_cat)
            score_powers = [
                max((f['score'] - score_shift) + 1.0, score_floor)
                for f in selected_in_cat
            ]
            total_score_power = sum(score_powers)
            
            for idx, fund in enumerate(selected_in_cat):
                score_power = score_powers[idx]
                score_share = (
                    score_power / total_score_power
                    if total_score_power > 0
                    else (1 / len(selected_in_cat))
                )
                fund_weight = score_share * target_weight
                amc = fund.get('fund_house') or 'Unknown'
                if fund_weight < RecommendationEngine.MIN_ALLOC_FLOOR:
                    logger.info(
                        f"Skipping {fund.get('scheme_name', 'Unknown')} - below min allocation floor ({fund_weight:.2%})"
                    )
                    continue

                portfolio.append({
                    "scheme_code": fund['scheme_code'],
                    "scheme_name": fund.get('scheme_name', fund.get('schemeName', 'Unknown')),
                    "fund_house": fund.get('fund_house') or 'Unknown',
                    "category": fund['scheme_category'],
                    "broad_category": category,
                    "weight": fund_weight,
                    "score": fund['score'],
                    "score_share": score_share,
                    "score_power": score_power,
                    "cagr": fund['cagr'],
                    "alpha": fund['alpha'],
                    "beta": fund.get('beta', 1.0)
                })
                amc_exposure[amc] = amc_exposure.get(amc, 0) + fund_weight

            # Residual redistribution pass:
            # Re-allocate leftover category weight to selected funds.
            cat_rows = [i for i, p in enumerate(portfolio) if p['broad_category'] == category]
            cat_allocated = sum(portfolio[i]['weight'] for i in cat_rows)
            residual = target_weight - cat_allocated
            if residual > 1e-6 and cat_rows:
                max_rounds = 5
                for _ in range(max_rounds):
                    if residual <= 1e-6:
                        break
                    eligible_rows = []
                    for i in cat_rows:
                        fund = portfolio[i]
                        eligible_rows.append((i, max(fund.get("score_power", 1.0), 1e-6), residual))

                    if not eligible_rows:
                        break

                    total_power = sum(power for _, power, _ in eligible_rows)
                    moved = 0.0
                    for i, power, headroom in eligible_rows:
                        share = power / total_power if total_power > 0 else 1 / len(eligible_rows)
                        add_weight = min(residual * share, headroom)  # headroom == residual in no-AMC-cap mode
                        if add_weight <= 1e-6:
                            continue
                        portfolio[i]["weight"] += add_weight
                        amc = portfolio[i].get("fund_house", "Unknown")
                        amc_exposure[amc] = amc_exposure.get(amc, 0) + add_weight
                        moved += add_weight
                    residual -= moved
                    if moved <= 1e-6:
                        break

        # --- PHASE 4: CONSTRAINT RELAXATION (If target not met) ---
        # Calculate how much weight we actually managed to allocate
        total_allocated = sum(f['weight'] for f in portfolio)
        
        for category, target_weight in allocation_plan.items():
            cat_allocated = sum(f['weight'] for f in portfolio if f['broad_category'] == category)
            if cat_allocated < target_weight - 0.01:
                logger.warning(f"Shortfall in {category}: Allocated {cat_allocated:.1%} vs Target {target_weight:.1%}")

        def allocate_residual(residual):
            """Distribute residual by score power across already-selected funds."""
            if residual <= 1e-9 or not portfolio:
                return 0.0
            moved_total = 0.0
            max_rounds = 8
            for _ in range(max_rounds):
                if residual <= 1e-9:
                    break
                eligible_rows = []
                for i, fund in enumerate(portfolio):
                    eligible_rows.append((i, max(fund.get("score_power", 1.0), 1e-6), residual))
                if not eligible_rows:
                    break
                total_power = sum(power for _, power, _ in eligible_rows)
                moved = 0.0
                for i, power, headroom in eligible_rows:
                    share = power / total_power if total_power > 0 else 1.0 / len(eligible_rows)
                    add_weight = min(residual * share, headroom)  # headroom == residual in no-AMC-cap mode
                    if add_weight <= 1e-9:
                        continue
                    portfolio[i]["weight"] += add_weight
                    amc = portfolio[i].get("fund_house", "Unknown")
                    amc_exposure[amc] = amc_exposure.get(amc, 0.0) + add_weight
                    moved += add_weight
                residual -= moved
                moved_total += moved
                if moved <= 1e-9:
                    break
            return moved_total

        # Step 1: fill residual across existing positions.
        if portfolio and total_allocated < RecommendationEngine.TARGET_TOTAL_ALLOCATION - 1e-6:
            residual = RecommendationEngine.TARGET_TOTAL_ALLOCATION - total_allocated
            moved = allocate_residual(residual)
            total_allocated += moved

        # Step 2: second pass in case of tiny per-round leftovers.
        if portfolio and total_allocated < RecommendationEngine.TARGET_TOTAL_ALLOCATION - 1e-6:
            residual = RecommendationEngine.TARGET_TOTAL_ALLOCATION - total_allocated
            moved = allocate_residual(residual)
            total_allocated += moved

        # Step 3 (last resort): proportional normalization to reach 100% allocation.
        if portfolio and total_allocated < RecommendationEngine.TARGET_TOTAL_ALLOCATION - 1e-6 and total_allocated > 1e-9:
            pre_scale_alloc = total_allocated
            scale = RecommendationEngine.TARGET_TOTAL_ALLOCATION / pre_scale_alloc
            for fund in portfolio:
                fund["weight"] *= scale
            amc_exposure = {}
            for fund in portfolio:
                amc = fund.get("fund_house", "Unknown")
                amc_exposure[amc] = amc_exposure.get(amc, 0.0) + fund["weight"]
            total_allocated = sum(f['weight'] for f in portfolio)
            logger.warning(
                "Constraint relaxation: Final normalization scaled %.1f%% -> 100.0%%.",
                pre_scale_alloc * 100
            )

        return {
            "risk_profile": risk,
            "allocation": portfolio,
            "summary": {
                "total_funds": len(portfolio),
                "allocated_weight": total_allocated,
                "amc_distribution": amc_exposure,
                "target_allocation": allocation_plan
            }
        }

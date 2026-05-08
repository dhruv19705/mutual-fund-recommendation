import logging
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from data_fetcher import HybridFetcher, MFAPIFetcher, YahooFetcher
from data_validator import DataValidator, CacheManager
from aum_collector import build_aum_lookup, build_universe, lookup_aum_for_fund
from feature_engineering import QuantitativeMetrics
from scoring import FundScorer
from recommendation import RecommendationEngine
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MFEngineMain")

CLOSED_ENDED_NAME_KEYWORDS = [
    "debt fund series",
    "series",
    "fixed term",
    "fixed maturity",
    "fmp",
    "fixed interval",
    "close ended",
    "closed ended",
    "maturity fund",
]

AUM_FILTER_THRESHOLD_CR = 15000.0
DEBT_AUM_FILTER_THRESHOLD_CR = 5000.0
AUM_BONUS_THRESHOLD_CR = 25000.0
AUM_MAX_BONUS_POINTS = 10.0


def _normalize_scheme_name_for_match(text: str) -> str:
    normalized = str(text).lower()
    for token in [
        "direct plan",
        "regular plan",
        "growth option",
        "growth",
        "idcw",
        "dividend",
        "-",
        "(",
        ")",
    ]:
        normalized = normalized.replace(token, " ")
    return " ".join(normalized.split())


def _resolve_code_from_name_index(
    target_name: str,
    normalized_name_to_code: dict,
    scheme_name_lookup: dict,
) -> str | None:
    """
    Resolve MFAPI scheme code from a normalized-but-not-exact scheme name.
    This helps bridge ETMoney naming variants to MFAPI names.
    """
    target_norm = _normalize_scheme_name_for_match(target_name)
    if not target_norm:
        return None

    target_tokens = set(target_norm.split())
    if len(target_tokens) < 3:
        return None

    # Prioritize exact direct+growth matches with high token overlap.
    best_direct_growth = None
    best_overall = None
    for candidate_norm, candidate_code in normalized_name_to_code.items():
        candidate_tokens = set(str(candidate_norm).split())
        if not candidate_tokens:
            continue

        # Guardrail: keep AMC-family alignment by requiring first token match.
        target_first = target_norm.split()[0]
        candidate_first = str(candidate_norm).split()[0]
        if target_first != candidate_first:
            continue

        overlap = len(target_tokens.intersection(candidate_tokens))
        overlap_ratio = overlap / max(len(target_tokens), 1)
        if overlap_ratio < 0.70:
            continue

        similarity = SequenceMatcher(None, target_norm, str(candidate_norm)).ratio()
        score = (0.65 * overlap_ratio) + (0.35 * similarity)
        candidate_name = str(scheme_name_lookup.get(str(candidate_code), "")).lower()
        is_direct_growth = "direct" in candidate_name and "growth" in candidate_name

        choice = (score, str(candidate_code))
        if is_direct_growth and (best_direct_growth is None or score > best_direct_growth[0]):
            best_direct_growth = choice
        if best_overall is None or score > best_overall[0]:
            best_overall = choice

    if best_direct_growth and best_direct_growth[0] >= 0.80:
        return best_direct_growth[1]
    if best_overall and best_overall[0] >= 0.86:
        return best_overall[1]
    return None


def apply_aum_eligibility_filter(
    funds_df: pd.DataFrame,
    threshold_cr: float = AUM_FILTER_THRESHOLD_CR,
    debt_threshold_cr: float = DEBT_AUM_FILTER_THRESHOLD_CR,
) -> tuple[pd.DataFrame, dict]:
    """Apply strict category-aware AUM filter."""
    if funds_df.empty or "aum_cr" not in funds_df.columns:
        return funds_df, {
            "aum_filter_applied": False,
            "aum_filter_bypassed": False,
            "aum_filter_reason": "missing_aum_column_or_empty",
            "candidates_before_filter": int(len(funds_df)),
            "excluded_for_low_or_missing_aum": 0,
            "survivors_after_filter": int(len(funds_df)),
            "threshold_15000_pass_count": 0,
            "threshold_25000_pass_count": 0,
        }

    before_count = int(len(funds_df))
    aum_series = pd.to_numeric(funds_df["aum_cr"], errors="coerce")
    is_debt = funds_df.get("broad_category", pd.Series(index=funds_df.index, dtype=str)).astype(str).eq("Debt")
    debt_before = int(is_debt.sum())
    non_debt_before = int((~is_debt).sum())
    debt_pass_mask = is_debt & (aum_series > float(debt_threshold_cr))
    non_debt_pass_mask = (~is_debt) & (aum_series > float(threshold_cr))
    eligible_mask = debt_pass_mask | non_debt_pass_mask
    survivors_df = funds_df.loc[eligible_mask].copy()
    survivors_count = int(len(survivors_df))
    excluded_count = before_count - survivors_count
    debt_survivors = int((survivors_df.get("broad_category", pd.Series(dtype=str)).astype(str) == "Debt").sum())
    non_debt_survivors = int(survivors_count - debt_survivors)

    diagnostics = {
        "aum_filter_applied": True,
        "aum_filter_bypassed": False,
        "aum_filter_reason": "strict_applied",
        "candidates_before_filter": before_count,
        "excluded_for_low_or_missing_aum": excluded_count,
        "survivors_after_filter": survivors_count,
        "threshold_non_debt_15000_pass_count": int(non_debt_pass_mask.sum()),
        "threshold_debt_5000_pass_count": int(debt_pass_mask.sum()),
        "threshold_25000_pass_count": int((pd.to_numeric(survivors_df["aum_cr"], errors="coerce") > AUM_BONUS_THRESHOLD_CR).sum()),
        "debt_before_filter": debt_before,
        "non_debt_before_filter": non_debt_before,
        "debt_after_filter": debt_survivors,
        "non_debt_after_filter": non_debt_survivors,
    }
    return survivors_df, diagnostics


def run_engine(risk_appetite="medium", horizon="long"):
    """Main execution pipe with professional metrics and constraints."""
    
    fetcher = HybridFetcher()
    validator = DataValidator()
    cache = CacheManager()
    engine = RecommendationEngine()
    
    # 1. Fetch Benchmarks
    logger.info(f"Initializing Engine | Risk: {risk_appetite}, Horizon: {horizon}...")
    
    BENCHMARK_MAP = {
        "Equity - Large Cap":   "^NSEI",
        "Equity - Mid Cap":     "NIFTYMIDCAP150.NS",
        "Equity - Small Cap":   "^NSEI",
        "Equity - Flexi Cap":   "^NSEI",
        "Debt - Gilt":          "^IRX",        # Proxy — use with caution
        "Debt - Liquid":        None,          # Skip benchmark for liquid funds
        "Hybrid - Aggressive":  "^NSEI",
        "Hybrid - Conservative":"^NSEI",
    }
    
    # Pre-fetch common benchmarks
    benchmark_data = {}
    for ticker in set(BENCHMARK_MAP.values()):
        if ticker:
            df = YahooFetcher.fetch_benchmark_history(ticker)
            if df is not None:
                benchmark_data[ticker] = validator.clean_data(df)
    
    # 2. Define universe (AUM collector combined universe)
    logger.info("Fund Discovery...")
    combined_universe_df = build_universe(source_mode="combined")
    aum_lookup = build_aum_lookup(universe_source="combined", use_fast_sources=False)
    logger.info(
        "AUM lookup ready: universe=%s matched=%s coverage=%.1f%%",
        aum_lookup.universe_size,
        aum_lookup.matched_count,
        aum_lookup.coverage * 100,
    )
    combined_universe_df["aum_cr"] = combined_universe_df.apply(
        lambda row: lookup_aum_for_fund(
            scheme_code=str(row.get("scheme_code", "")),
            scheme_name=str(row.get("scheme_name", "")),
            lookup=aum_lookup,
        ).aum_cr,
        axis=1,
    )
    combined_universe_df["aum_cr"] = pd.to_numeric(combined_universe_df["aum_cr"], errors="coerce")
    high_aum_universe_df = combined_universe_df.loc[combined_universe_df["aum_cr"] > AUM_FILTER_THRESHOLD_CR].copy()

    all_funds = MFAPIFetcher.fetch_all_funds()
    mfapi_name_to_code = {}
    scheme_name_lookup = {}
    for f in all_funds:
        code = str(f.get("schemeCode", "")).strip()
        name = str(f.get("schemeName", "")).strip()
        if not code or not name:
            continue
        scheme_name_lookup[code] = name
        mfapi_name_to_code[_normalize_scheme_name_for_match(name)] = code

    fund_universe = []
    unmapped_high_aum = 0
    for _, row in high_aum_universe_df.iterrows():
        raw_code = str(row.get("scheme_code", "")).strip()
        name = str(row.get("scheme_name", "")).strip()
        if not name:
            continue
        name_lower = name.lower()
        if any(kw in name_lower for kw in CLOSED_ENDED_NAME_KEYWORDS):
            continue

        resolved_code = None
        if raw_code.isdigit():
            resolved_code = raw_code
        else:
            norm_name = _normalize_scheme_name_for_match(name)
            resolved_code = mfapi_name_to_code.get(norm_name)
            if not resolved_code:
                resolved_code = _resolve_code_from_name_index(
                    target_name=name,
                    normalized_name_to_code=mfapi_name_to_code,
                    scheme_name_lookup=scheme_name_lookup,
                )
            if not resolved_code:
                search_hits = MFAPIFetcher.search_funds(name)
                for hit in search_hits:
                    candidate_code = str(hit.get("schemeCode", "")).strip()
                    candidate_name = str(hit.get("schemeName", "")).strip()
                    if not candidate_code or not candidate_name:
                        continue
                    if "direct" in candidate_name.lower() and "growth" in candidate_name.lower():
                        resolved_code = candidate_code
                        scheme_name_lookup[candidate_code] = candidate_name
                        mfapi_name_to_code[_normalize_scheme_name_for_match(candidate_name)] = candidate_code
                        break
                if not resolved_code and search_hits:
                    top_hit = search_hits[0]
                    candidate_code = str(top_hit.get("schemeCode", "")).strip()
                    candidate_name = str(top_hit.get("schemeName", "")).strip()
                    if candidate_code and candidate_name:
                        resolved_code = candidate_code
                        scheme_name_lookup[candidate_code] = candidate_name
                        mfapi_name_to_code[_normalize_scheme_name_for_match(candidate_name)] = candidate_code

        if not resolved_code:
            unmapped_high_aum += 1
            continue

        resolved_name = scheme_name_lookup.get(resolved_code, name)
        fund_universe.append({"schemeCode": resolved_code, "schemeName": resolved_name})

    # Deduplicate by scheme code after ETMoney->MFAPI resolution.
    unique_funds = {}
    for item in fund_universe:
        unique_funds[str(item["schemeCode"])] = item
    fund_universe = list(unique_funds.values())

    logger.info(
        "Fund Discovery: combined=%s | high_aum_gt_%s=%s | nav_fetch_candidates=%s | unresolved_high_aum=%s",
        len(combined_universe_df),
        int(AUM_FILTER_THRESHOLD_CR),
        len(high_aum_universe_df),
        len(fund_universe),
        unmapped_high_aum,
    )

    # 3. Data Fetching (Parallel with Cache logic)
    processed_funds = []
    min_yrs = 1.0 if horizon in {"short", "long"} else 3.0
    
    # Check cache first to see what we actually need to fetch
    to_fetch = []
    funds_data = {} # code -> {nav, meta}
    
    for entry in fund_universe:
        code = str(entry['schemeCode'])
        cached = cache.get(code)
        if cached is not None and isinstance(cached, dict) and "nav" in cached:
            funds_data[code] = cached
        else:
            to_fetch.append(entry)

    if to_fetch:
        logger.info(f"Executing parallel fetch for {len(to_fetch)} funds...")
        fetched_dfs = fetcher.fetch_batch(to_fetch)
        
        for entry in to_fetch:
            code = str(entry['schemeCode'])
            df = fetched_dfs.get(code)
            if df is not None:
                # Also need meta
                meta_raw = MFAPIFetcher.fetch_nav_history(code)
                if meta_raw and "meta" in meta_raw:
                    meta = meta_raw['meta']
                    funds_data[code] = {"nav": df, "meta": meta}
                    cache.set(code, funds_data[code])

    logger.info(f"Analyzing {len(funds_data)} candidates (Min History: {min_yrs}y)...")
    
    stale_data_skipped = 0
    other_category_skipped = 0
    for code, data in funds_data.items():
        df = data['nav']
        meta = data['meta']
        name = (
            meta.get('scheme_name')
            or meta.get('schemeName')
            or scheme_name_lookup.get(code)
            or 'Unknown'
        )
        
        df = validator.clean_data(df)
        
        latest_date = df['date'].max()
        days_stale = (pd.Timestamp.now().normalize() - latest_date.normalize()).days
        if days_stale > 180:
            stale_data_skipped += 1
            continue

        # History and quality checks in validator
        if validator.validate_history(df, min_years=min_yrs, max_staleness_days=180):
            category = (
                meta.get('scheme_category')
                or meta.get('schemeCategory')
                or meta.get('scheme_type')
                or meta.get('schemeType')
                or name
                or 'Other'
            )
            fund_house = meta.get('fund_house') or meta.get('fundHouse')
            if not fund_house:
                # Keep candidate even when AMC metadata is unavailable.
                fund_house = "Unknown AMC"
            broad_cat = engine.categorize_fund(category)
            if broad_cat == "Other":
                other_category_skipped += 1
                continue
            
            benchmark_ticker = BENCHMARK_MAP.get(category, "^NSEI")
            bench_df = benchmark_data.get(benchmark_ticker) if benchmark_ticker else None
            
            metrics = QuantitativeMetrics.calculate_metrics(df, bench_df)
            
            if metrics:
                aum_match = lookup_aum_for_fund(code, name, aum_lookup)
                metrics.update({
                    "scheme_code": code,
                    "scheme_name": name,
                    "fund_house": fund_house,
                    "scheme_category": category,
                    "broad_category": broad_cat,
                    "aum_cr": float(aum_match.aum_cr) if aum_match.aum_cr is not None else np.nan,
                    "aum_source": aum_match.source,
                    "aum_matched": bool(aum_match.aum_cr is not None),
                })
                processed_funds.append(metrics)

    if stale_data_skipped:
        logger.info(f"Hard-excluded {stale_data_skipped} funds with NAV staleness > 180 days.")
    if other_category_skipped:
        logger.info(f"Excluded {other_category_skipped} funds categorized as Other.")

    if not processed_funds:
        logger.error("Analysis failed: No funds met the history or quality criteria.")
        return None, None

    # 3. Scoring (Horizon-Driven weights)
    funds_df = pd.DataFrame(processed_funds)
    metrics_required = [
        "cagr", "alpha", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "rolling_1y_positive_ratio", "volatility"
    ]
    existing_metrics = [m for m in metrics_required if m in funds_df.columns]
    if existing_metrics:
        invalid_mask = ~np.isfinite(funds_df[existing_metrics]).all(axis=1)
        invalid_funds = funds_df.loc[invalid_mask, "scheme_name"].astype(str).tolist()
        if invalid_funds:
            logger.info(
                "Excluded %s funds with NaN/inf metrics pre-scoring: %s",
                len(invalid_funds),
                ", ".join(invalid_funds[:25]) + (" ..." if len(invalid_funds) > 25 else "")
            )
        funds_df = funds_df.loc[~invalid_mask].copy()

    if funds_df.empty:
        logger.error("Analysis failed: All funds excluded due to invalid pre-scoring metrics.")
        return None, None

    funds_df, aum_filter_diag = apply_aum_eligibility_filter(
        funds_df,
        threshold_cr=AUM_FILTER_THRESHOLD_CR,
        debt_threshold_cr=DEBT_AUM_FILTER_THRESHOLD_CR,
    )
    logger.info(
        "AUM filter diagnostics | applied=%s bypassed=%s before=%s excluded=%s survivors=%s debt_after=%s non_debt_after=%s >25000=%s",
        aum_filter_diag["aum_filter_applied"],
        aum_filter_diag["aum_filter_bypassed"],
        aum_filter_diag["candidates_before_filter"],
        aum_filter_diag["excluded_for_low_or_missing_aum"],
        aum_filter_diag["survivors_after_filter"],
        aum_filter_diag["debt_after_filter"],
        aum_filter_diag["non_debt_after_filter"],
        aum_filter_diag["threshold_25000_pass_count"],
    )

    scored_df = FundScorer.calculate_scores(
        funds_df,
        horizon=horizon,
        aum_bonus_threshold=AUM_BONUS_THRESHOLD_CR,
        aum_bonus_points=AUM_MAX_BONUS_POINTS,
    )
    
    # 4. Recommendation (diversified with 25% AMC cap)
    recommendation = engine.recommend_portfolio(scored_df, risk_appetite, top_n=5)
    if recommendation is not None:
        recommendation.setdefault("summary", {})
        recommendation["summary"]["aum_diagnostics"] = {
            **aum_filter_diag,
            "aum_lookup_coverage": aum_lookup.coverage,
            "aum_lookup_matched_count": aum_lookup.matched_count,
            "aum_lookup_universe_size": aum_lookup.universe_size,
            "aum_source_hit_counts": aum_lookup.source_hit_counts,
            "aum_filter_threshold_cr_non_debt": AUM_FILTER_THRESHOLD_CR,
            "aum_filter_threshold_cr_debt": DEBT_AUM_FILTER_THRESHOLD_CR,
            "aum_bonus_threshold_cr": AUM_BONUS_THRESHOLD_CR,
            "aum_bonus_points": AUM_MAX_BONUS_POINTS,
        }
    
    return scored_df, recommendation

def display_results(scored_df, recommendation):
    if scored_df is None or scored_df.empty:
        print("\n[!] No scored funds available.")
        return

    risk_profile = (recommendation or {}).get("risk_profile", "n/a").upper()
    print("\n" + "=" * 240)
    print(f"TOP 20 MUTUAL FUND SUGGESTIONS | RISK: {risk_profile}")
    print("=" * 240)

    headers = (
        f"{'Rank':<4} | {'Scheme Name':<45} | {'Category':<20} | {'Broad':<8} | {'AUM Cr':>10} | "
        f"{'CAGR':>7} | {'Alpha':>7} | {'Sortino':>8} | {'Vol':>6} | {'MDD':>7} | {'Calmar':>7} | "
        f"{'1Y+%':>6} | {'StbCAGR':>8} | {'StbDown':>8} | {'StbVar':>7} | {'ExCAGR':>7} | "
        f"{'Outperf':>8} | {'AUMBon':>7} | {'Score':>7}"
    )
    print(headers)
    print("-" * 240)

    scored_df = scored_df.sort_values('score', ascending=False).reset_index(drop=True)
    top20 = scored_df.head(20).copy()
    for i, row in top20.iterrows():
        name = str(row.get('scheme_name', 'N/A'))[:43]
        category = str(row.get("scheme_category", "N/A"))[:18]
        print(
            f"{i+1:<4} | {name:<45} | {category:<20} | {str(row.get('broad_category','N/A')):<8} | "
            f"{float(row.get('aum_cr', np.nan)):>10.1f} | {float(row.get('cagr',0)):>7.1%} | {float(row.get('alpha',0)):>7.1%} | "
            f"{float(row.get('sortino_ratio',0)):>8.2f} | {float(row.get('volatility',0)):>6.1%} | {float(row.get('max_drawdown',0)):>7.1%} | "
            f"{float(row.get('calmar_ratio',0)):>7.2f} | {float(row.get('rolling_1y_positive_ratio',0)):>6.1%} | "
            f"{float(row.get('rolling_cagr_consistency',0)):>8.3f} | {float(row.get('downside_deviation_stability',0)):>8.3f} | "
            f"{float(row.get('return_variance_stability',0)):>7.3f} | {float(row.get('excess_cagr_vs_benchmark',0)):>7.2%} | "
            f"{float(row.get('benchmark_outperformance_ratio',0)):>8.1%} | {float(row.get('aum_bonus',0)):>7.2f} | {float(row.get('score',0)):>7.2f}"
        )

    print("\n" + "=" * 140)
    print("TOP FUNDS BY CATEGORY")
    print("=" * 140)
    category_top5_output = {}
    for broad in ["Equity", "Hybrid", "Debt", "Other"]:
        cat_df = scored_df[scored_df["broad_category"].astype(str) == broad].copy()
        score_col = "score_within_category" if "score_within_category" in cat_df.columns else "score"
        cat_df = cat_df[cat_df[score_col] > 0].copy()
        cat_df = cat_df.sort_values(score_col, ascending=False).reset_index(drop=True).head(5)
        if cat_df.empty:
            continue
        category_top5_output[broad] = cat_df
        print(f"\n[{broad}] (showing {len(cat_df)} funds)")
        print(f"{'Rank':<4} | {'Scheme Name':<45} | {'AUM Cr':>10} | {'CatScore':>8}")
        print("-" * 80)
        for rank, (_, row) in enumerate(cat_df.iterrows(), start=1):
            name = str(row.get("scheme_name", "N/A"))[:43]
            print(
                f"{rank:<4} | {name:<45} | {float(row.get('aum_cr', np.nan)):>10.1f} | {float(row.get(score_col, 0)):>8.2f}"
            )

    # 5. Save Output
    top_cols = [
        'scheme_name', 'scheme_category', 'broad_category', 'aum_cr', 'score', 'aum_bonus', 'cagr', 'alpha',
        'sortino_ratio', 'volatility', 'max_drawdown', 'calmar_ratio', 'rolling_1y_positive_ratio',
        'rolling_cagr_consistency', 'downside_deviation_stability', 'return_variance_stability',
        'excess_cagr_vs_benchmark', 'benchmark_outperformance_ratio'
    ]
    existing_top_cols = [c for c in top_cols if c in scored_df.columns]
    output = {
        "run_date": datetime.now().isoformat(),
        "risk": (recommendation or {}).get('risk_profile', 'n/a'),
        "top_funds_overall": scored_df.head(20)[existing_top_cols].to_dict('records'),
        "top_funds_by_broad_category": {
            broad: df[existing_top_cols].to_dict("records")
            for broad, df in category_top5_output.items()
        },
        "aum_diagnostics": (recommendation or {}).get("summary", {}).get("aum_diagnostics", {}),
    }
    filename = f"output_{recommendation['risk_profile']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved to %s", filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Professional Indian Mutual Fund Engine")
    parser.add_argument("--risk", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--horizon", choices=["short", "medium", "long"], default="long")
    
    args = parser.parse_args()
    
    print(f"--- INITIALIZING QUANTITATIVE ANALYSIS ---")
    scored_list, final_portfolio = run_engine(risk_appetite=args.risk, horizon=args.horizon)
    display_results(scored_list, final_portfolio)

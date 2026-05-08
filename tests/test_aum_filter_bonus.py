import unittest

import numpy as np
import pandas as pd

from aum_collector import AUMLookup, AUMMatch, lookup_aum_for_fund
from feature_engineering import QuantitativeMetrics
from main import apply_aum_eligibility_filter
from scoring import FundScorer


class TestAUMLookupJoin(unittest.TestCase):
    def test_lookup_prefers_scheme_code_then_name(self):
        lookup = AUMLookup(
            by_scheme_code={"123": AUMMatch(aum_cr=20000.0, source="amfi_average_aum")},
            by_norm_name={"fund alpha": AUMMatch(aum_cr=18000.0, source="etmoney")},
            coverage=0.5,
            source_hit_counts={"amfi_average_aum": 1},
            generated_at="2026-01-01T00:00:00",
            universe_size=2,
            matched_count=1,
        )
        match_code = lookup_aum_for_fund("123", "Fund Alpha", lookup)
        self.assertEqual(match_code.aum_cr, 20000.0)
        self.assertEqual(match_code.source, "amfi_average_aum")

        match_name = lookup_aum_for_fund("999", "Fund Alpha", lookup)
        self.assertEqual(match_name.aum_cr, 18000.0)
        self.assertEqual(match_name.source, "etmoney")


class TestAUMFilterAndFallback(unittest.TestCase):
    def test_filter_applied_when_survivors_enough(self):
        df = pd.DataFrame(
            {
                "scheme_code": [f"S{i}" for i in range(25)],
                "broad_category": ["Equity"] * 25,
                "aum_cr": [20000.0] * 22 + [12000.0, 14000.0, np.nan],
            }
        )
        filtered, diag = apply_aum_eligibility_filter(
            df,
            threshold_cr=15000.0,
            debt_threshold_cr=5000.0,
        )
        self.assertTrue(diag["aum_filter_applied"])
        self.assertFalse(diag["aum_filter_bypassed"])
        self.assertEqual(len(filtered), 22)
        self.assertEqual(diag["excluded_for_low_or_missing_aum"], 3)

    def test_filter_remains_strict_when_survivors_too_few(self):
        df = pd.DataFrame(
            {
                "scheme_code": [f"S{i}" for i in range(10)],
                "broad_category": ["Equity"] * 10,
                "aum_cr": [20000.0] * 5 + [12000.0] * 5,
            }
        )
        filtered, diag = apply_aum_eligibility_filter(
            df,
            threshold_cr=15000.0,
            debt_threshold_cr=5000.0,
        )
        self.assertTrue(diag["aum_filter_applied"])
        self.assertFalse(diag["aum_filter_bypassed"])
        self.assertEqual(len(filtered), 5)

    def test_debt_threshold_5000_and_non_debt_15000(self):
        df = pd.DataFrame(
            {
                "scheme_code": ["D1", "D2", "E1", "E2", "E3"],
                "broad_category": ["Debt", "Debt", "Equity", "Equity", "Equity"],
                "aum_cr": [6000.0, 4900.0, 16000.0, 14999.0, np.nan],
            }
        )
        filtered, diag = apply_aum_eligibility_filter(
            df,
            threshold_cr=15000.0,
            debt_threshold_cr=5000.0,
        )
        self.assertEqual(set(filtered["scheme_code"].tolist()), {"D1", "E1"})
        self.assertEqual(diag["threshold_debt_5000_pass_count"], 1)
        self.assertEqual(diag["threshold_non_debt_15000_pass_count"], 1)
        self.assertEqual(diag["debt_after_filter"], 1)
        self.assertEqual(diag["non_debt_after_filter"], 1)


class TestAUMBonusScoring(unittest.TestCase):
    def test_bonus_added_for_gt_25000(self):
        funds = pd.DataFrame(
            {
                "scheme_code": ["A", "B"],
                "scheme_name": ["Fund A", "Fund B"],
                "broad_category": ["Equity", "Equity"],
                "scheme_category": ["Equity - Large Cap", "Equity - Large Cap"],
                "cagr": [0.18, 0.18],
                "alpha": [0.07, 0.07],
                "sortino_ratio": [1.2, 1.2],
                "volatility": [0.15, 0.15],
                "max_drawdown": [-0.35, -0.35],
                "calmar_ratio": [0.5, 0.5],
                "rolling_1y_positive_ratio": [0.8, 0.8],
                "aum_cr": [30000.0, 20000.0],
            }
        )
        scored = FundScorer.calculate_scores(
            funds,
            horizon="long",
            aum_bonus_threshold=25000.0,
            aum_bonus_points=2.0,
        )
        by_code = scored.set_index("scheme_code")
        self.assertEqual(float(by_code.loc["A", "aum_bonus"]), 2.0)
        self.assertEqual(float(by_code.loc["B", "aum_bonus"]), 0.0)
        self.assertGreater(float(by_code.loc["A", "score"]), float(by_code.loc["B", "score"]))

    def test_dynamic_bonus_is_monotonic(self):
        funds = pd.DataFrame(
            {
                "scheme_code": ["A", "B", "C"],
                "scheme_name": ["Fund A", "Fund B", "Fund C"],
                "fund_house": ["H1", "H2", "H3"],
                "broad_category": ["Equity", "Equity", "Equity"],
                "scheme_category": ["Equity - Large Cap"] * 3,
                "cagr": [0.18, 0.18, 0.18],
                "alpha": [0.07, 0.07, 0.07],
                "sortino_ratio": [1.2, 1.2, 1.2],
                "volatility": [0.15, 0.15, 0.15],
                "max_drawdown": [-0.35, -0.35, -0.35],
                "calmar_ratio": [0.5, 0.5, 0.5],
                "rolling_1y_positive_ratio": [0.8, 0.8, 0.8],
                "rolling_cagr_consistency": [0.9, 0.9, 0.9],
                "downside_deviation_stability": [0.9, 0.9, 0.9],
                "return_variance_stability": [0.9, 0.9, 0.9],
                "benchmark_outperformance_ratio": [0.55, 0.55, 0.55],
                "excess_cagr_vs_benchmark": [0.03, 0.03, 0.03],
                "aum_cr": [26000.0, 50000.0, 90000.0],
            }
        )
        scored = FundScorer.calculate_scores(
            funds,
            horizon="long",
            aum_bonus_threshold=25000.0,
            aum_bonus_points=10.0,
        ).set_index("scheme_code")
        self.assertLessEqual(float(scored.loc["A", "aum_bonus"]), float(scored.loc["B", "aum_bonus"]))
        self.assertLessEqual(float(scored.loc["B", "aum_bonus"]), float(scored.loc["C", "aum_bonus"]))
        self.assertLessEqual(float(scored["aum_bonus"].max()), 10.0)


class TestDebtCategoryReweight(unittest.TestCase):
    def test_debt_reweight_prefers_stable_risk_profile(self):
        funds = pd.DataFrame(
            {
                "scheme_code": ["D_STABLE", "D_UNSTABLE", "E_REF"],
                "scheme_name": ["Debt Stable", "Debt Unstable", "Equity Ref"],
                "fund_house": ["DebtHouse1", "DebtHouse2", "EqHouse"],
                "broad_category": ["Debt", "Debt", "Equity"],
                "scheme_category": ["Debt Scheme", "Debt Scheme", "Equity Scheme"],
                "cagr": [0.09, 0.20, 0.16],
                "alpha": [0.01, 0.08, 0.06],
                "sortino_ratio": [1.4, 0.2, 1.0],
                "volatility": [0.025, 0.12, 0.16],
                "max_drawdown": [-0.02, -0.14, -0.30],
                "calmar_ratio": [1.6, 0.4, 0.5],
                "rolling_1y_positive_ratio": [0.95, 0.55, 0.75],
                "rolling_cagr_consistency": [0.92, 0.35, 0.80],
                "downside_deviation_stability": [0.95, 0.40, 0.78],
                "return_variance_stability": [0.93, 0.38, 0.76],
                "benchmark_outperformance_ratio": [0.58, 0.52, 0.54],
                "excess_cagr_vs_benchmark": [0.01, 0.03, 0.04],
                "aum_cr": [30000.0, 30000.0, 30000.0],
            }
        )
        scored = FundScorer.calculate_scores(
            funds,
            horizon="long",
            aum_bonus_threshold=25000.0,
            aum_bonus_points=0.0,
        ).set_index("scheme_code")

        self.assertGreater(float(scored.loc["D_STABLE", "score_within_category"]), float(scored.loc["D_UNSTABLE", "score_within_category"]))
        self.assertGreater(float(scored.loc["D_STABLE", "score"]), float(scored.loc["D_UNSTABLE", "score"]))


class TestStabilityMetrics(unittest.TestCase):
    def test_feature_engineering_outputs_stability_fields(self):
        dates = pd.date_range("2023-01-01", periods=400, freq="B")
        nav = pd.Series(np.linspace(100, 150, len(dates)) + np.random.normal(0, 0.2, len(dates)))
        bench = pd.Series(np.linspace(100, 145, len(dates)) + np.random.normal(0, 0.2, len(dates)))
        nav_df = pd.DataFrame({"date": dates, "nav": nav})
        bench_df = pd.DataFrame({"date": dates, "nav": bench})
        metrics = QuantitativeMetrics.calculate_metrics(nav_df, bench_df)
        self.assertIsNotNone(metrics)
        for key in [
            "rolling_cagr_consistency",
            "downside_deviation_stability",
            "return_variance_stability",
            "excess_cagr_vs_benchmark",
            "benchmark_outperformance_ratio",
        ]:
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()

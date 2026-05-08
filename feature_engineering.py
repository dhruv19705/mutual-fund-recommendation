"""
Mutual Fund Engine V3 - Feature Engineering & Metric Computation
Core metrics calculation with proper alignment, windowing, and validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger("MetricsEngine")
_METRIC_ENGINE_SINGLETON = None


def _get_metric_engine() -> "MetricComputationEngine":
    """Return a cached metric engine instance to avoid per-fund re-init churn."""
    global _METRIC_ENGINE_SINGLETON
    if _METRIC_ENGINE_SINGLETON is None:
        _METRIC_ENGINE_SINGLETON = MetricComputationEngine()
    return _METRIC_ENGINE_SINGLETON


class RiskFreeRateFetcher:
    """Fetch current risk-free rate from market sources."""
    
    @staticmethod
    def get_current_rfr() -> Tuple[float, str, datetime]:
        """
        Fetch current risk-free rate with fallback chain.
        Returns: (rfr_as_decimal, source, timestamp)
        """
        
        # Try Yahoo's 3-month T-bill rate (^IRX)
        try:
            import yfinance as yf
            irx_data = yf.Ticker("^IRX").history(period="1d")
            if not irx_data.empty:
                rfr = irx_data['Close'].iloc[-1] / 100
                logger.info(f"RFR fetched: {rfr:.2%} from Yahoo ^IRX")
                return rfr, "YAHOO_IRX", irx_data.index[-1]
        except Exception as e:
            logger.debug(f"Yahoo RFR fetch failed: {e}")
        
        # Fallback: Config default
        logger.warning("Using default RFR: 6.00%")
        return 0.06, "CONFIG_DEFAULT", datetime.now()


class MetricComputationEngine:
    """Computes quantitative metrics for funds."""
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.rfr = risk_free_rate
        logger.info(f"Metric Engine initialized with RFR={risk_free_rate:.2%}")

    @staticmethod
    def sanitize_returns(returns: pd.Series) -> pd.Series:
        """Keep only finite numeric returns to avoid nan/inf math warnings."""
        if returns is None:
            return pd.Series(dtype=float)
        clean = pd.to_numeric(returns, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        return clean
    
    @staticmethod
    def calculate_cagr(nav_start: float, nav_end: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate.
        CAGR = (Ending Value / Starting Value)^(1/years) - 1
        """
        if nav_start <= 0 or nav_end <= 0 or years <= 0:
            return np.nan
        
        cagr = (nav_end / nav_start) ** (1 / years) - 1
        return cagr if not np.isnan(cagr) else np.nan
    
    @staticmethod
    def calculate_volatility(returns: pd.Series) -> float:
        """
        Calculate annualized volatility from daily returns.
        Volatility = std(daily_returns) * sqrt(252)
        """
        returns = MetricComputationEngine.sanitize_returns(returns)
        
        if len(returns) < 2:
            return np.nan
        
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Enforce minimum volatility floor (0.1%)
        if annual_volatility < 0.001:
            logger.debug(f"Volatility floor applied: {annual_volatility:.4f} → 0.001")
            annual_volatility = 0.001
        
        return annual_volatility if annual_volatility > 0 else np.nan
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, rfr: float) -> float:
        """
        Sharpe Ratio = (Return - RFR) / Volatility
        """
        returns = MetricComputationEngine.sanitize_returns(returns)
        
        if len(returns) < 2:
            return np.nan
        
        excess_return = returns.mean() * 252 - rfr  # Annualized excess
        volatility = returns.std() * np.sqrt(252)
        
        if volatility == 0:
            return np.nan
        
        sharpe = excess_return / volatility
        return sharpe if not np.isnan(sharpe) else np.nan
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, rfr: float, target_return: float = 0) -> float:
        """
        Sortino Ratio = (Return - Target) / Downside Deviation
        Only penalizes downside volatility.
        """
        returns = MetricComputationEngine.sanitize_returns(returns)
        
        if len(returns) < 2:
            return np.nan
        
        excess_return = returns.mean() * 252 - rfr
        
        # Downside deviation: only negative returns
        downside_returns = returns[returns < target_return]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        else:
            downside_deviation = 1e-6  # Avoid division by zero
        
        if downside_deviation == 0:
            return np.nan
        
        sortino = excess_return / downside_deviation
        return sortino if not np.isnan(sortino) else np.nan
    
    @staticmethod
    def calculate_max_drawdown(nav_series: pd.Series) -> float:
        """
        Maximum Drawdown = (Trough - Peak) / Peak
        Measures the largest peak-to-trough decline.
        """
        nav_series = nav_series.dropna()
        
        if len(nav_series) < 2:
            return np.nan
        
        cumulative_max = nav_series.expanding().max()
        drawdown = (nav_series - cumulative_max) / cumulative_max
        max_dd = drawdown.min()
        
        return max_dd if max_dd < 0 else 0.0
    
    @staticmethod
    def calculate_calmar_ratio(cagr: float, max_dd: float) -> float:
        """
        Calmar Ratio = CAGR / |Max Drawdown|
        Measures return per unit of risk (drawdown).
        """
        if max_dd == 0 or max_dd > 0:  # Max DD should be negative
            return np.nan
        
        calmar = cagr / abs(max_dd)
        return calmar if not np.isnan(calmar) else np.nan
    
    def calculate_jensen_alpha(
        self,
        fund_returns: pd.Series,
        market_returns: pd.Series,
        fund_beta: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Jensen's Alpha = Fund Return - (RFR + Beta × (Market Return - RFR))
        
        Returns:
            (alpha, beta) both annualized
        """
        
        # Align on common dates
        fund_returns = MetricComputationEngine.sanitize_returns(fund_returns)
        market_returns = MetricComputationEngine.sanitize_returns(market_returns)
        common_idx = fund_returns.index.intersection(market_returns.index)
        
        if len(common_idx) < 252:
            logger.warning(
                f"Insufficient overlap for alpha: {len(common_idx)} days < 252 required"
            )
            return np.nan, np.nan
        
        fund_ret_aligned = MetricComputationEngine.sanitize_returns(fund_returns[common_idx])
        market_ret_aligned = MetricComputationEngine.sanitize_returns(market_returns[common_idx])
        common_len = min(len(fund_ret_aligned), len(market_ret_aligned))
        if common_len < 252:
            return np.nan, np.nan
        fund_ret_aligned = fund_ret_aligned.iloc[:common_len]
        market_ret_aligned = market_ret_aligned.iloc[:common_len]
        
        # Calculate beta
        covariance = np.cov(fund_ret_aligned, market_ret_aligned)[0, 1]
        market_variance = np.var(market_ret_aligned)
        
        if market_variance == 0:
            return np.nan, np.nan
        
        beta = covariance / market_variance
        
        # Calculate alpha
        fund_annual_return = fund_ret_aligned.mean() * 252
        market_annual_return = market_ret_aligned.mean() * 252
        
        expected_return = self.rfr + beta * (market_annual_return - self.rfr)
        alpha = fund_annual_return - expected_return
        
        return alpha, beta
    
    @staticmethod
    def calculate_information_ratio(
        fund_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Information Ratio = (Fund Return - Benchmark Return) / Tracking Error
        Measures active management skill.
        """
        
        # Align on common dates
        fund_returns = MetricComputationEngine.sanitize_returns(fund_returns)
        benchmark_returns = MetricComputationEngine.sanitize_returns(benchmark_returns)
        common_idx = fund_returns.index.intersection(benchmark_returns.index)
        
        if len(common_idx) < 252:
            return np.nan
        
        fund_ret_aligned = MetricComputationEngine.sanitize_returns(fund_returns[common_idx])
        bench_ret_aligned = MetricComputationEngine.sanitize_returns(benchmark_returns[common_idx])
        common_len = min(len(fund_ret_aligned), len(bench_ret_aligned))
        if common_len < 252:
            return np.nan
        fund_ret_aligned = fund_ret_aligned.iloc[:common_len]
        bench_ret_aligned = bench_ret_aligned.iloc[:common_len]
        
        # Excess returns
        excess_returns = fund_ret_aligned - bench_ret_aligned
        
        # Information Ratio
        annual_excess_return = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error == 0:
            return np.nan
        
        ir = annual_excess_return / tracking_error
        return ir if not np.isnan(ir) else np.nan


class DataAlignmentValidator:
    """Validates and aligns fund & benchmark data."""
    
    @staticmethod
    def check_overlap(
        fund_dates: pd.DatetimeIndex,
        benchmark_dates: pd.DatetimeIndex
    ) -> Tuple[int, str]:
        """
        Check overlap between fund and benchmark data.
        Returns: (overlap_days, status)
        """
        
        common_dates = fund_dates.intersection(benchmark_dates)
        overlap_days = len(common_dates)
        
        if overlap_days < 252:
            status = f"WARNING: Only {overlap_days} common dates (need ≥252)"
        elif overlap_days < 252 * 3:
            status = f"OK: {overlap_days} common dates (3Y)"
        else:
            status = f"✓ Good: {overlap_days} common dates ({overlap_days/252:.1f}Y)"
        
        return overlap_days, status
    
    @staticmethod
    def align_to_common_frequency(
        fund_nav: pd.Series,
        benchmark_nav: pd.Series,
        method: str = "forward_fill"
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Align fund and benchmark to common date range with consistent frequency.
        """
        
        # Get common date range
        min_date = max(fund_nav.index.min(), benchmark_nav.index.min())
        max_date = min(fund_nav.index.max(), benchmark_nav.index.max())
        
        # Create common business day index
        common_index = pd.bdate_range(start=min_date, end=max_date)
        
        # Reindex both to common dates
        fund_aligned = fund_nav.reindex(common_index)
        bench_aligned = benchmark_nav.reindex(common_index)
        
        # Handle missing data
        if method == "forward_fill":
            fund_aligned = fund_aligned.ffill()
            bench_aligned = bench_aligned.ffill()
        elif method == "interpolate":
            fund_aligned = fund_aligned.interpolate(method='linear')
            bench_aligned = bench_aligned.interpolate(method='linear')
        elif method == "drop":
            valid_mask = ~(fund_aligned.isna() | bench_aligned.isna())
            fund_aligned = fund_aligned[valid_mask]
            bench_aligned = bench_aligned[valid_mask]
        
        return fund_aligned, bench_aligned


class DataQualityError(Exception):
    """Raised when data quality check fails."""
    pass


def validate_nav_data(
    nav_df: pd.DataFrame,
    max_staleness_days: int = 180,
    min_history_days: int = 252,
    max_null_ratio: float = 0.20
) -> Tuple[bool, str]:
    """
    Validate NAV data quality.
    Returns: (is_valid, reason)
    """
    
    if nav_df.empty:
        return False, "Empty NAV DataFrame"
    
    # Check staleness
    latest_date = nav_df.index.max()
    days_old = (pd.Timestamp.now() - latest_date).days
    
    if days_old > max_staleness_days:
        return False, f"Stale data: {days_old} days old"
    
    # Check history length
    if len(nav_df) < min_history_days:
        return False, f"Insufficient history: {len(nav_df)} days < {min_history_days}"
    
    # Check null ratio
    null_ratio = nav_df.isnull().sum().sum() / (len(nav_df) * len(nav_df.columns))
    if null_ratio > max_null_ratio:
        return False, f"Too many nulls: {null_ratio:.1%} > {max_null_ratio:.1%}"
    
    return True, "✓ Data quality check passed"


class QuantitativeMetrics:
    """Compatibility wrapper for the main orchestration pipeline."""

    @staticmethod
    def calculate_metrics(nav_df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, float]]:
        """Compute the core fund metrics expected by `main.py`."""
        if nav_df is None or nav_df.empty or 'nav' not in nav_df.columns or 'date' not in nav_df.columns:
            return None

        df = nav_df.copy().dropna(subset=['date', 'nav']).sort_values('date')
        if len(df) < 2:
            return None

        engine = _get_metric_engine()
        nav_series = pd.Series(df['nav'].values, index=pd.to_datetime(df['date']))
        nav_series = pd.to_numeric(nav_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        nav_series = nav_series[nav_series > 0]
        returns = MetricComputationEngine.sanitize_returns(nav_series.pct_change())
        if returns.empty:
            return None

        latest_nav = float(nav_series.iloc[-1])
        earliest_nav = float(nav_series.iloc[0])
        years = max((nav_series.index[-1] - nav_series.index[0]).days / 365.25, 1 / 365.25)

        cagr = engine.calculate_cagr(earliest_nav, latest_nav, years)
        volatility = engine.calculate_volatility(returns)
        sharpe_ratio = engine.calculate_sharpe_ratio(returns, engine.rfr)
        sortino_ratio = engine.calculate_sortino_ratio(returns, engine.rfr)
        max_drawdown = engine.calculate_max_drawdown(nav_series)
        calmar_ratio = engine.calculate_calmar_ratio(cagr, max_drawdown) if pd.notna(cagr) and pd.notna(max_drawdown) else np.nan
        rolling_1y_returns = nav_series.pct_change(252).dropna()
        rolling_1y_positive_ratio = (
            float((rolling_1y_returns > 0).mean())
            if not rolling_1y_returns.empty
            else np.nan
        )
        # Stability factors to penalize "lucky recent winners".
        rolling_cagr_consistency = np.nan
        if len(rolling_1y_returns) >= 4:
            rolling_cagr_consistency = 1.0 / (1.0 + float(rolling_1y_returns.std()))

        downside_deviation_stability = np.nan
        downside_component = returns.where(returns < 0, 0.0)
        rolling_downside = downside_component.rolling(63).apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True).dropna()
        if len(rolling_downside) >= 5:
            downside_deviation_stability = 1.0 / (1.0 + float(rolling_downside.std()))

        return_variance_stability = np.nan
        rolling_var = returns.rolling(63).var().dropna()
        if len(rolling_var) >= 5:
            return_variance_stability = 1.0 / (1.0 + float(rolling_var.std()))

        alpha = np.nan
        beta = np.nan
        information_ratio = np.nan
        excess_cagr_vs_benchmark = np.nan
        benchmark_outperformance_ratio = np.nan
        if benchmark_df is not None and not benchmark_df.empty and 'nav' in benchmark_df.columns:
            bench = benchmark_df.copy()
            if 'date' in bench.columns:
                bench_series = pd.Series(bench['nav'].values, index=pd.to_datetime(bench['date']))
            else:
                bench_series = pd.Series(bench['nav'].values, index=pd.to_datetime(bench.index))
            bench_series = pd.to_numeric(bench_series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            bench_series = bench_series[bench_series > 0]
            benchmark_returns = MetricComputationEngine.sanitize_returns(bench_series.sort_index().pct_change())
            if not benchmark_returns.empty:
                alpha, beta = engine.calculate_jensen_alpha(returns, benchmark_returns)
                information_ratio = engine.calculate_information_ratio(returns, benchmark_returns)
                overlap_idx = returns.index.intersection(benchmark_returns.index)
                if len(overlap_idx) >= 63:
                    fund_overlap = returns.loc[overlap_idx]
                    bench_overlap = benchmark_returns.loc[overlap_idx]
                    benchmark_outperformance_ratio = float((fund_overlap > bench_overlap).mean())

                    fund_nav_overlap = (1.0 + fund_overlap).cumprod()
                    bench_nav_overlap = (1.0 + bench_overlap).cumprod()
                    overlap_years = max(len(overlap_idx) / 252.0, 1 / 252.0)
                    fund_cagr_overlap = (fund_nav_overlap.iloc[-1] / fund_nav_overlap.iloc[0]) ** (1.0 / overlap_years) - 1.0
                    bench_cagr_overlap = (bench_nav_overlap.iloc[-1] / bench_nav_overlap.iloc[0]) ** (1.0 / overlap_years) - 1.0
                    excess_cagr_vs_benchmark = fund_cagr_overlap - bench_cagr_overlap

        return {
            "cagr": float(cagr) if pd.notna(cagr) else 0.0,
            "alpha": float(alpha) if pd.notna(alpha) else 0.0,
            "beta": float(beta) if pd.notna(beta) else 1.0,
            "volatility": float(volatility) if pd.notna(volatility) else 0.0,
            "sharpe_ratio": float(sharpe_ratio) if pd.notna(sharpe_ratio) else 0.0,
            "sortino_ratio": float(sortino_ratio) if pd.notna(sortino_ratio) else 0.0,
            "calmar_ratio": float(calmar_ratio) if pd.notna(calmar_ratio) else 0.0,
            "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else 0.0,
            "rolling_1y_positive_ratio": (
                float(rolling_1y_positive_ratio)
                if pd.notna(rolling_1y_positive_ratio)
                else 0.0
            ),
            "information_ratio": float(information_ratio) if pd.notna(information_ratio) else 0.0,
            "rolling_cagr_consistency": float(rolling_cagr_consistency) if pd.notna(rolling_cagr_consistency) else 0.0,
            "downside_deviation_stability": (
                float(downside_deviation_stability) if pd.notna(downside_deviation_stability) else 0.0
            ),
            "return_variance_stability": (
                float(return_variance_stability) if pd.notna(return_variance_stability) else 0.0
            ),
            "excess_cagr_vs_benchmark": (
                float(excess_cagr_vs_benchmark) if pd.notna(excess_cagr_vs_benchmark) else 0.0
            ),
            "benchmark_outperformance_ratio": (
                float(benchmark_outperformance_ratio) if pd.notna(benchmark_outperformance_ratio) else 0.0
            ),
            "fund_age_years": float(years),
        }

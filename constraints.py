"""Shared thresholds used by the runtime validation layer."""


class DataQualityThresholds:
    """Single source of truth for data quality guardrails."""

    MIN_AUM_CRORE = 10               # Minimum AUM in crore.
    MAX_STALENESS_DAYS = 180         # Max allowed NAV staleness.
    MIN_HISTORY_DAYS = 252           # Minimum 1 year of daily history.
    MIN_BENCHMARK_OVERLAP_DAYS = 252 # Minimum overlap for benchmark metrics.
    MAX_NULL_RATIO = 0.20            # Max missing-value ratio.
    MIN_VOLATILITY = 0.001           # Volatility floor to avoid div-by-zero.

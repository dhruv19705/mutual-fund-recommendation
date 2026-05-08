import os
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from constraints import DataQualityThresholds

logger = logging.getLogger("DataValidator")

CLOSED_END_KEYWORDS = ["Fixed Term", "FTP", "Series", "FMP", "Close Ended", "Closed-End", "(Matured)"]

def is_closed_end(scheme_name: str) -> bool:
    name_lower = scheme_name.lower()
    return any(keyword.lower() in name_lower for keyword in CLOSED_END_KEYWORDS)

def validate_aum(aum: Optional[float], scheme_name: str = "") -> Tuple[bool, str]:
    """
    V3: Hard AUM floor check. Rejects funds below DataQualityThresholds.MIN_AUM_CRORE.
    
    Args:
        aum: AUM value in absolute ₹ (not Crores). Pass None if unavailable.
        scheme_name: For logging context.
    
    Returns:
        (is_valid, reason)
    """
    min_aum_abs = DataQualityThresholds.MIN_AUM_CRORE * 10_000_000
    
    if aum is None:
        logger.warning(f"AUM_REJECT | {scheme_name} | AUM=None (source returned nothing)")
        return False, "AUM data not available"
    
    if not isinstance(aum, (int, float)) or aum <= 0:
        logger.warning(f"AUM_REJECT | {scheme_name} | AUM={aum} (invalid value)")
        return False, f"Invalid AUM value: {aum}"
    
    if aum < min_aum_abs:
        aum_cr = aum / 10_000_000
        logger.warning(
            f"AUM_REJECT | {scheme_name} | AUM=₹{aum_cr:.2f}Cr < ₹{DataQualityThresholds.MIN_AUM_CRORE}Cr floor"
        )
        return False, f"AUM ₹{aum_cr:.1f}Cr below minimum ₹{DataQualityThresholds.MIN_AUM_CRORE}Cr"
    
    aum_cr = aum / 10_000_000
    logger.debug(f"AUM_OK | {scheme_name} | AUM=₹{aum_cr:.0f}Cr")
    return True, "✓ AUM valid"

class CacheManager:
    """Handles local caching of NAV data."""
    def __init__(self, cache_dir: str = ".cache", expiry_days: int = 1):
        self.cache_dir = cache_dir
        self.expiry_days = expiry_days
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def get(self, key: str) -> Optional[pd.DataFrame]:
        path = self._get_cache_path(key)
        if os.path.exists(path):
            # Check expiry
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - mtime < timedelta(days=self.expiry_days):
                try:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading cache for {key}: {e}")
        return None

    def set(self, key: str, df: pd.DataFrame):
        path = self._get_cache_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.error(f"Error saving cache for {key}: {e}")

class DataValidator:
    """Cleans and validates NAV dataframes."""

    @staticmethod
    def validate_aum(aum: Optional[float], scheme_name: str = "") -> Tuple[bool, str]:
        """Compatibility wrapper for V3 AUM validation."""
        return validate_aum(aum, scheme_name=scheme_name)
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates, sorts, and handles missing values."""
        if df is None or df.empty:
            return pd.DataFrame()
            
        df = df.copy()
        
        # Handle Date as Index (common in Yahoo data)
        if 'date' not in [c.lower() for c in df.columns]:
            df = df.reset_index()
            
        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        if 'date' not in df.columns:
            # Try to find date-like column
            date_cols = [c for c in df.columns if 'date' in c or 'time' in c]
            if date_cols:
                df = df.rename(columns={date_cols[0]: 'date'})
            else:
                logger.error("Could not find date column in data.")
                return pd.DataFrame()

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        # Drop duplicates and sort
        df = df.drop_duplicates(subset=['date']).sort_values('date')
        
        # Handle missing NAVs
        df = df.dropna(subset=['nav'])
        
        return df.reset_index(drop=True)

    @staticmethod
    def validate_history(df: pd.DataFrame, min_years: float = 1.0, max_staleness_days: int = 180) -> bool:
        """
        Checks if the data has sufficient history and is not stale.
        
        Args:
            df: DataFrame with 'date' and 'nav' columns
            min_years: Minimum years of history required
            max_staleness_days: Maximum allowed staleness (default 180 days = 6 months)
        
        Returns:
            True if data passes validation, False otherwise
        """
        if df is None or df.empty or 'date' not in df.columns or 'nav' not in df.columns:
            return False
        
        df_copy = df.dropna(subset=['nav'])
        if df_copy.empty:
            return False
        
        latest_date = df_copy['date'].max()
        
        # 1. Staleness Check (more aggressive: max 6 months)
        days_stale = (pd.Timestamp.now().normalize() - latest_date.normalize()).days
        if days_stale > max_staleness_days:
            logger.warning(f"Data is STALE. Latest record from {latest_date.date()} ({days_stale} days ago).")
            return False

        # 2. History Check
        earliest_date = df_copy['date'].min()
        duration = (latest_date - earliest_date).days / 365.25
        if duration < min_years:
            logger.warning(f"Insufficient history: {duration:.2f} years (min {min_years})")
            return False
        
        # 3. Data Quality Check (at least 80% non-null NAV values)
        null_ratio = df['nav'].isna().sum() / len(df)
        if null_ratio > 0.2:
            logger.warning(f"Data quality low: {null_ratio:.1%} null values")
            return False
            
        return True

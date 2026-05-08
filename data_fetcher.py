import requests
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataFetcher")

class MFAPIFetcher:
    """Fetcher for api.mfapi.in"""
    BASE_URL = "https://api.mfapi.in/mf"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    @staticmethod
    def search_funds(query: str) -> List[Dict[str, Any]]:
        """Search for funds by name."""
        try:
            url = f"{MFAPIFetcher.BASE_URL}/search?q={query}"
            response = requests.get(url, headers=MFAPIFetcher.HEADERS)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching MFAPI for '{query}': {e}", exc_info=True)
            return []

    @staticmethod
    def fetch_all_funds() -> List[Dict[str, Any]]:
        """Pulls the complete AMFI fund list (~1800 funds) and filters to Direct Growth only."""
        url = "https://api.mfapi.in/mf"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            all_funds = response.json()
            # Filter to Direct + Growth only at discovery stage
            return [
                f for f in all_funds
                if "direct" in f['schemeName'].lower()
                and "growth" in f['schemeName'].lower()
                and "idcw" not in f['schemeName'].lower()
                and "dividend" not in f['schemeName'].lower()
            ]
        except Exception as e:
            logger.error(f"AMFI master fetch failed: {e}")
            return []

    @staticmethod
    def fetch_nav_history(scheme_code: str) -> Optional[Dict[str, Any]]:
        """Fetch historical NAV for a scheme code."""
        try:
            url = f"{MFAPIFetcher.BASE_URL}/{scheme_code}"
            response = requests.get(url, headers=MFAPIFetcher.HEADERS)
            response.raise_for_status()
            data = response.json()
            if data.get("status", "").upper() == "SUCCESS":
                return data
            logger.warning(f"MFAPI returned status: {data.get('status')} for {scheme_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching NAV for scheme {scheme_code}: {e}", exc_info=True)
            return None

class YahooFetcher:
    """Fetcher for Yahoo Finance via yfinance"""
    
    @staticmethod
    def search_ticker(name: str) -> Optional[str]:
        """Try to find a ticker for a fund name."""
        try:
            # yfinance doesn't have a direct search in all versions, 
            # but we can try common formats or use yf.Ticker(...).info
            # Some versions of yfinance have Search.
            search = yf.Search(name, max_results=5)
            for result in search.quotes:
                # Look for things that look like mutual funds (.BO or .NS)
                symbol = result.get('symbol', '')
                if symbol.endswith(('.BO', '.NS')):
                    return symbol
            return None
        except Exception as e:
            logger.debug(f"Yahoo search failed for '{name}': {e}")
            return None

    @staticmethod
    def fetch_nav_history(ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical NAV from Yahoo Finance."""
        try:
            fund = yf.Ticker(ticker)
            hist = fund.history(period="max")
            if hist.empty:
                return None
            return hist[['Close']].rename(columns={'Close': 'nav'})
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {ticker}: {e}")
            return None

    @staticmethod
    def fetch_benchmark_history(ticker: str = "^NSEI") -> Optional[pd.DataFrame]:
        """Fetch benchmark history (e.g., Nifty 50)."""
        try:
            bench = yf.Ticker(ticker)
            hist = bench.history(period="max")
            if hist.empty:
                return None
            return hist[['Close']].rename(columns={'Close': 'nav'})
        except Exception as e:
            logger.error(f"Error fetching benchmark {ticker}: {e}")
            return None

class HybridFetcher:
    """Orchestrates data ingestion with fallback and validation."""

    def __init__(self):
        self.mf_fetcher = MFAPIFetcher()
        self.yf_fetcher = YahooFetcher()

    def fetch_batch(self, fund_list: list, max_workers: int = 10) -> dict:
        """Fetch NAV histories in parallel. Returns {scheme_code: dataframe}."""
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.get_fund_data, 
                    str(f['schemeCode']), 
                    f['schemeName']
                ): f['schemeCode']
                for f in fund_list
            }
            for future in as_completed(futures):
                code = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[str(code)] = df
                except Exception as e:
                    logger.warning(f"Fetch failed for {code}: {e}")
        return results

    def get_fund_data(self, scheme_code: str, name: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a fund using MFAPI as primary and Yahoo as backup.
        """
        import time
        time.sleep(1) # Rate limiting friendly
        
        # Primary: MFAPI
        mf_data = self.mf_fetcher.fetch_nav_history(scheme_code)
        
        df_mf = None
        if mf_data and "data" in mf_data:
            df_mf = pd.DataFrame(mf_data["data"])
            df_mf['nav'] = pd.to_numeric(df_mf['nav'], errors='coerce')
            df_mf['date'] = pd.to_datetime(df_mf['date'], format='%d-%m-%Y')
            df_mf = df_mf.sort_values('date').reset_index(drop=True)
            logger.info(f"MFAPI: Fetched {len(df_mf)} records for {name} ({scheme_code})")
        else:
            logger.warning(f"MFAPI: Failed to fetch for {name} ({scheme_code})")

        # Secondary: Yahoo Finance
        df_yf = None
        try:
            ticker = self.yf_fetcher.search_ticker(name)
            if ticker:
                df_yf = self.yf_fetcher.fetch_nav_history(ticker)
                if df_yf is not None:
                    df_yf = df_yf.reset_index()
                    df_yf.columns = [c.lower() for c in df_yf.columns]
                    # Convert to datetime and strip timezone
                    df_yf['date'] = pd.to_datetime(df_yf['date']).dt.tz_localize(None)
                    logger.info(f"Yahoo: Fetched {len(df_yf)} records for {name} (Ticker: {ticker})")
        except Exception as e:
            logger.warning(f"Yahoo Search/Fetch failed for {name}: {e}")

        # Consistency Check & Return
        if df_mf is not None and df_yf is not None:
            latest_mf = df_mf.iloc[-1]['nav']
            latest_yf = df_yf.iloc[-1]['nav']
            diff = abs(latest_mf - latest_yf) / latest_mf
            logger.info(f"NAV Validation for {name}: MF={latest_mf}, YF={latest_yf} (diff={diff:.2%})")
            return df_mf # Always prefer MFAPI if available
        
        if df_mf is not None:
            return df_mf
        
        if df_yf is not None:
            logger.info(f"Using Yahoo Finance as fallback for {name}")
            return df_yf

        return None

"""
AUM coverage collector scaffold.

Purpose
-------
1) Pull scheme names from the current fund universe.
2) Try AUM extraction from multiple sources in sequence.
3) Export per-scheme coverage report to CSV.
4) Print an automatic decision on whether AUM is safe for scoring.

This is intentionally a scaffold: add more source adapters over time to
improve coverage.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from data_fetcher import MFAPIFetcher

logger = logging.getLogger("AUMCollector")

DEFAULT_OUTPUT = "aum_coverage_report.csv"
DEFAULT_MIN_COVERAGE = 0.70
ETMONEY_SCHEME_SITEMAP_URL = "https://www.etmoney.com/mf-schemes-sitemap.xml"


@dataclass
class AUMMatch:
    aum_cr: Optional[float]
    source: str
    confidence: float = 0.0


@dataclass
class AUMLookup:
    by_scheme_code: Dict[str, AUMMatch]
    by_norm_name: Dict[str, AUMMatch]
    coverage: float
    source_hit_counts: Dict[str, int]
    generated_at: str
    universe_size: int
    matched_count: int


def _normalize_name(text: str) -> str:
    cleaned = str(text).lower()
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
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split())


def _extract_etmoney_slug_and_id(url: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(r"/mutual-funds/([^/]+)/(\d+)$", str(url).strip(), flags=re.IGNORECASE)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _etmoney_slug_to_name(slug: str) -> str:
    return str(slug).replace("-", " ").strip()


def fetch_etmoney_universe() -> pd.DataFrame:
    """
    Build a universe from ETMoney scheme sitemap.
    This generally includes far more schemes than the static listing snippets.
    """
    try:
        response = requests.get(
            ETMONEY_SCHEME_SITEMAP_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        response.raise_for_status()
        xml_text = response.text
    except Exception as exc:
        logger.warning("ETMoney sitemap fetch failed: %s", exc)
        return pd.DataFrame(columns=["scheme_code", "scheme_name", "universe_source"])

    urls = re.findall(r"<loc>(https://www\.etmoney\.com/mutual-funds/[^<]+)</loc>", xml_text, flags=re.IGNORECASE)
    rows = []
    for url in urls:
        slug, scheme_id = _extract_etmoney_slug_and_id(url)
        if not slug or not scheme_id:
            continue
        rows.append(
            {
                "scheme_code": f"etm_{scheme_id}",
                "scheme_name": _etmoney_slug_to_name(slug),
                "universe_source": "etmoney",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["scheme_code", "scheme_name", "universe_source"])
    return pd.DataFrame(rows).drop_duplicates(subset=["scheme_code"]).reset_index(drop=True)


class BaseAUMSource:
    source_name = "base"

    def prepare(self) -> None:
        """Optional source bootstrap."""
        return

    def extract(self, scheme_name: str) -> AUMMatch:
        raise NotImplementedError


class PPFASNavPageSource(BaseAUMSource):
    """
    Reuses test_aum.py style regex from PPFAS NAV page.
    Good as a proof-of-concept source, but limited coverage.
    """

    source_name = "ppfas_nav_page"
    url = "https://amc.ppfas.com/schemes/nav-history/"
    headers = {"User-Agent": "Mozilla/5.0"}

    def __init__(self) -> None:
        self.html: Optional[str] = None

    def prepare(self) -> None:
        try:
            response = requests.get(self.url, headers=self.headers, timeout=20)
            response.raise_for_status()
            self.html = response.text
            logger.info("Loaded PPFAS source page.")
        except Exception as exc:
            logger.warning("PPFAS source unavailable: %s", exc)
            self.html = None

    def extract(self, scheme_name: str) -> AUMMatch:
        if not self.html:
            return AUMMatch(aum_cr=None, source=self.source_name)

        # exact test_aum.py compatible pattern (works for Parag Parikh page)
        exact = re.search(
            r"Parag Parikh Flexi Cap Fund.*?AUM.*?([\d,]+\.\d+)",
            self.html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if exact and "parag parikh" in str(scheme_name).lower():
            try:
                return AUMMatch(aum_cr=float(exact.group(1).replace(",", "")), source=self.source_name, confidence=1.0)
            except ValueError:
                return AUMMatch(aum_cr=None, source=self.source_name)

        # generic fallback for same page shape
        name = _normalize_name(scheme_name)
        if not name:
            return AUMMatch(aum_cr=None, source=self.source_name)
        generic = re.search(
            rf"{re.escape(name)}.*?AUM.*?([\d,]+\.\d+)",
            self.html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if generic:
            try:
                return AUMMatch(aum_cr=float(generic.group(1).replace(",", "")), source=self.source_name, confidence=0.55)
            except ValueError:
                return AUMMatch(aum_cr=None, source=self.source_name)

        return AUMMatch(aum_cr=None, source=self.source_name)


class AMFIAverageAUMSource(BaseAUMSource):
    """
    Universal source from AMFI average AUM APIs.
    Uses latest financial year and latest period automatically.
    """

    source_name = "amfi_average_aum"
    base_url = "https://www.amfiindia.com"
    headers = {"User-Agent": "Mozilla/5.0"}

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.name_to_aum: Dict[str, float] = {}

    def _safe_get_json(self, path: str) -> Optional[dict]:
        try:
            response = self.session.get(f"{self.base_url}{path}", timeout=25)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("AMFI API call failed for %s: %s", path, exc)
            return None

    @staticmethod
    def _pick_latest_id(rows: list) -> Optional[int]:
        if not rows:
            return None
        # API currently appears ordered latest-first; keep robust fallback.
        try:
            return int(rows[0]["id"])
        except Exception:
            ids = [int(r["id"]) for r in rows if isinstance(r, dict) and "id" in r]
            return max(ids) if ids else None

    def prepare(self) -> None:
        fy_payload = self._safe_get_json("/api/average-aum-fundwise")
        if not fy_payload:
            return
        fy_rows = fy_payload.get("data", [])
        fy_id = self._pick_latest_id(fy_rows)
        if fy_id is None:
            logger.warning("AMFI FY list empty.")
            return

        period_payload = self._safe_get_json(f"/api/average-aum-fundwise?fyId={fy_id}")
        if not period_payload:
            return
        period_rows = (period_payload.get("data") or {}).get("periods", [])
        period_id = self._pick_latest_id(period_rows)
        if period_id is None:
            logger.warning("AMFI period list empty for fyId=%s.", fy_id)
            return

        data_payload = self._safe_get_json(
            f"/api/average-aum-schemewise?strType=Categorywise&fyId={fy_id}&periodId={period_id}&MF_ID=0"
        )
        if not data_payload:
            return

        rows = data_payload.get("data", [])
        loaded = 0
        for block in rows:
            for scheme in block.get("schemes", []) if isinstance(block, dict) else []:
                scheme_name = scheme.get("SchemeNAVName")
                if not scheme_name:
                    continue
                month = scheme.get("AverageAumForTheMonth", {})
                ex_domestic = float(month.get("ExcludingFundOfFundsDomesticButIncludingFundOfFundsOverseas") or 0.0)
                fof_domestic = float(month.get("FundOfFundsDomestic") or 0.0)
                total_lakh = ex_domestic + fof_domestic
                total_cr = total_lakh / 100.0  # 1 crore = 100 lakh
                norm = _normalize_name(scheme_name)
                if norm:
                    self.name_to_aum[norm] = total_cr
                    loaded += 1
        logger.info("AMFI AUM index loaded: %d schemes.", loaded)

    def extract(self, scheme_name: str) -> AUMMatch:
        norm = _normalize_name(scheme_name)
        if not norm:
            return AUMMatch(aum_cr=None, source=self.source_name)

        # Exact normalized match
        if norm in self.name_to_aum:
            return AUMMatch(aum_cr=self.name_to_aum[norm], source=self.source_name, confidence=1.0)

        # Fallback fuzzy containment for naming variants.
        tokens = set(norm.split())
        best_name = None
        best_score = 0
        for candidate in self.name_to_aum:
            if norm in candidate or candidate in norm:
                overlap = len(tokens.intersection(set(candidate.split())))
                overlap_ratio = overlap / max(len(tokens), 1)
                if overlap > best_score and overlap_ratio >= 0.65:
                    best_score = overlap
                    best_name = candidate
        if best_name:
            overlap = len(tokens.intersection(set(best_name.split())))
            confidence = min(0.9, max(0.6, overlap / max(len(tokens), 1)))
            return AUMMatch(aum_cr=self.name_to_aum[best_name], source=self.source_name, confidence=confidence)
        return AUMMatch(aum_cr=None, source=self.source_name)


class ETMoneyUniversalSource(BaseAUMSource):
    """
    Broader source using ETMoney:
    - Build an index from all-funds listing page.
    - Resolve scheme name -> detail URL.
    - Extract AUM from fund detail page.
    """

    source_name = "etmoney"
    listing_url = "https://www.etmoney.com/mutual-funds/all-funds-listing"
    base_url = "https://www.etmoney.com"
    headers = {"User-Agent": "Mozilla/5.0"}

    def __init__(self) -> None:
        self.name_to_href: Dict[str, str] = {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.aum_cache: Dict[str, Optional[float]] = {}

    def prepare(self) -> None:
        # First choice: full scheme sitemap
        sitemap_df = fetch_etmoney_universe()
        if not sitemap_df.empty:
            for _, row in sitemap_df.iterrows():
                code = str(row.get("scheme_code", ""))
                name = str(row.get("scheme_name", ""))
                if not code.startswith("etm_") or not name:
                    continue
                scheme_id = code.replace("etm_", "", 1)
                href = f"/mutual-funds/{name.replace(' ', '-').lower()}/{scheme_id}"
                norm = _normalize_name(name)
                if norm and norm not in self.name_to_href:
                    self.name_to_href[norm] = href
            logger.info("ETMoney sitemap index loaded: %d scheme links.", len(self.name_to_href))
            return

        # Fallback: listing page parser (smaller coverage)
        try:
            html = self.session.get(self.listing_url, timeout=25).text
            entries = re.findall(
                r'href="(/mutual-funds/[^"]+/\d+)"[^>]*title="([^"]+)"',
                html,
                flags=re.IGNORECASE,
            )
            for href, title in entries:
                norm = _normalize_name(title)
                if norm and norm not in self.name_to_href:
                    self.name_to_href[norm] = href
            logger.info("ETMoney listing index loaded: %d scheme links.", len(self.name_to_href))
        except Exception as exc:
            logger.warning("ETMoney listing unavailable: %s", exc)

    def _resolve_href(self, scheme_name: str) -> Optional[str]:
        norm = _normalize_name(scheme_name)
        if not norm:
            return None
        if norm in self.name_to_href:
            return self.name_to_href[norm]

        # Fallback: containment-based fuzzy match.
        scheme_tokens = set(norm.split())
        best_href = None
        best_score = 0
        for candidate, href in self.name_to_href.items():
            if norm in candidate or candidate in norm:
                overlap = len(scheme_tokens.intersection(set(candidate.split())))
                overlap_ratio = overlap / max(len(scheme_tokens), 1)
                if overlap > best_score and overlap_ratio >= 0.60:
                    best_score = overlap
                    best_href = href
        return best_href

    @staticmethod
    def _extract_aum_from_detail_html(html: str) -> Optional[float]:
        patterns = [
            r">\s*AUM\s*<.*?>\s*(?:₹|&#8377;)?\s*([\d,]+(?:\.\d+)?)\s*Cr",
            r"AUM[^0-9]{0,120}(?:₹|&#8377;)?\s*([\d,]+(?:\.\d+)?)\s*Cr",
        ]
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            raw = match.group(1).replace(",", "")
            try:
                return float(raw)
            except ValueError:
                continue
        return None

    def _fetch_aum_by_href(self, href: str) -> Optional[float]:
        if href in self.aum_cache:
            return self.aum_cache[href]
        try:
            url = f"{self.base_url}{href}"
            html = self.session.get(url, timeout=20).text
            value = self._extract_aum_from_detail_html(html)
            self.aum_cache[href] = value
            return value
        except Exception:
            self.aum_cache[href] = None
            return None

    def extract(self, scheme_name: str) -> AUMMatch:
        href = self._resolve_href(scheme_name)
        if not href:
            return AUMMatch(aum_cr=None, source=self.source_name)
        aum = self._fetch_aum_by_href(href)
        return AUMMatch(
            aum_cr=aum,
            source=self.source_name if aum is not None else self.source_name,
            confidence=0.70 if aum is not None else 0.0,
        )


class NullSource(BaseAUMSource):
    """
    Placeholder adapter to make source chaining explicit.
    Replace this with real AMC-specific sources over time.
    """

    source_name = "null_placeholder"

    def extract(self, scheme_name: str) -> AUMMatch:
        _ = scheme_name
        return AUMMatch(aum_cr=None, source=self.source_name, confidence=0.0)


class AUMCollector:
    def __init__(self, sources: List[BaseAUMSource]) -> None:
        self.sources = sources

    def prepare_sources(self) -> None:
        for source in self.sources:
            source.prepare()

    def get_aum(self, scheme_name: str) -> AUMMatch:
        for source in self.sources:
            result = source.extract(scheme_name)
            if result.aum_cr is not None:
                return result
        return AUMMatch(aum_cr=None, source="unmatched")


def get_default_sources() -> List[BaseAUMSource]:
    """Return default source chain (best coverage first)."""
    return [
        AMFIAverageAUMSource(),
        ETMoneyUniversalSource(),
        PPFASNavPageSource(),
        NullSource(),
    ]


def get_fast_sources() -> List[BaseAUMSource]:
    """
    Faster runtime source chain for `main.py`:
    avoid high-latency detail-page scraping.
    """
    return [
        AMFIAverageAUMSource(),
        PPFASNavPageSource(),
        NullSource(),
    ]


def build_universe(limit: Optional[int] = None, source_mode: str = "combined") -> pd.DataFrame:
    source_mode = str(source_mode).lower()

    mfapi_df = pd.DataFrame(columns=["scheme_code", "scheme_name", "universe_source"])
    etmoney_df = pd.DataFrame(columns=["scheme_code", "scheme_name", "universe_source"])

    if source_mode in {"mfapi", "combined"}:
        all_funds = MFAPIFetcher.fetch_all_funds()
        rows = []
        for fund in all_funds:
            code = fund.get("schemeCode")
            name = fund.get("schemeName")
            if code is None or not name:
                continue
            rows.append(
                {
                    "scheme_code": str(code),
                    "scheme_name": str(name),
                    "universe_source": "mfapi",
                }
            )
        mfapi_df = pd.DataFrame(rows)

    if source_mode in {"etmoney", "combined"}:
        etmoney_df = fetch_etmoney_universe()

    if source_mode == "mfapi":
        universe_df = mfapi_df.copy()
    elif source_mode == "etmoney":
        universe_df = etmoney_df.copy()
    else:
        universe_df = pd.concat([mfapi_df, etmoney_df], ignore_index=True)

    if universe_df.empty:
        return universe_df

    # Keep one row per normalized name to avoid near-duplicates across sources.
    universe_df["norm_name"] = universe_df["scheme_name"].map(_normalize_name)
    universe_df = universe_df.sort_values(["norm_name", "universe_source"]).drop_duplicates(
        subset=["norm_name"], keep="first"
    )
    universe_df = universe_df.drop(columns=["norm_name"]).reset_index(drop=True)

    if limit is not None:
        universe_df = universe_df.head(limit).copy()
    return universe_df


def evaluate_coverage(df: pd.DataFrame, min_coverage: float) -> str:
    coverage = float(df["aum_cr"].notna().mean()) if not df.empty else 0.0
    if coverage >= min_coverage:
        return f"SAFE_TO_USE (coverage={coverage:.1%} >= threshold={min_coverage:.1%})"
    return f"NOT_SAFE_YET (coverage={coverage:.1%} < threshold={min_coverage:.1%})"


def _source_rank(source: str) -> int:
    ranking = {
        "amfi_average_aum": 3,
        "etmoney": 2,
        "ppfas_nav_page": 1,
        "unmatched": 0,
        "null_placeholder": 0,
    }
    return ranking.get(str(source), 0)


def _is_better_match(candidate: AUMMatch, current: Optional[AUMMatch]) -> bool:
    if current is None:
        return True
    if current.aum_cr is None and candidate.aum_cr is not None:
        return True
    if candidate.aum_cr is None:
        return False
    if current.aum_cr is None:
        return True
    if candidate.confidence > current.confidence:
        return True
    if candidate.confidence == current.confidence and _source_rank(candidate.source) > _source_rank(current.source):
        return True
    return False


def build_aum_lookup(
    limit: Optional[int] = None,
    universe_source: str = "combined",
    use_fast_sources: bool = False,
) -> AUMLookup:
    """
    Build in-memory AUM lookup for runtime usage by the scoring pipeline.
    """
    universe_df = build_universe(limit=limit, source_mode=universe_source)
    if universe_df.empty:
        return AUMLookup(
            by_scheme_code={},
            by_norm_name={},
            coverage=0.0,
            source_hit_counts={},
            generated_at=datetime.now().isoformat(),
            universe_size=0,
            matched_count=0,
        )

    collector = AUMCollector(sources=get_fast_sources() if use_fast_sources else get_default_sources())
    collector.prepare_sources()
    matches = universe_df["scheme_name"].apply(collector.get_aum)
    enriched_df = universe_df.copy()
    enriched_df["aum_cr"] = matches.apply(lambda x: x.aum_cr)
    enriched_df["aum_source"] = matches.apply(lambda x: x.source)
    enriched_df["aum_matched"] = enriched_df["aum_cr"].notna()

    by_scheme_code: Dict[str, AUMMatch] = {}
    for _, row in enriched_df.iterrows():
        code = str(row.get("scheme_code", ""))
        if not code:
            continue
        candidate = AUMMatch(
            aum_cr=float(row["aum_cr"]) if pd.notna(row["aum_cr"]) else None,
            source=str(row.get("aum_source", "unmatched")),
            confidence=1.0 if pd.notna(row["aum_cr"]) else 0.0,
        )
        current = by_scheme_code.get(code)
        if _is_better_match(candidate, current):
            by_scheme_code[code] = candidate

    by_norm_name: Dict[str, AUMMatch] = {}
    for _, row in enriched_df.iterrows():
        norm = _normalize_name(row.get("scheme_name", ""))
        if not norm:
            continue
        current = by_norm_name.get(norm)
        candidate = AUMMatch(
            aum_cr=float(row["aum_cr"]) if pd.notna(row["aum_cr"]) else None,
            source=str(row.get("aum_source", "unmatched")),
            confidence=1.0 if pd.notna(row["aum_cr"]) else 0.0,
        )
        if _is_better_match(candidate, current):
            by_norm_name[norm] = candidate

    coverage = float(enriched_df["aum_matched"].mean())
    source_hit_counts = {
        str(source): int(count)
        for source, count in enriched_df["aum_source"].value_counts(dropna=False).items()
    }
    matched_count = int(enriched_df["aum_matched"].sum())

    return AUMLookup(
        by_scheme_code=by_scheme_code,
        by_norm_name=by_norm_name,
        coverage=coverage,
        source_hit_counts=source_hit_counts,
        generated_at=datetime.now().isoformat(),
        universe_size=int(len(enriched_df)),
        matched_count=matched_count,
    )


def lookup_aum_for_fund(scheme_code: str, scheme_name: str, lookup: AUMLookup) -> AUMMatch:
    """
    Resolve AUM in this order:
    1) Exact scheme code match
    2) Normalized name match
    3) unmatched
    """
    code = str(scheme_code or "")
    if code in lookup.by_scheme_code:
        return lookup.by_scheme_code[code]
    norm_name = _normalize_name(scheme_name)
    if norm_name and norm_name in lookup.by_norm_name:
        return lookup.by_norm_name[norm_name]
    return AUMMatch(aum_cr=None, source="unmatched")


def summarize_high_aum_coverage(universe_df: pd.DataFrame, high_aum_threshold_cr: float = 25000.0) -> dict:
    """Return diagnostics for high-AUM extraction coverage."""
    if universe_df is None or universe_df.empty:
        return {
            "high_aum_threshold_cr": high_aum_threshold_cr,
            "high_aum_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "by_source": {},
            "by_category_hint": {},
        }

    df = universe_df.copy()
    df["scheme_name"] = df["scheme_name"].astype(str)
    df["aum_cr"] = pd.to_numeric(df["aum_cr"], errors="coerce")
    high_df = df[df["aum_cr"] > float(high_aum_threshold_cr)].copy()
    matched_df = high_df[high_df["aum_cr"].notna()]

    category_hint = "unknown"
    if "scheme_category" in high_df.columns:
        high_df["category_hint"] = high_df["scheme_category"].astype(str)
    else:
        high_df["category_hint"] = high_df["scheme_name"].str.extract(
            r"(small cap|mid cap|large cap|flexi cap|multi cap|debt|hybrid|liquid|gilt)",
            flags=re.IGNORECASE,
            expand=False,
        ).fillna(category_hint)

    return {
        "high_aum_threshold_cr": high_aum_threshold_cr,
        "high_aum_count": int(len(high_df)),
        "matched_count": int(len(matched_df)),
        "unmatched_count": int(max(len(high_df) - len(matched_df), 0)),
        "by_source": {str(k): int(v) for k, v in high_df["aum_source"].value_counts(dropna=False).items()},
        "by_category_hint": {str(k): int(v) for k, v in high_df["category_hint"].value_counts(dropna=False).items()},
    }


def run_collection(output_csv: str, min_coverage: float, limit: Optional[int], universe_source: str) -> None:
    universe_df = build_universe(limit=limit, source_mode=universe_source)
    if universe_df.empty:
        print("No funds in universe. Nothing to process.")
        return

    lookup = build_aum_lookup(limit=limit, universe_source=universe_source)
    matches = universe_df.apply(
        lambda row: lookup_aum_for_fund(
            scheme_code=str(row.get("scheme_code", "")),
            scheme_name=str(row.get("scheme_name", "")),
            lookup=lookup,
        ),
        axis=1,
    )
    universe_df["aum_cr"] = matches.apply(lambda x: x.aum_cr)
    universe_df["aum_source"] = matches.apply(lambda x: x.source)
    universe_df["aum_matched"] = universe_df["aum_cr"].notna()

    coverage = float(universe_df["aum_matched"].mean())
    matched = int(universe_df["aum_matched"].sum())
    verdict = evaluate_coverage(universe_df, min_coverage=min_coverage)

    universe_df.to_csv(output_csv, index=False)

    print("\n=== AUM COVERAGE REPORT ===")
    print(f"Universe size     : {len(universe_df)}")
    print(f"AUM matched       : {matched}")
    print(f"AUM coverage      : {coverage:.1%}")
    print(f"Threshold         : {min_coverage:.1%}")
    print(f"Decision          : {verdict}")
    print(f"CSV output        : {output_csv}")

    universe_counts = universe_df["universe_source"].value_counts(dropna=False)
    if not universe_counts.empty:
        print("\nUniverse source counts:")
        for src, count in universe_counts.items():
            print(f"  - {src}: {int(count)}")

    source_counts = universe_df["aum_source"].value_counts(dropna=False)
    if not source_counts.empty:
        print("\nSource hit counts:")
        for source, count in source_counts.items():
            print(f"  - {source}: {int(count)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Collect AUM coverage for current fund universe.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="Coverage threshold to mark AUM as safe for scoring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional universe size limit for faster trial runs",
    )
    parser.add_argument(
        "--universe-source",
        choices=["mfapi", "etmoney", "combined"],
        default="combined",
        help="Universe source mode",
    )
    args = parser.parse_args()

    run_collection(
        output_csv=args.output,
        min_coverage=args.min_coverage,
        limit=args.limit,
        universe_source=args.universe_source,
    )

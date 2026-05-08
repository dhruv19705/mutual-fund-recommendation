"""
AUM integration experiment (safe, standalone).

This script does NOT modify the production pipeline.
It runs the existing engine, tries to fetch AUM using the same
regex-style logic from test_aum.py, and reports if AUM is practical
to include as a model feature.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
import requests

from main import run_engine

logger = logging.getLogger("AUMExperiment")

PPFAS_AUM_URL = "https://amc.ppfas.com/schemes/nav-history/"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _normalize_scheme_name(scheme_name: str) -> str:
    """Normalize noisy suffixes to improve regex hit-rate."""
    text = str(scheme_name).lower()
    for token in [
        " direct plan",
        " regular plan",
        " growth option",
        " growth",
        " idcw",
        " - ",
    ]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def fetch_ppfas_page_html() -> Optional[str]:
    try:
        response = requests.get(PPFAS_AUM_URL, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.error("Could not fetch AUM source page: %s", exc)
        return None


def extract_aum_for_scheme(scheme_name: str, html: str) -> Optional[float]:
    """
    Reuse test_aum.py style:
    '<scheme_name> ... AUM ... <number>'
    """
    # 1) Exact test_aum.py-style extraction for the known PPFAS page.
    # This source URL is scheme-specific (Parag Parikh Flexi Cap Fund), so
    # this direct pattern is the most reliable one for this page.
    known_pattern = r"Parag Parikh Flexi Cap Fund.*?AUM.*?([\d,]+\.\d+)"
    known_match = re.search(known_pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if known_match and "parag parikh" in str(scheme_name).lower():
        try:
            return float(known_match.group(1).replace(",", ""))
        except ValueError:
            return None

    # 2) Generic fallback (still regex-based) if scheme names line up.
    normalized_name = _normalize_scheme_name(scheme_name)
    if not normalized_name:
        return None
    generic_pattern = rf"{re.escape(normalized_name)}.*?AUM.*?([\d,]+\.\d+)"
    match = re.search(generic_pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    aum_str = match.group(1).replace(",", "")
    try:
        return float(aum_str)
    except ValueError:
        return None


def run_aum_experiment(risk: str, horizon: str, sample_size: int = 30) -> None:
    scored_df, recommendation = run_engine(risk_appetite=risk, horizon=horizon)
    if scored_df is None or scored_df.empty:
        print("Experiment aborted: engine returned no scored funds.")
        return

    baseline_top = (
        scored_df.sort_values("score", ascending=False)
        .head(5)["scheme_name"]
        .astype(str)
        .tolist()
    )

    html = fetch_ppfas_page_html()
    if not html:
        print("AUM check failed: source page unavailable.")
        return

    # Sanity proof: same pattern used in test_aum.py
    sanity = re.search(r"Parag Parikh Flexi Cap Fund.*?AUM.*?([\d,]+\.\d+)", html, re.IGNORECASE | re.DOTALL)
    if sanity:
        print(f"Sanity extraction (test_aum pattern): {sanity.group(1)} Cr")
    else:
        print("Sanity extraction failed for test_aum pattern.")

    work_df = scored_df.sort_values("score", ascending=False).head(sample_size).copy()
    work_df["aum_cr"] = work_df["scheme_name"].apply(lambda name: extract_aum_for_scheme(name, html))

    coverage = float(work_df["aum_cr"].notna().mean())
    matched = int(work_df["aum_cr"].notna().sum())

    print("\n=== AUM EXPERIMENT REPORT ===")
    print(f"Risk/Horizon            : {risk}/{horizon}")
    print(f"Sample size             : {len(work_df)}")
    print(f"AUM matches found       : {matched}")
    print(f"AUM coverage            : {coverage:.1%}")

    if coverage < 0.30:
        print("\nVerdict: NOT USABLE right now.")
        print(
            "Reason: The current scraping source/logic gives low AUM coverage, "
            "so AUM would bias rankings due to missing values."
        )
        print(
            "Suggestion: Keep this as an experiment file and delete it if not needed, "
            "or switch to a broader AUM source before integration."
        )
        return

    # If enough coverage, test a small blended scoring impact.
    # Fill missing AUM with median so missing rows are neutral.
    aum_filled = work_df["aum_cr"].fillna(work_df["aum_cr"].median())
    work_df["aum_log"] = np.log1p(aum_filled)
    aum_norm = (work_df["aum_log"] - work_df["aum_log"].min()) / (
        (work_df["aum_log"].max() - work_df["aum_log"].min()) or 1.0
    )
    work_df["score_with_aum"] = 0.90 * work_df["score"] + 10.0 * aum_norm

    aum_top = (
        work_df.sort_values("score_with_aum", ascending=False)
        .head(5)["scheme_name"]
        .astype(str)
        .tolist()
    )
    overlap = len(set(baseline_top) & set(aum_top))

    print("\nVerdict: USABLE (experimentally) in this sample.")
    print("Baseline Top-5:")
    for i, name in enumerate(baseline_top, start=1):
        print(f"  {i}. {name}")
    print("\nAUM-blended Top-5:")
    for i, name in enumerate(aum_top, start=1):
        print(f"  {i}. {name}")
    print(f"\nTop-5 overlap            : {overlap}/5")
    if recommendation and recommendation.get("allocation"):
        print(f"Portfolio funds (base)   : {len(recommendation.get('allocation', []))}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Test whether AUM can be used in model scoring.")
    parser.add_argument("--risk", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--horizon", choices=["short", "medium", "long"], default="long")
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()

    run_aum_experiment(risk=args.risk, horizon=args.horizon, sample_size=args.sample_size)

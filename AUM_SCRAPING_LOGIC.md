# AUM Scraping Logic

This document explains how AUM (Assets Under Management) values are discovered, matched, and used in this project.

## Where the logic lives

- Primary implementation: `aum_collector.py`
- Main source chain definitions:
  - `get_default_sources()` for higher coverage
  - `get_fast_sources()` for lower-latency runtime use

## High-level flow

1. Build a fund universe (`build_universe`).
2. Initialize and prepare AUM sources (`AUMCollector.prepare_sources`).
3. For each scheme, query sources in priority order (`AUMCollector.get_aum`).
4. Store the best match in lookup maps (`build_aum_lookup`).
5. Report coverage and source hit statistics.

## Universe construction

`build_universe(source_mode=...)` supports:

- `mfapi`: from `MFAPIFetcher.fetch_all_funds()`
- `etmoney`: from ETMoney sitemap parsing (`fetch_etmoney_universe`)
- `combined`: concatenates both and deduplicates by normalized scheme name

Deduplication uses `_normalize_name(...)` so naming variants are collapsed.

## Name normalization strategy

`_normalize_name(text)` standardizes names by:

- lowercasing text
- removing tokens such as `direct plan`, `regular plan`, `growth`, `idcw`, `dividend`
- removing punctuation-like tokens (`-`, `(`, `)`)
- collapsing extra spaces

This normalization is used for:

- deduplicating universe entries
- exact/fuzzy AUM matching by scheme name
- fallback lookup when scheme code match is unavailable

## Source adapters and extraction logic

All sources inherit from `BaseAUMSource` and implement:

- `prepare()`: optional preloading/bootstrap
- `extract(scheme_name) -> AUMMatch`

### 1) AMFI average AUM source (`AMFIAverageAUMSource`)

- Source name: `amfi_average_aum`
- Calls AMFI APIs to discover latest:
  - financial year (`/api/average-aum-fundwise`)
  - period (`/api/average-aum-fundwise?fyId=...`)
  - scheme-wise data (`/api/average-aum-schemewise?...`)
- Builds in-memory map: `normalized_scheme_name -> aum_cr`
- Converts AMFI values from lakh to crore:
  - `total_cr = (ex_domestic + fof_domestic) / 100`
- Match mode:
  - exact normalized name match (confidence `1.0`)
  - fallback token-overlap containment (confidence approx `0.6` to `0.9`)

### 2) ETMoney universal source (`ETMoneyUniversalSource`)

- Source name: `etmoney`
- Index bootstrap:
  - first preference: ETMoney sitemap (`fetch_etmoney_universe`)
  - fallback: parse all-funds listing page
- For each scheme:
  - resolve detail-page href by exact/fuzzy normalized name match
  - fetch detail page
  - parse AUM with regex patterns expecting values in `Cr`
- Uses per-href cache (`aum_cache`) to avoid repeated network fetches
- Returns confidence `0.70` when AUM is found

### 3) PPFAS NAV page source (`PPFASNavPageSource`)

- Source name: `ppfas_nav_page`
- Preloads a single NAV history page
- Supports:
  - specific pattern for Parag Parikh Flexi Cap Fund
  - generic fallback regex against normalized input name
- Intended as a limited proof-of-concept source

### 4) Null placeholder (`NullSource`)

- Source name: `null_placeholder`
- Always returns no AUM
- Keeps source chaining explicit and easy to extend

## Source priority

Default order (`get_default_sources()`):

1. `amfi_average_aum`
2. `etmoney`
3. `ppfas_nav_page`
4. `null_placeholder`

Fast order (`get_fast_sources()`):

1. `amfi_average_aum`
2. `ppfas_nav_page`
3. `null_placeholder`

The fast mode intentionally skips ETMoney detail-page scraping to reduce runtime latency.

## Match selection and lookup build

`build_aum_lookup(...)` generates:

- `by_scheme_code: Dict[str, AUMMatch]`
- `by_norm_name: Dict[str, AUMMatch]`
- aggregate metadata (`coverage`, `source_hit_counts`, `matched_count`, etc.)

When multiple candidate matches exist, `_is_better_match(...)` prefers:

1. records with non-null AUM
2. higher confidence
3. higher source rank tie-breaker (`amfi` > `etmoney` > `ppfas`)

Runtime resolution (`lookup_aum_for_fund`) is:

1. exact `scheme_code`
2. normalized `scheme_name`
3. `unmatched`

## Coverage reporting

`run_collection(...)` writes `aum_coverage_report.csv` with:

- `scheme_code`
- `scheme_name`
- `universe_source`
- `aum_cr`
- `aum_source`
- `aum_matched`

It also prints:

- universe size
- matched count and coverage ratio
- threshold decision (`SAFE_TO_USE` or `NOT_SAFE_YET`)
- source hit counts

## Extending scraping logic

To add a new source:

1. Create a new `BaseAUMSource` subclass.
2. Implement robust `prepare()` and `extract()`.
3. Return `AUMMatch(aum_cr=..., source=..., confidence=...)`.
4. Insert the source in `get_default_sources()` (and optionally `get_fast_sources()`).
5. Add/adjust tests for matching and coverage behavior.

Recommended improvements:

- stronger fuzzy matching (token weighting or phonetic similarity)
- retry/backoff and response validation for network calls
- source freshness metadata (as-of dates)
- optional persistence of source-level snapshots for debugging

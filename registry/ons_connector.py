"""
Connector for the Office for National Statistics (ONS) API.

This is a real, open, unauthenticated REST API at https://api.beta.ons.gov.uk/v1.
It's the one source in the registry that behaves like a normal API: you can
query a specific dataset/edition/version and get observation-level data back.

Docs: https://developer.ons.gov.uk/

NOTE ON SANDBOXING: this code makes real HTTP calls to api.beta.ons.gov.uk.
It has NOT been live-tested from this environment because that domain isn't
reachable from the build sandbox's network allowlist. The request shapes
below match ONS's published API docs exactly, but run a smoke test against
the live API once you deploy this somewhere with open egress.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from .base import Connector, FetchResult, Source, SourceType

ONS_API_BASE = "https://api.beta.ons.gov.uk/v1"


def parse_timeseries_observations(fetch_result: FetchResult) -> tuple[pd.Series, pd.Series]:
    """
    Convert a raw ONS /observations payload into (time_index, values) series,
    ready to feed into stat_tests.run_linear_regression(x=time_index, y=values)
    for a trend claim ("X has risen/fallen since...").

    Built directly against ONS's documented observations response shape:
        {"observations": [{"observation": "3.2", "dimensions": {"time": {"id": "2024-01", ...}, ...}}, ...]}

    NOT yet live-tested (same caveat as the rest of this connector — this
    sandbox can't reach api.beta.ons.gov.uk). If ONS's actual response
    shape differs in some field name, this is the first place to check
    when a TREND claim fails after the dataset ID itself resolves fine.
    """
    observations = fetch_result.data.get("observations", [])
    if not observations:
        raise ValueError(
            "No 'observations' in ONS response — check the dataset's actual "
            "dimension names via get_dataset_metadata() before assuming this "
            "adapter's field names are wrong."
        )

    rows = []
    for obs in observations:
        try:
            value = float(obs["observation"])
        except (KeyError, ValueError, TypeError):
            continue  # skip missing/non-numeric observations rather than crash the whole series
        time_dim = obs.get("dimensions", {}).get("time", {})
        time_label = time_dim.get("id") or time_dim.get("label")
        if time_label is None:
            continue
        rows.append((time_label, value))

    if len(rows) < 3:
        raise ValueError(
            f"Only {len(rows)} usable observations parsed — not enough for a "
            "trend test. Check the fetch's geography/time filters."
        )

    # Sort by the time label as given (ONS time IDs are lexically sortable
    # for both YYYY and YYYY-MM formats) and use position as the numeric
    # x-axis — the actual calendar spacing doesn't matter for a trend
    # direction test, only the ordering does.
    rows.sort(key=lambda r: r[0])
    time_index = pd.Series(range(len(rows)))
    values = pd.Series([v for _, v in rows])
    return time_index, values


class ONSConnector(Connector):
    def __init__(self, source: Optional[Source] = None):
        source = source or Source(
            id="ons",
            name="Office for National Statistics API",
            type=SourceType.LIVE_API,
            org="ONS",
            description=(
                "UK's national statistical institute. Live queryable API "
                "covering employment, population, migration, and economic "
                "statistics."
            ),
            base_url=ONS_API_BASE,
            license="Open Government Licence v3.0",
            update_frequency="varies by dataset (monthly/quarterly/annual)",
            topics=[
                "unemployment", "employment", "jobs", "labour market",
                "population", "migration", "gdp", "inflation", "wages",
                "crime rate", "crime", "census",
            ],
            methodology_url="https://developer.ons.gov.uk/",
        )
        super().__init__(source)

    def list_datasets(self) -> list[dict]:
        """List all datasets available via the ONS API."""
        resp = requests.get(f"{ONS_API_BASE}/datasets", timeout=15)
        resp.raise_for_status()
        return resp.json().get("items", [])

    def get_dataset_metadata(self, dataset_id: str) -> dict:
        resp = requests.get(f"{ONS_API_BASE}/datasets/{dataset_id}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def fetch(
        self,
        dataset_id: str,
        edition: str = "time-series",
        version: Optional[int] = None,
        **dimension_filters,
    ) -> FetchResult:
        """
        Fetch observation-level data for a dataset.

        Example (unemployment claim):
            connector.fetch(
                dataset_id="labour-market",
                geography="K02000001",   # UK
                time="*",                # wildcard: all time periods
            )

        `dimension_filters` are passed straight through as query params —
        ONS dimensions vary per dataset (geography, time, sex, age, etc.),
        so check get_dataset_metadata() first to see what's available.
        """
        # resolve latest version if not given
        if version is None:
            meta = self.get_dataset_metadata(dataset_id)
            latest_href = meta.get("links", {}).get("latest_version", {}).get("href")
            if not latest_href:
                raise ValueError(
                    f"Could not resolve latest version for dataset '{dataset_id}'"
                )
            obs_url = latest_href.rstrip("/") + "/observations"
        else:
            obs_url = (
                f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{edition}"
                f"/versions/{version}/observations"
            )

        resp = requests.get(obs_url, params=dimension_filters, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        return FetchResult(
            source_id=self.source.id,
            fetched_at=datetime.now(timezone.utc),
            data=payload,
            provenance_url=resp.url,
            notes=(
                "Raw ONS observation payload. Check 'observation level "
                "metadata' in the response for coefficients of variation "
                "before treating values as precise."
            ),
        )

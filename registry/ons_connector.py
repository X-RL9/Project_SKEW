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

    def get_dataset_dimensions(self, dataset_id: str, edition: str, version: int) -> list[dict]:
        """List the actual dimensions a dataset/edition/version requires.
        This is how we find out, e.g., that 'labour-market' needs age
        group, economic activity, seasonal adjustment, sex, and unit of
        measure -- not just geography and time -- instead of guessing."""
        resp = requests.get(
            f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{edition}"
            f"/versions/{version}/dimensions",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])

    def get_dimension_options(
        self, dataset_id: str, edition: str, version: int, dimension_id: str, limit: int = 200
    ) -> list[dict]:
        resp = requests.get(
            f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{edition}"
            f"/versions/{version}/dimensions/{dimension_id}/options",
            params={"limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])

    def _resolve_edition_version(self, dataset_id: str, edition, version):
        """Figure out the actual edition string + version number we're
        querying, whether the caller gave one or we need the latest."""
        if version is not None:
            return edition or "time-series", version
        meta = self.get_dataset_metadata(dataset_id)
        latest = meta.get("links", {}).get("latest_version", {})
        latest_href = latest.get("href")
        latest_id = latest.get("id")
        if not latest_href or not latest_id:
            raise ValueError(f"Could not resolve latest version for dataset '{dataset_id}'")
        # edition is embedded in the href: .../editions/{edition}/versions/{id}
        parts = latest_href.rstrip("/").split("/")
        resolved_edition = parts[parts.index("editions") + 1]
        return resolved_edition, int(latest_id)

    def fetch(
        self,
        dataset_id: str,
        edition: Optional[str] = None,
        version: Optional[int] = None,
        preferred_options: Optional[dict] = None,
        **dimension_filters,
    ) -> FetchResult:
        """
        Fetch observation-level data for a dataset.

        Example (unemployment claim):
            connector.fetch(
                dataset_id="labour-market",
                geography="K02000001",   # UK
                time="*",                # wildcard: all time periods
                preferred_options={
                    "seasonaladjustment": ["Seasonally Adjusted"],
                    "economicactivity": ["Unemployed"],
                    "unitofmeasure": ["Rates"],
                    "sex": ["All adults"],
                    "agegroups": ["16+"],
                },
            )

        `dimension_filters` are passed straight through as query params for
        anything you already know the exact value for. For anything you
        DON'T specify, this method looks up the dataset's real dimensions
        from ONS (get_dataset_dimensions) and fills in a default for each
        missing one -- required, because some datasets (labour-market,
        cpih01, etc.) 400 if every dimension isn't given a value, and the
        set of dimensions differs per dataset so it can't be hardcoded.

        `preferred_options` lets you bias that auto-fill: for a given
        dimension id, give an ordered list of label substrings to look for
        (case-insensitive) among its real options, e.g. prefer "Seasonally
        Adjusted" over "Not Seasonally Adjusted" for headline unemployment.
        If nothing matches (or no preference given), it falls back to
        whatever option ONS lists first -- and that fallback is recorded in
        the result's notes, not hidden, since an unreviewed default could
        be the wrong slice of the data.
        """
        preferred_options = preferred_options or {}
        resolved_edition, resolved_version = self._resolve_edition_version(
            dataset_id, edition, version
        )

        auto_filled_notes = []
        try:
            dims = self.get_dataset_dimensions(dataset_id, resolved_edition, resolved_version)
        except requests.HTTPError:
            dims = []  # if this itself fails, fall through and let the real request surface the error

        for dim in dims:
            dim_id = dim.get("id")
            if not dim_id or dim_id in dimension_filters:
                continue
            if dim_id == "time":
                dimension_filters["time"] = "*"
                continue
            try:
                options = self.get_dimension_options(
                    dataset_id, resolved_edition, resolved_version, dim_id
                )
            except requests.HTTPError:
                continue  # can't resolve this one -- let the observations call itself report the real error
            if not options:
                continue

            chosen = None
            for wanted_label in preferred_options.get(dim_id, []):
                for opt in options:
                    label = (opt.get("label") or opt.get("id") or "")
                    if label.strip().lower() == wanted_label.strip().lower():
                        chosen = opt
                        break
                if chosen:
                    break
            if chosen is None:
                chosen = options[0]
                auto_filled_notes.append(
                    f"{dim_id}='{chosen.get('label', chosen.get('id'))}' (no preference given -- "
                    f"picked ONS's first listed option, not necessarily the headline one)"
                )
            else:
                auto_filled_notes.append(
                    f"{dim_id}='{chosen.get('label', chosen.get('id'))}' (matched preference)"
                )
            dimension_filters[dim_id] = chosen.get("id")

        obs_url = (
            f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{resolved_edition}"
            f"/versions/{resolved_version}/observations"
        )
        resp = requests.get(obs_url, params=dimension_filters, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        notes = (
            "Raw ONS observation payload. Check 'observation level "
            "metadata' in the response for coefficients of variation "
            "before treating values as precise."
        )
        if auto_filled_notes:
            notes += " Auto-filled dimensions: " + "; ".join(auto_filled_notes)

        return FetchResult(
            source_id=self.source.id,
            fetched_at=datetime.now(timezone.utc),
            data=payload,
            provenance_url=resp.url,
            notes=notes,
        )

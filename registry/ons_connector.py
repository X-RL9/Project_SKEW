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
from functools import lru_cache
import re
from typing import Optional

import pandas as pd
import requests

from .base import Connector, FetchResult, Source, SourceType

ONS_API_BASE = "https://api.beta.ons.gov.uk/v1"


class ONSDataError(RuntimeError):
    """A data-selection or response-shape error from the ONS connector."""


def _response_error(resp: requests.Response, context: str) -> ONSDataError:
    body = (resp.text or "").strip().replace("\n", " ")[:500]
    detail = f" Response: {body}" if body else ""
    return ONSDataError(
        f"ONS {context} failed with HTTP {resp.status_code} for {resp.url}.{detail}"
    )


def _time_value(dimensions: dict) -> Optional[str]:
    for name, value in dimensions.items():
        if name.casefold() == "time" and isinstance(value, dict):
            return value.get("id") or value.get("label")
    return None


def _parse_ons_time(label: str) -> pd.Timestamp:
    """Parse common ONS labels without treating 1999 as 2099."""
    text = str(label).strip()
    monthly = re.fullmatch(r"([A-Za-z]{3})-(\d{2})", text)
    if monthly:
        short_year = int(monthly.group(2))
        current_short_year = datetime.now().year % 100
        century = 2000 if short_year <= current_short_year + 1 else 1900
        return pd.Timestamp(datetime.strptime(monthly.group(1), "%b").replace(
            year=century + short_year
        ))

    rolling = re.fullmatch(r"([A-Za-z]{3})-[A-Za-z]{3}\s+(\d{4})", text)
    if rolling:
        return pd.Timestamp(datetime.strptime(
            f"{rolling.group(1)} {rolling.group(2)}", "%b %Y"
        ))

    quarter = re.fullmatch(r"Q([1-4])\s+(\d{4})", text, re.IGNORECASE)
    if quarter:
        return pd.Timestamp(year=int(quarter.group(2)), month=3 * int(quarter.group(1)), day=1)

    return pd.to_datetime(text, errors="coerce")


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
        time_label = _time_value(obs.get("dimensions", {}))
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
    def sort_key(row):
        parsed = _parse_ons_time(row[0])
        return (1, str(row[0])) if pd.isna(parsed) else (0, parsed)

    rows.sort(key=sort_key)
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

    @lru_cache(maxsize=1)
    def list_datasets(self) -> list[dict]:
        """List all datasets available via the ONS API."""
        resp = requests.get(f"{ONS_API_BASE}/datasets", timeout=15)
        if not resp.ok:
            raise _response_error(resp, "dataset catalogue request")
        return resp.json().get("items", [])

    @lru_cache(maxsize=64)
    def get_dataset_metadata(self, dataset_id: str) -> dict:
        resp = requests.get(f"{ONS_API_BASE}/datasets/{dataset_id}", timeout=15)
        if not resp.ok:
            raise _response_error(resp, f"metadata request for '{dataset_id}'")
        return resp.json()

    @lru_cache(maxsize=128)
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
        if not resp.ok:
            raise _response_error(resp, f"dimension request for '{dataset_id}'")
        return resp.json().get("items", [])

    @lru_cache(maxsize=512)
    def get_dimension_options(
        self, dataset_id: str, edition: str, version: int, dimension_id: str, limit: int = 200
    ) -> list[dict]:
        resp = requests.get(
            f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{edition}"
            f"/versions/{version}/dimensions/{dimension_id}/options",
            params={"limit": limit},
            timeout=15,
        )
        if not resp.ok:
            raise _response_error(resp, f"option request for '{dataset_id}/{dimension_id}'")
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
        from ONS (get_dataset_dimensions). Every required dimension must
        either be supplied explicitly or have a verified preference. ONS
        returns HTTP 400 when dimensions are missing.

        `preferred_options` lets you bias that auto-fill: for a given
        dimension id, give an ordered list of exact labels or option IDs to
        match case-insensitively. If no verified selection is available,
        the request fails safely instead of choosing an arbitrary series.
        """
        preferred_options = preferred_options or {}
        resolved_edition, resolved_version = self._resolve_edition_version(
            dataset_id, edition, version
        )

        selected_notes = []
        dims = self.get_dataset_dimensions(dataset_id, resolved_edition, resolved_version)
        supplied = {key.casefold(): value for key, value in dimension_filters.items()}
        normalized_filters = {}

        for dim in dims:
            # The live API returns `name`; support `id` for older fixtures.
            dim_id = dim.get("name") or dim.get("id")
            if not dim_id:
                continue
            if dim_id.casefold() in supplied:
                normalized_filters[dim_id] = supplied[dim_id.casefold()]
                continue
            if dim_id.casefold() == "time":
                normalized_filters[dim_id] = "*"
                continue

            options = self.get_dimension_options(
                dataset_id, resolved_edition, resolved_version, dim_id
            )
            chosen = None
            for wanted_label in preferred_options.get(dim_id, []):
                wanted = wanted_label.strip().casefold()
                for opt in options:
                    option_id = opt.get("option") or opt.get("id") or opt.get("node_id")
                    label = str(opt.get("label") or option_id or "").strip().casefold()
                    if label == wanted or str(option_id).casefold() == wanted:
                        chosen = opt
                        break
                if chosen:
                    break
            if chosen is None:
                raise ONSDataError(
                    f"No verified option was configured for required ONS dimension "
                    f"'{dim_id}' in dataset '{dataset_id}'. Refusing to select an "
                    "arbitrary series."
                )
            option_id = chosen.get("option") or chosen.get("id") or chosen.get("node_id")
            if not option_id:
                raise ONSDataError(
                    f"ONS option for '{dim_id}' did not contain an option identifier."
                )
            normalized_filters[dim_id] = option_id
            selected_notes.append(f"{dim_id}='{chosen.get('label', option_id)}'")

        obs_url = (
            f"{ONS_API_BASE}/datasets/{dataset_id}/editions/{resolved_edition}"
            f"/versions/{resolved_version}/observations"
        )
        # Large wildcard series can be slow on the beta API. A 45-second
        # read timeout avoids treating ordinary ONS latency as bad data.
        resp = requests.get(obs_url, params=normalized_filters, timeout=45)
        if not resp.ok:
            raise _response_error(resp, f"observation request for '{dataset_id}'")
        payload = resp.json()
        if not payload.get("observations"):
            raise ONSDataError(
                f"ONS returned zero observations for '{dataset_id}' using "
                f"dimensions {normalized_filters}. URL: {resp.url}"
            )

        notes = (
            "Raw ONS observation payload. Check 'observation level "
            "metadata' in the response for coefficients of variation "
            "before treating values as precise."
        )
        if selected_notes:
            notes += " Verified dimensions: " + "; ".join(selected_notes)

        return FetchResult(
            source_id=self.source.id,
            fetched_at=datetime.now(timezone.utc),
            data=payload,
            provenance_url=resp.url,
            notes=notes,
        )

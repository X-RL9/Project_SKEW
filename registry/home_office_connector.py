"""
Connector for Home Office immigration/migration statistics.

Important difference from ONS: the Home Office does NOT expose a queryable
API for these statistics. It publishes dated spreadsheet releases (XLS/ODS)
on GOV.UK, on a quarterly cadence, under pages like:

    https://www.gov.uk/government/statistical-data-sets/immigration-system-statistics-data-tables
    https://www.gov.uk/government/statistical-data-sets/migration-transparency-data

Each release replaces the previous one's "latest" link but old versions
stay archived. So this connector's job is: given a registered release
page, find the current file link and download+parse it — not "query an
endpoint," because that endpoint doesn't exist.

WHY THIS MATTERS FOR SKEW: your claim → test pipeline can't treat this the
same way it treats ONS. Fetches here should be cached with the release date
attached, and your provenance line needs to say "Home Office release for
[period], published [date]" rather than implying real-time data.

NOTE ON SANDBOXING: not live-tested here (gov.uk isn't reachable from this
build sandbox). Verify the CSS selector logic against the live page once
deployed — GOV.UK page structure occasionally changes between releases.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .base import Connector, FetchResult, Source, SourceType

HOME_OFFICE_RELEASES = {
    "immigration_system_stats": (
        "https://www.gov.uk/government/statistical-data-sets/"
        "immigration-system-statistics-data-tables"
    ),
    "migration_transparency": (
        "https://www.gov.uk/government/statistical-data-sets/"
        "migration-transparency-data"
    ),
}


class HomeOfficeConnector(Connector):
    def __init__(self, source: Optional[Source] = None):
        source = source or Source(
            id="home_office",
            name="Home Office Immigration & Migration Statistics",
            type=SourceType.STATIC_RELEASE,
            org="Home Office",
            description=(
                "Quarterly statistical releases on immigration, asylum, "
                "visas, settlement, and border enforcement. Published as "
                "downloadable spreadsheets, not a live API."
            ),
            base_url="https://www.gov.uk/government/statistical-data-sets/",
            license="Open Government Licence v3.0",
            update_frequency="quarterly",
            topics=[
                "immigration", "asylum", "visas", "settlement",
                "border force", "deportation", "small boats", "work permits",
                "sponsorship", "citizenship",
            ],
            methodology_url=(
                "https://www.gov.uk/government/collections/"
                "migration-statistics"
            ),
        )
        super().__init__(source)

    def _find_latest_file_link(self, release_page_url: str, keyword: str = "") -> tuple[str, str]:
        """
        Scrape a GOV.UK statistical-data-set page for the most recent
        download link. Returns (file_url, link_text).

        If `keyword` is given, prefers the first link whose text contains it
        (e.g. "asylum", "extensions") since these pages list many tables.
        """
        resp = requests.get(release_page_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        candidates = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(href.lower().endswith(ext) for ext in (".xls", ".xlsx", ".ods", ".csv")):
                candidates.append((href, a.get_text(strip=True)))

        if not candidates:
            raise ValueError(f"No downloadable data files found on {release_page_url}")

        if keyword:
            for href, text in candidates:
                if keyword.lower() in text.lower():
                    return href, text

        # fall back to first match (pages usually list newest first)
        return candidates[0]

    def fetch(self, release: str = "immigration_system_stats", keyword: str = "", **_) -> FetchResult:
        """
        release: key into HOME_OFFICE_RELEASES
        keyword: optional text to match a specific table on that page
                 (e.g. "asylum", "settlement", "extensions")
        """
        if release not in HOME_OFFICE_RELEASES:
            raise ValueError(
                f"Unknown release '{release}'. Options: {list(HOME_OFFICE_RELEASES)}"
            )

        page_url = HOME_OFFICE_RELEASES[release]
        file_url, link_text = self._find_latest_file_link(page_url, keyword)

        if not file_url.startswith("http"):
            file_url = "https://www.gov.uk" + file_url

        file_resp = requests.get(file_url, timeout=30)
        file_resp.raise_for_status()

        # Parse with pandas — works for .xls/.xlsx/.ods/.csv given the
        # right engine; caller can re-parse raw_bytes if a different sheet
        # is needed.
        import pandas as pd

        raw_bytes = file_resp.content
        try:
            df = pd.read_excel(io.BytesIO(raw_bytes))
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_bytes))

        return FetchResult(
            source_id=self.source.id,
            fetched_at=datetime.now(timezone.utc),
            data=df,
            provenance_url=file_url,
            notes=(
                f"Downloaded file: '{link_text}'. This is a point-in-time "
                "quarterly release, not live data — check the release "
                "page for the covered period before running time-series "
                "comparisons."
            ),
        )

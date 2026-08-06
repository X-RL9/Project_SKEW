"""
Connector for the Migration Observatory (University of Oxford).

Unlike ONS or Home Office, Migration Observatory does not publish raw,
machine-readable datasets of its own — it produces expert analysis,
briefings, and commentary that itself draws on ONS/Home Office data.

So this is a REFERENCE connector, not a data connector. Its job in Skew's
pipeline is NOT to feed a statistical test — it's to:
  1. surface existing expert analysis on a claim topic, so a human
     reviewer (or the claim-classifier) can check whether an authoritative
     body has already examined this exact claim, and
  2. flag known methodological traps (e.g. "unemployment rate by
     nationality confounds with age structure — see Migration Observatory's
     note on this") that should inform which statistical test you choose.

Calling fetch() here returns links + summaries to read, not numbers to
plug into a regression. Don't feed this into the same pipeline stage as
ONS/Home Office data.

NOTE ON SANDBOXING: search/scrape logic not live-tested (site isn't
reachable from this build sandbox). Migration Observatory has no public
API, so this uses their site search page — check their robots.txt /
terms before scraping in production, and consider reaching out for
permission given you'd be citing them regularly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .base import Connector, FetchResult, Source, SourceType

MIGOBS_SEARCH_URL = "https://migrationobservatory.ox.ac.uk/?s={query}"


class MigrationObservatoryConnector(Connector):
    def __init__(self, source: Optional[Source] = None):
        source = source or Source(
            id="migration_observatory",
            name="Migration Observatory",
            type=SourceType.REFERENCE,
            org="University of Oxford",
            description=(
                "Independent academic analysis of UK migration data. No "
                "raw data API — use for expert commentary, methodology "
                "caveats, and cross-checking claims against existing "
                "authoritative analysis."
            ),
            base_url="https://migrationobservatory.ox.ac.uk",
            license="Check individual publication terms",
            update_frequency="ad hoc (briefings published as needed)",
            topics=[
                "immigration", "migration", "asylum", "labour market",
                "public opinion", "integration", "migrant crime",
                "migrant employment",
            ],
            methodology_url="https://migrationobservatory.ox.ac.uk/about/",
        )
        super().__init__(source)

    def fetch(self, query: str, max_results: int = 5, **_) -> FetchResult:
        """
        Search Migration Observatory's site for briefings relevant to a
        claim. Returns a list of {title, url, snippet} in `.data` — this
        is reading material, not numeric data.
        """
        url = MIGOBS_SEARCH_URL.format(query=requests.utils.quote(query))
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for article in soup.select("article")[:max_results]:
            title_el = article.find(["h2", "h3"])
            link_el = article.find("a", href=True)
            snippet_el = article.find("p")
            if title_el and link_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": link_el["href"],
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })

        return FetchResult(
            source_id=self.source.id,
            fetched_at=datetime.now(timezone.utc),
            data=results,
            provenance_url=url,
            notes=(
                "Reference material only — expert commentary, not raw "
                "data. Use to check for existing analysis and known "
                "confounds before designing your statistical test."
            ),
        )

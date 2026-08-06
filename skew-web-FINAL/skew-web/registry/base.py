"""
Core types for Skew's source registry.

A `Source` is metadata describing where data comes from and what claims
it can speak to. A `Connector` is the thing that actually knows how to
get data out of that source. We split these because sources fall into
genuinely different shapes:

  - LIVE_API:   a queryable REST API (e.g. ONS) — request the exact slice
                of data you need, on demand.
  - STATIC_RELEASE: a government body publishes a spreadsheet/CSV on a
                schedule (e.g. Home Office quarterly releases) — you fetch
                the latest file and parse it yourself, there's no query
                layer.
  - REFERENCE:  a body that publishes analysis/reports but no raw queryable
                data (e.g. Migration Observatory) — useful for citation and
                cross-checking claims against expert commentary, but not
                something you can pull a time series from.

Treating all three as if they were the same "API" is the mistake that
breaks the moment you try to build it. This module keeps them distinct.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    LIVE_API = "live_api"
    STATIC_RELEASE = "static_release"
    REFERENCE = "reference"


@dataclass
class Source:
    """Metadata describing a single data source."""

    id: str
    name: str
    type: SourceType
    org: str
    description: str
    base_url: str
    license: str
    update_frequency: str
    # keywords used to match an incoming claim to relevant sources,
    # e.g. ["unemployment", "jobs", "employment", "labour market"]
    topics: list[str] = field(default_factory=list)
    # optional: link to the human-readable methodology / caveats page,
    # important since your verdicts need to cite limitations
    methodology_url: Optional[str] = None


@dataclass
class FetchResult:
    """Normalized result returned by any connector's fetch()."""

    source_id: str
    fetched_at: datetime
    # raw payload: a pandas DataFrame for tabular data, a dict for API
    # JSON, or None for reference-only sources
    data: Any
    # where this specific data came from (exact URL/query), for the
    # transparency requirement in the operating framework
    provenance_url: str
    notes: Optional[str] = None


class Connector(ABC):
    """Every source type implements this interface."""

    def __init__(self, source: Source):
        self.source = source

    @abstractmethod
    def fetch(self, **params) -> FetchResult:
        """Retrieve data relevant to a claim. Params vary by connector."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.source.name} ({self.source.org}) — {self.source.description}"

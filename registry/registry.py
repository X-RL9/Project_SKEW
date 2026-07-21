"""
SourceRegistry: the lookup layer between "a claim came in" and "here are
the sources + connectors relevant to testing it."

This is deliberately dumb keyword matching for now, not semantic search.
That's a reasonable v1: the claim-classification stage (your next build
step) can call registry.find_relevant(claim_text) to get a shortlist,
then apply smarter NLP on top of that shortlist rather than searching
all of Skew's future sources from scratch every time.
"""

from __future__ import annotations

from typing import Optional

from .base import Connector, FetchResult, Source
from .ons_connector import ONSConnector
from .home_office_connector import HomeOfficeConnector
from .migration_observatory_connector import MigrationObservatoryConnector


class SourceRegistry:
    def __init__(self):
        self._connectors: dict[str, Connector] = {}
        self._register_defaults()

    def _register_defaults(self):
        for connector in (
            ONSConnector(),
            HomeOfficeConnector(),
            MigrationObservatoryConnector(),
        ):
            self.register(connector)

    def register(self, connector: Connector):
        self._connectors[connector.source.id] = connector

    def get(self, source_id: str) -> Connector:
        if source_id not in self._connectors:
            raise KeyError(
                f"No source registered with id '{source_id}'. "
                f"Available: {list(self._connectors)}"
            )
        return self._connectors[source_id]

    def all_sources(self) -> list[Source]:
        return [c.source for c in self._connectors.values()]

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip simple plurals so 'boats' matches 'boat'."""
        words = text.lower().replace("-", " ").split()
        stemmed = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
        return " ".join(stemmed)

    def find_relevant(self, claim_text: str) -> list[Source]:
        """
        Return sources whose topic keywords appear in the claim text,
        ranked by number of keyword matches (most relevant first).

        Matching is substring-on-normalized-text rather than exact phrase
        match, so singular/plural variants ("small boat" vs "small boats")
        both hit. This is still simple keyword matching, not semantic
        search — good enough to shortlist sources for the claim-classifier
        to refine, not precise enough to be the final word on relevance.
        """
        claim_norm = self._normalize(claim_text)
        scored: list[tuple[int, Source]] = []

        for connector in self._connectors.values():
            hits = sum(
                1 for topic in connector.source.topics
                if self._normalize(topic) in claim_norm
            )
            if hits > 0:
                scored.append((hits, connector.source))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [source for _, source in scored]

    def fetch(self, source_id: str, **params) -> FetchResult:
        return self.get(source_id).fetch(**params)

    def fetch_all_relevant(self, claim_text: str, **params) -> list[FetchResult]:
        """
        Convenience: find relevant sources for a claim and fetch from
        each, passing the same params through. Useful for a first pass;
        real usage will likely need per-source params (dataset IDs etc.)
        since ONS/Home Office/MigObs all take different arguments.
        """
        results = []
        for source in self.find_relevant(claim_text):
            try:
                results.append(self.fetch(source.id, **params))
            except Exception as e:
                results.append(
                    FetchResult(
                        source_id=source.id,
                        fetched_at=None,  # type: ignore
                        data=None,
                        provenance_url="",
                        notes=f"Fetch failed: {e}",
                    )
                )
        return results

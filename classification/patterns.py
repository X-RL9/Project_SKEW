"""
Registered claim patterns.

Each pattern is a (trigger, claim_type, tests, confounds, fetch_plan)
bundle. This file is the place where domain judgment calls live — e.g.
"immigrant employment claims need age-structure control" is encoded once,
here, rather than reimplemented ad hoc every time a similar claim comes
through.

Patterns are checked in order; first match wins. Keep more specific
patterns above more general ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .types import ClaimType, FetchPlanItem, StatisticalTest

# --- named-individual detection ------------------------------------------
# Naive but useful: two+ consecutive capitalized words, e.g. "Nigel
# Farage", "Rishi Sunak". This is a heuristic, not a proper NER model —
# it will also false-positive on things like "United Kingdom" or "Home
# Office". That's an acceptable trade for this specific check: it only
# gates whether a claim needs human sign-off before publishing, so
# over-flagging costs a reviewer a few extra seconds, while under-flagging
# risks publishing an unreviewed claim about a real person. A proper NER
# model (spaCy et al.) is worth swapping in once claim volume justifies it.
_NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")


def mentions_named_individual(claim_text: str) -> Optional[str]:
    match = _NAME_PATTERN.search(claim_text)
    return match.group(1) if match else None


@dataclass
class ClaimPattern:
    name: str
    trigger: Callable[[str], bool]
    claim_type: ClaimType
    tests: list[StatisticalTest]
    confounds: list[str]
    fetch_plan: Callable[[str], list[FetchPlanItem]]
    base_review_reason: Optional[str] = None


def _contains_all(claim: str, groups: list[list[str]]) -> bool:
    """True if claim contains at least one keyword from each group."""
    claim_lower = claim.lower()
    return all(any(kw in claim_lower for kw in group) for group in groups)


# --- Pattern: immigration & employment -------------------------------------

def _employment_fetch_plan(claim_text: str) -> list[FetchPlanItem]:
    return [
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "unemployment-rate", "geography": "K02000001", "time": "*"},
            purpose="outcome variable: unemployment/employment rate over time",
        ),
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "population-estimates", "geography": "K02000001", "time": "*"},
            purpose="control variable: population by nationality/region, for confound adjustment",
        ),
        FetchPlanItem(
            source_id="home_office",
            params={"release": "migration_transparency", "keyword": "work"},
            purpose="context: work-visa volumes, for the DiD comparison window",
        ),
        FetchPlanItem(
            source_id="migration_observatory",
            params={"query": claim_text},
            purpose="check for existing expert analysis of this exact claim before testing from scratch",
        ),
    ]


PATTERN_IMMIGRATION_EMPLOYMENT = ClaimPattern(
    name="immigration_employment",
    trigger=lambda c: _contains_all(c, [
        ["immigra", "migrant"],
        ["job", "employ", "unemploy", "work"],
    ]),
    claim_type=ClaimType.ASSOCIATION,
    tests=[
        StatisticalTest.DIFFERENCE_IN_DIFFERENCES,
        StatisticalTest.LINEAR_REGRESSION,
        StatisticalTest.WHITE_TEST_HETEROSCEDASTICITY,
    ],
    confounds=[
        "regional economic conditions (compare against similar regions, not national average)",
        "sector composition (immigration concentrated in specific sectors distorts headline correlation)",
        "time trend (rising immigration and falling unemployment can both trend without causal link)",
    ],
    fetch_plan=_employment_fetch_plan,
)


# --- Pattern: immigration & crime -------------------------------------------

def _crime_fetch_plan(claim_text: str) -> list[FetchPlanItem]:
    return [
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "crime-in-england-and-wales", "geography": "K02000001", "time": "*"},
            purpose="outcome variable: crime rate over time",
        ),
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "population-estimates", "geography": "K02000001", "time": "*"},
            purpose="control variable: age distribution by nationality — required, not optional",
        ),
        FetchPlanItem(
            source_id="home_office",
            params={"release": "migration_transparency", "keyword": "border"},
            purpose="context: small boat / crossing volumes, for the trend window",
        ),
        FetchPlanItem(
            source_id="migration_observatory",
            params={"query": claim_text},
            purpose="check for existing expert analysis and known methodological traps",
        ),
    ]


PATTERN_IMMIGRATION_CRIME = ClaimPattern(
    name="immigration_crime",
    trigger=lambda c: _contains_all(c, [
        ["immigra", "migrant", "asylum", "boat"],
        ["crime", "criminal", "offend", "arrest", "convict"],
    ]),
    claim_type=ClaimType.ASSOCIATION,
    tests=[
        StatisticalTest.CHI_SQUARE,
        StatisticalTest.LINEAR_REGRESSION,
    ],
    confounds=[
        "age distribution (younger populations have higher crime rates regardless of origin — "
        "this is the single most common confound in this claim type and MUST be controlled for)",
        "reporting/recording differences across police forces",
        "definition of 'crime' (all offences vs. specific categories changes the result materially)",
    ],
    fetch_plan=_crime_fetch_plan,
)


# --- Pattern: quote attribution ---------------------------------------------

PATTERN_QUOTE_ATTRIBUTION = ClaimPattern(
    name="quote_attribution",
    trigger=lambda c: _contains_all(c, [
        ["did", "quote", "said", "claim"],
        ["say", "said", "state", "tweet", "post"],
    ]) or bool(re.search(r'"[^"]{5,}"', c)),  # a quoted string of some length
    claim_type=ClaimType.QUOTE_ATTRIBUTION,
    tests=[StatisticalTest.NONE_QUOTE_VERIFICATION],
    confounds=[],
    fetch_plan=lambda c: [],  # no registered source fetches a quote match yet
    base_review_reason="Quote-attribution claims need a dedicated news/transcript search connector, not yet built.",
)


# --- Pattern: opinion / not factual -----------------------------------------

_OPINION_MARKERS = [
    "should", "shouldn't", "ought to", "is bad", "is good", "is wrong",
    "is right", "best", "worst", "i think", "i believe", "in my opinion",
]

PATTERN_OPINION = ClaimPattern(
    name="opinion_not_factual",
    trigger=lambda c: any(m in c.lower() for m in _OPINION_MARKERS),
    claim_type=ClaimType.OPINION,
    tests=[StatisticalTest.NONE_NOT_FACTUAL],
    confounds=[],
    fetch_plan=lambda c: [],
)


ALL_PATTERNS: list[ClaimPattern] = [
    PATTERN_IMMIGRATION_CRIME,       # check crime before employment: "immigrants
    PATTERN_IMMIGRATION_EMPLOYMENT,  # taking jobs and committing crime" should hit both if asked separately
    PATTERN_QUOTE_ATTRIBUTION,
    PATTERN_OPINION,
]

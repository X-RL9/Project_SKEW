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
            params={
                "dataset_id": "labour-market",
                "geography": "K02000001",
                "time": "*",
                "preferred_options": {
                    "seasonaladjustment": ["Seasonally Adjusted"],
                    "economicactivity": ["Unemployed"],
                    "unitofmeasure": ["Rates"],
                    "sex": ["All adults"],
                    "agegroups": ["16+"],
                },
            },
            purpose="outcome variable: unemployment/employment rate over time",
        ),
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "populationestimatestimeseriesdataset", "geography": "K02000001", "time": "*"},
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
            params={"dataset_id": "crimeinenglandandwalesappendixtables", "geography": "K02000001", "time": "*"},
            purpose="outcome variable: crime rate over time",
        ),
        FetchPlanItem(
            source_id="ons",
            params={"dataset_id": "populationestimatestimeseriesdataset", "geography": "K02000001", "time": "*"},
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


# --- Pattern: simple economic trend claims -----------------------------
#
# These cover claims like "inflation has risen", "GDP is growing",
# "wages have gone up", "house prices are falling" — a single metric
# moving over time, not a claimed relationship between two things (that's
# still ASSOCIATION territory, above). The test that fits is a straight
# regression of the metric against time itself: is there a real,
# statistically significant trend, and which direction.
#
# All dataset IDs below were individually confirmed against ONS's own
# dataset pages (not guessed). What's NOT yet confirmed: whether
# ons_connector.parse_timeseries_observations() correctly parses every
# one of these datasets' actual observation shape, since dimension names
# can vary dataset-to-dataset and this hasn't been live-tested.

def _single_ons_series_fetch_plan(dataset_id: str, purpose: str):
    def _plan(claim_text: str) -> list[FetchPlanItem]:
        return [
            FetchPlanItem(
                source_id="ons",
                params={"dataset_id": dataset_id, "geography": "K02000001", "time": "*"},
                purpose=purpose,
            ),
        ]
    return _plan


PATTERN_INFLATION = ClaimPattern(
    name="inflation_trend",
    trigger=lambda c: (
        _contains_all(c, [["inflation", "cost of living", "cpih", "cpi"]])
        or ("prices" in c.lower() and "house" not in c.lower() and "property" not in c.lower())
    ),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=[
        "base-year rebasing (CPIH indices are periodically re-referenced, e.g. to 2015=100 — "
        "compare like-for-like base years across the claimed time window)",
        "which measure is meant (CPIH vs CPI vs RPI give materially different figures)",
    ],
    fetch_plan=_single_ons_series_fetch_plan("cpih01", "outcome variable: CPIH inflation index over time"),
)

PATTERN_GDP = ClaimPattern(
    name="gdp_growth_trend",
    trigger=lambda c: _contains_all(c, [["gdp", "economic growth", "economy grow", "economy shrink", "recession"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=[
        "seasonal adjustment (compare seasonally adjusted figures, not raw)",
        "revisions (early GDP estimates are routinely revised later — check which vintage the claim refers to)",
    ],
    fetch_plan=_single_ons_series_fetch_plan(
        "gdp-to-four-decimal-places", "outcome variable: monthly GDP index over time"
    ),
)

PATTERN_WAGES = ClaimPattern(
    name="wages_earnings_trend",
    trigger=lambda c: _contains_all(c, [["wage", "earning", "pay ", "salary", "salaries"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=[
        "nominal vs real terms (wage claims often conflate cash pay rises with inflation-adjusted "
        "purchasing power — check which one the claim actually means)",
        "regular pay vs total pay (bonuses swing 'total pay' figures much more than regular pay)",
    ],
    fetch_plan=_single_ons_series_fetch_plan("averageweeklyearnings", "outcome variable: average weekly earnings over time"),
)

PATTERN_HOUSE_PRICES = ClaimPattern(
    name="house_prices_trend",
    trigger=lambda c: _contains_all(c, [["house price", "housing market", "property price", "house prices"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=[
        "regional variation (a UK-wide average can mask opposite trends in different regions)",
        "this ONS dataset is a summary — HM Land Registry runs the fuller authoritative UK HPI "
        "separately; cross-check there for anything published as an official headline figure",
    ],
    fetch_plan=_single_ons_series_fetch_plan(
        "ukhousepriceindexmonthlypricestatistics", "outcome variable: UK house price index over time"
    ),
)

PATTERN_PRODUCTIVITY = ClaimPattern(
    name="productivity_trend",
    trigger=lambda c: _contains_all(c, [["productivity", "output per hour", "output per worker"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=["industry mix (a UK-wide figure blends very different sectors — check whether the claim is sector-specific)"],
    fetch_plan=_single_ons_series_fetch_plan("labourproductivity", "outcome variable: UK output per hour over time"),
)

PATTERN_INEQUALITY = ClaimPattern(
    name="inequality_trend",
    trigger=lambda c: _contains_all(c, [["inequality", "gini", "income gap", "wealth gap"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=["before vs after housing costs (the Gini coefficient differs materially depending on which is used)"],
    fetch_plan=_single_ons_series_fetch_plan(
        "householddisposableincomeandinequality", "outcome variable: Gini coefficient over time"
    ),
)

PATTERN_POPULATION_TREND = ClaimPattern(
    name="population_trend",
    trigger=lambda c: _contains_all(c, [["population"]]) and not _contains_all(c, [["immigra", "migrant"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=["mid-year estimates are provisional and revised later — check which vintage the claim uses"],
    fetch_plan=_single_ons_series_fetch_plan(
        "populationestimatestimeseriesdataset", "outcome variable: UK population over time"
    ),
)

PATTERN_TRADE = ClaimPattern(
    name="trade_trend",
    trigger=lambda c: _contains_all(c, [["trade deficit", "trade surplus", "export", "import"]]),
    claim_type=ClaimType.TREND,
    tests=[StatisticalTest.LINEAR_REGRESSION],
    confounds=["this dataset ('trade') covers goods by country/commodity specifically — "
               "confirm it matches the claim's scope before treating it as the overall UK trade balance"],
    fetch_plan=_single_ons_series_fetch_plan("trade", "outcome variable: UK goods trade over time"),
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
    PATTERN_HOUSE_PRICES,             # kept above inflation: both can trigger on "prices"
    PATTERN_INFLATION,
    PATTERN_GDP,
    PATTERN_WAGES,
    PATTERN_PRODUCTIVITY,
    PATTERN_INEQUALITY,
    PATTERN_TRADE,
    PATTERN_POPULATION_TREND,        # generic "population" — kept below the immigration
                                      # patterns so immigration-flavoured population claims
                                      # still hit those more specific rules first
    PATTERN_QUOTE_ATTRIBUTION,
    PATTERN_OPINION,
]

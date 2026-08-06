"""
Core types for Skew's claim classifier.

This sits between the source registry (which sources are relevant?) and
the actual statistical pipeline (run the test, get a verdict). Its job:
given claim text, decide

  1. what KIND of claim this is (an association claim about jobs? a
     quote-attribution claim? not a factual claim at all?)
  2. which statistical test(s) fit that kind of claim
  3. what known confounds to control for — this is the part that came
     directly out of the earlier framework discussion (e.g. age
     structure confounding immigrant crime/employment comparisons) and
     is worth encoding explicitly rather than leaving to whoever writes
     the test code later to remember
  4. a concrete fetch plan: which registered source(s) and what params

Deliberately NOT using an LLM call here as the primary classifier. Two
reasons: (a) the statistical test choice and confound list need to be
consistent and auditable — a fixed rule firing is easier to defend in a
correction/dispute than "the model decided," and (b) this is exactly the
kind of decision the framework agreed should be reproducible, not a
black box. An LLM is a good *fallback* for claims that don't match any
rule (see classifier.py) — flagged as ai_assisted so it goes through the
extra review lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimType(str, Enum):
    ASSOCIATION = "association"            # "X is linked to / causes / increases Y"
    POPULATION_STAT = "population_stat"     # "there are N of X" / simple magnitude claims
    TREND = "trend"                         # "X has gone up/down since..."
    QUOTE_ATTRIBUTION = "quote_attribution"  # "did person say X"
    OPINION = "opinion"                     # value judgment, not checkable
    UNKNOWN = "unknown"                     # didn't match any rule


class StatisticalTest(str, Enum):
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    LINEAR_REGRESSION = "linear_regression"
    TWO_SAMPLE_T_TEST = "two_sample_t_test"
    CHI_SQUARE = "chi_square"
    TIME_SERIES_CORRELATION = "time_series_correlation"
    WHITE_TEST_HETEROSCEDASTICITY = "white_test_heteroscedasticity"
    NONE_QUOTE_VERIFICATION = "none_quote_verification"  # not a stats test at all
    NONE_NOT_FACTUAL = "none_not_factual"


@dataclass
class FetchPlanItem:
    """One source + the params to call registry.fetch() with."""
    source_id: str
    params: dict = field(default_factory=dict)
    purpose: str = ""  # why this fetch, e.g. "outcome variable" vs "control variable"


@dataclass
class ClassificationResult:
    claim_text: str
    claim_type: ClaimType
    recommended_tests: list[StatisticalTest]
    confounds_to_control: list[str]
    fetch_plan: list[FetchPlanItem]
    requires_human_review: bool
    review_reason: Optional[str] = None
    matched_rule: Optional[str] = None  # which pattern fired, for auditability
    confidence: str = "rule_matched"     # "rule_matched" | "ai_assisted" | "no_match"

"""
Core types for Skew's statistical pipeline.

This is the layer that turns fetched data + a ClassificationResult into
an actual verdict. Two design decisions worth being explicit about:

1. CLAIM DIRECTION IS AN INPUT, NOT AN INFERENCE. Deciding whether "the
   coefficient came out positive" means a claim is SUPPORTED or
   CONTRADICTED depends on what the claim actually asserted — and that's
   a natural-language judgment call, not a statistics problem. Silently
   guessing this from claim text risks the exact kind of automation
   failure the framework discussion was trying to avoid (an unreviewed
   AI verdict about a contested factual claim). So `expected_direction`
   is a required, explicit parameter someone (a human, or a well-tested
   claim-parser you build and validate later) has to set — not something
   this module invents.

2. MULTIPLE TESTS -> ONE VERDICT uses Bonferroni correction on the
   p-values before deciding significance, per the framework's agreement
   that claims should get more than one test where possible. This is
   the conservative choice (raises the bar for "Supported"), which is
   the right default for a fact-checker's reputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MIXED = "mixed"
    UNPROVEN = "unproven"
    INSUFFICIENT_DATA = "insufficient_data"
    NOT_A_FACTUAL_CLAIM = "not_a_factual_claim"


Direction = Literal["positive", "negative"]  # the direction the claim asserts


@dataclass
class TestResult:
    """Output of a single statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_direction: Optional[Direction]  # sign of the estimated effect, if applicable
    effect_size: Optional[float] = None
    n_observations: Optional[int] = None
    caveats: list[str] = field(default_factory=list)
    raw: Optional[dict] = None  # full underlying output, for auditability


@dataclass
class VerdictResult:
    claim_text: str
    verdict: Verdict
    hedge_statement: str  # the actual public-facing hedged sentence
    test_results: list[TestResult]
    bonferroni_alpha: Optional[float] = None
    confounds_noted: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_reason: Optional[str] = None

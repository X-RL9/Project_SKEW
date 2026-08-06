"""
ClaimClassifier — ties the registered patterns together and applies the
one review rule that isn't claim-type-specific: any claim naming a real
individual gets flagged for human review before publication, regardless
of which statistical pattern it matched. That was agreed as a baseline
protection in the operating framework, so it's enforced here rather than
left to each pattern to remember.
"""

from __future__ import annotations

from .patterns import ALL_PATTERNS, mentions_named_individual
from .types import ClassificationResult, ClaimType, StatisticalTest


class ClaimClassifier:
    def __init__(self, patterns=None):
        self.patterns = patterns if patterns is not None else ALL_PATTERNS

    def classify(self, claim_text: str) -> ClassificationResult:
        named_individual = mentions_named_individual(claim_text)

        for pattern in self.patterns:
            if pattern.trigger(claim_text):
                requires_review = bool(named_individual) or bool(pattern.base_review_reason)
                reasons = []
                if pattern.base_review_reason:
                    reasons.append(pattern.base_review_reason)
                if named_individual:
                    reasons.append(
                        f"Claim names an individual ('{named_individual}') — "
                        "requires human sign-off before publishing per Skew's review policy."
                    )

                return ClassificationResult(
                    claim_text=claim_text,
                    claim_type=pattern.claim_type,
                    recommended_tests=pattern.tests,
                    confounds_to_control=pattern.confounds,
                    fetch_plan=pattern.fetch_plan(claim_text),
                    requires_human_review=requires_review,
                    review_reason=" / ".join(reasons) if reasons else None,
                    matched_rule=pattern.name,
                    confidence="rule_matched",
                )

        # No pattern matched. Flag for human triage rather than guessing —
        # this is the honest "we don't have a rule for this yet" case,
        # and it's exactly the kind of gap that should surface so you can
        # decide whether to add a new pattern.
        return ClassificationResult(
            claim_text=claim_text,
            claim_type=ClaimType.UNKNOWN,
            recommended_tests=[],
            confounds_to_control=[],
            fetch_plan=[],
            requires_human_review=True,
            review_reason=(
                "No registered claim pattern matched. Needs manual "
                "classification, and — if this claim type recurs — a new "
                "pattern added to classification/patterns.py."
            ),
            matched_rule=None,
            confidence="no_match",
        )

"""
VerdictEngine — combines one or more TestResults into a single verdict
from the six-category taxonomy agreed in the operating framework:
Supported, Contradicted, Mixed, Unproven, Insufficient Data,
Not a Factual Claim.

Key design choices:

- `expected_direction` is required and must come from whoever parsed the
  claim (human or a validated claim-parser) — see the module docstring
  in types.py for why this isn't inferred here.
- Only DIRECTIONAL tests (regression, DiD, t-test, correlation) count
  toward the Supported/Contradicted decision. Diagnostic-only tests
  (White's test) and non-directional tests (chi-square) inform caveats
  and confidence but can't themselves flip a verdict, since they don't
  have a "direction" to agree or disagree with the claim.
- Bonferroni correction is applied across the directional tests actually
  used for the decision, before checking significance. This is the
  conservative choice — it makes "Supported" harder to reach than
  eyeballing one test's p-value would, which is the right default for a
  reputation that depends on not overclaiming.
"""

from __future__ import annotations

from .types import Direction, TestResult, Verdict, VerdictResult

DIRECTIONAL_TESTS = {
    "linear_regression", "difference_in_differences",
    "two_sample_t_test", "time_series_correlation",
}
DIAGNOSTIC_ONLY_TESTS = {"white_test_heteroscedasticity"}
ASSOCIATION_ONLY_TESTS = {"chi_square"}  # significant but no direction


class VerdictEngine:
    def __init__(self, base_alpha: float = 0.05):
        self.base_alpha = base_alpha

    def decide(
        self,
        claim_text: str,
        test_results: list[TestResult],
        expected_direction: Direction | None,
        confounds_noted: list[str] | None = None,
        data_sources: list[str] | None = None,
        requires_human_review: bool = False,
        review_reason: str | None = None,
    ) -> VerdictResult:
        confounds_noted = confounds_noted or []
        data_sources = data_sources or []

        if not test_results:
            return VerdictResult(
                claim_text=claim_text,
                verdict=Verdict.INSUFFICIENT_DATA,
                hedge_statement=(
                    "There is not enough data available to statistically "
                    "evaluate this claim."
                ),
                test_results=[],
                confounds_noted=confounds_noted,
                data_sources=data_sources,
                requires_human_review=requires_human_review,
                review_reason=review_reason,
            )

        directional = [t for t in test_results if t.test_name in DIRECTIONAL_TESTS]
        diagnostics = [t for t in test_results if t.test_name in DIAGNOSTIC_ONLY_TESTS]
        associations = [t for t in test_results if t.test_name in ASSOCIATION_ONLY_TESTS]

        # heteroscedasticity check: if White's test is significant, the
        # regression's p-value is unreliable as reported. Flag it — don't
        # silently trust a "Supported" built on broken standard errors.
        heteroscedastic_warning = any(
            d.p_value < self.base_alpha for d in diagnostics
        )

        if expected_direction is None or not directional:
            # No directional test available (e.g. only chi-square ran) —
            # we can say "there IS a significant association" at most,
            # not which way the claim's direction points. That's Mixed
            # at best, since we can't confirm the specific claim.
            if associations and any(a.p_value < self.base_alpha for a in associations):
                verdict = Verdict.MIXED
                hedge = (
                    "The data shows a statistically significant association, "
                    "but the tests run cannot confirm the specific direction "
                    "the claim asserts. Further directional testing is needed."
                )
            else:
                verdict = Verdict.UNPROVEN
                hedge = (
                    "The statistical tests run do not show a significant "
                    "effect either supporting or contradicting this claim."
                )
            return VerdictResult(
                claim_text=claim_text,
                verdict=verdict,
                hedge_statement=hedge,
                test_results=test_results,
                confounds_noted=confounds_noted,
                data_sources=data_sources,
                requires_human_review=requires_human_review,
                review_reason=review_reason,
            )

        # Bonferroni correction across directional tests used for the decision
        n_tests = len(directional)
        corrected_alpha = self.base_alpha / n_tests

        significant = [t for t in directional if t.p_value < corrected_alpha]
        agreeing = [t for t in significant if t.effect_direction == expected_direction]
        opposing = [t for t in significant if t.effect_direction != expected_direction]

        if not significant:
            verdict = Verdict.UNPROVEN
            hedge = (
                f"Based on {n_tests} statistical test(s), the data "
                "does not show a statistically significant effect in either "
                "direction. This claim is currently unproven, not disproven — "
                "a null result with the available data and tests."
            )
        elif agreeing and not opposing:
            verdict = Verdict.SUPPORTED
            hedge = (
                f"The statistical tests suggest evidence consistent with this "
                f"claim ({len(agreeing)} of {n_tests} test(s) statistically "
                f"significant at the 5% level, in the direction the "
                "claim asserts)."
            )
        elif opposing and not agreeing:
            verdict = Verdict.CONTRADICTED
            hedge = (
                f"The statistical tests suggest evidence against this claim "
                f"({len(opposing)} of {n_tests} test(s) statistically "
                "significant at the 5% level, in the opposite direction "
                "to what the claim asserts)."
            )
        else:
            verdict = Verdict.MIXED
            hedge = (
                f"The statistical tests give mixed signals: {len(agreeing)} "
                f"test(s) support the claim's direction and {len(opposing)} "
                f"contradict it, out of {n_tests} total. This suggests the "
                "true picture is more complicated than the claim implies."
            )

        if heteroscedastic_warning:
            hedge += (
                " Note: a diagnostic test detected non-constant error variance "
                "in the underlying regression, which means its reported "
                "significance may be unreliable until re-run with robust "
                "standard errors — treat this verdict as provisional."
            )

        return VerdictResult(
            claim_text=claim_text,
            verdict=verdict,
            hedge_statement=hedge,
            test_results=test_results,
            bonferroni_alpha=corrected_alpha,
            confounds_noted=confounds_noted,
            data_sources=data_sources,
            requires_human_review=requires_human_review or heteroscedastic_warning,
            review_reason=review_reason or (
                "Heteroscedasticity detected — needs re-run with robust SEs "
                "before this verdict is safe to publish as-is."
                if heteroscedastic_warning else None
            ),
        )

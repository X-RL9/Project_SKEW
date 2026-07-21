"""
SkewPipeline — the single callable that wires together everything built
so far:

    claim text
      -> ClaimClassifier.classify()        (what kind of claim, which tests, confounds, fetch plan)
      -> SourceRegistry.fetch() per item    (pull the data)
      -> stat_tests.*                       (run the recommended tests)
      -> VerdictEngine.decide()             (produce a verdict)

What this orchestrator does NOT solve (both already flagged, repeated
here because this is where they actually bite):

1. There's no generic "FetchResult -> x/y Series" adapter, because each
   dataset shape is different (ONS observations JSON vs. a Home Office
   spreadsheet vs. Migration Observatory's reading list). This class
   accepts an optional `data_adapter` callable per source_id so you can
   plug in real parsing logic once you've confirmed real API/response
   shapes — without one, fetched data is passed through as-is and the
   test functions will likely need it pre-shaped anyway.
2. `expected_direction` still isn't inferred — same reasoning as
   VerdictEngine. You pass it in when you call run().
3. Confound *control variables* aren't automatically wired into the
   regression — the classifier tells you what to control for, but this
   orchestrator doesn't yet turn "control for age structure" into an
   extra regressor. That's real modeling work per claim type, not
   something to fake here.

Given (1) and (2), `run()` supports two modes:
  - `test_data={...}` : you provide pre-fetched, pre-shaped data directly
    (what the demo below uses, since live fetches aren't testable from
    this sandbox)
  - no `test_data` : it will actually call the registry's fetch() per the
    classifier's fetch_plan — untested against live sources for the same
    network-access reason as the connectors themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from classification import ClaimClassifier, ClaimType
from pipeline.types import Direction, VerdictResult
from pipeline.stat_tests import TEST_REGISTRY
from pipeline.verdict import VerdictEngine
from registry import SourceRegistry


@dataclass
class PipelineRunResult:
    claim_text: str
    classification_matched_rule: Optional[str]
    verdict_result: VerdictResult


class SkewPipeline:
    def __init__(
        self,
        classifier: Optional[ClaimClassifier] = None,
        registry: Optional[SourceRegistry] = None,
        verdict_engine: Optional[VerdictEngine] = None,
    ):
        self.classifier = classifier or ClaimClassifier()
        self.registry = registry or SourceRegistry()
        self.verdict_engine = verdict_engine or VerdictEngine()

    def run(
        self,
        claim_text: str,
        expected_direction: Optional[Direction] = None,
        test_data: Optional[dict[str, Any]] = None,
        data_adapters: Optional[dict[str, Callable]] = None,
    ) -> PipelineRunResult:
        """
        test_data: maps test_name -> kwargs dict for that test function,
            e.g. {"linear_regression": {"x": series_a, "y": series_b}}.
            Use this to bypass live fetching (required in this sandbox,
            and useful generally for re-running a test with corrected data).

        data_adapters: maps source_id -> callable(FetchResult) -> dict of
            kwargs ready for the relevant test function(s). Only used if
            test_data isn't provided. Without an adapter for a given
            source, its FetchResult is fetched but not automatically
            usable by the stat tests — flagged in the result's caveats.
        """
        classification = self.classifier.classify(claim_text)

        # Opinions and unclassifiable claims short-circuit before any
        # fetching or testing — there's nothing to test.
        if classification.claim_type == ClaimType.OPINION:
            verdict = self.verdict_engine.decide(
                claim_text=claim_text,
                test_results=[],
                expected_direction=None,
                requires_human_review=False,
            )
            verdict.verdict = verdict.verdict  # INSUFFICIENT_DATA by default from empty tests
            from pipeline.types import Verdict
            verdict.verdict = Verdict.NOT_A_FACTUAL_CLAIM
            verdict.hedge_statement = (
                "This is a statement of opinion or value judgment, not a "
                "checkable factual claim."
            )
            return PipelineRunResult(
                claim_text=claim_text,
                classification_matched_rule=classification.matched_rule,
                verdict_result=verdict,
            )

        if classification.claim_type == ClaimType.QUOTE_ATTRIBUTION:
            verdict = self.verdict_engine.decide(
                claim_text=claim_text,
                test_results=[],
                expected_direction=None,
                requires_human_review=True,
                review_reason=classification.review_reason,
            )
            return PipelineRunResult(
                claim_text=claim_text,
                classification_matched_rule=classification.matched_rule,
                verdict_result=verdict,
            )

        if classification.claim_type == ClaimType.UNKNOWN:
            verdict = self.verdict_engine.decide(
                claim_text=claim_text,
                test_results=[],
                expected_direction=expected_direction,
                requires_human_review=True,
                review_reason=classification.review_reason,
            )
            return PipelineRunResult(
                claim_text=claim_text,
                classification_matched_rule=classification.matched_rule,
                verdict_result=verdict,
            )

        # --- gather data -----------------------------------------------
        data_sources_used = [item.source_id for item in classification.fetch_plan]

        if test_data is None:
            # Live-fetch path — untested in this sandbox (no network
            # access to ONS/gov.uk). Included so this is real, callable
            # code once deployed, not a stub.
            data_adapters = data_adapters or {}
            fetched = {}
            for item in classification.fetch_plan:
                fetch_result = self.registry.fetch(item.source_id, **item.params)
                adapter = data_adapters.get(item.source_id)
                if adapter:
                    fetched.update(adapter(fetch_result))
                # without an adapter, fetch_result is retrieved but not
                # automatically wired into a test — see class docstring
            test_data = fetched

        # --- run recommended tests --------------------------------------
        test_results = []
        skipped_tests = []
        for test_enum in classification.recommended_tests:
            test_name = test_enum.value
            run_fn = TEST_REGISTRY.get(test_name)
            if run_fn is None:
                skipped_tests.append(f"{test_name} (no test implementation registered)")
                continue
            kwargs = test_data.get(test_name) if test_data else None
            if not kwargs:
                skipped_tests.append(f"{test_name} (no data provided/fetched for this test)")
                continue
            test_results.append(run_fn(**kwargs))

        verdict = self.verdict_engine.decide(
            claim_text=claim_text,
            test_results=test_results,
            expected_direction=expected_direction,
            confounds_noted=classification.confounds_to_control,
            data_sources=data_sources_used,
            requires_human_review=classification.requires_human_review,
            review_reason=classification.review_reason,
        )

        if skipped_tests:
            note = "Tests recommended but not run: " + "; ".join(skipped_tests)
            verdict.hedge_statement += f" [{note}]"

        return PipelineRunResult(
            claim_text=claim_text,
            classification_matched_rule=classification.matched_rule,
            verdict_result=verdict,
        )

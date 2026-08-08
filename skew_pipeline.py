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
import re
from typing import Any, Callable, Optional

import pandas as pd

from classification import ClaimClassifier, ClaimType
from pipeline.types import Direction, Verdict, VerdictResult
from pipeline.stat_tests import TEST_REGISTRY
from pipeline.verdict import VerdictEngine
from registry import SourceRegistry


@dataclass
class PipelineRunResult:
    claim_text: str
    classification_matched_rule: Optional[str]
    verdict_result: VerdictResult
    analysis_details: Optional[dict[str, Any]] = None


def _claim_window(claim_text: str, latest: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    """Return start/end dates and whether the user supplied the period."""
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", claim_text)]
    if len(years) >= 2:
        return pd.Timestamp(year=min(years), month=1, day=1), pd.Timestamp(
            year=max(years), month=12, day=31
        ), True
    if len(years) == 1:
        return pd.Timestamp(year=years[0], month=1, day=1), latest, True
    last_years = re.search(r"(?:last|past)\s+(\d+)\s+years?", claim_text, re.I)
    if last_years:
        return latest - pd.DateOffset(years=int(last_years.group(1))), latest, True
    return latest - pd.DateOffset(years=5), latest, False


def _period_label(value: pd.Timestamp) -> str:
    return value.strftime("%B %Y")


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
        analysis_details = None

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
                elif (
                    classification.claim_type == ClaimType.TREND
                    and item.source_id == "ons"
                    and "linear_regression" not in fetched
                ):
                    # TREND claims (inflation/GDP/wages/etc.) are a single
                    # ONS time series checked against time itself — we know
                    # the expected observation shape (see
                    # ons_connector.parse_timeseries_observations), so wire
                    # this automatically rather than requiring the caller
                    # to supply a custom adapter for every trend pattern.
                    from registry.ons_connector import parse_timeseries_points
                    points = parse_timeseries_points(fetch_result).dropna(subset=["date"])

                    # CPIH is an index level. Convert it to the economically
                    # meaningful 12-month inflation rate before testing an
                    # inflation claim.
                    is_inflation = classification.matched_rule == "inflation_trend"
                    if is_inflation:
                        points = points.copy()
                        points["value"] = points["value"].pct_change(12) * 100
                        points = points.dropna(subset=["value"])

                    latest = points["date"].max()
                    start, end, user_supplied_period = _claim_window(claim_text, latest)
                    selected = points[
                        (points["date"] >= start) & (points["date"] <= end)
                    ].copy()
                    if len(selected) < 3:
                        raise ValueError(
                            f"Only {len(selected)} observations are available between "
                            f"{_period_label(start)} and {_period_label(end)}."
                        )

                    time_index = pd.Series(range(len(selected)))
                    values = selected["value"].reset_index(drop=True)
                    fetched["linear_regression"] = {"x": time_index, "y": values}

                    first_value = float(values.iloc[0])
                    last_value = float(values.iloc[-1])
                    if is_inflation:
                        change = last_value - first_value
                        change_unit = "percentage points"
                    else:
                        change = ((last_value / first_value) - 1) * 100 if first_value else None
                        change_unit = "percent"
                    claimed = re.search(r"(-?\d+(?:\.\d+)?)\s*%", claim_text)
                    if is_inflation:
                        series_name = "the 12-month UK CPIH inflation rate"
                    elif classification.matched_rule == "gdp_growth_trend":
                        series_name = "UK monthly GDP"
                    elif classification.matched_rule == "trade_trend":
                        series_name = (
                            "UK goods imports" if item.params.get("direction") == "IM"
                            else "UK goods exports"
                        )
                    else:
                        series_name = item.purpose.replace("outcome variable: ", "")
                    analysis_details = {
                        "method": "ordinary least-squares linear time-trend regression",
                        "hypothesis_test": "two-sided test of whether the regression trend is zero",
                        "significance_level": "5%",
                        "period_start": _period_label(selected["date"].iloc[0]),
                        "period_end": _period_label(selected["date"].iloc[-1]),
                        "period_was_user_supplied": user_supplied_period,
                        "used_default_five_year_period": not user_supplied_period,
                        "n_observations": len(selected),
                        "first_value": first_value,
                        "last_value": last_value,
                        "observed_change": change,
                        "change_unit": change_unit,
                        "claimed_percentage_change": float(claimed.group(1)) if claimed else None,
                        "measure": "12-month CPIH inflation rate" if is_inflation else item.purpose,
                        "series_name": series_name,
                        "unit_of_measure": (
                            "Percent" if is_inflation else str(
                                fetch_result.data.get("unit_of_measure") or ""
                            ).replace("Â£", "£")
                        ),
                        "source_name": "Office for National Statistics",
                        "source_url": fetch_result.provenance_url,
                    }
                # otherwise fetch_result is retrieved but not automatically
                # wired into a test — see class docstring
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

        if analysis_details and test_results:
            primary = test_results[0]
            analysis_details.update({
                "p_value": primary.p_value,
                "effect_direction": primary.effect_direction,
                "slope": primary.effect_size,
                "confidence_interval_95": (
                    primary.raw or {}
                ).get("slope_confidence_interval_95"),
                "r_squared": (primary.raw or {}).get("r_squared"),
            })
            claimed_change = analysis_details.get("claimed_percentage_change")
            observed_change = analysis_details.get("observed_change")
            if claimed_change is not None and observed_change is not None:
                difference = observed_change - claimed_change
                magnitude_matches = abs(difference) <= 0.5
                analysis_details.update({
                    "magnitude_difference_percentage_points": difference,
                    "claimed_magnitude_matches_calculation": magnitude_matches,
                    "directional_verdict": verdict.verdict.value,
                })
                if not magnitude_matches:
                    verdict.verdict = Verdict.MIXED
                    verdict.hedge_statement = (
                        "The direction of the claim is supported, but its stated "
                        f"magnitude is not: the data shows {observed_change:.1f}% "
                        f"rather than {claimed_change:.1f}% over the tested period."
                    )

        return PipelineRunResult(
            claim_text=claim_text,
            classification_matched_rule=classification.matched_rule,
            verdict_result=verdict,
            analysis_details=analysis_details,
        )

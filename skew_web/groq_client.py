"""
Groq-backed chat layer for Skew.

The core rule this file exists to enforce: the LLM may chat, clarify,
and explain — but it can never state a verdict, statistic, or number
about a factual claim except by calling `run_skew_verdict` and reporting
exactly what came back. That's not a suggestion in the system prompt,
it's the whole point of building this as tool-use rather than just
piping questions straight to a chat model.

Uses Groq's free tier (OpenAI-compatible API) with Llama 3.3 70B, which
supports tool/function calling. Needs a GROQ_API_KEY environment
variable — get one free, no card required, at console.groq.com.
"""

from __future__ import annotations

import json
import os
import traceback

import requests

from classification import ClaimClassifier, ClaimType
from skew_pipeline import SkewPipeline

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

_classifier = ClaimClassifier()
_pipeline = SkewPipeline()

SKEW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_skew_verdict",
        "description": (
            "Run Skew's real statistical fact-checking pipeline on a single, "
            "clearly-stated factual claim about UK macro-economic, social, or "
            "immigration statistics. This is the ONLY way to get a verdict — "
            "never state a verdict, statistic, or conclusion yourself without "
            "calling this first and reporting exactly what it returns. If the "
            "claim describes a relationship or trend, determine the direction "
            "from the claim's own wording whenever it's reasonably clear "
            "('push up'/'increase' = positive, 'push down'/'reduce' = "
            "negative) and call this immediately — only ask the user first if "
            "the claim truly gives no directional cue at all."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_text": {
                    "type": "string",
                    "description": "The claim, restated clearly and specifically enough to classify and test.",
                },
                "expected_direction": {
                    "type": "string",
                    "enum": ["positive", "negative"],
                    "description": (
                        "The direction the claim asserts, ONLY if the user has "
                        "explicitly stated it (e.g. they said 'increases' or "
                        "'decreases', 'risen' or 'fallen'). Omit this entirely "
                        "if the claim doesn't need a direction, if the user said "
                        "they're unsure/don't know, or if you haven't yet asked "
                        "which direction they mean — the tool still runs and "
                        "returns real test data either way, just without a "
                        "supported/contradicted verdict label."
                    ),
                },
            },
            "required": ["claim_text"],
        },
    },
}

SYSTEM_PROMPT = """You are Skew, an AI that fact-checks UK statistical claims. Refer to yourself simply as "Skew". Never call yourself "Skew chat". Never mention "the tool", "the pipeline", "calling a function", or any internal mechanism to the user — just speak as though you did the analysis yourself (e.g. "I checked this against ONS data and found..."), since exposing that architecture to the user adds nothing for them.

Your ONLY source of truth for any verdict, statistic, or factual conclusion is the run_skew_verdict tool. You must never state whether a claim is true, false, supported, or contradicted from your own knowledge — always call it and report exactly what it returns, including its hedged language, confounds, and sources.

NEVER describe your own outputs as "accurate", "reliable", "trustworthy", or similar unqualified guarantees. You run real statistical tests against official government data and report hedged, uncertain findings — describe it factually (e.g. "I ran a regression against ONS data and it shows...") never as a guarantee of correctness. This isn't a style preference — overclaiming accuracy is a real legal and credibility risk.

Rules:
1. If a claim describes a relationship ("X increases/decreases Y") or a trend ("Y has risen/fallen"), determine the direction directly from the claim's own wording wherever it's reasonably clear, and go straight to calling the tool — do NOT ask the user to clarify if the wording already states or clearly implies a direction. Examples: "push up", "increase", "boost", "raise", "worsen" -> positive. "push down", "decrease", "reduce", "lower", "improve" (for a bad thing like unemployment) -> negative. Use your best reading of ordinary English; you don't need certainty, just a reasonable interpretation.
2. Only ask the user to clarify direction if the claim is GENUINELY ambiguous with no directional language at all (e.g. "does immigration affect unemployment" — no up/down implied either way) AND getting it wrong would materially change the answer. Even then, if the user says they don't know or are unsure, do not keep asking — call the tool anyway with expected_direction omitted, then report the actual empirical result (its real direction and significance) as a plain descriptive finding rather than a supported/contradicted verdict.
3. Never ask more than one clarifying question before attempting the claim — get straight to a real answer as fast as possible.
4. If the tool returns an error (e.g. a live data fetch failed), say so honestly — don't paper over it or make up a number instead. ALWAYS include the exact raw error text the tool returned (in full, even if technical) so the user can share it for debugging, alongside your plain-English explanation.
5. If the tool reports requires_human_review=True, tell the user this finding needs human sign-off before being treated as final.
6. Keep responses conversational, direct, and concise — lead with the actual finding, not a preamble.
7. For every successful statistical result, format the answer in Markdown using separate paragraphs in this exact order:
   - First, a bold plain-English outcome stating supported, contradicted, or unproven; the measured direction and calculated change; and the exact period. Never repeat a percentage asserted by the user as though it was calculated. Use `observed_change` from `analysis_details`.
   - Second, explain the named statistical method, observation count, estimated direction, hypothesis tested, 5% significance level, p-value, and 95% confidence interval when supplied. Do not mention Bonferroni in the public answer.
   - Third, write `**Data source:**`, name the source, and on the next line include a Markdown link labelled `View the exact ONS data request` whose target is `source_url` exactly.
   - Fourth, write `**Points to consider:**` and preserve the supplied confounds and caveats in accessible language.
   - Finally, if `period_was_user_supplied` is false, state the default five-year period and ask: `Would you like me to test a different period or adjust the values for inflation?` If a period was supplied, offer to test another period.
8. Always name every statistical test actually run. Never describe a multiple-testing adjustment as a statistical test.
9. If `claimed_percentage_change` is present, compare it explicitly with `observed_change`. A direction-only result does not establish that the claimed magnitude is correct.
"""


def _run_tool(claim_text: str, expected_direction: str | None = None) -> dict:
    """Actually call the real pipeline. Returns a JSON-serializable dict —
    never lets an exception escape silently, so the LLM always gets an
    honest result to report, success or failure."""
    try:
        result = _pipeline.run(claim_text, expected_direction=expected_direction)
        vr = result.verdict_result
        return {
            "ok": True,
            "verdict": vr.verdict.value,
            "hedge_statement": vr.hedge_statement,
            "confounds_noted": vr.confounds_noted,
            "data_sources": vr.data_sources,
            "requires_human_review": vr.requires_human_review,
            "review_reason": vr.review_reason,
            "matched_rule": result.classification_matched_rule,
            "raw_test_results": [
                {
                    "test_name": t.test_name,
                    "p_value": t.p_value,
                    "effect_direction": t.effect_direction,
                    "effect_size": t.effect_size,
                    "n_observations": t.n_observations,
                }
                for t in vr.test_results
            ],
            "analysis_details": result.analysis_details,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "note": (
                "The live data fetch or pipeline failed. Report this honestly "
                "to the user rather than guessing a verdict."
            ),
        }


def _format_statistical_result(result: dict) -> str | None:
    """Create the public result deterministically after the analysis runs."""
    details = result.get("analysis_details")
    tests = result.get("raw_test_results") or []
    if not result.get("ok") or not details or not tests:
        return None

    change = details.get("observed_change")
    change_unit = details.get("change_unit")
    direction = "increased" if change is not None and change >= 0 else "decreased"
    if change_unit == "percentage points":
        change_text = f"{abs(change):.1f} percentage points"
    else:
        change_text = f"{abs(change):.1f}%"
    series = details.get("series_name", "the measured series")
    start = details.get("period_start")
    end = details.get("period_end")

    magnitude_matches = details.get("claimed_magnitude_matches_calculation")
    if magnitude_matches is False and details.get("directional_verdict") == "supported":
        claimed = details.get("claimed_percentage_change")
        outcome = (
            f"**The direction of the claim is supported, but its stated magnitude is not: "
            f"{series} {direction} by {change_text} between {start} and {end}, "
            f"rather than the claimed {claimed:.1f}%.**"
        )
    else:
        verdict_phrase = {
            "supported": "The evidence is consistent with the claim",
            "contradicted": "The evidence is inconsistent with the claim",
            "unproven": "The claim is not established by this analysis",
            "mixed": "The evidence is mixed",
        }.get(result.get("verdict"), "The analysis produced a result")
        outcome = (
            f"**{verdict_phrase}: {series} {direction} by {change_text} "
            f"between {start} and {end}.**"
        )

    test = tests[0]
    p_value = test.get("p_value")
    p_text = f"{p_value:.4g}" if isinstance(p_value, (int, float)) else "not available"
    ci = details.get("confidence_interval_95")
    ci_text = ""
    if ci:
        unit = details.get("unit_of_measure") or "units"
        if unit == "£million":
            unit = "£ million"
        ci_text = (
            f" The 95% confidence interval for the monthly trend was "
            f"{ci[0]:.3g} to {ci[1]:.3g} {unit} per month."
        )
    method = details.get("method")
    n = details.get("n_observations")
    effect = details.get("effect_direction")
    methods = (
        f"I calculated the change and fitted an {method} using {n} observations. "
        f"The estimated trend was {effect}. I used a two-sided hypothesis test of "
        f"whether the trend was zero at the 5% significance level (p = {p_text}).{ci_text}"
    )

    source = (
        f"**Data source:** {details.get('source_name')}  \n"
        f"[View the exact ONS data request]({details.get('source_url')})"
    )
    considerations = result.get("confounds_noted") or []
    caveats = "\n" + "\n".join(f"- {item}" for item in considerations) if considerations else (
        "Statistical significance does not by itself establish causation, and official data may be revised."
    )
    points = f"**Points to consider:** {caveats}"
    if details.get("period_was_user_supplied"):
        follow_up = "Would you like me to test a different period?"
    else:
        follow_up = (
            f"I used the latest five-year period available, {start} to {end}. "
            "Would you like me to test a different period or adjust the values for inflation?"
        )
    return "\n\n".join([outcome, methods, source, points, follow_up])


def chat_turn(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    messages: prior conversation as [{"role": "user"/"assistant", "content": "..."}].
    Returns (assistant_reply_text, updated_messages_including_tool_calls).

    Runs the tool-call loop: ask Groq, execute any tool calls locally
    against the real pipeline, feed results back, ask again for the
    final natural-language reply.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return (
            "Skew's chat isn't configured yet — the site is missing a "
            "GROQ_API_KEY environment variable. (This is a setup issue, not "
            "something you can fix from the chat.)",
            messages,
        )

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for _ in range(4):  # cap tool-call round-trips so a loop can't run forever
        resp = requests.post(
            GROQ_API_URL,
            headers=headers,
            json={
                "model": GROQ_MODEL,
                "messages": full_messages,
                "tools": [SKEW_TOOL_SCHEMA],
                "tool_choice": "auto",
            },
            timeout=30,
        )
        if not resp.ok:
            # requests' default HTTPError hides the response body, which is
            # exactly where the actual reason lives — surface it directly
            # instead of a generic "400 Bad Request" with no explanation.
            raise RuntimeError(
                f"Groq API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        choice = data["choices"][0]["message"]
        full_messages.append(choice)

        tool_calls = choice.get("tool_calls")
        if not tool_calls:
            # No tool call — final natural-language reply.
            return choice.get("content", ""), full_messages[1:]  # drop the system message before returning

        for call in tool_calls:
            args = json.loads(call["function"]["arguments"] or "{}")
            result = _run_tool(
                claim_text=args.get("claim_text", ""),
                expected_direction=args.get("expected_direction"),
            )
            full_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            })
            formatted = _format_statistical_result(result)
            if formatted and len(tool_calls) == 1:
                full_messages.append({"role": "assistant", "content": formatted})
                return formatted, full_messages[1:]

    return (
        "Something went wrong resolving this over several tool calls — try rephrasing your question.",
        full_messages[1:],
    )

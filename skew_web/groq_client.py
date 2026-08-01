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
            "claim describes a relationship or trend (e.g. 'X causes/increases "
            "Y', 'Y has risen/fallen') and the user hasn't already told you "
            "which direction they believe is true, ASK them directly before "
            "calling this tool — never guess the direction yourself from the "
            "claim's wording."
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
                        "if the claim doesn't need a direction (e.g. a quote-"
                        "attribution or opinion claim), or if you haven't yet "
                        "asked the user which direction they mean."
                    ),
                },
            },
            "required": ["claim_text"],
        },
    },
}

SYSTEM_PROMPT = """You are the conversational front-end for Skew, a UK statistical fact-checking tool.

Your ONLY source of truth for any verdict, statistic, or factual conclusion is the run_skew_verdict tool. You must never state whether a claim is true, false, supported, or contradicted from your own knowledge — always call the tool and report exactly what it returns, including its hedged language, confounds, and sources.

Rules:
1. If a claim describes a relationship ("X increases/decreases Y") or a trend ("Y has risen/fallen"), you MUST ask the user which direction they believe is true BEFORE calling the tool, unless they've already told you. Never infer the direction yourself from how the claim is phrased.
2. If the claim is vague or could mean several different specific things, ask a clarifying question first rather than guessing.
3. If the tool returns an error (e.g. a live data fetch failed), say so honestly — don't paper over it or make up a number instead.
4. If the tool reports requires_human_review=True, tell the user this verdict needs human sign-off before being treated as final.
5. You can have a normal, friendly conversation — greet the user, explain what Skew does, ask what they'd like to check. Just never skip the tool for the actual verdict.
6. Keep responses conversational and concise, not overly formal."""


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
        resp.raise_for_status()
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

    return (
        "Something went wrong resolving this over several tool calls — try rephrasing your question.",
        full_messages[1:],
    )

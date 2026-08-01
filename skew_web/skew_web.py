"""
Skew web interface — a Reflex app that puts a real page in front of the
engine already built in classification/, registry/, and pipeline/.

Flow:
  1. User types a claim, hits "Check this claim".
  2. We classify it first (cheap, no network) to see if it's the kind of
     claim that needs an explicit direction (see pipeline/types.py for
     why direction isn't auto-inferred).
  3. If it does, we ask the user which direction the claim asserts, then
     run the full pipeline (which will attempt a LIVE fetch against ONS/
     Home Office — this is the first time that's ever been tried outside
     the sandbox, so failures here are expected and shown honestly, not
     hidden).
  4. Render the verdict.

This is intentionally a thin UI — the goal right now is "does the real
engine work when deployed with real network access", not a polished
design pass.
"""

import traceback

import reflex as rx

from classification import ClaimClassifier, ClaimType
from skew_pipeline import SkewPipeline
from skew_web.groq_client import chat_turn

classifier = ClaimClassifier()
pipeline = SkewPipeline()

VERDICT_COLORS = {
    "supported": "green",
    "contradicted": "red",
    "mixed": "amber",
    "unproven": "gray",
    "insufficient_data": "gray",
    "not_a_factual_claim": "blue",
}


class State(rx.State):
    claim_text: str = ""
    is_loading: bool = False
    needs_direction: bool = False
    has_result: bool = False
    error_message: str = ""

    verdict: str = ""
    hedge_statement: str = ""
    confounds: list[str] = []
    data_sources: list[str] = []
    requires_review: bool = False
    review_reason: str = ""
    matched_rule: str = ""

    def set_claim_text(self, value: str):
        self.claim_text = value

    def _reset_result(self):
        self.has_result = False
        self.needs_direction = False
        self.error_message = ""
        self.verdict = ""
        self.hedge_statement = ""
        self.confounds = []
        self.data_sources = []
        self.requires_review = False
        self.review_reason = ""
        self.matched_rule = ""

    def check_claim(self):
        """Step 1: classify. Decide whether we need a direction from the user."""
        self._reset_result()
        if not self.claim_text.strip():
            self.error_message = "Type a claim first."
            return

        classification = classifier.classify(self.claim_text)
        if classification.claim_type in (ClaimType.ASSOCIATION, ClaimType.TREND):
            self.needs_direction = True
        else:
            yield from self._run_pipeline(expected_direction=None)

    def submit_with_direction(self, direction: str):
        """Step 2 (only for association claims): user picked a direction, now run it."""
        self.needs_direction = False
        yield from self._run_pipeline(expected_direction=direction)

    def _run_pipeline(self, expected_direction):
        self.is_loading = True
        yield
        try:
            result = pipeline.run(self.claim_text, expected_direction=expected_direction)
            vr = result.verdict_result
            self.verdict = vr.verdict.value
            self.hedge_statement = vr.hedge_statement
            self.confounds = vr.confounds_noted
            self.data_sources = vr.data_sources
            self.requires_review = vr.requires_human_review
            self.review_reason = vr.review_reason or ""
            self.matched_rule = result.classification_matched_rule or ""
            self.has_result = True
        except Exception as e:
            # Live fetches can fail (network, dataset ID, source down) —
            # show that honestly instead of crashing the page.
            self.error_message = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.is_loading = False


class ChatState(rx.State):
    messages: list[dict[str, str]] = []
    input_text: str = ""
    is_loading: bool = False

    def set_input_text(self, value: str):
        self.input_text = value

    def send(self):
        if not self.input_text.strip():
            return
        user_message = self.input_text
        self.messages.append({"role": "user", "content": user_message})
        self.input_text = ""
        self.is_loading = True
        yield
        try:
            # chat_turn does its own tool-call loop synchronously — fine
            # for now, though a long-running one will block this session's
            # other events until it returns.
            reply, updated = chat_turn(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            traceback.print_exc()
            self.messages.append({
                "role": "assistant",
                "content": f"Something went wrong talking to the chat model: {type(e).__name__}: {e}",
            })
        finally:
            self.is_loading = False


def chat_bubble(message: dict) -> rx.Component:
    return rx.box(
        rx.text(message["content"]),
        background=rx.cond(message["role"] == "user", "#E8F0FE", "#F3F3F3"),
        padding="0.75em 1em",
        border_radius="12px",
        max_width="80%",
        align_self=rx.cond(message["role"] == "user", "flex-end", "flex-start"),
    )


def chat_page() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.heading("Skew chat", size="7"),
                rx.link("Switch to form →", href="/", size="2", color="gray"),
                justify="between",
                width="100%",
            ),
            rx.text(
                "Ask about a UK statistic naturally — I'll ask follow-up "
                "questions if I need them, but every verdict comes from "
                "Skew's real statistical pipeline, not from me guessing.",
                color="gray",
                size="2",
            ),
            rx.vstack(
                rx.foreach(ChatState.messages, chat_bubble),
                width="100%",
                spacing="3",
                align_items="stretch",
                min_height="300px",
            ),
            rx.cond(
                ChatState.is_loading,
                rx.text("Thinking…", color="gray", size="2"),
            ),
            rx.hstack(
                rx.input(
                    placeholder="e.g. does immigration push up unemployment?",
                    value=ChatState.input_text,
                    on_change=ChatState.set_input_text,
                    on_key_down=lambda k: rx.cond(k == "Enter", ChatState.send(), rx.noop()),
                    width="100%",
                ),
                rx.button("Send", on_click=ChatState.send, loading=ChatState.is_loading),
                width="100%",
            ),
            spacing="4",
            width="100%",
            max_width="640px",
            padding="2em",
        ),
        width="100%",
        padding_top="2em",
    )


def result_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.badge(
                State.verdict,
                color_scheme=rx.cond(State.verdict == "supported", "green",
                    rx.cond(State.verdict == "contradicted", "red",
                        rx.cond(State.verdict == "mixed", "amber", "gray"))),
                size="2",
            ),
            rx.text(State.hedge_statement, size="3"),
            rx.cond(
                State.confounds.length() > 0,
                rx.vstack(
                    rx.text("Confounds to control for:", weight="bold", size="2"),
                    rx.foreach(State.confounds, lambda c: rx.text(f"• {c}", size="2")),
                    align_items="start",
                    spacing="1",
                ),
            ),
            rx.cond(
                State.data_sources.length() > 0,
                rx.text(
                    "Sources: " + State.data_sources.join(", "),
                    size="2",
                    color="gray",
                ),
            ),
            rx.cond(
                State.requires_review,
                rx.callout(
                    rx.cond(
                        State.review_reason != "",
                        State.review_reason,
                        "This verdict requires human sign-off before publishing.",
                    ),
                    icon="triangle_alert",
                    color_scheme="orange",
                ),
            ),
            align_items="start",
            spacing="3",
        ),
        width="100%",
    )


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.hstack(
                rx.heading("Skew", size="8"),
                rx.link("Try the chat →", href="/chat", size="2", color="gray"),
                justify="between",
                width="100%",
            ),
            rx.text(
                "Statistically-driven fact verification. Paste a claim below.",
                color="gray",
            ),
            rx.text_area(
                placeholder='e.g. "Immigrants are taking all the jobs in the UK"',
                value=State.claim_text,
                on_change=State.set_claim_text,
                width="100%",
                rows="3",
            ),
            rx.button(
                "Check this claim",
                on_click=State.check_claim,
                loading=State.is_loading,
                size="3",
            ),
            rx.cond(
                State.needs_direction,
                rx.vstack(
                    rx.text(
                        "Which direction does this claim assert?",
                        weight="bold",
                    ),
                    rx.hstack(
                        rx.button(
                            "Positive (rising / more of X → more of Y)",
                            on_click=lambda: State.submit_with_direction("positive"),
                        ),
                        rx.button(
                            "Negative (falling / more of X → less of Y)",
                            on_click=lambda: State.submit_with_direction("negative"),
                        ),
                    ),
                    padding="1em",
                    border="1px solid #ddd",
                    border_radius="8px",
                    width="100%",
                ),
            ),
            rx.cond(
                State.error_message != "",
                rx.callout(
                    State.error_message,
                    icon="circle_alert",
                    color_scheme="red",
                ),
            ),
            rx.cond(State.has_result, result_card()),
            spacing="4",
            width="100%",
            max_width="640px",
            padding="2em",
        ),
        width="100%",
        padding_top="3em",
    )


app = rx.App()
app.add_page(index, title="Skew — Fact Verification")
app.add_page(chat_page, route="/chat", title="Skew — Chat")

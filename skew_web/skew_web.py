"""
Skew web interface — a Reflex app that puts a real page in front of the
engine already built in classification/, registry/, and pipeline/.

This is the chat-only version: an LLM (via Groq's free API, see
skew_web/groq_client.py) handles the conversation, but it can only state
a verdict by calling the real SkewPipeline as a tool — see
groq_client.SYSTEM_PROMPT for the rule that enforces this. The earlier
form-based UI has been removed now that the chat interface works end to
end; the underlying classifier/registry/pipeline code is unchanged.
"""

import traceback

import reflex as rx

from skew_web.groq_client import chat_turn


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


def index() -> rx.Component:
    return rx.center(
        rx.vstack(
            rx.image(src="/skew_logo.png", height="3em"),
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
                rx.button(
                    "Send",
                    on_click=ChatState.send,
                    loading=ChatState.is_loading,
                    background="#1E9E5A",
                    color="white",
                    _hover={"background": "#177A46"},
                ),
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


app = rx.App()
app.add_page(index, title="Skew — Fact Verification")

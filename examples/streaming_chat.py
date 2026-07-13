"""Streaming: render text deltas live, then print the finished trace tree.

``.stream()`` returns an iterable of events from the same bus tracing uses
-- token deltas interleaved with span started/finished events -- so a live
UI and the trace are always in sync by construction.

Runs completely offline by default (FakeModel synthesizes word-level
deltas). Stream from a real model with:

    COMPOSE_EXAMPLE_MODEL="anthropic/claude-sonnet-5" python examples/streaming_chat.py
"""

import os

import composeai as compose
from composeai.testing import FakeModel


def pick_model():
    real = os.environ.get("COMPOSE_EXAMPLE_MODEL")
    if real:
        return real
    return FakeModel(
        [
            "Streams and traces, one and the same,\n"
            "every token a span event's frame."
        ]
    )


@compose.agent(model=pick_model())
def poet(subject: str) -> str:
    """You are a concise poet. Answer with exactly two short lines."""
    return f"Write two short lines about {subject}."


if __name__ == "__main__":
    stream = poet.stream("event buses")

    for event in stream:
        if event.kind == "text_delta" and event.text:
            print(event.text, end="", flush=True)

    print("\n")
    stream.run.trace.print()  # blocks until the run settles, then the full tree

"""Quickstart: a typed ``@agent`` with a ``@tool`` and structured output.

Runs completely offline by default (a scripted FakeModel -- no API key, no
network). Point it at a real model instead with:

    COMPOSE_EXAMPLE_MODEL="anthropic/claude-sonnet-5" python examples/quickstart_agent.py

(requires ``pip install "composeai[anthropic]"`` and ``ANTHROPIC_API_KEY``;
``"openai/gpt-5.2"``-style strings work too, with the openai extra).
"""

import os

from pydantic import BaseModel, Field

import composeai as compose
from composeai.testing import FakeModel


class FactSheet(BaseModel):
    topic: str
    key_facts: list[str] = Field(description="Three crisp, verifiable facts")
    word_count: int = Field(description="Word count of the key facts combined")


@compose.tool
def count_words(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text whose words should be counted.
    """
    return len(text.split())


def pick_model():
    """A real "provider/model-id" from the environment, or a scripted FakeModel."""
    real = os.environ.get("COMPOSE_EXAMPLE_MODEL")
    if real:
        return real
    facts = [
        "Qubits lose coherence in microseconds, so error correction is mandatory at scale.",
        "A single logical qubit can require ~1000 physical qubits.",
        "Cryogenic control electronics are a major scaling bottleneck.",
    ]
    return FakeModel(
        [
            {"tool_calls": [{"name": "count_words", "arguments": {"text": " ".join(facts)}}]},
            {"json": {"topic": "quantum computing", "key_facts": facts, "word_count": 33}},
        ]
    )


@compose.agent(model=pick_model(), tools=[count_words])
def researcher(topic: str) -> FactSheet:
    """You are a meticulous researcher. Extract crisp, verifiable facts.
    Use the count_words tool to report how long your fact list is."""
    return f"Build a fact sheet about: {topic}"


if __name__ == "__main__":
    run = researcher.run("quantum computing")

    print(f"topic:      {run.output.topic}")
    for fact in run.output.key_facts:
        print(f"  - {fact}")
    print(f"word count: {run.output.word_count}")

    print()
    run.trace.print()  # always-on tracing: no setup, the trace is just there

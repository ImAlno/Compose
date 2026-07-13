"""A durable ``@flow``: ``@task`` steps, ``compose.map``, an ``@agent`` step,
and a human approval gate -- cross-process pause/resume in one file.

First invocation starts a run and pauses at ``approve("publish")``:

    python examples/research_flow.py

It prints the pending interrupt and the exact command to resume. The second
invocation -- a brand-new process, seconds or days later -- resumes it:

    python examples/research_flow.py --resume RUN_ID --approve
    python examples/research_flow.py --resume RUN_ID --deny

Completed steps replay from the journal (the agent's LLM call is not
re-made and not re-billed); only the tail after the gate actually runs.

Runs completely offline by default (a scripted FakeModel -- no API key).
Point the agent step at a real model with:

    COMPOSE_EXAMPLE_MODEL="anthropic/claude-sonnet-5" python examples/research_flow.py
"""

import argparse
import os

import composeai as compose
from composeai.testing import FakeModel


def pick_model():
    real = os.environ.get("COMPOSE_EXAMPLE_MODEL")
    if real:
        return real
    return FakeModel(
        ["Draft report: sources agree the field is advancing quickly but scaling is hard."]
    )


@compose.task
def fetch_sources(topic: str) -> list[str]:
    # Side effects (HTTP calls, randomness, clock reads) belong in @task
    # bodies like this one -- the flow body itself must stay deterministic.
    return [f"{topic}: primary source", f"{topic}: industry report", f"{topic}: recent paper"]


@compose.task
def summarize(source: str) -> str:
    return f"summary of ({source})"


@compose.task
def publish(report: str) -> str:
    return f"PUBLISHED: {report}"


@compose.agent(model=pick_model())
def editor(summaries: list[str]) -> str:
    """You are a concise editor. Merge the summaries into a short report."""
    return "Write a short report based on these summaries:\n" + "\n".join(summaries)


@compose.flow
def research(topic: str) -> str:
    sources = fetch_sources(topic)  # journaled step 1
    summaries = compose.map(summarize, sources)  # journaled steps, fanned out in parallel
    draft = editor(summaries)  # the whole agent run journals as one step
    if compose.approve("publish", payload={"draft": draft}):  # named interrupt
        return publish(draft)
    return f"not published, draft was: {draft}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", metavar="RUN_ID", help="resume a paused run by id")
    parser.add_argument("--approve", action="store_true", help="answer the publish gate: yes")
    parser.add_argument("--deny", action="store_true", help="answer the publish gate: no")
    args = parser.parse_args()

    if args.resume:
        answered = args.approve or args.deny
        answers = {"publish": args.approve and not args.deny} if answered else None
        run = compose.resume(args.resume, answers=answers)  # re-pauses if unanswered
    else:
        run = research.run("quantum computing")

    if run.status == "paused":
        assert run.pending is not None
        print(f"run {run.id} is paused on interrupt {run.pending.id!r}")
        print(f"payload: {run.pending.payload}")
        print()
        print("this process can now exit -- resume from a fresh one with:")
        print(f"    python examples/research_flow.py --resume {run.id} --approve")
        print(f"    python examples/research_flow.py --resume {run.id} --deny")
        return 0

    print(f"output: {run.output}")
    print()
    run.trace.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

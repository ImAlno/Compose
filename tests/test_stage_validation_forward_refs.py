"""I-1 regression: an unresolvable string annotation degrades to unvalidated
pass-through instead of crashing a valid run (v0.5.0 final-review fix).

Unlike ``test_stage_validation.py``, this module DOES use
``from __future__ import annotations`` -- that is the exact trigger for the
bug. With it, a stage annotated by a *closure-local* model name comes back
from :func:`~composeai._schema.resolve_annotations` as an inert string
(``get_type_hints`` resolves against ``fn.__globals__`` only, never an
enclosing function's local scope), so ``_stage_input_type`` yields that raw
string.

0.4.1 (no runtime validation) passed such values straight through and ran
fine. 0.5.0's runtime validator must NOT build a lazy ``TypeAdapter("X")``:
that adapter is never rebuilt, so the FIRST run -- even with a perfectly
valid instance -- used to die with a raw
``PydanticUserError: 'TypeAdapter[X]' is not fully defined``. An unresolvable
annotation is statically useless; the boundary degrades to unvalidated
pass-through, exactly like ``Any``.
"""

from __future__ import annotations

from pydantic import BaseModel

from composeai._schema import _adapter_for
from composeai.combinators import aggregate, pipe


def test_adapter_for_string_annotation_is_none_passthrough():
    # The unit contract behind the fix: an unresolved string annotation is
    # unvalidatable, so `_adapter_for` returns None (Any-like pass-through) and
    # NEVER constructs a lazy `TypeAdapter("LocalTopic")`.
    assert _adapter_for("LocalTopic") is None


def test_closure_local_model_pipeline_builds_and_runs_clean_on_valid_input():
    def build_and_run():
        class LocalTopic(BaseModel):
            name: str

        def head(t: LocalTopic) -> LocalTopic:
            return t

        def tail(t: LocalTopic) -> LocalTopic:
            return t

        # Build-time eager warm must not choke on the string annotation either.
        p = pipe(head, tail)
        # Precondition of the bug: the annotation resolved to a raw string.
        assert p.input_type == "LocalTopic"
        original = LocalTopic(name="ai")
        # The FIRST run with a VALID instance must run clean (0.4.1 behavior),
        # not raise a raw PydanticUserError.
        return p(original), original

    out, original = build_and_run()
    # Degraded to pass-through: the exact object flows through untouched.
    assert out is original


def test_closure_local_model_aggregate_builds_and_runs_clean():
    def build_and_run():
        class LocalTopic(BaseModel):
            name: str

        def branch(t: LocalTopic) -> LocalTopic:
            return t

        # Aggregate's own eager warm must not choke on the string annotation.
        agg = aggregate(only=branch)
        original = LocalTopic(name="x")
        return agg(original), original

    out, original = build_and_run()
    assert out["only"] is original

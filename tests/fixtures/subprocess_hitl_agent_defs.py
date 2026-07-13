"""Shared ``@tool``/``@agent`` factory for the HITL *agent* cross-process test.

Unlike the flow fixtures, the ``@agent`` itself is built once **per driver
process** (via :func:`make_agent`) rather than shared as a single module-level
object -- each process needs its own :class:`~composeai.testing.FakeModel`,
scripted with only the responses *that* process will actually request:
process A only ever asks the model once (the tool-call turn, since
``dangerous_tool`` requires approval and is left unanswered); process B,
resuming from the durable ``agent_state`` snapshot, never re-asks that turn
(the whole point of the snapshot), so its ``FakeModel`` only needs the final
response. Both processes register the agent under the same name (``runner``)
so ``resume()`` can route to it via ``composeai.agentfn._AGENT_REGISTRY``.
"""

from __future__ import annotations

from composeai.agentfn import agent
from composeai.testing import FakeModel
from composeai.tools import tool


@tool(requires_approval=True)
def dangerous_tool() -> str:
    """Do something requiring approval."""
    return "done"


def make_agent(model: FakeModel):
    @agent(model=model, tools=[dangerous_tool], max_turns=5)
    def runner() -> str:
        """Runner."""
        return "go"

    return runner

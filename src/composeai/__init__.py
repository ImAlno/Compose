"""composeai: a radically simple, functional framework for multi-agent AI workflows.

Typed agent functions, pipe/aggregate/map composition with build-time type
checks, always-on local tracing, and durable, resumable flows.
"""

from .agentfn import agent, prompt
from .combinators import aggregate, map, pipe
from .flow import flow, resume, task
from .hitl import Interrupt, approve, ask_human
from .models.compatible import openai_compatible
from .runs import Budget, Run
from .tools import tool

__version__ = "0.1.1"

__all__ = [
    "Budget",
    "Interrupt",
    "Run",
    "agent",
    "aggregate",
    "approve",
    "ask_human",
    "flow",
    "map",
    "openai_compatible",
    "pipe",
    "prompt",
    "resume",
    "task",
    "tool",
]

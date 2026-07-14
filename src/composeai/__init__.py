"""composeai: a radically simple, functional framework for multi-agent AI workflows.

Typed agent functions, pipe/aggregate/map composition with build-time type
checks, always-on local tracing, and durable, resumable flows.
"""

from ._encoding import register_serializable
from ._schema import register_module_types
from .agentfn import agent, prompt
from .combinators import MapResult, aggregate, map, pipe
from .flow import flow, now, random, resume, task
from .hitl import Interrupt, approve, ask_human
from .models.compatible import openai_compatible
from .models.prices import ModelPrice, register_price
from .runs import Budget, Run
from .tools import tool

__version__ = "0.2.1"

__all__ = [
    "Budget",
    "Interrupt",
    "MapResult",
    "ModelPrice",
    "Run",
    "agent",
    "aggregate",
    "approve",
    "ask_human",
    "flow",
    "map",
    "now",
    "openai_compatible",
    "pipe",
    "prompt",
    "random",
    "register_module_types",
    "register_price",
    "register_serializable",
    "resume",
    "task",
    "tool",
]

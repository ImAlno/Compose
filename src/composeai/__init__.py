"""composeai: a radically simple, functional framework for multi-agent AI workflows.

Typed agent functions, pipe/aggregate/map composition with build-time type
checks, always-on local tracing, and durable, resumable flows.
"""

from ._encoding import register_serializable
from ._schema import register_module_types
from .agentfn import agent, prompt
from .combinators import MapResult, aggregate, amap, map, pipe
from .flow import anow, arandom, aresume, flow, now, random, resume, task
from .hitl import Interrupt, approve, ask_human
from .mcp import mcp_tools
from .models.compatible import openai_compatible
from .models.prices import ModelPrice, register_price
from .runs import Budget, Run
from .tools import tool

__version__ = "0.3.0"

__all__ = [
    "Budget",
    "Interrupt",
    "MapResult",
    "ModelPrice",
    "Run",
    "agent",
    "aggregate",
    "amap",
    "anow",
    "approve",
    "arandom",
    "aresume",
    "ask_human",
    "flow",
    "map",
    "mcp_tools",
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

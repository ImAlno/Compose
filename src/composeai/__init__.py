"""composeai: multi-agent workflows as the typed Python functions you already write.

Typed agent functions, pipe/aggregate/map composition with build-time type
checks, always-on local tracing, and durable, resumable flows.
"""

from ._encoding import register_serializable
from ._schema import register_module_types
from .agentfn import agent, prompt
from .chat import Chat, chat, load_chat
from .combinators import MapResult, aggregate, amap, map, pipe
from .flow import anow, arandom, aresume, flow, now, random, resume, task
from .hitl import Interrupt, approve, ask_human
from .mcp import mcp_tools
from .models.compatible import openai_compatible
from .models.prices import ModelPrice, register_price
from .runs import Budget, Run
from .tools import tool

__version__ = "0.8.0"

__all__ = [
    "Budget",
    "Chat",
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
    "chat",
    "flow",
    "load_chat",
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

# compose ── 𝑓(𝑔(𝑥))

A lightweight, functional framework for building multi-agent AI workflows. Stop fighting heavy graph configurations, deep abstraction wrappers, and rigid framework boilerplate. Treat your LLMs, prompts, and tools as pure, composable functions.

---

## 💡 The Philosophy

Most modern AI frameworks suffer from **The Abstraction Trap**. They force you to learn proprietary classes (`Runnable`, `AgentExecutor`, `StateGraph`), wrap raw strings in multiple layers of objects, and leave you with 15-level-deep stack traces when an API call fails.

`compose` takes a step back. It treats an AI Agent as a mathematical function:
1. **Inputs go in.**
2. **The LLM processes.**
3. **Structured data comes out.**

By treating agents like pure functions, you can chain, pipe, and compose them using native language primitives. No black boxes. No unnecessary boilerplate. Just clean, deterministic execution loops.

---

## 🚀 Quick Start

Here is how simple it is to build a multi-agent research pipeline using `compose` in Python.

```python
from compose import agent, pipe
from pydantic import BaseModel, Field

# Define clean data structures for your agent boundaries
class FactSheet(BaseModel):
    core_discoveries: list[str] = Field(description="Key historical facts discovered")
    technical_hurdles: list[str] = Field(description="Main engineering bottlenecks")

# 1. Define your agents as decorated functions
@agent(model="gpt-4o", temperature=0.2)
def researcher(topic: str) -> FactSheet:
    """Analyze the topic and extract technical facts."""
    # The docstring acts as part of the prompt context automatically!
    return topic

@agent(model="claude-3-5-sonnet", temperature=0.7)
def copywriter(facts: FactSheet) -> str:
    """Turn the structured fact sheet into an engaging blog post."""
    return f"Write a tech blog based on these points: {facts}"

# 2. Functional Composition: f(g(x))
# The output of researcher() automatically matches the input type of copywriter()
generate_blog = pipe(researcher, copywriter)

# 3. Execute
final_article = generate_blog("Quantum Computing Scalability")
print(final_article)
```

---

## 🛠 Why `compose` beats the Alternatives

| Feature | LangChain / LangGraph | `compose` |
| :--- | :--- | :--- |
| **Core Architecture** | Configuration & State Graphs | Pure Functional Programming |
| **Learning Curve** | High (Proprietary class ecosystem) | Zero (It's just regular functions & types) |
| **Debugging** | Painful (Deeply nested internal traces) | Trivial (Drop a standard breakpoint between functions) |
| **Dependencies** | Heavy footprint (Hundreds of transitive pins) | Minimalist (Lightweight wrapper around official SDKs) |
| **Type Safety** | Complex runtime dictionary checking | Native Compile-Time (Pydantic / TypeScript types) |

---

## 🧩 Advanced Patterns

### 1. Conditional Branching
Since agents are just functions, routing doesn't require a special "Conditional Edge" class. Use a standard Python match case or if-statement.

```python
@agent(model="gpt-4o-mini")
def route_intent(user_query: str) -> str:
    """Classify intent into: SUPPORT, BILLING, or TECHNICAL."""
    return user_query

# Route dynamically using vanilla Python
intent = route_intent("How do I update my credit card?")

match intent:
    case "BILLING": billing_agent(user_query)
    case "TECHNICAL": tech_agent(user_query)
    case _: general_support(user_query)
```

### 2. Parallel Composition (Fan-Out / Fan-In)
Run independent analysis tasks in parallel using native concurrency primitives, then aggregate their results.

```python
from compose import aggregate

@agent(model="gpt-4o")
def security_audit(code: str) -> str: ...

@agent(model="gpt-4o")
def performance_audit(code: str) -> str: ...

# Run both concurrently and gather outputs into a single dictionary
audit_pipeline = aggregate(sec=security_audit, perf=performance_audit)
results = audit_pipeline("def unstable_function()...")
```

---

## 🛣 Roadmap

- [ ] Native support for Anthropic Model Context Protocol (MCP) to inject tools seamlessly.
- [ ] Direct telemetry hooks for OpenInference/OpenTelemetry without monkey-patching.
- [ ] TypeScript parallel package (`@compose/core`) with true functional piping (`pipe(f, g, h)`).

---

## 🤝 Contributing

We love minimalists! If you want to contribute, please keep your PRs focused on expanding core developer experience without introducing bloated classes. Check out `CONTRIBUTING.md` to get started.

---

License: MIT
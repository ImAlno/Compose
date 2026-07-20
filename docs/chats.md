# Chats

`compose.chat` is the middle layer between one-shot agents and durable flows:
a persistent, resumable conversation over an ordinary `@agent` function.

```python
import composeai as compose

@compose.agent(model="anthropic/claude-sonnet-5")
def buddy() -> str:
    """You are a helpful pair programmer."""
    return "unused for chats"

c = compose.chat(buddy)
run = c.send("What does this traceback mean? ...")
print(run.output)
run2 = c.send("And how do I fix it?")   # the model sees the full history
```

Every `send` is a normal durable agent run: it appears in `compose runs`, is
traced and budgeted like any other call (`send(..., budget=Budget(usd=0.5))`),
and pauses on `requires_approval` tools exactly like `.run()` does.

## Continuing later — even in another process

```python
c = compose.load_chat(chat_id)     # the agent's module must be imported first
c.send("picking this back up ...")
```

History is persisted in the run store (`./.compose`) after every completed
send. `chat.messages` returns the full history; `chat.id` is the stable handle.

## Interactive control

```python
def ask(interrupt):                             # inline approval
    return input(f"allow {interrupt.payload['tool']}? [y/N] ") == "y"

def compact(messages, last_input_tokens):       # context management
    return summarize_older_turns(messages) if last_input_tokens > 100_000 else messages

c = compose.chat(buddy, approver=ask, context_manager=compact,
                 system="Optional per-chat system prompt override.")
```

- `approver=` is called synchronously for `requires_approval` tools instead of
  pausing; return `True`/`False`, or an `ApprovalReply(allow: bool, message:
  str | None = None)` — a denial with a `message` is shown to the agent as
  the denied tool's result instead of the default `"denied by user"`. This
  is **live-approver only**: a paused `resume({"tool:<name>:<call_id>":
  False})` denial is journaled as a plain `bool` and always falls back to
  `"denied by user"`. Without it, a gated send returns a paused
  `Run` — answer via `c.resume({"tool:<name>:<call_id>": True})`.
- `context_manager=` receives `(messages, last_input_tokens)` before every
  provider call; whatever it returns is what gets sent (and becomes the
  ongoing history — compaction persists).
- `c.resume(...)` re-supplies the chat's `approver`/`context_manager` to the
  resumed run, so inline approval and compaction keep working on every turn
  after a pause, not just the ones before it.
- `system=`/`model=` override the decoration defaults for this chat only.
  All four are also accepted directly by `.run()`/`.stream()` on any agent.

## Streaming

```python
for event in c.stream("walk me through the diff"):
    if event.kind == "text_delta":
        print(event.text, end="", flush=True)
```

`c.stream(...)` yields the same unified event bus as `.stream()`; the chat's
history updates when the stream's run completes.

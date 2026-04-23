# Phase 7: LLM Integration

## What We Built

The full agentic loop — Claude connected to the MCP server via the Anthropic API, with a domain-tuned system prompt, context window management, and an HTTP endpoint.

```
User question
     │
     ▼
POST /chat  (Starlette HTTP API)
     │
     ▼
FinancialAgent.chat()
     │
     ├── ContextManager.trim(history)
     │
     ├── ClaudeClient.complete(messages, system_prompt, tools)
     │        │
     │        ▼  stop_reason = "tool_use"
     │   MCPClient.call_tool(name, input)
     │        │
     │        └── retrieve_docs / query_metrics / list_sources
     │                │
     │                └── result fed back as tool_result message
     │
     │  (loop until stop_reason = "end_turn")
     │
     ▼
AgentResponse(text, tool_calls, updated_history)
     │
     ▼
{"response": "...", "tool_calls": [...], "history": [...]}
```

---

## Key Concepts

### The Agentic Loop

The term "agentic" means the LLM doesn't just answer in one shot — it takes actions (tool calls), observes results, and decides what to do next. The loop has exactly three cases:

**Case 1 — `stop_reason = "end_turn"`:** Claude has everything it needs. Extract the text block and return.

**Case 2 — `stop_reason = "tool_use"`:** Claude wants to call one or more tools. Execute them (in parallel using `asyncio.gather`), append both the tool-use request and the results to the message history, and loop.

**Case 3 — Any other stop reason (e.g. `max_tokens`):** The response was cut off. Return what we have with a truncation note.

```python
for _ in range(max_iterations):
    response = await claude.complete(messages, system, tools)

    if response.stop_reason == "end_turn":
        return AgentResponse(text=extract_text(response), ...)

    elif response.stop_reason == "tool_use":
        # Execute all requested tools in parallel
        results = await asyncio.gather(*[execute_tool(b) for b in tool_blocks])
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
        # → loop again
```

The `max_iterations` guard (default 10) prevents infinite loops — if Claude keeps calling tools without ever answering, the agent raises `RuntimeError`.

---

### Tool Use in the Anthropic API

Claude does not call tools directly. The protocol is a structured dialogue:

```
You → Claude: [messages] + [tool definitions]
Claude → You: ToolUseBlock(name="query_metrics", input={"metric_name": "revenue"}, id="use_abc")
You → Claude: ToolResultBlock(tool_use_id="use_abc", content="Revenue: 100,000")
Claude → You: TextBlock("Based on the data, revenue was 100,000.")
```

Each step is a full API round-trip. The `tool_use_id` links each result to the request that generated it — important when Claude requests multiple tools in one turn.

**Multi-tool turns:** Claude can request several tools at once. We execute all of them with `asyncio.gather` (concurrent, not sequential) and return all results in one `user` message. This cuts latency roughly proportional to the number of parallel tools.

---

### MCP → Anthropic Tool Format Conversion

The MCP protocol uses camelCase for JSON Schema:
```json
{"name": "retrieve_docs", "description": "...", "inputSchema": {...}}
```

The Anthropic API requires snake_case:
```json
{"name": "retrieve_docs", "description": "...", "input_schema": {...}}
```

`ClaudeClient.convert_mcp_tools()` handles this translation and also ensures `"type": "object"` is present at the schema root (required by Anthropic, sometimes omitted by MCP tool definitions).

This conversion happens once per `chat()` call — tools are fetched from the MCP server and immediately converted before being sent to Claude.

---

### System Prompt Engineering for Financial AI

The system prompt is the most important lever for LLM reliability in a production financial system. The MASSA prompt enforces three non-negotiable behaviours:

**1. Tool selection discipline**
```
Use query_metrics for any specific financial figure.
Use retrieve_docs for qualitative questions.
NEVER answer a quantitative question without calling query_metrics.
```
Without explicit rules, Claude will sometimes answer from training data ("I believe EBITDA margins in this sector are typically 20-25%") rather than retrieving the actual data.

**2. Mandatory citation**
```
Cite every factual claim: source file, page number, and period.
```
Citations make the output verifiable. An analyst can pull up `board_report.pdf, page 3` and check the claim. Uncited claims from an LLM are unverifiable.

**3. Explicit uncertainty permission**
```
IF data is unavailable: say "I don't have data for [...]"
Do not estimate or infer from related data points.
```
LLMs are trained to be helpful. Without explicit permission to say "I don't know", they will extrapolate, interpolate, and hallucinate rather than admit a gap.

`build_system_prompt(schema_description)` optionally embeds the live DB schema into the system prompt. This lets Claude write correct SQL/filters on the first try, without needing to call `list_sources` on every turn.

---

### Context Window Management — `ContextManager`

Claude's context window is ~200K tokens. A long session with many tool calls (each potentially returning thousands of tokens) can approach this limit.

**Strategy: trim from the front (oldest first)**

The most recent exchange is almost always the most relevant. Dropping old turns is almost always correct.

**Structural constraint: tool pairs must stay together**

A `tool_result` message is only valid if the matching `tool_use` message is present earlier in the history. The trimmer removes messages in pairs — if the oldest message is an `assistant` turn, it removes the following `user` (tool result) turn with it. This keeps the history structurally valid.

```python
# Trim in pairs: assistant (tool_use) + user (tool_result)
if trimmed[0].role == "assistant" and trimmed[1].role == "user":
    trimmed = trimmed[2:]  # remove both together
```

**Token counting:** Uses `tiktoken` with `cl100k_base` encoding — a close approximation to Claude's tokenizer. Conservative: errs toward over-counting (safer than under-counting and hitting the API limit).

---

### The HTTP API — `src/api/chat.py`

A thin Starlette endpoint that wraps the agent:

```
POST /chat
{"message": "What was EBITDA margin in Q3 2024?", "history": [...]}

→ 200 OK
{"response": "25.00%", "tool_calls": [...], "history": [...]}
```

**Multi-turn conversations:** The client sends its `history` array with every request and receives an `updated_history` back. The server is stateless — history management is the client's responsibility. This design scales horizontally (any server instance can handle any request) and avoids server-side session state.

**Factory pattern:** `create_app(agent)` takes the agent as a parameter, making the app fully testable with mock agents. No global state.

---

## File Structure

```
src/
├── llm/
│   ├── __init__.py
│   ├── system_prompt.py    — build_system_prompt(schema_description)
│   ├── context_manager.py  — ContextManager: count_tokens(), trim()
│   ├── client.py           — ClaudeClient: complete(), convert_mcp_tools()
│   └── agent.py            — FinancialAgent: chat(), agentic loop
│                             AgentResponse, ToolCall dataclasses
└── api/
    ├── __init__.py
    └── chat.py             — create_app(agent), POST /chat handler
```

---

## How It Connects to Previous Phases

| Phase | Connection to Phase 7 |
|---|---|
| Phase 5 (SchemaIntrospector) | `build_system_prompt(schema_description)` embeds the live schema |
| Phase 6 (MCP server) | `FinancialAgent` takes a FastMCP server and calls tools via `MCPClient` |
| Phase 4 (retrieval) + Phase 5 (metrics) | Tool calls executed by Phase 6's tools, which call Phase 4/5 engines |

Phase 7 is the top of the stack. It adds nothing to the data layer — it only orchestrates what was already built.

---

## Tests — `tests/test_llm.py`

24 tests across 5 categories:

| Category | Tests | What's verified |
|---|---|---|
| System prompt | 5 | Role definition; both tools mentioned; citations required; fabrication forbidden; schema embedded when provided |
| ContextManager | 5 | Token counting (string + list content); no trim under limit; oldest trimmed first; never below 2 messages |
| Tool conversion | 4 | inputSchema → input_schema rename; name/description preserved; type:object added if missing; multiple tools converted |
| Agent loop | 6 | Direct response (no tools); query_metrics called; retrieve_docs called; multi-step sequence; history structure; max_iterations guard |
| HTTP API | 4 | 200 with response/tool_calls/history; history forwarded to agent; 400 on missing message; 400 on invalid JSON |

**Mock strategy:** Claude responses are mocked as `MagicMock` objects with `.stop_reason`, `.content`, `.type`, `.text`, `.name`, `.input`, `.id` attributes — matching the Anthropic SDK's object structure without requiring a live API key. The FastMCP server in agent tests is a real (minimal) in-memory server with stub tools, so `list_tools()` and `call_tool()` work correctly.

---

## Running Phase 7 Tests

```bash
# Phase 7 only
python -m uv run pytest tests/test_llm.py -v

# Full suite (all phases)
python -m uv run pytest -v
```

Expected result: **122 passed, 1 skipped**.

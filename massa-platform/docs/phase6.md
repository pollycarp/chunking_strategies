# Phase 6: MCP Server

## What We Built

An MCP (Model Context Protocol) server that exposes both data surfaces — document retrieval and structured metrics — as typed tools an LLM can call by name.

```
LLM (Claude)
    │
    │  MCP Protocol (JSON over stdio or HTTP)
    │
    ▼
FastMCP Server  "MASSA Financial Intelligence"
    │
    ├── retrieve_docs(query, top_k, doc_type, source_file)
    │       └── HybridRetriever → RRF → CohereReranker → formatted chunks
    │
    ├── query_metrics(metric_name, company, period_label)
    │       └── metric_query() → QueryEngine → financial_metrics view
    │
    └── list_sources()
            └── SchemaIntrospector + documents table → full inventory
```

---

## Key Concepts

### MCP — Model Context Protocol

MCP is an open standard (published by Anthropic) for connecting LLMs to external tools and data. Before MCP, every AI application invented its own tool-calling format. MCP standardises this:

- The **server** (our FastMCP app) declares what tools are available and their parameter schemas
- The **client** (Claude or any MCP-compatible LLM) queries the tool list and calls tools by name
- The protocol handles serialisation, errors, and the request/response lifecycle

Think of it as USB-C for AI tools: a single plug that works across all compatible devices. Once your tools are MCP-compliant, any MCP host (Claude Desktop, Claude API with tool use, other MCP clients) can use them without code changes.

---

### FastMCP

FastMCP is the Python framework that makes building MCP servers easy. The entire tool registration pattern is:

```python
mcp = FastMCP("My Server")

@mcp.tool()
async def my_tool(query: str, top_k: int = 5) -> str:
    """Tool description the LLM will read."""
    ...
    return result_text
```

FastMCP automatically:
- Converts the Python type annotations (`str`, `int`, `str | None`) into JSON Schema for the MCP tool spec
- Parses incoming LLM requests into typed Python arguments
- Returns your string as a `TextContent` block in the MCP response

---

### Tool Descriptions as Prompt Engineering

The docstring you write for an MCP tool is the exact text the LLM reads when deciding whether to call it. This is why tool descriptions are a form of **prompt engineering**.

A vague description gets misused:
```python
# Bad — LLM will use this for everything
async def search(query: str) -> str:
    """Search for information."""
```

A precise description guides correct usage:
```python
# Good — LLM knows exactly when to use this vs query_metrics
async def retrieve_docs(query: str, ...) -> str:
    """
    Search the document library for text relevant to a question.
    Use this for QUALITATIVE questions: board discussions, risk factors,
    strategic decisions, management commentary.
    Do NOT use for specific numbers — use query_metrics instead.
    """
```

The three tools are designed with distinct, non-overlapping scopes:

| Tool | When to use | When NOT to use |
|---|---|---|
| `retrieve_docs` | Qualitative questions about documents | Specific numbers/metrics |
| `query_metrics` | Specific financial figures | Document context/narrative |
| `list_sources` | "What data do you have?" / orientation | Answering a substantive question |

---

### Factory Pattern — `create_server()`

Rather than a module-level singleton, the server is created by a factory function:

```python
def create_server(pool, embedder, reranker=None) -> FastMCP:
    mcp = FastMCP("MASSA Financial Intelligence", instructions="...")
    retriever = HybridRetriever(pool, embedder)
    retrieve_docs.register(mcp, retriever, reranker)
    query_metrics.register(mcp, pool)
    list_sources.register(mcp, pool)
    return mcp
```

**Why a factory?**
- **Testability** — pass a mock embedder and the test pool; no real API key needed
- **Flexibility** — swap `CohereReranker` for `PassThroughReranker` for local dev
- **No global state** — multiple server instances can coexist (useful in tests)

Each tool file's `register()` function uses a **closure** to capture its dependencies:

```python
def register(mcp, retriever, reranker):
    @mcp.tool()
    async def retrieve_docs(query: str, ...) -> str:
        # retriever and reranker captured from the enclosing scope
        candidates = await retriever.search(query, ...)
        final = await reranker.rerank(query, candidates, ...)
        return format_chunks(final)
```

The LLM never sees the implementation — only the tool name, description, and parameter schema.

---

### Output Formatters — `src/mcp/output_formatters.py`

How you present retrieved information affects LLM answer quality. Three formatters:

**`format_chunks(chunks)`** — numbered list with source citations:
```
[1] report.pdf — page 3, section: Financial Highlights  (score: 0.87)
EBITDA margin improved to 23.4% from 21.1% in the prior year period.

[2] memo.docx — page 1  (score: 0.72)
Revenue grew 12% year-over-year.
```
The rank, score, source file, and page number are all present so the LLM can:
1. Cite the source accurately ("according to report.pdf, page 3...")
2. Know which sources are most relevant (score)
3. Omit weak matches if needed (score threshold)

**`format_metric_rows(rows, metric_name, unit)`** — labelled table:
```
Metric: ebitda_margin_pct [percent]

Company        | Period   | Value
---------------|----------|-------
Test Agri Co   | Q3 2024  | 25.00
Test Agri Co   | Q4 2024  | 26.67
```

**`format_sources(tables, documents, schema_description)`** — full inventory for LLM orientation.

---

### Server Instructions

FastMCP supports an `instructions` string on the server itself — this is included in the MCP server manifest and acts as a top-level system prompt that applies across all tools:

```python
mcp = FastMCP(
    name="MASSA Financial Intelligence",
    instructions=(
        "You are a financial analyst assistant with access to two data sources...\n"
        "Always cite your sources. When unsure what data is available, call list_sources first.\n"
        "For quantitative questions prefer query_metrics. For qualitative use retrieve_docs."
    ),
)
```

This means even if the LLM forgets to read individual tool descriptions, the server-level instruction guides it to the right tool selection behaviour.

---

## File Structure

```
src/mcp/
├── __init__.py
├── server.py               — create_server() factory, wires all tools
├── output_formatters.py    — format_chunks, format_metric_rows, format_sources
└── tools/
    ├── __init__.py
    ├── retrieve_docs.py    — register() attaches retrieve_docs tool
    ├── query_metrics.py    — register() attaches query_metrics tool
    └── list_sources.py     — register() attaches list_sources tool
```

---

## How It Connects to Other Phases

| Phase | Connection to Phase 6 |
|---|---|
| Phase 4 (HybridRetriever, reranker) | `retrieve_docs` tool wraps retriever + reranker |
| Phase 5 (metrics, QueryEngine, SchemaIntrospector) | `query_metrics` and `list_sources` wrap these directly |
| Phase 7 (LLM integration, next) | Phase 7's Claude agent will use `create_server()` to create the MCP server and connect Claude to it |

Phase 6 is the adapter layer. It doesn't contain any business logic — it translates MCP protocol calls into calls on the engines built in Phases 4 and 5.

---

## Tests — `tests/test_mcp.py`

19 tests across 5 categories:

| Category | Tests | What's verified |
|---|---|---|
| Output formatters | 8 | format_chunks single/multiple/empty/no-section; format_metric_rows values/empty; format_sources all-sections/no-docs |
| Tool registration | 2 | All 3 tools discoverable via FastMCP Client; all have non-empty descriptions (>20 chars) |
| `retrieve_docs` | 2 | Mock retriever returns formatted output; empty results return no-results message |
| `query_metrics` | 4 | Correct revenue value; EBITDA margin computed correctly; unknown metric → error string; no data → helpful message |
| `list_sources` | 3 | Schema tables present; metric names present; query rules present |

**Key testing pattern — `_extract_text()`:**
FastMCP's `Client.call_tool` wraps the tool's return value in MCP content blocks. The helper handles both direct string returns and list-of-content-block returns, making tests robust across FastMCP versions.

**Key testing pattern — tool registration tests:**
Tool registration is silent on failure — if `@mcp.tool()` fails, the LLM simply won't see the tool. Testing registration explicitly via `client.list_tools()` catches these silent failures.

---

## Running Phase 6 Tests

```bash
# Phase 6 only
python -m uv run pytest tests/test_mcp.py -v

# Full suite (all phases)
python -m uv run pytest -v
```

Expected result: **98 passed, 1 skipped**.

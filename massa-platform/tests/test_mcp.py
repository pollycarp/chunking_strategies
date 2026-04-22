"""
Phase 6 Tests: MCP Server

Test categories:
- formatter tests    : format_chunks, format_metric_rows, format_sources output shape
- tool registration  : all three tools are discoverable via FastMCP client
- retrieve_docs      : returns formatted chunks for a known query (mock retriever)
- query_metrics      : returns correct metric values from fixture data
- list_sources       : returns schema + document inventory
- safety             : query_metrics blocks unknown metrics cleanly
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client

from src.mcp.output_formatters import format_chunks, format_metric_rows, format_sources
from src.mcp.server import create_server
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import PassThroughReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: int = 1,
    content: str = "EBITDA margin improved to 23.4%.",
    source_file: str = "report.pdf",
    page_number: int = 3,
    section_title: str = "Financial Highlights",
    score: float = 0.87,
    rank: int = 1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        page_number=page_number,
        section_title=section_title,
        doc_type="pdf",
        chunk_strategy="fixed",
        score=score,
        rank=rank,
        parent_id=None,
    )


def _extract_text(result) -> str:
    """
    Extract plain text from a FastMCP call_tool result.
    Handles both list-of-content and direct string return styles
    across FastMCP versions.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        item = result[0]
        return item.text if hasattr(item, "text") else str(item)
    return str(result)


@pytest.fixture
async def financial_fixture(db_pool):
    """Inserts fixture data for query_metrics tests. Cleans up after."""
    async with db_pool.acquire() as conn:
        company_id = await conn.fetchval(
            "INSERT INTO dim_company (name, sector) VALUES ('MCP Test Co', 'Finance') RETURNING id"
        )
        period_id = await conn.fetchval(
            """
            INSERT INTO dim_period (year, quarter, period_label, start_date, end_date)
            VALUES (2024, 2, 'Q2 2024', '2024-04-01', '2024-06-30') RETURNING id
            """
        )
        await conn.execute(
            """
            INSERT INTO fact_financials
                (company_id, period_id, revenue, gross_profit, ebitda, ebit,
                 net_income, total_assets, total_debt, cash)
            VALUES ($1, $2, 80000, 32000, 20000, 16000, 10000, 150000, 40000, 12000)
            """,
            company_id, period_id,
        )

    yield {"company": "MCP Test Co", "company_id": company_id, "period_id": period_id}

    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM dim_company WHERE id = $1", company_id)
        await conn.execute("DELETE FROM dim_period WHERE id = $1", period_id)


# ---------------------------------------------------------------------------
# Test 1: Output formatters (pure — no DB)
# ---------------------------------------------------------------------------

def test_format_chunks_single_result():
    """format_chunks produces a numbered entry with source, score, and content."""
    chunk = _make_chunk()
    output = format_chunks([chunk])

    assert "[1]" in output
    assert "report.pdf" in output
    assert "page 3" in output
    assert "Financial Highlights" in output
    assert "0.87" in output
    assert "EBITDA margin improved" in output


def test_format_chunks_multiple_results():
    """Each chunk gets its own numbered entry."""
    chunks = [
        _make_chunk(chunk_id=1, rank=1, content="First chunk."),
        _make_chunk(chunk_id=2, rank=2, content="Second chunk.", score=0.60),
    ]
    output = format_chunks(chunks)

    assert "[1]" in output
    assert "[2]" in output
    assert "First chunk." in output
    assert "Second chunk." in output


def test_format_chunks_empty_returns_helpful_message():
    """Empty chunk list → informative 'no results' message, not an error."""
    output = format_chunks([])
    assert "No relevant documents" in output


def test_format_chunks_no_section_title():
    """Chunks without section_title should still format cleanly."""
    chunk = _make_chunk(section_title=None)
    output = format_chunks([chunk])
    assert "report.pdf" in output
    assert "section" not in output  # no section line if title is None


def test_format_metric_rows_correct_values():
    """format_metric_rows produces a labelled table with all values."""
    rows = [
        {"company": "Agri Co", "period_label": "Q3 2024", "value": 25.00},
        {"company": "Agri Co", "period_label": "Q4 2024", "value": 26.67},
    ]
    output = format_metric_rows(rows, "ebitda_margin_pct", unit="percent")

    assert "ebitda_margin_pct" in output
    assert "percent" in output
    assert "Agri Co" in output
    assert "Q3 2024" in output
    assert "25.00" in output
    assert "26.67" in output


def test_format_metric_rows_empty_returns_helpful_message():
    """Empty rows → informative 'no data' message."""
    output = format_metric_rows([], "revenue")
    assert "No data found" in output
    assert "revenue" in output


def test_format_sources_contains_all_sections():
    """format_sources output must contain documents, tables, and schema."""
    docs = [{"filename": "report.pdf", "doc_type": "pdf", "chunk_count": 45}]
    tables = ["dim_company", "financial_metrics"]
    schema = "TABLE dim_company\n  id (integer)"

    output = format_sources(tables, docs, schema)

    assert "report.pdf" in output
    assert "dim_company" in output
    assert "financial_metrics" in output
    assert "TABLE dim_company" in output


def test_format_sources_no_documents():
    """When no documents are ingested, output says so clearly."""
    output = format_sources(["dim_company"], [], "schema text")
    assert "No documents ingested" in output


# ---------------------------------------------------------------------------
# Test 2: Tool registration
# ---------------------------------------------------------------------------

async def test_all_tools_are_registered(db_pool):
    """
    All three tools must be discoverable via the MCP list_tools endpoint.

    Why test this separately?
    Tool registration happens at server construction time. If a tool fails to
    register (e.g. due to a syntax error in its decorator), the LLM simply
    won't see it — no exception is raised. This test catches silent failures.
    """
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        tools = await client.list_tools()

    tool_names = {t.name for t in tools}
    assert "retrieve_docs"  in tool_names, "retrieve_docs tool must be registered"
    assert "query_metrics"  in tool_names, "query_metrics tool must be registered"
    assert "list_sources"   in tool_names, "list_sources tool must be registered"


async def test_tool_descriptions_are_not_empty(db_pool):
    """
    Every tool must have a non-empty description.
    Blank descriptions prevent the LLM from knowing when to use the tool.
    """
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        tools = await client.list_tools()

    for tool in tools:
        assert tool.description, f"Tool '{tool.name}' has no description"
        assert len(tool.description) > 20, (
            f"Tool '{tool.name}' description is too short to be useful"
        )


# ---------------------------------------------------------------------------
# Test 3: retrieve_docs tool
# ---------------------------------------------------------------------------

async def test_retrieve_docs_returns_formatted_output(db_pool):
    """
    retrieve_docs with a mock retriever+reranker returns formatted chunk text.
    The mock bypasses the embedding API and DB retrieval.
    """
    known_chunk = _make_chunk(
        content="Revenue grew 18% driven by strong agribusiness performance.",
        source_file="board_report.pdf",
        rank=1,
    )

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[known_chunk])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[known_chunk])

    # Patch the HybridRetriever constructor so the server uses our mock
    from src.mcp.tools import retrieve_docs as rd_module
    from fastmcp import FastMCP
    mcp = FastMCP("test")
    rd_module.register(mcp, mock_retriever, mock_reranker)

    async with Client(mcp) as client:
        result = await client.call_tool("retrieve_docs", {"query": "revenue growth"})

    text = _extract_text(result)
    assert "board_report.pdf" in text
    assert "Revenue grew 18%" in text
    assert "[1]" in text


async def test_retrieve_docs_empty_results(db_pool):
    """retrieve_docs with no matching chunks returns the no-results message."""
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    from src.mcp.tools import retrieve_docs as rd_module
    from fastmcp import FastMCP
    mcp = FastMCP("test")
    rd_module.register(mcp, mock_retriever, mock_reranker)

    async with Client(mcp) as client:
        result = await client.call_tool("retrieve_docs", {"query": "nonexistent topic"})

    text = _extract_text(result)
    assert "No relevant documents" in text


# ---------------------------------------------------------------------------
# Test 4: query_metrics tool
# ---------------------------------------------------------------------------

async def test_query_metrics_returns_correct_value(db_pool, financial_fixture):
    """
    query_metrics for revenue of 'MCP Test Co' returns the inserted value (80,000).
    Uses real DB data — tests the full stack from tool call to DB query.
    """
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool(
            "query_metrics",
            {
                "metric_name": "revenue",
                "company": financial_fixture["company"],
                "period_label": "Q2 2024",
            },
        )

    text = _extract_text(result)
    assert "80000.00" in text or "80,000" in text.replace(",", "") or "80000" in text
    assert "MCP Test Co" in text
    assert "Q2 2024" in text


async def test_query_metrics_computes_ebitda_margin(db_pool, financial_fixture):
    """
    EBITDA margin = ebitda/revenue*100 = 20000/80000*100 = 25.00%.
    Tests the view's computed column is surfaced correctly through the tool.
    """
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool(
            "query_metrics",
            {
                "metric_name": "ebitda_margin_pct",
                "company": financial_fixture["company"],
                "period_label": "Q2 2024",
            },
        )

    text = _extract_text(result)
    assert "25.00" in text


async def test_query_metrics_unknown_metric_returns_error_message(db_pool):
    """
    An unknown metric name should return an error string, not raise an exception.
    The LLM needs to see the error message to recover gracefully.
    """
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool(
            "query_metrics",
            {"metric_name": "made_up_metric"},
        )

    text = _extract_text(result)
    assert "Unknown metric" in text or "made_up_metric" in text


async def test_query_metrics_no_data_returns_helpful_message(db_pool):
    """
    A valid metric with no matching data returns a 'no data found' message.
    """
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool(
            "query_metrics",
            {
                "metric_name": "revenue",
                "company": "Nonexistent Company XYZ",
            },
        )

    text = _extract_text(result)
    assert "No data found" in text


# ---------------------------------------------------------------------------
# Test 5: list_sources tool
# ---------------------------------------------------------------------------

async def test_list_sources_contains_schema_tables(db_pool):
    """list_sources output must mention the financial schema tables."""
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool("list_sources", {})

    text = _extract_text(result)
    assert "dim_company"     in text
    assert "fact_financials" in text
    assert "financial_metrics" in text


async def test_list_sources_contains_metric_names(db_pool):
    """list_sources output must include named metric definitions."""
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool("list_sources", {})

    text = _extract_text(result)
    assert "ebitda_margin_pct" in text
    assert "revenue" in text


async def test_list_sources_mentions_query_rules(db_pool):
    """list_sources must include the read-only query rules."""
    mock_embedder = MagicMock()
    server = create_server(db_pool, mock_embedder)

    async with Client(server) as client:
        result = await client.call_tool("list_sources", {})

    text = _extract_text(result)
    assert "SELECT only" in text

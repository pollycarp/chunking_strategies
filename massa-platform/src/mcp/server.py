"""
MCP Server — assembles the FastMCP application.

Architecture: factory function
Rather than a module-level singleton, create_server() takes its dependencies
as arguments. This keeps the server testable (pass mock pool/embedder in tests)
and flexible (swap embedder or reranker without touching server code).

Why thin?
The MCP layer is intentionally thin — it translates LLM tool calls into
calls on the retrieval/SQL engines. All business logic stays in the engines,
not in the tool handlers. This keeps each layer independently testable.

Usage:
    pool = await create_pool()
    embedder = OpenAIEmbedder(api_key=settings.openai_api_key)
    server = create_server(pool, embedder)
    server.run()  # starts MCP server over stdio (for Claude Desktop) or HTTP
"""

from __future__ import annotations

from asyncpg import Pool
from fastmcp import FastMCP

from src.embeddings.base import EmbeddingModel
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import BaseReranker, PassThroughReranker
from src.mcp.tools import retrieve_docs, query_metrics, list_sources


def create_server(
    pool: Pool,
    embedder: EmbeddingModel,
    reranker: BaseReranker | None = None,
    candidate_k: int = 20,
) -> FastMCP:
    """
    Creates and configures the MASSA MCP server.

    Parameters:
        pool        : asyncpg connection pool (already initialised with pgvector codec)
        embedder    : embedding model for semantic search (OpenAI, Voyage, or mock)
        reranker    : cross-encoder reranker; defaults to PassThroughReranker (no-op)
        candidate_k : how many candidates each sub-retriever fetches before RRF fusion

    Returns a FastMCP instance with three tools registered:
        retrieve_docs   — hybrid document search
        query_metrics   — named financial metric queries
        list_sources    — schema + document inventory
    """
    if reranker is None:
        reranker = PassThroughReranker()

    mcp = FastMCP(
        name="MASSA Financial Intelligence",
        instructions=(
            "You are a financial analyst assistant with access to two data sources:\n"
            "1. A document library (PDFs, Word docs, spreadsheets) — use retrieve_docs\n"
            "2. A structured financial metrics database — use query_metrics\n\n"
            "Always cite your sources. When unsure what data is available, call list_sources first.\n"
            "For quantitative questions (revenue, margins, ratios) prefer query_metrics.\n"
            "For qualitative questions (context, narrative, risk factors) use retrieve_docs."
        ),
    )

    retriever = HybridRetriever(pool=pool, embedder=embedder, candidate_k=candidate_k)

    retrieve_docs.register(mcp, retriever, reranker, candidate_k=candidate_k)
    query_metrics.register(mcp, pool)
    list_sources.register(mcp, pool)

    return mcp

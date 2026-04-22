"""
MCP Tool: retrieve_docs

Exposes hybrid document retrieval (Phase 4) as an MCP tool.

Tool description design principle:
The docstring is what the LLM reads when deciding whether to call this tool.
It must answer three questions clearly:
  1. WHEN should I call this? (qualitative questions, document content)
  2. WHEN should I NOT call this? (quantitative questions → use query_metrics)
  3. WHAT do I get back? (ranked text chunks with source citations)
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.mcp.output_formatters import format_chunks
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalFilter
from src.retrieval.reranker import BaseReranker


def register(
    mcp: FastMCP,
    retriever: HybridRetriever,
    reranker: BaseReranker,
    candidate_k: int = 20,
) -> None:
    """
    Registers the retrieve_docs tool on the given FastMCP server instance.

    Uses closures to capture retriever and reranker — no global state needed.
    """

    @mcp.tool()
    async def retrieve_docs(
        query: str,
        top_k: int = 5,
        doc_type: str | None = None,
        source_file: str | None = None,
    ) -> str:
        """
        Search the document library for text relevant to a question.

        Use this tool for QUALITATIVE questions about document content:
        - Board meeting discussions and decisions
        - Risk factors and strategic context
        - Covenant compliance narratives
        - Management commentary and outlook
        - Any question whose answer lives in a PDF, Word doc, or spreadsheet

        Do NOT use this tool for specific quantitative metrics like revenue,
        EBITDA margin, or debt ratios — use query_metrics instead, which
        returns exact numbers from the structured database.

        Parameters:
            query       : The natural-language question to search for
            top_k       : Number of results to return (default 5, max 20)
            doc_type    : Optional — filter to "pdf", "docx", or "xlsx" only
            source_file : Optional — filter to a specific file by name

        Returns ranked text chunks, each with source file, page number, and
        section title for citation purposes.
        """
        top_k = min(top_k, 20)

        filters = None
        if doc_type is not None or source_file is not None:
            filters = RetrievalFilter(
                doc_type=doc_type,
                source_file=source_file,
            )

        # Step 1: hybrid retrieval — gets top candidate_k from both sub-retrievers
        candidates = await retriever.search(query, top_k=candidate_k, filters=filters)

        # Step 2: re-rank candidates down to top_k
        final = await reranker.rerank(query, candidates, top_k=top_k)

        return format_chunks(final)

"""
MCP Tool: list_sources

Returns a complete description of all available data — ingested documents,
database tables, and named metrics — so the LLM can orient itself before
deciding which tool to use for an unfamiliar query.

Tool description design principle:
This is the LLM's map of the territory. It should be called at the start
of a session or when the LLM is unsure what data exists. It answers the
meta-question "what can I look up?" before the object-level question "what
is the answer?".
"""

from __future__ import annotations

from asyncpg import Pool
from fastmcp import FastMCP

from src.mcp.output_formatters import format_sources
from src.structured.schema_introspector import SchemaIntrospector


def register(mcp: FastMCP, pool: Pool) -> None:
    """
    Registers the list_sources tool on the given FastMCP server instance.
    """
    introspector = SchemaIntrospector(pool)

    @mcp.tool()
    async def list_sources() -> str:
        """
        Returns a description of all available data sources.

        Call this tool when:
        - The user asks "what data do you have?" or "what can you look up?"
        - You are unsure whether to use retrieve_docs or query_metrics
        - You need to know the exact company names or period labels to use
          as filters in query_metrics

        Returns:
        - A list of all ingested documents with their type and chunk count
        - The database schema: tables, columns, and data types
        - All named financial metrics with descriptions
        - Query rules (what operations are permitted)
        """
        # Fetch ingested documents
        async with pool.acquire() as conn:
            doc_rows = await conn.fetch(
                """
                SELECT filename, doc_type, chunk_count
                FROM   documents
                ORDER  BY filename
                """
            )

        documents = [dict(row) for row in doc_rows]
        tables = await introspector.list_tables()
        schema_description = await introspector.get_schema_description()

        return format_sources(tables, documents, schema_description)

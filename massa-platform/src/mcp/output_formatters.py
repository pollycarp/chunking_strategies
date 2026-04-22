"""
Output formatters — shapes raw Python data into LLM-readable text.

Why this matters:
How you present retrieved information directly affects LLM answer quality.
Well-formatted output with clear source citations:
  - Reduces hallucination (LLM can quote directly)
  - Enables citation ("according to report.pdf, page 3...")
  - Makes the LLM's job easier (less parsing work = fewer errors)

All formatters return plain text, not JSON or HTML. Plain text works across
all LLM contexts (chat, tool response, context window).
"""

from __future__ import annotations

from src.retrieval.models import RetrievedChunk


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    """
    Formats retrieved document chunks as a numbered list with source metadata.

    Each result includes:
    - Rank and relevance score (so the LLM knows which sources are strongest)
    - Source file, page, and section (for citation)
    - The actual text content

    Example output:
        [1] report.pdf — page 3, section: Financial Highlights  (score: 0.87)
        EBITDA margin improved to 23.4% from 21.1% in the prior year period.

        [2] memo.docx — page 1  (score: 0.72)
        Revenue grew 12% year-over-year driven by strong agribusiness performance.
    """
    if not chunks:
        return "No relevant documents found for this query."

    lines: list[str] = []
    for chunk in chunks:
        # Build source line
        source = chunk.source_file
        if chunk.page_number is not None:
            source += f" — page {chunk.page_number}"
        if chunk.section_title:
            source += f", section: {chunk.section_title}"

        lines.append(f"[{chunk.rank}] {source}  (score: {chunk.score:.2f})")
        lines.append(chunk.content.strip())
        lines.append("")  # blank line between results

    return "\n".join(lines).rstrip()


def format_metric_rows(
    rows: list[dict],
    metric_name: str,
    unit: str = "",
) -> str:
    """
    Formats metric query results as a labelled table.

    Example output:
        Metric: ebitda_margin_pct [percent]

        Company        | Period   | Value
        ---------------|----------|-------
        Test Agri Co   | Q3 2024  | 25.00
        Test Agri Co   | Q4 2024  | 26.67
    """
    if not rows:
        return f"No data found for metric '{metric_name}'."

    unit_label = f" [{unit}]" if unit else ""
    lines: list[str] = [
        f"Metric: {metric_name}{unit_label}",
        "",
    ]

    # Header
    lines.append(f"{'Company':<20} | {'Period':<12} | Value")
    lines.append(f"{'-' * 20}-+-{'-' * 12}-+-------")

    for row in rows:
        company = str(row.get("company", ""))
        period = str(row.get("period_label", ""))
        value = row.get("value")
        value_str = f"{float(value):.2f}" if value is not None else "N/A"
        lines.append(f"{company:<20} | {period:<12} | {value_str}")

    return "\n".join(lines)


def format_sources(
    tables: list[str],
    documents: list[dict],
    schema_description: str,
) -> str:
    """
    Formats a summary of all available data for the LLM to orient itself.

    Includes:
    - Ingested documents (unstructured data, searchable via retrieve_docs)
    - Database tables/views (structured data, queryable via query_metrics)
    - Full schema description

    This output is what the LLM receives when it calls list_sources, which
    it should do when it encounters an unfamiliar query to understand what
    data is available before deciding how to answer.
    """
    lines: list[str] = ["=" * 60, "AVAILABLE DATA SOURCES", "=" * 60, ""]

    # Unstructured documents
    lines.append(f"INGESTED DOCUMENTS ({len(documents)} files):")
    if documents:
        for doc in documents:
            chunks = doc.get("chunk_count", 0)
            doc_type = doc.get("doc_type", "")
            lines.append(f"  • {doc['filename']} ({doc_type}, {chunks} chunks)")
    else:
        lines.append("  No documents ingested yet.")
    lines.append("")

    # Structured tables
    lines.append(f"DATABASE TABLES/VIEWS ({len(tables)}):")
    for table in tables:
        lines.append(f"  • {table}")
    lines.append("")

    # Full schema description
    lines.append(schema_description)

    return "\n".join(lines)

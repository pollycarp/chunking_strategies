"""
MCP Tool: query_metrics

Exposes the structured financial metrics layer (Phase 5) as an MCP tool.

Tool description design principle:
The LLM must understand that this is the tool for NUMBERS — exact figures
from the database. Keeping this distinct from retrieve_docs prevents the LLM
from confusing "look up a number" with "search a document".
"""

from __future__ import annotations

from asyncpg import Pool
from fastmcp import FastMCP

from src.mcp.output_formatters import format_metric_rows
from src.structured.metrics import get_metric, metric_query
from src.structured.query_engine import QueryEngine, UnsafeSQLError


def register(mcp: FastMCP, pool: Pool) -> None:
    """
    Registers the query_metrics tool on the given FastMCP server instance.
    """
    engine = QueryEngine(pool)

    @mcp.tool()
    async def query_metrics(
        metric_name: str,
        company: str | None = None,
        period_label: str | None = None,
    ) -> str:
        """
        Retrieve a named financial metric from the structured database.

        Use this tool for QUANTITATIVE questions about specific financial figures:
        - Revenue, gross profit, EBITDA, net income
        - Margin percentages (gross margin, EBITDA margin, net margin)
        - Balance sheet items (total assets, total debt, cash)
        - Leverage ratios (debt-to-EBITDA)

        Do NOT use this tool for qualitative questions or document content —
        use retrieve_docs instead for narrative, context, or explanations.

        Available metric names:
            revenue, gross_profit, ebitda, ebit, net_income,
            total_assets, total_debt, cash,
            gross_margin_pct, ebitda_margin_pct, net_margin_pct,
            debt_to_ebitda

        Parameters:
            metric_name  : One of the metric names listed above
            company      : Optional — exact company name to filter by
            period_label : Optional — period label to filter by, e.g. "Q3 2024" or "FY 2024"

        Returns a table of metric values, one row per matching company/period combination.
        """
        # Validate metric name before building SQL
        try:
            metric = get_metric(metric_name)
        except KeyError as exc:
            return str(exc)

        sql, params = metric_query(metric_name, company=company, period_label=period_label)

        try:
            rows = await engine.execute(sql, params)
        except UnsafeSQLError as exc:
            # Should never happen via metric_query — safety net
            return f"Query blocked: {exc}"

        return format_metric_rows(rows, metric_name, unit=metric.unit)

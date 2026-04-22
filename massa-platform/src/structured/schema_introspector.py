"""
Schema introspector — generates LLM-readable descriptions of the financial schema.

Why this exists:
An LLM writing SQL needs to know the schema. Two options:
  1. Hard-code the schema in the system prompt — gets stale as the DB evolves.
  2. Read it from the DB at runtime — always accurate, works after migrations.

This module implements option 2 by querying information_schema.columns.
The output is plain text structured for LLM consumption, not humans.
"""

from __future__ import annotations

from asyncpg import Pool

from src.structured.metrics import list_metrics

# Tables/views the LLM is allowed to query.
# Explicitly allowlisted so we never expose internal system tables.
_QUERYABLE_OBJECTS = (
    "dim_company",
    "dim_period",
    "fact_financials",
    "financial_metrics",
)


class SchemaIntrospector:
    """
    Reads the live DB schema and formats it for LLM consumption.

    Usage:
        introspector = SchemaIntrospector(pool)
        description = await introspector.get_schema_description()
        # → pass description into the system prompt or tool context
    """

    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def list_tables(self) -> list[str]:
        """
        Returns the names of all queryable tables and views, sorted.
        Used by MCP's list_sources tool to tell the LLM what data is available.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT table_name
                FROM   information_schema.tables
                WHERE  table_schema = 'public'
                  AND  table_name   = ANY($1::text[])
                ORDER  BY table_name
                """,
                list(_QUERYABLE_OBJECTS),
            )
        return [row["table_name"] for row in rows]

    async def get_schema_description(self) -> str:
        """
        Returns a plain-text schema description formatted for an LLM.

        Format:
            TABLE dim_company
              id (integer) — NOT NULL
              name (text) — NOT NULL
              ...

            VIEW financial_metrics
              company (text)
              ...

            METRICS
              revenue — Total revenue (top line) for the period [currency]
              ...

        The LLM can include this verbatim in its reasoning to write correct SQL.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    t.table_name,
                    t.table_type,
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.ordinal_position
                FROM  information_schema.tables  t
                JOIN  information_schema.columns c
                      ON  c.table_name   = t.table_name
                      AND c.table_schema = t.table_schema
                WHERE t.table_schema = 'public'
                  AND t.table_name   = ANY($1::text[])
                ORDER BY t.table_name, c.ordinal_position
                """,
                list(_QUERYABLE_OBJECTS),
            )

        # Group columns by table
        tables: dict[str, list] = {}
        table_types: dict[str, str] = {}
        for row in rows:
            name = row["table_name"]
            if name not in tables:
                tables[name] = []
                table_types[name] = row["table_type"]
            nullable = "" if row["is_nullable"] == "YES" else " — NOT NULL"
            tables[name].append(f"  {row['column_name']} ({row['data_type']}){nullable}")

        lines: list[str] = ["=" * 60, "FINANCIAL DATABASE SCHEMA", "=" * 60, ""]

        for table_name in sorted(tables):
            kind = "VIEW" if table_types[table_name] == "VIEW" else "TABLE"
            lines.append(f"{kind} {table_name}")
            lines.extend(tables[table_name])
            lines.append("")

        # Append named metric definitions
        lines += ["─" * 60, "NAMED METRICS (query via financial_metrics view)", "─" * 60]
        for m in list_metrics():
            lines.append(f"  {m['name']} [{m['unit']}]")
            lines.append(f"    {m['description']}")
        lines.append("")

        lines += [
            "─" * 60,
            "QUERY RULES",
            "─" * 60,
            "  • SELECT only — INSERT/UPDATE/DELETE/DDL are blocked",
            "  • Use financial_metrics view for metric queries (avoids manual JOINs)",
            "  • Filter by: company (exact name), period_label (e.g. 'Q3 2024')",
            "  • Monetary values are in the currency of the source document",
            "",
        ]

        return "\n".join(lines)

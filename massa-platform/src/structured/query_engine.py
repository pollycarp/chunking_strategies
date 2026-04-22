"""
Safe SQL query engine for LLM-generated queries.

Three safety layers:
1. Write detection   — rejects INSERT/UPDATE/DELETE/DDL before reaching the DB
2. Statement timeout — Postgres kills long-running queries automatically
3. Row limit         — LIMIT is injected if absent to cap result set size

Why enforce safety here (application layer) rather than via a read-only DB user?
Both are best practice — defence in depth. A read-only user at the DB level is
the primary guard. This layer catches mistakes earlier (before a round-trip to
Postgres) and produces clear error messages for the LLM to reason about.
"""

from __future__ import annotations

import re

from asyncpg import Pool


# ---------------------------------------------------------------------------
# Write / DDL detection
# ---------------------------------------------------------------------------

_WRITE_PATTERN = re.compile(
    r"""
    \b(
        INSERT   |
        UPDATE   |
        DELETE   |
        DROP     |
        TRUNCATE |
        ALTER    |
        CREATE   |
        GRANT    |
        REVOKE   |
        COPY     |
        MERGE
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when SQL contains a write or DDL operation."""


class QueryEngine:
    """
    Executes read-only SQL queries against the financial schema.

    Usage:
        engine = QueryEngine(pool)
        rows = await engine.execute("SELECT * FROM financial_metrics WHERE company = $1", ["Agri Co"])

    Raises:
        UnsafeSQLError  : if the SQL contains any write/DDL keyword
        asyncpg.QueryCanceledError : if the query exceeds the timeout
    """

    def __init__(
        self,
        pool: Pool,
        max_rows: int = 500,
        timeout_ms: int = 10_000,  # 10 seconds
    ) -> None:
        self._pool = pool
        self._max_rows = max_rows
        self._timeout_ms = timeout_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        sql: str,
        params: list | None = None,
    ) -> list[dict]:
        """
        Validates and executes a SELECT query.

        Returns a list of dicts (one per row), with column names as keys.
        An empty result set returns [].
        """
        self._validate(sql)
        safe_sql = self._apply_limit(sql)

        async with self._pool.acquire() as conn:
            # SET LOCAL applies only within this transaction — does not affect
            # the connection after it's returned to the pool.
            async with conn.transaction():
                await conn.execute(
                    f"SET LOCAL statement_timeout = '{self._timeout_ms}'"
                )
                rows = await conn.fetch(safe_sql, *(params or []))

        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, sql: str) -> None:
        """
        Raises UnsafeSQLError if the SQL contains a write/DDL keyword.

        This is intentionally conservative: a false positive (blocking a safe
        query) is far preferable to a false negative (allowing a destructive one).
        """
        match = _WRITE_PATTERN.search(sql)
        if match:
            raise UnsafeSQLError(
                f"Write operation '{match.group()}' is not permitted. "
                "Only SELECT queries are allowed."
            )

    def _apply_limit(self, sql: str) -> str:
        """
        Appends LIMIT {max_rows} if the query has no LIMIT clause.

        Strips trailing semicolons first — Postgres does not need them for
        programmatic queries and they can cause parse errors when appending.
        """
        if _LIMIT_PATTERN.search(sql):
            return sql
        return f"{sql.rstrip().rstrip(';')} LIMIT {self._max_rows}"

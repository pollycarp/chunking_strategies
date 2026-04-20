"""
Phase 1 Tests: Infrastructure

Three tests covering:
1. DB connection — can we connect and run a query?
2. Schema smoke test — did migrations create the expected tables?
3. Config validation — does missing env var raise a clear error?
"""

import pytest
from pydantic import ValidationError

from src.config import Settings


# ---------------------------------------------------------------------------
# Test 1: DB Connection
# ---------------------------------------------------------------------------

async def test_db_connection(db_pool):
    """
    Assert that the pool can acquire a connection and run a trivial query.

    'SELECT 1' is the standard DB liveness check — if this works,
    the connection, authentication, and network path are all correct.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")

    assert result == 1, "Expected SELECT 1 to return 1"


# ---------------------------------------------------------------------------
# Test 2: Schema Smoke Test
# ---------------------------------------------------------------------------

async def test_schema_migrations_table_exists(db_pool):
    """
    Assert that the schema_migrations table was created by our migration runner.

    We query information_schema.tables — a standard Postgres catalog that lists
    every table in the database. If our migration ran, this table must be there.
    """
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND   table_name   = 'schema_migrations'
            )
        """)

    assert exists is True, "schema_migrations table should exist after migrations"


async def test_pgvector_extension_enabled(db_pool):
    """
    Assert that the pgvector extension is active.

    pg_extension is a Postgres catalog table listing installed extensions.
    Without pgvector, we cannot store or search vector columns in later phases.
    """
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM pg_extension
                WHERE extname = 'vector'
            )
        """)

    assert exists is True, "pgvector extension should be enabled after migrations"


async def test_migration_was_recorded(db_pool):
    """
    Assert that migration 001_init.sql is recorded in schema_migrations.

    This verifies the migration runner inserted a row after applying the file,
    which is the mechanism that prevents the same migration from running twice.
    """
    async with db_pool.acquire() as conn:
        version = await conn.fetchval("""
            SELECT version FROM schema_migrations WHERE version = '001_init.sql'
        """)

    assert version == "001_init.sql", "001_init.sql should be recorded as applied"


# ---------------------------------------------------------------------------
# Test 3: Config Validation
# ---------------------------------------------------------------------------

def test_config_raises_on_missing_required_field():
    """
    Assert that instantiating Settings without a required field raises ValidationError.

    This is a pure unit test — no DB needed. We deliberately omit postgres_password
    and expect pydantic to raise immediately with a clear error, rather than letting
    the app start and fail silently later.
    """
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_db="test",
            postgres_user="test",
            # postgres_password intentionally omitted
        )

    # The error message should tell us exactly which field is missing
    assert "postgres_password" in str(exc_info.value)


def test_config_dsn_format():
    """
    Assert that the database_dsn property produces a correctly formatted connection string.

    asyncpg will fail with a cryptic error if the DSN format is wrong.
    Testing it here catches formatting bugs before they hit the DB layer.
    """
    s = Settings(
        postgres_host="myhost",
        postgres_port=5433,
        postgres_db="mydb",
        postgres_user="myuser",
        postgres_password="mypass",
    )

    assert s.database_dsn == "postgresql://myuser:mypass@myhost:5433/mydb"

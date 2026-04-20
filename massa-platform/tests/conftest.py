import pytest
import asyncpg
from asyncpg import Pool

from src.config import settings
from src.db.migrate import run_migrations


@pytest.fixture(scope="session")
async def db_pool() -> Pool:
    """
    Session-scoped fixture: creates the pool once for all tests.

    'scope=session' means this setup runs once per test session, not once
    per test function. Avoids the overhead of creating/destroying a pool
    for every single test.
    """
    # Run migrations so the schema is up to date before any test touches the DB
    await run_migrations()

    pool = await asyncpg.create_pool(dsn=settings.database_dsn, min_size=1, max_size=3)
    yield pool
    await pool.close()

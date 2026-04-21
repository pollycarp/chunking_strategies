import asyncio

import pytest
import asyncpg
from asyncpg import Pool
from pgvector.asyncpg import register_vector

from src.config import settings
from src.db.migrate import run_migrations


@pytest.fixture(scope="session", autouse=True)
def run_db_migrations():
    """
    Sync fixture — runs migrations once before any test using its own event loop.

    Why sync? pytest-asyncio gives each async fixture a loop tied to its scope.
    A session-scoped async fixture would run in the session loop, but our
    function-scoped db_pool runs in the test's function loop — they never match.
    Making this sync means it calls asyncio.run() and exits cleanly with no
    shared loop state, so the test loops are never polluted.
    """
    asyncio.run(run_migrations())


@pytest.fixture
async def db_pool() -> Pool:
    """
    Function-scoped fixture: creates a fresh pool for each test.

    This avoids the 'attached to a different loop' error that occurs when
    an asyncpg pool created in one event loop is used in another.
    pytest-asyncio gives each test its own event loop, so the pool must
    be created inside that same loop.
    """
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        min_size=1,
        max_size=3,
        init=register_vector,   # register pgvector codec on every connection
    )
    yield pool
    await pool.close()

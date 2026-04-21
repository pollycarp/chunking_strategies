import asyncpg
from asyncpg import Connection, Pool
from pgvector.asyncpg import register_vector

from src.config import settings

# Module-level pool — created once at startup, shared across the app
_pool: Pool | None = None


async def _init_connection(conn: Connection) -> None:
    """
    Called by asyncpg on every new connection the pool creates.

    Why register_vector here?
    asyncpg has no built-in codec for pgvector's `vector` type.
    register_vector() tells asyncpg how to:
      - Encode: Python list[float] → Postgres vector literal
      - Decode: Postgres vector bytes → Python list[float]
    Without this, passing a list[float] to a vector column raises:
      DataError: invalid input for query argument $n (expected str, got list)
    """
    await register_vector(conn)


async def create_pool() -> Pool:
    """
    Creates the asyncpg connection pool.
    Call once at application startup.

    min_size=2  — always keep 2 connections open (avoids cold-start latency)
    max_size=10 — never open more than 10 connections (protects the DB)
    init        — registers the pgvector codec on every new connection
    """
    global _pool
    _pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        database=settings.postgres_db,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )
    return _pool


async def get_pool() -> Pool:
    """
    Returns the active pool. Raises if create_pool() was never called.
    Use this everywhere in the app that needs a DB access.
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialised. Call create_pool() at startup.")
    return _pool


async def close_pool() -> None:
    """
    Gracefully closes all connections in the pool.
    Call at application shutdown.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

import asyncpg
from asyncpg import Pool

from src.config import settings

# Module-level pool — created once at startup, shared across the app
_pool: Pool | None = None


async def create_pool() -> Pool:
    """
    Creates the asyncpg connection pool.
    Call once at application startup.

    min_size=2  — always keep 2 connections open (avoids cold-start latency)
    max_size=10 — never open more than 10 connections (protects the DB)
    """
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.database_dsn,
        min_size=2,
        max_size=10,
    )
    return _pool


async def get_pool() -> Pool:
    """
    Returns the active pool. Raises if create_pool() was never called.
    Use this everywhere in the app that needs a DB connection.
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

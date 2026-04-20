import asyncio
from pathlib import Path

import asyncpg

from src.config import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


async def run_migrations() -> None:
    """
    Applies any unapplied SQL migrations in order.

    How it works:
    1. Connect directly (not via pool — pool needs schema to exist first)
    2. Ensure schema_migrations table exists
    3. For each .sql file sorted by name, check if it's already been applied
    4. If not, run it and record it in schema_migrations
    """
    conn = await asyncpg.connect(dsn=settings.database_dsn)

    try:
        # Ensure migration tracking table exists before we check it
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version     TEXT        PRIMARY KEY,
                applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        # Get the set of already-applied migrations
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        applied = {row["version"] for row in rows}

        # Run migrations in filename order (001_, 002_, ...)
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        for migration_file in migration_files:
            version = migration_file.name

            if version in applied:
                print(f"  [skip] {version} already applied")
                continue

            sql = migration_file.read_text(encoding="utf-8")

            # Run the migration and record it atomically
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    version,
                )

            print(f"  [ok]   {version} applied")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migrations())

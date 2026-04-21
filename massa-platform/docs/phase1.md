# Phase 1: Project Foundation & Infrastructure

## Overview

Phase 1 establishes the foundation that every subsequent phase builds on. It covers the development environment, database setup, configuration management, async database access, and schema migrations. By the end of this phase you have a running Postgres + pgvector instance, a validated config system, a connection pool, and a migration runner — all verified by 6 passing tests.

---

## Table of Contents

1. [Concepts](#concepts)
   - [Docker & Docker Compose](#docker--docker-compose)
   - [PostgreSQL & pgvector](#postgresql--pgvector)
   - [asyncpg & Connection Pooling](#asyncpg--connection-pooling)
   - [Pydantic Settings](#pydantic-settings)
   - [SQL Migrations](#sql-migrations)
   - [Async I/O in Python](#async-io-in-python)
2. [Project Structure](#project-structure)
3. [File Reference](#file-reference)
4. [Setup & Running](#setup--running)
5. [Tests](#tests)
6. [Lessons Learned](#lessons-learned)

---

## Concepts

### Docker & Docker Compose

**What it is:**
Docker packages an application and all its dependencies into a self-contained unit called a *container*. The container runs identically on any machine — your laptop, a teammate's machine, or a production server — because it includes everything the software needs to run.

**Why we use it here:**
Instead of installing Postgres directly on your machine (dealing with version conflicts, OS-specific config, and leftover state), we define the entire database stack in a single `docker-compose.yml` file and spin it up with one command.

**Key concepts:**
- **Image** — a read-only blueprint (e.g. `pgvector/pgvector:pg16` is Postgres 16 with pgvector pre-compiled)
- **Container** — a running instance of an image
- **Volume** — persists data between container restarts; without it all data is lost when the container stops
- **Port mapping** — `"5433:5432"` means port 5433 on your machine maps to port 5432 inside the container
- **Healthcheck** — Docker polls `pg_isready` to know when Postgres is actually ready to accept connections, not just started

**Why port 5433 instead of 5432:**
Windows often has a local Postgres installation running on 5432. Mapping Docker to 5433 avoids the collision — the app connects to Docker on 5433, not the local Postgres on 5432.

---

### PostgreSQL & pgvector

**PostgreSQL:**
A production-grade relational database. We use it for both structured data (financial metrics, schema migrations) and vector data (embeddings). Using one database for both avoids maintaining two separate systems.

**pgvector:**
An open-source Postgres extension that adds:
- A `vector` column type to store dense float arrays (embeddings)
- Approximate Nearest Neighbor (ANN) index types: `ivfflat` and `hnsw`
- Distance operators for similarity search: `<=>` (cosine), `<->` (L2), `<#>` (inner product)

**Why pgvector over a dedicated vector DB:**
For our use case (institutional clients, moderate data volumes, need for transactional integrity alongside vector search) pgvector is a pragmatic choice. It avoids operational complexity of running a second database system. Dedicated vector databases (Pinecone, Weaviate, Qdrant) offer advantages at very high scale or with advanced filtering requirements.

**Key Postgres concepts used:**
- `information_schema.tables` — a standard catalog listing all tables in the database; used in tests to verify schema state
- `pg_extension` — a catalog table listing installed extensions
- `TIMESTAMPTZ` — timezone-aware timestamp; always store timestamps with timezone in production
- `TEXT PRIMARY KEY` — we use migration filenames as primary keys (e.g. `001_init.sql`)

---

### asyncpg & Connection Pooling

**Why async matters:**
Our platform will eventually run an API server, an ingestion pipeline, and a retrieval engine — all making DB queries concurrently. Synchronous drivers block the thread while waiting for a DB response. With `asyncpg`, while one query waits on the database, the Python event loop can run other coroutines — the thread is never idle.

**asyncpg vs psycopg2:**
| | asyncpg | psycopg2 |
|---|---|---|
| I/O model | Async (non-blocking) | Sync (blocking) |
| Performance | Fastest Python Postgres driver | Moderate |
| API | `await conn.fetch(...)` | `cursor.execute(...)` |
| Use case | High-concurrency servers | Simple scripts |

**Connection pooling:**
Opening a new TCP connection to Postgres costs ~50–100ms (DNS lookup, TCP handshake, TLS, auth). A pool keeps connections open and reuses them:

```
min_size=1  → always keep at least 1 connection open (avoids cold-start latency)
max_size=3  → never open more than 3 connections in tests (protects DB from overload)
```

In the production app (non-test), we use `min_size=2, max_size=10`.

**The event loop problem (and how we solved it):**
asyncpg connections are bound to the event loop they were created in. pytest-asyncio creates a fresh event loop per test function. If the connection pool is created in one loop (e.g. a session-scoped fixture loop) and used in another (the test's function loop), asyncpg raises:

```
RuntimeError: ... got Future ... attached to a different loop
```

**Solution:** The `db_pool` fixture is function-scoped — a fresh pool is created inside each test's event loop. The migration fixture is made synchronous (calls `asyncio.run()` internally) so it never participates in the test event loop at all.

---

### Pydantic Settings

**What it is:**
`pydantic-settings` reads configuration from environment variables and `.env` files, validates types, and raises a clear error if required values are missing.

**Why not just use `os.environ`:**
```python
# Bad — silent failure, wrong type, no validation
host = os.environ.get("POSTGRES_HOST", "localhost")
port = int(os.environ.get("POSTGRES_PORT", "5432"))  # crashes if value isn't a number

# Good — validated at startup, clear error if missing
class Settings(BaseSettings):
    postgres_host: str
    postgres_port: int = 5432  # default value, type-checked
```

**How it works:**
1. On import, `Settings()` is instantiated
2. pydantic-settings looks for each field in: environment variables → `.env` file → default value
3. If a required field (no default) isn't found anywhere, it raises `ValidationError` immediately
4. Type coercion is automatic: the string `"5433"` from `.env` becomes the integer `5433`

**The `_env_file=None` pattern:**
In tests, we pass `_env_file=None` to the Settings constructor to prevent it from reading `.env`. This lets us test validation behaviour without the real credentials interfering.

**Security rule:**
`.env` is in `.gitignore` and never committed. `.env.example` is committed as a template showing what variables are needed. Real secrets never appear in git history.

---

### SQL Migrations

**What they are:**
Numbered SQL files that define the database schema changes over time:
```
001_init.sql   → enable pgvector, create migration tracking table
002_vectors.sql → (Phase 2) add embeddings table with hnsw index
003_financial_facts.sql → (Phase 5) add fact/dimension tables
```

**Why not just run CREATE TABLE manually:**
- Manual changes aren't reproducible — another developer or deployment environment won't have them
- There's no audit trail of what changed and when
- You can't roll forward safely in CI/CD

**How our migration runner works (`src/db/migrate.py`):**
1. Connect directly (not via pool — pool requires schema to exist first)
2. Read all `.sql` files in `migrations/` sorted by filename
3. For each file, check if its name is in `schema_migrations` table
4. If not: run it in a transaction, then insert its name into `schema_migrations`
5. If yes: skip it

**Idempotency:**
Running the migration runner twice is safe — already-applied migrations are skipped. This is critical for CI pipelines where the runner might execute on every deployment.

---

### Async I/O in Python

**The event loop:**
Python's `asyncio` runs a single-threaded event loop. Instead of blocking on I/O (waiting for a DB response, HTTP response, file read), async code suspends itself with `await` and lets the loop run other tasks.

```python
# Synchronous — thread blocks here until DB responds
result = conn.fetchval("SELECT 1")

# Asynchronous — event loop runs other tasks while waiting
result = await conn.fetchval("SELECT 1")
```

**`async def` vs `def`:**
- `async def` defines a coroutine — a function that can be paused and resumed
- Must be called with `await` from another coroutine, or run with `asyncio.run()`
- `asyncio.run()` is the entry point that creates an event loop and runs one top-level coroutine

**`async with` and `async for`:**
Context managers and iterators can also be async. `async with pool.acquire() as conn:` acquires a connection from the pool asynchronously, then releases it back when the block exits — even if an exception is raised.

---

## Project Structure

```
massa-platform/
├── docker-compose.yml              # Postgres + pgvector container definition
├── .env                            # Local secrets — never committed
├── .env.example                    # Template showing required env vars
├── .gitignore                      # Excludes .env, __pycache__, .venv
├── pyproject.toml                  # Project metadata, dependencies, pytest config
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic settings — single source of truth for config
│   └── db/
│       ├── __init__.py
│       ├── connection.py           # asyncpg pool factory (create, get, close)
│       ├── migrate.py              # Migration runner
│       └── migrations/
│           └── 001_init.sql        # Enable pgvector, create schema_migrations table
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures: migrations runner, db pool
    └── test_db_connection.py       # 6 tests covering DB, schema, and config
```

---

## File Reference

### `docker-compose.yml`
Defines the Postgres service using the `pgvector/pgvector:pg16` image. Maps host port 5433 to container port 5432 (avoids conflict with any local Postgres on 5432). Persists data with a named volume. Includes a healthcheck so dependent services wait for readiness.

### `.env` / `.env.example`
Holds all environment-specific configuration. `.env` is local only. `.env.example` is the committed template. The app reads these via `pydantic-settings`.

### `src/config.py`
Single `Settings` class using `pydantic-settings`. All config fields are typed. The `database_dsn` property builds the asyncpg connection string. Instantiated once at module level — import `settings` everywhere rather than re-instantiating.

### `src/db/connection.py`
Module-level pool management. Three functions:
- `create_pool()` — call once at startup
- `get_pool()` — call everywhere in the app that needs DB access
- `close_pool()` — call at shutdown

### `src/db/migrate.py`
Migration runner. Reads `.sql` files in order, tracks applied migrations in `schema_migrations`, and runs each one in a transaction. Safe to run multiple times.

### `src/db/migrations/001_init.sql`
Two operations:
1. `CREATE EXTENSION IF NOT EXISTS vector` — enables pgvector
2. Creates `schema_migrations` table for migration tracking

### `tests/conftest.py`
Two fixtures:
- `run_db_migrations` — sync, session-scoped, autouse; runs migrations once per test session using `asyncio.run()`
- `db_pool` — async, function-scoped; creates a fresh asyncpg pool per test to avoid event loop conflicts

### `tests/test_db_connection.py`
Six tests across three categories — see [Tests](#tests) section.

---

## Setup & Running

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ installed
- `uv` accessible via `python -m uv` (install with `pip install uv`)

### Step-by-step

**1. Start the database:**
```bash
cd massa-platform
docker compose up -d
```

Verify it's healthy:
```bash
docker ps
# massa_db should show (healthy)
```

**2. Install dependencies:**
```bash
python -m uv sync --all-extras
```

**3. Run migrations:**
```bash
python -m uv run python -m src.db.migrate
```

Expected output:
```
  [ok]   001_init.sql applied
```

Running again produces:
```
  [skip] 001_init.sql already applied
```

**4. Run tests:**
```bash
python -m uv run pytest tests/ -v
```

Expected output:
```
tests/test_db_connection.py::test_db_connection PASSED
tests/test_db_connection.py::test_schema_migrations_table_exists PASSED
tests/test_db_connection.py::test_pgvector_extension_enabled PASSED
tests/test_db_connection.py::test_migration_was_recorded PASSED
tests/test_db_connection.py::test_config_raises_on_missing_required_field PASSED
tests/test_db_connection.py::test_config_dsn_format PASSED
6 passed in 0.47s
```

**5. Stop the database:**
```bash
docker compose down
```

To also delete all stored data:
```bash
docker compose down -v
```

---

## Tests

### `test_db_connection`
**Type:** Integration
**What it tests:** The asyncpg pool can acquire a connection and `SELECT 1` returns 1.
**Why it matters:** Verifies the full connection path — Docker networking, port mapping, credentials, and asyncpg are all working together.

### `test_schema_migrations_table_exists`
**Type:** Integration
**What it tests:** The `schema_migrations` table exists in the `public` schema by querying `information_schema.tables`.
**Why it matters:** Confirms the migration runner created its tracking table. If this fails, no migrations will be tracked.

### `test_pgvector_extension_enabled`
**Type:** Integration
**What it tests:** The `vector` extension is present in `pg_extension`.
**Why it matters:** Without this, any attempt to create a `vector` column in Phase 2 will fail with a cryptic error.

### `test_migration_was_recorded`
**Type:** Integration
**What it tests:** `001_init.sql` has a row in `schema_migrations`.
**Why it matters:** Validates the idempotency mechanism — if recording fails, the same migration would run again on the next startup.

### `test_config_raises_on_missing_required_field`
**Type:** Unit (no DB)
**What it tests:** Instantiating `Settings` without `postgres_password` (and with `_env_file=None`) raises `pydantic_core.ValidationError` containing the field name.
**Why it matters:** Confirms the app fails fast with a clear error rather than starting and crashing later with a cryptic DB error.

### `test_config_dsn_format`
**Type:** Unit (no DB)
**What it tests:** `settings.database_dsn` produces a correctly formatted `postgresql://user:pass@host:port/db` string.
**Why it matters:** asyncpg will raise a cryptic parse error if the DSN is malformed. Testing it here catches formatting bugs before they reach the DB layer.

---

## Lessons Learned

These are real issues encountered during Phase 1 that you will encounter in production work:

### Port conflicts
**Problem:** Two processes listening on the same port — a local Postgres and Docker — cause the app to connect to the wrong one silently.
**Detection:** `netstat -ano | findstr :5432` shows multiple PIDs on the same port. `tasklist /FI "PID eq <pid>"` identifies each process.
**Fix:** Map Docker to a different host port (`5433:5432`).
**Lesson:** Always verify what's actually listening on a port before debugging authentication errors.

### asyncpg + pytest-asyncio event loop mismatch
**Problem:** asyncpg connections and pools are bound to the event loop they were created in. pytest-asyncio creates a new loop per test. A pool created in a session-scoped fixture runs in a different loop than the test — asyncpg raises `RuntimeError: attached to a different loop`.
**Fix:** Make the pool fixture function-scoped (one pool per test, same loop). Make one-time setup (migrations) a sync fixture using `asyncio.run()`.
**Lesson:** When mixing asyncio with test frameworks, always be explicit about which loop owns which resource.

### pydantic-settings reads environment even in tests
**Problem:** Testing that a missing field raises `ValidationError` fails because pydantic-settings finds the value in `.env` and fills it in automatically.
**Fix:** Pass `_env_file=None` to the Settings constructor in the test to disable `.env` reading.
**Lesson:** Isolation matters in tests. External state (files, environment variables) must be explicitly controlled.

### scram-sha-256 authentication with asyncpg on Python 3.14
**Problem:** asyncpg failed with `InvalidPasswordError` even with correct credentials. Postgres 16 defaults to `scram-sha-256`; asyncpg had compatibility issues on Python 3.14.
**Fix:** Changed `pg_hba.conf` to use `md5` for host connections via `sed` inside the container, then restarted.
**Lesson:** Always check Postgres `pg_hba.conf` when debugging authentication failures — it controls which auth method is used per connection type.

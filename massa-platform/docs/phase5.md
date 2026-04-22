# Phase 5: Structured Data Layer

## What We Built

A typed, safe interface for financial structured data — the quantitative complement to the document retrieval system built in Phases 3 and 4.

```
LLM question: "What was Agri Co's EBITDA margin in Q3 2024?"
        │
        ├──► metric_query("ebitda_margin_pct", company="Agri Co", period_label="Q3 2024")
        │           │
        │    QueryEngine.execute(sql, params)
        │           │
        │    ┌──────▼──────────────────────────┐
        │    │ financial_metrics VIEW           │
        │    │ (joins fact + dims, computes     │
        │    │  ratios like ebitda/revenue*100) │
        │    └──────────────────────────────────┘
        │           │
        └──► "25.00%"
```

---

## Key Concepts

### Star Schema — `src/db/migrations/004_financial_facts.sql`

A **star schema** is the standard design for financial data warehouses. It separates data into two categories:

**Dimension tables** — the context (who, when, where):
- `dim_company` — portfolio company metadata (name, sector, country)
- `dim_period` — time period definitions (year, quarter, label, date range)

**Fact table** — the measurements:
- `fact_financials` — numeric financial data, with foreign keys to both dimension tables

The "star" shape is the fact table at the centre with dimension tables radiating outward. This design is simple, fast to query, and easy for an LLM to understand because the JOIN structure is always the same.

```
      dim_company ─────┐
                        ├── fact_financials
      dim_period  ─────┘
```

**Why this structure over a flat table?**

A flat table `(company_name, quarter, revenue, ebitda, ...)` duplicates company metadata on every row. If a company changes its sector classification, you'd need to update every historical row. With a dimension table, you update one row in `dim_company`. This is called **normalisation**.

---

### `NUMERIC` vs `FLOAT` — the most important financial data rule

```sql
-- WRONG for financial data:
revenue FLOAT

-- CORRECT:
revenue NUMERIC(18, 2)
```

`FLOAT` uses **IEEE 754 binary floating-point** arithmetic — it trades precision for speed. In binary, many decimal fractions (like 0.1) cannot be represented exactly:

```python
>>> 0.1 + 0.2
0.30000000000000004   ← not 0.3
```

For most engineering this is fine. For financial data, a 0.00000000000000004 error on a single row compounds into material misstatements at aggregate scale (millions of rows, billions of dollars).

`NUMERIC(18, 2)` stores an **exact decimal value** up to 18 total digits with 2 decimal places. It's slower but always precise. Every monetary column in `fact_financials` uses it.

---

### The `financial_metrics` View — Semantic Layer

A **view** in SQL is a stored query that looks like a table. The `financial_metrics` view joins the three tables and pre-computes derived ratios:

```sql
CREATE OR REPLACE VIEW financial_metrics AS
SELECT
    c.name      AS company,
    p.period_label,
    f.revenue,
    f.ebitda,
    CASE WHEN f.revenue <> 0
         THEN ROUND(f.ebitda / f.revenue * 100, 2)
    END AS ebitda_margin_pct,
    ...
```

**Why a view instead of computing in the application?**

1. **Consistency** — every consumer (LLM, analyst, dashboard) gets the same formula. If you update the EBITDA margin definition, you change one place.
2. **LLM simplicity** — the LLM queries `SELECT ebitda_margin_pct FROM financial_metrics WHERE company = 'X'`. It doesn't need to know the formula. This dramatically reduces hallucination risk.
3. **Division-by-zero safety** — the `CASE WHEN revenue <> 0` guard means a company with zero revenue returns `NULL` for margin ratios rather than crashing with a division error.

This concept — pre-defining business metrics in a shared layer — is called a **semantic layer** in data engineering. Modern tools like dbt, Looker's LookML, and Cube.js implement this at scale.

---

### Named Metrics — `src/structured/metrics.py`

The `METRICS` registry maps machine-readable names to their descriptions, view column names, and units:

```python
"ebitda_margin_pct": Metric(
    name="ebitda_margin_pct",
    description="EBITDA as a percentage of revenue",
    column="ebitda_margin_pct",
    unit="percent",
)
```

`metric_query(name, company, period_label)` builds a parameterised SQL query:

```python
sql, params = metric_query("ebitda_margin_pct", company="Agri Co", period_label="Q3 2024")
# sql   → "SELECT company, period_label, ebitda_margin_pct AS value FROM financial_metrics WHERE company = $1 AND period_label = $2 ORDER BY year, quarter NULLS LAST"
# params → ["Agri Co", "Q3 2024"]
```

The LLM never constructs SQL by hand — it calls `metric_query()` with named arguments. This removes an entire class of SQL injection and formula errors.

---

### Query Safety — `src/structured/query_engine.py`

An LLM connected to a live database is dangerous without guards. `QueryEngine` implements three safety layers:

**Layer 1 — Write detection (application level)**

A regex checks for write/DDL keywords before the query reaches Postgres:

```python
_WRITE_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY|MERGE)\b',
    re.IGNORECASE,
)
```

If matched, `UnsafeSQLError` is raised immediately — no round-trip to the database. The error message names the blocked keyword so the LLM can reason about why its query failed.

**Layer 2 — Statement timeout (database level)**

```python
await conn.execute(f"SET LOCAL statement_timeout = '{self._timeout_ms}'")
```

`SET LOCAL` applies only within the current transaction. When the connection is returned to the pool, the timeout resets. This prevents accidental or adversarial queries (e.g. `SELECT * FROM fact_financials` on a 10M-row table) from blocking the connection pool indefinitely.

**Layer 3 — Row limit**

```python
def _apply_limit(self, sql: str) -> str:
    if not _LIMIT_PATTERN.search(sql):
        return f"{sql.rstrip().rstrip(';')} LIMIT {self._max_rows}"
    return sql
```

If the LLM forgets `LIMIT`, the engine adds one. This prevents accidentally loading millions of rows into application memory.

**Why all three?**

Defence in depth. The read-only DB user (a best practice we'll add to the Docker setup) is the primary guard at the infrastructure level. The application-level checks catch mistakes earlier with better error messages. The three layers together make the system safe even if one layer is misconfigured.

---

### Schema Introspection — `src/structured/schema_introspector.py`

Before an LLM can write SQL, it needs to know what tables and columns exist. `SchemaIntrospector` reads `information_schema.columns` at runtime and formats it as LLM-consumable text:

```
TABLE dim_company
  id (integer) — NOT NULL
  name (text) — NOT NULL
  sector (text)
  country (text) — NOT NULL

VIEW financial_metrics
  company (text)
  period_label (text)
  revenue (numeric)
  ebitda_margin_pct (numeric)
  ...

NAMED METRICS
  ebitda_margin_pct [percent]
    EBITDA as a percentage of revenue
  ...

QUERY RULES
  • SELECT only — INSERT/UPDATE/DELETE/DDL are blocked
  • Use financial_metrics view for metric queries
```

**Why read from `information_schema` instead of hard-coding?**

Hard-coded schema descriptions go stale. When you run a migration (e.g. add a new column), the description is automatically correct on the next call. This is especially important as the project evolves through Phase 6 (MCP) where this description will be injected into the LLM's system prompt.

---

## File Structure

```
src/
├── db/migrations/
│   └── 004_financial_facts.sql   — dim_company, dim_period, fact_financials, financial_metrics view
└── structured/
    ├── __init__.py
    ├── metrics.py                — Metric dataclass, METRICS registry, metric_query(), list_metrics()
    ├── query_engine.py           — QueryEngine with write detection, timeout, row limit
    └── schema_introspector.py   — SchemaIntrospector reads information_schema, formats for LLM
```

---

## How It Connects to Other Phases

| Phase | Connection to Phase 5 |
|---|---|
| Phase 1 (DB pool) | `asyncpg.Pool` passed into `QueryEngine` and `SchemaIntrospector` |
| Phase 4 (retrieval) | Retrieval answers "what does the document say?"; Phase 5 answers "what are the exact numbers?" — two complementary question types |
| Phase 6 (MCP, next) | `query_metrics` tool calls `metric_query()` + `QueryEngine`; `list_sources` calls `SchemaIntrospector.list_tables()` |

The platform now has two data surfaces:
- **Unstructured** (Phase 3+4): documents → chunks → embeddings → retrieval
- **Structured** (Phase 5): financial metrics → star schema → safe SQL queries

Phase 6 (MCP) will expose both surfaces as tools the LLM can call.

---

## Tests — `tests/test_structured.py`

27 tests across 4 categories:

| Category | Tests | What's verified |
|---|---|---|
| Schema | 4 | NUMERIC on all monetary columns; FK constraints exist; metrics view queryable; UNIQUE on company name |
| Metrics | 9 | get_metric returns correct object; unknown raises KeyError; list_metrics returns all 12; SQL built correctly; revenue/EBITDA margin/debt-to-EBITDA computed correctly against fixture data; multi-period ordering |
| QueryEngine | 9 | SELECT executes; INSERT/UPDATE/DELETE/DROP/TRUNCATE all blocked; row limit applied; existing LIMIT respected; case-insensitive detection |
| SchemaIntrospector | 5 | All 4 tables listed; description contains table names, key columns, metric names, query rules |

**Fixture design:**
The `financial_fixture` fixture inserts one company (`Test Agri Co`) with two periods (Q3 and Q4 2024) and two fact rows with known values. After the test it cleans up via `DELETE FROM dim_company WHERE id = $1` — the `CASCADE` constraint removes related facts automatically. This keeps the DB clean between test runs without needing a full schema reset.

---

## Running Phase 5 Tests

```bash
# After first run: migration must be applied first
python -m uv run python -c "import asyncio; from src.db.migrate import run_migrations; asyncio.run(run_migrations())"

# Phase 5 only
python -m uv run pytest tests/test_structured.py -v

# Full suite (all phases)
python -m uv run pytest -v
```

Expected result: **79 passed, 1 skipped**.

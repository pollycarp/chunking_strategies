"""
Phase 5 Tests: Structured Data Layer

Test categories:
- schema tests       : NUMERIC types, FK constraints, view exists
- metrics tests      : named metric queries return correct values
- query engine tests : write detection, row limit, timeout enforcement
- introspector tests : schema description contains expected tables/columns
"""

import pytest

from src.structured.metrics import get_metric, list_metrics, metric_query
from src.structured.query_engine import QueryEngine, UnsafeSQLError
from src.structured.schema_introspector import SchemaIntrospector


# ---------------------------------------------------------------------------
# Fixture: seed financial data for metric tests
# ---------------------------------------------------------------------------

@pytest.fixture
async def financial_fixture(db_pool):
    """
    Inserts one company, two periods, and two fact rows.
    Cleans up after the test so data doesn't accumulate across runs.

    Company : "Test Agri Co"
    Periods : Q3 2024, Q4 2024
    Facts   :
        Q3 2024 — revenue=100_000, gross_profit=40_000, ebitda=25_000,
                  ebit=20_000, net_income=12_000, total_debt=50_000, cash=15_000
        Q4 2024 — revenue=120_000, gross_profit=50_000, ebitda=32_000,
                  ebit=26_000, net_income=16_000, total_debt=48_000, cash=20_000
    """
    async with db_pool.acquire() as conn:
        company_id = await conn.fetchval(
            """
            INSERT INTO dim_company (name, sector, country)
            VALUES ('Test Agri Co', 'Agriculture', 'ZA')
            RETURNING id
            """
        )

        period_q3 = await conn.fetchval(
            """
            INSERT INTO dim_period (year, quarter, period_label, start_date, end_date)
            VALUES (2024, 3, 'Q3 2024', '2024-07-01', '2024-09-30')
            RETURNING id
            """
        )
        period_q4 = await conn.fetchval(
            """
            INSERT INTO dim_period (year, quarter, period_label, start_date, end_date)
            VALUES (2024, 4, 'Q4 2024', '2024-10-01', '2024-12-31')
            RETURNING id
            """
        )

        await conn.execute(
            """
            INSERT INTO fact_financials
                (company_id, period_id, revenue, gross_profit, ebitda, ebit,
                 net_income, total_assets, total_debt, cash, source_file)
            VALUES ($1, $2, 100000, 40000, 25000, 20000, 12000, 200000, 50000, 15000, 'q3_report.pdf')
            """,
            company_id, period_q3,
        )
        await conn.execute(
            """
            INSERT INTO fact_financials
                (company_id, period_id, revenue, gross_profit, ebitda, ebit,
                 net_income, total_assets, total_debt, cash, source_file)
            VALUES ($1, $2, 120000, 50000, 32000, 26000, 16000, 210000, 48000, 20000, 'q4_report.pdf')
            """,
            company_id, period_q4,
        )

    yield {"company": "Test Agri Co", "company_id": company_id}

    # Cleanup — CASCADE from dim_company deletes related facts
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM dim_company WHERE id = $1", company_id)
        await conn.execute("DELETE FROM dim_period WHERE id = ANY($1::int[])", [period_q3, period_q4])


# ---------------------------------------------------------------------------
# Test 1: Schema
# ---------------------------------------------------------------------------

async def test_fact_financials_monetary_columns_are_numeric(db_pool):
    """
    NUMERIC type must be used for all monetary columns — never FLOAT.
    Floating-point arithmetic loses precision on financial calculations.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name, data_type
            FROM   information_schema.columns
            WHERE  table_name = 'fact_financials'
              AND  column_name IN (
                  'revenue','gross_profit','ebitda','ebit',
                  'net_income','total_assets','total_debt','cash'
              )
            """
        )

    assert len(rows) == 8, "All 8 monetary columns should exist"
    for row in rows:
        assert row["data_type"] == "numeric", (
            f"Column '{row['column_name']}' should be NUMERIC, got {row['data_type']}"
        )


async def test_foreign_key_constraints_exist(db_pool):
    """
    fact_financials must have FK constraints to dim_company and dim_period.
    Without FK constraints, orphan rows can accumulate silently.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT tc.constraint_name, kcu.column_name, ccu.table_name AS referenced_table
            FROM   information_schema.table_constraints       tc
            JOIN   information_schema.key_column_usage        kcu
                   ON kcu.constraint_name = tc.constraint_name
            JOIN   information_schema.constraint_column_usage ccu
                   ON ccu.constraint_name = tc.constraint_name
            WHERE  tc.constraint_type = 'FOREIGN KEY'
              AND  tc.table_name      = 'fact_financials'
            """
        )

    referenced = {row["referenced_table"] for row in rows}
    assert "dim_company" in referenced, "FK to dim_company must exist"
    assert "dim_period"  in referenced, "FK to dim_period must exist"


async def test_financial_metrics_view_exists(db_pool):
    """The financial_metrics view should be queryable."""
    async with db_pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM financial_metrics")
    assert count is not None  # just verify it runs without error


async def test_dim_company_name_is_unique(db_pool):
    """dim_company.name has a UNIQUE constraint — no duplicate company names."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT constraint_name
            FROM   information_schema.table_constraints
            WHERE  table_name       = 'dim_company'
              AND  constraint_type  = 'UNIQUE'
            """
        )
    assert len(rows) >= 1, "dim_company.name should have a UNIQUE constraint"


# ---------------------------------------------------------------------------
# Test 2: Metrics
# ---------------------------------------------------------------------------

def test_get_metric_returns_correct_metric():
    """get_metric() returns the right Metric object for a known name."""
    m = get_metric("ebitda_margin_pct")
    assert m.name == "ebitda_margin_pct"
    assert m.unit == "percent"
    assert m.column == "ebitda_margin_pct"


def test_get_metric_unknown_raises():
    """get_metric() raises KeyError for an unrecognised metric name."""
    with pytest.raises(KeyError, match="Unknown metric"):
        get_metric("made_up_metric")


def test_list_metrics_returns_all():
    """list_metrics() should return all 12 defined metrics."""
    metrics = list_metrics()
    names = {m["name"] for m in metrics}
    assert "revenue" in names
    assert "ebitda_margin_pct" in names
    assert "debt_to_ebitda" in names
    assert len(metrics) == 12


def test_metric_query_builds_correct_sql():
    """metric_query() should embed the column name and correct placeholders."""
    sql, params = metric_query("ebitda_margin_pct", company="Agri Co", period_label="Q3 2024")

    assert "ebitda_margin_pct" in sql
    assert "$1" in sql
    assert "$2" in sql
    assert params == ["Agri Co", "Q3 2024"]


def test_metric_query_no_filters():
    """metric_query() with no filters should have no WHERE clause."""
    sql, params = metric_query("revenue")

    assert "WHERE" not in sql
    assert params == []


async def test_metric_query_returns_correct_revenue(db_pool, financial_fixture):
    """
    Revenue for Q3 2024 should equal the inserted value (100,000).
    Tests that metric_query produces correct SQL against real data.
    """
    sql, params = metric_query(
        "revenue",
        company=financial_fixture["company"],
        period_label="Q3 2024",
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    assert len(rows) == 1
    assert float(rows[0]["value"]) == 100_000.0


async def test_metric_query_computes_ebitda_margin(db_pool, financial_fixture):
    """
    EBITDA margin for Q3 2024 = ebitda / revenue * 100 = 25000/100000*100 = 25.00%.
    Verifies the view correctly computes the derived ratio.
    """
    sql, params = metric_query(
        "ebitda_margin_pct",
        company=financial_fixture["company"],
        period_label="Q3 2024",
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    assert len(rows) == 1
    assert float(rows[0]["value"]) == 25.00


async def test_metric_query_computes_debt_to_ebitda(db_pool, financial_fixture):
    """
    Debt/EBITDA for Q3 2024 = 50000/25000 = 2.00×.
    Verifies the leverage ratio computation.
    """
    sql, params = metric_query(
        "debt_to_ebitda",
        company=financial_fixture["company"],
        period_label="Q3 2024",
    )
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    assert len(rows) == 1
    assert float(rows[0]["value"]) == 2.00


async def test_metric_query_returns_multiple_periods(db_pool, financial_fixture):
    """
    Without a period filter, the query returns one row per period (Q3 + Q4).
    Results should be ordered chronologically (year, quarter).
    """
    sql, params = metric_query("revenue", company=financial_fixture["company"])
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    assert len(rows) == 2
    # Q3 should come before Q4 (ordered by year, quarter)
    assert rows[0]["period_label"] == "Q3 2024"
    assert rows[1]["period_label"] == "Q4 2024"


# ---------------------------------------------------------------------------
# Test 3: QueryEngine safety
# ---------------------------------------------------------------------------

async def test_query_engine_executes_select(db_pool):
    """A valid SELECT should return results without raising."""
    engine = QueryEngine(db_pool)
    rows = await engine.execute("SELECT 1 AS n")
    assert rows == [{"n": 1}]


async def test_query_engine_rejects_insert(db_pool):
    """INSERT must raise UnsafeSQLError before reaching the DB."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError, match="INSERT"):
        await engine.execute("INSERT INTO dim_company (name) VALUES ('hack')")


async def test_query_engine_rejects_update(db_pool):
    """UPDATE must be blocked."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError, match="UPDATE"):
        await engine.execute("UPDATE fact_financials SET revenue = 0")


async def test_query_engine_rejects_delete(db_pool):
    """DELETE must be blocked."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError, match="DELETE"):
        await engine.execute("DELETE FROM dim_company WHERE id = 1")


async def test_query_engine_rejects_drop(db_pool):
    """DROP TABLE must be blocked."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError, match="DROP"):
        await engine.execute("DROP TABLE fact_financials")


async def test_query_engine_rejects_truncate(db_pool):
    """TRUNCATE must be blocked."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError, match="TRUNCATE"):
        await engine.execute("TRUNCATE TABLE fact_financials")


async def test_query_engine_applies_row_limit(db_pool):
    """
    When max_rows=2, the engine appends LIMIT 2 to unbounded queries.
    Generates 10 rows with generate_series, expects only 2 back.
    """
    engine = QueryEngine(db_pool, max_rows=2)
    rows = await engine.execute("SELECT generate_series(1, 10) AS n")
    assert len(rows) == 2


async def test_query_engine_respects_existing_limit(db_pool):
    """
    A query that already has LIMIT should not have another LIMIT appended.
    """
    engine = QueryEngine(db_pool, max_rows=100)
    rows = await engine.execute("SELECT generate_series(1, 10) AS n LIMIT 3")
    assert len(rows) == 3


async def test_query_engine_case_insensitive_write_detection(db_pool):
    """Write detection must work regardless of keyword casing."""
    engine = QueryEngine(db_pool)
    with pytest.raises(UnsafeSQLError):
        await engine.execute("insert into dim_company (name) values ('x')")
    with pytest.raises(UnsafeSQLError):
        await engine.execute("Insert Into dim_company (name) VALUES ('x')")


# ---------------------------------------------------------------------------
# Test 4: SchemaIntrospector
# ---------------------------------------------------------------------------

async def test_introspector_lists_financial_tables(db_pool):
    """list_tables() must include all four queryable objects."""
    introspector = SchemaIntrospector(db_pool)
    tables = await introspector.list_tables()

    assert "dim_company"       in tables
    assert "dim_period"        in tables
    assert "fact_financials"   in tables
    assert "financial_metrics" in tables


async def test_introspector_description_contains_table_names(db_pool):
    """get_schema_description() must mention all four tables/views by name."""
    introspector = SchemaIntrospector(db_pool)
    description = await introspector.get_schema_description()

    assert "dim_company"       in description
    assert "dim_period"        in description
    assert "fact_financials"   in description
    assert "financial_metrics" in description


async def test_introspector_description_contains_key_columns(db_pool):
    """Schema description must name key columns the LLM will query on."""
    introspector = SchemaIntrospector(db_pool)
    description = await introspector.get_schema_description()

    assert "revenue"        in description
    assert "ebitda"         in description
    assert "period_label"   in description
    assert "company"        in description


async def test_introspector_description_contains_metric_names(db_pool):
    """Schema description must include the named metric list."""
    introspector = SchemaIntrospector(db_pool)
    description = await introspector.get_schema_description()

    assert "ebitda_margin_pct" in description
    assert "debt_to_ebitda"    in description
    assert "gross_margin_pct"  in description


async def test_introspector_description_contains_query_rules(db_pool):
    """Schema description must include the read-only query rules."""
    introspector = SchemaIntrospector(db_pool)
    description = await introspector.get_schema_description()

    assert "SELECT only" in description

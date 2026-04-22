"""
Semantic metric layer — named financial metrics that map to the financial_metrics view.

Why a semantic layer?
Without it, an LLM must re-derive metric formulas on every query:
  "EBITDA margin = ebitda / revenue * 100"  → prone to hallucination.
With it, the LLM queries by name:
  SELECT ebitda_margin_pct FROM financial_metrics WHERE ...
  → formula is always correct because it lives in the view definition.

This module defines the registry of available metrics and builds
parameterised queries against the financial_metrics view.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    """
    Describes one named financial metric.

    name        : machine-readable key used in queries
    description : human/LLM-readable explanation of what the metric means
    column      : the column name in the financial_metrics view
    unit        : "currency" | "percent" | "ratio" (helps LLM format output)
    """
    name: str
    description: str
    column: str
    unit: str


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRICS: dict[str, Metric] = {
    "revenue": Metric(
        name="revenue",
        description="Total revenue (top line) for the period",
        column="revenue",
        unit="currency",
    ),
    "gross_profit": Metric(
        name="gross_profit",
        description="Revenue minus cost of goods sold",
        column="gross_profit",
        unit="currency",
    ),
    "ebitda": Metric(
        name="ebitda",
        description=(
            "Earnings Before Interest, Taxes, Depreciation and Amortisation — "
            "a proxy for operating cash flow"
        ),
        column="ebitda",
        unit="currency",
    ),
    "ebit": Metric(
        name="ebit",
        description="Earnings Before Interest and Taxes (operating profit)",
        column="ebit",
        unit="currency",
    ),
    "net_income": Metric(
        name="net_income",
        description="Net profit after all expenses, interest, and taxes",
        column="net_income",
        unit="currency",
    ),
    "total_assets": Metric(
        name="total_assets",
        description="Sum of all assets on the balance sheet",
        column="total_assets",
        unit="currency",
    ),
    "total_debt": Metric(
        name="total_debt",
        description="Total interest-bearing debt (short + long term)",
        column="total_debt",
        unit="currency",
    ),
    "cash": Metric(
        name="cash",
        description="Cash and cash equivalents",
        column="cash",
        unit="currency",
    ),
    "gross_margin_pct": Metric(
        name="gross_margin_pct",
        description="Gross profit as a percentage of revenue",
        column="gross_margin_pct",
        unit="percent",
    ),
    "ebitda_margin_pct": Metric(
        name="ebitda_margin_pct",
        description="EBITDA as a percentage of revenue",
        column="ebitda_margin_pct",
        unit="percent",
    ),
    "net_margin_pct": Metric(
        name="net_margin_pct",
        description="Net income as a percentage of revenue",
        column="net_margin_pct",
        unit="percent",
    ),
    "debt_to_ebitda": Metric(
        name="debt_to_ebitda",
        description=(
            "Total debt divided by EBITDA — measures financial leverage; "
            "lower is safer; >4x is typically considered high leverage"
        ),
        column="debt_to_ebitda",
        unit="ratio",
    ),
}


def get_metric(name: str) -> Metric:
    """
    Returns the Metric for the given name.

    Raises KeyError with a helpful message if the metric is unknown.
    This is the function the query engine / LLM layer calls to validate
    a metric name before executing a query.
    """
    if name not in METRICS:
        available = ", ".join(sorted(METRICS))
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}")
    return METRICS[name]


def metric_query(
    metric_name: str,
    company: str | None = None,
    period_label: str | None = None,
) -> tuple[str, list]:
    """
    Builds a parameterised SELECT against financial_metrics for one named metric.

    Optionally filters by company name (exact match) and/or period_label
    (e.g. "Q3 2024", "FY 2024").

    Returns:
        (sql, params) — ready to pass to conn.fetch(sql, *params)

    Example:
        sql, params = metric_query("ebitda_margin_pct", company="Agri Co", period_label="Q3 2024")
        rows = await conn.fetch(sql, *params)
    """
    metric = get_metric(metric_name)

    conditions: list[str] = []
    params: list = []
    i = 1

    if company is not None:
        conditions.append(f"company = ${i}")
        params.append(company)
        i += 1

    if period_label is not None:
        conditions.append(f"period_label = ${i}")
        params.append(period_label)
        i += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT company, period_label, year, quarter, {metric.column} AS value
        FROM   financial_metrics
        {where}
        ORDER  BY year, quarter NULLS LAST
    """

    return sql.strip(), params


def list_metrics() -> list[dict]:
    """
    Returns all metrics as a list of dicts — used by the schema introspector
    to generate LLM-readable metric documentation.
    """
    return [
        {
            "name": m.name,
            "description": m.description,
            "unit": m.unit,
        }
        for m in METRICS.values()
    ]

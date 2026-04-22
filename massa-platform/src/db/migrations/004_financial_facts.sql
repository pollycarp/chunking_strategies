-- Migration 004: Financial facts star schema
-- Creates dimension tables, fact table, and a computed metrics view.
--
-- Star schema overview:
--   dim_company    — who    (portfolio company metadata)
--   dim_period     — when   (year/quarter definitions)
--   fact_financials — what  (numeric measurements, FK to both dims)
--   financial_metrics VIEW  — derived ratios computed from fact_financials
--
-- Why NUMERIC(18,2) for all monetary columns?
-- FLOAT uses binary fractions — 0.1 + 0.2 ≠ 0.3 in IEEE 754.
-- NUMERIC stores exact decimal values. For financial data this is non-negotiable:
-- a 1-cent rounding error in a single fact row can cascade into material
-- misstatements at aggregate level.

-- ─────────────────────────────────────────────────────────────────────────────
-- Dimension: company
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_company (
    id         SERIAL      PRIMARY KEY,
    name       TEXT        NOT NULL UNIQUE,
    sector     TEXT,
    country    TEXT        NOT NULL DEFAULT 'ZA',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Dimension: time period
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_period (
    id           SERIAL  PRIMARY KEY,
    year         INT     NOT NULL,
    quarter      INT     CHECK (quarter BETWEEN 1 AND 4),  -- NULL = full year
    period_label TEXT    NOT NULL UNIQUE,  -- human label: "Q3 2024" or "FY 2024"
    start_date   DATE    NOT NULL,
    end_date     DATE    NOT NULL
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Fact: financial measurements
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_financials (
    id           SERIAL       PRIMARY KEY,
    company_id   INT          NOT NULL REFERENCES dim_company(id) ON DELETE CASCADE,
    period_id    INT          NOT NULL REFERENCES dim_period(id)  ON DELETE CASCADE,

    -- Income statement
    revenue      NUMERIC(18,2),
    gross_profit NUMERIC(18,2),
    ebitda       NUMERIC(18,2),
    ebit         NUMERIC(18,2),
    net_income   NUMERIC(18,2),

    -- Balance sheet
    total_assets NUMERIC(18,2),
    total_debt   NUMERIC(18,2),
    cash         NUMERIC(18,2),

    -- Provenance
    source_file  TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- One measurement per company per period
    UNIQUE (company_id, period_id)
);

CREATE INDEX IF NOT EXISTS fact_financials_company_idx ON fact_financials(company_id);
CREATE INDEX IF NOT EXISTS fact_financials_period_idx  ON fact_financials(period_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- View: financial_metrics
--
-- Joins dims to facts and computes derived ratios.
-- This is the primary surface an LLM queries — it never needs to write
-- the JOIN itself; it just queries this view with WHERE filters.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW financial_metrics AS
SELECT
    f.id,
    c.name        AS company,
    c.sector,
    c.country,
    p.period_label,
    p.year,
    p.quarter,
    p.start_date,
    p.end_date,

    -- Raw measurements
    f.revenue,
    f.gross_profit,
    f.ebitda,
    f.ebit,
    f.net_income,
    f.total_assets,
    f.total_debt,
    f.cash,

    -- Derived ratios (NULL-safe: CASE avoids division-by-zero)
    CASE WHEN f.revenue  <> 0 THEN ROUND(f.gross_profit / f.revenue  * 100, 2) END AS gross_margin_pct,
    CASE WHEN f.revenue  <> 0 THEN ROUND(f.ebitda       / f.revenue  * 100, 2) END AS ebitda_margin_pct,
    CASE WHEN f.revenue  <> 0 THEN ROUND(f.net_income   / f.revenue  * 100, 2) END AS net_margin_pct,
    CASE WHEN f.ebitda   <> 0 THEN ROUND(f.total_debt   / f.ebitda,          2) END AS debt_to_ebitda,

    f.source_file
FROM  fact_financials f
JOIN  dim_company     c ON c.id = f.company_id
JOIN  dim_period      p ON p.id = f.period_id;

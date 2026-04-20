-- Migration 001: Initial setup
-- Enables pgvector extension and creates migration tracking table.

-- pgvector adds the `vector` column type and ANN index support to Postgres.
-- Must be enabled before any table uses a vector column.
CREATE EXTENSION IF NOT EXISTS vector;

-- Tracks which migrations have been applied.
-- The migration runner inserts a row here after each successful migration.
-- This prevents running the same migration twice.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT        PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

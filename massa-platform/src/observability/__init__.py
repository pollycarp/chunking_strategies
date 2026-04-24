"""
Observability & Data Quality — Phase 9.

Three complementary concerns:

    logger.py        — records every agent interaction to the DB so quality
                       can be trended over time (latency, token usage, scores)

    data_quality.py  — inspects the chunks and documents tables for structural
                       problems: missing embeddings, empty content, duplicates

    health.py        — aggregates both sources into a single SystemHealthReport
                       suitable for a dashboard or CI health check
"""

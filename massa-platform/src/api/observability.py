"""
GET /health  — returns the full SystemHealthReport as JSON.
GET /logs    — returns the most recent agent interactions as JSON.

These endpoints power the Streamlit dashboard's health and activity tabs.
They are read-only and safe to call frequently.
"""

from __future__ import annotations

import json

from starlette.requests import Request
from starlette.responses import JSONResponse

from src.observability.health import HealthReporter, to_json
from src.observability.logger import AgentLogger


async def health_endpoint(request: Request) -> JSONResponse:
    """
    GET /health

    Runs DataQualityChecker + AgentLogger.stats() and returns the aggregated
    SystemHealthReport as JSON. The report includes both data-layer health
    (embedding coverage, duplicates, empty chunks) and runtime health
    (avg latency, faithfulness score, hallucination rate).
    """
    reporter: HealthReporter = request.app.state.health_reporter
    report = await reporter.report()
    # to_json() returns a JSON string — parse back to dict for JSONResponse
    return JSONResponse(json.loads(to_json(report)))


async def logs_endpoint(request: Request) -> JSONResponse:
    """
    GET /logs?limit=N  (default 50)

    Returns the N most recent agent interactions, newest first.
    Timestamps are serialised as ISO strings for JSON compatibility.
    """
    try:
        limit = int(request.query_params.get("limit", 50))
        limit = max(1, min(limit, 200))   # clamp to [1, 200]
    except ValueError:
        limit = 50

    agent_logger: AgentLogger = request.app.state.agent_logger
    rows = await agent_logger.recent(limit=limit)

    # Convert datetime objects to ISO strings (not JSON-serialisable by default)
    serialisable = []
    for row in rows:
        d = dict(row)
        if d.get("logged_at") is not None:
            d["logged_at"] = d["logged_at"].isoformat()
        serialisable.append(d)

    return JSONResponse(serialisable)

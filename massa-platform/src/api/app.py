"""
Full MASSA Starlette application — all routes wired together.

Routes
------
POST /chat    → FinancialAgent agentic loop
POST /ingest  → IngestionPipeline (parse → chunk → embed → store)
GET  /health  → SystemHealthReport (data quality + agent stats)
GET  /logs    → Recent agent interaction log

Startup / Shutdown
------------------
Uses Starlette's lifespan context manager to:
1. Create the DB connection pool (shared by all routes)
2. Instantiate every component exactly once (embedder, MCP server, agent,
   pipeline, logger, health reporter)
3. Attach everything to app.state so endpoints can access it via request.app.state
4. Close the pool cleanly on shutdown

Run with:
    uvicorn src.api.app:app --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route

from src.api.chat import chat_endpoint
from src.api.ingest import ingest_endpoint
from src.api.observability import health_endpoint, logs_endpoint
from src.config import settings
from src.db.connection import close_pool, create_pool
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.ingestion.pipeline import IngestionPipeline
from src.llm.agent import FinancialAgent
from src.llm.client import ClaudeClient
from src.mcp.server import create_server
from src.observability.health import HealthReporter
from src.observability.logger import AgentLogger


@asynccontextmanager
async def lifespan(app: Starlette):
    """
    Runs once at startup (before the yield) and once at shutdown (after).

    Why lifespan instead of on_startup/on_shutdown events?
    Lifespan is the modern Starlette/ASGI standard. It also ensures cleanup
    always runs even if startup raises partway through.
    """
    # ── Database ────────────────────────────────────────────────────────────
    pool = await create_pool()

    # ── Embeddings ──────────────────────────────────────────────────────────
    embedder = OpenAIEmbedder(api_key=settings.openai_api_key)

    # ── MCP server + Agent ──────────────────────────────────────────────────
    server = create_server(pool=pool, embedder=embedder)
    claude_client = ClaudeClient(api_key=settings.anthropic_api_key)
    agent = FinancialAgent(client=claude_client, server=server)

    # ── Ingestion ───────────────────────────────────────────────────────────
    pipeline = IngestionPipeline(pool=pool, embedder=embedder)

    # ── Observability ───────────────────────────────────────────────────────
    agent_logger = AgentLogger(pool=pool)
    health_reporter = HealthReporter(pool=pool)

    # ── Attach to app.state (accessible in every endpoint via request.app.state)
    app.state.agent = agent
    app.state.pipeline = pipeline
    app.state.agent_logger = agent_logger
    app.state.health_reporter = health_reporter

    print("MASSA platform started.")
    print(f"  DB:      {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    print(f"  Env:     {settings.env}")

    yield  # ← server is live here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    await close_pool()
    print("MASSA platform stopped.")


app = Starlette(
    routes=[
        Route("/chat",   chat_endpoint,   methods=["POST"]),
        Route("/ingest", ingest_endpoint, methods=["POST"]),
        Route("/health", health_endpoint, methods=["GET"]),
        Route("/logs",   logs_endpoint,   methods=["GET"]),
    ],
    lifespan=lifespan,
)

# Allow the Streamlit frontend (running on port 8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

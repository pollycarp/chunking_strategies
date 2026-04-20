# Financial Intelligence Platform

A production-grade AI platform that lets an LLM answer questions over both documents and structured financial data. Built to demonstrate the core competencies required for a Data & AI Engineer role at MASSA Advisors.

---

## What It Does

- Ingests PDFs, Word docs, and spreadsheets (financials, reports, memos)
- Ingests structured data (financial metrics, portfolio data)
- Answers natural-language questions by retrieving the right context
- Exposes tools to an LLM via MCP (Model Context Protocol)
- Evaluates answer quality with a benchmark harness

---

## Stack

Python (FastMCP, Starlette/ASGI, asyncpg), PostgreSQL + pgvector, embedding models, vector search, Docker, MCP (Model Context Protocol), Claude API

---

## Phased Build Plan

---

### Phase 1: Project Foundation & Infrastructure

**Goal:** Set up the development environment, database, and project skeleton that every other phase builds on.

**Key Concepts:**
- Docker & Docker Compose — why we containerize (reproducibility, isolation)
- PostgreSQL fundamentals — roles, schemas, connection pooling
- `asyncpg` — why async database access matters in a pipeline that does I/O constantly
- `pgvector` — a Postgres extension that adds a vector column type and ANN (Approximate Nearest Neighbor) index
- Project structure for a production Python service (not a notebook)

**Project Structure:**
```
massa-platform/
├── docker-compose.yml         # Postgres + pgvector container
├── .env                       # Secrets (never committed)
├── pyproject.toml             # Dependencies (uv or poetry)
├── src/
│   ├── db/
│   │   ├── connection.py      # asyncpg pool factory
│   │   ├── migrations/        # SQL migration files
│   │   │   └── 001_init.sql
│   └── config.py              # Pydantic settings from .env
└── tests/
    └── test_db_connection.py
```

**Tests:**
- DB connection test — assert pool connects and returns a valid connection
- Schema smoke test — assert all expected tables exist after migration runs
- Config test — assert missing env vars raise a clear error, not a cryptic one

---

### Phase 2: Embeddings & Vector Storage

**Goal:** Understand what embeddings are, generate them, store them in pgvector, and run your first similarity searches.

**Key Concepts:**
- What an embedding is — a dense vector that encodes semantic meaning; why "king - man + woman ≈ queen" works
- Embedding model landscape: OpenAI `text-embedding-3-small`, Cohere `embed-v3`, Voyage `voyage-2`, open-source (`bge-m3`, `nomic-embed`)
- Trade-offs: cost, dimensionality, multilingual support, domain fit
- pgvector index types — `ivfflat` vs `hnsw` and when each makes sense
- Cosine similarity vs L2 distance vs inner product — which to use and why

**Project Structure:**
```
src/
├── embeddings/
│   ├── base.py               # Abstract EmbeddingModel interface
│   ├── openai_embedder.py    # OpenAI implementation
│   ├── voyage_embedder.py    # Voyage AI implementation
│   └── cache.py              # DB-backed embedding cache (avoid re-embedding)
├── db/
│   └── migrations/
│       └── 002_vectors.sql   # Add vectors table + hnsw index
```

**Tests:**
- Embedder interface test — both implementations return same shape vector for same input
- Cache test — second call for same text hits DB, not API (mock the API)
- Similarity sanity test — "revenue" is closer to "income" than to "cat"
- Index existence test — assert hnsw index is present on the vectors table

---

### Phase 3: Document Ingestion & Chunking

**Goal:** Build the pipeline that takes raw files (PDFs, DOCX, XLSX) and turns them into clean, embedded, retrievable chunks.

**Key Concepts:**
- Why chunking matters — LLMs have context limits; you can't feed a 200-page report verbatim
- Fixed-size chunking — simple, predictable, but breaks sentences mid-thought
- Semantic chunking — split on meaning boundaries (sentences, paragraphs); better context preservation
- Hierarchical chunking — store both a summary chunk and its child detail chunks; lets you retrieve at the right granularity
- Metadata — why every chunk must carry: `source_file`, `page_number`, `section_title`, `created_at`, `doc_type`
- Document parsers: `pdfplumber` for PDFs, `python-docx` for Word, `openpyxl` for Excel

**Project Structure:**
```
src/
├── ingestion/
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── xlsx_parser.py
│   ├── chunkers/
│   │   ├── fixed_size.py
│   │   ├── semantic.py
│   │   └── hierarchical.py
│   ├── pipeline.py           # Orchestrates: parse → chunk → embed → store
│   └── deduplication.py      # Hash-based dedup; don't re-ingest unchanged files
```

**Tests:**
- Parser tests — each parser extracts known text from a fixture file
- Chunker tests — fixed: assert no chunk exceeds max tokens; semantic: assert no chunk splits mid-sentence
- Metadata test — every chunk produced has all required metadata fields populated
- Deduplication test — ingesting the same file twice does not create duplicate rows
- End-to-end pipeline test — one document in → chunks with embeddings in DB

---

### Phase 4: Hybrid Retrieval & Re-ranking

**Goal:** Build a retrieval system that combines semantic search with keyword search and re-ranking — the core of a production RAG system.

**Key Concepts:**
- Semantic search — embedding similarity; good at meaning, bad at exact terms ("Q3 2024")
- Keyword search (BM25/full-text) — Postgres `tsvector`; exact term matching; good at named entities, numbers, codes
- Hybrid retrieval — why neither alone is enough; Reciprocal Rank Fusion (RRF) as a simple, effective fusion method
- Re-ranking — a second-pass cross-encoder model (Cohere Rerank, `cross-encoder/ms-marco`) that scores query-chunk relevance more precisely but is too slow to run over the whole index
- Metadata filtering — pre-filter by `doc_type`, `date_range`, `client_id` before searching; reduces noise dramatically
- Retrieval metrics: Recall@K, Precision@K, MRR (Mean Reciprocal Rank) — how to measure if you're finding the right chunks

**Project Structure:**
```
src/
├── retrieval/
│   ├── semantic.py           # pgvector ANN search
│   ├── keyword.py            # Postgres full-text search
│   ├── hybrid.py             # RRF fusion of both
│   ├── reranker.py           # Cross-encoder re-ranking pass
│   └── filters.py            # Metadata filter builder
```

**Tests:**
- Semantic retrieval test — known query returns known relevant chunk in top-3
- Keyword retrieval test — exact term "EBITDA margin Q3" is found even if semantically distant
- Hybrid beats both test — on a small fixture dataset, hybrid recall@5 >= max(semantic, keyword) alone
- Re-ranker test — re-ranked list places the gold chunk higher than pre-rerank order
- Filter test — results with `doc_type=financial_report` filter contain no chunks from other types

---

### Phase 5: Structured Data Layer

**Goal:** Build the SQL schema and query interface that lets an LLM access quantitative financial data precisely.

**Key Concepts:**
- Dimensional modeling — star schema vs snowflake; fact tables (what happened) vs dimension tables (context: who, when, where)
- Why this matters for LLMs — a well-designed schema lets an LLM write correct SQL on the first try; a poorly designed one leads to hallucinated joins
- Semantic layer — metric definitions in code/SQL (`revenue`, `gross_margin`, `portfolio_irr`) so the LLM queries named metrics, not raw columns
- Precision in financial data — using `NUMERIC` not `FLOAT` in Postgres; why floats lose money
- Query safety — read-only DB user, query timeouts, result size limits; the LLM must never be able to mutate data

**Project Structure:**
```
src/
├── structured/
│   ├── schema/
│   │   └── 003_financial_facts.sql   # fact_financials, dim_company, dim_period
│   ├── metrics.py                    # Named metric definitions (revenue, EBITDA...)
│   ├── query_engine.py               # Safe SQL executor with timeout + row limit
│   └── schema_introspector.py        # Generates LLM-readable schema description
```

**Tests:**
- Schema test — foreign keys are valid; `NUMERIC` used for all monetary columns
- Metric test — `revenue` metric query returns correct value against fixture data
- Safety test — INSERT/UPDATE/DELETE statements raise an error before reaching DB
- Introspector test — generated schema description contains all table and column names

---

### Phase 6: MCP Server

**Goal:** Expose both the retrieval pipeline and the SQL query engine as tools an LLM can call via the Model Context Protocol.

**Key Concepts:**
- MCP (Model Context Protocol) — an open standard (by Anthropic) for connecting LLMs to external tools and data sources; like a USB-C standard for AI tools
- Tool design — how you write a tool description is prompt engineering; a badly described tool gets misused by the LLM
- Parameter schemas — JSON Schema for inputs; how to constrain what the LLM can pass in
- FastMCP — Python framework for building MCP servers quickly; sits on top of Starlette/ASGI
- Separation of concerns — the MCP layer is thin; it translates LLM calls into calls on your retrieval/SQL engines; business logic stays in the engine, not the tool handler

**Project Structure:**
```
src/
├── mcp/
│   ├── server.py             # FastMCP app, tool registration
│   ├── tools/
│   │   ├── retrieve_docs.py  # Tool: search document chunks
│   │   ├── query_metrics.py  # Tool: run named metric queries
│   │   └── list_sources.py   # Tool: tell LLM what data exists
│   └── output_formatters.py  # Shape tool outputs for LLM consumption
```

**Tests:**
- Tool registration test — all tools are discoverable via MCP tool-list endpoint
- Schema validation test — calling a tool with missing required params returns a schema error
- Retrieve docs tool test — returns correctly formatted chunks for a known query
- Query metrics tool test — returns correct metric value for fixture data
- Read-only enforcement test — query tool rejects mutation attempts

---

### Phase 7: LLM Integration

**Goal:** Wire the MCP server to Claude, shape the system prompt for the financial domain, and build the full conversational loop.

**Key Concepts:**
- Claude API — Messages API, tool use / function calling flow (how Claude decides when to call a tool, and how you return the result)
- System prompt engineering — for financial AI, precision and citation of sources matters more than creativity; how to encode that in a system prompt
- Context window management — you can't fit everything; how to prioritize what context goes in
- Multi-turn conversations — maintaining chat history without blowing the context limit
- Agentic loop — the pattern of: LLM decides → calls tool → gets result → decides again; when to stop

**Project Structure:**
```
src/
├── llm/
│   ├── client.py             # Claude API wrapper
│   ├── agent.py              # Agentic loop: thinks, calls tools, answers
│   ├── system_prompt.py      # Domain-specific system prompt builder
│   └── context_manager.py   # Trims history to fit context window
├── api/
│   └── chat.py               # Starlette HTTP endpoint: POST /chat
```

**Tests:**
- Agent tool-call test — given a question about financials, agent calls `query_metrics` tool (mock Claude, assert tool was invoked)
- Agent RAG test — given a question about a document, agent calls `retrieve_docs`
- Context manager test — history beyond token limit is trimmed from oldest first
- End-to-end integration test — full question → tools → answer flow against fixture data (uses real Claude, marked slow)

---

### Phase 8: Evaluation Harness

**Goal:** Build a rigorous benchmark to measure whether your system is actually getting better or worse as you change things.

**Key Concepts:**
- Why vibes-based evaluation fails — "it looks right" doesn't catch regressions; you need numbers
- Retrieval metrics — Recall@K (did the right chunk appear in the top K?), Precision@K, MRR (how high was the first correct chunk ranked?)
- Answer fidelity / faithfulness — is the answer grounded in the retrieved context? (LLM-as-judge pattern)
- Hallucination detection — checking if the answer makes claims not supported by retrieved chunks
- Benchmark dataset design — 200+ question/answer pairs with known correct sources; how to build one without labeling everything manually

**Project Structure:**
```
src/
├── eval/
│   ├── benchmark.py          # Load & run the Q&A benchmark dataset
│   ├── retrieval_metrics.py  # Recall@K, Precision@K, MRR
│   ├── answer_metrics.py     # Faithfulness, correctness (LLM-as-judge)
│   ├── hallucination.py      # Claim extraction + source grounding check
│   └── reporter.py           # Outputs results as JSON + markdown table
└── data/
    └── benchmark_qa.json     # 50-question fixture dataset (you'll build this)
```

**Tests:**
- Metrics math test — Recall@5 = 1.0 when gold doc is in top 5, 0.0 when not
- MRR test — known ranked list produces correct MRR value
- Faithfulness judge test — answer containing info not in context is flagged
- Regression gate test — if recall@5 drops below a threshold, the test suite fails (CI guard)

---

### Phase 9: Observability & Data Quality

**Goal:** Make the system trustworthy in production — log everything, detect bad data, alert on quality degradation.

**Key Concepts:**
- Structured logging — why JSON logs are better than print statements in production
- Query audit trail — every LLM query, every tool call, every retrieved chunk gets logged with a `trace_id`
- Data quality checks — null rates, value range checks, staleness detection (when was this record last updated?)
- Deduplication & reconciliation — when the same financial metric comes from two sources with different values, you need a rule for which wins

**Project Structure:**
```
src/
├── observability/
│   ├── logger.py             # Structured JSON logger with trace_id
│   ├── query_tracker.py      # Logs every tool call + result
│   └── audit_log.py          # Append-only audit table in Postgres
├── quality/
│   ├── validators.py         # Per-table data quality rules
│   ├── staleness.py          # Flags records not updated in N days
│   └── reconciler.py         # Conflict resolution across sources
```

**Tests:**
- Logger test — every log line is valid JSON and contains `trace_id`
- Audit log test — tool calls are persisted and queryable after a request
- Staleness test — records older than threshold are correctly flagged
- Validator test — out-of-range financial values raise a quality warning

---

## Phase Summary

| Phase | Builds | Core Concept |
|-------|--------|--------------|
| 1 | Infrastructure | Async Postgres, Docker, pgvector |
| 2 | Embeddings | Semantic vectors, ANN index |
| 3 | Ingestion | Chunking strategies, metadata |
| 4 | Retrieval | Hybrid search, RRF, re-ranking |
| 5 | Structured Data | Star schema, safe SQL for LLMs |
| 6 | MCP Server | Tool design, FastMCP |
| 7 | LLM Integration | Agentic loop, Claude API |
| 8 | Evaluation | Retrieval metrics, hallucination |
| 9 | Observability | Audit logs, data quality |

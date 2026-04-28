# Vector Database Selection Guide

## What MASSA Uses and Why

MASSA is built on **pgvector** — the vector extension for PostgreSQL. This is the right choice for financial intelligence platforms of this scale. The reasons are covered below alongside guidance for when to consider dedicated vector databases.

---

## pgvector (What We Built)

**Best for:** datasets under ~5–10 million vectors where SQL joins matter.

| Strength | Why it matters for MASSA |
|---|---|
| Vectors and relational data in one DB | Chunks, embeddings, and financial metrics (star schema) live in the same Postgres instance — no cross-system joins needed |
| ACID transactions | Ingestion is atomic: document record + all chunks commit together or not at all |
| SQL joins | Phase 5 queries JOIN vector retrieval results directly against `fact_financials` — impossible in a pure vector DB |
| Hybrid retrieval | pgvector provides the HNSW index for semantic search; Postgres GIN index handles BM25 keyword search — both in one query |
| Operational simplicity | One database to run, back up, and monitor instead of two |
| Cost | No additional service; runs in the same Docker container |

**When pgvector starts to strain:**
- Dataset grows beyond ~5–10 million chunks
- Query latency requirements fall below 10ms at high concurrency
- You need multi-region replication of the vector index specifically

---

## Pinecone

**Best for:** very large-scale deployments where operational simplicity and extreme query speed matter more than SQL integration.

| Situation | Why Pinecone |
|---|---|
| 100M+ vectors | Fully managed horizontal scaling — no infrastructure tuning required |
| Sub-10ms latency at scale | Purpose-built indexes outperform Postgres HNSW at extreme scale |
| No engineering bandwidth for ops | Fully managed SaaS — nothing to run, patch, or tune |
| Serverless billing model | Pay per query rather than per running instance |

**Tradeoffs:**
- No SQL. Metadata filtering is limited compared to WHERE clauses.
- Vectors are isolated from your relational data — JOINs must happen in application code after fetching results from both systems.
- Vendor lock-in: proprietary API, no self-hosted option.
- For MASSA's use case, you would lose the Phase 5 structured data integration without significant re-engineering.

---

## Weaviate

**Best for:** teams that need built-in hybrid search, multi-modal data, or want a self-hosted dedicated vector DB at scale.

| Situation | Why Weaviate |
|---|---|
| Built-in hybrid search | BM25 + vector search in a single native query — MASSA had to build this manually in Phase 4 |
| Multi-modal data | Text, images, and audio can be indexed together in one store |
| Self-hosted at scale | Open-source, runs on your own infrastructure, scales horizontally |
| Schema-defined objects | Stores structured objects with typed properties alongside vectors, closer to a document DB |
| GraphQL API | Rich query language for complex retrieval patterns |

**Tradeoffs:**
- More complex to operate than pgvector.
- Less natural fit with SQL-heavy workloads like MASSA's financial metrics layer.
- You would still need a separate Postgres instance for the star schema (Phase 5), meaning two databases to operate and keep in sync.

---

## Decision Guide

```
How many vectors do you expect?

  Under 5 million
  └── pgvector                    ← MASSA's choice; simplest, most integrated

  5–50 million
  └── Do you need SQL joins with structured data?
        YES → pgvector with read replicas + table partitioning
              or Weaviate + separate Postgres for structured data
        NO  → Pinecone (managed) or Weaviate (self-hosted)

  50M+
  └── Do you need SQL joins?
        YES → Weaviate + separate Postgres
        NO  → Pinecone
```

---

## Recommendation for MASSA

**Stay on pgvector** unless one of the following becomes true:

1. The chunk count exceeds ~5 million (unlikely for a single firm's document corpus)
2. A specific latency SLA requires sub-10ms retrieval at high concurrency
3. The platform expands to multi-modal data (images of charts, audio earnings calls)

If you ever migrate, the abstraction in `src/retrieval/` means only the `SemanticRetriever` and `KeywordRetriever` classes need to change — the rest of the platform (MCP tools, agent, evaluation) stays the same.

---

## Summary Table

| | pgvector | Pinecone | Weaviate |
|---|---|---|---|
| Max practical scale | ~10M vectors | 1B+ vectors | 100M+ vectors |
| SQL integration | Native | Manual (app code) | Partial (separate DB) |
| Hybrid search | Manual (Phase 4) | Basic | Native |
| Operational complexity | Low (one DB) | None (SaaS) | Medium (self-hosted) |
| Self-hosted | Yes | No | Yes |
| Cost model | Infrastructure | Per query | Infrastructure |
| Best fit for MASSA | Yes | No | Possible at scale |

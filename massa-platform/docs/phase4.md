# Phase 4: Hybrid Retrieval & Re-ranking

## What We Built

A complete retrieval pipeline that combines two fundamentally different search strategies — vector similarity and keyword matching — then re-ranks the fused results using a cross-encoder model.

```
Query
  │
  ├──► Embed (once)        ──► SemanticRetriever ──► top-20 chunks (by cosine)
  │                                                         │
  └──► Raw text            ──► KeywordRetriever  ──► top-20 chunks (by ts_rank)
                                                           │
                                              rrf_fusion (k=60)
                                                           │
                                              top-10 fused candidates
                                                           │
                                              CohereReranker (cross-encoder)
                                                           │
                                              top-5 final results
```

---

## Key Concepts

### Why Two Retrievers?

No single retrieval strategy is perfect for all financial queries:

| Query type | Example | Best strategy |
|---|---|---|
| Conceptual | "What drove profitability improvement?" | Semantic (embeddings) |
| Exact identifier | "EBITDA margin Q3 2024 Portfolio Co A" | Keyword (tsvector) |
| Mixed | "covenant compliance Section 5.2" | Both |

Semantic search understands meaning — it knows "operating profitability" relates to "EBITDA" — but it struggles with precise numbers, dates, and identifiers. Keyword search is the opposite: perfect for exact matches, blind to synonyms.

Hybrid retrieval gets the best of both.

---

### Semantic Retrieval — `src/retrieval/semantic.py`

Uses pgvector's `<=>` operator (cosine distance) to find chunks whose embedding vectors point in the same direction as the query vector.

```sql
SELECT ..., 1 - (embedding <=> $1::vector) AS score
FROM   chunks
WHERE  embedding IS NOT NULL
ORDER  BY embedding <=> $1::vector   -- ascending distance = descending similarity
LIMIT  $2
```

- `<=>` returns the **cosine distance** (0 = identical, 2 = opposite)
- `1 - distance` converts it to a **cosine similarity** score (1 = identical, -1 = opposite)
- The HNSW index (created in migration 003) makes this fast at scale

**Important:** The query vector is embedded by the caller (HybridRetriever) and passed in — SemanticRetriever never calls an embedding API. This keeps concerns separate and avoids redundant API calls.

---

### Keyword Retrieval — `src/retrieval/keyword.py`

Uses PostgreSQL's full-text search stack:

```sql
SELECT ..., ts_rank(content_tsv, plainto_tsquery('english', $1)) AS score
FROM   chunks
WHERE  content_tsv @@ plainto_tsquery('english', $1)
ORDER  BY score DESC
LIMIT  $2
```

**Key components:**

- `content_tsv` — a `tsvector` column (GIN-indexed) that Postgres builds automatically via `GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`. It tokenises, stems, and removes stop words from every chunk as it's inserted.
- `plainto_tsquery('english', $1)` — converts the query text into a tsquery (words joined by AND), applying the same stemming. "EBITDA margins" becomes `'ebitda' & 'margin'`.
- `@@` — the match operator: true if the tsvector contains all terms in the tsquery.
- `ts_rank` — scores the match by term frequency and position (terms near the start score higher).

**Why this matters for finance:** Queries like `"Q3 2024"`, `"23.4%"`, ticker symbols, and section references (`"5.2"`) have no semantic relationship to their meaning — they need exact lexical matching.

---

### Metadata Filters — `src/retrieval/filters.py`

Both retrievers accept an optional `RetrievalFilter` that narrows the search space:

```python
@dataclass
class RetrievalFilter:
    doc_type: str | None = None        # "pdf", "docx", "xlsx"
    source_file: str | None = None     # exact filename
    section_title: str | None = None   # e.g. "Executive Summary"
    chunk_strategy: str | None = None  # "fixed", "semantic", "hierarchical"
    document_id: int | None = None     # specific document by DB id
```

`build_filter_clause()` converts this into a parameterised SQL fragment:

```python
clause, params = build_filter_clause(
    RetrievalFilter(doc_type="pdf", source_file="report.pdf"),
    param_offset=2
)
# clause → "AND doc_type = $2 AND source_file = $3"
# params → ["pdf", "report.pdf"]
```

**Security:** values are always bound as parameters, never interpolated into the SQL string. This prevents SQL injection regardless of what the filter contains.

---

### Reciprocal Rank Fusion — `src/retrieval/hybrid.py`

RRF merges two ranked lists by position rather than by raw score. This avoids the problem of scores from different systems (cosine similarity vs ts_rank) being on incomparable scales.

**Formula:**
```
RRF_score(doc) = Σ  1 / (k + rank_i)
```

Where `k=60` (empirically validated default from the original Cormack 2009 paper).

**Why k=60?** Without a dampening constant, a document ranked 1st in one list would always dominate over a document ranked 2nd in both lists. k=60 smooths this: a document that's consistently good in both lists will outscore one that's exceptional in only one.

```python
# Doc A: rank 1 (semantic) only
# 1/(60+1) = 0.0164

# Doc B: rank 2 (semantic) + rank 2 (keyword)
# 1/(60+2) + 1/(60+2) = 0.0323  ← Doc B wins (consistently good)
```

The `HybridRetriever` runs both retrievers concurrently with `asyncio.gather()` — network/DB calls overlap, cutting latency roughly in half.

---

### Small-to-Big Retrieval — `HybridRetriever.fetch_parent()`

When using hierarchical chunking (Phase 3), the DB stores small child chunks (200 tokens) and large parent sections (800 tokens). Retrieval searches over children (precise matches) but can expand to parents (richer context for the LLM).

```
Search over children → find precise match → fetch parent → send full section to LLM
```

`fetch_parent(parent_id)` does a direct lookup by primary key:
```python
parent = await retriever.fetch_parent(chunk.parent_id)
# Returns the full parent section for richer LLM context
```

---

### Re-ranking — `src/retrieval/reranker.py`

Re-ranking is a second-pass scoring step. After hybrid retrieval returns the top 20 candidates, a cross-encoder re-scores them more accurately.

**Bi-encoder vs cross-encoder:**

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
|---|---|---|
| How it works | `embed(query)` · `embed(doc)` → similarity | `score(query, doc)` → relevance |
| When computed | Doc embeddings pre-computed at ingest | Computed fresh at query time |
| Query-doc interaction | None (vectors are independent) | Full — query sees the document |
| Speed | Fast (dot product) | Slow (~200-400ms per batch) |
| Accuracy | Good | Much higher |

The cross-encoder reads both the query and the document simultaneously, so it can understand phrases like "Q3 2024 revenue" in context — not just as individual token similarities.

**`CohereReranker`** calls Cohere's Rerank API:
```python
response = await self._client.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=[c.content for c in chunks],
    top_n=top_k,
)
```
Cost: ~$1 per 1,000 searches (assuming 20 candidates → 5 results).

**`PassThroughReranker`** is a no-op that returns `chunks[:top_k]` unchanged. Used when no Cohere API key is configured, or in tests that don't need actual re-ranking.

---

## File Structure

```
src/retrieval/
├── __init__.py
├── models.py       — RetrievalFilter, RetrievedChunk dataclasses
├── filters.py      — build_filter_clause() SQL builder
├── semantic.py     — SemanticRetriever (pgvector <=> cosine)
├── keyword.py      — KeywordRetriever (tsvector + plainto_tsquery)
├── hybrid.py       — rrf_fusion() + HybridRetriever
└── reranker.py     — BaseReranker, CohereReranker, PassThroughReranker
```

---

## How It All Connects to Previous Phases

| Dependency | Used by Phase 4 |
|---|---|
| Phase 1 (DB pool + migrations) | `asyncpg.Pool` passed into every retriever |
| Phase 2 (EmbeddingModel) | `HybridRetriever` calls `embedder.embed(query)` once |
| Phase 3 (chunks table, content_tsv, HNSW index) | All three indexes (HNSW, GIN, document_id) are queried here |

Phase 4 is the read side of the system. Phase 3 was the write side. They share the same `chunks` table as the boundary.

---

## Tests — `tests/test_retrieval.py`

25 tests across 7 categories:

| Category | Tests | What's verified |
|---|---|---|
| `build_filter_clause` | 5 | None → empty; single/multi/all fields; all-None |
| `rrf_fusion` | 6 | Disjoint lists; boost for dual-list chunks; top_k truncation; sequential ranks; empty inputs |
| `SemanticRetriever` | 3 | Closest vector = rank 1; metadata populated; doc_type filter |
| `KeywordRetriever` | 3 | Exact term found; term frequency affects rank; source_file filter |
| `HybridRetriever` | 2 | Fused results from both sub-retrievers; fetch_parent returns parent/None |
| `PassThroughReranker` | 3 | Order preserved; top_k truncation; empty input |
| `CohereReranker` | 3 | Mock API re-orders correctly; top_n forwarded; empty input skips API call |

**Test isolation technique:**
Each DB test uses `uuid.uuid4().hex[:8]` as a suffix in `source_file` (e.g., `semantic_test_a3f2c891.pdf`). This prevents chunks from one test run appearing in another test's results — the same technique used for embedding cache tests in Phase 2.

**Why filter in the semantic rank-1 test:**
When run alone, `top_k=2` is sufficient. When run alongside Phase 3 ingestion tests (which leave real chunks with real embeddings in the DB), those chunks can crowd out the test data. Adding `filters=RetrievalFilter(source_file=source_file)` scopes the query to only the test's own chunks — reliable regardless of DB state.

---

## Running Phase 4 Tests

```bash
# Phase 4 only
python -m uv run pytest tests/test_retrieval.py -v

# Full suite (all phases)
python -m uv run pytest -v
```

Expected result: **52 passed, 1 skipped** (the OpenAI integration test requires a live API key).

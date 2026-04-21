# Phase 2: Embeddings & Vector Storage

## Overview

Phase 2 builds the semantic layer of the platform — the ability to convert text into mathematical representations (embeddings) that capture meaning, store them efficiently in Postgres, and retrieve them by similarity. This is the core technology that makes RAG (Retrieval-Augmented Generation) possible.

By the end of this phase you have:
- An abstract embedding interface supporting multiple providers
- OpenAI and Voyage AI implementations
- A Postgres-backed embedding cache (no text is ever embedded twice)
- An HNSW index on the vector column for fast similarity search
- 13 passing tests (12 unit + 1 integration)

---

## Table of Contents

1. [Concepts](#concepts)
   - [What is an Embedding?](#what-is-an-embedding)
   - [How Embeddings Encode Meaning](#how-embeddings-encode-meaning)
   - [Embedding Model Landscape](#embedding-model-landscape)
   - [Vector Dimensions](#vector-dimensions)
   - [Distance Metrics](#distance-metrics)
   - [pgvector Index Types](#pgvector-index-types)
   - [Type Codecs in asyncpg](#type-codecs-in-asyncpg)
   - [Embedding Caching Strategy](#embedding-caching-strategy)
2. [Project Structure](#project-structure)
3. [File Reference](#file-reference)
4. [Setup & Running](#setup--running)
5. [Tests](#tests)
6. [Lessons Learned](#lessons-learned)

---

## Concepts

### What is an Embedding?

An embedding is a **dense vector** — a list of floating-point numbers — that represents the semantic meaning of a piece of text. The key property is:

> Texts with similar meaning produce vectors that are close together in high-dimensional space.

```
"revenue"        → [0.21, -0.54, 0.87, 0.03, ...]   ← 1536 numbers
"income"         → [0.23, -0.51, 0.85, 0.01, ...]   ← close to "revenue"
"annual earnings"→ [0.19, -0.56, 0.89, 0.05, ...]   ← also close
"cat"            → [-0.88, 0.12, -0.33, 0.77, ...]  ← far from "revenue"
```

This is what makes semantic search possible. Instead of matching exact keywords ("EBITDA" only matches "EBITDA"), we match meaning — a query for "operating profit" will find documents containing "EBITDA", "earnings before interest", "operating income", and similar terms.

---

### How Embeddings Encode Meaning

Embedding models are neural networks trained on massive text corpora. During training, they learn to place related words and phrases near each other in vector space. The famous example:

```
king − man + woman ≈ queen
```

This arithmetic works because the model learned that:
- `king` and `queen` differ along a gender dimension
- `man` and `woman` also differ along that same dimension
- So subtracting `man` and adding `woman` moves in the "female royalty" direction

For financial text, a domain-trained model learns relationships like:
```
revenue − sales + receipts ≈ income
EBITDA − depreciation ≈ EBIT
leverage + equity ≈ enterprise value
```

---

### Embedding Model Landscape

| Model | Provider | Dimensions | Context Window | Best For | Cost |
|---|---|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | 8191 tokens | General purpose, fast, cheap | ~$0.02/M tokens |
| `text-embedding-3-large` | OpenAI | 3072 | 8191 tokens | Higher accuracy | ~$0.13/M tokens |
| `voyage-finance-2` | Voyage AI | 1024 | 32000 tokens | Financial documents (SEC, earnings) | ~$0.12/M tokens |
| `voyage-3` | Voyage AI | 1024 | 32000 tokens | General high accuracy | ~$0.06/M tokens |
| `bge-m3` | BAAI (open-source) | 1024 | 8192 tokens | Self-hosted, multilingual | Free |
| `nomic-embed-text` | Nomic (open-source) | 768 | 8192 tokens | Self-hosted, fast | Free |

**Which to use for this project:**

- **Default:** `text-embedding-3-small` — good balance of quality and cost; handles financial text well
- **High-precision retrieval:** `voyage-finance-2` — outperforms general models on domain-specific financial queries (private equity terminology, agribusiness metrics, regulatory language)
- **No API cost:** `bge-m3` — strong open-source option; requires local GPU for production throughput

**Key trade-offs to understand:**
- Higher dimensions → better accuracy but more storage and slower search
- Larger context window → can embed longer documents without truncation
- Domain-trained → better recall on specialised terminology

---

### Vector Dimensions

The dimension of an embedding is the length of the output list. `text-embedding-3-small` outputs 1536 numbers per text.

Why does dimension matter?

1. **Storage:** Each dimension is a 4-byte float. One embedding = 1536 × 4 = ~6KB. One million embeddings = ~6GB.
2. **Index size:** The HNSW index stores additional graph structure — roughly 2–3× the raw vector data.
3. **Search speed:** More dimensions = more computation per similarity comparison.
4. **Accuracy:** More dimensions give the model more "room" to encode nuance.

In our schema, we use `vector(1536)` — fixed at OpenAI's dimension. If you switch to Voyage (1024 dims) or a 3072-dim model, you would need a separate table or a schema migration.

---

### Distance Metrics

pgvector supports three distance metrics, each corresponding to a different operator and index operator class:

| Operator | Metric | Formula | Index Class | Use When |
|---|---|---|---|---|
| `<=>` | Cosine distance | `1 - (A·B / |A||B|)` | `vector_cosine_ops` | Text embeddings (most common) |
| `<->` | L2 (Euclidean) | `√Σ(Aᵢ - Bᵢ)²` | `vector_l2_ops` | When magnitude matters |
| `<#>` | Negative inner product | `-(A·B)` | `vector_ip_ops` | When vectors are unit-normalised |

**We use cosine similarity (`<=>`)** for two reasons:
1. OpenAI and Voyage both recommend it for their models
2. Cosine similarity is magnitude-independent — a short sentence and a long document about the same topic produce similar cosine similarity scores, even though their raw vectors have very different magnitudes

**Cosine similarity vs cosine distance:**
- pgvector's `<=>` returns *distance* (0 = identical, 2 = opposite)
- To get *similarity* (1 = identical, -1 = opposite): `1 - (A <=> B)`
- For ranking: order by `<=>` ascending (smallest distance = most similar)

---

### pgvector Index Types

Without an index, every similarity query does a **sequential scan** — comparing the query vector against every stored vector one by one. At 1 million rows this takes seconds. An ANN (Approximate Nearest Neighbor) index makes it milliseconds.

pgvector provides two index types:

#### IVFFlat (Inverted File with Flat quantization)
- Divides vectors into `lists` clusters (like bins)
- Query searches only the nearest `probes` clusters, not all vectors
- **Pros:** Fast to build, low memory
- **Cons:** Lower recall; must rebuild when data distribution changes significantly
- **Best for:** Large static datasets, infrequent writes

```sql
CREATE INDEX ON embeddings_cache USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- sqrt(row_count) is a common heuristic
```

#### HNSW (Hierarchical Navigable Small World)
- Builds a multi-layer graph where each node connects to its nearest neighbours
- Query traverses the graph from top layer down to find approximate nearest neighbours
- **Pros:** Much higher recall, faster queries, handles insertions well
- **Cons:** Slower to build, higher memory usage
- **Best for:** Production workloads with continuous inserts and frequent queries

```sql
CREATE INDEX ON embeddings_cache USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**HNSW parameters:**
- `m` — number of connections per node per layer. Higher = better recall, more memory. Default 16 is good for most cases.
- `ef_construction` — search depth during index build. Higher = better quality index, slower build. 64 is a safe default.
- At query time, `SET hnsw.ef_search = 100` increases recall at the cost of latency (default 40).

**We use HNSW** because our platform continuously ingests new documents and queries run constantly — the graph structure handles both well.

---

### Type Codecs in asyncpg

asyncpg is a high-performance driver that communicates with Postgres using the binary protocol. Every Postgres type (INTEGER, TEXT, TIMESTAMPTZ, etc.) has a corresponding codec — a pair of functions that:
- **Encode:** Convert a Python object → Postgres binary format
- **Decode:** Convert Postgres binary format → Python object

asyncpg ships with codecs for all built-in Postgres types. But `vector` is not a built-in type — it's added by the pgvector extension. Without a codec, asyncpg falls back to treating it as text, which means:

```python
# Without codec registration:
await conn.execute("INSERT INTO t (embedding) VALUES ($1)", [0.1, 0.2, 0.3])
# → DataError: invalid input for query argument $1 (expected str, got list)

# With codec registration:
await register_vector(conn)
await conn.execute("INSERT INTO t (embedding) VALUES ($1)", [0.1, 0.2, 0.3])
# → Works correctly
```

The `pgvector` Python package provides `register_vector(conn)` which teaches asyncpg how to handle the `vector` type. We call it in two places:

1. **`connection.py`** — via the `init` parameter in `create_pool()`, which runs on every new connection the pool creates
2. **`conftest.py`** — via the `init` parameter in the test pool fixture

---

### Embedding Caching Strategy

#### Why cache?
- **Cost:** Embedding 1 million tokens with OpenAI costs ~$0.02. Re-embedding the same tokens repeatedly wastes money.
- **Latency:** An API call takes ~100ms. A DB lookup takes ~1ms. For a document ingested 10 times, the first costs 100ms, the next 9 cost 1ms each.
- **Stability:** If the API is temporarily unavailable, cached embeddings keep the system running.

#### How the cache works

```
embed("EBITDA margin improved Q3")
  │
  ├── SHA256("text-embedding-3-small:EBITDA margin improved Q3")
  │      = "a3f9c2..." (cache key)
  │
  ├── SELECT embedding FROM embeddings_cache
  │   WHERE content_hash = 'a3f9c2...' AND model_name = 'text-embedding-3-small'
  │
  ├── HIT  → return list[float] immediately (no API call)
  └── MISS → call API → INSERT into cache → return list[float]
```

#### Hash design decisions

The hash includes the model name (`model_name + ":" + text`) because:
- The same text embedded by OpenAI and Voyage produces *different* vectors
- They should not share a cache entry
- `UNIQUE (content_hash, model_name)` enforces this at the DB level

#### Concurrency safety

If two coroutines try to cache the same text simultaneously:
- Both get a cache miss
- Both call the API (small waste, but rare)
- Both try to INSERT
- The second INSERT hits the `UNIQUE` constraint
- `ON CONFLICT DO NOTHING` silently ignores it
- No error, no duplicate row

---

## Project Structure

```
massa-platform/
├── pyproject.toml                        # Added: openai, voyageai, numpy, pgvector
│
├── src/
│   ├── config.py                         # Added: openai_api_key, voyage_api_key fields
│   ├── db/
│   │   ├── connection.py                 # Updated: init=register_vector in create_pool()
│   │   └── migrations/
│   │       └── 002_vectors.sql           # NEW: embeddings_cache table + HNSW index
│   └── embeddings/
│       ├── __init__.py
│       ├── base.py                       # NEW: abstract EmbeddingModel interface
│       ├── openai_embedder.py            # NEW: OpenAI text-embedding-3-small/large
│       ├── voyage_embedder.py            # NEW: Voyage AI voyage-finance-2
│       └── cache.py                      # NEW: DB-backed embedding cache
│
└── tests/
    ├── conftest.py                       # Updated: init=register_vector in test pool
    └── test_embeddings.py                # NEW: 8 tests (7 unit + 1 integration)
```

---

## File Reference

### `002_vectors.sql`
Creates the `embeddings_cache` table with:
- `content_hash TEXT` — SHA256 of `(model_name + text)`, used as cache lookup key
- `model_name TEXT` — embedding model identifier; part of the unique constraint
- `content TEXT` — original text stored for debugging and auditing
- `embedding vector(1536)` — the actual embedding vector
- `UNIQUE (content_hash, model_name)` — enforces one embedding per text per model
- HNSW index on `embedding` with `vector_cosine_ops` — enables fast cosine similarity queries

### `src/embeddings/base.py`
Abstract base class `EmbeddingModel` with four required methods/properties:
- `model_name` — string identifier used as cache key
- `dimensions` — output vector length
- `embed(text)` — single text → `list[float]`
- `embed_batch(texts)` — multiple texts → `list[list[float]]` (one API call)

Any new embedding provider (Cohere, Mistral, local model) implements this interface and is immediately usable everywhere in the platform.

### `src/embeddings/openai_embedder.py`
Wraps OpenAI's async embeddings API. Supports `text-embedding-3-small` (1536 dims) and `text-embedding-3-large` (3072 dims). Sorts batch results by index to guarantee order consistency with the input list.

### `src/embeddings/voyage_embedder.py`
Wraps Voyage AI's async embeddings API. Default model is `voyage-finance-2` — trained on financial corpora. Uses `input_type="document"` for storage (vs `"query"` for search-time queries — Voyage recommends different embeddings for each).

### `src/embeddings/cache.py`
`CachedEmbedder` wraps any `EmbeddingModel`. Key behaviours:
- `embed()` — single-text cache with DB fallback
- `embed_batch()` — fetches all cached entries in one query; only calls API for uncached texts; preserves input order in output
- `_make_hash()` — SHA256 of `model_name:text`

### `src/db/connection.py`
Updated to pass `init=_init_connection` to `create_pool()`. The `_init_connection` callback calls `register_vector(conn)` on each new connection, registering the pgvector type codec with asyncpg.

### `src/config.py`
Added two optional string fields:
- `openai_api_key` — defaults to `""` if not set (unit tests do not need it)
- `voyage_api_key` — defaults to `""` if not set

---

## Setup & Running

### Prerequisites
Completed Phase 1 setup (Docker running, dependencies installed, migrations applied).

### Add API keys to `.env`

Open `.env` and replace the placeholder values:
```
OPENAI_API_KEY=sk-...your-real-key...
VOYAGE_API_KEY=pa-...your-real-key...
```

Get keys from:
- OpenAI: https://platform.openai.com/api-keys
- Voyage AI: https://dash.voyageai.com/api-keys

### Install new dependencies
```bash
python -m uv sync --all-extras
```

### Run the new migration
```bash
python -m uv run python -m src.db.migrate
```

Expected output:
```
  [skip] 001_init.sql already applied
  [ok]   002_vectors.sql applied
```

### Run all tests
```bash
python -m uv run pytest tests/ -v
```

Expected output:
```
tests/test_embeddings.py::test_openai_embedder_returns_correct_shape PASSED
tests/test_embeddings.py::test_openai_embedder_batch_returns_correct_count PASSED
tests/test_embeddings.py::test_openai_embedder_unknown_model_raises PASSED
tests/test_embeddings.py::test_hnsw_index_exists PASSED
tests/test_embeddings.py::test_embeddings_cache_table_schema PASSED
tests/test_embeddings.py::test_cache_stores_and_retrieves_embedding PASSED
tests/test_embeddings.py::test_cache_different_texts_get_different_entries PASSED
tests/test_embeddings.py::test_semantic_similarity_financial_terms SKIPPED
13 passed, 1 skipped
```

### Run the integration test (requires API key)
```bash
python -m uv run pytest tests/test_embeddings.py::test_semantic_similarity_financial_terms -v
```

This test:
1. Calls the real OpenAI API
2. Embeds "revenue", "income", "cat"
3. Computes cosine similarity between each pair
4. Asserts `sim(revenue, income) > sim(revenue, cat)`

---

## Tests

### `test_openai_embedder_returns_correct_shape`
**Type:** Unit (mocked API)
**What it tests:** `embed()` returns a `list[float]` of exactly 1536 elements.
**Why it matters:** If the wrapper returns the wrong type or length, every downstream component (cache, retriever, indexer) will fail with confusing errors.

### `test_openai_embedder_batch_returns_correct_count`
**Type:** Unit (mocked API)
**What it tests:** `embed_batch()` returns exactly one vector per input text, in the same order.
**Why it matters:** Order preservation is a contract. If the batch result at index 2 corresponds to input at index 0, you've silently associated the wrong embedding with the wrong text — a data corruption bug that's very hard to detect.

### `test_openai_embedder_unknown_model_raises`
**Type:** Unit
**What it tests:** Passing an unsupported model name raises `ValueError` immediately.
**Why it matters:** Fail fast at construction time. If this check didn't exist, the error would surface as a cryptic OpenAI API error after the first API call.

### `test_hnsw_index_exists`
**Type:** Integration (DB, no API)
**What it tests:** The HNSW index `embeddings_cache_hnsw_idx` exists in `pg_indexes` after migration.
**Why it matters:** Without this index, similarity searches work but do a full sequential scan. At scale this is catastrophically slow. This test acts as a guard — if someone drops the index accidentally, CI fails.

### `test_embeddings_cache_table_schema`
**Type:** Integration (DB, no API)
**What it tests:** All required columns exist with correct types. Specifically that `created_at` is `TIMESTAMPTZ` (timezone-aware), not a naive `TIMESTAMP`.
**Why it matters:** Schema drift is a real risk when multiple developers run migrations. Catching it early prevents runtime errors buried in asyncpg type-codec failures.

### `test_cache_stores_and_retrieves_embedding`
**Type:** Integration (DB + mock API)
**What it tests:** The cache hits the mock embedder exactly once for two calls with the same text.
**Why it matters:** This is the core cache correctness test. A broken cache that always calls the API would still return correct results but burn money and latency silently.

### `test_cache_different_texts_get_different_entries`
**Type:** Integration (DB + mock API)
**What it tests:** Two different texts each trigger one API call — they don't share a cache entry.
**Why it matters:** If the hash function had a collision or the uniqueness logic was wrong, two different texts could return the same embedding — a subtle semantic corruption that would be very hard to debug in production.

### `test_semantic_similarity_financial_terms` *(integration)*
**Type:** Integration (real OpenAI API)
**What it tests:** `cosine_similarity("revenue", "income") > cosine_similarity("revenue", "cat")`
**Why it matters:** This is the fundamental property the entire RAG system depends on. If embeddings don't encode semantic proximity correctly, retrieval is broken at its foundation. Running this test after switching embedding models confirms the new model meets the basic semantic requirement.

---

## Lessons Learned

### asyncpg requires explicit codec registration for pgvector

**Problem:** Passing a `list[float]` to a `vector` column raised:
```
DataError: invalid input for query argument $4 (expected str, got list)
```

**Root cause:** pgvector is a Postgres extension — asyncpg has no built-in knowledge of the `vector` type. Without a codec, it treats it as `text` and tries to cast the Python list to a string.

**Fix:** Install the `pgvector` Python package and call `register_vector(conn)` on every connection. The `init` parameter in `asyncpg.create_pool()` is the right place — it runs on every new connection automatically.

**Lesson:** When using Postgres extensions that add custom types (pgvector, PostGIS, hstore, etc.), always check whether your driver needs a type codec registered. Silent type coercion failures are hard to diagnose.

---

### The `init` parameter in asyncpg pools

`asyncpg.create_pool(init=callback)` runs `callback(conn)` every time the pool opens a new connection. This is the correct place for per-connection setup like:
- Registering custom type codecs (`register_vector`)
- Setting session-level Postgres parameters (`SET search_path`, `SET hnsw.ef_search`)
- Setting statement timeouts for safety (`SET statement_timeout`)

Do not use `after_acquire` for codec registration — that runs after the connection is taken from the pool, which is too late for the codec to be available during query preparation.

---

### Unit tests vs integration tests for embeddings

Embedding tests fall into two clear categories:

**Unit tests (no API key needed):**
- Shape/type contract tests — mock the API response, verify our wrapper handles it correctly
- Error handling — invalid model names, missing keys
- Cache logic — mock the embedder, verify DB read/write behaviour

**Integration tests (real API key required):**
- Semantic correctness — does the model actually encode meaning correctly?
- Domain fit — are financial terms correctly clustered?

Separating them with `pytest.mark.skipif` means the unit suite runs in CI without secrets, and the integration suite runs manually or in a separate CI job with credentials.

---

### Batch embedding over single embedding

Always prefer `embed_batch()` over calling `embed()` in a loop:

```python
# Bad — N API calls, N × latency
embeddings = [await embedder.embed(text) for text in texts]

# Good — 1 API call regardless of N
embeddings = await embedder.embed_batch(texts)
```

OpenAI charges the same per token whether you send 1 text or 100. The latency difference is dramatic: 100 sequential calls × 100ms = 10 seconds vs 1 batch call × 150ms = 0.15 seconds. The `CachedEmbedder.embed_batch()` method further optimises this by only sending uncached texts to the API.

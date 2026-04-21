"""
Phase 2 Tests: Embeddings & Vector Storage

Test categories:
- Unit tests   : mock the API — run with no API key, always fast
- Integration  : call real API — skipped unless OPENAI_API_KEY is set
"""

import hashlib
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import settings
from src.embeddings.base import EmbeddingModel
from src.embeddings.cache import CachedEmbedder
from src.embeddings.openai_embedder import OpenAIEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_embedding(text: str, dims: int = 1536) -> list[float]:
    """
    Produces a deterministic fake embedding for a given text.
    Different texts → different vectors; same text → same vector.
    Used in unit tests to avoid real API calls.
    """
    rng = np.random.default_rng(seed=abs(hash(text)) % (2**32))
    vec = rng.standard_normal(dims).astype(float)
    # Normalise to unit length (cosine similarity requires this)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


requires_openai = pytest.mark.skipif(
    not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here",
    reason="OPENAI_API_KEY not set — skipping integration test",
)


# ---------------------------------------------------------------------------
# Test 1: Embedder interface — shape and type contract
# ---------------------------------------------------------------------------

async def test_openai_embedder_returns_correct_shape():
    """
    Assert that OpenAIEmbedder.embed() returns a list of floats
    with the correct number of dimensions.

    We mock the OpenAI client so no real API call is made.
    This tests our wrapper code, not OpenAI itself.
    """
    fake_embedding = make_fake_embedding("test", dims=1536)

    with patch("src.embeddings.openai_embedder.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding, index=0)]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="fake-key")
        result = await embedder.embed("What is the EBITDA margin?")

    assert isinstance(result, list), "embed() should return a list"
    assert len(result) == 1536, "text-embedding-3-small has 1536 dimensions"
    assert all(isinstance(v, float) for v in result), "all values should be floats"


async def test_openai_embedder_batch_returns_correct_count():
    """
    Assert that embed_batch() returns one vector per input text,
    in the same order as the input.
    """
    texts = ["revenue growth", "EBITDA margin", "portfolio IRR"]
    fake_embeddings = [make_fake_embedding(t) for t in texts]

    with patch("src.embeddings.openai_embedder.AsyncOpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=emb, index=i)
            for i, emb in enumerate(fake_embeddings)
        ]
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)

        embedder = OpenAIEmbedder(api_key="fake-key")
        results = await embedder.embed_batch(texts)

    assert len(results) == 3, "should return one embedding per input text"
    assert len(results[0]) == 1536, "each embedding should be 1536 dimensions"


def test_openai_embedder_unknown_model_raises():
    """
    Assert that passing an unknown model name raises ValueError immediately,
    not later when an API call fails.
    """
    with pytest.raises(ValueError, match="Unknown model"):
        OpenAIEmbedder(api_key="fake-key", model="gpt-4-turbo")


# ---------------------------------------------------------------------------
# Test 2: pgvector schema — index exists after migration
# ---------------------------------------------------------------------------

async def test_hnsw_index_exists(db_pool):
    """
    Assert that the HNSW index was created on embeddings_cache by migration 002.

    pg_indexes is a Postgres catalog view listing all indexes.
    If this index is missing, similarity search will fall back to a sequential
    scan — correct results but catastrophically slow at scale.
    """
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM   pg_indexes
                WHERE  tablename  = 'embeddings_cache'
                AND    indexname  = 'embeddings_cache_hnsw_idx'
            )
        """)

    assert exists is True, "HNSW index should exist after 002_vectors.sql migration"


async def test_embeddings_cache_table_schema(db_pool):
    """
    Assert that embeddings_cache has all required columns with correct types.
    Catching schema drift early prevents cryptic INSERT errors later.
    """
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM   information_schema.columns
            WHERE  table_name = 'embeddings_cache'
            ORDER  BY ordinal_position
        """)

    columns = {row["column_name"]: row["data_type"] for row in rows}

    assert "content_hash" in columns, "content_hash column missing"
    assert "model_name"   in columns, "model_name column missing"
    assert "content"      in columns, "content column missing"
    assert "embedding"    in columns, "embedding column missing"
    assert "created_at"   in columns, "created_at column missing"
    assert columns["created_at"] == "timestamp with time zone", \
        "created_at should be TIMESTAMPTZ, not naive timestamp"


# ---------------------------------------------------------------------------
# Test 3: Cache — API called once, DB used on second call
# ---------------------------------------------------------------------------

async def test_cache_stores_and_retrieves_embedding(db_pool):
    """
    Assert that the second embed() call for the same text returns from DB,
    not from the API.

    This is the core cache correctness test:
    - Call 1: API mock is called, result stored in DB
    - Call 2: DB row found, API mock is NOT called
    """
    fake_embedding = make_fake_embedding("operating cash flow")
    call_count = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return fake_embedding

    # Build a mock embedder that tracks how many times it was called
    mock_embedder = MagicMock(spec=EmbeddingModel)
    # Unique model name per test run — prevents cache hits from previous runs
    mock_embedder.model_name = f"test-model-cache-{uuid.uuid4().hex[:8]}"
    mock_embedder.dimensions = 1536
    mock_embedder.embed = fake_embed

    cached = CachedEmbedder(embedder=mock_embedder, pool=db_pool)

    # First call — should hit the mock embedder
    result1 = await cached.embed("operating cash flow")
    assert call_count == 1, "API should be called on first embed"

    # Second call — should return from DB without calling the embedder again
    result2 = await cached.embed("operating cash flow")
    assert call_count == 1, "API should NOT be called on second embed (cache hit)"

    assert result1 == result2, "cached result should match original"


async def test_cache_different_texts_get_different_entries(db_pool):
    """
    Assert that two different texts produce two separate cache entries.
    Verifies the hash function differentiates content correctly.
    """
    call_count = 0

    async def fake_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        return make_fake_embedding(text)

    mock_embedder = MagicMock(spec=EmbeddingModel)
    mock_embedder.model_name = f"test-model-diff-{uuid.uuid4().hex[:8]}"
    mock_embedder.dimensions = 1536
    mock_embedder.embed = fake_embed

    cached = CachedEmbedder(embedder=mock_embedder, pool=db_pool)

    await cached.embed("revenue")
    await cached.embed("EBITDA")

    assert call_count == 2, "two different texts should each call the API once"


# ---------------------------------------------------------------------------
# Test 4: Similarity sanity — semantic proximity (integration, needs API key)
# ---------------------------------------------------------------------------

@requires_openai
async def test_semantic_similarity_financial_terms():
    """
    Integration test: real API call.
    Assert that 'revenue' is semantically closer to 'income' than to 'cat'.

    This is the fundamental property embeddings must have to be useful.
    Cosine similarity = dot product of unit vectors = ranges from -1 to 1.
    Higher = more similar.

    Skipped automatically if OPENAI_API_KEY is not set.
    """
    embedder = OpenAIEmbedder(api_key=settings.openai_api_key)

    vectors = await embedder.embed_batch(["revenue", "income", "cat"])
    v_revenue, v_income, v_cat = [np.array(v) for v in vectors]

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_revenue_income = cosine_sim(v_revenue, v_income)
    sim_revenue_cat    = cosine_sim(v_revenue, v_cat)

    assert sim_revenue_income > sim_revenue_cat, (
        f"'revenue' should be closer to 'income' ({sim_revenue_income:.3f}) "
        f"than to 'cat' ({sim_revenue_cat:.3f})"
    )

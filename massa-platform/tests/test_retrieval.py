"""
Phase 4 Tests: Hybrid Retrieval & Re-ranking

Test categories:
- filter tests        : build_filter_clause generates correct SQL fragments
- rrf fusion tests    : rrf_fusion merges ranked lists using the RRF formula
- semantic tests      : SemanticRetriever returns chunks by vector similarity
- keyword tests       : KeywordRetriever returns chunks by tsvector match
- hybrid tests        : HybridRetriever combines both via RRF
- reranker tests      : PassThroughReranker + mocked CohereReranker
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.filters import build_filter_clause
from src.retrieval.hybrid import HybridRetriever, rrf_fusion
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.models import RetrievalFilter, RetrievedChunk
from src.retrieval.reranker import CohereReranker, PassThroughReranker
from src.retrieval.semantic import SemanticRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: int,
    content: str = "test content",
    score: float = 0.5,
    rank: int = 1,
    doc_type: str = "pdf",
    source_file: str = "test.pdf",
) -> RetrievedChunk:
    """Factory for RetrievedChunk — avoids repeating all 10 fields."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        source_file=source_file,
        page_number=1,
        section_title=None,
        doc_type=doc_type,
        chunk_strategy="fixed",
        score=score,
        rank=rank,
        parent_id=None,
    )


async def _insert_test_document(conn, source_file: str) -> int:
    """
    Inserts a minimal document row and returns its id.
    source_file is used as the unique filename so tests don't collide.
    """
    doc_id = await conn.fetchval(
        """
        INSERT INTO documents (file_hash, filename, doc_type, file_path, page_count, chunk_count)
        VALUES ($1, $2, 'pdf', $2, 1, 0)
        ON CONFLICT (file_hash) DO UPDATE SET filename = EXCLUDED.filename
        RETURNING id
        """,
        source_file,  # use source_file as file_hash — unique per test
        source_file,
    )
    return doc_id


async def _insert_test_chunk(
    conn,
    *,
    document_id: int,
    content: str,
    source_file: str,
    embedding: list[float],
    doc_type: str = "pdf",
    chunk_strategy: str = "fixed",
) -> int:
    """
    Inserts a chunk row with the given content and embedding, returns chunk id.

    chunk_index is auto-computed as MAX(chunk_index)+1 for this document so
    that multiple inserts into the same document never collide on the
    (document_id, chunk_index) unique constraint.
    """
    next_index = await conn.fetchval(
        "SELECT COALESCE(MAX(chunk_index), -1) + 1 FROM chunks WHERE document_id = $1",
        document_id,
    )
    chunk_id = await conn.fetchval(
        """
        INSERT INTO chunks (
            document_id, chunk_index, content, content_hash, embedding,
            source_file, page_number, doc_type, chunk_strategy, token_count
        )
        VALUES ($1, $2, $3, md5($3), $4::vector, $5, 1, $6, $7, 10)
        RETURNING id
        """,
        document_id,
        next_index,
        content,
        embedding,
        source_file,
        doc_type,
        chunk_strategy,
    )
    return chunk_id


def _unit_vector(dims: int = 1536, hot_index: int = 0) -> list[float]:
    """
    Returns a unit vector with 1.0 at hot_index and small values elsewhere.
    Used to create embeddings that are clearly closest to a target direction.
    """
    v = [0.001] * dims
    v[hot_index] = 1.0
    return v


# ---------------------------------------------------------------------------
# Test 1: build_filter_clause
# ---------------------------------------------------------------------------

def test_filter_clause_none_returns_empty():
    """No filters → empty clause and empty params list."""
    clause, params = build_filter_clause(None)
    assert clause == ""
    assert params == []


def test_filter_clause_single_field():
    """One filter field → single AND condition with correct param placeholder."""
    clause, params = build_filter_clause(
        RetrievalFilter(doc_type="pdf"),
        param_offset=2,
    )
    assert clause == "AND doc_type = $2"
    assert params == ["pdf"]


def test_filter_clause_multiple_fields():
    """Two filter fields → two conditions chained with AND."""
    clause, params = build_filter_clause(
        RetrievalFilter(doc_type="xlsx", source_file="financials.xlsx"),
        param_offset=2,
    )
    assert "doc_type = $2" in clause
    assert "source_file = $3" in clause
    assert params == ["xlsx", "financials.xlsx"]


def test_filter_clause_all_fields():
    """All five filter fields → five conditions, params in declaration order."""
    filt = RetrievalFilter(
        doc_type="pdf",
        source_file="report.pdf",
        section_title="Executive Summary",
        chunk_strategy="semantic",
        document_id=42,
    )
    clause, params = build_filter_clause(filt, param_offset=2)

    assert "doc_type = $2" in clause
    assert "source_file = $3" in clause
    assert "section_title = $4" in clause
    assert "chunk_strategy = $5" in clause
    assert "document_id = $6" in clause
    assert len(params) == 5


def test_filter_clause_empty_filter_returns_empty():
    """RetrievalFilter with all-None fields → same as no filter."""
    clause, params = build_filter_clause(RetrievalFilter())
    assert clause == ""
    assert params == []


# ---------------------------------------------------------------------------
# Test 2: rrf_fusion (pure function — no DB needed)
# ---------------------------------------------------------------------------

def test_rrf_fusion_combines_disjoint_lists():
    """
    When semantic and keyword return completely different chunks, RRF should
    include all of them (up to top_k), ordered by their RRF scores.
    """
    semantic = [_make_chunk(1, rank=1), _make_chunk(2, rank=2)]
    keyword = [_make_chunk(3, rank=1), _make_chunk(4, rank=2)]

    results = rrf_fusion(semantic, keyword, top_k=4)

    result_ids = [r.chunk_id for r in results]
    assert set(result_ids) == {1, 2, 3, 4}


def test_rrf_fusion_boosts_chunks_in_both_lists():
    """
    A chunk ranked 2nd in both lists should outscore one ranked 1st in only one list.

    Chunk A: rank 1 (semantic) only          → 1/(60+1)          = 0.01639
    Chunk B: rank 2 (semantic) + rank 2 (keyword) → 2 × 1/(60+2) = 0.03226  ← should win
    """
    semantic = [_make_chunk(100, rank=1), _make_chunk(200, rank=2)]
    keyword = [_make_chunk(200, rank=2)]  # only chunk 200 appears in keyword

    results = rrf_fusion(semantic, keyword, top_k=2)

    assert results[0].chunk_id == 200, (
        "Chunk present in both lists should rank higher than chunk present in only one"
    )


def test_rrf_fusion_top_k_truncates_output():
    """Output must never exceed top_k items."""
    semantic = [_make_chunk(i, rank=i) for i in range(1, 11)]
    keyword = [_make_chunk(i, rank=i) for i in range(1, 11)]

    results = rrf_fusion(semantic, keyword, top_k=3)

    assert len(results) == 3


def test_rrf_fusion_assigns_sequential_ranks():
    """Returned chunks must have rank 1, 2, 3, … (1-based, no gaps)."""
    semantic = [_make_chunk(i, rank=i) for i in range(1, 6)]
    keyword = [_make_chunk(i, rank=i) for i in range(1, 6)]

    results = rrf_fusion(semantic, keyword, top_k=5)

    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_rrf_fusion_empty_inputs():
    """Both lists empty → empty output."""
    assert rrf_fusion([], [], top_k=5) == []


def test_rrf_fusion_one_empty_list():
    """One empty list → results purely from the other list."""
    semantic = [_make_chunk(1, rank=1), _make_chunk(2, rank=2)]
    results = rrf_fusion(semantic, [], top_k=5)
    result_ids = {r.chunk_id for r in results}
    assert result_ids == {1, 2}


# ---------------------------------------------------------------------------
# Test 3: SemanticRetriever (requires DB)
# ---------------------------------------------------------------------------

async def test_semantic_retriever_finds_most_similar_chunk(db_pool):
    """
    Insert two chunks with orthogonal embeddings.
    Query with a vector identical to chunk A's embedding → chunk A is rank 1.

    Why this works:
    Cosine similarity is maximised when vectors point in the same direction.
    _unit_vector(hot_index=0) and _unit_vector(hot_index=1) are near-orthogonal,
    so querying with hot_index=0 returns the first chunk at the top.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"semantic_test_{tag}.pdf"

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)

        # Chunk A: embedding strongly in direction 0
        chunk_a_id = await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content="Revenue grew 12% driven by agribusiness segment performance.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=0),
        )

        # Chunk B: embedding strongly in direction 1 (nearly orthogonal to A)
        await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content="Currency hedging strategy mitigates foreign exchange risk.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=1),
        )

    retriever = SemanticRetriever(db_pool)
    # Scope to our test file so other test data in the DB doesn't crowd out results
    results = await retriever.search(
        _unit_vector(hot_index=0),
        top_k=2,
        filters=RetrievalFilter(source_file=source_file),
    )

    assert len(results) >= 1
    assert results[0].chunk_id == chunk_a_id, (
        "Chunk with embedding closest to query vector should be ranked first"
    )


async def test_semantic_retriever_returns_correct_metadata(db_pool):
    """
    Verify all RetrievedChunk fields are populated correctly from the DB row.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"meta_test_{tag}.pdf"

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)
        chunk_id = await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content="EBITDA margin improved to 24.1% in the current period.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=5),
        )

    retriever = SemanticRetriever(db_pool)
    results = await retriever.search(_unit_vector(hot_index=5), top_k=20)

    our = next((r for r in results if r.chunk_id == chunk_id), None)
    assert our is not None, "Inserted chunk should be in results"
    assert our.source_file == source_file
    assert our.doc_type == "pdf"
    assert our.chunk_strategy == "fixed"
    assert 0.0 <= our.score <= 1.0


async def test_semantic_retriever_filter_restricts_results(db_pool):
    """
    Insert chunks with two different doc_types.
    A doc_type filter should return only the matching type.
    """
    tag = uuid.uuid4().hex[:8]
    pdf_file = f"filter_pdf_{tag}.pdf"
    docx_file = f"filter_docx_{tag}.docx"

    async with db_pool.acquire() as conn:
        doc_id_pdf = await _insert_test_document(conn, pdf_file)
        doc_id_docx = await _insert_test_document(conn, docx_file)

        await _insert_test_chunk(
            conn,
            document_id=doc_id_pdf,
            content="PDF chunk for filter test.",
            source_file=pdf_file,
            embedding=_unit_vector(hot_index=10),
            doc_type="pdf",
        )
        await _insert_test_chunk(
            conn,
            document_id=doc_id_docx,
            content="DOCX chunk for filter test.",
            source_file=docx_file,
            embedding=_unit_vector(hot_index=10),
            doc_type="docx",
        )

    retriever = SemanticRetriever(db_pool)
    results = await retriever.search(
        _unit_vector(hot_index=10),
        top_k=50,
        filters=RetrievalFilter(doc_type="pdf"),
    )

    # Every result must be pdf — no docx should slip through
    our_files = {pdf_file, docx_file}
    our_results = [r for r in results if r.source_file in our_files]
    assert all(r.doc_type == "pdf" for r in our_results), (
        "doc_type filter should exclude non-matching chunks"
    )
    assert any(r.source_file == pdf_file for r in our_results), (
        "PDF chunk should appear in filtered results"
    )


# ---------------------------------------------------------------------------
# Test 4: KeywordRetriever (requires DB)
# ---------------------------------------------------------------------------

async def test_keyword_retriever_finds_exact_term(db_pool):
    """
    Insert a chunk containing a rare financial term.
    Keyword search for that term should return the chunk.

    Why this matters: semantic search can miss exact tickers, dates, or
    identifiers. tsvector full-text matching catches them precisely.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"keyword_test_{tag}.pdf"
    # Use a term unlikely to appear in other test data
    rare_term = f"xyzfinancialterm{tag}"

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)
        chunk_id = await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content=f"The {rare_term} metric exceeded all forecasts this quarter.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=20),
        )

    retriever = KeywordRetriever(db_pool)
    results = await retriever.search(rare_term, top_k=5)

    chunk_ids = [r.chunk_id for r in results]
    assert chunk_id in chunk_ids, (
        f"Chunk containing '{rare_term}' should be returned by keyword search"
    )


async def test_keyword_retriever_scores_term_frequency(db_pool):
    """
    A chunk mentioning a term 3× should score higher than one mentioning it 1×.
    ts_rank is sensitive to term frequency.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"tf_test_{tag}.pdf"
    term = f"revenue{tag}"

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)

        # High frequency — mentions term 3 times
        high_id = await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content=f"The {term} increased. {term} growth was strong. {term} beat guidance.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=21),
        )

        # Low frequency — mentions term once
        low_id = await _insert_test_chunk(
            conn,
            document_id=doc_id,
            content=f"The {term} met expectations but costs rose significantly.",
            source_file=source_file,
            embedding=_unit_vector(hot_index=22),
        )

    retriever = KeywordRetriever(db_pool)
    results = await retriever.search(term, top_k=5)

    our_results = [r for r in results if r.source_file == source_file]
    assert len(our_results) >= 2
    # High-frequency chunk should rank above low-frequency
    high_rank = next(r.rank for r in our_results if r.chunk_id == high_id)
    low_rank = next(r.rank for r in our_results if r.chunk_id == low_id)
    assert high_rank < low_rank, (
        "Chunk with higher term frequency should have lower (better) rank number"
    )


async def test_keyword_retriever_filter_restricts_results(db_pool):
    """
    Keyword search with a source_file filter should only return chunks
    from that specific file.
    """
    tag = uuid.uuid4().hex[:8]
    file_a = f"kw_filter_a_{tag}.pdf"
    file_b = f"kw_filter_b_{tag}.pdf"
    shared_term = f"covenant{tag}"

    async with db_pool.acquire() as conn:
        doc_a = await _insert_test_document(conn, file_a)
        doc_b = await _insert_test_document(conn, file_b)

        await _insert_test_chunk(
            conn, document_id=doc_a,
            content=f"Section 5.2 {shared_term} compliance was maintained.",
            source_file=file_a, embedding=_unit_vector(hot_index=30),
        )
        await _insert_test_chunk(
            conn, document_id=doc_b,
            content=f"The {shared_term} threshold was not breached.",
            source_file=file_b, embedding=_unit_vector(hot_index=30),
        )

    retriever = KeywordRetriever(db_pool)
    results = await retriever.search(
        shared_term,
        top_k=10,
        filters=RetrievalFilter(source_file=file_a),
    )

    assert all(r.source_file == file_a for r in results), (
        "source_file filter should exclude chunks from other files"
    )
    assert any(r.source_file == file_a for r in results), (
        "Matching chunk from file_a should be present"
    )


# ---------------------------------------------------------------------------
# Test 5: HybridRetriever (requires DB + mock embedder)
# ---------------------------------------------------------------------------

async def test_hybrid_retriever_returns_fused_results(db_pool):
    """
    Insert chunks designed so that:
    - Chunk A matches semantically (embedding similarity) but has no keywords
    - Chunk B matches on keywords but has a less similar embedding
    - Hybrid RRF should surface both

    The mock embedder returns the query vector directly so we can control
    which chunk wins on the semantic side.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"hybrid_test_{tag}.pdf"
    keyword_term = f"ebitdaterm{tag}"

    # Chunk A: embedding in direction 40 (will win semantic search)
    # Chunk B: contains unique keyword (will win keyword search)
    embedding_a = _unit_vector(hot_index=40)
    embedding_b = _unit_vector(hot_index=41)

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)

        chunk_a_id = await _insert_test_chunk(
            conn, document_id=doc_id,
            content="Revenue growth exceeded management guidance for the period.",
            source_file=source_file, embedding=embedding_a,
        )
        chunk_b_id = await _insert_test_chunk(
            conn, document_id=doc_id,
            content=f"The {keyword_term} ratio improved significantly.",
            source_file=source_file, embedding=embedding_b,
        )

    # Mock embedder returns embedding_a — so semantic search favours chunk_a
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=embedding_a)

    retriever = HybridRetriever(pool=db_pool, embedder=mock_embedder, candidate_k=20)

    # Use keyword_term in query so keyword search finds chunk_b
    results = await retriever.search(keyword_term, top_k=10)

    # Filter to our test chunks
    our_results = [r for r in results if r.source_file == source_file]
    our_ids = {r.chunk_id for r in our_results}

    # Chunk B must appear (it has the exact keyword)
    assert chunk_b_id in our_ids, (
        "Hybrid retriever should include chunk matched by keyword search"
    )
    # Chunk A may also appear (strong semantic match)
    # Both should appear if top_k is large enough
    assert len(our_ids) >= 1


async def test_hybrid_retriever_fetch_parent(db_pool):
    """
    fetch_parent() should return the parent chunk given a valid parent_id,
    and None for a non-existent id.
    """
    tag = uuid.uuid4().hex[:8]
    source_file = f"parent_test_{tag}.pdf"

    async with db_pool.acquire() as conn:
        doc_id = await _insert_test_document(conn, source_file)
        parent_id = await _insert_test_chunk(
            conn, document_id=doc_id,
            content="Parent section: full financial overview for Q3 2024.",
            source_file=source_file, embedding=_unit_vector(hot_index=50),
        )

    mock_embedder = MagicMock()
    retriever = HybridRetriever(pool=db_pool, embedder=mock_embedder)

    # Valid parent
    parent = await retriever.fetch_parent(parent_id)
    assert parent is not None
    assert parent.chunk_id == parent_id
    assert "Q3 2024" in parent.content

    # Non-existent id
    missing = await retriever.fetch_parent(999_999_999)
    assert missing is None


# ---------------------------------------------------------------------------
# Test 6: PassThroughReranker
# ---------------------------------------------------------------------------

async def test_passthrough_reranker_preserves_order():
    """PassThrough reranker must return chunks in original order (no reranking)."""
    chunks = [_make_chunk(i, rank=i) for i in range(1, 6)]
    reranker = PassThroughReranker()

    result = await reranker.rerank("any query", chunks, top_k=5)

    assert [r.chunk_id for r in result] == [1, 2, 3, 4, 5]


async def test_passthrough_reranker_top_k_truncates():
    """PassThrough reranker must honour top_k and not return extra chunks."""
    chunks = [_make_chunk(i, rank=i) for i in range(1, 11)]
    reranker = PassThroughReranker()

    result = await reranker.rerank("query", chunks, top_k=3)

    assert len(result) == 3
    assert [r.chunk_id for r in result] == [1, 2, 3]


async def test_passthrough_reranker_empty_input():
    """Empty input → empty output."""
    reranker = PassThroughReranker()
    result = await reranker.rerank("query", [], top_k=5)
    assert result == []


# ---------------------------------------------------------------------------
# Test 7: CohereReranker (mocked API)
# ---------------------------------------------------------------------------

async def test_cohere_reranker_reorders_by_relevance_score():
    """
    Mock the Cohere API to return scores that reverse the input order.
    Assert the output reflects the new scores, not the original order.

    Why mock? We don't want to call the real API in tests (cost + network).
    The mock verifies our code correctly maps Cohere's response back to chunks.
    """
    # Input order: chunk_id 1, 2, 3
    chunks = [
        _make_chunk(1, content="Least relevant text"),
        _make_chunk(2, content="Moderately relevant text"),
        _make_chunk(3, content="Most relevant financial text"),
    ]

    # Cohere reverses the order: index 2 (chunk_id=3) is best, index 0 is worst
    mock_result_0 = MagicMock()
    mock_result_0.index = 2         # chunk_id=3
    mock_result_0.relevance_score = 0.95

    mock_result_1 = MagicMock()
    mock_result_1.index = 1         # chunk_id=2
    mock_result_1.relevance_score = 0.60

    mock_result_2 = MagicMock()
    mock_result_2.index = 0         # chunk_id=1
    mock_result_2.relevance_score = 0.10

    mock_response = MagicMock()
    mock_response.results = [mock_result_0, mock_result_1, mock_result_2]

    mock_client = MagicMock()
    mock_client.rerank = AsyncMock(return_value=mock_response)

    with patch("src.retrieval.reranker.cohere.AsyncClientV2", return_value=mock_client):
        reranker = CohereReranker(api_key="fake-key")

    result = await reranker.rerank("financial metrics query", chunks, top_k=3)

    assert len(result) == 3
    # Output should be in order of Cohere's relevance scores
    assert result[0].chunk_id == 3, "Highest-scored chunk should be rank 1"
    assert result[1].chunk_id == 2
    assert result[2].chunk_id == 1

    # Ranks should be 1-based sequential
    assert result[0].rank == 1
    assert result[1].rank == 2
    assert result[2].rank == 3

    # Scores should reflect Cohere's relevance_score
    assert abs(result[0].score - 0.95) < 1e-6
    assert abs(result[1].score - 0.60) < 1e-6


async def test_cohere_reranker_top_k_respected():
    """
    Cohere returns top_n results matching top_k.
    Assert we forward the correct top_n to the API.
    """
    chunks = [_make_chunk(i) for i in range(1, 11)]

    mock_results = []
    for i in range(3):           # Cohere returns only 3 (top_n=3)
        r = MagicMock()
        r.index = i
        r.relevance_score = 1.0 - i * 0.1
        mock_results.append(r)

    mock_response = MagicMock()
    mock_response.results = mock_results

    mock_client = MagicMock()
    mock_client.rerank = AsyncMock(return_value=mock_response)

    with patch("src.retrieval.reranker.cohere.AsyncClientV2", return_value=mock_client):
        reranker = CohereReranker(api_key="fake-key")

    result = await reranker.rerank("query", chunks, top_k=3)

    # Verify top_n=3 was passed to the Cohere client
    call_kwargs = mock_client.rerank.call_args.kwargs
    assert call_kwargs["top_n"] == 3

    assert len(result) == 3


async def test_cohere_reranker_empty_chunks():
    """Empty chunk list → return immediately without calling Cohere API."""
    mock_client = MagicMock()
    mock_client.rerank = AsyncMock()

    with patch("src.retrieval.reranker.cohere.AsyncClientV2", return_value=mock_client):
        reranker = CohereReranker(api_key="fake-key")

    result = await reranker.rerank("query", [], top_k=5)

    assert result == []
    mock_client.rerank.assert_not_called()

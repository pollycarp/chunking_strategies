"""
Retrieval evaluation metrics — Recall@K, Precision@K, MRR.

These are the standard metrics for measuring retrieval system quality.
They are pure functions (no DB, no API) and can be unit-tested precisely.

Definitions:
    retrieved_ids : list of chunk IDs returned by the retriever, in rank order
    relevant_ids  : set of chunk IDs considered correct for this query
    k             : the cutoff — "how many results did we give the user?"

Why these three metrics together?
    Recall@K alone → a retriever that returns every chunk scores 1.0 (useless)
    Precision@K alone → a retriever that returns 1 correct chunk scores 1.0 (ignores coverage)
    MRR alone → doesn't capture whether all relevant docs were found
    Together → they capture the quality vs noise vs ranking tradeoff
"""

from __future__ import annotations


def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Fraction of relevant chunks that appear in the top-k retrieved results.

    recall@k = |{relevant} ∩ {top-k retrieved}| / |{relevant}|

    Score of 1.0 → all relevant chunks were found in the top k.
    Score of 0.0 → no relevant chunk appeared in the top k.

    Returns 0.0 if relevant_ids is empty (undefined case).
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    """
    Fraction of the top-k retrieved results that are relevant.

    precision@k = |{relevant} ∩ {top-k retrieved}| / k

    Score of 1.0 → every result in the top k was relevant (no noise).
    Score of 0.0 → no relevant chunk in the top k.

    Returns 0.0 if k == 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> float:
    """
    1 / rank of the first relevant chunk in the retrieved list.

    Examples:
        Retrieved: [3, 1, 5, 2]  Relevant: {1}  →  1/2 = 0.5  (found at rank 2)
        Retrieved: [1, 3, 5, 2]  Relevant: {1}  →  1/1 = 1.0  (found at rank 1)
        Retrieved: [3, 5, 2, 4]  Relevant: {1}  →  0.0        (not found)

    Used to compute MRR — rewards systems that put the right answer first.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    query_results: list[tuple[list[int], set[int]]],
) -> float:
    """
    Mean Reciprocal Rank over a set of queries.

    MRR = (1/|Q|) × Σ  1/rank_i

    Parameters:
        query_results: list of (retrieved_ids, relevant_ids) pairs, one per query

    Returns 0.0 for an empty list.

    Example:
        query_results = [
            ([1, 2, 3], {1}),    # rank 1 → RR = 1.0
            ([3, 1, 2], {1}),    # rank 2 → RR = 0.5
            ([3, 2, 4], {1}),    # not found → RR = 0.0
        ]
        MRR = (1.0 + 0.5 + 0.0) / 3 = 0.5
    """
    if not query_results:
        return 0.0
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in query_results
    ]
    return sum(rr_scores) / len(rr_scores)


def average_recall_at_k(
    query_results: list[tuple[list[int], set[int]]],
    k: int,
) -> float:
    """
    Average Recall@K over multiple queries.
    Used to compute a single dataset-level recall metric.
    """
    if not query_results:
        return 0.0
    scores = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in query_results]
    return sum(scores) / len(scores)


def average_precision_at_k(
    query_results: list[tuple[list[int], set[int]]],
    k: int,
) -> float:
    """Average Precision@K over multiple queries."""
    if not query_results:
        return 0.0
    scores = [precision_at_k(retrieved, relevant, k) for retrieved, relevant in query_results]
    return sum(scores) / len(scores)

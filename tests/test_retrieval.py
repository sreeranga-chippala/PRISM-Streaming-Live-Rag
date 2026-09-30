from __future__ import annotations

from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.multi_query_retriever import MultiQueryRetriever


class FakeHybridSearch:
    """Deterministic hybrid-search double for retrieval tests."""

    def __init__(self):
        self.calls = []

        self.data = {
            "phone bluetooth": {
                "semantic_results": [
                    {
                        "vector_id": 1,
                        "chunk_id": "chunk_a",
                        "text": "Phone Bluetooth settings.",
                        "metadata": {"source": "phone.txt"},
                        "score": 0.90,
                    },
                    {
                        "vector_id": 2,
                        "chunk_id": "chunk_b",
                        "text": "Pair a phone with Bluetooth.",
                        "metadata": {"source": "pair.txt"},
                        "score": 0.80,
                    },
                ],
                "keyword_results": [
                    {
                        "vector_id": 1,
                        "chunk_id": "chunk_a",
                        "text": "Phone Bluetooth settings.",
                        "metadata": {"source": "phone.txt"},
                        "score": 1.0,
                    }
                ],
            },
            "tv bluetooth": {
                "semantic_results": [
                    {
                        "vector_id": 3,
                        "chunk_id": "chunk_c",
                        "text": "TV Bluetooth settings.",
                        "metadata": {"source": "tv.txt"},
                        "score": 0.91,
                    },
                    {
                        "vector_id": 1,
                        "chunk_id": "chunk_a",
                        "text": "Phone Bluetooth settings.",
                        "metadata": {"source": "phone.txt"},
                        "score": 0.70,
                    },
                ],
                "keyword_results": [
                    {
                        "vector_id": 3,
                        "chunk_id": "chunk_c",
                        "text": "TV Bluetooth settings.",
                        "metadata": {"source": "tv.txt"},
                        "score": 1.0,
                    }
                ],
            },
        }

    def search(
        self,
        query,
        semantic_top_k=None,
        keyword_top_k=None,
    ):
        self.calls.append(query)
        return self.data[query]


class FakeReranker:
    """Deterministic reranker double."""

    def __init__(self):
        self.calls = []

    def rerank(
        self,
        query,
        candidates,
        top_k=None,
    ):
        self.calls.append(
            {
                "query": query,
                "candidate_count": len(candidates),
                "top_k": top_k,
            }
        )

        results = []

        for index, candidate in enumerate(candidates):
            result = candidate.copy()
            result["reranker_score"] = float(
                len(candidates) - index
            )
            results.append(result)

        results.sort(
            key=lambda item: (
                -item["reranker_score"],
                item["chunk_id"],
            )
        )

        if top_k is not None:
            results = results[:top_k]

        for rank, result in enumerate(results, start=1):
            result["reranker_rank"] = rank

        return results


def build_retriever():
    hybrid = FakeHybridSearch()
    reranker = FakeReranker()

    retriever = MultiQueryRetriever(
        hybrid_search=hybrid,
        fusion=ReciprocalRankFusion(k=60),
        reranker=reranker,
        default_subquery_top_k=2,
        default_final_top_k=3,
    )

    return retriever, hybrid, reranker


def test_empty_subqueries():
    retriever, _, _ = build_retriever()

    try:
        retriever.retrieve_many([])
        raise AssertionError(
            "Expected ValueError for empty subqueries"
        )
    except ValueError:
        pass


def test_multi_query_retrieval_and_deduplication():
    retriever, hybrid, reranker = build_retriever()

    subqueries = [
        "phone bluetooth",
        "tv bluetooth",
    ]

    results = retriever.retrieve_many(
        subqueries
    )

    assert hybrid.calls == subqueries
    assert len(results) == 3

    chunk_ids = [
        result["chunk_id"]
        for result in results
    ]

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )

    by_id = {
        result["chunk_id"]: result
        for result in results
    }

    assert set(
        by_id["chunk_a"]["source_subqueries"]
    ) == set(subqueries)

    assert by_id["chunk_a"][
        "source_subquery_indices"
    ] == [0, 1]

    assert "rrf_score" in by_id["chunk_a"]
    assert "reranker_score" in by_id["chunk_a"]
    assert "reranker_rank" in by_id["chunk_a"]

    assert "retrieval_latency_ms" in by_id["chunk_a"]
    assert "fusion_latency_ms" in by_id["chunk_a"]
    assert "reranking_latency_ms" in by_id["chunk_a"]
    assert "total_latency_ms" in by_id["chunk_a"]

    assert (
        reranker.calls[0]["query"]
        == "phone bluetooth | tv bluetooth"
    )

    assert (
        reranker.calls[0]["candidate_count"]
        == 3
    )


def test_duplicate_subqueries_are_removed():
    retriever, hybrid, _ = build_retriever()

    results = retriever.retrieve_many(
        [
            "phone bluetooth",
            "phone bluetooth",
            "  phone bluetooth  ",
        ]
    )

    assert hybrid.calls == [
        "phone bluetooth"
    ]

    assert len(results) == 2


def test_retrieve_alias():
    retriever, _, _ = build_retriever()

    results = retriever.retrieve(
        ["tv bluetooth"],
        final_top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_c"


def test_invalid_subquery_input():
    retriever, _, _ = build_retriever()

    try:
        retriever.retrieve_many(
            "phone bluetooth"
        )
        raise AssertionError(
            "Expected TypeError for a single string"
        )
    except TypeError:
        pass


def main():
    test_empty_subqueries()
    test_multi_query_retrieval_and_deduplication()
    test_duplicate_subqueries_are_removed()
    test_retrieve_alias()
    test_invalid_subquery_input()

    print(
        "\nALL RETRIEVAL TESTS PASSED"
    )


if __name__ == "__main__":
    main()

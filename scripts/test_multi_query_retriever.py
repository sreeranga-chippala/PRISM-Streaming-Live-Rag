from __future__ import annotations

from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.multi_query_retriever import MultiQueryRetriever


class FakeHybridSearch:
    def __init__(self):
        self.calls = []
        self.data = {
            "phone bluetooth": {
                "semantic_results": [
                    {"vector_id": 1, "chunk_id": "a", "text": "Phone Bluetooth settings", "metadata": {}, "score": .90},
                    {"vector_id": 2, "chunk_id": "b", "text": "Pair phone Bluetooth", "metadata": {}, "score": .80},
                ],
                "keyword_results": [
                    {"vector_id": 1, "chunk_id": "a", "text": "Phone Bluetooth settings", "metadata": {}, "score": 1.0},
                ],
            },
            "tv bluetooth": {
                "semantic_results": [
                    {"vector_id": 3, "chunk_id": "c", "text": "TV Bluetooth settings", "metadata": {}, "score": .91},
                    {"vector_id": 1, "chunk_id": "a", "text": "Phone Bluetooth settings", "metadata": {}, "score": .70},
                ],
                "keyword_results": [
                    {"vector_id": 3, "chunk_id": "c", "text": "TV Bluetooth settings", "metadata": {}, "score": 1.0},
                ],
            },
        }

    def search(self, query, semantic_top_k=None, keyword_top_k=None):
        self.calls.append(query)
        return self.data[query]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k=None):
        self.calls.append((query, len(candidates), top_k))
        results = []
        for i, candidate in enumerate(candidates):
            item = candidate.copy()
            item["reranker_score"] = float(len(candidates) - i)
            results.append(item)
        results.sort(key=lambda x: (-x["reranker_score"], x["chunk_id"]))
        results = results[:top_k]
        for rank, item in enumerate(results, 1):
            item["reranker_rank"] = rank
        return results


def build():
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


def test_empty():
    retriever, _, _ = build()
    try:
        retriever.retrieve_many([])
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_multi_query_and_dedup():
    retriever, hybrid, reranker = build()
    results = retriever.retrieve_many(["phone bluetooth", "tv bluetooth"])

    assert hybrid.calls == ["phone bluetooth", "tv bluetooth"]
    assert len(results) == 3

    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))

    by_id = {r["chunk_id"]: r for r in results}
    assert set(by_id["a"]["source_subqueries"]) == {"phone bluetooth", "tv bluetooth"}
    assert by_id["a"]["source_subquery_indices"] == [0, 1]
    assert "rrf_score" in by_id["a"]
    assert "reranker_score" in by_id["a"]
    assert "reranker_rank" in by_id["a"]
    assert "total_latency_ms" in by_id["a"]
    assert reranker.calls[0][0] == "phone bluetooth | tv bluetooth"


def test_duplicate_queries_are_removed():
    retriever, hybrid, _ = build()
    results = retriever.retrieve_many([
        "phone bluetooth",
        "phone bluetooth",
        "  phone bluetooth  ",
    ])
    assert hybrid.calls == ["phone bluetooth"]
    assert len(results) == 2


def test_alias():
    retriever, _, _ = build()
    results = retriever.retrieve(["tv bluetooth"], final_top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] in {"a", "c"}


def main():
    test_empty()
    test_multi_query_and_dedup()
    test_duplicate_queries_are_removed()
    test_alias()
    print("\nALL MULTI-QUERY RETRIEVER TESTS PASSED")


if __name__ == "__main__":
    main()

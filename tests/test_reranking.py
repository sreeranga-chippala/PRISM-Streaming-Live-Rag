from __future__ import annotations

from src.retrieval.reranker import Reranker


class FakeCrossEncoder:
    """
    Deterministic CrossEncoder replacement for unit tests.

    The fake score is the number of query words that appear in
    the candidate text. This lets the test verify ranking logic
    without downloading or loading the real Hugging Face model.
    """

    def predict(
        self,
        pairs,
        batch_size=None,
        show_progress_bar=False,
    ):
        scores = []

        for query, text in pairs:

            query_words = set(
                query.lower().split()
            )

            text_words = set(
                text.lower().split()
            )

            overlap = len(
                query_words.intersection(
                    text_words
                )
            )

            scores.append(
                float(overlap)
            )

        return scores


def build_reranker():
    """
    Build a Reranker instance without loading the real
    CrossEncoder model.
    """

    reranker = object.__new__(
        Reranker
    )

    reranker.model_name = (
        "test-cross-encoder"
    )

    reranker.batch_size = 2

    reranker.model = (
        FakeCrossEncoder()
    )

    return reranker


def test_rerank_orders_by_score():
    reranker = build_reranker()

    candidates = [
        {
            "chunk_id": "low",
            "text": "camera battery",
            "metadata": {},
            "fusion_rank": 1,
        },
        {
            "chunk_id": "high",
            "text": (
                "phone bluetooth settings "
                "connections"
            ),
            "metadata": {},
            "fusion_rank": 2,
        },
        {
            "chunk_id": "medium",
            "text": "phone bluetooth",
            "metadata": {},
            "fusion_rank": 3,
        },
    ]

    results = reranker.rerank(
        query="phone bluetooth settings",
        candidates=candidates,
    )

    assert [
        result["chunk_id"]
        for result in results
    ] == [
        "high",
        "medium",
        "low",
    ]

    assert (
        results[0]["reranker_score"]
        > results[1]["reranker_score"]
    )

    assert (
        results[1]["reranker_score"]
        > results[2]["reranker_score"]
    )


def test_rerank_assigns_ranks():
    reranker = build_reranker()

    candidates = [
        {
            "chunk_id": "a",
            "text": "phone bluetooth",
            "metadata": {},
            "fusion_rank": 1,
        },
        {
            "chunk_id": "b",
            "text": "camera",
            "metadata": {},
            "fusion_rank": 2,
        },
    ]

    results = reranker.rerank(
        query="phone bluetooth",
        candidates=candidates,
    )

    assert (
        results[0]["reranker_rank"]
        == 1
    )

    assert (
        results[1]["reranker_rank"]
        == 2
    )


def test_rerank_top_k():
    reranker = build_reranker()

    candidates = [
        {
            "chunk_id": "a",
            "text": (
                "phone bluetooth settings"
            ),
            "metadata": {},
            "fusion_rank": 1,
        },
        {
            "chunk_id": "b",
            "text": "phone",
            "metadata": {},
            "fusion_rank": 2,
        },
        {
            "chunk_id": "c",
            "text": "camera",
            "metadata": {},
            "fusion_rank": 3,
        },
    ]

    results = reranker.rerank(
        query="phone bluetooth",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert [
        result["chunk_id"]
        for result in results
    ] == [
        "a",
        "b",
    ]

    assert [
        result["reranker_rank"]
        for result in results
    ] == [
        1,
        2,
    ]


def test_empty_candidates():
    reranker = build_reranker()

    results = reranker.rerank(
        query="phone bluetooth",
        candidates=[],
    )

    assert results == []


def test_invalid_query():
    reranker = build_reranker()

    try:

        reranker.rerank(
            query="",
            candidates=[],
        )

        raise AssertionError(
            "Expected ValueError for empty query"
        )

    except ValueError:
        pass


def test_invalid_top_k():
    reranker = build_reranker()

    candidates = [
        {
            "chunk_id": "a",
            "text": "phone",
            "metadata": {},
        }
    ]

    try:

        reranker.rerank(
            query="phone",
            candidates=candidates,
            top_k=0,
        )

        raise AssertionError(
            "Expected ValueError for top_k=0"
        )

    except ValueError:
        pass


def main():
    test_rerank_orders_by_score()
    test_rerank_assigns_ranks()
    test_rerank_top_k()
    test_empty_candidates()
    test_invalid_query()
    test_invalid_top_k()

    print(
        "\nALL RERANKING TESTS PASSED"
    )


if __name__ == "__main__":
    main()

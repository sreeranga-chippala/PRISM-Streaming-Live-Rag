from src.intelligence.query_decomposer import (
    QueryDecomposer,
)


def test_single_intent_query_remains_single():
    decomposer = QueryDecomposer()

    result = decomposer.decompose(
        "What is the cancellation policy?"
    )

    assert len(result.sub_queries) == 1
    assert result.sub_queries[0].query == (
        "What is the cancellation policy?"
    )


def test_and_creates_multiple_sub_queries():
    decomposer = QueryDecomposer()

    result = decomposer.decompose(
        "What is the cancellation policy and what is the pricing?"
    )

    assert len(result.sub_queries) == 2
    assert result.sub_queries[0].query == (
        "What is the cancellation policy"
    )
    assert result.sub_queries[1].query == (
        "what is the pricing?"
    )


def test_multiple_intents_are_classified():
    decomposer = QueryDecomposer()

    result = decomposer.decompose(
        "Tell me the cancellation policy and compare the pricing"
    )

    assert len(result.sub_queries) == 2
    assert result.sub_queries[0].intent == "information"
    assert result.sub_queries[1].intent == "comparison"


def test_comparison_query_is_detected():
    decomposer = QueryDecomposer()

    result = decomposer.decompose(
        "Compare the pricing and availability"
    )

    assert len(result.sub_queries) == 2


def test_original_query_is_preserved():
    decomposer = QueryDecomposer()

    query = "What is the policy and what is the price?"

    result = decomposer.decompose(query)

    assert result.original_query == query


def test_empty_query_rejected():
    decomposer = QueryDecomposer()

    try:
        decomposer.decompose("   ")
        assert False
    except ValueError:
        assert True
from __future__ import annotations

from pathlib import Path

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import Reranker


# =========================================================
# CONFIGURATION
# =========================================================

VECTOR_STORE_DIR = (
    "data/processed/vector_store"
)

EMBEDDING_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

TOP_K = 5


# =========================================================
# BUILD PERSON 2 RETRIEVAL PIPELINE
# =========================================================

def build_pipeline():
    """
    Build the complete Person 2 retrieval pipeline.

    Pipeline:

        Query
          ↓
        Embedding
          ↓
        Semantic Retrieval
          ↓
        Keyword Retrieval
          ↓
        Hybrid Search
          ↓
        RRF Fusion
          ↓
        Cross-Encoder Reranking
          ↓
        Final Results
    """

    print()
    print("=" * 80)
    print("BUILDING PERSON 2 RETRIEVAL PIPELINE")
    print("=" * 80)

    # -----------------------------------------------------
    # Embedding model
    # -----------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=32,
    )

    # -----------------------------------------------------
    # Persistent vector store
    # -----------------------------------------------------

    vector_store = VectorStore(
        VECTOR_STORE_DIR
    )

    # -----------------------------------------------------
    # Semantic retriever
    # -----------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=TOP_K,
    )

    # -----------------------------------------------------
    # Hybrid search
    # -----------------------------------------------------

    hybrid_search = HybridSearch(
        retriever=retriever,
        vector_store=vector_store,
        semantic_top_k=TOP_K,
        keyword_top_k=TOP_K,
    )

    # -----------------------------------------------------
    # RRF fusion
    # -----------------------------------------------------

    fusion = ReciprocalRankFusion(
        k=60
    )

    # -----------------------------------------------------
    # Cross-encoder reranker
    # -----------------------------------------------------

    reranker = Reranker(
        model_name=RERANKER_MODEL_NAME,
        batch_size=16,
    )

    print()
    print(
        "[PASS] Person 2 retrieval components "
        "initialized."
    )

    return (
        embedding_model,
        vector_store,
        retriever,
        hybrid_search,
        fusion,
        reranker,
    )


# =========================================================
# TEST VECTOR STORE
# =========================================================

def test_vector_store(
    vector_store,
):
    """
    Verify that the persistent vector store
    contains indexed data.
    """

    print()
    print("=" * 80)
    print("TEST 1 - VECTOR STORE")
    print("=" * 80)

    vector_count = (
        vector_store.count()
    )

    keyword_count = (
        vector_store.keyword_count()
    )

    print(
        f"FAISS vectors : {vector_count}"
    )

    print(
        f"SQLite/FTS5 records : {keyword_count}"
    )

    assert vector_count > 0, (
        "Vector store contains no vectors."
    )

    assert keyword_count > 0, (
        "FTS5 contains no keyword records."
    )

    assert vector_count == keyword_count, (
        "FAISS and FTS5 record counts "
        "are inconsistent."
    )

    print()
    print(
        "[PASS] Vector store contains "
        "synchronized indexed data."
    )


# =========================================================
# TEST SEMANTIC RETRIEVAL
# =========================================================

def test_semantic_retrieval(
    retriever,
):
    """
    Test semantic FAISS retrieval.
    """

    print()
    print("=" * 80)
    print("TEST 2 - SEMANTIC RETRIEVAL")
    print("=" * 80)

    query = (
        "How do I enable Bluetooth "
        "on my Samsung phone?"
    )

    results = retriever.retrieve(
        query=query,
        top_k=TOP_K,
    )

    assert results, (
        "Semantic retrieval returned no results."
    )

    print()
    print(
        f"Query: {query}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"Rank: {index}"
        )

        print(
            f"Score: "
            f"{result.get('score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{result.get('text')}"
        )

    print()
    print(
        "[PASS] Semantic retrieval returned results."
    )

    return results


# =========================================================
# TEST KEYWORD RETRIEVAL
# =========================================================

def test_keyword_retrieval(
    vector_store,
):
    """
    Test SQLite FTS5 keyword retrieval.
    """

    print()
    print("=" * 80)
    print("TEST 3 - KEYWORD RETRIEVAL")
    print("=" * 80)

    query = "Bluetooth Samsung Phone"

    results = vector_store.keyword_search(
        query=query,
        top_k=TOP_K,
    )

    assert results, (
        "Keyword retrieval returned no results."
    )

    print()
    print(
        f"Query: {query}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"Rank: {index}"
        )

        print(
            f"Score: "
            f"{result.get('score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{result.get('text')}"
        )

    print()
    print(
        "[PASS] Keyword retrieval returned results."
    )

    return results


# =========================================================
# TEST HYBRID SEARCH
# =========================================================

def test_hybrid_search(
    hybrid_search,
):
    """
    Test semantic + keyword retrieval together.
    """

    print()
    print("=" * 80)
    print("TEST 4 - HYBRID SEARCH")
    print("=" * 80)

    query = (
        "How do I enable Bluetooth "
        "on my Samsung phone?"
    )

    result = hybrid_search.search(
        query=query,
        semantic_top_k=TOP_K,
        keyword_top_k=TOP_K,
    )

    semantic_results = (
        result.get(
            "semantic_results",
            [],
        )
    )

    keyword_results = (
        result.get(
            "keyword_results",
            [],
        )
    )

    candidates = (
        result.get(
            "candidates",
            [],
        )
    )

    assert semantic_results, (
        "Hybrid search returned no semantic results."
    )

    assert keyword_results, (
        "Hybrid search returned no keyword results."
    )

    assert candidates, (
        "Hybrid search returned no candidates."
    )

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        f"Semantic results: "
        f"{len(semantic_results)}"
    )

    print(
        f"Keyword results: "
        f"{len(keyword_results)}"
    )

    print(
        f"Candidate pool: "
        f"{len(candidates)}"
    )

    print()
    print(
        "[PASS] Hybrid search returned "
        "semantic and keyword candidates."
    )

    return result


# =========================================================
# TEST RRF FUSION
# =========================================================

def test_fusion(
    fusion,
    hybrid_result,
):
    """
    Test Reciprocal Rank Fusion.
    """

    print()
    print("=" * 80)
    print("TEST 5 - RRF FUSION")
    print("=" * 80)

    fused_results = (
        fusion.fuse_hybrid_result(
            hybrid_result
        )
    )

    assert fused_results, (
        "RRF fusion returned no results."
    )

    print()

    for result in fused_results:

        print(
            f"Fusion rank: "
            f"{result.get('fusion_rank')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Retrieval sources: "
            f"{result.get('retrieval_sources')}"
        )

        print("-" * 80)

    # Fusion ranks must be sequential.
    expected_ranks = list(
        range(
            1,
            len(fused_results) + 1,
        )
    )

    actual_ranks = [
        result.get("fusion_rank")
        for result in fused_results
    ]

    assert actual_ranks == expected_ranks, (
        "Fusion ranks are not sequential."
    )

    print()
    print(
        "[PASS] RRF fusion produced "
        "ranked candidates."
    )

    return fused_results


# =========================================================
# TEST RERANKING
# =========================================================

def test_reranker(
    reranker,
    query,
    fused_results,
):
    """
    Test cross-encoder reranking.
    """

    print()
    print("=" * 80)
    print("TEST 6 - CROSS-ENCODER RERANKING")
    print("=" * 80)

    final_results = reranker.rerank(
        query=query,
        candidates=fused_results,
        top_k=TOP_K,
    )

    assert final_results, (
        "Reranker returned no results."
    )

    print()

    for result in final_results:

        print(
            f"Reranker rank: "
            f"{result.get('reranker_rank')}"
        )

        print(
            f"Reranker score: "
            f"{result.get('reranker_score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{result.get('text')}"
        )

        print("-" * 80)

    expected_ranks = list(
        range(
            1,
            len(final_results) + 1,
        )
    )

    actual_ranks = [
        result.get("reranker_rank")
        for result in final_results
    ]

    assert actual_ranks == expected_ranks, (
        "Reranker ranks are not sequential."
    )

    print()
    print(
        "[PASS] Cross-encoder reranking "
        "produced final ranked results."
    )

    return final_results


# =========================================================
# COMPLETE END-TO-END TEST
# =========================================================

def test_complete_pipeline(
    hybrid_search,
    fusion,
    reranker,
):
    """
    Test the complete Person 2 retrieval flow.

    Query
        ↓
    Hybrid Search
        ↓
    RRF Fusion
        ↓
    Reranker
        ↓
    Final Top-K
    """

    print()
    print("=" * 80)
    print("TEST 7 - COMPLETE PERSON 2 PIPELINE")
    print("=" * 80)

    query = (
        "How do I enable Bluetooth "
        "on my Samsung phone?"
    )

    # -----------------------------------------------------
    # Hybrid retrieval
    # -----------------------------------------------------

    hybrid_result = hybrid_search.search(
        query=query,
        semantic_top_k=TOP_K,
        keyword_top_k=TOP_K,
    )

    assert hybrid_result

    # -----------------------------------------------------
    # RRF fusion
    # -----------------------------------------------------

    fused_results = (
        fusion.fuse_hybrid_result(
            hybrid_result
        )
    )

    assert fused_results

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    final_results = reranker.rerank(
        query=query,
        candidates=fused_results,
        top_k=TOP_K,
    )

    assert final_results

    # -----------------------------------------------------
    # Display final pipeline
    # -----------------------------------------------------

    print()
    print(
        f"Query: {query}"
    )

    print()
    print(
        "FINAL RETRIEVAL RESULTS"
    )

    print()

    for result in final_results:

        print(
            f"Final rank: "
            f"{result.get('reranker_rank')}"
        )

        print(
            f"Reranker score: "
            f"{result.get('reranker_score')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Chunk ID: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Text: "
            f"{result.get('text')}"
        )

        print("-" * 80)

    # -----------------------------------------------------
    # Validate final output structure
    # -----------------------------------------------------

    required_fields = [
        "chunk_id",
        "text",
        "metadata",
        "rrf_score",
        "reranker_score",
        "reranker_rank",
    ]

    for result in final_results:

        for field in required_fields:

            assert field in result, (
                f"Missing required field: {field}"
            )

    print()
    print(
        "[PASS] Complete Person 2 retrieval "
        "pipeline works end-to-end."
    )

    return final_results


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 80)
    print(
        "PERSON 2 - FINAL RETRIEVAL PIPELINE TEST"
    )
    print("=" * 80)

    (
        embedding_model,
        vector_store,
        retriever,
        hybrid_search,
        fusion,
        reranker,
    ) = build_pipeline()

    try:

        # -------------------------------------------------
        # Test 1
        # -------------------------------------------------

        test_vector_store(
            vector_store
        )

        # -------------------------------------------------
        # Test 2
        # -------------------------------------------------

        test_semantic_retrieval(
            retriever
        )

        # -------------------------------------------------
        # Test 3
        # -------------------------------------------------

        test_keyword_retrieval(
            vector_store
        )

        # -------------------------------------------------
        # Test 4
        # -------------------------------------------------

        hybrid_result = (
            test_hybrid_search(
                hybrid_search
            )
        )

        # -------------------------------------------------
        # Test 5
        # -------------------------------------------------

        fused_results = test_fusion(
            fusion,
            hybrid_result,
        )

        # -------------------------------------------------
        # Test 6
        # -------------------------------------------------

        query = (
            "How do I enable Bluetooth "
            "on my Samsung phone?"
        )

        test_reranker(
            reranker,
            query,
            fused_results,
        )

        # -------------------------------------------------
        # Test 7
        # -------------------------------------------------

        test_complete_pipeline(
            hybrid_search,
            fusion,
            reranker,
        )

        # -------------------------------------------------
        # FINAL RESULT
        # -------------------------------------------------

        print()
        print("=" * 80)
        print(
            "ALL PERSON 2 RETRIEVAL TESTS PASSED"
        )
        print("=" * 80)

        print()
        print(
            "Person 2 retrieval pipeline is ready "
            "for integration."
        )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()
from pathlib import Path

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.multi_query_retriever import MultiQueryRetriever


VECTOR_STORE_DIR = Path("data/processed/vector_store")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SUBQUERY_TOP_K = 5
FINAL_TOP_K = 5
# ============================================================
# TEST QUERIES
# ============================================================

TEST_CASES = [

    {
        "query": "What are the rules for taking leave?",
        "expected_source": "02_Leave_Policy.pdf",
        "expected_evidence": [
            "leave",
        ],
    },

    {
        "query": "What are the rules for working from home?",
        "expected_source": "03_Work_From_Home_Policy.pdf",
        "expected_evidence": [
            "work from home",
            "WFH",
        ],
    },

    {
        "query": "What are the employee code of conduct requirements?",
        "expected_source": "04_Code_of_Conduct.pdf",
        "expected_evidence": [
            "conduct",
            "employee",
        ],
    },

    {
        "query": "What is the performance improvement process?",
        "expected_source": "05_Performance_Review_Policy.pdf",
        "expected_evidence": [
            "performance",
            "improvement",
        ],
    },

    {
        "query": "What benefits are provided to employees?",
        "expected_source": "06_Compensation_and_Benefits_Policy.pdf",
        "expected_evidence": [
            "benefits",
        ],
    },

    {
        "query": "How should an employee report a security incident?",
        "expected_source": "07_IT_and_Data_Security_Policy.pdf",
        "expected_evidence": [
            "incident",
            "security",
        ],
    },

    {
        "query": "How can an employee report workplace harassment?",
        "expected_source": "08_Prevention_of_Sexual_Harassment_Policy.pdf",
        "expected_evidence": [
            "harassment",
        ],
    },

    {
        "query": "What documents are required during employee onboarding?",
        "expected_source": "09_Onboarding_and_Separation_Policy.pdf",
        "expected_evidence": [
            "onboarding",
        ],
    },

    {
        "query": "What travel expenses can employees claim?",
        "expected_source": "10_Travel_and_Expense_Policy.pdf",
        "expected_evidence": [
            "travel",
            "expense",
        ],
    },

]


# ============================================================
# UNRELATED QUERIES
# ============================================================

UNRELATED_QUERIES = [

    "How do I enable Bluetooth on a Samsung phone?",

    "What is the weather in Bengaluru today?",

    "How do I cook pasta?",

]


# ============================================================
# HELPERS
# ============================================================

def print_result(
    rank: int,
    result: dict,
):
    """
    Print one retrieval result.
    """

    metadata = result.get(
        "metadata",
        {},
    )

    print()
    print(
        f"Result {rank}"
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    if "score" in result:

        print(
            f"Score              : "
            f"{result['score']:.4f}"
        )

    if "rerank_score" in result:

        print(
            f"Reranker score      : "
            f"{result['rerank_score']:.4f}"
        )

    if "rrf_score" in result:

        print(
            f"RRF score           : "
            f"{result['rrf_score']:.6f}"
        )

    # --------------------------------------------------------
    # Retrieval information
    # --------------------------------------------------------

    print(
        f"Chunk ID            : "
        f"{result.get('chunk_id')}"
    )

    print(
        f"Source              : "
        f"{metadata.get('source')}"
    )

    # --------------------------------------------------------
    # Page / row / record / sheet / slide
    # --------------------------------------------------------

    for field in (
        "page",
        "row",
        "record",
        "sheet",
        "slide",
    ):

        if field in metadata:

            print(
                f"{field.capitalize():20}: "
                f"{metadata[field]}"
            )

    # --------------------------------------------------------
    # Retrieval source information
    # --------------------------------------------------------

    if "retrieval_sources" in result:

        print(
            f"Retrieval sources   : "
            f"{result['retrieval_sources']}"
        )

    if "source_subquery" in result:

        print(
            f"Source subquery    : "
            f"{result['source_subquery']}"
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    print(
        "Text                : "
        f"{result.get('text', '')}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("THEME 4 - COMPLETE RETRIEVAL PIPELINE TEST")
    print("=" * 70)

    print()
    print(
        "Pipeline:"
    )

    print(
        "Query"
    )

    print(
        "  ↓"
    )

    print(
        "Hybrid Search"
    )

    print(
        "  ↓"
    )

    print(
        "Reciprocal Rank Fusion"
    )

    print(
        "  ↓"
    )

    print(
        "Cross-Encoder Reranking"
    )

    print(
        "  ↓"
    )

    print(
        "Final Results"
    )

    # ========================================================
    # LOAD EMBEDDING MODEL
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "INITIALIZING EMBEDDING MODEL"
    )
    print(
        "=" * 70
    )

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL_NAME,
        batch_size=32,
    )

    # ========================================================
    # LOAD VECTOR STORE
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "LOADING VECTOR STORE"
    )
    print(
        "=" * 70
    )

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    print()
    print(
        f"FAISS vectors       : "
        f"{vector_store.count()}"
    )

    print(
        f"SQLite records      : "
        f"{vector_store.metadata_count()}"
    )

    # ========================================================
    # INITIALIZE COMPLETE RETRIEVER
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "INITIALIZING MULTI-QUERY RETRIEVER"
    )
    print(
        "=" * 70
    )

    semantic_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=SUBQUERY_TOP_K,
    )
    
    hybrid_search = HybridSearch(
        retriever=semantic_retriever,
        vector_store=vector_store,
        semantic_top_k=SUBQUERY_TOP_K,
        keyword_top_k=SUBQUERY_TOP_K,
    )

    fusion = ReciprocalRankFusion()

    reranker = Reranker(
        model_name=RERANKER_MODEL_NAME,
    )

    retriever = MultiQueryRetriever(
        hybrid_search=hybrid_search,
        fusion=fusion,
        reranker=reranker,
        default_subquery_top_k=SUBQUERY_TOP_K,
        default_final_top_k=FINAL_TOP_K,
    )

    # ========================================================
    # POSITIVE TESTS
    # ========================================================

    passed = 0
    failed = 0

    print()
    print("=" * 70)
    print("POSITIVE RETRIEVAL TESTS")
    print("=" * 70)

    for test_number, test_case in enumerate(
        TEST_CASES,
        start=1,
    ):

        query = test_case["query"]

        expected_source = (
            test_case["expected_source"]
        )

        expected_evidence = (
            test_case["expected_evidence"]
        )

        print()
        print("-" * 70)

        print(
            f"TEST {test_number}: "
            f"{query}"
        )

        print(
            f"Expected source: "
            f"{expected_source}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # COMPLETE RETRIEVAL
        # ----------------------------------------------------

        results = retriever.retrieve(
            [query],
            subquery_top_k=SUBQUERY_TOP_K,
            final_top_k=FINAL_TOP_K,
        )

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not results:

            print(
                "[FAIL] No retrieval results returned."
            )

            failed += 1

            continue

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print_result(
                rank,
                result,
            )

        # ----------------------------------------------------
        # CHECK TOP RESULT
        # ----------------------------------------------------

        top_result = results[0]

        top_metadata = top_result.get(
            "metadata",
            {},
        )

        top_source = top_metadata.get(
            "source"
        )

        # ----------------------------------------------------
        # SOURCE CHECK
        # ----------------------------------------------------

        if top_source == expected_source:

            print()
            print(
                "[PASS] Correct source "
                "is the top final result."
            )

            passed += 1

        else:

            print()
            print(
                "[FAIL] Expected source "
                "was not the top final result."
            )

            print(
                f"Expected: "
                f"{expected_source}"
            )

            print(
                f"Received: "
                f"{top_source}"
            )

            failed += 1

        # ----------------------------------------------------
        # EVIDENCE CHECK
        # ----------------------------------------------------

        combined_text = " ".join(
            result.get(
                "text",
                "",
            ).lower()
            for result in results
        )

        missing_evidence = []

        for evidence in expected_evidence:

            if evidence.lower() not in combined_text:

                missing_evidence.append(
                    evidence
                )

        if not missing_evidence:

            print(
                "[PASS] Expected evidence "
                "found in retrieved results."
            )

        else:

            print(
                "[FAIL] Missing expected evidence:"
            )

            for evidence in missing_evidence:

                print(
                    f"  - {evidence}"
                )

            failed += 1

    # ========================================================
    # UNRELATED QUERY TESTS
    # ========================================================

    print()
    print("=" * 70)
    print("UNRELATED QUERY TESTS")
    print("=" * 70)

    for query in UNRELATED_QUERIES:

        print()
        print("-" * 70)

        print(
            f"QUERY: {query}"
        )

        print("-" * 70)

        results = retriever.retrieve(
            [query],
            subquery_top_k=SUBQUERY_TOP_K,
            final_top_k=FINAL_TOP_K,
        )

        if not results:

            print(
                "[PASS] No results returned "
                "for unrelated query."
            )

            passed += 1

            continue

        top_result = results[0]

        # ----------------------------------------------------
        # Reranker score
        # ----------------------------------------------------

        rerank_score = top_result.get(
            "rerank_score"
        )

        if rerank_score is not None:

            print(
                f"Top reranker score: "
                f"{rerank_score:.4f}"
            )

        else:

            print(
                f"Top score: "
                f"{top_result.get('score', 0.0):.4f}"
            )

        metadata = top_result.get(
            "metadata",
            {},
        )

        print(
            f"Top source: "
            f"{metadata.get('source')}"
        )

        print(
            f"Top text: "
            f"{top_result.get('text', '')}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT use the old 0.15 semantic threshold here.
        #
        # The cross-encoder score has a different scale from
        # the FAISS cosine similarity score.
        #
        # For now we only report the result.
        # ----------------------------------------------------

        print(
            "[INFO] Unrelated-query result "
            "reported; no arbitrary reranker "
            "threshold is applied."
        )

    # ========================================================
    # CLOSE VECTOR STORE
    # ========================================================

    vector_store.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("RETRIEVAL PIPELINE TEST SUMMARY")
    print("=" * 70)

    print(
        f"Passed checks : {passed}"
    )

    print(
        f"Failed checks : {failed}"
    )

    if failed == 0:

        print()
        print(
            "ALL RETRIEVAL TESTS PASSED."
        )

        print(
            "Hybrid search + RRF + "
            "cross-encoder reranking "
            "is working end-to-end."
        )

    else:

        print()
        print(
            f"RETRIEVAL TEST COMPLETED "
            f"WITH {failed} FAILURE(S)."
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
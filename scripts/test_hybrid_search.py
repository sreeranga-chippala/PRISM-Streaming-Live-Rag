from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch


VECTOR_STORE_DIR = (
    "data/processed/vector_store"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


def print_semantic_results(
    results: list[dict],
):
    print("\n" + "-" * 70)
    print("SEMANTIC RESULTS")
    print("-" * 70)

    if not results:
        print("No semantic results found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {rank}"
        )

        print(
            f"Chunk ID       : "
            f"{result['chunk_id']}"
        )

        print(
            f"Semantic score : "
            f"{result['score']:.6f}"
        )

        print(
            f"Text           : "
            f"{result['text']}"
        )


def print_keyword_results(
    results: list[dict],
):
    print("\n" + "-" * 70)
    print("KEYWORD RESULTS")
    print("-" * 70)

    if not results:
        print("No keyword results found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nResult {rank}"
        )

        print(
            f"Chunk ID       : "
            f"{result['chunk_id']}"
        )

        print(
            f"Keyword score  : "
            f"{result['score']:.6f}"
        )

        print(
            f"Keyword rank   : "
            f"{result['keyword_rank']:.6f}"
        )

        print(
            f"Text           : "
            f"{result['text']}"
        )


def print_candidate_results(
    results: list[dict],
):
    print("\n" + "-" * 70)
    print("HYBRID CANDIDATE POOL")
    print("-" * 70)

    if not results:
        print("No hybrid candidates found.")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\nCandidate {rank}"
        )

        print(
            f"Chunk ID          : "
            f"{result['chunk_id']}"
        )

        print(
            f"Semantic score    : "
            f"{result['semantic_score']}"
        )

        print(
            f"Semantic rank     : "
            f"{result['semantic_rank']}"
        )

        print(
            f"Keyword score     : "
            f"{result['keyword_score']}"
        )

        print(
            f"Keyword rank      : "
            f"{result['keyword_rank']}"
        )

        print(
            f"Retrieval sources : "
            f"{result['retrieval_sources']}"
        )

        print(
            f"Text              : "
            f"{result['text']}"
        )


def main():

    print("=" * 70)
    print("HYBRID SEARCH TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name=EMBEDDING_MODEL,
        batch_size=32,
    )

    # ---------------------------------------------------------
    # Open vector store
    # ---------------------------------------------------------

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    print(
        f"\nFAISS vectors : "
        f"{vector_store.count()}"
    )

    print(
        f"SQLite records: "
        f"{vector_store.metadata_count()}"
    )

    print(
        f"FTS5 records  : "
        f"{vector_store.keyword_count()}"
    )

    # ---------------------------------------------------------
    # Create semantic retriever
    # ---------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=5,
    )

    # ---------------------------------------------------------
    # Create hybrid search
    # ---------------------------------------------------------

    hybrid_search = HybridSearch(
        retriever=retriever,
        vector_store=vector_store,
        semantic_top_k=5,
        keyword_top_k=5,
    )

    # ---------------------------------------------------------
    # Test queries
    # ---------------------------------------------------------

    queries = [
        "How do I enable Bluetooth on a Samsung phone?",
        "How can I turn on WiFi on a Samsung phone?",
        "What are the steps to turn Bluetooth on?",
    ]

    for query in queries:

        print("\n\n" + "=" * 70)

        print(
            f"QUERY: {query}"
        )

        print("=" * 70)

        result = hybrid_search.search(
            query
        )

        # -----------------------------------------------------
        # Semantic results
        # -----------------------------------------------------

        print_semantic_results(
            result[
                "semantic_results"
            ]
        )

        # -----------------------------------------------------
        # Keyword results
        # -----------------------------------------------------

        print_keyword_results(
            result[
                "keyword_results"
            ]
        )

        # -----------------------------------------------------
        # Hybrid candidate pool
        # -----------------------------------------------------

        print_candidate_results(
            result[
                "candidates"
            ]
        )

    # ---------------------------------------------------------
    # Close vector store
    # ---------------------------------------------------------

    vector_store.close()

    print("\n" + "=" * 70)
    print("HYBRID SEARCH TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
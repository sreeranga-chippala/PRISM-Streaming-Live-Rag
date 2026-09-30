from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion


VECTOR_STORE_DIR = (
    "data/processed/vector_store"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


def print_fused_results(
    results: list[dict],
):
    print("\n" + "-" * 70)
    print("RRF FUSED RESULTS")
    print("-" * 70)

    if not results:

        print(
            "No fused results found."
        )

        return

    for result in results:

        print(
            f"\nFusion Rank     : "
            f"{result['fusion_rank']}"
        )

        print(
            f"Chunk ID        : "
            f"{result['chunk_id']}"
        )

        print(
            f"RRF Score       : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Retrieval ranks : "
            f"{result['retrieval_ranks']}"
        )

        print(
            f"Sources         : "
            f"{result['retrieval_sources']}"
        )

        print(
            f"Text            : "
            f"{result['text']}"
        )


def main():

    print("=" * 70)
    print("RECIPROCAL RANK FUSION TEST")
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
    # Create retriever
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
    # Create RRF fusion
    # ---------------------------------------------------------

    fusion = ReciprocalRankFusion(
        k=60,
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

        # -----------------------------------------------------
        # Hybrid retrieval
        # -----------------------------------------------------

        hybrid_result = (
            hybrid_search.search(
                query
            )
        )

        # -----------------------------------------------------
        # RRF fusion
        # -----------------------------------------------------

        fused_results = (
            fusion.fuse_hybrid_result(
                hybrid_result,
                semantic_weight=1.0,
                keyword_weight=1.0,
            )
        )

        print_fused_results(
            fused_results
        )

    # ---------------------------------------------------------
    # Manual RRF sanity test
    # ---------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("MANUAL RRF SANITY TEST")
    print("=" * 70)

    semantic_results = [
        {
            "chunk_id": "A",
            "text": "Document A",
            "score": 0.95,
        },
        {
            "chunk_id": "B",
            "text": "Document B",
            "score": 0.90,
        },
        {
            "chunk_id": "C",
            "text": "Document C",
            "score": 0.80,
        },
    ]

    keyword_results = [
        {
            "chunk_id": "B",
            "text": "Document B",
            "score": 0.95,
        },
        {
            "chunk_id": "A",
            "text": "Document A",
            "score": 0.90,
        },
        {
            "chunk_id": "D",
            "text": "Document D",
            "score": 0.80,
        },
    ]

    manual_results = fusion.fuse(
        result_lists=[
            semantic_results,
            keyword_results,
        ],
        weights=[
            1.0,
            1.0,
        ],
    )

    for result in manual_results:

        print(
            f"\nFusion Rank : "
            f"{result['fusion_rank']}"
        )

        print(
            f"Chunk ID    : "
            f"{result['chunk_id']}"
        )

        print(
            f"RRF Score   : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Ranks       : "
            f"{result['retrieval_ranks']}"
        )

    # ---------------------------------------------------------
    # Close vector store
    # ---------------------------------------------------------

    vector_store.close()

    print("\n" + "=" * 70)
    print("RRF FUSION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
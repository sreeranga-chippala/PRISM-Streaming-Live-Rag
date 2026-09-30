from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import Reranker


VECTOR_STORE_DIR = (
    "data/processed/vector_store"
)

EMBEDDING_MODEL = (
    "all-MiniLM-L6-v2"
)


def print_results(
    results: list[dict],
):
    print("\n" + "-" * 70)
    print("RERANKED RESULTS")
    print("-" * 70)

    if not results:

        print(
            "No results found."
        )

        return

    for result in results:

        print(
            f"\nFinal Rank       : "
            f"{result['reranker_rank']}"
        )

        print(
            f"Chunk ID         : "
            f"{result['chunk_id']}"
        )

        print(
            f"Reranker score   : "
            f"{result['reranker_score']:.6f}"
        )

        print(
            f"Fusion rank      : "
            f"{result.get('fusion_rank')}"
        )

        print(
            f"RRF score        : "
            f"{result.get('rrf_score')}"
        )

        print(
            f"Text             : "
            f"{result['text']}"
        )


def main():

    print("=" * 70)
    print("RERANKER TEST")
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
    # Semantic retriever
    # ---------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=5,
    )

    # ---------------------------------------------------------
    # Hybrid search
    # ---------------------------------------------------------

    hybrid_search = HybridSearch(
        retriever=retriever,
        vector_store=vector_store,
        semantic_top_k=5,
        keyword_top_k=5,
    )

    # ---------------------------------------------------------
    # RRF fusion
    # ---------------------------------------------------------

    fusion = ReciprocalRankFusion(
        k=60,
    )

    # ---------------------------------------------------------
    # Reranker
    # ---------------------------------------------------------

    reranker = Reranker(
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        batch_size=16,
    )

    # ---------------------------------------------------------
    # Queries
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

        print(
            f"\nCandidates before "
            f"reranking: "
            f"{len(fused_results)}"
        )

        # -----------------------------------------------------
        # Reranking
        # -----------------------------------------------------

        reranked_results = (
            reranker.rerank(
                query=query,
                candidates=fused_results,
                top_k=5,
            )
        )

        print_results(
            reranked_results
        )

    # ---------------------------------------------------------
    # Close vector store
    # ---------------------------------------------------------

    vector_store.close()

    print("\n" + "=" * 70)
    print("RERANKER TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
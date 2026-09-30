from pathlib import Path

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.chunker import Chunk
from src.retrieval.retriever import Retriever


# ============================================================
# CONFIGURATION
# ============================================================

TEST_STORE_DIR = Path(
    "data/processed/test_retriever_store"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RETRIEVER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )

    # --------------------------------------------------------
    # STEP 2: TEST CHUNKS
    # --------------------------------------------------------

    chunks = [
        Chunk(
            text=(
                "Machine learning allows computers "
                "to learn patterns from data."
            ),
            metadata={
                "source": "test_document.txt",
                "chunk_id": "retriever_chunk_1",
            },
        ),

        Chunk(
            text=(
                "Deep learning uses neural networks "
                "with multiple layers."
            ),
            metadata={
                "source": "test_document.txt",
                "chunk_id": "retriever_chunk_2",
            },
        ),

        Chunk(
            text=(
                "Python is a popular programming "
                "language for machine learning."
            ),
            metadata={
                "source": "test_document.txt",
                "chunk_id": "retriever_chunk_3",
            },
        ),

        Chunk(
            text=(
                "The weather forecast predicts "
                "heavy rain tomorrow."
            ),
            metadata={
                "source": "test_document.txt",
                "chunk_id": "retriever_chunk_4",
            },
        ),
    ]

    # --------------------------------------------------------
    # STEP 3: CREATE EMBEDDINGS
    # --------------------------------------------------------

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = (
        embedding_model.encode_batch(
            texts
        )
    )

    print()
    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # STEP 4: VECTOR STORE
    # --------------------------------------------------------

    store = VectorStore(
        index_dir=TEST_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    store.clear()

    added = store.add(
        embeddings,
        chunks,
    )

    print(
        f"Vectors added: {added}"
    )

    if added != len(chunks):

        raise RuntimeError(
            "Not all test chunks were added."
        )

    # --------------------------------------------------------
    # STEP 5: CREATE RETRIEVER
    # --------------------------------------------------------

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=store,
        default_top_k=3,
    )

    print()
    print(
        "Retriever created successfully."
    )

    # --------------------------------------------------------
    # STEP 6: TEST QUERY
    # --------------------------------------------------------

    query = (
        "How do computers learn from data?"
    )

    print()
    print(
        f"Query: {query}"
    )

    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    # --------------------------------------------------------
    # STEP 7: DISPLAY RESULTS
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RETRIEVAL RESULTS")
    print("-" * 70)

    for index, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"Result {index}"
        )

        print(
            f"Score    : "
            f"{result['score']:.4f}"
        )

        print(
            f"Chunk ID : "
            f"{result['chunk_id']}"
        )

        print(
            f"Source   : "
            f"{result['metadata'].get('source')}"
        )

        print(
            f"Text     : "
            f"{result['text']}"
        )

        print(
            f"Query    : "
            f"{result['query']}"
        )

    # --------------------------------------------------------
    # STEP 8: VALIDATE RESULTS
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "Retriever returned no results."
        )

    if len(results) > 3:

        raise RuntimeError(
            "Retriever returned more than top_k results."
        )

    # The machine-learning chunk should be highly relevant
    # to this query.

    if (
        results[0]["chunk_id"]
        != "retriever_chunk_1"
    ):

        raise RuntimeError(
            "Unexpected top retrieval result."
        )

    print()
    print(
        "Basic retrieval test: PASSED"
    )

    # --------------------------------------------------------
    # STEP 9: TEST retrieve_texts()
    # --------------------------------------------------------

    texts = retriever.retrieve_texts(
        query=(
            "What is deep learning?"
        ),
        top_k=2,
    )

    print()
    print(
        "Text-only retrieval:"
    )

    for text in texts:

        print(
            f"  - {text}"
        )

    if not texts:

        raise RuntimeError(
            "retrieve_texts() returned no results."
        )

    print()
    print(
        "Text retrieval test: PASSED"
    )

    # --------------------------------------------------------
    # STEP 10: TEST SCORE FILTER
    # --------------------------------------------------------

    filtered_results = (
        retriever.retrieve(
            query=(
                "How do computers learn "
                "patterns from data?"
            ),
            top_k=4,
            min_score=0.50,
        )
    )

    print()
    print(
        "Results with score >= 0.50:"
    )

    for result in filtered_results:

        print(
            f"  {result['score']:.4f} "
            f"-> {result['chunk_id']}"
        )

    for result in filtered_results:

        if result["score"] < 0.50:

            raise RuntimeError(
                "Score filtering failed."
            )

    print()
    print(
        "Score filtering test: PASSED"
    )

    # --------------------------------------------------------
    # STEP 11: TEST EMPTY QUERY
    # --------------------------------------------------------

    try:

        retriever.retrieve(
            query="   "
        )

        raise RuntimeError(
            "Empty query should have raised "
            "ValueError."
        )

    except ValueError:

        print(
            "Empty query validation: PASSED"
        )

    # --------------------------------------------------------
    # STEP 12: CLOSE
    # --------------------------------------------------------

    store.close()

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL RETRIEVER TESTS PASSED")
    print("=" * 70)

    print()
    print("Verified:")
    print("  [PASS] Query embedding")
    print("  [PASS] FAISS retrieval")
    print("  [PASS] Top-K retrieval")
    print("  [PASS] Metadata preservation")
    print("  [PASS] Score filtering")
    print("  [PASS] Text-only retrieval")
    print("  [PASS] Query validation")

    print()
    print(
        "Retriever is ready for the next stage."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
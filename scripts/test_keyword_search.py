from src.retrieval.vector_store import VectorStore


VECTOR_STORE_DIR = "data/processed/test_vector_store"


def main():
    print("=" * 70)
    print("FTS5 KEYWORD SEARCH TEST")
    print("=" * 70)

    store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=384,
    )

    print(
        f"\nFAISS vectors : {store.count()}"
    )

    print(
        f"SQLite records: {store.metadata_count()}"
    )

    print(
        f"FTS5 records  : {store.keyword_count()}"
    )

    # ---------------------------------------------------------
    # Test 1
    # ---------------------------------------------------------

    query = "machine learning"

    print("\n" + "-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    results = store.keyword_search(
        query,
        top_k=5,
    )

    if not results:
        print("No results found.")

    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                f"Score    : "
                f"{result['score']:.6f}"
            )

            print(
                f"Chunk ID : "
                f"{result['chunk_id']}"
            )

            print(
                f"Text     : "
                f"{result['text']}"
            )

    # ---------------------------------------------------------
    # Test 2
    # ---------------------------------------------------------

    query = "neural networks"

    print("\n" + "-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    results = store.keyword_search(
        query,
        top_k=5,
    )

    if not results:
        print("No results found.")

    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                f"Score    : "
                f"{result['score']:.6f}"
            )

            print(
                f"Chunk ID : "
                f"{result['chunk_id']}"
            )

            print(
                f"Text     : "
                f"{result['text']}"
            )

    # ---------------------------------------------------------
    # Test 3
    # ---------------------------------------------------------

    query = "heavy rain"

    print("\n" + "-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    results = store.keyword_search(
        query,
        top_k=5,
    )

    if not results:
        print("No results found.")

    else:

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(
                f"\nResult {index}"
            )

            print(
                f"Score    : "
                f"{result['score']:.6f}"
            )

            print(
                f"Chunk ID : "
                f"{result['chunk_id']}"
            )

            print(
                f"Text     : "
                f"{result['text']}"
            )

    # ---------------------------------------------------------
    # Test 4: Unknown keyword
    # ---------------------------------------------------------

    query = "quantum computing"

    print("\n" + "-" * 70)
    print(f"QUERY: {query}")
    print("-" * 70)

    results = store.keyword_search(
        query,
        top_k=5,
    )

    if not results:

        print(
            "No results found: PASSED"
        )

    else:

        print(
            "Unexpected results:"
        )

        for result in results:

            print(
                result["text"]
            )

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    store.close()

    print("\n" + "=" * 70)
    print("FTS5 KEYWORD SEARCH TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.retrieval.chunker import Chunk
from src.retrieval.vector_store import VectorStore


def create_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "source": "failure_test.txt",
        },
    )


def print_counts(
    store: VectorStore,
    label: str,
) -> None:
    print()
    print(label)
    print("-" * 50)

    print(
        f"FAISS : {store.count()}"
    )

    print(
        f"SQLite: {store.metadata_count()}"
    )

    print(
        f"FTS5  : {store.keyword_count()}"
    )


def check_consistency(
    store: VectorStore,
) -> bool:

    faiss_count = store.count()
    sqlite_count = store.metadata_count()
    fts_count = store.keyword_count()

    print()
    print("Consistency check:")
    print(
        f"FAISS={faiss_count}, "
        f"SQLite={sqlite_count}, "
        f"FTS5={fts_count}"
    )

    return (
        faiss_count
        == sqlite_count
        == fts_count
    )


def test_normal_add() -> None:

    print()
    print("=" * 70)
    print("TEST 1: NORMAL ADD")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        store = VectorStore(
            index_dir=Path(temp_dir),
            dimension=4,
        )

        embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        chunks = [
            create_chunk(
                "chunk_1",
                "Bluetooth connection instructions",
            ),
            create_chunk(
                "chunk_2",
                "WiFi connection instructions",
            ),
        ]

        added = store.add(
            embeddings,
            chunks,
        )

        print(
            f"\nVectors added: {added}"
        )

        print_counts(
            store,
            "Store after normal add",
        )

        if not check_consistency(store):

            raise AssertionError(
                "Normal add produced "
                "an inconsistent store."
            )

        if added != 2:

            raise AssertionError(
                f"Expected 2 vectors, "
                f"but added {added}."
            )

        print(
            "\n[PASS] Normal add "
            "is consistent."
        )

        store.close()


def test_sqlite_failure_after_faiss() -> None:

    print()
    print("=" * 70)
    print(
        "TEST 2: SQLITE FAILURE "
        "AFTER FAISS INSERT"
    )
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        store = VectorStore(
            index_dir=Path(temp_dir),
            dimension=4,
        )

        # ---------------------------------------------------------
        # First create a valid store containing one vector.
        # ---------------------------------------------------------

        initial_embedding = np.array(
            [
                [1.0, 0.0, 0.0, 0.0]
            ],
            dtype=np.float32,
        )

        initial_chunk = [
            create_chunk(
                "initial_chunk",
                "Initial valid document",
            )
        ]

        store.add(
            initial_embedding,
            initial_chunk,
        )

        print_counts(
            store,
            "Before forced SQLite failure",
        )

        if not check_consistency(store):

            raise AssertionError(
                "Store was already inconsistent "
                "before the failure test."
            )

        # ---------------------------------------------------------
        # Save the state before the failed operation.
        # ---------------------------------------------------------

        faiss_before = store.count()
        sqlite_before = store.metadata_count()
        fts_before = store.keyword_count()

        # ---------------------------------------------------------
        # Save the real SQLite executemany method.
        # ---------------------------------------------------------

        original_executemany = (
            store.connection.executemany
        )

        # ---------------------------------------------------------
        # Replace executemany with a function that deliberately
        # fails.
        #
        # IMPORTANT:
        #
        # This is called AFTER FAISS has already received the
        # vectors because VectorStore.add() calls:
        #
        #     self.index.add_with_ids(...)
        #
        # before:
        #
        #     self.connection.executemany(...)
        # ---------------------------------------------------------

        def failing_executemany(
            *args,
            **kwargs,
        ):

            raise RuntimeError(
                "INTENTIONAL TEST FAILURE: "
                "SQLite insertion failed."
            )

        # Python's sqlite connection methods are implemented in
        # C and cannot always be replaced directly.
        #
        # Therefore this test uses a proxy connection object.

        class ConnectionProxy:

            def __init__(
                self,
                real_connection,
            ):
                self.real_connection = (
                    real_connection
                )

            def executemany(
                self,
                *args,
                **kwargs,
            ):

                return failing_executemany(
                    *args,
                    **kwargs,
                )

            def __getattr__(
                self,
                name,
            ):

                return getattr(
                    self.real_connection,
                    name,
                )

        # Replace the store's connection temporarily.
        store.connection = ConnectionProxy(
            store.connection
        )

        # ---------------------------------------------------------
        # Try adding another vector.
        # ---------------------------------------------------------

        failing_embedding = np.array(
            [
                [0.0, 1.0, 0.0, 0.0]
            ],
            dtype=np.float32,
        )

        failing_chunk = [
            create_chunk(
                "failing_chunk",
                "This document should trigger "
                "the SQLite failure.",
            )
        ]

        failure_occurred = False

        try:

            store.add(
                failing_embedding,
                failing_chunk,
            )

        except RuntimeError as error:

            failure_occurred = True

            print()
            print(
                "Expected failure occurred:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

        finally:

            # Restore the original connection.
            store.connection = (
                store.connection.real_connection
            )

        if not failure_occurred:

            raise AssertionError(
                "The intentional SQLite "
                "failure did not occur."
            )

        # ---------------------------------------------------------
        # Check the state AFTER failure.
        # ---------------------------------------------------------

        print_counts(
            store,
            "Store after forced SQLite failure",
        )

        faiss_after = store.count()
        sqlite_after = store.metadata_count()
        fts_after = store.keyword_count()

        print()
        print("Before failure:")
        print(
            f"FAISS : {faiss_before}"
        )
        print(
            f"SQLite: {sqlite_before}"
        )
        print(
            f"FTS5  : {fts_before}"
        )

        print()
        print("After failure:")
        print(
            f"FAISS : {faiss_after}"
        )
        print(
            f"SQLite: {sqlite_after}"
        )
        print(
            f"FTS5  : {fts_after}"
        )

        # ---------------------------------------------------------
        # Determine whether FAISS changed while SQLite/FTS5 did
        # not.
        # ---------------------------------------------------------

        if (
            faiss_after != faiss_before
            and sqlite_after == sqlite_before
            and fts_after == fts_before
        ):

            print()
            print(
                "[CONFIRMED] FAISS changed "
                "but SQLite/FTS5 did not."
            )

            print(
                "The current add() operation "
                "is not fully transactional "
                "across all three stores."
            )

        elif (
            faiss_after
            == sqlite_after
            == fts_after
        ):

            print()
            print(
                "[PASS] FAISS, SQLite, and "
                "FTS5 remained consistent."
            )

        else:

            print()
            print(
                "[WARNING] Store state changed "
                "in an unexpected way."
            )

        store.close()


def main() -> None:

    print()
    print("=" * 70)
    print("VECTOR STORE FAILURE-SCENARIO TEST")
    print("=" * 70)

    test_normal_add()

    test_sqlite_failure_after_faiss()

    print()
    print("=" * 70)
    print("FAILURE-SCENARIO TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
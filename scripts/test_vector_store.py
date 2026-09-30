from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.retrieval.chunker import Chunk
from src.retrieval.vector_store import VectorStore


def create_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "source": "failure_test.txt",
        },
    )


def counts(store: VectorStore) -> tuple[int, int, int]:
    return (
        store.count(),
        store.metadata_count(),
        store.keyword_count(),
    )


def assert_consistent(store: VectorStore) -> None:
    faiss_count, sqlite_count, fts_count = counts(store)

    print(
        f"FAISS={faiss_count}, "
        f"SQLite={sqlite_count}, "
        f"FTS5={fts_count}"
    )

    if not (
        faiss_count
        == sqlite_count
        == fts_count
    ):
        raise AssertionError(
            "FAISS, SQLite, and FTS5 are inconsistent."
        )


def test_normal_add() -> None:
    print("\n" + "=" * 70)
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

        print(f"Added: {added}")
        print(f"Counts: {counts(store)}")

        if added != 2:
            raise AssertionError(
                f"Expected 2 added vectors, got {added}."
            )

        assert_consistent(store)

        store.close()

    print("[PASS] Normal add remains consistent.")


def test_sqlite_failure_rolls_back_faiss() -> None:
    print("\n" + "=" * 70)
    print("TEST 2: SQLITE FAILURE AFTER FAISS INSERT")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorStore(
            index_dir=Path(temp_dir),
            dimension=4,
        )

        initial_embedding = np.array(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        store.add(
            initial_embedding,
            [
                create_chunk(
                    "initial_chunk",
                    "Initial valid document",
                )
            ],
        )

        before = counts(store)

        print(f"Before forced failure: {before}")
        assert_consistent(store)

        # Proxy the connection so executemany() fails after
        # VectorStore.add() has already modified FAISS.
        real_connection = store.connection

        class ConnectionProxy:
            def __init__(self, connection):
                self.real_connection = connection

            def executemany(self, *args, **kwargs):
                raise RuntimeError(
                    "INTENTIONAL TEST FAILURE: "
                    "SQLite insertion failed."
                )

            def __getattr__(self, name):
                return getattr(
                    self.real_connection,
                    name,
                )

        store.connection = ConnectionProxy(
            real_connection
        )

        failing_embedding = np.array(
            [[0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        failure_occurred = False

        try:
            store.add(
                failing_embedding,
                [
                    create_chunk(
                        "failing_chunk",
                        "This should be rolled back.",
                    )
                ],
            )

        except RuntimeError as error:
            failure_occurred = True
            print(
                f"Expected failure: "
                f"{type(error).__name__}: {error}"
            )

        finally:
            store.connection = real_connection

        if not failure_occurred:
            raise AssertionError(
                "The intentional SQLite failure "
                "did not occur."
            )

        after = counts(store)

        print(f"After forced failure:  {after}")

        # The failed operation must not leave any new record
        # in FAISS, SQLite, or FTS5.
        if after != before:
            raise AssertionError(
                "Rollback failed: store counts changed "
                f"from {before} to {after}."
            )

        assert_consistent(store)

        # Confirm that the failed chunk is absent.
        if store.contains_chunk("failing_chunk"):
            raise AssertionError(
                "Failed chunk was left in SQLite."
            )

        print(
            "[PASS] SQLite failure rolled back "
            "the FAISS change and kept all stores consistent."
        )

        store.close()


def test_fts_failure_rolls_back_everything() -> None:
    print("\n" + "=" * 70)
    print("TEST 3: FTS5 FAILURE AFTER SQLITE INSERT")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorStore(
            index_dir=Path(temp_dir),
            dimension=4,
        )

        initial_embedding = np.array(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        store.add(
            initial_embedding,
            [
                create_chunk(
                    "initial_chunk",
                    "Initial valid document",
                )
            ],
        )

        before = counts(store)

        real_execute = store.connection.execute

        class ConnectionProxy:
            def __init__(self, connection):
                self.real_connection = connection

            def execute(self, sql, *args, **kwargs):
                normalized = " ".join(
                    str(sql).split()
                ).lower()

                if (
                    "insert into chunks_fts" in normalized
                ):
                    raise RuntimeError(
                        "INTENTIONAL TEST FAILURE: "
                        "FTS5 insertion failed."
                    )

                return self.real_connection.execute(
                    sql,
                    *args,
                    **kwargs,
                )

            def __getattr__(self, name):
                return getattr(
                    self.real_connection,
                    name,
                )

        store.connection = ConnectionProxy(
            store.connection
        )

        failing_embedding = np.array(
            [[0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        failure_occurred = False

        try:
            store.add(
                failing_embedding,
                [
                    create_chunk(
                        "fts_failing_chunk",
                        "This should also be rolled back.",
                    )
                ],
            )

        except RuntimeError as error:
            failure_occurred = True
            print(
                f"Expected failure: "
                f"{type(error).__name__}: {error}"
            )

        finally:
            store.connection = (
                store.connection.real_connection
            )

        if not failure_occurred:
            raise AssertionError(
                "The intentional FTS5 failure "
                "did not occur."
            )

        after = counts(store)

        print(f"After forced failure:  {after}")

        if after != before:
            raise AssertionError(
                "FTS5 rollback failed: store counts "
                f"changed from {before} to {after}."
            )

        assert_consistent(store)

        if store.contains_chunk("fts_failing_chunk"):
            raise AssertionError(
                "FTS failure left the failed chunk in SQLite."
            )

        print(
            "[PASS] FTS5 failure rolled back "
            "FAISS and SQLite successfully."
        )

        store.close()


def main() -> None:
    print("=" * 70)
    print("VECTOR STORE FAILURE / ROLLBACK TESTS")
    print("=" * 70)

    test_normal_add()
    test_sqlite_failure_rolls_back_faiss()
    test_fts_failure_rolls_back_everything()

    print("\n" + "=" * 70)
    print("ALL VECTOR STORE FAILURE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

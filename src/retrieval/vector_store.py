from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np


class VectorStore:
    """
    Persistent hybrid-ready vector store.

    FAISS:
        Stores and searches embedding vectors using cosine
        similarity through normalized inner product.

    SQLite:
        Stores chunk text and metadata.

    SQLite FTS5:
        Provides persistent keyword / lexical search over
        chunk text.

    Storage structure:

        vector_store/
        ├── index.faiss
        └── metadata.db

    Database tables:

        chunks
            vector_id
            chunk_id
            text
            metadata

        chunks_fts
            FTS5 full-text search index over chunk text

    Design goals:
        - Persistent storage
        - Incremental additions
        - Batch insertion
        - Semantic search
        - Keyword search
        - Metadata retrieval
        - Duplicate prevention
        - Dimension validation
        - Memory-conscious operation
        - FAISS/SQLite consistency checks
        - Safe FAISS persistence
    """

    def __init__(
        self,
        index_dir: str | Path = "data/processed/vector_store",
        dimension: int = 384,
    ):
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than 0"
            )

        self.index_dir = Path(index_dir)

        self.dimension = dimension

        self.index_path = (
            self.index_dir / "index.faiss"
        )

        self.database_path = (
            self.index_dir / "metadata.db"
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self._configure_database()

        self.index = (
            self._load_or_create_index()
        )

        self._validate_store_consistency()

    # =========================================================
    # DATABASE CONFIGURATION
    # =========================================================

    def _configure_database(self) -> None:
        """
        Configure SQLite and create the required tables.

        FTS5 is used for persistent lexical search.
        """

        self.connection.execute(
            "PRAGMA journal_mode=WAL"
        )

        self.connection.execute(
            "PRAGMA synchronous=NORMAL"
        )

        # -----------------------------------------------------
        # Main metadata table
        # -----------------------------------------------------

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_chunks_chunk_id
            ON chunks(chunk_id)
            """
        )

        # -----------------------------------------------------
        # FTS5 virtual table
        #
        # content='chunks' means the actual text remains in
        # the chunks table while FTS5 maintains its search
        # index.
        # -----------------------------------------------------

        self.connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                text,
                content='chunks',
                content_rowid='vector_id'
            )
            """
        )

        self.connection.commit()

        # -----------------------------------------------------
        # Synchronize FTS5 with existing chunks.
        #
        # This is especially important when upgrading an
        # existing database that was created before FTS5
        # support was added.
        # -----------------------------------------------------

        self._synchronize_fts()

    # =========================================================
    # FTS5 SYNCHRONIZATION
    # =========================================================

    def _synchronize_fts(self) -> None:
        """
        Ensure the FTS5 index represents the current chunks
        table.

        Rebuild is intentionally used here because this method
        is called during initialization and guarantees that an
        existing database gets a correct FTS5 index.
        """

        try:

            self.connection.execute(
                """
                INSERT INTO chunks_fts(
                    chunks_fts
                )
                VALUES('rebuild')
                """
            )

            self.connection.commit()

        except sqlite3.OperationalError as error:

            raise RuntimeError(
                "SQLite FTS5 is not available in this "
                "Python environment."
            ) from error

    # =========================================================
    # FAISS INDEX CREATION
    # =========================================================

    def _create_index(self):
        """
        Create a FAISS index.

        Because embeddings are normalized, inner product is
        equivalent to cosine similarity.
        """

        base_index = faiss.IndexFlatIP(
            self.dimension
        )

        return faiss.IndexIDMap2(
            base_index
        )

    # =========================================================
    # FAISS INDEX LOADING
    # =========================================================

    def _load_or_create_index(self):
        """
        Load an existing FAISS index if available.

        Otherwise create a new index.
        """

        if self.index_path.exists():

            index = faiss.read_index(
                str(self.index_path)
            )

            if index.d != self.dimension:

                raise ValueError(
                    "FAISS index dimension mismatch. "
                    f"Expected {self.dimension}, "
                    f"found {index.d}."
                )

            return index

        return self._create_index()

    # =========================================================
    # CONSISTENCY CHECK
    # =========================================================

    def _validate_store_consistency(self) -> None:
        """
        Verify that FAISS and SQLite contain the same number
        of records.
        """

        faiss_count = int(
            self.index.ntotal
        )

        sqlite_count = (
            self.metadata_count()
        )

        if faiss_count != sqlite_count:

            raise RuntimeError(
                "Vector store is inconsistent: "
                f"FAISS contains {faiss_count} vectors, "
                f"while SQLite contains {sqlite_count} "
                "metadata records."
            )

    # =========================================================
    # ADD DOCUMENTS
    # =========================================================

    def add(
        self,
        embeddings: np.ndarray,
        chunks: Iterable,
    ) -> int:
        """
        Add embeddings and corresponding chunks.

        The operation updates:

            1. FAISS
            2. SQLite chunks
            3. SQLite FTS5

        Duplicate chunk IDs are skipped.

        If a database/FTS5 operation fails after FAISS has
        already been updated, the newly added FAISS IDs are
        removed so that FAISS, SQLite, and FTS5 remain
        consistent.
        """

        embeddings = self._prepare_embeddings(
            embeddings
        )

        chunks = list(chunks)

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings must match "
                "number of chunks."
            )

        if not chunks:
            return 0

        records = []
        vectors = []

        next_vector_id = (
            self._next_vector_id()
        )

        for embedding, chunk in zip(
            embeddings,
            chunks,
        ):

            chunk_id = str(
                chunk.metadata.get(
                    "chunk_id",
                    "",
                )
            ).strip()

            if not chunk_id:

                raise ValueError(
                    "Every chunk must contain "
                    "'chunk_id' in metadata."
                )

            # -------------------------------------------------
            # Duplicate protection
            # -------------------------------------------------

            if self.contains_chunk(
                chunk_id
            ):
                continue

            text = str(
                chunk.text
            ).strip()

            if not text:
                continue

            vector_id = next_vector_id

            next_vector_id += 1

            metadata = dict(
                chunk.metadata
            )

            records.append(
                (
                    vector_id,
                    chunk_id,
                    text,
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                )
            )

            vectors.append(
                embedding
            )

        if not records:
            return 0

        vectors_array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        vector_ids = np.asarray(
            [
                record[0]
                for record in records
            ],
            dtype=np.int64,
        )

        # -----------------------------------------------------
        # Add vectors to FAISS.
        #
        # The database operations below are protected by a
        # rollback handler. If they fail, these IDs are removed
        # from FAISS before the exception is re-raised.
        # -----------------------------------------------------

        self.index.add_with_ids(
            vectors_array,
            vector_ids,
        )

        try:

            # -------------------------------------------------
            # Add metadata to SQLite.
            # -------------------------------------------------

            self.connection.executemany(
                """
                INSERT INTO chunks
                (
                    vector_id,
                    chunk_id,
                    text,
                    metadata
                )
                VALUES (?, ?, ?, ?)
                """,
                records,
            )

            # -------------------------------------------------
            # Update FTS5 in the SAME SQLite transaction.
            #
            # We intentionally do not commit SQLite metadata
            # before FTS5 succeeds.
            # -------------------------------------------------

            for record in records:

                vector_id = record[0]

                self.connection.execute(
                    """
                    INSERT INTO chunks_fts(
                        rowid,
                        text
                    )
                    VALUES (?, ?)
                    """,
                    (
                        vector_id,
                        record[2],
                    ),
                )

            # -------------------------------------------------
            # Commit SQLite + FTS5 only after both succeed.
            # -------------------------------------------------

            self.connection.commit()

        except Exception:

            # -------------------------------------------------
            # Roll back SQLite/FTS5 changes.
            # -------------------------------------------------

            try:
                self.connection.rollback()
            except Exception:
                pass

            # -------------------------------------------------
            # Roll back the already-added FAISS vectors.
            # IndexIDMap2 supports removal by vector IDs.
            # -------------------------------------------------

            try:

                self.index.remove_ids(
                    vector_ids
                )

            except Exception as rollback_error:

                raise RuntimeError(
                    "Vector store rollback failed. "
                    "FAISS may be inconsistent with "
                    "SQLite/FTS5."
                ) from rollback_error

            # -------------------------------------------------
            # Rebuild FTS5 from the authoritative chunks table.
            # -------------------------------------------------

            try:

                self._synchronize_fts()

            except Exception as fts_error:

                raise RuntimeError(
                    "Vector store rollback completed for "
                    "FAISS/SQLite, but FTS5 synchronization "
                    "failed."
                ) from fts_error

            raise

        return len(records)

    # =========================================================
    # BATCH ADD
    # =========================================================

    def add_batch(
        self,
        embeddings: np.ndarray,
        chunks: list,
        batch_size: int = 256,
    ) -> int:
        """
        Add a large collection in smaller batches.
        """

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than 0"
            )

        embeddings = self._prepare_embeddings(
            embeddings
        )

        if len(embeddings) != len(chunks):

            raise ValueError(
                "Number of embeddings must match "
                "number of chunks."
            )

        total_added = 0

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(chunks),
            )

            batch_embeddings = (
                embeddings[start:end]
            )

            batch_chunks = (
                chunks[start:end]
            )

            total_added += self.add(
                batch_embeddings,
                batch_chunks,
            )

        return total_added

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic vector search using FAISS.
        """

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0"
            )

        if self.index.ntotal == 0:
            return []

        query = self._prepare_query(
            query_embedding
        )

        actual_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, ids = self.index.search(
            query,
            actual_k,
        )

        results = []

        for score, vector_id in zip(
            scores[0],
            ids[0],
        ):

            if vector_id == -1:
                continue

            row = self.connection.execute(
                """
                SELECT
                    vector_id,
                    chunk_id,
                    text,
                    metadata
                FROM chunks
                WHERE vector_id = ?
                """,
                (int(vector_id),),
            ).fetchone()

            if row is None:
                continue

            results.append(
                self._row_to_result(
                    row,
                    score=float(score),
                )
            )

        return results

    # =========================================================
    # KEYWORD SEARCH
    # =========================================================

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Perform lexical keyword search using SQLite FTS5.

        FTS5 ranks results using its BM25 ranking function.

        The returned score is converted to a positive relevance
        score where larger values represent better matches.
        """

        if not isinstance(query, str):

            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:

            raise ValueError(
                "Keyword query cannot be empty"
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0"
            )

        # -----------------------------------------------------
        # Convert natural query into an FTS5-safe query.
        #
        # We use individual terms rather than passing arbitrary
        # FTS5 syntax directly from the user.
        # -----------------------------------------------------

        tokens = self._tokenize_query(
            query
        )

        if not tokens:
            return []

        fts_query = " OR ".join(
            self._escape_fts_token(token)
            for token in tokens
        )

        rows = self.connection.execute(
            """
            SELECT
                chunks.vector_id,
                chunks.chunk_id,
                chunks.text,
                chunks.metadata,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks
                ON chunks.vector_id =
                   chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (
                fts_query,
                top_k,
            ),
        ).fetchall()

        results = []

        for row in rows:

            # SQLite FTS5 bm25 returns lower values for
            # more relevant matches. Convert to a positive
            # relevance score.
            raw_rank = float(
                row["rank"]
            )

            relevance = (
                1.0
                / (
                    1.0
                    + max(
                        raw_rank,
                        0.0,
                    )
                )
            )

            result = self._row_to_result(
                row,
                score=relevance,
            )

            result["keyword_rank"] = raw_rank

            results.append(
                result
            )

        return results

    # =========================================================
    # QUERY TOKENIZATION
    # =========================================================

    def _tokenize_query(
        self,
        query: str,
    ) -> list[str]:
        """
        Convert a query into simple lexical tokens.

        This deliberately stays lightweight. More advanced
        query processing will be handled by the hybrid-search
        layer.
        """

        cleaned = []

        current = []

        for character in query:

            if character.isalnum():

                current.append(
                    character.lower()
                )

            elif current:

                token = "".join(
                    current
                )

                if len(token) >= 2:

                    cleaned.append(
                        token
                    )

                current.clear()

        if current:

            token = "".join(
                current
            )

            if len(token) >= 2:

                cleaned.append(
                    token
                )

        # Remove duplicates while preserving order.

        unique_tokens = []

        seen = set()

        for token in cleaned:

            if token not in seen:

                seen.add(token)

                unique_tokens.append(
                    token
                )

        return unique_tokens

    # =========================================================
    # FTS TOKEN ESCAPING
    # =========================================================

    def _escape_fts_token(
        self,
        token: str,
    ) -> str:
        """
        Safely quote an individual FTS5 token.
        """

        escaped = token.replace(
            '"',
            '""',
        )

        return f'"{escaped}"'

    # =========================================================
    # RESULT CONVERSION
    # =========================================================

    def _row_to_result(
        self,
        row,
        score: float,
    ) -> dict[str, Any]:
        """
        Convert a SQLite row into the common retrieval result
        structure.
        """

        metadata = json.loads(
            row["metadata"]
        )

        return {
            "vector_id": int(
                row["vector_id"]
            ),
            "chunk_id": row["chunk_id"],
            "score": float(score),
            "text": row["text"],
            "metadata": metadata,
        }

    # =========================================================
    # DUPLICATE CHECK
    # =========================================================

    def contains_chunk(
        self,
        chunk_id: str,
    ) -> bool:
        """
        Check whether a chunk already exists.
        """

        row = self.connection.execute(
            """
            SELECT 1
            FROM chunks
            WHERE chunk_id = ?
            LIMIT 1
            """,
            (chunk_id,),
        ).fetchone()

        return row is not None

    # =========================================================
    # NEXT VECTOR ID
    # =========================================================

    def _next_vector_id(self) -> int:
        """
        Return the next available vector ID.
        """

        row = self.connection.execute(
            """
            SELECT COALESCE(
                MAX(vector_id),
                0
            ) + 1
            FROM chunks
            """
        ).fetchone()

        return int(
            row[0]
        )

    # =========================================================
    # EMBEDDING VALIDATION
    # =========================================================

    def _prepare_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize an embedding matrix.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:

            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if embeddings.shape[1] != self.dimension:

            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        if not np.isfinite(
            embeddings
        ).all():

            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        valid = (
            norms.squeeze() > 0
        )

        embeddings[valid] = (
            embeddings[valid]
            / norms[valid]
        )

        return embeddings

    # =========================================================
    # QUERY VALIDATION
    # =========================================================

    def _prepare_query(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize a query vector.
        """

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:

            query = query.reshape(
                1,
                -1,
            )

        if query.ndim != 2:

            raise ValueError(
                "Query embedding must be a 1D "
                "or 2D array."
            )

        if query.shape[1] != self.dimension:

            raise ValueError(
                "Query embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"received {query.shape[1]}."
            )

        if not np.isfinite(
            query
        ).all():

            raise ValueError(
                "Query embedding contains "
                "NaN or infinite values."
            )

        norms = np.linalg.norm(
            query,
            axis=1,
            keepdims=True,
        )

        valid = (
            norms.squeeze() > 0
        )

        query[valid] = (
            query[valid]
            / norms[valid]
        )

        return query

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def save(self) -> None:
        """
        Persist the FAISS index to disk.

        Uses a temporary file followed by atomic replacement.
        """

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".faiss",
                dir=self.index_dir,
                delete=False,
            ) as temporary_file:

                temporary_path = Path(
                    temporary_file.name
                )

            faiss.write_index(
                self.index,
                str(temporary_path),
            )

            os.replace(
                temporary_path,
                self.index_path,
            )

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                temporary_path.unlink(
                    missing_ok=True
                )

    # =========================================================
    # INFORMATION
    # =========================================================

    def count(self) -> int:
        """
        Return the number of vectors in FAISS.
        """

        return int(
            self.index.ntotal
        )

    def metadata_count(self) -> int:
        """
        Return the number of chunks in SQLite.
        """

        row = self.connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()

        return int(
            row[0]
        )

    def keyword_count(self) -> int:
        """
        Return the number of rows represented in the FTS5
        index.
        """

        row = self.connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks_fts
            """
        ).fetchone()

        return int(
            row[0]
        )

    def get_dimension(self) -> int:
        """
        Return vector dimension.
        """

        return self.dimension

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> None:
        """
        Completely clear FAISS, SQLite metadata, and FTS5.
        """

        self.index = (
            self._create_index()
        )

        self.connection.execute(
            "DELETE FROM chunks"
        )

        self.connection.commit()

        # Rebuild the now-empty FTS5 index.

        self._synchronize_fts()

        if self.index_path.exists():

            self.index_path.unlink()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self) -> None:
        """
        Save FAISS and close SQLite.
        """

        self.save()

        self.connection.close()

    # =========================================================
    # CONTEXT MANAGER
    # =========================================================

    def __enter__(self):

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

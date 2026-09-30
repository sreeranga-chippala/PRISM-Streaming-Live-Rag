from __future__ import annotations

from typing import Any

from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


class HybridSearch:
    """
    Hybrid retrieval layer.

    Combines two independent retrieval strategies:

        1. Semantic retrieval
           - SentenceTransformer embeddings
           - FAISS vector search

        2. Keyword retrieval
           - SQLite FTS5
           - Lexical keyword matching

    This class is responsible for candidate generation.

    It does NOT perform:
        - rank fusion
        - final scoring
        - reranking
        - query decomposition

    Those responsibilities belong to later retrieval stages.
    """

    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        semantic_top_k: int = 5,
        keyword_top_k: int = 5,
    ):
        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than 0"
            )

        if keyword_top_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than 0"
            )

        self.retriever = retriever
        self.vector_store = vector_store

        self.semantic_top_k = (
            semantic_top_k
        )

        self.keyword_top_k = (
            keyword_top_k
        )

    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        semantic_top_k: int | None = None,
        keyword_top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Run semantic and keyword retrieval for the same query.

        Returns both retrieval lists and a merged candidate pool.

        No final ranking is performed here.

        Example return structure:

            {
                "query": "...",

                "semantic_results": [...],

                "keyword_results": [...],

                "candidates": [...]
            }
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty"
            )

        if semantic_top_k is None:
            semantic_top_k = (
                self.semantic_top_k
            )

        if keyword_top_k is None:
            keyword_top_k = (
                self.keyword_top_k
            )

        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than 0"
            )

        if keyword_top_k <= 0:
            raise ValueError(
                "keyword_top_k must be greater than 0"
            )

        # -----------------------------------------------------
        # 1. Semantic retrieval
        # -----------------------------------------------------

        semantic_results = (
            self.retriever.retrieve(
                query=query,
                top_k=semantic_top_k,
            )
        )

        # -----------------------------------------------------
        # 2. Keyword retrieval
        # -----------------------------------------------------

        keyword_results = (
            self.vector_store.keyword_search(
                query=query,
                top_k=keyword_top_k,
            )
        )

        # -----------------------------------------------------
        # 3. Build unified candidate pool
        # -----------------------------------------------------

        candidates = (
            self._build_candidate_pool(
                semantic_results,
                keyword_results,
            )
        )

        return {
            "query": query,
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "candidates": candidates,
        }

    # =========================================================
    # CANDIDATE POOL
    # =========================================================

    def _build_candidate_pool(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Merge semantic and keyword results using chunk_id.

        A chunk appearing in both retrieval systems becomes a
        single candidate containing both signals.

        Example:

            semantic_rank = 1
            keyword_rank = 2

        means the same chunk was found by both retrievers.

        No final combined score is calculated here.
        """

        candidates = {}

        # -----------------------------------------------------
        # Semantic results
        # -----------------------------------------------------

        for rank, result in enumerate(
            semantic_results,
            start=1,
        ):

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "vector_id": result[
                        "vector_id"
                    ],
                    "chunk_id": chunk_id,
                    "text": result[
                        "text"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                    "semantic_score": (
                        result["score"]
                    ),
                    "semantic_rank": rank,
                    "keyword_score": None,
                    "keyword_rank": None,
                    "retrieval_sources": [
                        "semantic"
                    ],
                }

            else:

                candidates[
                    chunk_id
                ][
                    "semantic_score"
                ] = result["score"]

                candidates[
                    chunk_id
                ][
                    "semantic_rank"
                ] = rank

                candidates[
                    chunk_id
                ][
                    "retrieval_sources"
                ].append(
                    "semantic"
                )

        # -----------------------------------------------------
        # Keyword results
        # -----------------------------------------------------

        for rank, result in enumerate(
            keyword_results,
            start=1,
        ):

            chunk_id = result[
                "chunk_id"
            ]

            if chunk_id not in candidates:

                candidates[chunk_id] = {
                    "vector_id": result[
                        "vector_id"
                    ],
                    "chunk_id": chunk_id,
                    "text": result[
                        "text"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                    "semantic_score": None,
                    "semantic_rank": None,
                    "keyword_score": (
                        result["score"]
                    ),
                    "keyword_rank": rank,
                    "retrieval_sources": [
                        "keyword"
                    ],
                }

            else:

                candidates[
                    chunk_id
                ][
                    "keyword_score"
                ] = result["score"]

                candidates[
                    chunk_id
                ][
                    "keyword_rank"
                ] = rank

                candidates[
                    chunk_id
                ][
                    "retrieval_sources"
                ].append(
                    "keyword"
                )

        # -----------------------------------------------------
        # Sort candidates only for deterministic output.
        #
        # This is NOT final relevance ranking.
        #
        # Candidates found by both retrievers are shown first,
        # followed by candidates found by one retriever.
        # Final ranking will be handled by fusion.py.
        # -----------------------------------------------------

        candidate_list = list(
            candidates.values()
        )

        candidate_list.sort(
            key=lambda candidate: (
                -len(
                    candidate[
                        "retrieval_sources"
                    ]
                ),
                self._best_rank(
                    candidate
                ),
                candidate[
                    "chunk_id"
                ],
            )
        )

        return candidate_list

    # =========================================================
    # BEST RANK
    # =========================================================

    def _best_rank(
        self,
        candidate: dict[str, Any],
    ) -> int:
        """
        Return the best rank obtained by this candidate from
        either retrieval method.

        Used only for deterministic candidate ordering.
        """

        ranks = []

        semantic_rank = candidate[
            "semantic_rank"
        ]

        keyword_rank = candidate[
            "keyword_rank"
        ]

        if semantic_rank is not None:
            ranks.append(
                semantic_rank
            )

        if keyword_rank is not None:
            ranks.append(
                keyword_rank
            )

        if not ranks:
            return 10**9

        return min(ranks)

    # =========================================================
    # CONVENIENCE METHODS
    # =========================================================

    def semantic_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run only semantic retrieval.
        """

        if top_k is None:
            top_k = self.semantic_top_k

        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run only keyword retrieval.
        """

        if top_k is None:
            top_k = self.keyword_top_k

        return self.vector_store.keyword_search(
            query=query,
            top_k=top_k,
        )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def get_semantic_top_k(self) -> int:
        return self.semantic_top_k

    def get_keyword_top_k(self) -> int:
        return self.keyword_top_k
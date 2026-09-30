from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Combines multiple ranked retrieval lists without requiring
    their scores to be on the same numerical scale.

    This is useful for combining:

        - Semantic FAISS retrieval
        - Keyword FTS5 retrieval
        - Future additional retrieval methods

    RRF formula:

        RRF(d) = sum(
            weight / (k + rank)
        )

    where:

        d      = document/chunk
        rank   = rank assigned by a retriever
        k      = RRF smoothing constant
        weight = optional retriever weight
    """

    def __init__(
        self,
        k: int = 60,
    ):
        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        self.k = k

    # =========================================================
    # FUSE RETRIEVAL RESULTS
    # =========================================================

    def fuse(
        self,
        result_lists: list[list[dict[str, Any]]],
        weights: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fuse multiple ranked result lists using RRF.

        Parameters
        ----------
        result_lists:
            A list containing ranked retrieval result lists.

        weights:
            Optional weight for each retrieval list.

            Example:

                [
                    1.0,   # semantic
                    1.0,   # keyword
                ]

        Returns
        -------
        list[dict]
            Results sorted by descending RRF score.
        """

        if not result_lists:
            return []

        if weights is None:

            weights = [
                1.0
                for _ in result_lists
            ]

        if len(weights) != len(
            result_lists
        ):
            raise ValueError(
                "Number of weights must match "
                "number of result lists."
            )

        for weight in weights:

            if weight < 0:

                raise ValueError(
                    "Weights cannot be negative."
                )

        fused = {}

        # -----------------------------------------------------
        # Process every retrieval system
        # -----------------------------------------------------

        for list_index, results in enumerate(
            result_lists
        ):

            weight = weights[
                list_index
            ]

            for rank, result in enumerate(
                results,
                start=1,
            ):

                chunk_id = result.get(
                    "chunk_id"
                )

                if not chunk_id:
                    continue

                # -------------------------------------------------
                # Create candidate if this is the first occurrence
                # -------------------------------------------------

                if chunk_id not in fused:

                    fused[chunk_id] = {
                        "vector_id": result.get(
                            "vector_id"
                        ),
                        "chunk_id": chunk_id,
                        "text": result.get(
                            "text",
                            "",
                        ),
                        "metadata": result.get(
                            "metadata",
                            {},
                        ),
                        "rrf_score": 0.0,
                        "retrieval_ranks": [],
                        "retrieval_sources": [],
                        "original_scores": [],
                    }

                candidate = fused[
                    chunk_id
                ]

                # -------------------------------------------------
                # Calculate RRF contribution
                # -------------------------------------------------

                contribution = (
                    weight
                    / (
                        self.k
                        + rank
                    )
                )

                candidate[
                    "rrf_score"
                ] += contribution

                # -------------------------------------------------
                # Keep retrieval information
                # -------------------------------------------------

                candidate[
                    "retrieval_ranks"
                ].append(
                    {
                        "retriever_index": (
                            list_index
                        ),
                        "rank": rank,
                        "weight": weight,
                        "contribution": (
                            contribution
                        ),
                    }
                )

                candidate[
                    "original_scores"
                ].append(
                    {
                        "retriever_index": (
                            list_index
                        ),
                        "score": result.get(
                            "score"
                        ),
                    }
                )

                if (
                    list_index
                    not in candidate[
                        "retrieval_sources"
                    ]
                ):

                    candidate[
                        "retrieval_sources"
                    ].append(
                        list_index
                    )

        # -----------------------------------------------------
        # Sort by descending RRF score
        # -----------------------------------------------------

        fused_results = list(
            fused.values()
        )

        fused_results.sort(
            key=lambda result: (
                -result[
                    "rrf_score"
                ],
                result[
                    "chunk_id"
                ],
            )
        )

        # -----------------------------------------------------
        # Add final fusion rank
        # -----------------------------------------------------

        for rank, result in enumerate(
            fused_results,
            start=1,
        ):

            result[
                "fusion_rank"
            ] = rank

        return fused_results

    # =========================================================
    # FUSE HYBRID SEARCH RESULT
    # =========================================================

    def fuse_hybrid_result(
        self,
        hybrid_result: dict[str, Any],
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Fuse the semantic and keyword results returned by
        HybridSearch.

        Parameters
        ----------
        hybrid_result:
            Result returned by HybridSearch.search().

        semantic_weight:
            Weight assigned to semantic retrieval.

        keyword_weight:
            Weight assigned to keyword retrieval.
        """

        if not isinstance(
            hybrid_result,
            dict,
        ):
            raise TypeError(
                "hybrid_result must be a dictionary."
            )

        semantic_results = (
            hybrid_result.get(
                "semantic_results",
                [],
            )
        )

        keyword_results = (
            hybrid_result.get(
                "keyword_results",
                [],
            )
        )

        return self.fuse(
            result_lists=[
                semantic_results,
                keyword_results,
            ],
            weights=[
                semantic_weight,
                keyword_weight,
            ],
        )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def get_k(self) -> int:
        return self.k
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder based reranker.

    The reranker receives a query and a set of candidate
    documents produced by the retrieval + fusion stages.

    Unlike embedding retrieval, which independently embeds the
    query and document, a cross-encoder evaluates the query and
    document together.

    Pipeline:

        Query
          +
        Candidate document
          ↓
        CrossEncoder
          ↓
        Relevance score
          ↓
        Final ranking
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self.model_name = model_name
        self.batch_size = batch_size

        print(
            f"Loading reranker model: "
            f"{model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

        print(
            "Reranker model loaded successfully."
        )

    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidate documents for a query.

        Parameters
        ----------
        query:
            User's natural-language query.

        candidates:
            Candidate documents returned by the fusion stage.

        top_k:
            Number of final results to return.

            If None, return all candidates.

        Returns
        -------
        list[dict]
            Candidates sorted by reranker score.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty"
            )

        if not isinstance(
            candidates,
            list,
        ):
            raise TypeError(
                "candidates must be a list"
            )

        if top_k is not None:

            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0"
                )

        if not candidates:
            return []

        # -----------------------------------------------------
        # Build query-document pairs
        # -----------------------------------------------------

        pairs = []

        valid_candidates = []

        for candidate in candidates:

            if not isinstance(
                candidate,
                dict,
            ):
                continue

            text = candidate.get(
                "text",
                "",
            )

            if not isinstance(
                text,
                str,
            ):
                text = str(text)

            text = text.strip()

            if not text:
                continue

            pairs.append(
                (
                    query,
                    text,
                )
            )

            valid_candidates.append(
                candidate
            )

        if not pairs:
            return []

        # -----------------------------------------------------
        # Cross-encoder prediction
        # -----------------------------------------------------

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # -----------------------------------------------------
        # Attach reranker scores
        # -----------------------------------------------------

        reranked = []

        for candidate, score in zip(
            valid_candidates,
            scores,
        ):

            result = dict(
                candidate
            )

            result[
                "reranker_score"
            ] = float(score)

            reranked.append(
                result
            )

        # -----------------------------------------------------
        # Sort by reranker score
        # -----------------------------------------------------

        reranked.sort(
            key=lambda result: (
                -result[
                    "reranker_score"
                ],
                result.get(
                    "fusion_rank",
                    10**9,
                ),
                result.get(
                    "chunk_id",
                    "",
                ),
            )
        )

        # -----------------------------------------------------
        # Add final rank
        # -----------------------------------------------------

        for rank, result in enumerate(
            reranked,
            start=1,
        ):

            result[
                "reranker_rank"
            ] = rank

        # -----------------------------------------------------
        # Return requested number
        # -----------------------------------------------------

        if top_k is not None:

            reranked = reranked[
                :top_k
            ]

            # Recalculate final rank after truncation.

            for rank, result in enumerate(
                reranked,
                start=1,
            ):

                result[
                    "reranker_rank"
                ] = rank

        return reranked

    # =========================================================
    # RERANK FUSED RESULTS
    # =========================================================

    def rerank_fused(
        self,
        query: str,
        fused_results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Convenience method for reranking the output of
        ReciprocalRankFusion.
        """

        return self.rerank(
            query=query,
            candidates=fused_results,
            top_k=top_k,
        )

    # =========================================================
    # CONFIGURATION
    # =========================================================

    def get_model_name(self) -> str:
        return self.model_name

    def get_batch_size(self) -> int:
        return self.batch_size
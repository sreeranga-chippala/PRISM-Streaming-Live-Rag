from __future__ import annotations

from typing import Any

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore


class Retriever:
    """
    Semantic retriever for the RAG pipeline.

    Responsibilities:
        1. Accept a natural-language query.
        2. Convert the query into an embedding.
        3. Search the persistent vector store.
        4. Return the most relevant chunks.

    This class intentionally does NOT perform:
        - query decomposition
        - keyword search
        - hybrid retrieval
        - rank fusion
        - reranking

    Those responsibilities belong to later retrieval stages.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        default_top_k: int = 5,
        min_score: float | None = None,
    ):
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater than 0"
            )

        if min_score is not None:
            if not -1.0 <= min_score <= 1.0:
                raise ValueError(
                    "min_score must be between -1.0 and 1.0"
                )

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.min_score = min_score

        # ----------------------------------------------------
        # Make sure the embedding model and vector store use
        # the same vector dimension.
        # ----------------------------------------------------

        embedding_dimension = (
            self.embedding_model.get_dimension()
        )

        store_dimension = (
            self.vector_store.get_dimension()
        )

        if embedding_dimension != store_dimension:

            raise ValueError(
                "Embedding model and vector store "
                "dimensions do not match. "
                f"Embedding model: "
                f"{embedding_dimension}, "
                f"Vector store: "
                f"{store_dimension}."
            )

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Parameters
        ----------
        query:
            Natural-language user query.

        top_k:
            Maximum number of results to return.
            Uses default_top_k if omitted.

        min_score:
            Optional minimum similarity score.

            If omitted, the retriever uses the value
            configured during initialization.

        Returns
        -------
        list[dict[str, Any]]
            Retrieved chunks sorted by descending
            similarity score.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not isinstance(query, str):

            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:

            raise ValueError(
                "Query cannot be empty"
            )

        # ----------------------------------------------------
        # Resolve top_k
        # ----------------------------------------------------

        if top_k is None:

            top_k = self.default_top_k

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than 0"
            )

        # ----------------------------------------------------
        # Resolve minimum score
        # ----------------------------------------------------

        if min_score is None:

            min_score = self.min_score

        if min_score is not None:

            if not -1.0 <= min_score <= 1.0:

                raise ValueError(
                    "min_score must be between "
                    "-1.0 and 1.0"
                )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                query
            )
        )

        # ----------------------------------------------------
        # Search vector store
        # ----------------------------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        # ----------------------------------------------------
        # Apply optional score threshold
        # ----------------------------------------------------

        if min_score is not None:

            results = [
                result
                for result in results
                if result["score"] >= min_score
            ]

        # ----------------------------------------------------
        # Add query information to results
        #
        # This makes downstream components easier to build.
        # ----------------------------------------------------

        for result in results:

            result["query"] = query

        return results

    # =========================================================
    # RETRIEVE WITH SCORES
    # =========================================================

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Explicitly named convenience method for callers that
        need similarity scores.

        The underlying results already contain scores, so this
        method currently delegates to retrieve().
        """

        return self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

    # =========================================================
    # SIMPLE TEXT RESULTS
    # =========================================================

    def retrieve_texts(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[str]:
        """
        Return only the retrieved chunk texts.

        Useful when a downstream component only needs the
        textual context.
        """

        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
        )

        return [
            result["text"]
            for result in results
        ]

    # =========================================================
    # INFORMATION
    # =========================================================

    def get_default_top_k(self) -> int:
        """
        Return the default number of results.
        """

        return self.default_top_k

    def get_min_score(self) -> float | None:
        """
        Return the configured minimum score.
        """

        return self.min_score
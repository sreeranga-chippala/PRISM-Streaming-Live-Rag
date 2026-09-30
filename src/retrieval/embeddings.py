from typing import Iterable, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Embedding model wrapper for the RAG retrieval pipeline.

    Responsibilities:
    - Load the embedding model once.
    - Generate an embedding for one text.
    - Generate embeddings for a batch of texts.
    - Generate embeddings incrementally for large datasets.
    - Keep embeddings normalized for cosine-similarity search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        # Current SentenceTransformer API
        self.dimension = (
            self.model.get_embedding_dimension()
        )

        print(
            f"Embedding dimension: {self.dimension}"
        )

    # =========================================================
    # SINGLE TEXT
    # =========================================================

    def encode(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate an embedding for one piece of text.
        """

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Cannot create embedding for empty text"
            )

        vector = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vector.astype(
            np.float32
        )

    # =========================================================
    # BATCH ENCODING
    # =========================================================

    def encode_batch(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Batch processing is considerably more efficient than
        encoding every text individually.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        cleaned_texts = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "Every item must be a string"
                )

            text = text.strip()

            if not text:
                raise ValueError(
                    "Empty text found in batch"
                )

            cleaned_texts.append(text)

        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.astype(
            np.float32
        )

    # =========================================================
    # LARGE DATASET / ITERATIVE ENCODING
    # =========================================================

    def encode_iter(
        self,
        texts: Iterable[str],
    ):
        """
        Generate embeddings incrementally.

        Texts are processed in batches instead of loading the
        entire dataset into memory.

        Example:

            for embedding in model.encode_iter(texts):
                save_embedding(embedding)
        """

        batch = []

        for text in texts:

            if not isinstance(text, str):
                raise TypeError(
                    "Every item must be a string"
                )

            text = text.strip()

            if not text:
                continue

            batch.append(text)

            if len(batch) >= self.batch_size:

                embeddings = self.encode_batch(
                    batch
                )

                for embedding in embeddings:
                    yield embedding

                batch.clear()

        # Process the final incomplete batch.

        if batch:

            embeddings = self.encode_batch(
                batch
            )

            for embedding in embeddings:
                yield embedding

    # =========================================================
    # INFORMATION
    # =========================================================

    def get_dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        return self.dimension

    def get_model_name(self) -> str:
        """
        Return the embedding model name.
        """

        return self.model_name
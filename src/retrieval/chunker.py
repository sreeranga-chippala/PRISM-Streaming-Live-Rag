from dataclasses import dataclass
from typing import Any, Iterable, Iterator
import hashlib


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller "
                "than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ========================================================
    # MULTIPLE DOCUMENTS
    # ========================================================

    def chunk_documents(
        self,
        documents: Iterable,
    ) -> Iterator[Chunk]:
        """
        Chunk multiple documents incrementally.
        """

        for document in documents:

            yield from self.chunk_document(
                document
            )

    # ========================================================
    # SINGLE DOCUMENT
    # ========================================================

    def chunk_document(
        self,
        document,
    ) -> Iterator[Chunk]:
        """
        Convert one Document into one or more Chunks.
        """

        if not document.text:
            return

        text = document.text.strip()

        if not text:
            return

        # ----------------------------------------------------
        # Create a stable base identifier for this document.
        #
        # Metadata such as page, row, slide, or record is
        # included so chunks from the same source remain unique.
        # ----------------------------------------------------

        base_id = self._create_base_id(
            document
        )

        chunk_index = 0

        for chunk_text in self._chunk_text(
            text
        ):

            metadata = (
                document.metadata.copy()
            )

            # ------------------------------------------------
            # Unique chunk identifier
            # ------------------------------------------------

            metadata["chunk_id"] = (
                f"{base_id}_chunk_{chunk_index}"
            )

            metadata["chunk_index"] = (
                chunk_index
            )

            metadata["chunk_word_count"] = (
                len(chunk_text.split())
            )

            yield Chunk(
                text=chunk_text,
                metadata=metadata,
            )

            chunk_index += 1

    # ========================================================
    # BASE ID
    # ========================================================

    def _create_base_id(
        self,
        document,
    ) -> str:
        """
        Create a stable identifier for the document unit.

        Metadata such as:

            source
            page
            row
            slide
            record
            sheet

        is included when available.

        A short content hash is also included to reduce the
        possibility of collisions between otherwise identical
        metadata combinations.
        """

        metadata = document.metadata

        source = str(
            metadata.get(
                "source",
                "unknown_source"
            )
        )

        source = (
            source
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )

        parts = [source]

        # ----------------------------------------------------
        # Format-specific identifiers
        # ----------------------------------------------------

        for key in (
            "page",
            "slide",
            "row",
            "record",
            "sheet",
        ):

            if key in metadata:

                value = str(
                    metadata[key]
                )

                safe_value = (
                    value
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(":", "_")
                    .replace(" ", "_")
                )

                parts.append(
                    f"{key}_{safe_value}"
                )

        # ----------------------------------------------------
        # Content hash
        # ----------------------------------------------------

        content_hash = hashlib.sha1(
            document.text.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()[:10]

        parts.append(
            content_hash
        )

        return "_".join(parts)

    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    def _chunk_text(
        self,
        text: str,
    ) -> Iterator[str]:
        """
        Split text into overlapping word-based chunks.

        chunk_size and chunk_overlap are measured in words.
        """

        words = text.split()

        start = 0

        step = (
            self.chunk_size
            - self.chunk_overlap
        )

        while start < len(words):

            end = min(
                start + self.chunk_size,
                len(words),
            )

            chunk_words = words[
                start:end
            ]

            if chunk_words:

                yield " ".join(
                    chunk_words
                )

            start += step
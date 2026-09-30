from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import re


@dataclass(frozen=True)
class Citation:
    """A source citation associated with a retrieved evidence item."""

    citation_id: str
    source: str
    chunk_id: str | None
    score: float | None
    excerpt: str


class CitationManager:
    """Builds citations from evidence that supports the generated answer."""

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "with", "as", "by", "from", "that",
        "this", "it", "be", "has", "have", "had", "at", "into", "than",
        "then", "they", "their", "its", "also", "can", "may", "will",
    }

    def build_citations(
        self,
        answer: str,
        retrieved_results: Iterable[dict[str, Any]],
        max_citations: int = 6,
        min_support: float = 0.35,
        max_rank: int = 3,
    ) -> list[Citation]:
        if max_citations <= 0:
            raise ValueError("max_citations must be greater than 0")
        if max_rank <= 0:
            raise ValueError("max_rank must be greater than 0")
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1")

        answer = answer.strip()

        if not answer:
            raise ValueError("answer cannot be empty")

        answer_sentences = self._sentences(answer)

        if not answer_sentences:
            return []

        candidates = []

        for result in retrieved_results:
            if not isinstance(result, dict):
                continue

            reranker_rank = result.get("reranker_rank")

            if (
                isinstance(reranker_rank, int)
                and reranker_rank > max_rank
            ):
                continue

            text = str(result.get("text", "")).strip()

            if not text:
                continue

            evidence_tokens = self._content_tokens(text)

            if not evidence_tokens:
                continue

            # A citation is relevant if it strongly supports at least
            # one complete sentence of the generated answer.
            best_support = max(
                self._sentence_support(
                    sentence,
                    evidence_tokens,
                )
                for sentence in answer_sentences
            )

            candidates.append(
                (best_support, result)
            )

        relevant_results = [
            result
            for support, result in candidates
            if support >= min_support
        ]

        # Preserve the strongest evidence if nothing passes the threshold.
        if not relevant_results and candidates:
            relevant_results = [
                max(
                    candidates,
                    key=lambda item: item[0],
                )[1]
            ]

        citations: list[Citation] = []

        for index, result in enumerate(
            relevant_results[:max_citations],
            start=1,
        ):
            metadata = result.get("metadata") or {}

            source = (
                metadata.get("source")
                or metadata.get("filename")
                or metadata.get("title")
                or result.get("source")
                or "Unknown source"
            )

            chunk_id = result.get("chunk_id")
            score = self._score(result)
            text = str(result.get("text", "")).strip()

            citations.append(
                Citation(
                    citation_id=f"S{index}",
                    source=str(source),
                    chunk_id=(
                        str(chunk_id)
                        if chunk_id is not None
                        else None
                    ),
                    score=score,
                    excerpt=self._excerpt(text),
                )
            )

        return citations

    def support_coverage(
        self,
        answer: str,
        retrieved_results: Iterable[dict[str, Any]],
        citations: Iterable[Citation],
    ) -> float:
        """Measure factual sentence support using cited chunks only."""
        results_by_id = {
            str(item.get("chunk_id")): item
            for item in retrieved_results
            if isinstance(item, dict) and item.get("chunk_id") is not None
        }
        cited_text = " ".join(
            str(results_by_id[citation.chunk_id].get("text", ""))
            for citation in citations
            if citation.chunk_id is not None and citation.chunk_id in results_by_id
        )
        evidence_tokens = self._content_tokens(cited_text)
        sentences = self._sentences(answer)
        if not sentences:
            return 0.0
        supported = sum(
            1
            for sentence in sentences
            if self._sentence_support(sentence, evidence_tokens) >= 0.85
        )
        return round(supported / len(sentences), 4)

    def attach(
        self,
        answer: str,
        citations: Iterable[Citation],
    ) -> str:
        answer = answer.strip()
        citations = list(citations)

        if not answer:
            raise ValueError("answer cannot be empty")

        if not citations:
            return answer

        lines = ["", "Sources:"]

        for citation in citations:
            lines.append(
                f"[{citation.citation_id}] {citation.source}"
            )

        return answer + "\n".join(lines)

    @staticmethod
    def _score(
        result: dict[str, Any],
    ) -> float | None:
        for key in (
            "reranker_score",
            "rrf_score",
            "score",
            "semantic_score",
        ):
            value = result.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        return None

    @classmethod
    def _content_tokens(
        cls,
        text: str,
    ) -> set[str]:
        tokens = set(
            re.findall(
                r"[A-Za-z0-9][A-Za-z0-9_-]*",
                text.lower(),
            )
        )

        return tokens - cls.STOPWORDS

    @staticmethod
    def _excerpt(
        text: str,
        limit: int = 240,
    ) -> str:
        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if len(text) <= limit:
            return text

        return text[: limit - 1].rstrip() + "…"

    @classmethod
    def _sentence_support(
        cls,
        sentence: str,
        evidence_tokens: set[str],
    ) -> float:
        sentence_tokens = cls._content_tokens(sentence)

        if not sentence_tokens:
            return 0.0

        return (
            len(sentence_tokens & evidence_tokens)
            / len(sentence_tokens)
        )


    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(
                r"(?<=[.!?])\s+",
                text,
            )
            if part.strip()
        ]
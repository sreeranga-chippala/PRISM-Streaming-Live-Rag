from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GroundingResult:
    grounded: bool
    score: float
    supported_tokens: int
    answer_tokens: int
    unsupported_sentences: tuple[str, ...]


class GroundingChecker:
    """
    Lightweight deterministic grounding baseline.

    This is intentionally transparent rather than pretending to be a
    semantic truth judge. It measures lexical support of answer content
    against retrieved evidence and is useful for hackathon telemetry.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "with", "as", "by", "from", "that",
        "this", "it", "be", "has", "have", "had", "at", "into", "than",
        "then", "they", "their", "its", "also", "can", "may", "will",
    }

    def check(
        self,
        answer: str,
        retrieved_results: Iterable[dict[str, Any]],
        threshold: float = 0.45,
    ) -> GroundingResult:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        answer = answer.strip()
        if not answer:
            raise ValueError("answer cannot be empty")

        evidence = " ".join(
            str(item.get("text", ""))
            for item in retrieved_results
            if isinstance(item, dict)
        ).strip()

        answer_tokens = self._content_tokens(answer)
        evidence_tokens = self._content_tokens(evidence)

        if not answer_tokens:
            return GroundingResult(False, 0.0, 0, 0, (answer,))

        supported = sum(token in evidence_tokens for token in answer_tokens)
        score = supported / len(answer_tokens)

        unsupported = tuple(
            sentence
            for sentence in self._sentences(answer)
            if self._sentence_score(sentence, evidence_tokens) < threshold
        )

        grounded = score >= threshold and not unsupported

        return GroundingResult(
            grounded=grounded,
            score=round(score, 4),
            supported_tokens=supported,
            answer_tokens=len(answer_tokens),
            unsupported_sentences=unsupported,
        )

    def _sentence_score(
        self,
        sentence: str,
        evidence_tokens: set[str],
    ) -> float:
        tokens = self._content_tokens(sentence)
        if not tokens:
            return 1.0
        return sum(token in evidence_tokens for token in tokens) / len(tokens)

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text.lower()))
        return tokens - cls.STOPWORDS

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", text)
            if part.strip()
        ]

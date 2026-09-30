from dataclasses import dataclass
from enum import Enum
import re

from src.streaming.transcript_stream import TranscriptChunk


class RetrievalAction(str, Enum):
    """Possible decisions made by the retrieval controller."""

    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVAL = "NO_RETRIEVAL"


@dataclass(frozen=True)
class RetrievalDecision:
    """
    Decision produced by the retrieval controller.

    Attributes:
        action: WAIT, RETRIEVE, or NO_RETRIEVAL.
        reason: Human-readable explanation for the decision.
        query: Search query when retrieval is appropriate.
        confidence: Controller confidence in the decision.
        timestamp: Timestamp of the transcript chunk that triggered it.
    """

    action: RetrievalAction
    reason: str
    query: str | None
    confidence: float
    timestamp: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if self.timestamp < 0:
            raise ValueError("timestamp cannot be negative")

        if (
            self.action == RetrievalAction.RETRIEVE
            and not self.query
        ):
            raise ValueError(
                "RETRIEVE decision must contain a query"
            )

        if (
            self.action != RetrievalAction.RETRIEVE
            and self.query is not None
        ):
            raise ValueError(
                "Only RETRIEVE decisions may contain a query"
            )


class StreamController:
    """
    Retrieval controller for incremental transcript streams.

    The controller evaluates the accumulated utterance and decides
    whether the system should:

        WAIT          -> insufficiently stable information
        RETRIEVE      -> enough information for retrieval
        NO_RETRIEVAL  -> retrieval is unnecessary or already performed
    """

    # Phrases that generally indicate the user is asking to
    # transform or reuse an existing answer rather than retrieve.
    _NO_RETRIEVAL_PATTERNS = (
        r"\brepeat\b",
        r"\brephrase\b",
        r"\brephrase that\b",
        r"\bsummarize\b",
        r"\bshorten\b",
        r"\bmake it shorter\b",
        r"\bin bullet points\b",
        r"\bbullet points\b",
        r"\bformat\b",
        r"\brewrite\b",
        r"\bexplain your last answer\b",
        r"\bprevious answer\b",
        r"\blast answer\b",
    )

    # Weak opening fragments that should not trigger retrieval.
    _INCOMPLETE_PATTERNS = (
        r"\bi need to\b",
        r"\bi want to\b",
        r"\bcan you\b",
        r"\bcould you\b",
        r"\bplease\b",
        r"\bwhat is the\b",
        r"\bwhat are the\b",
        r"\bhow do i\b",
        r"\btell me about\b",
        r"\bi am looking for\b",
        r"\bi'm looking for\b",
    )

    # Terms that often indicate a concrete information need.
    _INFORMATION_WORDS = (
        "what",
        "which",
        "where",
        "when",
        "how",
        "why",
        "policy",
        "rules",
        "requirements",
        "procedure",
        "price",
        "cost",
        "capacity",
        "availability",
        "cancellation",
        "refund",
        "deadline",
        "eligibility",
        "eligible",
        "support",
        "options",
        "details",
        "information",
    )

    def __init__(
        self,
        minimum_words: int = 5,
        minimum_confidence: float = 0.70,
    ) -> None:
        if minimum_words < 1:
            raise ValueError("minimum_words must be positive")

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1"
            )

        self.minimum_words = minimum_words
        self.minimum_confidence = minimum_confidence

        # Stores the normalized form of the most recently retrieved
        # query so that repeated transcript chunks do not trigger
        # duplicate retrieval.
        self._last_retrieved_query: dict[str, str] = {}

    def decide(
        self,
        chunk: TranscriptChunk,
        accumulated_text: str,
    ) -> RetrievalDecision:
        """
        Decide what the system should do with the current transcript.

        Args:
            chunk: Latest transcript chunk.
            accumulated_text: Full transcript accumulated so far.

        Returns:
            RetrievalDecision.
        """

        # Normalized text is used only for classification.
        # The original accumulated text is preserved as the
        # retrieval query.
        text = self._normalize(accumulated_text)

        # 1. Empty/insufficient input.
        if not text:
            return self._wait(
                chunk,
                reason="empty_transcript",
                confidence=0.99,
            )

        # 2. Explicit requests that operate on the existing answer.
        if self._is_no_retrieval_request(text):
            return self._no_retrieval(
                chunk,
                reason="presentation_or_previous_answer_request",
                confidence=0.94,
            )

        # 3. Very short/incomplete transcript.
        if self._is_obviously_incomplete(text):
            return self._wait(
                chunk,
                reason="utterance_not_stable",
                confidence=0.90,
            )

        words = self._word_count(text)

        if words < self.minimum_words:
            return self._wait(
                chunk,
                reason="insufficient_context",
                confidence=0.88,
            )

        # 4. A final chunk or sufficiently stable early chunk can
        # be retrieved if it looks like an information request.
        if self._looks_like_information_request(text):
            confidence = self._retrieval_confidence(
                text=text,
                is_final=chunk.is_final,
            )

            if confidence >= self.minimum_confidence:
                query = accumulated_text.strip()

                # Avoid retrieving the same query repeatedly as
                # additional transcript chunks arrive.
                if self._is_duplicate_retrieval(
                    session_id=chunk.session_id,
                    query=query,
                ):
                    return self._no_retrieval(
                        chunk,
                        reason="query_already_retrieved",
                        confidence=0.96,
                    )

                return self._retrieve(
                    chunk,
                    query=query,
                    reason=(
                        "final_information_request"
                        if chunk.is_final
                        else "sufficient_context_for_early_retrieval"
                    ),
                    confidence=confidence,
                )

        # 5. If there is concrete entity/context but the request
        # is not yet stable, wait for another chunk.
        if self._has_concrete_context(text):
            return self._wait(
                chunk,
                reason="context_present_but_intent_not_stable",
                confidence=0.76,
            )

        return self._wait(
            chunk,
            reason="intent_not_stable",
            confidence=0.72,
        )

    def _normalize(self, text: str) -> str:
        """Normalize whitespace and punctuation for analysis."""

        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)

        # Treat terminal punctuation as equivalent for retrieval
        # deduplication and intent analysis.
        text = re.sub(r"[.!?]+$", "", text)

        return text

    def _word_count(self, text: str) -> int:
        """Count word-like tokens."""

        return len(
            re.findall(
                r"\b[\w'-]+\b",
                text,
            )
        )

    def _is_no_retrieval_request(self, text: str) -> bool:
        """Detect requests that operate on an existing answer."""

        return any(
            re.search(pattern, text)
            for pattern in self._NO_RETRIEVAL_PATTERNS
        )

    def _is_obviously_incomplete(self, text: str) -> bool:
        """Detect common incomplete conversational fragments."""

        if text.endswith(("...", "…")):
            return True

        has_incomplete_pattern = any(
            re.search(pattern, text)
            for pattern in self._INCOMPLETE_PATTERNS
        )

        if not has_incomplete_pattern:
            return False

        # An opening phrase such as "I need to" is not necessarily
        # incomplete once enough concrete context has arrived.
        if self._has_concrete_context(text):
            return False

        return not self._looks_like_information_request(text)

    def _looks_like_information_request(self, text: str) -> bool:
        """Determine whether the text contains information-seeking signals."""

        if "?" in text:
            return True

        if any(
            re.search(
                rf"\b{re.escape(word)}\b",
                text,
            )
            for word in self._INFORMATION_WORDS
        ):
            return True

        # Common request forms that don't necessarily contain
        # interrogative words.
        request_patterns = (
            r"\bfind\b",
            r"\bsearch\b",
            r"\bshow me\b",
            r"\blook for\b",
            r"\bcheck\b",
            r"\bcompare\b",
            r"\bneed information\b",
            r"\bi need to\b",
            r"\bi want to\b",
        )

        return any(
            re.search(pattern, text)
            for pattern in request_patterns
        )

    def _has_concrete_context(self, text: str) -> bool:
        """
        Detect whether the utterance contains useful concrete context.

        This is intentionally lightweight. More sophisticated semantic
        stability detection can be introduced later.
        """

        tokens = re.findall(
            r"\b[\w'-]+\b",
            text,
        )

        has_number = bool(
            re.search(
                r"\b\d+\b",
                text,
            )
        )

        contextual_terms = (
            "pune",
            "bangalore",
            "bengaluru",
            "mumbai",
            "workshop",
            "venue",
            "travel",
            "international",
            "refund",
            "reimbursement",
            "customer",
            "booking",
            "policy",
        )

        has_contextual_term = any(
            term in tokens
            for term in contextual_terms
        )

        return has_number or has_contextual_term

    def _retrieval_confidence(
        self,
        text: str,
        is_final: bool,
    ) -> float:
        """
        Estimate retrieval confidence.

        This is deliberately simple for the initial controller.
        The value is used for the controller decision, not as a
        probability claim.
        """

        confidence = 0.72

        if self._word_count(text) >= 8:
            confidence += 0.06

        if self._has_concrete_context(text):
            confidence += 0.07

        if "?" in text:
            confidence += 0.04

        if is_final:
            confidence += 0.06

        return min(
            confidence,
            0.99,
        )

    def _is_duplicate_retrieval(
        self,
        session_id: str,
        query: str,
    ) -> bool:
        """Check whether this query was already retrieved in this session."""

        normalized_query = self._normalize(query)

        last_query = self._last_retrieved_query.get(
            session_id
        )

        if last_query == normalized_query:
            return True

        self._last_retrieved_query[session_id] = normalized_query
        return False

    def _wait(
        self,
        chunk: TranscriptChunk,
        reason: str,
        confidence: float,
    ) -> RetrievalDecision:
        return RetrievalDecision(
            action=RetrievalAction.WAIT,
            reason=reason,
            query=None,
            confidence=confidence,
            timestamp=chunk.timestamp,
        )

    def _retrieve(
        self,
        chunk: TranscriptChunk,
        query: str,
        reason: str,
        confidence: float,
    ) -> RetrievalDecision:
        # Store normalized form for future duplicate detection,
        # while preserving the original query in the decision.
        self._last_retrieved_query[chunk.session_id] = self._normalize(query)

        return RetrievalDecision(
            action=RetrievalAction.RETRIEVE,
            reason=reason,
            query=query,
            confidence=confidence,
            timestamp=chunk.timestamp,
        )

    def _no_retrieval(
        self,
        chunk: TranscriptChunk,
        reason: str,
        confidence: float,
    ) -> RetrievalDecision:
        return RetrievalDecision(
            action=RetrievalAction.NO_RETRIEVAL,
            reason=reason,
            query=None,
            confidence=confidence,
            timestamp=chunk.timestamp,
        )

    def clear_session(self, session_id: str) -> None:
        """Release retrieval-controller state for a closed session."""

        self._last_retrieved_query.pop(
            session_id,
            None,
        )
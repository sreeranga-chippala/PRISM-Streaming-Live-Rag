from dataclasses import dataclass
from enum import Enum


class IntentType(str, Enum):
    INFORMATION = "information"
    ACTION = "action"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    CLARIFICATION = "clarification"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentResult:
    intent: IntentType
    confidence: float
    query: str
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not self.query.strip():
            raise ValueError("query cannot be empty")


class IntentDetector:
    """
    Detect the primary intent expressed in an accumulated transcript.

    This component identifies intent only. It does not decide whether
    retrieval should happen; that responsibility belongs to the
    streaming retrieval controller.
    """

    _COMPARISON_TERMS = (
        "compare",
        "comparison",
        "difference",
        "versus",
        "vs",
        "better than",
    )

    _SUMMARY_TERMS = (
        "summarize",
        "summary",
        "briefly",
        "in short",
        "key points",
    )

    _CLARIFICATION_TERMS = (
        "what do you mean",
        "can you clarify",
        "clarify",
        "explain that",
        "what does that mean",
    )

    _ACTION_TERMS = (
        "plan",
        "book",
        "schedule",
        "create",
        "find",
        "get",
        "show me",
        "help me",
    )

    _INFORMATION_TERMS = (
        "what",
        "which",
        "where",
        "when",
        "who",
        "why",
        "how",
        "policy",
        "rules",
        "requirements",
        "procedure",
        "price",
        "cost",
        "availability",
        "deadline",
        "eligibility",
        "information",
        "details",
    )

    def detect(self, query: str) -> IntentResult:
        """
        Detect the primary intent of a query.
        """

        normalized = " ".join(query.strip().lower().split())

        if not normalized:
            raise ValueError("query cannot be empty")

        # More specific intents are checked before broader ones.
        if self._contains_any(normalized, self._COMPARISON_TERMS):
            return IntentResult(
                intent=IntentType.COMPARISON,
                confidence=0.90,
                query=query.strip(),
                reason="comparison_signal_detected",
            )

        if self._contains_any(normalized, self._SUMMARY_TERMS):
            return IntentResult(
                intent=IntentType.SUMMARY,
                confidence=0.90,
                query=query.strip(),
                reason="summary_signal_detected",
            )

        if self._contains_any(normalized, self._CLARIFICATION_TERMS):
            return IntentResult(
                intent=IntentType.CLARIFICATION,
                confidence=0.88,
                query=query.strip(),
                reason="clarification_signal_detected",
            )

        if self._contains_any(normalized, self._ACTION_TERMS):
            return IntentResult(
                intent=IntentType.ACTION,
                confidence=0.84,
                query=query.strip(),
                reason="action_signal_detected",
            )

        if self._contains_any(normalized, self._INFORMATION_TERMS):
            return IntentResult(
                intent=IntentType.INFORMATION,
                confidence=0.82,
                query=query.strip(),
                reason="information_signal_detected",
            )

        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.50,
            query=query.strip(),
            reason="no_strong_intent_signal",
        )

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)
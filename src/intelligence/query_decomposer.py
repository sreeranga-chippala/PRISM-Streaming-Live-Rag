from dataclasses import dataclass


@dataclass(frozen=True)
class SubQuery:
    """
    Represents one independent retrieval query extracted from
    a multi-intent user request.
    """

    query: str
    intent: str
    index: int

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty")

        if not self.intent.strip():
            raise ValueError("intent cannot be empty")

        if self.index < 0:
            raise ValueError("index cannot be negative")


@dataclass(frozen=True)
class DecompositionResult:
    """
    Result produced by the query decomposer.
    """

    original_query: str
    sub_queries: tuple[SubQuery, ...]

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError("original_query cannot be empty")

        if not self.sub_queries:
            raise ValueError("at least one sub-query is required")


class QueryDecomposer:
    """
    Decomposes a user query into independent retrieval sub-queries.

    The first version uses deterministic linguistic signals.
    It does not perform retrieval and does not generate answers.
    """

    _INTENT_SEPARATORS = (
        " and ",
        " also ",
        " as well as ",
    )

    _COMPARISON_TERMS = (
        "compare",
        "comparison",
        "difference",
        "versus",
        " vs ",
    )

    _INFORMATION_TERMS = (
        "policy",
        "rules",
        "requirements",
        "procedure",
        "price",
        "pricing",
        "cost",
        "availability",
        "deadline",
        "eligibility",
        "details",
        "information",
    )

    def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a query into one or more retrieval sub-queries.
        """

        original_query = query.strip()

        if not original_query:
            raise ValueError("query cannot be empty")

        normalized = " ".join(original_query.lower().split())

        if not self._is_multi_intent(normalized):
            return DecompositionResult(
                original_query=original_query,
                sub_queries=(
                    SubQuery(
                        query=original_query,
                        intent="single_intent",
                        index=0,
                    ),
                ),
            )

        parts = self._split_query(original_query)

        sub_queries: list[SubQuery] = []

        for index, part in enumerate(parts):
            cleaned = part.strip(" ,.")
            if not cleaned:
                continue

            sub_queries.append(
                SubQuery(
                    query=cleaned,
                    intent=self._classify_sub_query(cleaned),
                    index=len(sub_queries),
                )
            )

        if not sub_queries:
            sub_queries.append(
                SubQuery(
                    query=original_query,
                    intent="single_intent",
                    index=0,
                )
            )

        return DecompositionResult(
            original_query=original_query,
            sub_queries=tuple(sub_queries),
        )

    def _is_multi_intent(self, normalized_query: str) -> bool:
        """
        Detect whether the query contains multiple information needs.
        """

        if self._contains_comparison(normalized_query):
            return True

        separator_count = sum(
            normalized_query.count(separator)
            for separator in self._INTENT_SEPARATORS
        )

        return separator_count > 0

    def _split_query(self, query: str) -> list[str]:
        """
        Split a multi-intent query while preserving the original casing.
        """

        normalized = query.lower()

        positions: list[tuple[int, int]] = []

        for separator in self._INTENT_SEPARATORS:
            start = 0

            while True:
                position = normalized.find(separator, start)

                if position == -1:
                    break

                positions.append(
                    (position, position + len(separator))
                )

                start = position + len(separator)

        if not positions:
            return [query]

        positions.sort()

        parts: list[str] = []
        start = 0

        for position, end in positions:
            parts.append(query[start:position])
            start = end

        parts.append(query[start:])

        return parts

    def _classify_sub_query(self, query: str) -> str:
        """
        Assign a lightweight intent label to a decomposed query.
        """

        normalized = query.lower()

        if self._contains_comparison(normalized):
            return "comparison"

        if any(
            term in normalized
            for term in self._INFORMATION_TERMS
        ):
            return "information"

        return "general"

    def _contains_comparison(self, query: str) -> bool:
        return any(
            term in query
            for term in self._COMPARISON_TERMS
        )
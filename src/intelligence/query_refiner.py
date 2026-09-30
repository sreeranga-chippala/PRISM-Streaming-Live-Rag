from dataclasses import dataclass


@dataclass(frozen=True)
class RefinedQuery:
    original_query: str
    refined_query: str
    changed: bool

    def __post_init__(self) -> None:
        if not self.original_query.strip():
            raise ValueError("original_query cannot be empty")
        if not self.refined_query.strip():
            raise ValueError("refined_query cannot be empty")


class QueryRefiner:
    _FILLER_PREFIXES = (
        "can you tell me ",
        "could you tell me ",
        "please tell me ",
        "i want to know ",
        "i need to know ",
        "can you explain ",
        "could you explain ",
    )

    _FILLER_WORDS = (
        "please",
        "actually",
        "basically",
        "just",
    )

    def refine(
        self,
        query: str,
        session_context: str | None = None,
    ) -> RefinedQuery:
        original_query = query.strip()

        if not original_query:
            raise ValueError("query cannot be empty")

        refined = " ".join(original_query.split())
        refined = self._remove_filler_prefix(refined)
        refined = self._remove_filler_words(refined)
        refined = refined.strip()

        if session_context and session_context.strip():
            refined = self._resolve_context_reference(
                refined,
                session_context.strip(),
            )

        return RefinedQuery(
            original_query=original_query,
            refined_query=refined,
            changed=refined != original_query,
        )

    def _remove_filler_prefix(self, query: str) -> str:
        normalized = query.lower()

        for prefix in self._FILLER_PREFIXES:
            if normalized.startswith(prefix):
                return query[len(prefix):].strip()

        return query

    def _remove_filler_words(self, query: str) -> str:
        words = query.split()
        filtered: list[str] = []

        for word in words:
            normalized_word = word.lower().strip(",.!?")

            if normalized_word in self._FILLER_WORDS:
                continue

            filtered.append(word)

        return " ".join(filtered)

    def _resolve_context_reference(
        self,
        query: str,
        session_context: str,
    ) -> str:
        normalized = query.lower()

        reference_terms = (
            "that policy",
            "this policy",
            "that venue",
            "this venue",
            "that option",
            "this option",
            "that procedure",
            "this procedure",
        )

        follow_up = normalized.startswith(
            ("and ", "also ", "what about ", "how about ")
        )

        if not follow_up and not any(
            term in normalized for term in reference_terms
        ):
            return query

        return f"{query} Context: {session_context}"

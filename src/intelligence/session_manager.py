from dataclasses import dataclass, field


@dataclass(frozen=True)
class SessionEvent:
    """
    One meaningful event recorded during a streaming session.
    """

    event_type: str
    content: str
    timestamp: float

    def __post_init__(self) -> None:
        if not self.event_type.strip():
            raise ValueError("event_type cannot be empty")

        if not self.content.strip():
            raise ValueError("content cannot be empty")

        if self.timestamp < 0:
            raise ValueError("timestamp cannot be negative")


@dataclass
class SessionState:
    """
    Mutable state associated with one live RAG session.
    """

    session_id: str
    transcript: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    current_answer: str = ""
    current_query: str = ""
    current_citations: list[dict] = field(default_factory=list)
    answer_version: int = 0
    answer_history: list[dict] = field(default_factory=list)
    last_answer_timestamp: float | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")


class SessionManager:
    """
    Maintains state for active streaming RAG sessions.

    The manager stores conversational state so later components can
    perform session-aware refinement and incremental synthesis.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, session_id: str) -> SessionState:
        """
        Create a new session.

        Raises:
            ValueError: If the session already exists.
        """

        if not session_id.strip():
            raise ValueError("session_id cannot be empty")

        if session_id in self._sessions:
            raise ValueError(
                f"Session already exists: {session_id}"
            )

        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session

        return session

    def get_session(self, session_id: str) -> SessionState:
        """
        Retrieve an existing session.
        """

        if session_id not in self._sessions:
            raise KeyError(
                f"Session not found: {session_id}"
            )

        return self._sessions[session_id]

    def get_or_create(self, session_id: str) -> SessionState:
        """
        Return an existing session or create a new one.
        """

        if session_id in self._sessions:
            return self._sessions[session_id]

        return self.create_session(session_id)

    def add_transcript(
        self,
        session_id: str,
        text: str,
        timestamp: float,
    ) -> None:
        """
        Add transcript text to the session.
        """

        session = self.get_session(session_id)

        if not text.strip():
            raise ValueError("transcript text cannot be empty")

        if timestamp < 0:
            raise ValueError("timestamp cannot be negative")

        session.transcript.append(text.strip())

        session.events.append(
            SessionEvent(
                event_type="transcript",
                content=text.strip(),
                timestamp=timestamp,
            )
        )

    def add_query(
        self,
        session_id: str,
        query: str,
        timestamp: float,
    ) -> None:
        """
        Record a query that was considered for retrieval.
        """

        session = self.get_session(session_id)

        if not query.strip():
            raise ValueError("query cannot be empty")

        session.queries.append(query.strip())

        session.events.append(
            SessionEvent(
                event_type="query",
                content=query.strip(),
                timestamp=timestamp,
            )
        )

    def add_intent(
        self,
        session_id: str,
        intent: str,
        timestamp: float,
    ) -> None:
        """
        Record a detected intent.
        """

        session = self.get_session(session_id)

        if not intent.strip():
            raise ValueError("intent cannot be empty")

        session.intents.append(intent.strip())

        session.events.append(
            SessionEvent(
                event_type="intent",
                content=intent.strip(),
                timestamp=timestamp,
            )
        )

    def add_retrieved_context(
        self,
        session_id: str,
        context: str,
        timestamp: float,
    ) -> None:
        """
        Store retrieved context for later session-aware synthesis.
        """

        session = self.get_session(session_id)

        if not context.strip():
            raise ValueError("context cannot be empty")

        session.retrieved_context.append(context.strip())

        session.events.append(
            SessionEvent(
                event_type="retrieval",
                content=context.strip(),
                timestamp=timestamp,
            )
        )

    def update_answer(
        self,
        session_id: str,
        answer: str,
        timestamp: float,
        *,
        query: str | None = None,
        citations: list[dict] | None = None,
        reason: str = "initial",
    ) -> int:
        """
        Replace the current provisional/final answer.
        """

        session = self.get_session(session_id)

        if not answer.strip():
            raise ValueError("answer cannot be empty")

        previous_answer = session.current_answer
        session.answer_version += 1
        session.current_answer = answer.strip()
        if query is not None and query.strip():
            session.current_query = query.strip()
        if citations is not None:
            session.current_citations = list(citations)
        session.last_answer_timestamp = timestamp

        session.answer_history.append(
            {
                "version": session.answer_version,
                "answer": session.current_answer,
                "query": session.current_query,
                "citations": list(session.current_citations),
                "reason": reason,
                "timestamp": timestamp,
                "replaced_previous_answer": bool(previous_answer),
            }
        )

        session.events.append(
            SessionEvent(
                event_type="answer",
                content=answer.strip(),
                timestamp=timestamp,
            )
        )
        return session.answer_version

    def set_current_query(
        self,
        session_id: str,
        query: str,
        timestamp: float,
    ) -> None:
        session = self.get_session(session_id)
        if not query.strip():
            raise ValueError("query cannot be empty")
        session.current_query = query.strip()
        session.events.append(
            SessionEvent(
                event_type="query_state",
                content=query.strip(),
                timestamp=timestamp,
            )
        )

    def set_current_citations(
        self,
        session_id: str,
        citations: list[dict],
    ) -> None:
        self.get_session(session_id).current_citations = list(citations)

    def get_answer_state(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        return {
            "answer": session.current_answer,
            "query": session.current_query,
            "citations": list(session.current_citations),
            "version": session.answer_version,
        }

    def get_context(self, session_id: str) -> str:
        """
        Return the most useful stored conversational context.
        """

        session = self.get_session(session_id)

        parts: list[str] = []

        if session.transcript:
            parts.append(
                "Transcript: " + " ".join(session.transcript)
            )

        if session.queries:
            parts.append(
                "Queries: " + " | ".join(session.queries)
            )

        if session.intents:
            parts.append(
                "Intents: " + " | ".join(session.intents)
            )

        if session.retrieved_context:
            parts.append(
                "Retrieved context: "
                + " | ".join(session.retrieved_context)
            )

        if session.current_query:
            parts.append("Current query: " + session.current_query)

        if session.current_answer:
            parts.append(
                "Current answer: " + session.current_answer
            )

        if session.answer_version:
            parts.append(f"Answer version: {session.answer_version}")

        return "\n".join(parts)

    def close_session(self, session_id: str) -> SessionState:
        """
        Remove and return a completed session.
        """

        if session_id not in self._sessions:
            raise KeyError(
                f"Session not found: {session_id}"
            )

        return self._sessions.pop(session_id)

    def has_session(self, session_id: str) -> bool:
        """
        Check whether a session currently exists.
        """

        return session_id in self._sessions
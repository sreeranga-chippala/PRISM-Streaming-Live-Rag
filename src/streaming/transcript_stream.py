from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TranscriptChunk:
    """
    Represents one incremental transcript fragment.

    A chunk may arrive while the user is still speaking.
    """

    session_id: str
    text: str
    timestamp: float
    is_final: bool = False
    sequence_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id cannot be empty")

        if not self.text.strip():
            raise ValueError("Transcript chunk text cannot be empty")

        if self.timestamp < 0:
            raise ValueError("timestamp cannot be negative")


class TranscriptStream:
    """
    Maintains the incremental transcript for one conversation session.
    """

    def __init__(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id cannot be empty")

        self.session_id = session_id
        self._chunks: list[TranscriptChunk] = []

    def add_chunk(self, chunk: TranscriptChunk) -> None:
        """
        Add an incoming transcript chunk.

        Chunks must belong to the current session.
        """

        if chunk.session_id != self.session_id:
            raise ValueError(
                "Transcript chunk belongs to a different session"
            )

        self._chunks.append(chunk)

    @property
    def chunks(self) -> list[TranscriptChunk]:
        """Return all received transcript chunks."""
        return list(self._chunks)

    @property
    def current_text(self) -> str:
        """Return the complete transcript accumulated so far."""
        return " ".join(chunk.text.strip() for chunk in self._chunks)

    @property
    def is_complete(self) -> bool:
        """Return True when a final transcript chunk has arrived."""
        return bool(self._chunks and self._chunks[-1].is_final)

    def reset(self) -> None:
        """Clear the current utterance."""
        self._chunks.clear()
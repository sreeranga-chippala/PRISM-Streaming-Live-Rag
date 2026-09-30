from dataclasses import dataclass

from src.streaming.transcript_stream import TranscriptChunk, TranscriptStream


@dataclass(frozen=True)
class ChunkProcessingResult:
    """
    Result produced after processing an incoming transcript chunk.
    """

    session_id: str
    accumulated_text: str
    is_final: bool
    chunk_count: int


class ChunkHandler:
    """
    Processes incremental transcript chunks and maintains
    the current utterance state.
    """

    def __init__(self, session_id: str) -> None:
        self.stream = TranscriptStream(session_id)

    def process(self, chunk: TranscriptChunk) -> ChunkProcessingResult:
        """
        Process one incoming transcript chunk.
        """

        self.stream.add_chunk(chunk)

        return ChunkProcessingResult(
            session_id=self.stream.session_id,
            accumulated_text=self.stream.current_text,
            is_final=self.stream.is_complete,
            chunk_count=len(self.stream.chunks),
        )

    def reset(self) -> None:
        """Reset the current utterance."""
        self.stream.reset()
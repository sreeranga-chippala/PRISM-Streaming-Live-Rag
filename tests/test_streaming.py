from src.streaming.stream_controller import (
    RetrievalAction,
    StreamController,
)
from src.streaming.transcript_stream import TranscriptChunk


def make_chunk(
    text: str,
    timestamp: float,
    is_final: bool = False,
) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="test-session",
        text=text,
        timestamp=timestamp,
        is_final=is_final,
    )


def test_short_fragment_waits():
    controller = StreamController()

    decision = controller.decide(
        make_chunk("I need to", 0.0),
        "I need to",
    )

    assert decision.action == RetrievalAction.WAIT
    assert decision.query is None


def test_early_retrieval_from_sufficient_context():
    controller = StreamController()

    text = "I need to plan a customer workshop in Pune"

    decision = controller.decide(
        make_chunk(text, 0.8),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.query == text
    assert decision.reason == "sufficient_context_for_early_retrieval"


def test_final_information_request_retrieves():
    controller = StreamController()

    text = "What is the cancellation policy for this venue?"

    decision = controller.decide(
        make_chunk(text, 2.1, is_final=True),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.reason == "final_information_request"


def test_presentation_request_suppresses_retrieval():
    controller = StreamController()

    text = "Please repeat your last answer in bullet points."

    decision = controller.decide(
        make_chunk(text, 2.0, is_final=True),
        text,
    )

    assert decision.action == RetrievalAction.NO_RETRIEVAL
    assert decision.query is None


def test_retrieve_requires_query():
    controller = StreamController()

    text = "What is the refund policy?"

    decision = controller.decide(
        make_chunk(text, 1.0),
        text,
    )

    assert decision.action == RetrievalAction.RETRIEVE
    assert decision.query


def test_decision_timestamp_matches_chunk():
    controller = StreamController()

    chunk = make_chunk(
        "I need to plan a workshop in Pune",
        0.8,
    )

    decision = controller.decide(
        chunk,
        chunk.text,
    )

    assert decision.timestamp == 0.8


def test_duplicate_retrieval_is_suppressed():
    controller = StreamController()

    first_text = "I need to plan a customer workshop in Pune"

    first_decision = controller.decide(
        make_chunk(first_text, 0.8),
        first_text,
    )

    assert first_decision.action == RetrievalAction.RETRIEVE

    second_text = "I need to plan a customer workshop in Pune."

    second_decision = controller.decide(
        make_chunk(second_text, 2.1, is_final=True),
        second_text,
    )

    assert second_decision.action == RetrievalAction.NO_RETRIEVAL
    assert second_decision.reason == "query_already_retrieved"
    assert second_decision.query is None


def test_changed_query_triggers_new_retrieval():
    controller = StreamController()

    first_text = "I need to plan a customer workshop in Pune"

    first_decision = controller.decide(
        make_chunk(first_text, 0.8),
        first_text,
    )

    assert first_decision.action == RetrievalAction.RETRIEVE

    changed_text = (
        "I need to plan a customer workshop in Pune for 50 people"
    )

    changed_decision = controller.decide(
        make_chunk(changed_text, 1.6),
        changed_text,
    )

    assert changed_decision.action == RetrievalAction.RETRIEVE
    assert changed_decision.query == changed_text


def test_duplicate_retrieval_is_session_scoped():
    controller = StreamController()

    chunk_a = TranscriptChunk(
        session_id="session-a",
        text="I need to plan a customer workshop in Pune",
        timestamp=0.0,
    )

    chunk_b = TranscriptChunk(
        session_id="session-b",
        text="I need to plan a customer workshop in Pune",
        timestamp=0.0,
    )

    first = controller.decide(
        chunk=chunk_a,
        accumulated_text=chunk_a.text,
    )

    second = controller.decide(
        chunk=chunk_b,
        accumulated_text=chunk_b.text,
    )

    assert first.action == RetrievalAction.RETRIEVE
    assert second.action == RetrievalAction.RETRIEVE


def test_duplicate_retrieval_is_suppressed_within_session():
    controller = StreamController()

    chunk = TranscriptChunk(
        session_id="session-a",
        text="I need to plan a customer workshop in Pune",
        timestamp=0.0,
    )

    first = controller.decide(
        chunk=chunk,
        accumulated_text=chunk.text,
    )

    second = controller.decide(
        chunk=chunk,
        accumulated_text=chunk.text,
    )

    assert first.action == RetrievalAction.RETRIEVE
    assert second.action == RetrievalAction.NO_RETRIEVAL
    assert second.reason == "query_already_retrieved"


def test_final_streaming_question_retrieves_after_fragment():
    controller = StreamController()

    first_text = "Who is eligible"

    first_decision = controller.decide(
        make_chunk(first_text, 0.5),
        first_text,
    )

    assert first_decision.action == RetrievalAction.WAIT
    assert first_decision.query is None

    final_text = "Who is eligible for work from home?"

    final_decision = controller.decide(
        make_chunk(
            "for work from home?",
            1.0,
            is_final=True,
        ),
        final_text,
    )

    assert final_decision.action == RetrievalAction.RETRIEVE
    assert final_decision.query == final_text
    assert final_decision.reason == "final_information_request"
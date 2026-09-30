from src.orchestration.pipeline import StreamingRAGPipeline
from src.streaming.stream_controller import RetrievalAction
from src.streaming.transcript_stream import TranscriptChunk


def make_chunk(
    text: str,
    timestamp: float,
    is_final: bool = False,
) -> TranscriptChunk:
    return TranscriptChunk(
        session_id="pipeline-session",
        text=text,
        timestamp=timestamp,
        is_final=is_final,
    )


def test_pipeline_waits_for_incomplete_query():
    pipeline = StreamingRAGPipeline()

    result = pipeline.process_chunk(
        make_chunk("I need to", 0.0)
    )

    assert result.retrieval_decision.action == RetrievalAction.WAIT
    assert result.intent is None
    assert result.refined_queries == ()


def test_pipeline_processes_early_retrieval():
    pipeline = StreamingRAGPipeline()

    result = pipeline.process_chunk(
        make_chunk(
            "I need to plan a customer workshop in Pune",
            0.8,
        )
    )

    assert result.retrieval_decision.action == RetrievalAction.RETRIEVE
    assert result.intent is not None
    assert result.decomposition is not None
    assert result.refined_queries


def test_pipeline_suppresses_previous_answer_request():
    pipeline = StreamingRAGPipeline()

    result = pipeline.process_chunk(
        make_chunk(
            "Please repeat your last answer in bullet points.",
            2.0,
            is_final=True,
        )
    )

    assert (
        result.retrieval_decision.action
        == RetrievalAction.NO_RETRIEVAL
    )

    assert result.intent is None
    assert result.refined_queries == ()


def test_pipeline_preserves_session_state():
    pipeline = StreamingRAGPipeline()

    result = pipeline.process_chunk(
        make_chunk(
            "I need to plan a customer workshop in Pune",
            0.8,
        )
    )

    session = result.session

    assert session.session_id == "pipeline-session"
    assert len(session.transcript) == 1
    assert len(session.queries) == 1
    assert len(session.intents) == 1

    
class MockMultiQueryRetriever:
    def __init__(self):
        self.received_queries = []

    def retrieve(self, subqueries):
        self.received_queries = list(subqueries)

        return [
            {
                "chunk_id": "test_chunk_001",
                "text": "Customer workshops require advance planning.",
                "metadata": {
                    "source": "test_document.txt",
                    "section": "1.1",
                },
                "reranker_score": 0.91,
                "reranker_rank": 1,
            }
        ]

    
def test_pipeline_integrates_retrieval_results():
    mock_retriever = MockMultiQueryRetriever()

    pipeline = StreamingRAGPipeline(
        multi_query_retriever=mock_retriever,
    )

    result = pipeline.process_chunk(
        make_chunk(
            "I need to plan a customer workshop in Pune",
            0.8,
        )
    )

    assert result.retrieval_decision.action == RetrievalAction.RETRIEVE

    assert result.refined_queries

    assert mock_retriever.received_queries == [
        refined_query.refined_query
        for refined_query in result.refined_queries
    ]

    assert result.retrieved_results

    retrieved = result.retrieved_results[0]

    assert retrieved["chunk_id"] == "test_chunk_001"
    assert retrieved["text"] == (
        "Customer workshops require advance planning."
    )
    assert retrieved["metadata"]["source"] == (
        "test_document.txt"
    )
    assert retrieved["metadata"]["section"] == "1.1"
    assert retrieved["reranker_score"] == 0.91
    assert retrieved["reranker_rank"] == 1

    assert result.session.retrieved_context == [
        "Customer workshops require advance planning."
    ]
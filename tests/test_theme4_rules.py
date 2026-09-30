from src.generation.citation_manager import CitationManager
from src.generation.grounding_checker import GroundingChecker
from src.intelligence.session_manager import SessionManager
from src.orchestration.pipeline import StreamingRAGPipeline
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.streaming.stream_controller import RetrievalAction
from src.streaming.transcript_stream import TranscriptChunk


class FakeHybrid:
    def __init__(self):
        self.calls = []

    def search(self, query, semantic_top_k=None, keyword_top_k=None):
        self.calls.append(query)
        return {
            "semantic_results": [{
                "chunk_id": f"s-{query}",
                "text": query,
                "metadata": {"source": "corpus"},
                "score": 1.0,
            }],
            "keyword_results": [{
                "chunk_id": f"k-{query}",
                "text": query,
                "metadata": {"source": "corpus"},
                "score": 1.0,
            }],
        }


class FakeReranker:
    def rerank(self, query, candidates, top_k=None):
        return candidates[:top_k] if top_k else candidates


def test_parallel_multi_query_preserves_all_subqueries():
    hybrid = FakeHybrid()
    retriever = MultiQueryRetriever(
        hybrid,
        ReciprocalRankFusion(),
        FakeReranker(),
        max_workers=2,
    )
    results = retriever.retrieve(["alpha policy", "beta policy"])
    assert set(hybrid.calls) == {"alpha policy", "beta policy"}
    assert results


def test_session_answer_version_increments_and_preserves_history():
    manager = SessionManager()
    manager.create_session("s")
    v1 = manager.update_answer(
        "s", "Initial answer.", 1.0, query="travel policy", citations=[], reason="initial"
    )
    v2 = manager.update_answer(
        "s", "Refined answer.", 2.0, query="international travel", citations=[], reason="late_detail_refinement"
    )
    session = manager.get_session("s")
    assert (v1, v2) == (1, 2)
    assert session.answer_version == 2
    assert len(session.answer_history) == 2


def test_late_detail_is_delta_retrieval_and_keeps_session():
    class FakeRetriever:
        def __init__(self):
            self.queries = []
        def retrieve(self, queries):
            self.queries.append(list(queries))
            return [{
                "chunk_id": "delta-1",
                "text": "International travel requires additional verification.",
                "metadata": {"source": "travel.pdf"},
            }]

    fake = FakeRetriever()
    pipeline = StreamingRAGPipeline(multi_query_retriever=fake)

    first = pipeline.process_chunk(
        TranscriptChunk(
            session_id="s",
            text="I need to understand the travel reimbursement rule for an employee trip",
            timestamp=0.8,
            is_final=True,
        )
    )
    pipeline.session_manager.update_answer(
        "s",
        "The standard reimbursement rule applies.",
        1.0,
        query=first.retrieval_decision.query,
        citations=[],
        reason="initial",
    )

    second = pipeline.process_chunk(
        TranscriptChunk(
            session_id="s",
            text="The trip was international and the booking was made after travel.",
            timestamp=2.0,
            is_final=True,
        )
    )

    assert second.is_refinement is True
    assert second.effective_query
    assert "international" in second.effective_query.lower()
    assert fake.queries[-1] == [
        item.refined_query for item in second.refined_queries
    ]
    assert pipeline.session_manager.get_session("s").current_answer == (
        "The standard reimbursement rule applies."
    )


def test_presentation_request_does_not_retrieve():
    pipeline = StreamingRAGPipeline()
    pipeline.session_manager.create_session("s")
    pipeline.session_manager.update_answer(
        "s",
        "One established fact.",
        1.0,
        query="policy",
        citations=[],
    )
    result = pipeline.process_chunk(
        TranscriptChunk(
            session_id="s",
            text="Please repeat your last answer in two bullet points.",
            timestamp=2.0,
            is_final=True,
        )
    )
    assert result.retrieval_decision.action == RetrievalAction.NO_RETRIEVAL
    assert result.retrieved_results == ()


def test_citations_only_reference_retrieved_chunks_and_grounding():
    evidence = [{
        "chunk_id": "doc-1",
        "text": "Employees must complete six months of service before WFH.",
        "source": "03_Work_From_Home_Policy.pdf",
        "reranker_rank": 1,
    }]
    answer = "Employees must complete six months of service before WFH."
    citations = CitationManager().build_citations(answer, evidence)
    grounding = GroundingChecker().check(answer, evidence)
    assert citations
    assert all(item.chunk_id == "doc-1" for item in citations)
    assert grounding.grounded
    assert CitationManager().support_coverage(answer, evidence, citations) == 1.0

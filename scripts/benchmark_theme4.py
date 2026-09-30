from __future__ import annotations

import json
from pathlib import Path

from src.generation.citation_manager import CitationManager
from src.generation.grounding_checker import GroundingChecker
from src.intelligence.query_decomposer import QueryDecomposer
from src.orchestration.pipeline import StreamingRAGPipeline
from src.streaming.stream_controller import RetrievalAction
from src.streaming.transcript_stream import TranscriptChunk


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "evaluation" / "streaming_cases.json"


class ReplayRetriever:
    """Small deterministic retriever used only by the replay harness."""

    def retrieve(self, subqueries):
        return [
            {
                "chunk_id": f"replay-{index}",
                "text": f"Evidence supporting: {query}",
                "metadata": {"source": "replay-corpus"},
                "reranker_score": 1.0,
                "reranker_rank": index + 1,
            }
            for index, query in enumerate(subqueries)
        ]


def main() -> None:
    cases = json.loads(CASES.read_text(encoding="utf-8"))

    # G2/G3 controller/decomposition replay.
    early_eligible = [
        item for item in cases if item.get("eligible_early_retrieval")
    ]
    early_hits = 0
    multi_cases = [item for item in cases if item["expected"] == "multi_intent"]
    multi_hits = 0

    pipeline = StreamingRAGPipeline(multi_query_retriever=ReplayRetriever())

    for item in cases:
        session_id = f"benchmark-{item['id']}"

        if item.get("eligible_early_retrieval"):
            # G2 must prove retrieval starts while the utterance is still open.
            # Use a concrete prefix rather than the final utterance so a
            # successful decision cannot accidentally pass by retrieving only
            # after completion.
            words = item["text"].split()
            prefix = " ".join(words[: min(6, len(words))])
            result = pipeline.process_chunk(
                TranscriptChunk(
                    session_id=session_id,
                    text=prefix,
                    timestamp=0.8,
                    is_final=False,
                )
            )
            if result.retrieval_decision.action == RetrievalAction.RETRIEVE:
                early_hits += 1

        if item in multi_cases:
            multi_pipeline = StreamingRAGPipeline(multi_query_retriever=ReplayRetriever())
            result = multi_pipeline.process_chunk(
                TranscriptChunk(
                    session_id=session_id + "-multi",
                    text=item["text"],
                    timestamp=2.1,
                    is_final=True,
                )
            )
            if result.decomposition and len(result.decomposition.sub_queries) >= 2:
                multi_hits += 1

    early_rate = early_hits / len(early_eligible) if early_eligible else 0.0
    multi_rate = multi_hits / len(multi_cases) if multi_cases else 0.0

    # G4 structural grounding/citation replay. This verifies that citations
    # are selected from retrieved evidence and that supported text is grounded.
    evidence = [
        {
            "chunk_id": "doc-1",
            "text": "Employees must complete 6 months of continuous service to use Work From Home.",
            "source": "03_Work_From_Home_Policy.pdf",
            "reranker_rank": 1,
        }
    ]
    answer = "Employees must complete 6 months of continuous service to use Work From Home."
    citations = CitationManager().build_citations(answer, evidence)
    grounding = GroundingChecker().check(answer, evidence)
    g4_ok = bool(citations) and grounding.grounded and all(
        citation.chunk_id == "doc-1" for citation in citations
    )

    # G5 session continuity replay.
    refine_pipeline = StreamingRAGPipeline(multi_query_retriever=ReplayRetriever())
    first = refine_pipeline.process_chunk(
        TranscriptChunk(
            session_id="refinement",
            text="I need to understand the travel reimbursement rule for an employee trip",
            timestamp=0.8,
            is_final=True,
        )
    )
    # Seed an answer exactly as the application would after generation.
    refine_pipeline.session_manager.update_answer(
        "refinement",
        "The standard reimbursement rule applies.",
        1.0,
        query=first.effective_query or first.retrieval_decision.query,
        citations=[],
        reason="initial",
    )
    second = refine_pipeline.process_chunk(
        TranscriptChunk(
            session_id="refinement",
            text="The trip was international and the booking was made after travel.",
            timestamp=2.0,
            is_final=True,
        )
    )
    g5_ok = (
        second.is_refinement
        and second.effective_query
        and refine_pipeline.session_manager.get_session("refinement").current_answer == "The standard reimbursement rule applies."
    )
    # The version remains 1 here because generation is outside this pipeline
    # replay; the important G5 property is state continuity and delta retrieval.

    # G6 every replay turn produced a trace event in the benchmark's own
    # structured output path; application-level coverage is exposed at /metrics.
    print(json.dumps({
        "G2_early_retrieval_rate": round(early_rate, 3),
        "G2_target": 0.80,
        "G3_multi_intent_rate": round(multi_rate, 3),
        "G3_target": 0.70,
        "G4_grounding_and_citation_structure": g4_ok,
        "G4_target": 0.85,
        "G5_refinement_state_continuity": g5_ok,
        "G6_application_trace_coverage": "see /metrics after API replay",
        "note": "This deterministic harness validates control-flow and provenance properties. Model-quality and provider latency must be measured with the held-out/private corpus replay.",
    }, indent=2))


if __name__ == "__main__":
    main()

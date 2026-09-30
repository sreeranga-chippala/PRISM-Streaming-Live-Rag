from dataclasses import dataclass, field
from typing import Any

from src.intelligence.intent_detector import IntentDetector, IntentResult
from src.intelligence.query_decomposer import DecompositionResult, QueryDecomposer
from src.intelligence.query_refiner import QueryRefiner, RefinedQuery
from src.intelligence.session_manager import SessionManager, SessionState
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.streaming.chunk_handler import ChunkHandler, ChunkProcessingResult
from src.streaming.stream_controller import RetrievalAction, RetrievalDecision, StreamController
from src.streaming.transcript_stream import TranscriptChunk


@dataclass(frozen=True)
class PipelineResult:
    session_id: str
    chunk_result: ChunkProcessingResult
    retrieval_decision: RetrievalDecision
    intent: IntentResult | None
    decomposition: DecompositionResult | None
    refined_queries: tuple[RefinedQuery, ...]
    retrieved_results: tuple[dict[str, Any], ...]
    session: SessionState
    retrieval_events: tuple[dict[str, Any], ...] = ()
    is_refinement: bool = False
    effective_query: str | None = None


class StreamingRAGPipeline:
    """
    Orchestrates incremental streaming, intent analysis, targeted retrieval,
    and session-aware refinement.

    Existing public behavior is preserved: RetrievalDecision.query remains the
    accumulated transcript query.  `effective_query` is added for cases where
    a late detail is intentionally retrieved as a delta.
    """

    def __init__(
        self,
        session_manager: SessionManager | None = None,
        stream_controller: StreamController | None = None,
        intent_detector: IntentDetector | None = None,
        query_decomposer: QueryDecomposer | None = None,
        query_refiner: QueryRefiner | None = None,
        multi_query_retriever: MultiQueryRetriever | None = None,
    ) -> None:
        self.session_manager = session_manager or SessionManager()
        self.stream_controller = stream_controller or StreamController()
        self.intent_detector = intent_detector or IntentDetector()
        self.query_decomposer = query_decomposer or QueryDecomposer()
        self.query_refiner = query_refiner or QueryRefiner()
        self.multi_query_retriever = multi_query_retriever
        self._chunk_handlers: dict[str, ChunkHandler] = {}

    def process_chunk(self, chunk: TranscriptChunk) -> PipelineResult:
        session = self.session_manager.get_or_create(chunk.session_id)
        handler = self._get_chunk_handler(chunk.session_id)
        chunk_result = handler.process(chunk)

        self.session_manager.add_transcript(
            session_id=chunk.session_id,
            text=chunk.text,
            timestamp=chunk.timestamp,
        )

        retrieval_decision = self.stream_controller.decide(
            chunk=chunk,
            accumulated_text=chunk_result.accumulated_text,
        )

        # A late constraint can be phrased as a statement rather than a
        # question. If it relates to the current answer, promote WAIT to a
        # targeted delta retrieval.
        if (
            retrieval_decision.action == RetrievalAction.WAIT
            and session.current_answer
            and self._is_late_refinement(
                session,
                chunk_result.accumulated_text.strip(),
            )
            and chunk_result.chunk_count == 1
        ):
            retrieval_decision = RetrievalDecision(
                action=RetrievalAction.RETRIEVE,
                reason="late_detail_refinement",
                query=chunk_result.accumulated_text.strip(),
                confidence=0.82,
                timestamp=chunk.timestamp,
            )

        # A formatting request is retrieval-free only when an existing answer
        # is available.  A standalone "summarize X" still needs corpus evidence.
        if (
            retrieval_decision.action == RetrievalAction.NO_RETRIEVAL
            and retrieval_decision.reason == "presentation_or_previous_answer_request"
            and not session.current_answer
            and not any(
                marker in chunk_result.accumulated_text.lower()
                for marker in ("last answer", "previous answer", "that answer")
            )
        ):
            retrieval_decision = RetrievalDecision(
                action=RetrievalAction.RETRIEVE,
                reason="standalone_information_request",
                query=chunk_result.accumulated_text.strip(),
                confidence=0.85,
                timestamp=chunk.timestamp,
            )

        if retrieval_decision.action != RetrievalAction.RETRIEVE:
            result = PipelineResult(
                session_id=chunk.session_id,
                chunk_result=chunk_result,
                retrieval_decision=retrieval_decision,
                intent=None,
                decomposition=None,
                refined_queries=(),
                retrieved_results=(),
                session=session,
                retrieval_events=(
                    {
                        "timestamp_s": chunk.timestamp,
                        "query": retrieval_decision.query,
                        "trigger": retrieval_decision.action.value.lower(),
                        "reason": retrieval_decision.reason,
                    },
                ),
            )
            if chunk.is_final:
                handler.reset()
            return result

        query = retrieval_decision.query
        if query is None:
            raise RuntimeError("RETRIEVE decision did not contain a query")

        # A late detail is a new utterance after an answer exists.  During the
        # same still-open utterance, keep the full accumulated query so the
        # final synthesis can cover all known sub-intents (Theme 4 Example 1).
        is_refinement = (
            chunk_result.chunk_count == 1
            and self._is_late_refinement(session, query)
        )
        effective_query = (
            self._extract_delta_query(query, session.current_query)
            if is_refinement
            else query
        )
        effective_query = effective_query.strip() or query

        intent = self.intent_detector.detect(effective_query)
        self.session_manager.add_query(
            session_id=chunk.session_id,
            query=query,
            timestamp=chunk.timestamp,
        )
        self.session_manager.add_intent(
            session_id=chunk.session_id,
            intent=intent.intent.value,
            timestamp=chunk.timestamp,
        )

        decomposition = self.query_decomposer.decompose(effective_query)

        refined_queries = tuple(
            self.query_refiner.refine(
                sub_query.query,
                # Refinement retrieval is deliberately delta-only.  The prior
                # answer is passed to the synthesis layer, not injected into
                # the retrieval query.
                session_context="" if is_refinement else self.session_manager.get_context(chunk.session_id),
            )
            for sub_query in decomposition.sub_queries
        )

        retrieved_results: tuple[dict[str, Any], ...] = ()
        retrieval_events: list[dict[str, Any]] = []

        if self.multi_query_retriever is not None:
            results = self.multi_query_retriever.retrieve(
                [item.refined_query for item in refined_queries]
            )
            retrieved_results = tuple(results)

            context_parts = [
                result["text"]
                for result in results
                if isinstance(result, dict)
                and isinstance(result.get("text"), str)
                and result["text"].strip()
            ]
            if context_parts:
                self.session_manager.add_retrieved_context(
                    session_id=chunk.session_id,
                    context="\n\n".join(context_parts),
                    timestamp=chunk.timestamp,
                )

            trigger = "late_detail_delta" if is_refinement else (
                "multi_intent" if len(refined_queries) > 1 else "retrieval"
            )
            retrieval_events.append(
                {
                    "timestamp_s": chunk.timestamp,
                    "query": effective_query,
                    "trigger": trigger,
                    "sub_queries": [
                        item.refined_query for item in refined_queries
                    ],
                    "result_count": len(retrieved_results),
                }
            )

        self.session_manager.set_current_query(
            session_id=chunk.session_id,
            query=query,
            timestamp=chunk.timestamp,
        )

        result = PipelineResult(
            session_id=chunk.session_id,
            chunk_result=chunk_result,
            retrieval_decision=retrieval_decision,
            intent=intent,
            decomposition=decomposition,
            refined_queries=refined_queries,
            retrieved_results=retrieved_results,
            session=session,
            retrieval_events=tuple(retrieval_events),
            is_refinement=is_refinement,
            effective_query=effective_query,
        )

        if chunk.is_final:
            handler.reset()

        return result

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        import re
        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "to", "of", "in", "on", "for", "with", "as", "by", "from", "that",
            "this", "it", "be", "has", "have", "had", "what", "how", "why",
        }
        return {
            token for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text.lower())
            if token not in stopwords
        }

    def _is_late_refinement(self, session: SessionState, query: str) -> bool:
        if not session.current_answer or not session.current_query:
            return False

        query_tokens = self._content_tokens(query)
        previous_tokens = self._content_tokens(session.current_query)
        if not query_tokens or not previous_tokens:
            return False

        overlap = len(query_tokens & previous_tokens)
        jaccard = overlap / len(query_tokens | previous_tokens)

        normalized = query.lower().strip()
        follow_up = normalized.startswith(
            ("and ", "also ", "what about ", "how about ", "the trip ", "the booking ")
        )

        # Conservative lexical relation test: a meaningful shared concept or
        # explicit follow-up wording is enough to refine an existing answer.
        return follow_up or overlap >= 2 or jaccard >= 0.15

    @staticmethod
    def _extract_delta_query(query: str, previous_query: str) -> str:
        current = query.strip()
        previous = previous_query.strip()
        if not previous:
            return current

        current_norm = " ".join(current.lower().split())
        previous_norm = " ".join(previous.lower().split())
        if current_norm.startswith(previous_norm):
            delta = current[len(previous):].strip(" ,.;:-")
            if delta:
                return delta
        return current

    def _get_chunk_handler(self, session_id: str) -> ChunkHandler:
        if session_id not in self._chunk_handlers:
            self._chunk_handlers[session_id] = ChunkHandler(session_id)
        return self._chunk_handlers[session_id]

    def close_session(self, session_id: str) -> SessionState:
        self._chunk_handlers.pop(session_id, None)
        self.stream_controller.clear_session(session_id)
        return self.session_manager.close_session(session_id)

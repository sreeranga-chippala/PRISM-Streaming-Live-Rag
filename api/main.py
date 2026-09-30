from __future__ import annotations

import ipaddress
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import logging
import threading

logger = logging.getLogger("prism.api")


from src.evaluation.evaluator import Evaluator
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_manager import CitationManager
from src.generation.grounding_checker import GroundingChecker
from src.orchestration.pipeline import StreamingRAGPipeline

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

from src.streaming.transcript_stream import TranscriptChunk


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================


class ChatRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    text: str

    is_final: bool = True

    sequence_id: int | None = None


class SourceResponse(BaseModel):
    citation_id: str

    source: str

    chunk_id: str | None

    score: float | None

    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    action: str
    reason: str
    query: str | None
    answer: str | None
    citations: list[SourceResponse]
    groundedness: float | None
    grounded: bool | None
    retrieval_count: int
    latency_ms: float

    # Additive Theme 4 response fields. These preserve the existing response
    # contract while exposing the state/telemetry required by the evaluation
    # gates to API clients.
    answer_version: int | None = None
    retrieval_events: list[dict[str, Any]] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)
    uncertainty: str | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)


class MonitoringEvent(BaseModel):
    session_id: str

    event_type: str

    severity: str = "info"

    detail: str = ""


class AssessmentSubmission(BaseModel):
    session_id: str

    question: str

    answer: str


# ============================================================
# MONITORING SERVICE
# ============================================================


class MonitoringService:
    """
    Session-scoped integrity monitoring.

    Important architecture rule:

        Browser monitoring event
                    |
                    v
             client IP observed
                    |
                    v
             monitoring state

    Normal API requests such as:

        /chat
        /chat/latest
        /monitoring/{session_id}
        /assessment/{session_id}

    DO NOT modify the observed client IP.

    This avoids Docker/Streamlit server addresses being
    incorrectly interpreted as candidate network changes.
    """

    def __init__(self) -> None:

        self.sessions: dict[
            str,
            dict[str, Any],
        ] = {}

    # --------------------------------------------------------
    # SESSION
    # --------------------------------------------------------

    def _session(
        self,
        session_id: str,
    ) -> dict[str, Any]:

        return self.sessions.setdefault(
            session_id,
            {
                "camera_status": "NOT_STARTED",

                "focus_status": "ACTIVE",

                "initial_ip": None,

                "current_ip": None,

                "ip_history": [],

                "ip_consistent": True,

                "monitor_agent": {
                    "status": "WAITING_FOR_IP",
                    "client_ip": None,
                    "decision": "WAIT",
                    "action": "WAIT",
                },

                "events": [],

                "alert_count": 0,

                "last_seen": None,
            },
        )

    # --------------------------------------------------------
    # IP NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize_ip(
        ip: str | None,
    ) -> str | None:

        if not ip:
            return None

        ip = ip.strip()

        if not ip:
            return None

        # IPv6 localhost
        if ip in {
            "::1",
            "0:0:0:0:0:0:0:1",
        }:
            return "127.0.0.1"

        # IPv4 mapped IPv6
        if ip.startswith("::ffff:"):

            mapped = ip[7:]

            if mapped:
                return mapped

        return ip

    # --------------------------------------------------------
    # LOCAL / PRIVATE ADDRESS DETECTION
    # --------------------------------------------------------

    @staticmethod
    def _is_local_address(
        ip: str,
    ) -> bool:

        try:

            address = ipaddress.ip_address(ip)

            return (
                address.is_loopback
                or address.is_private
                or address.is_link_local
            )

        except ValueError:

            return False

    # --------------------------------------------------------
    # OBSERVE IP
    # --------------------------------------------------------

    def observe_ip(
        self,
        session_id: str,
        client_ip: str | None,
    ) -> dict[str, Any]:

        session = self._session(
            session_id
        )

        normalized_ip = (
            self._normalize_ip(client_ip)
        )

        if not normalized_ip:

            return session

        # ====================================================
        # LOCAL DEVELOPMENT
        # ====================================================

        if self._is_local_address(
            normalized_ip
        ):

            if session["initial_ip"] is None:

                session["initial_ip"] = (
                    normalized_ip
                )

            session["current_ip"] = (
                normalized_ip
            )

            if (
                normalized_ip
                not in session["ip_history"]
            ):

                session["ip_history"].append(
                    normalized_ip
                )

            session["ip_consistent"] = True

            session["monitor_agent"] = {
                "status": "IP_STABLE",
                "client_ip": normalized_ip,
                "decision": "KEEP_MONITORING",
                "action": "CONTINUE",
            }

            session["last_seen"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            return session

        # ====================================================
        # FIRST EXTERNAL IP
        # ====================================================

        if session["initial_ip"] is None:

            session["initial_ip"] = (
                normalized_ip
            )

            session["current_ip"] = (
                normalized_ip
            )

            session["ip_history"].append(
                normalized_ip
            )

            session["ip_consistent"] = True

            session["monitor_agent"] = {
                "status": "IP_STABLE",
                "client_ip": normalized_ip,
                "decision": "KEEP_MONITORING",
                "action": "CONTINUE",
            }

            session["last_seen"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            return session

        # ====================================================
        # SAME EXTERNAL IP
        # ====================================================

        previous_ip = session[
            "current_ip"
        ]

        if previous_ip == normalized_ip:

            session["current_ip"] = (
                normalized_ip
            )

            session["ip_consistent"] = True

            session["monitor_agent"] = {
                "status": "IP_STABLE",
                "client_ip": normalized_ip,
                "decision": "KEEP_MONITORING",
                "action": "CONTINUE",
            }

            session["last_seen"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            return session

        # ====================================================
        # ACTUAL EXTERNAL IP CHANGE
        # ====================================================

        session["current_ip"] = (
            normalized_ip
        )

        if (
            normalized_ip
            not in session["ip_history"]
        ):

            session["ip_history"].append(
                normalized_ip
            )

        session["ip_consistent"] = False

        now = datetime.now(
            timezone.utc
        ).isoformat()

        event = {
            "id": str(uuid.uuid4()),

            "timestamp": now,

            "event_type": "ip_changed",

            "severity": "warning",

            "detail": (
                "Client network address changed "
                "during the assessment session."
            ),

            "previous_ip": previous_ip,

            "current_ip": normalized_ip,
        }

        session["events"].append(
            event
        )

        session["events"] = (
            session["events"][-100:]
        )

        session["alert_count"] += 1

        session["last_seen"] = now

        session["monitor_agent"] = {
            "status": "IP_CHANGED",
            "client_ip": normalized_ip,
            "decision": "FLAG",
            "action": "RECORD_IP_CHANGE",
        }

        return session

    # --------------------------------------------------------
    # RECORD EVENT
    # --------------------------------------------------------

    def record(
        self,
        event: MonitoringEvent,
    ) -> dict[str, Any]:

        session = self._session(
            event.session_id
        )

        # IP observation is handled by
        # observe_ip().
        if event.event_type == "ip_observed":

            return self.observe_ip(
                event.session_id,
                event.detail,
            )

                # Heartbeats only prove the browser is alive; don't let them
        # fill (and evict) the 100-event history.
        # Timestamp must be created BEFORE any event branch uses it.
        now = datetime.now(
            timezone.utc
        ).isoformat()

        # Heartbeats only prove that the browser is alive.
        # Do not add them to the event history.
        if event.event_type in {
            "camera_heartbeat",
            "page_heartbeat",
        }:
            session["last_seen"] = now
            return {
                "status": "ok",
                "timestamp": now,
            }

        item = {
            "id": str(uuid.uuid4()),

            "timestamp": now,

            "event_type": event.event_type,

            "severity": event.severity,

            "detail": event.detail,
        }

        session["events"].append(
            item
        )

        session["events"] = (
            session["events"][-100:]
        )

        session["last_seen"] = now

        # ====================================================
        # CAMERA
        # ====================================================

        if event.event_type == "camera_started":

            session[
                "camera_status"
            ] = "ACTIVE"

        elif event.event_type in {
            "camera_stopped",
            "camera_unavailable",
        }:

            session[
                "camera_status"
            ] = "INTERRUPTED"

        # ====================================================
        # PAGE FOCUS
        # ====================================================

        elif event.event_type == (
            "page_focus_lost"
        ):

            session[
                "focus_status"
            ] = "LOST"

        elif event.event_type == (
            "page_focus_returned"
        ):

            session[
                "focus_status"
            ] = "ACTIVE"

        # ====================================================
        # WARNING COUNT
        # ====================================================

        if event.severity == "warning":

            session[
                "alert_count"
            ] += 1

        return item

    # --------------------------------------------------------
    # GET STATE
    # --------------------------------------------------------

    def get(
        self,
        session_id: str,
    ) -> dict[str, Any]:

        return self._session(
            session_id
        )


# ============================================================
# RAG APPLICATION
# ============================================================


class RAGApplication:
    """
    Application service joining:

        StreamingRAGPipeline
                    |
                    v
              retrieval
                    |
                    v
               generation
                    |
                    v
              grounding
                    |
                    v
               evaluation

    Monitor/integrity functionality is intentionally kept
    outside this RAG pipeline.
    """

    def __init__(
        self,
        pipeline: StreamingRAGPipeline | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:

        self.generator = (
            generator
            or AnswerGenerator()
        )
        self._pipeline_lock = threading.Lock()
        self.citations = (
            CitationManager()
        )

        self.grounding = (
            GroundingChecker()
        )

        self.evaluator = (
            Evaluator(
                self.grounding
            )
        )

        self.pipeline = pipeline

        self.latest_answers: dict[
            str,
            dict[str, Any],
        ] = {}

    # --------------------------------------------------------
    # LAZY PIPELINE INITIALIZATION
    # --------------------------------------------------------

    def _ensure_pipeline(self) -> StreamingRAGPipeline:
        """Build the retrieval stack once (thread-safe)."""
        if self.pipeline is None:
            with self._pipeline_lock:
                if self.pipeline is None:
                    self.pipeline = StreamingRAGPipeline(
                        multi_query_retriever=self._build_retriever()
                    )
                    self._pipeline_initialized = True
        return self.pipeline

    # --------------------------------------------------------
    # RETRIEVAL STACK
    # --------------------------------------------------------

    def _build_retriever(
        self,
    ) -> MultiQueryRetriever:

        embedding_model = (
            EmbeddingModel(
                model_name=(
                    "all-MiniLM-L6-v2"
                ),
                batch_size=32,
            )
        )

        vector_store = (
            VectorStore(
                "data/processed/vector_store"
            )
        )

        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
            default_top_k=5,
        )

        hybrid_search = HybridSearch(
            retriever=retriever,
            vector_store=vector_store,
            semantic_top_k=5,
            keyword_top_k=5,
        )

        fusion = (
            ReciprocalRankFusion(
                k=60
            )
        )

        reranker = Reranker(
            model_name=(
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
            batch_size=16,
        )

        return MultiQueryRetriever(
            hybrid_search=hybrid_search,
            fusion=fusion,
            reranker=reranker,
            default_subquery_top_k=5,
            default_final_top_k=5,
        )

    # --------------------------------------------------------
    # RESET CURRENT UTTERANCE
    # --------------------------------------------------------

    def reset_utterance(
        self,
        session_id: str,
    ) -> None:

        pipeline = (
            self._ensure_pipeline()
        )

        handler = (
            pipeline._get_chunk_handler(
                session_id
            )
        )

        handler.reset()

    # --------------------------------------------------------
    # PROCESS CHAT
    # --------------------------------------------------------

    def process(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        started = time.perf_counter()

        chunk = TranscriptChunk(
            session_id=request.session_id,
            text=request.text,
            timestamp=time.time(),
            is_final=request.is_final,
            sequence_id=request.sequence_id,
        )

        try:
            pipeline = self._ensure_pipeline()
            result = pipeline.process_chunk(chunk)
            # Count only turns that successfully passed through the pipeline.
            # Each normal path below emits exactly one end-to-end trace event.
            self.evaluator.register_turn()
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

        action = result.retrieval_decision.action.value
        retrieved = list(result.retrieved_results)
        session = result.session
        sub_queries = [
            item.refined_query for item in result.refined_queries
        ]

        # Presentation-only / previous-answer requests are deliberately
        # handled without retrieval.  A standalone "summarize" request with
        # no prior answer is promoted back to retrieval by the pipeline.
        if not retrieved:
            if result.is_refinement and session.current_answer:
                elapsed = (time.perf_counter() - started) * 1000.0
                uncertainty = "The new detail could not be verified from the corpus."
                self.evaluator.record_trace(
                    session_id=request.session_id,
                    event_type="late_detail_unverified",
                    timestamp_s=chunk.timestamp,
                    query=result.effective_query,
                    retrieval_required=True,
                    retrieval_trigger="late_detail_delta",
                    retrieval_events=result.retrieval_events,
                    citations=session.current_citations,
                    answer_version=session.answer_version or None,
                    latency_ms=elapsed,
                    uncertainty=uncertainty,
                )
                return ChatResponse(
                    session_id=request.session_id,
                    action=action,
                    reason=result.retrieval_decision.reason,
                    query=result.effective_query,
                    answer=session.current_answer,
                    citations=[SourceResponse(**item) for item in session.current_citations],
                    groundedness=None,
                    grounded=None,
                    retrieval_count=0,
                    latency_ms=round(elapsed, 2),
                    answer_version=session.answer_version or None,
                    retrieval_events=list(result.retrieval_events),
                    sub_queries=sub_queries,
                    uncertainty=uncertainty,
                    telemetry={
                        "event_type": "late_detail_unverified",
                        "answer_version": session.answer_version or None,
                        "retrieval_required": True,
                        "retrieval_trigger": "late_detail_delta",
                        "latency_ms": round(elapsed, 2),
                    },
                )

            if (
                action == "NO_RETRIEVAL"
                and session.current_answer
                and result.retrieval_decision.reason
                == "presentation_or_previous_answer_request"
            ):
                generated = self.generator.transform_existing_answer(
                    instruction=request.text,
                    previous_answer=session.current_answer,
                )
                previous_citations = list(session.current_citations)
                citations = [
                    SourceResponse(**item)
                    for item in previous_citations
                ]
                version = pipeline.session_manager.update_answer(
                    request.session_id,
                    generated.answer,
                    chunk.timestamp,
                    query=session.current_query,
                    citations=previous_citations,
                    reason="presentation_restructure",
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                telemetry = {
                    "event_type": "presentation_restructure",
                    "answer_version": version,
                    "retrieval_required": False,
                    "retrieval_trigger": "suppressed",
                    "latency_ms": round(elapsed, 2),
                    "ttft_ms": round(generated.latency_ms, 2),
                    "prompt_tokens": generated.prompt_tokens,
                    "output_tokens": generated.output_tokens,
                    "total_tokens": generated.total_tokens,
                    "estimated_cost_usd": generated.estimated_cost_usd,
                }
                self.latest_answers[request.session_id] = {
                    "session_id": request.session_id,
                    "answer": generated.answer,
                    "query": session.current_query,
                    "action": action,
                    "reason": result.retrieval_decision.reason,
                    "grounded": None,
                    "groundedness": None,
                    "answer_version": version,
                    "retrieval_events": [],
                    "sub_queries": [],
                    "uncertainty": None,
                    "citations": previous_citations,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.evaluator.record_trace(
                    session_id=request.session_id,
                    event_type="presentation_restructure",
                    timestamp_s=chunk.timestamp,
                    query=request.text,
                    retrieval_required=False,
                    retrieval_trigger="suppressed",
                    answer_version=version,
                    latency_ms=elapsed,
                    ttft_ms=generated.latency_ms,
                    prompt_tokens=generated.prompt_tokens,
                    output_tokens=generated.output_tokens,
                    total_tokens=generated.total_tokens,
                    estimated_cost_usd=generated.estimated_cost_usd,
                )
                return ChatResponse(
                    session_id=request.session_id,
                    action=action,
                    reason=result.retrieval_decision.reason,
                    query=result.retrieval_decision.query,
                    answer=generated.answer,
                    citations=citations,
                    groundedness=None,
                    grounded=None,
                    retrieval_count=0,
                    latency_ms=round(elapsed, 2),
                    answer_version=version,
                    retrieval_events=list(result.retrieval_events),
                    sub_queries=[],
                    uncertainty=None,
                    telemetry=telemetry,
                )

            elapsed = (time.perf_counter() - started) * 1000.0
            self.evaluator.record_trace(
                session_id=request.session_id,
                event_type=action.lower(),
                timestamp_s=chunk.timestamp,
                query=result.retrieval_decision.query,
                retrieval_required=False,
                retrieval_trigger=result.retrieval_decision.reason,
                retrieval_events=result.retrieval_events,
                latency_ms=elapsed,
            )
            return ChatResponse(
                session_id=request.session_id,
                action=action,
                reason=result.retrieval_decision.reason,
                query=result.retrieval_decision.query,
                answer=None,
                citations=[],
                groundedness=None,
                grounded=None,
                retrieval_count=0,
                latency_ms=round(elapsed, 2),
                answer_version=session.answer_version or None,
                retrieval_events=list(result.retrieval_events),
                sub_queries=sub_queries,
                uncertainty=None,
                telemetry={
                    "event_type": action.lower(),
                    "retrieval_required": False,
                    "retrieval_trigger": result.retrieval_decision.reason,
                    "latency_ms": round(elapsed, 2),
                },
            )

        query = result.effective_query or result.retrieval_decision.query or request.text

        # Existing answer + late detail => targeted delta refinement.
        if result.is_refinement and session.current_answer:
            generated = self.generator.refine(
                previous_answer=session.current_answer,
                delta_query=query,
                delta_results=retrieved,
            )
        else:
            session_context = pipeline.session_manager.get_context(
                result.session.session_id
            )
            generated = self.generator.generate(
                query=query,
                retrieved_results=retrieved,
                session_context=session_context,
            )

        new_citations = self.citations.build_citations(
            answer=generated.answer,
            retrieved_results=retrieved,
        )

        # Refinement preserves prior citations and adds only new evidence.
        citation_dicts: list[dict[str, Any]] = []
        if result.is_refinement:
            citation_dicts.extend(session.current_citations)
        seen_chunks = {
            item.get("chunk_id")
            for item in citation_dicts
            if item.get("chunk_id")
        }
        for item in new_citations:
            if item.chunk_id in seen_chunks:
                continue
            citation_dicts.append(
                {
                    "citation_id": item.citation_id,
                    "source": item.source,
                    "chunk_id": item.chunk_id,
                    "score": item.score,
                    "excerpt": item.excerpt,
                }
            )
            if item.chunk_id:
                seen_chunks.add(item.chunk_id)

        # Citation IDs are local to the final answer version. Re-number after
        # preserving prior citations so IDs are unique and deterministic.
        for index, item in enumerate(citation_dicts, start=1):
            item["citation_id"] = f"S{index}"
        citations = [SourceResponse(**item) for item in citation_dicts]

        grounding = self.grounding.check(
            generated.answer,
            retrieved if not result.is_refinement else retrieved + [
                {"text": session.current_answer}
            ],
        )

        version = pipeline.session_manager.update_answer(
            request.session_id,
            generated.answer,
            chunk.timestamp,
            query=query,
            citations=citation_dicts,
            reason="late_detail_refinement" if result.is_refinement else "initial",
        )

        elapsed = (time.perf_counter() - started) * 1000.0
        uncertainty = None
        if grounding.unsupported_sentences:
            uncertainty = (
                "Some answer statements were not fully supported by the "
                "available evidence: "
                + " | ".join(grounding.unsupported_sentences)
            )

        telemetry = {
            "event_type": "late_detail_refinement" if result.is_refinement else "answer_generated",
            "answer_version": version,
            "retrieval_required": True,
            "retrieval_trigger": (
                "late_detail_delta"
                if result.is_refinement
                else (
                    "multi_intent"
                    if len(sub_queries) > 1
                    else result.retrieval_decision.reason
                )
            ),
            "latency_ms": round(elapsed, 2),
            "ttft_ms": round(generated.latency_ms, 2),
            "prompt_tokens": generated.prompt_tokens,
            "output_tokens": generated.output_tokens,
            "total_tokens": generated.total_tokens,
            "estimated_cost_usd": generated.estimated_cost_usd,
        }

        self.evaluator.evaluate_turn(
            query=query,
            answer=generated.answer,
            retrieved_results=retrieved,
            citations=citations,
            retrieval_required=True,
            started_at=started,
            first_token_at=started + generated.latency_ms / 1000.0,
            completed_at=time.perf_counter(),
            answer_version=version,
            prompt_tokens=generated.prompt_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.total_tokens,
            estimated_cost_usd=generated.estimated_cost_usd,
        )
        self.evaluator.record_trace(
            session_id=request.session_id,
            event_type=telemetry["event_type"],
            timestamp_s=chunk.timestamp,
            query=query,
            retrieval_required=True,
            retrieval_trigger=telemetry["retrieval_trigger"],
            retrieval_events=result.retrieval_events,
            citations=citations,
            answer_version=version,
            latency_ms=elapsed,
            ttft_ms=generated.latency_ms,
            prompt_tokens=generated.prompt_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.total_tokens,
            estimated_cost_usd=generated.estimated_cost_usd,
            uncertainty=uncertainty,
        )

        response = ChatResponse(
            session_id=request.session_id,
            action=action,
            reason=result.retrieval_decision.reason,
            query=query,
            answer=generated.answer,
            citations=citations,
            groundedness=grounding.score,
            grounded=grounding.grounded,
            retrieval_count=len(retrieved),
            latency_ms=round(elapsed, 2),
            answer_version=version,
            retrieval_events=list(result.retrieval_events),
            sub_queries=sub_queries,
            uncertainty=uncertainty,
            telemetry=telemetry,
        )

        self.latest_answers[request.session_id] = {
            "session_id": request.session_id,
            "answer": generated.answer,
            "query": query,
            "action": action,
            "reason": result.retrieval_decision.reason,
            "grounded": grounding.grounded,
            "groundedness": grounding.score,
            "answer_version": version,
            "retrieval_events": list(result.retrieval_events),
            "sub_queries": sub_queries,
            "uncertainty": uncertainty,
            "citations": citation_dicts,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        return response



# ============================================================
# FASTAPI APPLICATION
# ============================================================


app = FastAPI(

    title=(
        "PRISM Streaming Live RAG"
    ),

    version="1.1.0",

    description=(
        "Samsung PRISM Theme 4 "
        "Streaming Live RAG API "
        "with session-scoped "
        "assessment integrity signals."
    ),
)


# ============================================================
# CORS
# ============================================================


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# GLOBAL SERVICES
# ============================================================


service = RAGApplication()

monitoring = MonitoringService()

assessment_submissions: dict[
    str,
    list[dict[str, Any]],
] = {}


# ============================================================
# CLIENT IP
# ============================================================


def get_client_ip(
    request: Request,
) -> str:

    """
    Return the address observed by FastAPI.

    X-Forwarded-For is trusted only when
    TRUST_PROXY_HEADERS=true.

    IMPORTANT:

    This function is used only by the browser
    monitoring endpoint.

    It must NOT be used by normal /chat
    or Streamlit polling requests to mutate
    candidate monitoring state.
    """

    import os

    trust_proxy = (
        os.getenv(
            "TRUST_PROXY_HEADERS",
            "false",
        ).lower()
        == "true"
    )

    if trust_proxy:

        forwarded = (
            request.headers.get(
                "x-forwarded-for"
            )
        )

        if forwarded:

            return (
                forwarded
                .split(",")[0]
                .strip()
            )

    return (
        request.client.host
        if request.client
        else "unknown"
    )


@app.on_event("startup")
def warm_up() -> None:
    service._ensure_pipeline()
    
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "prism-live-rag",
        "pipeline_ready": service.pipeline is not None,
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    try:
        return service.process(request)
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/voice/finalize", response_model=ChatResponse)
def voice_finalize(request: ChatRequest, http_request: Request) -> ChatResponse:
    try:
        service.reset_utterance(request.session_id)
        return service.process(request.model_copy(update={"is_final": True}))
    except Exception as exc:
        logger.exception("voice_finalize failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    
# ============================================================
# LATEST CHAT RESPONSE
# ============================================================


@app.get(
    "/chat/latest/{session_id}"
)
def latest_chat(
    session_id: str,
) -> dict[str, Any]:

    return service.latest_answers.get(

        session_id,

        {
            "session_id": session_id,

            "answer": None,

            "query": None,

            "citations": [],

            "action": None,

            "reason": None,

            "grounded": None,

            "groundedness": None,
            "answer_version": None,
            "retrieval_events": [],
            "sub_queries": [],
            "uncertainty": None,
            "telemetry": {},
        },
    )


# ============================================================
# MONITORING EVENT
# ============================================================


@app.post("/monitoring/event")
def monitoring_event(event: MonitoringEvent, request: Request) -> dict[str, Any]:
    if request.headers.get("X-PRISM-Browser") == "1":
        client_ip = get_client_ip(request)
        if client_ip != "unknown":
            monitoring.observe_ip(event.session_id, client_ip)
    return monitoring.record(event)


# ============================================================
# MONITORING STATE
# ============================================================


@app.get(
    "/monitoring/{session_id}"
)
def monitoring_state(
    session_id: str,
) -> dict[str, Any]:

    """
    Read-only monitoring state.

    This endpoint MUST NOT observe the API request IP.

    Streamlit polling this endpoint therefore cannot
    accidentally create IP changes.
    """

    return monitoring.get(
        session_id
    )


# ============================================================
# ASSESSMENT SUBMISSION
# ============================================================


@app.post(
    "/assessment/submit"
)
def assessment_submit(
    submission: AssessmentSubmission,
) -> dict[str, Any]:

    """
    Store a candidate assessment answer.

    Submission itself is recorded as a monitoring event,
    but the API request IP is NOT used for IP monitoring.
    """

    item = {

        "question": (
            submission.question
        ),

        "answer": (
            submission.answer
        ),

        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    assessment_submissions.setdefault(

        submission.session_id,

        [],
    ).append(item)

    # --------------------------------------------------------
    # Record assessment event.
    #
    # DO NOT pass an IP here.
    # --------------------------------------------------------

    monitoring.record(

        MonitoringEvent(

            session_id=(
                submission.session_id
            ),

            event_type=(
                "answer_submitted"
            ),

            severity="info",

            detail=(
                "Candidate submitted "
                "an assessment answer"
            ),
        )
    )

    return {

        "status": "submitted",

        "session_id": (
            submission.session_id
        ),

        "submission": item,
    }


# ============================================================
# ASSESSMENT STATE
# ============================================================


@app.get(
    "/assessment/{session_id}"
)
def assessment_state(
    session_id: str,
) -> dict[str, Any]:

    return {

        "session_id": session_id,

        "submissions": (
            assessment_submissions.get(
                session_id,
                [],
            )
        ),
    }


# ============================================================
# METRICS
# ============================================================


@app.get("/metrics")
def metrics() -> dict[str, Any]:

    return service.evaluator.summary()
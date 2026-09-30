from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.generation.grounding_checker import GroundingChecker
from src.generation.citation_manager import Citation, CitationManager


@dataclass
class EvaluationRecord:
    query: str
    retrieval_required: bool
    retrieval_count: int
    citation_count: int
    groundedness: float
    grounded: bool
    retrieval_recall: float | None
    time_to_first_token_ms: float | None
    total_latency_ms: float | None
    answer_version: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    citation_support: float | None = None


class Evaluator:
    """Quantitative evaluation and structured telemetry for Theme 4."""

    def __init__(self, grounding_checker: GroundingChecker | None = None):
        self.grounding_checker = grounding_checker or GroundingChecker()
        self.citation_manager = CitationManager()
        self.records: list[EvaluationRecord] = []
        self.trace_events: list[dict[str, Any]] = []
        self.expected_turns: int = 0

    def register_turn(self) -> None:
        """Register one successfully processed chat turn for trace coverage."""
        self.expected_turns += 1

    def evaluate_turn(
        self,
        *,
        query: str,
        answer: str,
        retrieved_results: Iterable[dict[str, Any]],
        citations: Iterable[Any] | None = None,
        gold_chunk_ids: Iterable[str] | None = None,
        retrieval_required: bool = True,
        started_at: float | None = None,
        first_token_at: float | None = None,
        completed_at: float | None = None,
        answer_version: int | None = None,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        citation_support: float | None = None,
    ) -> EvaluationRecord:
        retrieved = list(retrieved_results)
        citation_list = list(citations or [])
        grounding = self.grounding_checker.check(answer, retrieved)
        recall = self._retrieval_recall(retrieved, gold_chunk_ids)
        citation_objects = [
            item for item in citation_list if isinstance(item, Citation)
        ]
        if citation_support is None and citation_objects:
            citation_support = self.citation_manager.support_coverage(
                answer, retrieved, citation_objects
            )

        ttft = None
        total = None
        if started_at is not None and first_token_at is not None:
            ttft = max(0.0, (first_token_at - started_at) * 1000.0)
        if started_at is not None and completed_at is not None:
            total = max(0.0, (completed_at - started_at) * 1000.0)

        record = EvaluationRecord(
            query=query,
            retrieval_required=retrieval_required,
            retrieval_count=len(retrieved),
            citation_count=len(citation_list),
            groundedness=grounding.score,
            grounded=grounding.grounded,
            retrieval_recall=recall,
            time_to_first_token_ms=ttft,
            total_latency_ms=total,
            answer_version=answer_version,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            citation_support=citation_support,
        )
        self.records.append(record)
        return record

    def record_trace(
        self,
        *,
        session_id: str,
        event_type: str,
        timestamp_s: float | None = None,
        query: str | None = None,
        retrieval_required: bool | None = None,
        retrieval_trigger: str | None = None,
        retrieval_events: Iterable[dict[str, Any]] | None = None,
        citations: Iterable[Any] | None = None,
        answer_version: int | None = None,
        latency_ms: float | None = None,
        ttft_ms: float | None = None,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        estimated_cost_usd: float | None = None,
        uncertainty: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one end-to-end structured trace event."""
        citation_ids = []
        for item in citations or []:
            citation_id = getattr(item, "citation_id", None)
            if citation_id is None and isinstance(item, dict):
                citation_id = item.get("citation_id")
            if citation_id is not None:
                citation_ids.append(str(citation_id))

        event = {
            "timestamp_s": timestamp_s,
            "session_id": session_id,
            "event_type": event_type,
            "query": query,
            "retrieval_required": retrieval_required,
            "retrieval_trigger": retrieval_trigger,
            "retrieval_events": list(retrieval_events or []),
            "citations": citation_ids,
            "answer_version": answer_version,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
            "uncertainty": uncertainty,
            "metadata": metadata or {},
        }
        self.trace_events.append(event)
        return event

    def summary(self) -> dict[str, Any]:
        if not self.records:
            base = {
                "turns": 0,
                "avg_groundedness": 0.0,
                "grounded_turn_rate": 0.0,
                "avg_retrieval_recall": None,
                "avg_ttft_ms": None,
                "avg_total_latency_ms": None,
                "avg_retrieval_count": 0.0,
                "avg_citation_count": 0.0,
                "avg_citation_support": None,
            }
        else:
            def avg(values: list[float]) -> float | None:
                return round(sum(values) / len(values), 3) if values else None

            recalls = [r.retrieval_recall for r in self.records if r.retrieval_recall is not None]
            ttfts = [r.time_to_first_token_ms for r in self.records if r.time_to_first_token_ms is not None]
            totals = [r.total_latency_ms for r in self.records if r.total_latency_ms is not None]
            costs = [r.estimated_cost_usd for r in self.records if r.estimated_cost_usd is not None]
            citation_supports = [
                r.citation_support for r in self.records
                if r.citation_support is not None
            ]

            base = {
                "turns": len(self.records),
                "avg_groundedness": round(sum(r.groundedness for r in self.records) / len(self.records), 3),
                "grounded_turn_rate": round(sum(r.grounded for r in self.records) / len(self.records), 3),
                "avg_retrieval_recall": avg(recalls),
                "avg_ttft_ms": avg(ttfts),
                "avg_total_latency_ms": avg(totals),
                "avg_retrieval_count": round(sum(r.retrieval_count for r in self.records) / len(self.records), 3),
                "avg_citation_count": round(sum(r.citation_count for r in self.records) / len(self.records), 3),
                "estimated_cost_usd": round(sum(costs), 8) if costs else None,
                "avg_citation_support": (
                    round(sum(citation_supports) / len(citation_supports), 3)
                    if citation_supports else None
                ),
            }

        traced = len(self.trace_events)
        base["trace_events"] = traced
        base["processed_turns"] = self.expected_turns
        base["trace_coverage"] = (
            round(min(1.0, traced / self.expected_turns), 4)
            if self.expected_turns
            else 0.0
        )
        base["answer_versions"] = sorted(
            {
                int(event["answer_version"])
                for event in self.trace_events
                if event.get("answer_version") is not None
            }
        )
        return base

    @staticmethod
    def _retrieval_recall(
        retrieved: list[dict[str, Any]],
        gold_chunk_ids: Iterable[str] | None,
    ) -> float | None:
        if gold_chunk_ids is None:
            return None
        gold = {str(item) for item in gold_chunk_ids}
        if not gold:
            return None
        found = {
            str(item.get("chunk_id"))
            for item in retrieved
            if item.get("chunk_id") is not None
        }
        return round(len(found & gold) / len(gold), 4)

    def export(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]

    def export_traces(self) -> list[dict[str, Any]]:
        return list(self.trace_events)

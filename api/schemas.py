"""Shared API schemas for structured Theme 4 events.

The main FastAPI module retains its existing request/response models for
backward compatibility; these models document the machine-readable event
contract used by telemetry and evaluation.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalEvent(BaseModel):
    timestamp_s: float
    query: str | None = None
    trigger: str
    sub_queries: list[str] = Field(default_factory=list)
    result_count: int | None = None
    reason: str | None = None


class TelemetryEvent(BaseModel):
    timestamp_s: float | None = None
    session_id: str
    event_type: str
    query: str | None = None
    retrieval_required: bool | None = None
    retrieval_trigger: str | None = None
    retrieval_events: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    answer_version: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    uncertainty: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

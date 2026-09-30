# Streaming Live RAG — System Architecture Brief

## 1. Scope
The system implements incremental transcript processing, selective early retrieval,
multi-intent decomposition, hybrid retrieval/fusion/reranking, state-preserving
late-detail refinement, presentation-only query suppression, grounded citations,
and structured telemetry.

## 2. Pipeline
`TranscriptChunk -> RetrievalController -> IntentDetector -> QueryDecomposer ->
QueryRefiner -> Hybrid Retrieval (parallel subqueries) -> RRF -> Reranker ->
Grounded Generation -> Citation/Grounding -> Session State -> Telemetry`

For a late-arriving detail, only the delta query is retrieved. The previous answer
and prior citations remain session state and are passed to the refinement generator.

## 3. Retrieval Trigger Logic
The controller has three states:
- `WAIT`: transcript is incomplete or below the stability threshold.
- `RETRIEVE`: enough stable information exists for an evidence lookup.
- `NO_RETRIEVAL`: the request only changes presentation of an existing answer, or
  retrieval is already satisfied.

A presentation request is promoted to retrieval if no previous answer exists.

## 4. Multi-Intent Retrieval
Independent subqueries are executed concurrently. `ThreadPoolExecutor.map`
preserves subquery order, so fusion remains deterministic. Reciprocal Rank Fusion
merges candidate lists and the existing reranker produces the final top-k set.

## 5. Session Refinement
Each session stores:
- current query
- current answer
- answer version
- answer history
- current citations
- retrieved evidence
- event lineage

A late detail increments the answer version and preserves prior citations while
adding citations from newly retrieved delta evidence.

## 6. Grounding and Provenance
Generation is explicitly corpus-grounded. Citation IDs are created only from
retrieved results. The grounding checker exposes unsupported sentences rather
than silently treating lexical overlap as proof of semantic truth.

## 7. Observability
Every `/chat` turn records a structured trace containing retrieval decision,
retrieval events, citations, answer version, latency, TTFT estimate, token usage,
and optional configured cost estimate. `/metrics` exposes aggregate coverage.

## 8. Reproducibility
The repository contains pinned dependency ranges, Docker packaging, CI syntax/test
checks, and evaluation scripts. A clean machine can start the containers without a
local `.env`; a Gemini API key is required only for answer generation.

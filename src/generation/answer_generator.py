from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class GenerationResult:
    answer: str
    latency_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None


class AnswerGenerator:
    """
    Corpus-grounded answer synthesis plus state-preserving refinement.

    Retrieval remains outside this class.  Refinement receives only the
    existing answer and newly retrieved delta evidence; presentation-only
    requests receive only the existing answer and therefore cannot trigger
    corpus retrieval.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )
        self.client = client

    def _get_client(self) -> Any:
        if self.client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured.")
            self.client = genai.Client(api_key=api_key)
        return self.client

    def _result_from_response(self, answer: str, start: float, response: Any) -> GenerationResult:
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)

        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
        except (TypeError, ValueError):
            prompt_tokens = None
        try:
            output_tokens = int(output_tokens) if output_tokens is not None else None
        except (TypeError, ValueError):
            output_tokens = None
        try:
            total_tokens = int(total_tokens) if total_tokens is not None else None
        except (TypeError, ValueError):
            total_tokens = None

        estimated_cost = self._estimate_cost(prompt_tokens, output_tokens)
        return GenerationResult(
            answer=answer,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

    @staticmethod
    def _estimate_cost(prompt_tokens: int | None, output_tokens: int | None) -> float | None:
        if prompt_tokens is None and output_tokens is None:
            return None
        try:
            input_rate = float(os.getenv("GEMINI_INPUT_USD_PER_1M_TOKENS", "0"))
            output_rate = float(os.getenv("GEMINI_OUTPUT_USD_PER_1M_TOKENS", "0"))
        except ValueError:
            return None
        if input_rate == 0 and output_rate == 0:
            return None
        return round(
            ((prompt_tokens or 0) / 1_000_000) * input_rate
            + ((output_tokens or 0) / 1_000_000) * output_rate,
            8,
        )

    @staticmethod
    def _evidence(results: list[dict[str, Any]]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            text = str(result.get("text", "")).strip()
            if not text:
                continue
            source = str(
                result.get("source")
                or (result.get("metadata") or {}).get("source")
                or "unknown"
            )
            chunk_id = str(result.get("chunk_id", "unknown"))
            blocks.append(
                f"[Evidence {index}]\n"
                f"Source: {source}\n"
                f"Chunk: {chunk_id}\n"
                f"Content:\n{text}"
            )
        return "\n\n".join(blocks)

    def _generate_text(self, prompt: str, system_instruction: str) -> GenerationResult:
        start = time.perf_counter()
        try:
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )
            answer = (getattr(response, "text", None) or "").strip()
        except Exception as exc:
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        if not answer:
            answer = "I don't have enough retrieved evidence to answer that reliably."
        return self._result_from_response(answer, start, response)

    def generate(
        self,
        query: str,
        retrieved_results: list[dict[str, Any]] | None = None,
        session_context: str = "",
    ) -> GenerationResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        results = retrieved_results or []
        evidence = self._evidence(results)
        if not evidence:
            return GenerationResult(
                answer="I don't have enough retrieved evidence to answer that reliably.",
                latency_ms=0.0,
            )

        context_section = (
            f"\nPrevious session evidence:\n{session_context.strip()}\n"
            if session_context.strip()
            else ""
        )

        prompt = f"""
You are the answer-generation component of a Streaming Live RAG system.

Answer the user's question using ONLY the retrieved corpus evidence below.

STRICT RULES:
1. Do not use outside knowledge.
2. Do not invent facts, policies, numbers, dates, names, or sources.
3. Do not mention information that cannot be supported by the evidence.
4. If evidence is insufficient for a part, explicitly say so.
5. Synthesize the evidence instead of copying large source passages.
6. Answer every supported sub-question.
7. Preserve important numbers, conditions, exceptions, and dates exactly.
8. Do not fabricate citation IDs.
9. Previous session evidence is context only; it does not override current evidence.
10. Do not restart unrelated parts merely because a new constraint was introduced.

User question:
{query}
{context_section}
Current retrieved corpus evidence:
{evidence}

Produce only the final answer.
"""
        return self._generate_text(
            prompt,
            "You are a strict corpus-grounded RAG answer generator.",
        )

    def refine(
        self,
        *,
        previous_answer: str,
        delta_query: str,
        delta_results: list[dict[str, Any]],
    ) -> GenerationResult:
        previous_answer = previous_answer.strip()
        delta_query = delta_query.strip()
        if not previous_answer:
            raise ValueError("previous_answer cannot be empty")
        if not delta_query:
            raise ValueError("delta_query cannot be empty")

        evidence = self._evidence(delta_results)
        if not evidence:
            return GenerationResult(
                answer=previous_answer,
                latency_ms=0.0,
            )

        prompt = f"""
You are refining an existing answer in a corpus-grounded Streaming Live RAG system.

Existing answer (authoritative prior state):
{previous_answer}

New user detail:
{delta_query}

Newly retrieved evidence for ONLY that detail:
{evidence}

STRICT RULES:
1. Preserve every previously established fact unless the new evidence directly
   requires a correction.
2. Add only facts supported by the new evidence.
3. Do not restart the answer or perform a full-corpus search conceptually.
4. Do not infer missing exceptions or details.
5. If the new evidence is insufficient, explicitly state that for the affected
   detail while retaining the established answer.
6. Return the complete revised answer, not a patch.
7. Do not fabricate citations or sources.

Produce only the revised answer.
"""
        return self._generate_text(
            prompt,
            "You are a strict state-preserving RAG refinement component.",
        )

    def transform_existing_answer(
        self,
        *,
        instruction: str,
        previous_answer: str,
    ) -> GenerationResult:
        instruction = instruction.strip()
        previous_answer = previous_answer.strip()
        if not instruction:
            raise ValueError("instruction cannot be empty")
        if not previous_answer:
            raise ValueError("previous_answer cannot be empty")

        prompt = f"""
Transform the existing answer according to the user's presentation request.

Existing answer:
{previous_answer}

User presentation request:
{instruction}

Rules:
1. Use only the existing answer.
2. Do not add, remove, or alter factual meaning.
3. Do not introduce any new facts, sources, numbers, dates, or assumptions.
4. If asked for a number of bullets, satisfy that formatting request as closely
   as possible without changing factual content.
5. Return only the transformed answer.
"""
        return self._generate_text(
            prompt,
            "You transform existing text only. You never introduce new facts.",
        )

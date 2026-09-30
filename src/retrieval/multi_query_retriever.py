from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from concurrent.futures import ThreadPoolExecutor

class MultiQueryRetriever:
    """Retrieval-side orchestrator that consumes Person 1's subqueries."""

    def __init__(
        self,
        hybrid_search: HybridSearch,
        fusion: ReciprocalRankFusion,
        reranker: Reranker,
        default_subquery_top_k: int = 5,
        default_final_top_k: int = 5,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        max_workers: int | None = None,
    ):
        if hybrid_search is None or fusion is None or reranker is None:
            raise ValueError("hybrid_search, fusion, and reranker are required")
        if default_subquery_top_k <= 0 or default_final_top_k <= 0:
            raise ValueError("top_k values must be greater than 0")
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if semantic_weight == 0 and keyword_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than 0")
        if max_workers is not None and (
            not isinstance(max_workers, int)
            or isinstance(max_workers, bool)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer or None")

        self.hybrid_search = hybrid_search
        self.fusion = fusion
        self.reranker = reranker
        self.default_subquery_top_k = default_subquery_top_k
        self.default_final_top_k = default_final_top_k
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.max_workers = max_workers

    def retrieve_many(
        self,
        subqueries: Sequence[str],
        subquery_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid retrieval for each subquery, fuse, then rerank."""
        queries = self._validate_subqueries(subqueries)
        subquery_top_k = subquery_top_k or self.default_subquery_top_k
        final_top_k = final_top_k or self.default_final_top_k
        self._validate_top_k(subquery_top_k, "subquery_top_k")
        self._validate_top_k(final_top_k, "final_top_k")

        started = perf_counter()
        retrieval_started = perf_counter()

        semantic_lists: list[list[dict[str, Any]]] = []
        keyword_lists: list[list[dict[str, Any]]] = []
        source_subqueries: dict[str, list[str]] = {}
        source_indices: dict[str, list[int]] = {}

        def retrieve_one(item: tuple[int, str]) -> tuple[int, str, list[dict[str, Any]], list[dict[str, Any]]]:
            index, query = item
            hybrid = self.hybrid_search.search(
                query=query,
                semantic_top_k=subquery_top_k,
                keyword_top_k=subquery_top_k,
            )
            semantic = self._annotate(
                hybrid.get("semantic_results", []), query, index
            )
            keyword = self._annotate(
                hybrid.get("keyword_results", []), query, index
            )
            return index, query, semantic, keyword

        # Independent subqueries are safe to execute concurrently.  map()
        # preserves input order, keeping fusion and tests deterministic.
        workers = self.max_workers or min(8, len(queries))
        if len(queries) == 1:
            retrieved_batches = [retrieve_one((0, queries[0]))]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                retrieved_batches = list(
                    executor.map(retrieve_one, enumerate(queries))
                )

        for index, query, semantic, keyword in retrieved_batches:
            semantic_lists.append(semantic)
            keyword_lists.append(keyword)

            for result in semantic + keyword:
                chunk_id = result.get("chunk_id")
                if not chunk_id:
                    continue
                source_subqueries.setdefault(chunk_id, [])
                source_indices.setdefault(chunk_id, [])
                if query not in source_subqueries[chunk_id]:
                    source_subqueries[chunk_id].append(query)
                if index not in source_indices[chunk_id]:
                    source_indices[chunk_id].append(index)

        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000.0

        fusion_started = perf_counter()
        result_lists: list[list[dict[str, Any]]] = []
        weights: list[float] = []

        for results in semantic_lists:
            result_lists.append(results)
            weights.append(self.semantic_weight)
        for results in keyword_lists:
            result_lists.append(results)
            weights.append(self.keyword_weight)

        fused = self.fusion.fuse(result_lists=result_lists, weights=weights)

        for result in fused:
            chunk_id = result.get("chunk_id")
            result["source_subqueries"] = source_subqueries.get(chunk_id, [])
            result["source_subquery_indices"] = source_indices.get(chunk_id, [])

        fusion_latency_ms = (perf_counter() - fusion_started) * 1000.0

        if not fused:
            return []

        rerank_query = " | ".join(queries)
        rerank_started = perf_counter()
        final_results = self.reranker.rerank(
            query=rerank_query,
            candidates=fused,
            top_k=final_top_k,
        )
        reranking_latency_ms = (perf_counter() - rerank_started) * 1000.0
        total_latency_ms = (perf_counter() - started) * 1000.0

        for result in final_results:
            result["retrieval_latency_ms"] = retrieval_latency_ms
            result["fusion_latency_ms"] = fusion_latency_ms
            result["reranking_latency_ms"] = reranking_latency_ms
            result["total_latency_ms"] = total_latency_ms
            result["rerank_query"] = rerank_query

        return final_results

    def retrieve(
        self,
        subqueries: Sequence[str],
        subquery_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Stable high-level alias for retrieve_many()."""
        return self.retrieve_many(subqueries, subquery_top_k, final_top_k)

    @staticmethod
    def _validate_subqueries(subqueries: Sequence[str]) -> list[str]:
        if subqueries is None:
            raise ValueError("subqueries cannot be None")
        if isinstance(subqueries, (str, bytes)):
            raise TypeError("subqueries must be a sequence of strings")

        normalized: list[str] = []
        for index, query in enumerate(subqueries):
            if not isinstance(query, str):
                raise TypeError(f"subqueries[{index}] must be a string")
            query = query.strip()
            if query and query not in normalized:
                normalized.append(query)

        if not normalized:
            raise ValueError("subqueries must contain at least one non-empty query")
        return normalized

    @staticmethod
    def _validate_top_k(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

    @staticmethod
    def _annotate(
        results: list[dict[str, Any]],
        query: str,
        index: int,
    ) -> list[dict[str, Any]]:
        annotated = []
        for result in results:
            if not isinstance(result, dict):
                continue
            copied = result.copy()
            copied["source_subquery"] = query
            copied["source_subquery_index"] = index
            annotated.append(copied)
        return annotated

    def get_default_subquery_top_k(self) -> int:
        return self.default_subquery_top_k

    def get_default_final_top_k(self) -> int:
        return self.default_final_top_k

    def get_weights(self) -> dict[str, float]:
        return {
            "semantic": self.semantic_weight,
            "keyword": self.keyword_weight,
        }

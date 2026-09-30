from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.multi_query_retriever import MultiQueryRetriever


ROOT = Path(__file__).resolve().parents[1]


class FakeHybrid:
    def __init__(self):
        self.calls = []

    def search(self, query, semantic_top_k=None, keyword_top_k=None):
        self.calls.append(query)
        semantic = [
            {
                "chunk_id": f"semantic-{query}",
                "text": query,
                "score": 1.0,
                "metadata": {"source": "replay"},
            }
        ]
        keyword = [
            {
                "chunk_id": f"keyword-{query}",
                "text": query,
                "score": 1.0,
                "metadata": {"source": "replay"},
            }
        ]
        return {"semantic_results": semantic, "keyword_results": keyword}


class FakeDenseOnly(FakeHybrid):
    def search(self, query, semantic_top_k=None, keyword_top_k=None):
        self.calls.append(query)
        return {
            "semantic_results": [
                {
                    "chunk_id": f"semantic-{query}",
                    "text": query,
                    "score": 1.0,
                    "metadata": {"source": "replay"},
                }
            ],
            "keyword_results": [],
        }


class FakeReranker:
    def rerank(self, query, candidates, top_k=None):
        return candidates[:top_k] if top_k else candidates


def run(retriever, queries):
    start = perf_counter()
    results = retriever.retrieve(queries, final_top_k=5)
    return {
        "latency_ms": round((perf_counter() - start) * 1000, 3),
        "result_count": len(results),
    }


def main():
    queries = [
        "travel reimbursement policy",
        "international travel exception",
        "post travel booking",
    ]
    hybrid = MultiQueryRetriever(
        FakeHybrid(),
        ReciprocalRankFusion(),
        FakeReranker(),
        default_subquery_top_k=5,
        default_final_top_k=5,
    )
    dense = MultiQueryRetriever(
        FakeDenseOnly(),
        ReciprocalRankFusion(),
        FakeReranker(),
        default_subquery_top_k=5,
        default_final_top_k=5,
    )

    print(json.dumps({
        "dense_only": run(dense, queries),
        "hybrid": run(hybrid, queries),
        "note": "This is an architecture ablation harness. Replace the deterministic doubles with the held-out corpus replay for final quantitative claims.",
    }, indent=2))


if __name__ == "__main__":
    main()

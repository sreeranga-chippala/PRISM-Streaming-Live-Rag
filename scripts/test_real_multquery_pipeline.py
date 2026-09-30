from pathlib import Path

from src.orchestration.pipeline import StreamingRAGPipeline
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.streaming.transcript_stream import TranscriptChunk
from src.streaming.stream_controller import RetrievalAction


VECTOR_STORE_DIR = Path("data/processed/vector_store")

SUBQUERY_TOP_K = 5
FINAL_TOP_K = 5


def build_real_retriever():
    embedding_model = EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )

    vector_store = VectorStore(
        index_dir=VECTOR_STORE_DIR,
        dimension=embedding_model.get_dimension(),
    )

    semantic_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        default_top_k=SUBQUERY_TOP_K,
    )

    hybrid_search = HybridSearch(
        retriever=semantic_retriever,
        vector_store=vector_store,
        semantic_top_k=SUBQUERY_TOP_K,
        keyword_top_k=SUBQUERY_TOP_K,
    )

    fusion = ReciprocalRankFusion()

    reranker = Reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    return MultiQueryRetriever(
        hybrid_search=hybrid_search,
        fusion=fusion,
        reranker=reranker,
        default_subquery_top_k=SUBQUERY_TOP_K,
        default_final_top_k=FINAL_TOP_K,
    )


def main():
    retriever = build_real_retriever()

    pipeline = StreamingRAGPipeline(
        multi_query_retriever=retriever,
    )

    chunk = TranscriptChunk(
        session_id="real-multiquery-test",
        text=(
            "I want to understand our work from home policy, "
            "including who is eligible, how many days are allowed, "
            "and what security requirements apply when working remotely."
        ),
        timestamp=1.0,
        is_final=True,
    )

    result = pipeline.process_chunk(chunk)

    print("\n=== REAL MULTI-QUERY PIPELINE TEST ===")

    print("\nAction:")
    print(result.retrieval_decision.action)

    print("\nOriginal query:")
    print(result.retrieval_decision.query)

    print("\nDecomposition:")
    print(result.decomposition)

    print("\nRefined queries:")
    for i, query in enumerate(result.refined_queries, start=1):
        print(f"{i}. {query.refined_query}")

    print("\nRetrieved results:")

    for i, item in enumerate(result.retrieved_results, start=1):
        print(f"\n[{i}]")
        print("source:", item.get("metadata", {}).get("source"))
        print("score:", item.get("reranker_score"))
        print("text:", item.get("text", "")[:500])

    assert result.retrieval_decision.action == RetrievalAction.RETRIEVE
    assert result.decomposition is not None
    assert result.refined_queries

    # The query should be decomposed into multiple subqueries.
    assert len(result.refined_queries) > 1, (
        "Expected multiple refined queries, "
        f"got {len(result.refined_queries)}"
    )

    assert result.retrieved_results

    sources = [
        item.get("metadata", {}).get("source", "")
        for item in result.retrieved_results
    ]

    # The final results should contain the WFH policy.
    assert any(
        "03_Work_From_Home_Policy.pdf" in source
        for source in sources
    ), (
        f"Expected WFH policy document in results, got: {sources}"
    )

    # Security requirements may also surface the IT security policy.
    assert any(
        "07_IT_and_Data_Security_Policy.pdf" in source
        for source in sources
    ), (
        f"Expected IT/Data Security policy in results, got: {sources}"
    )

    assert result.session.retrieved_context

    print("\nREAL MULTI-QUERY PIPELINE TEST PASSED.")


if __name__ == "__main__":
    main()

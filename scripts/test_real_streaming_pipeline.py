from pathlib import Path

from src.orchestration.pipeline import StreamingRAGPipeline
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.fusion import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.multi_query_retriever import MultiQueryRetriever
from src.streaming.stream_controller import RetrievalAction
from src.streaming.transcript_stream import TranscriptChunk


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


def make_chunk(text, timestamp, is_final=False):
    return TranscriptChunk(
        session_id="real-streaming-test",
        text=text,
        timestamp=timestamp,
        is_final=is_final,
    )


def main():
    retriever = build_real_retriever()

    pipeline = StreamingRAGPipeline(
        multi_query_retriever=retriever,
    )

    print("\n=== REAL STREAMING PIPELINE TEST ===")

    # ---------------------------------------------------------
    # CHUNK 1: Incomplete utterance
    # ---------------------------------------------------------
    result1 = pipeline.process_chunk(
        make_chunk(
            "I need to",
            0.0,
        )
    )

    print("\n[Chunk 1]")
    print("Text: I need to")
    print("Action:", result1.retrieval_decision.action)
    print("Reason:", result1.retrieval_decision.reason)

    assert result1.retrieval_decision.action == RetrievalAction.WAIT
    assert result1.retrieved_results == ()
    assert result1.refined_queries == ()

    # ---------------------------------------------------------
    # CHUNK 2: First stable information request
    # ---------------------------------------------------------
    result2 = pipeline.process_chunk(
        make_chunk(
            "understand our work from home policy",
            1.0,
        )
    )

    print("\n[Chunk 2]")
    print("Accumulated query:", result2.retrieval_decision.query)
    print("Action:", result2.retrieval_decision.action)
    print("Reason:", result2.retrieval_decision.reason)

    assert result2.retrieval_decision.action == RetrievalAction.RETRIEVE
    assert result2.refined_queries
    assert result2.retrieved_results

    sources2 = [
        item.get("metadata", {}).get("source", "")
        for item in result2.retrieved_results
    ]

    print("Retrieved sources:", sources2)

    assert any(
        "03_Work_From_Home_Policy.pdf" in source
        for source in sources2
    ), (
        f"Expected WFH policy document, got: {sources2}"
    )

    # ---------------------------------------------------------
    # CHUNK 3: New information changes the query
    # ---------------------------------------------------------
    result3 = pipeline.process_chunk(
        make_chunk(
            "and what security requirements apply",
            2.0,
        )
    )

    print("\n[Chunk 3]")
    print("Accumulated query:", result3.retrieval_decision.query)
    print("Action:", result3.retrieval_decision.action)
    print("Reason:", result3.retrieval_decision.reason)

    assert result3.retrieval_decision.action == RetrievalAction.RETRIEVE
    assert result3.retrieval_decision.query
    assert result3.retrieval_decision.query != (
        result2.retrieval_decision.query
    )
    assert result3.retrieved_results

    sources3 = [
        item.get("metadata", {}).get("source", "")
        for item in result3.retrieved_results
    ]

    print("Retrieved sources:", sources3)

    assert any(
        "07_IT_and_Data_Security_Policy.pdf" in source
        for source in sources3
    ), (
        f"Expected IT/Data Security policy, got: {sources3}"
    )

    # ---------------------------------------------------------
    # CHUNK 4: Presentation request must not retrieve
    # ---------------------------------------------------------
    result4 = pipeline.process_chunk(
        make_chunk(
            "Please repeat your last answer in bullet points.",
            3.0,
            is_final=True,
        )
    )

    print("\n[Chunk 4]")
    print("Action:", result4.retrieval_decision.action)
    print("Reason:", result4.retrieval_decision.reason)

    assert (
        result4.retrieval_decision.action
        == RetrievalAction.NO_RETRIEVAL
    )
    assert result4.retrieved_results == ()
    assert result4.refined_queries == ()

    # ---------------------------------------------------------
    # SESSION STATE
    # ---------------------------------------------------------
    print("\nSession:")
    print("Session ID:", result4.session.session_id)
    print("Transcript chunks:", len(result4.session.transcript))
    print("Queries:", len(result4.session.queries))
    print("Retrieved context:", len(result4.session.retrieved_context))

    assert result4.session.session_id == "real-streaming-test"
    assert len(result4.session.transcript) == 4
    assert len(result4.session.queries) == 2
    assert result4.session.retrieved_context

    print("\nREAL STREAMING PIPELINE TEST PASSED.")


if __name__ == "__main__":
    main()

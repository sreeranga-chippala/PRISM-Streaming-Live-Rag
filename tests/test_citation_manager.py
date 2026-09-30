from src.generation.citation_manager import CitationManager


def test_citations_are_limited_to_evidence_supporting_answer():
    manager = CitationManager()

    answer = (
        "Employees must complete 6 months of continuous service "
        "to be eligible for WFH. "
        "Employees must also be grade L3 or above."
    )

    results = [
        {
            "text": (
                "Employees must have completed 6 months of continuous "
                "service to be eligible for Work From Home."
            ),
            "source": "03_Work_From_Home_Policy.pdf",
            "chunk_id": "wfh-1",
            "reranker_rank": 1,
            "reranker_score": -1.3,
        },
        {
            "text": (
                "Employees must currently be grade L3 or above "
                "to use the WFH arrangement."
            ),
            "source": "03_Work_From_Home_Policy.pdf",
            "chunk_id": "wfh-2",
            "reranker_rank": 2,
            "reranker_score": -2.2,
        },
        {
            "text": (
                "Employees may carry forward a maximum of 45 days "
                "of Earned Leave."
            ),
            "source": "02_Leave_Policy.pdf",
            "chunk_id": "leave-1",
            "reranker_rank": 4,
            "reranker_score": -6.2,
        },
        {
            "text": (
                "VPN must be active when accessing company systems "
                "from outside the office network."
            ),
            "source": "07_IT_and_Data_Security_Policy.pdf",
            "chunk_id": "security-1",
            "reranker_rank": 5,
            "reranker_score": -8.5,
        },
    ]

    citations = manager.build_citations(
        answer=answer,
        retrieved_results=results,
        max_citations=3,
        max_rank=3,
    )

    sources = [citation.source for citation in citations]

    assert "03_Work_From_Home_Policy.pdf" in sources
    assert "02_Leave_Policy.pdf" not in sources
    assert "07_IT_and_Data_Security_Policy.pdf" not in sources
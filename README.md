# PRISM Streaming Live RAG

## Samsung PRISM Generative AI Hackathon — 3rd Edition 2026–27

### Theme 4 — Streaming Live RAG

A streaming-aware Retrieval-Augmented Generation (RAG) system that processes continuously arriving context and decides when retrieval should happen instead of waiting for a complete query.

The system combines streaming context handling, retrieval decisioning, multi-intent decomposition, parallel hybrid retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, grounded generation, citations, session continuity, answer versioning, evaluation, and telemetry.

---

# 1. Problem Statement

Conventional RAG systems generally follow a sequential workflow:

```text
Complete User Query
        |
        v
Retrieve Documents
        |
        v
Generate Answer
```

This workflow is not designed specifically for continuously arriving information such as:

- Live transcripts
- Streaming text
- Voice interactions
- Meetings
- Lectures
- Continuously changing conversational context

Retrieving after every partial input can also result in unnecessary retrieval operations when the incoming context is incomplete or when retrieval is not required.

The objective of this project is to build a RAG system that can operate over streaming context and make an explicit decision about when retrieval should happen.

---

# 2. Proposed Solution

PRISM Streaming Live RAG processes incoming context incrementally.

Instead of immediately performing retrieval for every incoming piece of information, the system determines one of three actions:

```text
WAIT
RETRIEVE
NO_RETRIEVAL
```

The high-level workflow is:

```text
Streaming Context
        |
        v
Intent / Context Analysis
        |
        v
Retrieval Decision
   |        |        |
 WAIT   RETRIEVE   NO_RETRIEVAL
          |
          v
Multi-Intent Decomposition
          |
          v
Parallel Hybrid Retrieval
       /       \
      /         \
Semantic       Keyword
Retrieval      Retrieval
      \         /
       \       /
          RRF
          |
          v
Cross-Encoder Reranking
          |
          v
Grounded Generation
          |
          v
Citations + Groundedness
          |
          v
Answer Versioning
          |
          v
Session-Aware Response
```

---

# 3. Key Features

## 3.1 Streaming-Aware Retrieval Decision

Incoming context is processed incrementally.

The system can decide:

```text
WAIT
```

when more context is required.

```text
RETRIEVE
```

when sufficient information is available for retrieval.

```text
NO_RETRIEVAL
```

when retrieval is unnecessary for the current request.

This allows retrieval to be controlled according to the state of the incoming context.

---

## 3.2 Multi-Intent Query Decomposition

A user request may contain multiple information needs.

The system decomposes the request into sub-queries:

```text
User Request
      |
      v
Intent Detection
      |
      v
Query Decomposition
      |
      +----> Sub-query 1
      |
      +----> Sub-query 2
      |
      +----> Sub-query 3
```

The resulting sub-queries are passed to the retrieval pipeline.

---

## 3.3 Parallel Hybrid Retrieval

The retrieval system combines:

### Semantic Retrieval

Uses embedding-based retrieval with a vector store.

### Keyword Retrieval

Uses SQLite FTS5-based keyword retrieval.

The two retrieval approaches provide complementary signals.

```text
                 Query
                   |
          +--------+--------+
          |                 |
          v                 v
   Semantic Search    Keyword Search
          |                 |
          +--------+--------+
                   |
                   v
          Reciprocal Rank Fusion
```

---

## 3.4 Reciprocal Rank Fusion

Results from semantic and keyword retrieval are combined using Reciprocal Rank Fusion.

```text
Semantic Results
       |
       +----------------+
                        |
                        v
                Reciprocal Rank
                    Fusion
                        ^
                        |
       +----------------+
       |
Keyword Results
```

The fused results are then passed to the reranking stage.

---

## 3.5 Cross-Encoder Reranking

After hybrid retrieval and fusion, retrieved candidates are reranked using a cross-encoder.

```text
Hybrid Retrieval Results
          |
          v
Reciprocal Rank Fusion
          |
          v
Cross-Encoder Reranker
          |
          v
Final Relevant Context
```

This provides a final relevance ordering before the context is passed to generation.

---

## 3.6 Grounded Generation

The generation layer produces responses using retrieved context.

The system also evaluates the generated response for groundedness and maintains citation information associated with retrieved sources.

```text
Retrieved Context
       |
       v
Grounded Generation
       |
       +---------> Answer
       |
       +---------> Citations
       |
       +---------> Groundedness
```

---

## 3.7 Answer Versioning

When additional information becomes available, the system can refine the existing answer.

```text
Initial Context
      |
      v
Answer v1
      |
      v
Additional Context
      |
      v
Late / Delta Retrieval
      |
      v
Refined Answer v2
```

This supports incremental answer refinement during a streaming session.

---

## 3.8 Session Continuity

The system maintains session-level state including:

- Transcript context
- Previous queries
- Detected intents
- Retrieved context
- Current answer
- Session events

This allows later requests to use relevant information from the same session.

---

## 3.9 Evaluation and Telemetry

The project includes evaluation and telemetry support for:

- Retrieval count
- Retrieval recall when ground truth is available
- Citation count
- Groundedness
- Time to First Token (TTFT)
- Total latency
- Retrieval events
- Answer versions
- Token usage
- Estimated generation cost
- Uncertainty information

---

## 3.10 Monitoring Agent

The prototype also contains a session-scoped monitoring component.

It can process browser/session signals such as:

- Camera status
- Page focus
- Browser heartbeat
- Client IP observation
- IP consistency
- Warning events

The monitoring component is kept separate from the core RAG retrieval and generation pipeline.

---

# 4. System Architecture

```text
                         STREAMING INPUT
                    Transcript / Text / Context
                                |
                                v
                    +-------------------------+
                    | Streaming Controller    |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Intent Detection        |
                    | Query Refinement        |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Retrieval Decision      |
                    |                         |
                    | WAIT                    |
                    | RETRIEVE                |
                    | NO_RETRIEVAL            |
                    +------------+------------+
                                 |
                         RETRIEVE
                                 |
                                 v
                    +-------------------------+
                    | Multi-Query Decomposer  |
                    +------------+------------+
                                 |
                  +--------------+--------------+
                  |                             |
                  v                             v
        +-------------------+         +-------------------+
        | Semantic Search   |         | Keyword Search    |
        | SentenceTransform.|         | SQLite FTS5       |
        | + FAISS           |         |                   |
        +---------+---------+         +---------+---------+
                  |                             |
                  +--------------+--------------+
                                 |
                                 v
                    +-------------------------+
                    | Reciprocal Rank Fusion  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Cross-Encoder Reranker  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Grounded Generation      |
                    | + Citations              |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Groundedness Evaluation  |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Answer Versioning        |
                    | + Session Continuity     |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    | Streamlit Frontend       |
                    | Answers / Sources        |
                    | Retrieval / Metrics      |
                    +-------------------------+


                 SUPPORTING COMPONENTS
                 ---------------------

        +-------------------+    +-------------------+
        | Evaluation        |    | Telemetry         |
        +-------------------+    +-------------------+

        +-------------------+    +-------------------+
        | Monitoring Agent  |    | Session Manager   |
        +-------------------+    +-------------------+
```

---

# 5. Technology Stack

## Backend

- Python
- FastAPI
- Pydantic

## RAG and Retrieval

- Sentence Transformers
- FAISS
- SQLite FTS5
- Semantic Retrieval
- Keyword Retrieval
- Reciprocal Rank Fusion
- Cross-Encoder Reranking

## Generation

- Gemini API
- Grounded Answer Generation
- Citation Management
- Groundedness Checking

## Streaming and Intelligence

- Streaming Transcript Processing
- Intent Detection
- Query Decomposition
- Query Refinement
- Session Management
- Retrieval Decisioning

## Frontend

- Streamlit

## Evaluation

- Retrieval Metrics
- Groundedness Evaluation
- Latency Measurement
- Time-to-First-Token Measurement
- Citation Metrics
- Benchmarking
- Ablation Experiments
- Telemetry

## Deployment

- Docker
- Docker Compose

## Development

- Git
- GitHub
- Pytest

---

# 6. Repository Structure

```text
PRISM-Streaming-Live-Rag/
|
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── evaluation/
│   │   ├── ground_truth.json
│   │   └── test_queries.json
│   │
│   └── raw/
│       └── documents/
│           └── .gitkeep
│
├── docker/
│   └── api-entrypoint.sh
│
├── docs/
│   ├── benchmarking_evaluation.md
│   ├── benchmarking_evaluation_report.md
│   ├── system_architecture_brief.md
│   └── telemetry_schema.md
│
├── frontend/
│   ├── app.py
│   ├── question_bank.py
│   └── components/
│
├── scripts/
│   ├── benchmark_theme4.py
│   ├── build_index.py
│   ├── ingest_documents.py
│   ├── run_ablations.py
│   ├── run_demo.py
│   └── test_*.py
│
├── src/
│   ├── agents/
│   │   └── monitor_agent.py
│   │
│   ├── evaluation/
│   │   └── evaluator.py
│   │
│   ├── generation/
│   │   ├── answer_generator.py
│   │   ├── citation_manager.py
│   │   └── grounding_checker.py
│   │
│   ├── intelligence/
│   │   ├── intent_detector.py
│   │   ├── query_decomposer.py
│   │   ├── query_refiner.py
│   │   └── session_manager.py
│   │
│   ├── orchestration/
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── fusion.py
│   │   ├── hybrid_search.py
│   │   ├── multi_query_retriever.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── streaming/
│   │   ├── chunk_handler.py
│   │   ├── stream_controller.py
│   │   └── transcript_stream.py
│   │
│   └── utils/
│
├── tests/
│   ├── test_citation_manager.py
│   ├── test_decomposition.py
│   ├── test_generation.py
│   ├── test_intent.py
│   ├── test_pipeline.py
│   ├── test_reranking.py
│   ├── test_retrieval.py
│   ├── test_streaming.py
│   └── test_theme4_rules.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.lock
└── README.md
```

---

# 7. Prerequisites

Recommended environment:

- Python 3.11+
- Git
- Docker Desktop
- Internet connection
- API key for the configured generation provider

---

# 8. Environment Setup

Clone the repository:

```bash
git clone https://github.com/sreeranga-chippala/PRISM-Streaming-Live-Rag.git
cd PRISM-Streaming-Live-Rag
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

Add the required API credentials to `.env`.

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not commit `.env`.

The repository contains `.env.example` so that the required configuration structure can be reproduced without exposing credentials.

---

# 9. Running the Application Locally

## 9.1 Start the FastAPI Backend

From the repository root:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "prism-live-rag",
  "pipeline_ready": true
}
```

---

## 9.2 Start the Streamlit Frontend

Open another terminal and activate the environment:

```bash
source .venv/bin/activate
```

Run:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL displayed in the terminal.

For local execution, the frontend communicates with:

```text
http://127.0.0.1:8000
```

---

# 10. Running with Docker

The repository includes Docker configuration for running the application.

Build the Docker services:

```bash
docker compose build
```

Start the services:

```bash
docker compose up
```

Run in detached mode:

```bash
docker compose up -d
```

Stop the services:

```bash
docker compose down
```

Docker-related files:

```text
Dockerfile
docker-compose.yml
docker/api-entrypoint.sh
```

---

# 11. Document Ingestion and Indexing

The repository provides utilities for preparing documents and constructing retrieval indexes.

Document ingestion:

```bash
python scripts/ingest_documents.py
```

Index construction:

```bash
python scripts/build_index.py
```

Generated indexes, model artifacts, local databases, and temporary files are excluded from version control through `.gitignore`.

---

# 12. Running Tests

Run the test suite:

```bash
pytest
```

The repository contains tests covering areas including:

- Streaming
- Retrieval
- Reranking
- Intent detection
- Query decomposition
- Generation
- Citations
- Pipeline behavior
- Theme 4 rules

---

# 13. Evaluation and Benchmarking

The project contains evaluation and benchmarking utilities.

Run evaluation:

```bash
python scripts/evaluate.py
```

Run the Theme 4 benchmark:

```bash
python scripts/benchmark_theme4.py
```

Run ablation experiments:

```bash
python scripts/run_ablations.py
```

Additional evaluation and benchmarking documentation is available in:

```text
docs/benchmarking_evaluation.md
docs/benchmarking_evaluation_report.md
```

---

# 14. Demo Flow

A typical demonstration follows this workflow:

```text
1. Start FastAPI
        |
        v
2. Start Streamlit
        |
        v
3. Provide streaming/live context
        |
        v
4. System analyzes incoming context
        |
        v
5. Retrieval decision
        |
        +---- WAIT
        |
        +---- NO_RETRIEVAL
        |
        +---- RETRIEVE
                    |
                    v
             Query Decomposition
                    |
                    v
             Hybrid Retrieval
                    |
                    v
                   RRF
                    |
                    v
               Reranking
                    |
                    v
             Grounded Answer
                    |
                    v
          Citations + Metrics
                    |
                    v
             Answer Refinement
```

---

# 15. Innovation Highlights

## Streaming Retrieval Decision

The system does not automatically perform retrieval for every incoming fragment.

It explicitly exposes:

```text
WAIT
RETRIEVE
NO_RETRIEVAL
```

---

## Multi-Intent Retrieval

Complex requests can be decomposed into multiple sub-queries before retrieval.

---

## Parallel Hybrid Retrieval

Semantic and keyword retrieval are combined across multiple retrieval intents.

---

## RRF + Cross-Encoder Pipeline

Retrieved results are first fused using Reciprocal Rank Fusion and subsequently reranked using a cross-encoder.

---

## Incremental Answer Refinement

Additional context can trigger retrieval and refinement of an existing answer.

---

## Session-Aware RAG

Session state allows the system to maintain relevant conversational context across a streaming interaction.

---

## Grounding and Evaluation

The system exposes grounding, citation, retrieval, latency, and telemetry information for evaluating the behavior of the RAG pipeline.

---

# 16. Expected Use Cases

The architecture can be applied to scenarios where information arrives continuously, including:

- Live meeting assistants
- Live lecture assistants
- Voice assistants
- Customer-support conversations
- Live technical support
- Streaming document analysis
- Interactive knowledge assistants
- Real-time enterprise search

The current implementation is a hackathon prototype demonstrating the underlying streaming RAG architecture.

---

# 17. Limitations

This project is a hackathon prototype.

Current limitations include:

- Generation depends on the configured external model/API provider.
- Retrieval quality depends on the indexed document collection.
- Local execution requires the required model dependencies and API credentials.
- Latency depends on hardware, model loading, network conditions, and external API response time.
- The monitoring component is a prototype/session-level implementation and should not be considered a production security or proctoring system.
- Docker execution depends on the local Docker environment and successful dependency installation.
- Large-scale distributed deployment is outside the scope of the current prototype.

---

# 18. Future Improvements

Potential extensions include:

- More efficient incremental embedding and indexing
- Improved streaming retrieval decision models
- Adaptive retrieval thresholds
- Learned retrieval timing policies
- Larger-scale vector databases
- Distributed retrieval
- Improved streaming generation
- More advanced delta retrieval
- Larger evaluation datasets
- Production-grade observability
- Scalable cloud deployment
- Multimodal streaming inputs

---

# 19. Documentation

Detailed project documentation is available in:

```text
docs/system_architecture_brief.md
docs/benchmarking_evaluation.md
docs/benchmarking_evaluation_report.md
docs/telemetry_schema.md
```

---

# 20. Submission Materials

## GitHub Repository

Public repository:

https://github.com/sreeranga-chippala/PRISM-Streaming-Live-Rag

---

## Final Presentation

PPT / PDF:

> ADD_FINAL_PPT_LINK_HERE

The presentation covers:

- Theme
- Existing solutions and gaps
- Proposed solution
- System architecture
- Demo and product walkthrough
- Technology stack
- Impact and use cases
- Innovation highlights
- Results
- Limitations
- Future work
- Differentiation

---

## Demo Video

Maximum duration:

```text
5 minutes
```

Demo video:

> ADD_DEMO_VIDEO_LINK_HERE

The video demonstrates the working prototype and the primary Streaming Live RAG workflow.

---

# 21. Team

## Samsung PRISM Generative AI Hackathon — Theme 4

| Member | Email |
|---|---|
| Chippala Sree Ranganath | 1MS24AI017@msrit.edu |
| Adarsha S Biradar | 1MS24AI003@msrit.edu |
| Vikas H V | 1MS24AI072@msrit.edu |

---

# 22. Final Submission Checklist

| Requirement | Status |
|---|---|
| Working prototype code | Complete |
| Public/shared GitHub repository | Complete |
| README with reproducible setup instructions | Complete |
| Requirements file | Complete |
| Dockerfile | Complete |
| Docker Compose configuration | Complete |
| Project documentation | Complete |
| Presentation file/link | Add final PPT link |
| Demo video link | Add final video link |
| Final Git tag | `PRISM_GENAI_HACKATHON_Y2026` |

# 23. Project Status

```text
Project: PRISM Streaming Live RAG

Hackathon:
Samsung PRISM Generative AI Hackathon
3rd Edition 2026–27

Theme:
Theme 4 — Streaming Live RAG

Repository:
https://github.com/sreeranga-chippala/PRISM-Streaming-Live-Rag

Final Tag:
PRISM_GENAI_HACKATHON_Y2026
```

---

# 24. Acknowledgement

This project was developed as a submission for the Samsung PRISM Generative AI Hackathon — 3rd Edition 2026–27, Theme 4: Streaming Live RAG.

---

# Thank You

## PRISM Streaming Live RAG

### Samsung PRISM Generative AI Hackathon — 3rd Edition 2026–27
### Theme 4 — Streaming Live RAG
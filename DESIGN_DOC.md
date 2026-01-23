This document provides the high-fidelity engineering specification for rebuilding the `knowledge_base` system. It is designed to be given directly to your **@Architect** agent to provide a "dead start" without requiring access to the previous codebase.

# DESIGN_DOC.md: Agentic Knowledge Ingestion & Management System

## 1. System Overview & Standard Compliance

This system is a high-fidelity information extraction engine that transforms unstructured data into a structured, temporally-aware knowledge graph.

* **Google Engineering Standards:**
* **AIP-121 (Resource-oriented):** The API surface is a collection of addressable resources (`Documents`, `Chunks`, `Entities`, `Communities`). [AIP-121]
* **AIP-122 (Naming):** Standardized hierarchical paths (e.g., `documents/{doc_id}/chunks/{chunk_id}`). [AIP-122]
* **AIP-191 (Layout):** Strictly follows the `src/` directory layout where packages match the directory structure.


* **Operational Mandate:** Direct local machine hosting (bare-metal) using `uv` for Python. No Docker.

## 2. Technical Stack Configuration

* **Language:** Python 3.12+ (Type-hinted, Google Style Guide compliant). 


* **Database:** PostgreSQL 16+ with `pgvector` 0.8.1.
* *Optimization:* `maintenance_work_mem = '4GB'`, `hnsw.ef_search = 100`, `hnsw.iterative_scan = strict_order`. 




* **Embeddings:** Google `gemini-embedding-001` (Task type: `RETRIEVAL_DOCUMENT` for storage, `RETRIEVAL_QUERY` for search). 


* **LLM Gateway:** Self-hosted OpenAI-compatible API (vLLM/Ollama/LM Studio).
* **Observability:** Pydantic Logfire for SRE-lite tracing and token tracking. 



## 3. Core Feature Specification

### Feature 1: High-Resolution Adaptive Gleaning

Unlike fixed extraction, this uses a density-aware 2-pass strategy.

* **Pass 1 (Discovery):** Extract obvious entities and relationships.
* **Density Assessment:** The agent evaluates the "remaining information density." If density > threshold, trigger Pass 2.
* **Pass 2 (Gleaning):** Focus specifically on subtle, nested, or technical relationships missed in Pass 1.

### Feature 2: Verbatim-Grounded Entity Resolution

Deduplication of nodes (e.g., "Dr. Vance" and "Elena Vance") must be cognitive, not just mathematical.

* **Hybrid Matching:** Use vector similarity to find candidates, then use the LLM to reason.
* **Mandatory Grounding:** Every resolution decision MUST include a `grounding_quote` (verbatim excerpt from the source).
* **Confidence Scoring:** Assign a `confidence_score` (0.0-1.0). Merges < 0.7 are flagged for "Human-in-the-loop" review.

### Feature 3: Hierarchical Leiden Clustering

Groups nodes into meaningful communities for sense-making at scale.

* **Algorithm:** Leiden Community Detection (optimized for modularity).
* **Stability Knobs:** Use a lower `gamma` (resolution parameter) for Macro-themes and a higher `gamma` for Micro-communities.
* **Incremental Updates:** Use a "Dynamic Leiden" approach to update clusters when new nodes arrive without re-clustering the entire graph.

### Feature 4: Map-Reduce Recursive Summarization

Generates "Intelligence Reports" for the graph hierarchy.

* **Micro-Reports:** factual, detail-heavy summaries of individual clusters.
* **Macro-Reports:** Strategic synthesis of child reports.
* **Fidelity Rule:** When summarizing, the agent must have access to "High-Confidence Edges" (raw relationships) of children, not just their text summaries, to prevent "Information Smoothing."

### Feature 5: Temporal Information Extraction (TIE)

* **Classification:** Label every claim as `Atemporal` (eternal facts), `Static` (valid from a point in time), or `Dynamic` (states that evolve).
* **Normalization:** Identify relative dates ("three days ago") and resolve to absolute **ISO-8601 (TIMEX3)** based on document metadata.
* **Versioning:** Newer Dynamic claims with later timestamps invalidate older ones via an `invalidated_by` edge.

## 4. Architectural Components

### I. Ingestion Plane (Custom ReAct Loop)

A custom `asyncio` loop following the **Reason+Act** pattern. No heavy frameworks.

1. **Partition:** Local `unstructured` partitioning (PDF/XLSX/DOCX). 


2. **Chunk:** Page-level or Semantic chunking (256-512 tokens for factoid accuracy). 


3. **Extract:** Adaptive gleaning extraction.
4. **Embed:** Batch-process through Google Embeddings. 



### II. Persistence Plane (pgvector HNSW)

* **Schema:** Relational metadata (AIP-122 compliant) + high-dimensional vectors.
* **Index Tuning:** HNSW index with `m=16` and `ef_construction=64`. 



### III. SRE-Lite Observability

* **Instrumentation:** Logfire `configure()` at startup.
* **Metrics:** Real-time dashboarding of **Latency**, **Success Rate**, and **Token Throughput**. 


* **Traces:** Single end-to-end timeline from file-drop to vector-store. 



## 5. Directory Structure (AIP-191)

src/
└── knowledge_base/
├── ingestion/
│   └── v1/
│       ├── partitioning_service.py   # unstructured local logic
│       ├── gleaning_service.py       # 2-pass adaptive logic
│       └── embedding_client.py       # Google embedding wrapper
├── persistence/
│   └── v1/
│       ├── vector_store.py           # pgvector/HNSW logic
│       └── schema.py                 # Pydantic models
├── intelligence/
│   └── v1/
│       ├── resolution_agent.py       # Grounded deduplication
│       ├── clustering_service.py     # Leiden algorithm
│       └── synthesis_agent.py        # Map-Reduce reporting
└── common/
├── temporal_utils.py             # ISO-8601/TIMEX3 logic
└── gateway.py                    # Local LLM API client

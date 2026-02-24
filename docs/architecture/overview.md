# KBV2 System Architecture

## Overview

KBV2 is a high-fidelity information extraction engine that transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction techniques.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KBV2 KNOWLEDGE BASE                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   USER INPUT │
│  (Document)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION ORCHESTRATOR                               │
│                      (SelfImprovingOrchestrator)                            │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DOCUMENT CREATION                                                        │
│  - Initialize document record (PENDING status)                              │
│  - Store metadata (file path, MIME type, size)                              │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. DOMAIN DETECTION                                                         │
│  - Keyword screening for crypto indicators                                  │
│  - LLM analysis for domain classification                                    │
│  - Auto-detection or manual specification                                    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DOCUMENT PARTITIONING                                                   │
│  - Semantic chunking (default 1536 tokens, 25% overlap)                      │
│  - Extract titles and structure                                             │
│  - Create chunk records                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. ADAPTIVE ANALYSIS                                                        │
│  - LLM analyzes document complexity                                         │
│  - Recommends processing strategy:                                          │
│    * Simple: Gleaning mode (~3-5 LLM calls)                                 │
│    * Moderate: Enhanced gleaning (~12-15 LLM calls)                         │
│    * Complex: Full multi-agent (~25-30 LLM calls)                           │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. EMBEDDING GENERATION                                                     │
│  - Generate 1024-dim vectors using bge-m3 (Ollama)                          │
│  - Batch processing for performance                                         │
│  - Store in pgvector with IVFFlat indexes                                   │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. ENTITY EXTRACTION                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Multi-Agent Extractor (GraphMaster-style)                           │  │
│  │  ├─ Perception Agent: Initial entity extraction                       │  │
│  │  ├─ Enhancement Agent: Improve entity quality                        │  │
│  │  └─ Evaluation Agent: Quality scoring                                │  │
│  │                                                                      │  │
│  │  Self-Improvement Integration:                                       │  │
│  │  ├─ Experience Bank: Retrieve similar examples                       │  │
│  │  ├─ Prompt Evolution: Use domain-optimized prompts                   │  │
│  │  └─ Store high-quality extractions (quality ≥ 0.75)                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. ENTITY PROCESSING                                                       │
│  - Resolution: Merge duplicates with verbatim grounding                      │
│  - Typing: Domain-aware classification                                      │
│  - Clustering: Hierarchical Leiden (macro + micro)                          │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. QUALITY ASSURANCE                                                        │
│  - Ontology validation (15+ crypto rules)                                   │
│  - Hallucination detection (LLM-as-Judge)                                   │
│  - Schema validation                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  9. STORAGE                                                                  │
│  - Document metadata → PostgreSQL                                            │
│  - Entities/Edges → PostgreSQL                                               │
│  - Embeddings → pgvector (1024 dims)                                         │
│  - High-quality extractions → Experience Bank                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### API Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── Document API     - Document management                                   │
│  ├── Query API        - Natural language to SQL                               │
│  ├── Review API       - Human review workflow                                │
│  ├── Graph API        - Knowledge graph operations                            │
│  ├── Unified Search   - Multi-mode search (hybrid/reranked)                  │
│  └── MCP Server       - WebSocket protocol for real-time                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Intelligence Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Intelligence Layer (LLM-Powered Services)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── Multi-Agent Extractor      - GraphMaster-style extraction               │
│  ├── Entity Typing Service      - Domain-aware classification                 │
│  ├── Hallucination Detector     - LLM-as-Judge validation                     │
│  ├── Resolution Agent           - Verbatim-grounded deduplication            │
│  ├── Clustering Service         - Hierarchical Leiden clustering              │
│  ├── Adaptive Ingestion Engine  - Document complexity analysis               │
│  ├── Hybrid Retriever           - Vector + Graph search                       │
│  └── Self-Improvement:                                                    │
│      ├── Experience Bank        - Few-shot learning                           │
│      ├── Prompt Evolution        - Automated optimization                     │
│      ├── Ontology Validator      - Rule-based validation                      │
│      └── Domain Detection Feedback - Learning from classification            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Ingestion Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Ingestion Layer                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── Partitioning Service     - Document chunking                            │
│  ├── Embedding Client         - 1024-dim vectors (bge-m3)                    │
│  └── Gleaning Service         - 2-pass adaptive extraction                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Persistence Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Persistence Layer                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ├── Vector Store             - pgvector with IVFFlat indexes                │
│  ├── Graph Store              - Knowledge graph storage                      │
│  ├── Schema                   - SQLAlchemy models                            │
│  └── Hybrid Search            - BM25 + Vector search                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Input: Document File                                                         │
│  - Supported formats: .md, .txt, .pdf, .docx, .html, .json                   │
│  - Domain: Auto-detected or specified (16 domains)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Partitioning Data

```python
Chunk {
    id: UUID
    document_id: UUID
    text: str
    chunk_index: int
    token_count: int
    page_number: Optional[int]
    metadata: JSON
    embedding: vector(1024)  # bge-m3
}
```

### Extraction Data

```python
Entity {
    id: UUID
    name: str
    entity_type: str
    description: Optional[str]
    properties: JSON
    confidence: float
    embedding: vector(1024)
    uri: str  # RDF-style unique identifier
    source_text: str
    domain: str
    community_id: Optional[UUID]
}

Edge {
    id: UUID
    source_id: UUID
    target_id: UUID
    edge_type: EdgeType  # 30+ types
    properties: JSON
    confidence: float
    temporal_validity_start: Optional[datetime]
    temporal_validity_end: Optional[datetime]
    provenance: str
    source_text: str
    is_directed: bool
    domain: str
}
```

### Clustering Data

```python
Community {
    id: UUID
    name: str
    level: int  # 0 = macro, 1 = micro
    resolution: float
    summary: str
    entity_count: int
    parent_id: Optional[UUID]  # Hierarchical
}
```

---

## Key Features

### 1. Adaptive Gleaning (2-Pass Extraction)

- **Pass 1 (Discovery):** Extract obvious entities and relationships
- **Pass 2 (Gleaning):** Find subtle, nested, technical relationships
- **Adaptive Continuation:** Only runs Pass 2 if information density > 0.3

### 2. Verbatim-Grounded Entity Resolution

- **Hybrid Matching:** Vector similarity + LLM reasoning
- **Verbatim Grounding:** Requires direct quotes from source text
- **Conservative Approach:** Keeps entities separate when uncertain
- **Human Review:** Low-confidence decisions added to ReviewQueue

### 3. Hierarchical Leiden Clustering

- **Level 0 (Macro):** Broad communities (resolution 0.8)
- **Level 1 (Micro):** Tight communities (resolution 1.2)
- **Hierarchy:** Micro communities have macro parents
- **Incremental Updates:** Supports adding new entities

### 4. Self-Improvement

- **Experience Bank:** Few-shot learning from high-quality extractions (quality ≥ 0.75)
- **Prompt Evolution:** Automated prompt optimization for 5 crypto domains
- **Ontology Validation:** 15+ crypto-specific validation rules
- **Domain Detection Feedback:** Learning from classification accuracy

### 5. Temporal Knowledge Graph

- **Temporal Claims:** Classified as atemporal, static, or dynamic
- **ISO-8601 Normalization:** All dates normalized to ISO-8601 format
- **Temporal Validity:** Edges have validity start/end times
- **Claim Invalidation:** Newer claims invalidate older ones

### 6. Hybrid Search

- **BM25:** Keyword-based search
- **Vector:** Semantic similarity search (1024 dims)
- **Reranking:** Cross-encoder for improved results
- **RRF:** Reciprocal Rank Fusion for multi-query

---

## Technology Stack

- **Language:** Python 3.12+
- **Database:** PostgreSQL 16+ with pgvector
- **Vector Search:** pgvector (IVFFlat indexes, 1024-dim vectors)
- **Embeddings:** Ollama (bge-m3, 1024 dimensions)
- **LLM Gateway:** OpenAI-compatible API (http://localhost:8087/v1)
- **LLM Client:** AsyncOpenAI SDK with random model rotation
- **Clustering:** igraph + leidenalg
- **API:** FastAPI
- **Observability:** Logfire

---

## Edge Types (30+)

### Hierarchical
- PART_OF, SUBCLASS_OF, INSTANCE_OF, CONTAINS

### Causal
- CAUSES, CAUSED_BY, INFLUENCES, INFLUENCED_BY, ENABLES, PREVENTS

### Temporal
- PRECEDES, FOLLOWS, CO_OCCURS_WITH, OVERLAPS_WITH

### Spatial
- LOCATED_IN, LOCATED_NEAR, ADJACENT_TO, WITHIN

### Social/Organizational
- WORKS_FOR, WORKS_WITH, REPORTS_TO, KNOWS, COLLEAGUE_OF

### Ownership
- OWNS, MANAGES, OPERATES

### Activity
- PARTICIPATES_IN, PERFORMS, TARGETS, AFFECTS

### Long-tail
- UNKNOWN, NOTA (None-of-the-Above), HYPOTHETICAL

---

## Related Documentation

- [Database Schema](../database/schema.md)
- [API Endpoints](../api/endpoints.md)
- [Ingestion Guide](../guides/ingestion.md)
- [Deployment Guide](../guides/deployment.md)

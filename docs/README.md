# KBV2 Documentation

This directory contains ASCII diagrams and documentation for the KBV2 (Knowledge Base Version 2) system.

## Overview

KBV2 is a high-fidelity information extraction engine that transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction techniques.

## Documentation Files

### 1. [architecture/system_overview.md](architecture/system_overview.md)
**Purpose:** Visual representation of the 9-stage ingestion pipeline

**Contents:**
- Complete ReAct orchestration flow
- Stage-by-stage breakdown:
  1. Create Document
  2. Partition Document
  3. Extract Knowledge (Adaptive Gleaning)
  4. Embed Content
  5. Resolve Entities (Verbatim-Grounded)
  6. Cluster Entities (Hierarchical Leiden)
  7. Generate Reports (Map-Reduce)
  8. Update Domain
  9. Complete

**Key Features:**
- 2-pass adaptive extraction (Discovery + Gleaning)
- Verbatim-grounded entity resolution
- Hierarchical clustering (Macro + Micro)
- Map-reduce summarization with edge fidelity

---

### 2. [architecture/data_flow.md](architecture/data_flow.md)
**Purpose:** Detailed data transformation through the pipeline

**Contents:**
- Input data processing (PDF/DOCX/Text)
- Partitioning data structure
- Extraction data (Pass 1 + Pass 2)
- Entity creation and embedding
- Entity resolution process
- Clustering data (Level 0 + Level 1)
- Synthesis data (Micro + Macro reports)
- Domain propagation
- Final data state

**Key Features:**
- Step-by-step data transformation
- Example data structures
- Confidence scores and thresholds
- URI generation (RDF-style)
- Temporal claim normalization

---

### 3. [development/folder_structure.md](development/folder_structure.md)
**Purpose:** Complete directory structure and file organization

**Contents:**
- Root level files and directories
- Source code structure (`src/knowledge_base/`)
- Module breakdown:
  - Core modules (orchestrator, APIs, services)
  - Common module (gateway, temporal utils, resilient gateway)
  - Ingestion module (partitioning, gleaning, embedding)
  - Persistence module (schema, vector store)
  - Intelligence module (resolution, clustering, synthesis)
- Test structure
- Documentation files
- Configuration files

**Key Features:**
- File naming conventions
- Class and method naming
- Version convention (v1/)
- Dependency injection pattern

---

### 4. [database/schema.md](database/schema.md)
**Purpose:** Database schema relationships and table definitions

**Contents:**
- Entity-Relationship diagram
- Table descriptions:
  - Document
  - Chunk
  - Entity
  - Edge
  - ChunkEntity (junction)
  - Community
  - ReviewQueue

**Key Relationships:**
- One-to-Many: Document → Chunk, Community → Entity
- Many-to-Many: Chunk ↔ Entity, Entity ↔ Edge
- Self-Referential: Community → Community (hierarchy)

**Edge Types (30+):**
- Hierarchical: PART_OF, SUBCLASS_OF, INSTANCE_OF
- Causal: CAUSES, INFLUENCES
- Temporal: PRECEDES, FOLLOWS
- Social: WORKS_FOR, KNOWS
- Ownership: OWNS, MANAGES
- Long-tail: UNKNOWN, NOTA, HYPOTHETICAL

---

### 5. [api/endpoints.md](api/endpoints.md)
**Purpose:** Complete API reference for all endpoints

**Contents:**

#### Query API (`/api/v1/query`)
- `POST /translate` - Natural language to SQL (no execution)
- `POST /execute` - Translate and execute query
- `GET /schema` - Get database schema

#### Review API (`/api/v1/review`)
- `GET /pending` - Get pending reviews by priority
- `GET /{review_id}` - Get specific review
- `POST /{review_id}/approve` - Approve review with notes
- `POST /{review_id}/reject` - Reject with corrections

#### MCP Server (WebSocket `/ws`)
- `kbv2/ingest_document` - Ingest document
- `kbv2/query_text_to_sql` - Execute text-to-SQL
- `kbv2/search_entities` - Vector search entities
- `kbv2/search_chunks` - Vector search chunks
- `kbv2/get_document_status` - Get processing status

**Security Features:**
- SQL injection prevention
- Schema validation
- Query timeout (5s)
- Result limits (1000 rows)
- Authentication/authorization

---

### 6. [configuration/environment.md](configuration/environment.md)
**Purpose:** Configuration management and service initialization

**Contents:**

#### Environment Variables
- `DATABASE_URL` - PostgreSQL connection
- `OLLAMA_URL` - Ollama embedding service
- `OLLAMA_MODEL` - Embedding model (nomic-embed-text)
- `LLM_GATEWAY_URL` - LLM API gateway
- `GOOGLE_API_KEY` - Optional Google embeddings

#### Configuration Classes
- **GleaningConfig** - Adaptive extraction parameters
  - Density thresholds (0.3-0.8)
  - Max passes (2)
  - Diminishing returns (5%)
  - Stability (90%)

- **ResolutionConfig** - Entity deduplication parameters
  - Confidence threshold (0.7)
  - Similarity threshold (0.85)
  - Max candidates (10)

- **ClusteringConfig** - Hierarchical clustering parameters
  - Min community size (3)
  - Iterations (10)
  - Macro resolution (0.8)
  - Micro resolution (1.2)

- **SynthesisConfig** - Report generation parameters
  - Max tokens (2000)
  - Edge fidelity (true)

#### Service Initialization
- Dependency injection pattern
- Initialization sequence
- Cleanup procedures
- Connection management

**Best Practices:**
- Environment variable security
- Type-safe configuration (Pydantic)
- Startup validation
- Error handling

---

## Key System Features

### 1. Adaptive Gleaning (2-Pass Extraction)
- **Pass 1 (Discovery):** Extract obvious entities and relationships
- **Pass 2 (Gleaning):** Find subtle, nested, technical relationships
- **Adaptive Continuation:** Only runs Pass 2 if information density > 0.3 and stability < 90%

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

### 4. Map-Reduce Summarization
- **Micro Reports:** Detailed summaries of leaf communities
- **Macro Reports:** Strategic synthesis of child reports
- **Edge Fidelity:** Preserves raw relationships to prevent information smoothing

### 5. Temporal Knowledge Graph
- **Temporal Claims:** Classified as atemporal, static, or dynamic
- **ISO-8601 Normalization:** All dates normalized to ISO-8601 format
- **Temporal Validity:** Edges have validity start/end times
- **Claim Invalidation:** Newer claims invalidate older ones

### 6. Long-Tail Relation Handling
- **NOTA (None-of-the-Above):** For rare/unclassifiable relations
- **HYPOTHETICAL:** For uncertain relationships
- **Based on 2026 DOREMI Research:** Optimizing long-tail predictions in document-level relation extraction

### 7. Resilient Gateway
- **Circuit Breaker:** CLOSED → OPEN → HALF_OPEN states
- **Exponential Backoff:** Retry with jitter for transient failures
- **Model Switching:** Automatic fallback on rate limits (429)
- **Metrics Collection:** Requests, successes, failures, retries

### 8. Security-First Text-to-SQL
- **SQL Injection Prevention:** Pattern matching against dangerous keywords
- **Schema Validation:** Whitelist-based identifier validation
- **Query Timeout:** 5-second limit
- **Result Limits:** Maximum 1000 rows

## Technology Stack

- **Language:** Python 3.12+
- **Database:** PostgreSQL 16+ with pgvector
- **Vector Search:** pgvector (IVFFlat indexes, 768-dim vectors)
- **Document Parsing:** unstructured library
- **Embeddings:** Ollama (nomic-embed-text)
- **LLM Gateway:** OpenAI-compatible API (gemini-2.5-flash-lite)
- **Clustering:** igraph + leidenalg
- **API:** FastAPI
- **Observability:** Logfire

## Getting Started

1. **Install Dependencies:**
   ```bash
   uv sync
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Setup Database:**
   ```bash
   createdb knowledge_base
   python scripts/setup_db.py
   ```

4. **Run:**
   ```bash
   uv run knowledge-base
   ```

## Architecture Patterns

### 1. ReAct Orchestration
- Sequential pipeline with clear status transitions
- Each stage is independent and observable
- Error handling with status updates

### 2. Service Layer Pattern
- Stateless service classes
- Configuration injected via constructor
- Clear separation of concerns

### 3. Repository Pattern
- VectorStore abstracts database operations
- Async session management
- Connection pooling

### 4. Resilience Patterns
- Circuit breaker for fault tolerance
- Exponential backoff with jitter
- Automatic model switching

## Documentation Conventions

### ASCII Diagrams
- Uses box-drawing characters (┌─┐│└┘┼┬┴├┤)
- Monospace font for alignment
- Clear labels and arrows
- Consistent styling

### Code Examples
- Python syntax highlighting
- Type hints included
- Docstrings in Google format
- Error handling shown

### Tables
- Markdown tables for parameters
- Default values specified
- Types and descriptions included
- Examples provided

## Related Documentation

- **Root Level:**
  - [README.md](../README.md) - Project overview
  - [DESIGN_DOC.md](../DESIGN_DOC.md) - High-level system design
  - [SPEC.md](../SPEC.md) - Architecture specification
  - [SETUP.md](../SETUP.md) - Setup instructions

- **Implementation:**
  - [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Changes summary
  - [ENTITY_PROCESSING_PIPELINE.md](../ENTITY_PROCESSING_PIPELINE.md) - Entity processing
  - [VALIDATION_SUMMARY.md](../VALIDATION_SUMMARY.md) - Validation results

- **Planning:**
  - [implementation_plan.md](../implementation_plan.md) - Implementation roadmap

## Support

- **Issues:** Report bugs and feature requests at the project repository
- **Questions:** Check existing documentation and issues first
- **Contributions:** Follow Google Python Style Guide and existing patterns

## License

See project LICENSE file for details.
---
KBV2 Knowledge Base System - Comprehensive Architecture Overview
1. PROJECT TYPE AND OVERVIEW
KBV2 is an Agentic Knowledge Ingestion & Management System - a high-fidelity information extraction engine that transforms unstructured data into structured, temporally-aware knowledge graphs. It is a sophisticated RAG (Retrieval-Augmented Generation) system with advanced entity extraction capabilities.
Key Characteristics:
- Vector Database with pgvector (PostgreSQL) - 1024-dimensional embeddings (bge-m3 compatible)
- Knowledge Graph with entity-relationship modeling
- Hybrid Search (BM25 + Vector similarity + Cross-encoder reranking)
- Multi-agent LLM-powered extraction pipeline
- Domain-aware processing with auto-detection
- MCP (Model Context Protocol) server for external tool integration
---
2. PROJECT STRUCTURE
/home/muham/development/kbv2/
├── src/knowledge_base/               # Main source code
│   ├── main.py                       # FastAPI entry point
│   ├── orchestrator.py               # ReAct loop orchestrator
│   ├── mcp_server.py                 # MCP protocol server (WebSocket)
│   ├── api/                          # API modules
│   │   └── unified_search_api.py
│   ├── clients/                      # LLM clients & middleware
│   │   ├── llm_client.py
│   │   ├── unified_llm_client.py
│   │   ├── rotating_llm_client.py
│   │   ├── rotation_manager.py
│   │   ├── cli.py                    # CLI interface
│   │   └── middleware/               # Resilience patterns
│   │       ├── circuit_breaker.py
│   │       ├── retry_middleware.py
│   │       └── rotation_middleware.py
│   ├── common/                       # Shared utilities
│   │   ├── api_models.py
│   │   ├── dependencies.py
│   │   ├── error_handlers.py
│   │   ├── gateway.py                # LLM gateway client
│   │   ├── resilient_gateway/        # Circuit breaker patterns
│   │   ├── pagination.py
│   │   └── offset_service.py
│   ├── domain/                       # Domain management
│   │   ├── domain_models.py
│   │   ├── detection.py              # Auto domain detection
│   │   └── ontology_snippets.py      # Domain keywords/ontologies
│   ├── extraction/                   # Extraction components
│   │   ├── guided_extractor.py
│   │   └── template_registry.py
│   ├── ingestion/                    # Data ingestion pipeline
│   │   └── v1/
│   │       ├── embedding_client.py
│   │       ├── gleaning_service.py   # 2-pass adaptive extraction
│   │       └── partitioning_service.py
│   ├── intelligence/                 # LLM-powered intelligence
│   │   └── v1/
│   │       ├── adaptive_ingestion_engine.py
│   │       ├── clustering_service.py
│   │       ├── cross_domain_detector.py
│   │       ├── domain_schema_service.py
│   │       ├── entity_typing_service.py
│   │       ├── federated_query_router.py
│   │       ├── hallucination_detector.py
│   │       ├── hybrid_retriever.py
│   │       ├── multi_agent_extractor.py  # GraphMaster-style
│   │       ├── resolution_agent.py
│   │       └── synthesis_agent.py
│   ├── orchestration/                # Pipeline services
│   │   ├── base_service.py
│   │   ├── document_pipeline_service.py
│   │   ├── domain_detection_service.py
│   │   ├── entity_pipeline_service.py
│   │   └── quality_assurance_service.py
│   ├── persistence/                  # Database layer
│   │   └── v1/
│   │       ├── graph_store.py
│   │       ├── schema.py             # SQLAlchemy models
│   │       └── vector_store.py
│   ├── processing/                   # Batch processing
│   │   └── batch_processor.py
│   ├── reranking/                    # Search reranking
│   │   ├── cross_encoder.py
│   │   ├── reranking_pipeline.py
│   │   └── rrf_fuser.py
│   ├── storage/                      # Search indexes
│   │   ├── bm25_index.py
│   │   └── hybrid_search.py
│   ├── summaries/                    # Community summaries
│   │   └── community_summaries.py
│   ├── types/                        # Type system
│   │   ├── schema_inducer.py
│   │   ├── type_discovery.py
│   │   └── validation_layer.py
│   ├── document_api.py               # Document management API
│   ├── graph_api.py                  # Graph visualization API
│   ├── query_api.py                  # Query/search API
│   ├── review_api.py                 # Human review queue API
│   ├── review_service.py
│   ├── schema_api.py                 # Domain schema API
│   └── text_to_sql_agent.py
├── tests/                            # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── alembic/                          # Database migrations
│   └── versions/
├── docs/                             # Documentation
│   ├── architecture/
│   ├── api/
│   ├── database/
│   └── overview/
├── scripts/                          # Utility scripts
└── test_data/                        # Sample documents
---
3. CONFIGURATION FILES
/home/muham/development/kbv2/pyproject.toml
- Build System: uv_build
- Python: >=3.12
- Key Dependencies:
  - FastAPI, uvicorn (Web framework)
  - SQLAlchemy, asyncpg, pgvector (Database)
  - pydantic, pydantic-settings (Data validation)
  - unstructured (Document parsing)
  - igraph, leidenalg (Graph clustering)
  - sentence-transformers, rank-bm25 (Search)
  - websockets (Real-time communication)
  - tenacity (Resilience patterns)
/home/muham/development/kbv2/.env.example
# Database Configuration
DATABASE_URL=postgresql://agentzero@localhost:5432/knowledge_base
# LLM Gateway Configuration
LLM_GATEWAY_URL=http://localhost:8087/v1/
LLM_MODEL=gemini-2.5-flash-lite
# Ingestion Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_DENSITY_THRESHOLD=0.8
MIN_DENSITY_THRESHOLD=0.3
# Clustering Configuration
LEIDEN_RESOLUTION_MACRO=0.8
LEIDEN_RESOLUTION_MICRO=1.2
# Resolution Configuration
RESOLUTION_CONFIDENCE_THRESHOLD=0.7
RESOLUTION_SIMILARITY_THRESHOLD=0.85
# HNSW Index Configuration
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=100
/home/muham/development/kbv2/alembic.ini
- Database migration configuration
- Uses async PostgreSQL driver
---
4. DATABASE SCHEMA (SQLAlchemy Models)
Core Tables:
| Table | Purpose | Key Fields |
|-------|---------|------------|
| documents | Document metadata | id, name, source_uri, status, domain |
| chunks | Document chunks | id, document_id, text, embedding (1024-dim) |
| entities | Knowledge graph nodes | id, name, entity_type, embedding, properties |
| edges | Relationships | id, source_id, target_id, edge_type, confidence |
| communities | Leiden clusters | id, name, level, parent_id, summary |
| chunk_entities | Junction table | chunk_id, entity_id, grounding_quote |
| review_queue | Human review items | item_type, entity_id, confidence_score, status |
Edge Types (30+):
- Hierarchical: PART_OF, SUBCLASS_OF, INSTANCE_OF, CONTAINS
- Causal: CAUSES, INFLUENCES, PREVENTS, ENABLES
- Temporal: PRECEDES, FOLLOWS, DURING, CONCURRENT_WITH
- Social: WORKS_FOR, KNOWS, COLLEAGUE_OF
- Ownership: OWNS, MANAGES, OPERATES
- Activity: PARTICIPATES_IN, PERFORMS, TARGETS
- Document: MENTIONS, REFERENCES, DISCUSSES
- Long-tail: UNKNOWN, NOTA (none-of-the-above), HYPOTHETICAL
---
5. API ENDPOINTS
Query API (/api/v1/query)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /translate | POST | Natural language to SQL translation |
| /execute | POST | Execute NL query |
| /schema | GET | Database schema info |
| /federated | POST | Multi-domain query routing |
| /hybrid-search-v2 | POST | BM25 + Vector search |
| /reranked-search | POST | Hybrid + cross-encoder reranking |
Document API (/api/v1/documents)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /{id} | GET | Document metadata |
| /{id}/content | GET | Full document text |
| /{id}/spans | GET | Entity text spans (W3C annotations) |
| /{id}/entities | GET | Extracted entities |
| /:search | POST | Hybrid search across documents |
| /{id}/annotations | POST | Create W3C Web Annotations |
Graph API (/api/v1/graphs)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /{id}:summary | GET | Community-level overview |
| /{id}/nodes/{node}:neighborhood | GET | Entity neighborhood expansion |
| /{id}:findPath | POST | Path finding (shortest, most confident) |
| /{id}:export | GET | Export to Graphology/Sigma.js format |
Schema API (/api/v1/schemas)
| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET/POST | List/Register domain schemas |
| /{domain} | GET/DELETE | Get/Delete schema |
| /{domain}/entity-types | GET | List entity types |
Review API (/api/v1/review)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /pending | GET | Pending review items |
| /{id} | GET | Review item details |
| /{id}/approve | POST | Approve resolution |
| /{id}/reject | POST | Reject with corrections |
MCP WebSocket (/ws)
JSON-RPC over WebSocket for external tool integration:
- kbv2/ingest_document
- kbv2/query_text_to_sql
- kbv2/search_entities
- kbv2/search_chunks
- kbv2/get_document_status
---
6. DATA INGESTION PIPELINE (15 Stages)
1. Document Creation - Initialize record
2. Auto Domain Detection - Keyword screening + LLM analysis
3. Smart Partitioning - 1536 tokens, 25% overlap
4. Multi-Modal Extraction - Tables, images, figures
5. Guided Extraction - Domain-specific prompts
6. Multi-Agent Extraction - Manager, Perception, Enhancement, Evaluation agents
7. Embedding Generation - Batch processing with bge-m3 (1024-dim)
8. Global Entity Resolution - Cross-document deduplication
9. Entity Clustering - Hierarchical Leiden (macro → meso → micro → nano)
10. Enhanced Community Summaries - Map-reduce summarization
11. Adaptive Type Discovery - Schema induction
12. Schema Validation - Type checking
13. Hybrid Search Indexing - BM25 + Vector
14. Reranking Pipeline - Cross-encoder scoring
15. Intelligence Reports - AI-generated insights
Key Services:
- GleaningService - 2-pass adaptive extraction (Discovery + Gleaning)
- MultiAgentExtractor - GraphMaster-style coordinated agents
- ResolutionAgent - Verbatim-grounded entity deduplication
- ClusteringService - Leiden algorithm with igraph
- SynthesisAgent - Map-reduce community summarization
---
7. DOMAIN CONFIGURATIONS
Pre-configured Domains (in /home/muham/development/kbv2/src/knowledge_base/main.py):
| Domain | Parent | Key Entity Types |
|--------|--------|------------------|
| GENERAL | - | NamedEntity |
| TECHNOLOGY | GENERAL | Software, API, Framework |
| FINANCIAL | GENERAL | Company, FinancialInstrument |
| MEDICAL | GENERAL | Drug, Procedure |
| LEGAL | GENERAL | Contract, Court |
| HEALTHCARE | GENERAL | HealthcareProvider, InsurancePlan |
| ACADEMIC | GENERAL | Publication, ResearchField |
| SCIENTIFIC | GENERAL | Theory, Experiment |
Extended Domain Ontologies (in /home/muham/development/kbv2/src/knowledge_base/domain/ontology_snippets.py):
- CRYPTO_TRADING - With 200+ trading keywords, technical indicators, chart patterns
Domain Detection:
- Keyword screening with regex matching
- LLM analysis for ambiguous cases
- Hybrid confidence calibration
- Multi-domain detection support
---
8. KEY ARCHITECTURAL FEATURES
Multi-Agent Extraction System
- ManagerAgent: Orchestrates workflow
- PerceptionAgent: Boundary-aware entity extraction (BANER-style)
- EnhancementAgent: Refines entities using KG context
- EvaluationAgent: LLM-as-Judge quality validation
Hybrid Search Architecture
- BM25 Index: Keyword-based retrieval
- Vector Store: Semantic similarity (cosine)
- Cross-Encoder: Reranking for result quality
- RRF Fusion: Reciprocal Rank Fusion for combining scores
Resilience Patterns
- Circuit breaker for LLM calls
- Automatic model rotation
- Retry middleware with exponential backoff
- Graceful degradation
Observability
- Logfire integration
- Request ID tracking
- Performance metrics
- Extraction logging with WebSocket broadcast
---
9. FRONTEND
Status: The frontend has been removed from this repository. The system is now backend-only with:
- Interactive API documentation at /api/v1/docs (Swagger UI)
- Direct API calls
- WebSocket interface for real-time operations
- Static dashboard support (if files exist in static/ directory)
---
10. LLM INTEGRATION
Configuration:
- Endpoint: http://localhost:8087/v1/ (OpenAI-compatible)
- Model: gemini-2.5-flash-lite (configurable)
- Temperature: 0.7 (configurable per operation)
- Max tokens: 4096
Embedding:
- Endpoint: http://localhost:11434/ (Ollama)
- Model: bge-m3 (1024 dimensions)
- Supports batch processing
---
11. TESTING FRAMEWORK
- Unit Tests: pytest with async support
- Integration Tests: Full pipeline testing
- E2E Tests: End-to-end feature verification
- Test Data: Sample documents in /home/muham/development/kbv2/tests/test_data/
---
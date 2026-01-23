# KBV2 Folder Structure

```
kbv2/
│
├── .env                          # Environment variables (not in git)
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── .python-version               # Python version specification
├── .ruff_cache/                  # Ruff formatter cache
│   └── 0.14.13/
├── .mypy_cache/                  # MyPy type checker cache
│   └── 3.12/
│
├── DESIGN_DOC.md                 # High-level system design
├── ENTITY_PROCESSING_PIPELINE.md # Entity processing specification
├── IMPLEMENTATION_SUMMARY.md     # Implementation changes summary
├── README.md                     # Project overview and setup
├── SETUP.md                      # Setup instructions
├── SPEC.md                       # Architecture specification
├── VALIDATION_SUMMARY.md         # Validation results
├── implementation_plan.md        # Implementation roadmap
│
├── pyproject.toml                # Project configuration (uv)
├── requirements.txt              # Python dependencies
├── setup.py                      # Setup script
├── uv.lock                       # Dependency lock file
│
├── scripts/                      # Utility scripts
│   └── setup_db.py              # Database initialization script
│
├── src/                          # Source code
│   └── knowledge_base/          # Main package
│       │
│       ├── __init__.py          # Package initialization
│       │
│       ├── orchestrator.py      # Main ReAct loop orchestrator
│       │                        # └─ IngestionOrchestrator class
│       │                        #    ├─ process_document()
│       │                        #    ├─ _partition_document()
│       │                        #    ├─ _extract_knowledge()
│       │                        #    ├─ _embed_content()
│       │                        #    ├─ _resolve_entities()
│       │                        #    ├─ _cluster_entities()
│       │                        #    └─ _generate_reports()
│       │
│       ├── query_api.py         # FastAPI query endpoints
│       │                        # └─ /api/v1/query/*
│       │
│       ├── review_api.py        # FastAPI review endpoints
│       │                        # └─ /api/v1/review/*
│       │
│       ├── review_service.py    # Review queue management
│       │                        # └─ ReviewService class
│       │
│       ├── text_to_sql_agent.py # Natural language to SQL
│       │                        # └─ TextToSQLAgent class
│       │                        #    ├─ translate()
│       │                        #    ├─ _check_sql_security()
│       │                        #    └─ _is_safe_identifier()
│       │
│       ├── mcp_server.py        # Model Context Protocol server
│       │                        # └─ WebSocket: /ws
│       │                        # └─ Methods: kbv2/ingest_document, etc.
│       │
│       ├── observability.py     # SRE-Lite observability
│       │                        # └─ Observability class (singleton)
│       │                        #    ├─ trace_operation()
│       │                        #    ├─ trace_context()
│       │                        #    ├─ log_metric()
│       │                        #    └─ track_tokens()
│       │
│       │
│       ├── common/              # Shared utilities
│       │   ├── __init__.py
│       │   │
│       │   ├── gateway.py       # LLM API client
│       │   │                    # └─ GatewayClient class
│       │   │                    #    ├─ chat_completion()
│       │   │                    #    └─ generate_text()
│       │   │
│       │   ├── temporal_utils.py # Temporal normalization
│       │   │                    # └─ TemporalNormalizer class
│       │   │                    #    ├─ normalize_relative_date()
│       │   │                    #    ├─ classify_claim_temporal_type()
│       │   │                    #    ├─ extract_temporal_info()
│       │   │                    #    └─ check_invalidated()
│       │   │
│       │   └── resilient_gateway/ # Enhanced gateway with resilience
│       │       ├── __init__.py
│       │       ├── gateway.py   # ResilientGatewayClient class
│       │       │                #    ├─ Circuit Breaker (CLOSED/OPEN/HALF_OPEN)
│       │       │                #    ├─ Exponential Backoff
│       │       │                #    ├─ Model Switching
│       │       │                #    └─ Metrics Collection
│       │       ├── compatibility.py
│       │       └── example.py
│       │
│       │
│       ├── ingestion/           # Document ingestion pipeline
│       │   ├── __init__.py
│       │   └── v1/              # Ingestion v1 implementation
│       │       ├── __init__.py
│       │       │
│       │       ├── partitioning_service.py
│       │       │                # └─ PartitioningService class
│       │       │                #    ├─ partition_file()
│       │       │                #    ├─ chunk_elements()
│       │       │                #    └─ partition_and_chunk()
│       │       │
│       │       ├── gleaning_service.py
│       │       │                # └─ GleaningService class
│       │       │                #    ├─ extract() [2-pass adaptive]
│       │       │                #    ├─ should_continue_extraction()
│       │       │                #    ├─ _extract_pass()
│       │       │                #    ├─ _merge_results()
│       │       │                #    └─ _analyze_relation_distribution()
│       │       │
│       │       └── embedding_client.py
│       │                        # └─ EmbeddingClient class
│       │                        #    ├─ embed_text()
│       │                        #    └─ embed_batch()
│       │
│       │
│       ├── persistence/         # Database layer
│       │   ├── __init__.py
│       │   └── v1/              # Persistence v1 implementation
│       │       ├── __init__.py
│       │       │
│       │       ├── schema.py    # SQLAlchemy models
│       │       │                # └─ Models:
│       │       │                #    ├─ Document
│       │       │                #    ├─ Chunk
│       │       │                #    ├─ Entity
│       │       │                #    ├─ Edge
│       │       │                #    ├─ Community
│       │       │                #    ├─ ChunkEntity (junction)
│       │       │                #    ├─ ReviewQueue
│       │       │                #    ├─ DocumentStatus (enum)
│       │       │                #    ├─ EdgeType (enum, 30+ types)
│       │       │                #    ├─ ReviewStatus (enum)
│       │       │                #    └─ Vector (custom type)
│       │       │
│       │       └── vector_store.py
│       │                        # └─ VectorStore class
│       │                        #    ├─ initialize()
│       │                        #    ├─ search_similar_entities()
│       │                        #    ├─ search_similar_chunks()
│       │                        #    ├─ update_entity_embedding()
│       │                        #    └─ update_chunk_embedding()
│       │
│       │
│       └── intelligence/        # Analysis layer
│           ├── __init__.py
│           └── v1/              # Intelligence v1 implementation
│               ├── __init__.py
│               │
│               ├── resolution_agent.py
│               │                # └─ ResolutionAgent class
│               │                #    ├─ resolve_entity()
│               │                #    ├─ _llm_resolve()
│               │                #    ├─ batch_resolve_entities()
│               │                #    └─ _parse_resolution_response()
│               │
│               ├── clustering_service.py
│               │                # └─ ClusteringService class
│               │                #    ├─ cluster_entities()
│               │                #    ├─ build_hierarchy()
│               │                #    └─ incremental_update()
│               │
│               └── synthesis_agent.py
│                               # └─ SynthesisAgent class
│                               #    ├─ generate_micro_report()
│                               #    ├─ generate_macro_report()
│                               #    └─ generate_intelligence_report()
│
│
├── test_data/                   # Test data files
│   └── (sample documents for testing)
│
├── tests/                       # Test suite
│   ├── test_resilient_gateway.py
│   └── test_comprehensive_real_world.py
│
├── final_verification.py        # Verification script
└── ~/                          # Temporary directory
    └── .arxiv-mcp-server/      # ArXiv MCP server cache
```

## Directory Structure Overview

### Root Level
- **Configuration**: `.env`, `.env.example`, `pyproject.toml`, `requirements.txt`, `uv.lock`
- **Documentation**: `README.md`, `DESIGN_DOC.md`, `SPEC.md`, etc.
- **Scripts**: `scripts/` - Database setup and utility scripts
- **Source**: `src/` - All Python source code
- **Tests**: `tests/` - Test suite
- **Test Data**: `test_data/` - Sample documents for testing

### Source Code Structure (`src/knowledge_base/`)

#### Core Modules (Root Level)
- **orchestrator.py**: Main ReAct loop orchestrator
- **query_api.py**: FastAPI query endpoints
- **review_api.py**: FastAPI review endpoints
- **review_service.py**: Review queue management
- **text_to_sql_agent.py**: Natural language to SQL translation
- **mcp_server.py**: Model Context Protocol server
- **observability.py**: SRE-Lite observability (singleton)

#### Common Module (`common/`)
Shared utilities and infrastructure
- **gateway.py**: Basic LLM API client
- **temporal_utils.py**: Temporal normalization and ISO-8601 conversion
- **resilient_gateway/**: Enhanced gateway with circuit breaker, retry, model switching

#### Ingestion Module (`ingestion/v1/`)
Document processing pipeline
- **partitioning_service.py**: Document parsing and chunking
- **gleaning_service.py**: 2-pass adaptive knowledge extraction
- **embedding_client.py**: Vector embedding generation

#### Persistence Module (`persistence/v1/`)
Database layer
- **schema.py**: SQLAlchemy models and database schema
- **vector_store.py**: PostgreSQL + pgvector wrapper and vector operations

#### Intelligence Module (`intelligence/v1/`)
Analysis and reasoning layer
- **resolution_agent.py**: Verbatim-grounded entity deduplication
- **clustering_service.py**: Hierarchical Leiden community detection
- **synthesis_agent.py**: Map-reduce summarization of communities

## File Naming Conventions

- **Modules**: `snake_case.py` (e.g., `partitioning_service.py`)
- **Classes**: `PascalCase` (e.g., `PartitioningService`)
- **Functions/Methods**: `snake_case` (e.g., `partition_file`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_PASSES`)
- **Private members**: `_leading_underscore` (e.g., `_gateway`)

## Version Convention

The system uses versioned subdirectories (e.g., `v1/`) to allow for future parallel implementations and backward compatibility.
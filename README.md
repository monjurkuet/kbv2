# Agentic Knowledge Ingestion & Management System

A high-fidelity information extraction engine that transforms unstructured data into a structured, temporally-aware knowledge graph.

## Features

- **High-Resolution Adaptive Gleaning**: 2-pass density-aware extraction strategy
- **Verbatim-Grounded Entity Resolution**: Hybrid matching with mandatory grounding quotes
- **Hierarchical Leiden Clustering**: Macro and Micro community detection
- **Map-Reduce Recursive Summarization**: Intelligence reports with edge fidelity
- **Temporal Information Extraction (TIE)**: ISO-8601 normalized temporal claims

## Architecture

```
src/knowledge_base/
├── common/              # Shared utilities
│   ├── gateway.py       # LLM client
│   └── temporal_utils.py # TIE logic
├── persistence/         # Database layer
│   └── v1/
│       ├── schema.py    # SQLAlchemy models
│       └── vector_store.py # pgvector/HNSW
├── ingestion/           # Ingestion pipeline
│   └── v1/
│       ├── partitioning_service.py
│       ├── gleaning_service.py
│       └── embedding_client.py
├── intelligence/        # Analysis layer
│   └── v1/
│       ├── resolution_agent.py
│       ├── clustering_service.py
│       └── synthesis_agent.py
└── orchestrator.py      # ReAct loop
```

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Setup PostgreSQL**:
   ```bash
   # Create database
   createdb knowledge_base

   # Run setup script
   python scripts/setup_db.py
   ```

4. **Run**:
   ```bash
   uv run knowledge-base
   ```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `LLM_GATEWAY_URL`: Local LLM API endpoint
- `GOOGLE_API_KEY`: Google Embeddings API key

## Development

- **Lint**: `uv run ruff check`
- **Format**: `uv run ruff format`
- **Type check**: `uv run mypy src/`
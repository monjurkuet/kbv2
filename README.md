# Agentic Knowledge Ingestion & Management System

A high-fidelity information extraction engine that transforms unstructured data into a structured, temporally-aware knowledge graph.

## Key Features

- **Adaptive Extraction:** 2-pass density-aware extraction with document complexity analysis
- **Verbatim-Grounded Entity Resolution:** Dedupes similar entities with mandatory citations
- **Hierarchical Leiden Clustering:** Macro and micro community detection
- **Map-Reduce Recursive Summarization:** Intelligence reports with edge fidelity
- **Temporal Information Extraction:** ISO-8601 normalized temporal claims
- **Natural Language Query Interface:** Translate queries to SQL
- **Domain Tagging & Filtering:** 16 domains (10 crypto + 6 legacy)
- **Human Review Queue:** Flag low-confidence resolutions for manual review
- **LLM-Based Entity Typing:** Advanced entity classification
- **Multi-Domain Knowledge Management:** Unified data model across domains
- **Hybrid Search:** BM25 + Vector (1024 dims) with cross-encoder reranking
- **Auto Domain Detection:** Keyword screening + LLM analysis
- **Self-Improvement:** Experience Bank, Prompt Evolution, Ontology Validation

## Quick Start

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run database migrations
alembic upgrade head

# Start Ollama for embeddings
ollama pull bge-m3
ollama serve

# Ingest a document
./ingest_cli.py /path/to/document.md --domain BITCOIN

# Or with auto-detection
./ingest_cli.py /path/to/document.md
```

## Documentation

- **[Quick Start](QUICK_START.md)** - Get started in 5 minutes
- **[Documentation](docs/README.md)** - Complete documentation index
- **[Ingestion Guide](docs/guides/ingestion.md)** - Document ingestion methods
- **[Deployment Guide](docs/guides/deployment.md)** - Production deployment
- **[Self-Improvement Guide](docs/guides/self_improvement.md)** - Experience Bank, Prompt Evolution
- **[Architecture Overview](docs/architecture/overview.md)** - System architecture

## Development

```bash
# Lint
uv run ruff check

# Format
uv run ruff format

# Type check
uv run mypy src/

# Tests
uv run pytest tests/
```

## Technology Stack

- **Backend:** FastAPI (Python 3.12)
- **Database:** PostgreSQL 16+ with pgvector
- **Vector Search:** pgvector (IVFFlat indexes, 1024-dim vectors)
- **Embeddings:** Ollama (bge-m3)
- **LLM Integration:** OpenAI SDK (AsyncOpenAI) with random model rotation
- **Clustering:** igraph + leidenalg

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://agentzero@localhost/knowledge_base

# LLM Configuration (OpenAI-compatible API)
LLM_API_BASE=http://localhost:8087/v1
LLM_API_KEY=sk-dummy

# Embedding Configuration (Ollama)
EMBEDDING_API_BASE=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSIONS=1024

# Self-Improvement Features
ENABLE_EXPERIENCE_BANK=true
ENABLE_PROMPT_EVOLUTION=true
ENABLE_ONTOLOGY_VALIDATION=true
```

## License

MIT

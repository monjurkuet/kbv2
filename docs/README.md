# KBV2 Documentation

This directory contains documentation for the KBV2 (Knowledge Base Version 2) system.

## Quick Start

1. [Quick Start](../../QUICK_START.md) - Get started with ingestion in 5 minutes
2. [Ingestion Guide](guides/ingestion.md) - Complete ingestion documentation
3. [Deployment Guide](guides/deployment.md) - Production deployment instructions

---

## Documentation Structure

### Guides

Comprehensive guides for using KBV2 features:

- [Ingestion Guide](guides/ingestion.md) - Document ingestion methods and pipeline
- [Deployment Guide](guides/deployment.md) - Production setup and configuration
- [Self-Improvement Guide](guides/self_improvement.md) - Experience Bank, Prompt Evolution, Ontology Validation
- [Available Features](guides/available_features.md) - Features not in main pipeline

### Architecture

System architecture and design documentation:

- [System Overview](architecture/overview.md) - High-level architecture and data flow
- [Data Flow](architecture/data_flow.md) - Detailed data transformation pipeline

### API

API documentation and reference:

- [API Endpoints](api/endpoints.md) - Complete API reference for all endpoints

### Database

Database schema and relationships:

- [Schema](database/schema.md) - Entity-Relationship diagrams and table definitions

### Operations

Operations and maintenance documentation:

- [Setup Guide](operations/setup.md) - Installation and initial setup
- [Runbook](operations/runbook.md) - Operations, monitoring, and troubleshooting

### Development

Development documentation:

- [Folder Structure](development/folder_structure.md) - Directory structure and file organization

### Archive

Historical documentation and reports:

- [Migration Reports](archive/migration_reports/) - OpenAI client migration, gateway consolidation
- [Implementation Plans](archive/implementation_plans/) - Historical implementation plans

---

## System Architecture

KBV2 transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction techniques.

### Key Components

1. **Ingestion Pipeline** - Document parsing, chunking, domain detection
2. **Entity Extraction** - Multi-agent, gleaning, guided extraction
3. **Hybrid Search** - BM25 index, vector store (1024 dims), reranking
4. **Graph Management** - Hierarchical clustering, community summaries
5. **Query Engine** - Natural language to SQL, hybrid search, reranking
6. **Self-Improvement** - Experience Bank, Prompt Evolution, Ontology Validation

### Supported Domains

**Crypto Domains (Primary):**
- BITCOIN, DEFI, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION
- DIGITAL_ASSETS, BLOCKCHAIN_INFRA, CRYPTO_MARKETS, CRYPTO_AI, TOKENIZATION

**Legacy Domains:**
- TECHNOLOGY, FINANCIAL, MEDICAL, LEGAL, SCIENTIFIC, GENERAL

---

## Getting Started

### Prerequisites

- PostgreSQL 16+ with pgvector extension
- Python 3.12+
- uv package manager
- Ollama (for embeddings)
- OpenAI-compatible LLM API

### Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run database migrations
alembic upgrade head

# Start Ollama
ollama pull bge-m3
ollama serve

# Start server
uv run python -m knowledge_base.production
```

### Quick Ingestion

```bash
# Direct ingestion (no server needed)
./ingest_cli.py /path/to/document.md --domain BITCOIN

# Or with auto-detection
./ingest_cli.py /path/to/document.md
```

---

## Key Features

### Adaptive Extraction

- **Document Complexity Analysis** - LLM analyzes complexity and recommends strategy
- **Random Model Selection** - Each LLM call uses a different model
- **Error Rotation** - Automatic retry with different models on any error

### Self-Improvement

- **Experience Bank** - Stores high-quality extractions (quality ≥ 0.75) for few-shot learning
- **Prompt Evolution** - Automated prompt optimization for 5 crypto domains
- **Ontology Validation** - 15+ crypto-specific validation rules
- **Domain Detection Feedback** - Learning from classification accuracy

### Search Capabilities

- **Hybrid Search** - BM25 + Vector with weighted fusion
- **Reranking** - Cross-encoder for improved results
- **RRF Fusion** - Reciprocal Rank Fusion for multi-query

---

## Technology Stack

- **Backend:** FastAPI (Python 3.12)
- **Database:** PostgreSQL with async SQLAlchemy
- **Vector Search:** pgvector (1024-dim vectors, bge-m3)
- **LLM Integration:** OpenAI SDK (AsyncOpenAI) with random model rotation
- **Embeddings:** Ollama (bge-m3, 1024 dimensions)
- **Clustering:** igraph + leidenalg
- **Testing:** Pytest with async support

---

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://agentzero@localhost/knowledge_base

# LLM Configuration
LLM_API_BASE=http://localhost:8087/v1
LLM_API_KEY=sk-dummy

# Embedding Configuration
EMBEDDING_API_BASE=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSIONS=1024

# Self-Improvement Features
ENABLE_EXPERIENCE_BANK=true
ENABLE_PROMPT_EVOLUTION=true
ENABLE_ONTOLOGY_VALIDATION=true
```

---

## Development

### Code Quality

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run mypy src/

# Tests
uv run pytest tests/
```

### Project Structure

```
kbv2/
├── src/knowledge_base/          # Source code
│   ├── clients/                  # LLM and WebSocket clients
│   ├── intelligence/v1/         # AI services
│   ├── ingestion/v1/            # Document processing
│   ├── persistence/v1/          # Database layer
│   └── orchestration/            # Pipeline orchestration
├── docs/                        # Documentation
├── tests/                       # Test suite
└── alembic/                     # Database migrations
```

---

## Support

- **Issues:** Report bugs and feature requests at the project repository
- **Questions:** Check existing documentation and issues first
- **Contributions:** Follow Google Python Style Guide and existing patterns

---

## Related Documentation

- [README](../../README.md) - Project overview
- [QUICK_START](../../QUICK_START.md) - Quick start guide
- [AGENTS.md](../../AGENTS.md) - Agent instructions

---

## License

MIT

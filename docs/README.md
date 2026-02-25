# KBV2 Documentation

This directory contains documentation for the KBV2 (Knowledge Base Version 2) system.

## Quick Start

1. [Quick Start](../QUICK_START.md) - Get started in 5 minutes
2. [Ingestion Guide](guides/ingestion.md) - Document ingestion methods
3. [Deployment Guide](guides/deployment.md) - Production deployment
4. [API Endpoints](api/endpoints.md) - API reference

---

## Documentation Structure

### Guides

- [Ingestion Guide](guides/ingestion.md) - Document ingestion methods and pipeline
- [Deployment Guide](guides/deployment.md) - Production setup and configuration
- [Available Features](guides/available_features.md) - Feature overview

### Architecture

- [System Overview](architecture/overview.md) - High-level architecture
- [Data Flow](architecture/data_flow.md) - Data transformation pipeline

### API

- [API Endpoints](api/endpoints.md) - Complete API reference

### Operations

- [Environment Configuration](operations/environment.md) - Configuration reference
- [Runbook](operations/runbook.md) - Operations and troubleshooting

---

## System Architecture

KBV2 transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction techniques.

### Storage Architecture (Portable)

| Component | Location | Purpose |
|-----------|----------|---------|
| SQLite | `data/knowledge.db` | Documents + FTS5 full-text search |
| ChromaDB | `data/chroma/` | Vector embeddings (HNSW, 1024 dims) |
| Kuzu | `data/knowledge_graph.kuzu` | Knowledge graph (Cypher queries) |

**No external database servers required.**

### Key Components

1. **Ingestion Pipeline** - Document parsing, chunking, domain detection
2. **Entity Extraction** - Multi-agent extraction with domain ontologies
3. **Hybrid Search** - BM25 + Vector with reranking
4. **Knowledge Graph** - Entities, relationships, community detection
5. **RAG Pipeline** - 5 strategies for retrieval-augmented generation

### Supported Domains

**Crypto Domains (Primary):**
- BITCOIN, DEFI, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION
- DIGITAL_ASSETS, BLOCKCHAIN_INFRA, CRYPTO_MARKETS, CRYPTO_AI, TOKENIZATION

**Legacy Domains:**
- TECHNOLOGY, FINANCIAL, MEDICAL, LEGAL, SCIENTIFIC, GENERAL

---

## Getting Started

### Prerequisites

- Python 3.12+
- uv package manager
- Ollama (for embeddings)
- OpenAI-compatible LLM API

**No database setup required** - portable storage is embedded.

### Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start Ollama for embeddings
ollama pull bge-m3
ollama serve

# Start server
./start.sh
```

### Quick Ingestion

```bash
# Via CLI
uv run knowledge-base ingest /path/to/document.md --domain BITCOIN

# Via API
curl -X POST http://localhost:8088/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.md", "domain": "BITCOIN"}'
```

---

## Technology Stack

- **Backend:** FastAPI (Python 3.12)
- **Documents + FTS:** SQLite + FTS5
- **Vector Store:** ChromaDB (HNSW, 1024 dims)
- **Graph Database:** Kuzu (embedded, Cypher)
- **LLM Integration:** OpenAI SDK (AsyncOpenAI)
- **Embeddings:** Ollama (bge-m3, 1024 dimensions)
- **Clustering:** igraph + leidenalg
- **Testing:** pytest with async support

---

## Development

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

---

## Related Documentation

- [README](../README.md) - Project overview
- [QUICK_START](../QUICK_START.md) - Quick start guide
- [CLAUDE.md](../CLAUDE.md) - Development guidelines

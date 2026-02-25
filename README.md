# KBV2 - Portable Knowledge Base System

A high-fidelity information extraction engine that transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction.

## Key Features

- **Portable Storage:** SQLite + ChromaDB + Kuzu - no external database servers required
- **Hybrid Search:** BM25 + Vector (1024 dims) with cross-encoder reranking
- **Knowledge Graph:** Entity/relationship extraction with Cypher queries
- **Adaptive Extraction:** Multi-agent extraction pipeline with domain detection
- **RAG Strategies:** 5 modes (STANDARD, HYBRID, DUAL_LEVEL, GRAPH_ENHANCED, CORRECTIVE)
- **Temporal Information:** ISO-8601 normalized temporal claims
- **Domain Detection:** Auto keyword screening + LLM analysis
- **Community Detection:** Leiden clustering with hierarchical summarization

## Quick Start

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your API keys (secrets only)
# Edit config.yaml for all other settings

# Start Ollama for embeddings
ollama pull bge-m3
ollama serve

# Ingest a document
uv run knowledge-base ingest /path/to/document.md --domain BITCOIN

# Or with auto-detection
uv run knowledge-base ingest /path/to/document.md

# Start API server (one-stop script)
./start.sh

# Or with manual command
uv run uvicorn knowledge_base.main:app --reload --port 8088
```

## Storage Architecture

| Component | Technology | Purpose |
|-----------|------------|---------|
| SQLite + FTS5 | `data/kbv2.db` | Documents + Full-text search |
| ChromaDB | `data/chroma/` | Vector similarity (HNSW) |
| Kuzu | `data/kuzu/` | Knowledge graph (Cypher) |

**No external database servers required** - all data stored in portable files.

## Documentation

- **[Quick Start](QUICK_START.md)** - Get started in 5 minutes
- **[Documentation](docs/README.md)** - Complete documentation index
- **[Ingestion Guide](docs/guides/ingestion.md)** - Document ingestion methods
- **[Architecture Overview](docs/architecture/overview.md)** - System architecture

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

## Technology Stack

- **Language:** Python 3.12+
- **Backend:** FastAPI with async support
- **Documents + FTS:** SQLite + FTS5 (BM25)
- **Vector Store:** ChromaDB (HNSW, 1024 dims)
- **Graph Database:** Kuzu (embedded, Cypher)
- **LLM Client:** AsyncOpenAI SDK (OpenAI-compatible)
- **Embeddings:** Ollama (bge-m3)
- **CLI:** Typer with Rich formatting

## Configuration

Configuration is split between:

- **`config.yaml`** - All non-secret settings (LLM model, chunk sizes, etc.)
- **`.env`** - Secrets only (API keys)

See `config.yaml` for available options.

## License

MIT

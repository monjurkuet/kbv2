# CLAUDE.md - Knowledge Base V2 Development Guidelines

## Project Overview

KBV2 is a **Portable Knowledge Base System** - a high-fidelity information extraction engine that transforms unstructured documents into structured, temporally-aware knowledge graphs using adaptive AI extraction.

**Version:** 0.2.0 | **Python:** 3.12+ | **Package Manager:** uv

---

## Must Follow Rules

- **No CI/CD required** - keep it simple
- **No Docker required** - bare metal deployment
- **Always use `uv`** for Python package management
- **Always use `localhost:8087/v1/`** for LLM calls (OpenAI-compatible API)
- **Always use `localhost:11434/`** for embeddings (bge-m3, 1024 dimensions)
- **Clean slate architecture** - no backwards compatibility needed, no gradual migration
- **Portable databases only** - SQLite, ChromaDB, Kuzu (no external DB servers)
- **Self-host everything** - no cloud dependencies

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.12+ |
| **Backend** | FastAPI with async support |
| **CLI** | Typer with Rich formatting |
| **Documents + FTS** | SQLite + FTS5 (BM25) |
| **Vector Store** | ChromaDB (HNSW, 1024 dims) |
| **Graph Database** | Kuzu (embedded, Cypher queries) |
| **LLM Client** | AsyncOpenAI SDK (OpenAI-compatible) |
| **Embeddings** | Ollama bge-m3 |
| **Linting** | ruff (line length: 100) |
| **Typing** | mypy |
| **Testing** | pytest + pytest-asyncio |

---

## Project Structure

```
src/knowledge_base/
├── main.py              # FastAPI application entry
├── cli.py               # Typer CLI interface
├── clients/             # External service clients (LLM, WebSocket)
├── common/              # Shared utilities, exceptions, dependencies
├── config/              # Centralized constants
├── domain/              # Auto domain detection
├── extraction/          # Entity extraction pipeline
├── ingestion/           # Document processing, chunking
├── partitioning/        # Semantic chunking
├── rag/                 # RAG query pipeline (5 strategies)
├── reranking/           # Search reranking (RRF, cross-encoder)
├── storage/portable/    # SQLite, ChromaDB, Kuzu stores
├── summaries/           # Community summaries
├── types/               # Schema induction, type discovery
├── monitoring/          # Metrics collection
└── processing/          # Batch processing
```

---

## Coding Conventions

### Naming
- **Files/Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Code Style
- Line length: **100 characters** (E501 ignored)
- Ruff rules: E, F, I, N, W, UP, B, C4, SIM
- Type hints required on all public functions
- Google-style docstrings with Args/Returns

### Architecture Patterns

1. **Async-First**: All I/O operations are async
2. **Pydantic v2**: All data models use Pydantic
3. **Context Managers**: For resource lifecycle
4. **Singleton**: For shared clients (LLM, stores)
5. **Semaphore**: For concurrency control

```python
# Example: Async with semaphore
self._semaphore = asyncio.Semaphore(max_concurrent)

async def process(self, item):
    async with self._semaphore:
        return await self._process_internal(item)
```

---

## Storage Architecture

Portable, file-based storage in `data/` directory:

| Store | File | Purpose |
|-------|------|---------|
| SQLite | `data/kbv2.db` | Documents + FTS5 full-text search |
| ChromaDB | `data/chroma/` | Vector similarity (HNSW) |
| Kuzu | `data/kuzu/` | Knowledge graph (Cypher) |

**No external database servers required.**

---

## RAG Strategies

| Mode | Description |
|------|-------------|
| `STANDARD` | Basic vector + keyword search |
| `HYBRID` | BM25 + vector with RRF fusion |
| `DUAL_LEVEL` | LightRAG-style dual retrieval |
| `GRAPH_ENHANCED` | HippoRAG-style graph traversal |
| `CORRECTIVE` | CRAG-style corrective retrieval |

---

## Configuration

Configuration is split between two files:

**config.yaml** - All non-secret settings:
```yaml
storage:
  data_dir: data

llm:
  gateway_url: http://localhost:8087/v1/
  model: gemini-2.5-flash-lite
  temperature: 0.7
  max_tokens: 4096

embedding:
  api_base: http://localhost:11434
  model: bge-m3
  dimension: 1024

chunking:
  chunk_size: 512
  chunk_overlap: 50
```

**.env** - Secrets only:
```bash
LLM_API_KEY=
EMBEDDING_API_KEY=
```

---

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/stats` | GET | Storage statistics |
| `/documents` | POST | Create document |
| `/documents` | GET | List documents |
| `/search` | POST | Hybrid search |
| `/ingest` | POST | Ingest document |
| `/graph/entities` | GET | List entities |

---

## Development Commands

```bash
# Install dependencies
uv sync

# Start API server (one-stop script)
./start.sh

# Or with auto-reload for development
./start.sh --reload

# Or manual start
uv run uvicorn knowledge_base.main:app --reload --port 8088

# Run CLI
uv run knowledge-base --help

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run ruff format src/

# Type check
uv run mypy src/
```

---

## Supported Domains

**Crypto (Primary):** BITCOIN, DEFI, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION, DIGITAL_ASSETS, BLOCKCHAIN_INFRA, CRYPTO_MARKETS, CRYPTO_AI, TOKENIZATION

**Legacy:** TECHNOLOGY, FINANCIAL, MEDICAL, LEGAL, SCIENTIFIC, GENERAL

---

## Key Constants

```python
# src/knowledge_base/config/constants.py
LLM_GATEWAY_URL: Final[str] = "http://localhost:8087/v1"
WEBSOCKET_PORT: Final[int] = 8765
DEFAULT_CHUNK_SIZE: Final[int] = 512
SEMANTIC_CHUNK_SIZE: Final[int] = 1536
EMBEDDING_DIMENSION: Final[int] = 1024
DEFAULT_LLM_MODEL: Final[str] = "gemini-2.5-flash-lite"
```

---

## Exception Handling

Custom exception hierarchy in `src/knowledge_base/common/exceptions.py`:

```python
class KBV2BaseException(Exception):
    def __init__(self, message: str, error_code: str | None = None, context: dict | None = None):
        ...
```

---

## Notes

- `.env` files allowed for sensitive configuration
- CORS allows all origins (development mode)
- Follow Google Open Source coding standards
- Clean, simple, readable code preferred
- Stay current with 2026 LLM and agentic engineering patterns

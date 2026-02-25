# KBV2 Quick Start

Get started with KBV2 in 5 minutes.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Ollama (for embeddings)
- OpenAI-compatible LLM API

**No database setup required** - KBV2 uses portable SQLite/ChromaDB/Kuzu files.

## Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your API keys (secrets only)
# Edit config.yaml for all other settings
```

## Start Services

### Start Ollama (Embeddings)

```bash
# Pull bge-m3 model
ollama pull bge-m3

# Start Ollama server
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### Start LLM API

Ensure your OpenAI-compatible LLM API is running at `http://localhost:8087/v1`.

```bash
# Verify
curl http://localhost:8087/v1/health
```

## Ingest Documents

### Method 1: CLI (Recommended)

```bash
# Ingest with specified domain
uv run knowledge-base ingest /path/to/document.md --domain BITCOIN

# Ingest with auto-detection
uv run knowledge-base ingest /path/to/document.md
```

### Method 2: Python API

```python
import asyncio
from knowledge_base.common.dependencies import initialize_storage

async def ingest(file_path: str, domain: str = None):
    # Initialize storage
    stores = await initialize_storage()

    # Process document (simplified example)
    from knowledge_base.ingestion.document_processor import DocumentProcessor
    processor = DocumentProcessor()

    result = await processor.process(file_path, domain=domain)
    print(f"Ingested: {result}")

asyncio.run(ingest("/path/to/document.md", "BITCOIN"))
```

## Start API Server

```bash
# One-stop start script (recommended)
./start.sh

# Or with auto-reload for development
./start.sh --reload

# Or manual start
uv run uvicorn knowledge_base.main:app --reload --port 8088

# API will be available at http://localhost:8088
# Docs at http://localhost:8088/redoc
```

## Supported Domains

### Crypto Domains
- `BITCOIN` - Bitcoin protocol, mining, ETFs
- `DEFI` - DeFi protocols, TVL, yields
- `INSTITUTIONAL_CRYPTO` - ETFs, custody, treasuries
- `STABLECOINS` - USDC, USDT, backing
- `CRYPTO_REGULATION` - SEC, legislation, compliance

### Legacy Domains
- `TECHNOLOGY` - Software, APIs, ML/AI
- `FINANCIAL` - Financial reports, markets
- `GENERAL` - Mixed content

**Auto-Detection:** Omit `--domain` for automatic classification.

## What You Get

- **Entities** extracted with types and properties
- **Relationships** between entities
- **1024-dim vectors** for semantic search
- **Full-text search** via SQLite FTS5
- **Knowledge graph** with Cypher queries

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/stats` | GET | Storage statistics |
| `/documents` | POST | Create document |
| `/documents` | GET | List documents |
| `/search` | POST | Hybrid search |
| `/ingest` | POST | Ingest document from file |
| `/graph/entities` | GET | List entities |

## Storage Location

All data is stored in the `data/` directory:

```
data/
├── kbv2.db        # SQLite (documents + FTS)
├── chroma/        # ChromaDB (vectors)
└── kuzu/          # Kuzu (graph)
```

## Next Steps

- [Ingestion Guide](docs/guides/ingestion.md) - Complete ingestion documentation
- [Architecture Overview](docs/architecture/overview.md) - System architecture
- [API Endpoints](docs/api/endpoints.md) - Full API reference

## Troubleshooting

### Issue: Embedding generation failed

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama && ollama serve
```

### Issue: LLM API unavailable

```bash
# Check API health
curl http://localhost:8087/v1/health

# Check available models
curl http://localhost:8087/v1/models
```

### Issue: Import errors

```bash
# Verify installation
uv run python -c "from knowledge_base import __version__; print(__version__)"
```

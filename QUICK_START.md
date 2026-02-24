# KBV2 Quick Start

Get started with KBV2 in 5 minutes.

## Prerequisites

- PostgreSQL 16+ with pgvector extension
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Ollama (for embeddings)
- OpenAI-compatible LLM API

## Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your credentials

# Run database migrations
alembic upgrade head
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

### Method 1: Direct CLI (Recommended)

```bash
# Ingest with specified domain
./ingest_cli.py /path/to/document.md --domain BITCOIN

# Ingest with auto-detection
./ingest_cli.py /path/to/document.md

# With verbose output
./ingest_cli.py /path/to/document.md --domain DEFI --verbose
```

### Method 2: Simple Script

```bash
# Basic ingestion
./ingest.py /path/to/document.md BITCOIN
```

### Method 3: Python API

```python
import asyncio
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

async def ingest(file_path: str, domain: str):
    orchestrator = SelfImprovingOrchestrator()
    await orchestrator.initialize()
    
    document = await orchestrator.process_document(
        file_path=file_path,
        document_name="My Document",
        domain=domain,
    )
    
    print(f"Ingested: {document.id}")
    await orchestrator.close()

asyncio.run(ingest("/path/to/document.md", "BITCOIN"))
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

- **20-50 entities** extracted per document
- **30-80 relationships** extracted per document
- **1024-dim vectors** for semantic search
- **Self-improvement** via Experience Bank
- **Quality score** (target ≥ 0.75)

## Query Results

```bash
# Check document count
psql -d knowledge_base -c "SELECT COUNT(*) FROM documents;"

# Check entities
psql -d knowledge_base -c "SELECT COUNT(*) FROM entities;"

# Experience Bank stats
psql -d knowledge_base -c "SELECT COUNT(*) FROM extraction_experiences;"
```

## Next Steps

- [Ingestion Guide](docs/guides/ingestion.md) - Complete ingestion documentation
- [Deployment Guide](docs/guides/deployment.md) - Production deployment
- [Self-Improvement Guide](docs/guides/self_improvement.md) - Experience Bank features
- [Architecture Overview](docs/architecture/overview.md) - System architecture

## Troubleshooting

### Issue: Database connection failed

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify database exists
psql -U postgres -c "\l" | grep knowledge_base
```

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
```

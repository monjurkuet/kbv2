# KBV2 Ingestion Guide

## Overview

KBV2 provides multiple methods for ingesting documents into the knowledge base. This guide covers all ingestion options and best practices.

---

## Quick Start

### Method 1: CLI (Recommended)

```bash
# Ingest with specified domain
uv run knowledge-base ingest /path/to/document.md --domain BITCOIN

# Ingest with auto-detection
uv run knowledge-base ingest /path/to/document.md

# With verbose output
uv run knowledge-base ingest /path/to/document.md --domain DEFI --verbose
```

### Method 2: HTTP API

```bash
# Start server
./start.sh

# Ingest via API
curl -X POST http://localhost:8088/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.md", "domain": "BITCOIN"}'
```

### Method 3: Python API

```python
import asyncio
from pathlib import Path
from knowledge_base.ingestion import DocumentProcessor, SemanticChunker
from knowledge_base.storage.portable import SQLiteStore

async def ingest(file_path: str, domain: str = None):
    # Initialize processor and storage
    processor = DocumentProcessor()
    store = SQLiteStore()
    await store.initialize()

    # Process document
    processed = await processor.process(Path(file_path))

    # Create document record
    from knowledge_base.storage.portable.sqlite_store import Document
    doc = Document(
        name=processed.name,
        source_uri=str(processed.source_path),
        content=processed.content,
        domain=domain,
        status="processed",
    )
    doc_id = await store.add_document(doc)

    # Chunk and store
    chunker = SemanticChunker()
    chunks = chunker.chunk(processed.content, doc_id)
    await store.add_chunks_batch(chunks)

    print(f"Ingested: {doc_id}")
    return doc_id

asyncio.run(ingest("/path/to/document.md", "BITCOIN"))
```

---

## Supported Domains

### Crypto Domains (Primary)
- `BITCOIN` - Bitcoin protocol, mining, ETFs, Lightning Network
- `DEFI` - DeFi protocols, TVL, yields, liquidity pools
- `INSTITUTIONAL_CRYPTO` - ETFs, custody, treasuries, corporate adoption
- `STABLECOINS` - USDC, USDT, backing mechanisms
- `CRYPTO_REGULATION` - SEC, legislation, compliance
- `DIGITAL_ASSETS` - General cryptocurrencies
- `BLOCKCHAIN_INFRA` - L1/L2, consensus, scaling
- `CRYPTO_MARKETS` - Trading, on-chain metrics
- `CRYPTO_AI` - AI-blockchain convergence
- `TOKENIZATION` - RWA, asset tokenization

### Legacy Domains
- `TECHNOLOGY` - Software, APIs, ML/AI
- `FINANCIAL` - Financial reports, markets
- `MEDICAL` - Medical, clinical, research
- `LEGAL` - Law, contracts, litigation
- `SCIENTIFIC` - Academic research
- `GENERAL` - Mixed content

**Auto-Detection:** Omit `--domain` for automatic classification.

---

## Supported File Types

| Type | Extensions | Notes |
|------|------------|-------|
| Markdown | `.md`, `.markdown` | Best for structured content |
| Text | `.txt`, `.rst` | Plain text documents |
| PDF | `.pdf` | Requires vision model for OCR |
| Images | `.png`, `.jpg`, `.jpeg` | OCR via vision model |
| HTML | `.html`, `.htm` | Extracted to markdown |
| JSON | `.json` | Structured data |
| CSV | `.csv` | Tabular data |

---

## Ingestion Pipeline

### 1. Document Processing
- MIME type detection
- Content extraction (text, OCR for PDFs/images)
- Metadata extraction

### 2. Chunking
- Semantic chunking respecting document structure
- Configurable chunk size (default: 512 tokens)
- Overlap for context preservation

### 3. Storage
- SQLite: Document metadata + FTS5 full-text search
- ChromaDB: Vector embeddings (1024 dims)
- Kuzu: Entities and relationships

---

## Configuration

Edit `config.yaml`:

```yaml
chunking:
  chunk_size: 512
  chunk_overlap: 50
  semantic_chunk_size: 1536

embedding:
  api_base: http://localhost:11434
  model: bge-m3
  dimension: 1024
```

---

## Batch Ingestion

### CLI Batch

```bash
# Ingest multiple files
for file in documents/*.md; do
  uv run knowledge-base ingest "$file" --domain BITCOIN
done
```

### Python Batch

```python
import asyncio
from pathlib import Path
from knowledge_base.ingestion import DocumentProcessor

async def batch_ingest(directory: str, domain: str = None):
    processor = DocumentProcessor()
    results = await processor.process_directory(directory)
    for doc in results:
        print(f"Processed: {doc.name}")
    return results

asyncio.run(batch_ingest("./documents", "BITCOIN"))
```

---

## Verification

### Check Documents

```bash
# List documents via API
curl http://localhost:8088/documents

# Check storage stats
curl http://localhost:8088/stats
```

### Search Content

```bash
curl -X POST http://localhost:8088/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bitcoin ETF", "limit": 5}'
```

---

## Troubleshooting

### Embedding Failures

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Pull model if needed
ollama pull bge-m3
```

### Document Not Found

```bash
# Use absolute paths
uv run knowledge-base ingest /absolute/path/to/document.md

# Or relative to project root
cd /path/to/kbv2
uv run knowledge-base ingest ./documents/doc.md
```

### Large Files

For large PDFs or images, ensure:
- Vision model is running at `localhost:8087`
- Sufficient memory for OCR processing
- Consider splitting very large documents

---

## Related Documentation

- [Deployment Guide](deployment.md)
- [API Endpoints](../api/endpoints.md)
- [Quick Start](../QUICK_START.md)

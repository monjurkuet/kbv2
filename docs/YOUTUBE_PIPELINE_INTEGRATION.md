# KBV2 × YouTube Pipeline Integration Guide

**Date:** 2026-02-13  
**Status:** ✅ Production Ready

## Overview

This integration enables seamless flow of analyzed video content from the YouTube Content Pipeline into the KBV2 Knowledge Base. The external ingestion API accepts pre-processed documents with entities and relationships already extracted.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUTUBE CONTENT PIPELINE                         │
│                    (Port: 8001 - FastAPI)                           │
├─────────────────────────────────────────────────────────────────────┤
│  3-Agent Analysis:                                                  │
│  1. Transcript Intelligence (gemini-2.5-flash)                      │
│  2. Frame Intelligence (qwen3-vl-plus)                              │
│  3. Synthesis (gemini-2.5-flash)                                    │
└──────────────────────────────────────┬──────────────────────────────┘
                                       │ POST /api/v1/kb/export-to-kb
                                       │ (Transforms to KB format)
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    KBV2 KNOWLEDGE BASE                              │
│                    (Port: 8000 - FastAPI)                           │
├─────────────────────────────────────────────────────────────────────┤
│  POST /api/v1/documents/external                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Processing Pipeline:                                        │    │
│  │ 1. Create document record                                   │    │
│  │ 2. Partition into chunks (respects sections)                │    │
│  │ 3. Generate embeddings (bge-m3)                             │    │
│  │ 4. Store pre-extracted entities                             │    │
│  │ 5. Create entity-chunk associations                         │    │
│  │ 6. Create edges from relationships                          │    │
│  │ 7. Background Leiden clustering                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start KBV2 API

```bash
cd ~/development/kbv2
uv run python -m knowledge_base.main
# Or use the entry point:
uv run python -c "from knowledge_base import main; main()"
```

KBV2 will start on `http://localhost:8000`

### 2. Start YouTube Pipeline API

```bash
cd ~/development/youtube-content-pipeline
uv run python -m src.api.main
```

YouTube Pipeline API will start on `http://localhost:8001`

### 3. Run Integration Test

```bash
cd ~/development/kbv2
python test_external_ingestion.py
```

## API Reference

### KBV2 External Ingestion Endpoint

#### POST `/api/v1/documents/external`

Ingest a pre-processed document from an external pipeline.

**Request Body:**

```json
{
  "source_type": "youtube",
  "source_id": "e8jcSvurVQ0",
  "title": "Bitcoin Technical Analysis",
  "content": "# Full markdown content...",
  "url": "https://youtube.com/watch?v=e8jcSvurVQ0",
  "entities": [
    {
      "name": "Bitcoin",
      "entity_type": "CRYPTOCURRENCY",
      "description": "Leading cryptocurrency",
      "confidence": 0.95,
      "properties": {"symbol": "BTC"}
    }
  ],
  "relationships": [
    {
      "source": "Bitcoin",
      "target": "$45,000",
      "relationship_type": "HAS_RESISTANCE_LEVEL",
      "confidence": 0.92
    }
  ],
  "sections": [
    {
      "title": "Executive Summary",
      "content": "...",
      "section_type": "summary"
    }
  ],
  "domain": "FINANCIAL",
  "language": "en",
  "author": "Crypto Analyst",
  "published_at": "2026-02-13T12:00:00Z",
  "duration_seconds": 1847,
  "skip_embedding": false,
  "skip_entity_extraction": false
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "document_id": "uuid-string",
    "status": "completed",
    "chunks_created": 12,
    "entities_created": 9,
    "relationships_created": 6,
    "processing_time_ms": 3420
  },
  "error": null,
  "metadata": {
    "source_type": "youtube",
    "source_id": "e8jcSvurVQ0",
    "domain": "FINANCIAL"
  }
}
```

### Check Document Status

#### GET `/api/v1/documents/external/status/{source_type}/{source_id}`

Check if a document from a specific source has been ingested.

**Example:**
```bash
curl http://localhost:8000/api/v1/documents/external/status/youtube/e8jcSvurVQ0
```

## Entity Types Supported

The KBV2 schema supports 80+ edge types. Common ones for trading content:

| Edge Type | Description | Example |
|-----------|-------------|---------|
| `HAS_RESISTANCE_LEVEL` | Price resistance | Bitcoin → $45,000 |
| `HAS_SUPPORT_LEVEL` | Price support | Bitcoin → $42,500 |
| `INDICATES` | Signal relationship | MACD → Bitcoin |
| `TARGETS` | Price target | Pattern → $48,000 |
| `CAUSES` | Causal relationship | News → Price movement |
| `LOCATED_NEAR` | Proximity | Level A → Level B |
| `MENTIONS` | Reference | Video → Asset |

## Data Flow Example

### 1. Submit Video for Analysis (YouTube Pipeline)

```bash
curl -X POST http://localhost:8001/api/v1/videos/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://youtube.com/watch?v=VIDEO_ID",
    "priority": "high"
  }'
```

**Response:**
```json
{
  "job_id": "abc-123",
  "status": "accepted"
}
```

### 2. Check Analysis Status

```bash
curl http://localhost:8001/api/v1/videos/jobs/abc-123
```

### 3. Export to KBV2

```bash
curl -X POST http://localhost:8001/api/v1/kb/export-to-kb \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "VIDEO_ID",
    "include_visual_analysis": true
  }'
```

### 4. Query in KBV2

```bash
curl -X POST http://localhost:8000/api/v1/documents:search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bitcoin support resistance analysis",
    "search_type": "hybrid",
    "domains": ["FINANCIAL"],
    "limit": 10
  }'
```

## Configuration

### KBV2 Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/knowledge_base

# LLM Gateway
LLM_API_BASE=http://localhost:8087/v1
LLM_API_KEY=sk-dummy

# Embeddings
EMBEDDING_API_BASE=http://localhost:11434
DEFAULT_EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSION=1024
```

### YouTube Pipeline Environment Variables

```bash
# MongoDB
MONGODB_URL=mongodb://localhost:27017/video_pipeline

# LLM
LLM_API_BASE=http://localhost:8087/v1
LLM_TRANSSCRIPT_MODEL=gemini-2.5-flash
LLM_FRAME_MODEL=qwen3-vl-plus
LLM_SYNTHESIS_MODEL=gemini-2.5-flash

# KB Integration
KB_API_ENDPOINT=http://localhost:8000/api/v1
```

## Processing Pipeline Details

### Stage 1: Document Creation
- Creates document record with metadata
- Assigns UUID
- Stores source_type and source_id for deduplication

### Stage 2: Chunking
- Uses semantic chunker with 512 token chunks
- Respects provided sections (adds section headers)
- 50 token overlap between chunks

### Stage 3: Embedding
- bge-m3 model via Ollama (1024 dimensions)
- Batch processing for efficiency
- Stores in PostgreSQL pgvector

### Stage 4: Entity Storage
- Pre-extracted entities are stored with high confidence (0.9+)
- Entities are embedded for similarity search
- Properties preserved from external pipeline

### Stage 5: Entity-Chunk Associations
- Links entities to chunks where they appear
- Uses text matching to find mentions
- Stores grounding quotes

### Stage 6: Edge Creation
- Creates edges from pre-extracted relationships
- Validates edge types against schema
- Falls back to RELATED_TO for unknown types

### Stage 7: Background Clustering
- Triggers Leiden community detection
- Groups related entities into communities
- Updates entity community assignments

## Error Handling

### Retry Logic
- Failed exports are retried 3 times with exponential backoff
- Dead letter queue (DLQ) for persistent failures

### Validation
- Invalid edge types fall back to RELATED_TO
- Missing entities in relationships are skipped
- Malformed requests return 400 with details

### Monitoring
Check export status:
```bash
curl http://localhost:8001/api/v1/kb/export-status/{export_id}
```

## Performance Benchmarks

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Document Ingestion | 2-5 seconds | Excludes video analysis |
| Chunking | 100-500ms | Depends on content length |
| Embedding | 1-3 seconds | bge-m3, 1024-dim |
| Entity Storage | 200-800ms | Depends on entity count |
| Edge Creation | 100-400ms | Depends on relationship count |
| **Total** | **3-10 seconds** | For typical 10-min video |

## Best Practices

### 1. Batch Processing
For multiple videos, use batch export:
```bash
curl -X POST http://localhost:8001/api/v1/kb/export-to-kb/batch \
  -d '{"video_ids": ["id1", "id2", "id3"]}'
```

### 2. Domain Selection
Set appropriate domain for better entity extraction:
- `FINANCIAL` for trading content
- `TECHNOLOGY` for tech reviews
- `GENERAL` for mixed content

### 3. Section Structure
Provide sections for better chunking:
```json
"sections": [
  {"title": "Summary", "content": "...", "section_type": "summary"},
  {"title": "Analysis", "content": "...", "section_type": "analysis"},
  {"title": "Signals", "content": "...", "section_type": "signals"}
]
```

### 4. Confidence Scores
- Pre-extracted entities: 0.85-0.95
- High-confidence signals: 0.90+
- Speculative mentions: 0.60-0.75

## Troubleshooting

### Issue: Document not found after ingestion
**Check:**
```bash
# Check document status
curl http://localhost:8000/api/v1/documents/external/status/youtube/VIDEO_ID

# Check KBV2 logs for errors
tail -f ~/development/kbv2/logs/kbv2.log
```

### Issue: Entities not appearing in search
**Possible causes:**
1. Embeddings not generated (check `skip_embedding: false`)
2. Entity-chunk associations not created
3. Clustering not completed (background task)

### Issue: Relationships missing
**Check:**
1. Entity names must match exactly between entities and relationships
2. Edge type must be valid (falls back to RELATED_TO if invalid)
3. Source and target entities must exist in entities list

## Docker Compose Setup

```yaml
version: '3.8'

services:
  kbv2:
    build: ./kbv2
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@postgres/knowledge_base
      - LLM_API_BASE=http://llm-gateway:8087/v1
      - EMBEDDING_API_BASE=http://ollama:11434
    depends_on:
      - postgres
      - ollama

  youtube-pipeline:
    build: ./youtube-content-pipeline
    ports:
      - "8001:8000"
    environment:
      - MONGODB_URL=mongodb://mongo:27017/video_pipeline
      - KB_API_ENDPOINT=http://kbv2:8000/api/v1
    depends_on:
      - mongo
      - kbv2

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=knowledge_base
    volumes:
      - postgres_data:/var/lib/postgresql/data

  mongo:
    image: mongo:7
    volumes:
      - mongo_data:/data/db

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    # Pre-load bge-m3 model
    entrypoint: >
      sh -c "ollama serve &
             sleep 5
             ollama pull bge-m3
             wait"

volumes:
  postgres_data:
  mongo_data:
  ollama_data:
```

## Next Steps

1. **Run the integration test:** `python test_external_ingestion.py`
2. **Analyze a video:** Use YouTube Pipeline CLI or API
3. **Export to KB:** Call the export endpoint
4. **Explore knowledge graph:** Use KBV2 query endpoints

## Support

- KBV2 Issues: Check `~/development/kbv2/logs/`
- Pipeline Issues: Check `~/development/youtube-content-pipeline/logs/`
- Integration Issues: Run `test_external_ingestion.py` with `--verbose`

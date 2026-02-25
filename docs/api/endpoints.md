# KBV2 API Endpoints

## Overview

KBV2 provides a REST API for document ingestion, search, and knowledge graph operations.

**Base URL:** `http://localhost:8088`

**Interactive Docs:** `http://localhost:8088/redoc`

---

## Endpoints

### Health & Stats

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "components": {
    "sqlite": true,
    "chromadb": true,
    "kuzu": true,
    "vision_api": true
  }
}
```

#### GET /stats

Get storage statistics.

**Response:**
```json
{
  "sqlite": {
    "documents": 10,
    "chunks": 45,
    "vectors": 45,
    "db_size_mb": 0.5,
    "vector_enabled": false,
    "fts_enabled": true
  },
  "chromadb": {
    "collection_name": "knowledge_base",
    "embedding_count": 45,
    "storage_size_mb": 0.2,
    "distance_metric": "cosine"
  },
  "kuzu": {
    "entities": 120,
    "chunks": 45,
    "documents": 10,
    "communities": 5,
    "relationships": 200,
    "mentions": 80
  }
}
```

---

### Documents

#### POST /documents

Create a new document.

**Request:**
```json
{
  "name": "Bitcoin ETF Report",
  "content": "Document content here...",
  "domain": "INSTITUTIONAL_CRYPTO",
  "metadata": {
    "source": "web"
  }
}
```

**Response:**
```json
{
  "id": "uuid-123",
  "name": "Bitcoin ETF Report",
  "status": "pending",
  "created_at": "2026-02-25T00:00:00Z"
}
```

#### GET /documents

List all documents.

**Query Parameters:**
- `limit`: Number of results (default: 100)
- `offset`: Pagination offset
- `domain`: Filter by domain
- `status`: Filter by status

**Response:**
```json
{
  "documents": [
    {
      "id": "uuid-123",
      "name": "Bitcoin ETF Report",
      "status": "processed",
      "domain": "INSTITUTIONAL_CRYPTO",
      "created_at": "2026-02-25T00:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET /documents/{document_id}

Get a specific document.

**Response:**
```json
{
  "id": "uuid-123",
  "name": "Bitcoin ETF Report",
  "content": "Document content...",
  "status": "processed",
  "domain": "INSTITUTIONAL_CRYPTO",
  "metadata": {},
  "created_at": "2026-02-25T00:00:00Z"
}
```

---

### Ingestion

#### POST /ingest

Ingest a document from file path.

**Request:**
```json
{
  "file_path": "/path/to/document.md",
  "domain": "BITCOIN"
}
```

**Response:**
```json
{
  "document_id": "uuid-456",
  "name": "document.md",
  "status": "processed",
  "chunk_count": 5
}
```

---

### Search

#### POST /search

Hybrid search across documents and chunks.

**Request:**
```json
{
  "query": "Bitcoin ETF price targets",
  "limit": 10,
  "mode": "HYBRID"
}
```

**Modes:**
- `STANDARD` - Basic vector + keyword search
- `HYBRID` - BM25 + vector with RRF fusion
- `DUAL_LEVEL` - LightRAG-style dual retrieval
- `GRAPH_ENHANCED` - HippoRAG-style graph traversal
- `CORRECTIVE` - CRAG-style corrective retrieval

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "uuid-789",
      "document_id": "uuid-123",
      "text": "Bitcoin price target for 2024...",
      "score": 0.85,
      "document_name": "Bitcoin ETF Report"
    }
  ],
  "query": "Bitcoin ETF price targets",
  "count": 1
}
```

---

### Graph

#### GET /graph/entities

List entities from knowledge graph.

**Query Parameters:**
- `name`: Filter by entity name (partial match)
- `entity_type`: Filter by type
- `limit`: Number of results (default: 100)

**Response:**
```json
{
  "entities": [
    {
      "id": "entity-123",
      "name": "BlackRock IBIT",
      "type": "ETF",
      "properties": {
        "aum": "$45 billion"
      }
    }
  ],
  "count": 1
}
```

#### GET /graph/entities/{entity_id}

Get entity with relationships.

**Response:**
```json
{
  "id": "entity-123",
  "name": "BlackRock IBIT",
  "type": "ETF",
  "properties": {
    "aum": "$45 billion"
  },
  "relationships": [
    {
      "id": "rel-456",
      "type": "tracks",
      "target": {
        "id": "entity-789",
        "name": "Bitcoin",
        "type": "Cryptocurrency"
      }
    }
  ]
}
```

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400` - Bad Request (invalid parameters)
- `404` - Not Found
- `500` - Internal Server Error

---

## Example Usage

### cURL

```bash
# Health check
curl http://localhost:8088/health

# Ingest document
curl -X POST http://localhost:8088/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/doc.md", "domain": "BITCOIN"}'

# Search
curl -X POST http://localhost:8088/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Bitcoin ETF", "limit": 5}'
```

### Python

```python
import httpx

async def search(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8088/search",
            json={"query": query, "limit": 10}
        )
        return response.json()
```

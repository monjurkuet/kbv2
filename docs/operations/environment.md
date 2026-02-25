# KBV2 Environment Configuration

## Overview

KBV2 configuration is split between two files:
- **`config.yaml`** - All non-secret settings
- **`.env`** - Secrets only (API keys)

---

## config.yaml

The main configuration file for all non-secret settings.

```yaml
# KBV2 Configuration
# Secrets (API keys) should be set in .env file

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
  semantic_chunk_size: 1536

domain:
  default_domain: GENERAL
  confidence_threshold: 0.7

rag:
  default_mode: HYBRID
  top_k: 10

server:
  host: 0.0.0.0
  port: 8088
  websocket_port: 8765
```

---

## .env (Secrets Only)

```bash
# LLM API Key (if required by your gateway)
LLM_API_KEY=

# Embedding API Key (if required)
EMBEDDING_API_KEY=
```

---

## Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| KBV2 API | `http://localhost:8088` | Main REST API |
| LLM Gateway | `http://localhost:8087/v1` | OpenAI-compatible LLM API |
| Ollama | `http://localhost:11434` | Embedding generation |

---

## Embedding Model

| Setting | Value |
|---------|-------|
| Model | `bge-m3` |
| Dimensions | 1024 |
| API | Ollama (`localhost:11434`) |

```bash
# Pull embedding model
ollama pull bge-m3

# Verify
curl http://localhost:11434/api/tags
```

---

## LLM Gateway

KBV2 uses an OpenAI-compatible LLM API:

| Setting | Value |
|---------|-------|
| URL | `http://localhost:8087/v1` |
| Default Model | `gemini-2.5-flash-lite` |

```bash
# Verify gateway
curl http://localhost:8087/v1/models

# Check health
curl http://localhost:8087/v1/health
```

---

## Storage Architecture

Portable, file-based storage in `data/` directory:

| Component | Location | Purpose |
|-----------|----------|---------|
| SQLite | `data/knowledge.db` | Documents + FTS5 |
| ChromaDB | `data/chroma/` | Vector embeddings |
| Kuzu | `data/knowledge_graph.kuzu` | Knowledge graph |

**No DATABASE_URL required** - all databases are embedded.

---

## Verification

```bash
# Check config loads correctly
uv run python -c "from knowledge_base.config.loader import load_config; c = load_config(); print(f'LLM: {c.llm.model}, Port: {c.server.port}')"

# Check server health
curl http://localhost:8088/health
```

---

## Related Documentation

- [Deployment Guide](../guides/deployment.md)
- [Quick Start](../QUICK_START.md)

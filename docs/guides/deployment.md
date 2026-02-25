# KBV2 Deployment Guide

## Overview

KBV2 uses a **portable storage architecture** - no external database servers required. All data is stored in local files, making deployment simple and portable.

---

## Prerequisites

- **Python 3.12+**
- **uv** package manager
- **Ollama** for embeddings (localhost:11434)
- **OpenAI-compatible LLM API** (localhost:8087/v1)

**No database setup required** - SQLite, ChromaDB, and Kuzu are embedded.

---

## Quick Start

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your API keys (secrets only)
# Edit config.yaml for all other settings

# Start server
./start.sh
```

---

## Configuration

### config.yaml (Non-Secret Settings)

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

server:
  host: 0.0.0.0
  port: 8088
```

### .env (Secrets Only)

```bash
LLM_API_KEY=your-api-key-here
EMBEDDING_API_KEY=your-api-key-here
```

---

## Storage Architecture

| Component | Location | Purpose |
|-----------|----------|---------|
| SQLite | `data/knowledge.db` | Documents + FTS5 full-text search |
| ChromaDB | `data/chroma/` | Vector similarity (HNSW, 1024 dims) |
| Kuzu | `data/knowledge_graph.kuzu` | Knowledge graph (Cypher queries) |

**All data is portable** - just copy the `data/` directory to migrate or backup.

---

## Starting Services

### Start Ollama (Embeddings)

```bash
# Pull bge-m3 model
ollama pull bge-m3

# Start Ollama server
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

### Start LLM Gateway

Ensure your OpenAI-compatible LLM API is running at `http://localhost:8087/v1`.

```bash
# Verify
curl http://localhost:8087/v1/models
```

### Start KBV2 Server

```bash
# One-stop start script (recommended)
./start.sh

# Or with auto-reload for development
./start.sh --reload

# Or custom port
./start.sh --port 9000

# Or manual
uv run uvicorn knowledge_base.main:app --host 0.0.0.0 --port 8088
```

---

## Systemd Service (Production)

Create `/etc/systemd/system/kbv2.service`:

```ini
[Unit]
Description=KBV2 Knowledge Base
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/kbv2
Environment="PATH=/home/youruser/.local/bin:/usr/bin:/bin"
ExecStart=/path/to/kbv2/start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kbv2
sudo systemctl start kbv2
sudo systemctl status kbv2
```

---

## Verification

### Health Check

```bash
curl http://localhost:8088/health
# Expected: {"status": "healthy", "version": "0.2.0", ...}
```

### Statistics

```bash
curl http://localhost:8088/stats
```

### Test Ingestion

```bash
curl -X POST http://localhost:8088/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "test_data/markdown/sample_bitcoin_document.md", "domain": "BITCOIN"}'
```

---

## Backup and Recovery

### Backup

```bash
# Simple backup - just copy data directory
tar -czf kbv2_backup_$(date +%Y%m%d).tar.gz data/

# Or for continuous backup, sync to another location
rsync -av data/ /backup/kbv2/
```

### Recovery

```bash
# Stop server
sudo systemctl stop kbv2
# Or: pkill -f "uvicorn knowledge_base"

# Restore from backup
tar -xzf kbv2_backup_20260208.tar.gz

# Restart server
./start.sh
```

---

## Monitoring

### Health Endpoints

- `/health` - Basic health check
- `/stats` - Storage statistics

### Log Monitoring

```bash
# View server logs
journalctl -u kbv2 -f

# Or if running manually, redirect to file:
./start.sh 2>&1 | tee -a kbv2.log
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process on port
lsof -i :8088

# Kill process
kill -9 <PID>

# Or use start.sh which handles this automatically
./start.sh
```

### Ollama Not Running

```bash
# Check status
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull model if needed
ollama pull bge-m3
```

### LLM Gateway Unavailable

```bash
# Check API health
curl http://localhost:8087/v1/models

# Verify gateway is running
# (depends on your LLM gateway setup)
```

### Import Errors

```bash
# Verify installation
uv run python -c "from knowledge_base import __version__; print(__version__)"

# Reinstall if needed
uv sync --reinstall
```

---

## Performance Considerations

### Storage Size

- SQLite database grows with documents and chunks
- ChromaDB stores 1024-dim vectors (~4KB per chunk)
- Kuzu stores entities and relationships

### Memory Usage

- Embedding generation: ~2GB with bge-m3
- LLM calls: depends on model and concurrent requests
- Server overhead: ~500MB

### Concurrency

- SQLite handles concurrent reads well
- Writes are serialized at database level
- Use connection pooling (built-in)

---

## Security

### Network Security

```bash
# Firewall rules (if needed)
sudo ufw allow 8088/tcp  # API port
sudo ufw deny 11434/tcp  # Ollama (local only)
sudo ufw deny 8087/tcp   # LLM API (local only)
```

### API Authentication

For production, add authentication middleware in `main.py` or use a reverse proxy with auth.

---

## Related Documentation

- [Quick Start](../QUICK_START.md)
- [API Endpoints](../api/endpoints.md)
- [Ingestion Guide](ingestion.md)

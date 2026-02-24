# KBV2 Deployment Guide

## Overview

This guide covers production deployment of the KBV2 knowledge base system, including environment setup, database configuration, and monitoring.

---

## Prerequisites

- **PostgreSQL 16+** with pgvector extension
- **Python 3.12+**
- **uv** package manager
- **Ollama** for embeddings (localhost:11434)
- **OpenAI-compatible LLM API** (localhost:8087/v1)

---

## Step 1: Database Setup

### Install PostgreSQL and pgvector

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-16 postgresql-contrib-16

# Install pgvector
cd /tmp
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### Create Database

```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create database and user
sudo -u postgres psql <<EOF
CREATE DATABASE knowledge_base;
CREATE USER agentzero;
GRANT ALL PRIVILEGES ON DATABASE knowledge_base TO agentzero;
\c knowledge_base
CREATE EXTENSION vector;
GRANT ALL ON SCHEMA public TO agentzero;
EOF
```

### Configure PostgreSQL

Edit `/etc/postgresql/16/main/postgresql.conf`:

```ini
shared_preload_libraries = 'vector'
maintenance_work_mem = '4GB'
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

---

## Step 2: Environment Configuration

### Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Database
DATABASE_URL=postgresql://agentzero@localhost/knowledge_base

# LLM Configuration (OpenAI-compatible API)
LLM_API_BASE=http://localhost:8087/v1
LLM_API_KEY=sk-dummy

# Embedding Configuration (Ollama)
EMBEDDING_API_BASE=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSIONS=1024

# Self-Improvement Features
ENABLE_EXPERIENCE_BANK=true
ENABLE_PROMPT_EVOLUTION=true
ENABLE_ONTOLOGY_VALIDATION=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080
```

---

## Step 3: Install Dependencies

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

---

## Step 4: Run Database Migrations

```bash
# Verify current migration status
alembic current

# Run all migrations
alembic upgrade head

# Expected output: experience_bank_001
```

### Verify Migration

```bash
# Check tables exist
psql -d knowledge_base -c "\dt"

# Check extraction_experiences table
psql -d knowledge_base -c "\d extraction_experiences"

# Check vector dimensions (should be 1024)
psql -d knowledge_base -c "SELECT atttypmod FROM pg_attribute WHERE attname = 'embedding' AND attrelid = 'entities'::regclass;"
```

---

## Step 5: Start Ollama for Embeddings

```bash
# Pull bge-m3 model
ollama pull bge-m3

# Start Ollama server
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

## Step 6: Start Production Server

### Using uvicorn directly

```bash
# Start server
uv run python -m knowledge_base.production

# Or with custom host/port
PORT=8765 HOST=0.0.0.0 uv run python -m knowledge_base.production
```

### Using systemd service

Create `/etc/systemd/system/kbv2.service`:

```ini
[Unit]
Description=KBV2 Knowledge Base
After=network.target postgresql.service

[Service]
Type=simple
User=agentzero
WorkingDirectory=/home/muham/development/kbv2
Environment="PATH=/home/muham/.local/bin:/usr/bin:/bin"
ExecStart=/home/muham/.local/bin/uv run python -m knowledge_base.production
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

## Step 7: Verify Deployment

### Health Check

```bash
# Basic health check
curl http://localhost:8765/health

# Extended health check (v2)
curl http://localhost:8765/api/v2/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "2.0.0",
#   "components": {
#     "database": true,
#     "self_improving_orchestrator": true,
#     "experience_bank": true
#   }
# }
```

### Statistics

```bash
# Get statistics
curl http://localhost:8765/api/v2/stats

# Get Prometheus metrics
curl http://localhost:8765/metrics
```

### Test Ingestion

```bash
# Ingest a test document
./ingest_cli.py test_data/markdown/sample_bitcoin_document.md --domain BITCOIN

# Or via API
curl -X POST "http://localhost:8765/api/v2/documents/process" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "test_data/markdown/sample_bitcoin_document.md", "domain": "BITCOIN"}'
```

---

## Monitoring

### Prometheus Integration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'kbv2'
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key Metrics

- `kb_documents_processed_total` - Total documents processed
- `kb_extraction_quality_avg` - Average extraction quality
- `kb_experience_bank_hit_rate` - Experience Bank hit rate
- `kb_llm_calls_total` - Total LLM calls
- `kb_llm_failures_total` - Total LLM failures

### Grafana Dashboard

Import dashboard JSON (simplified):

```json
{
  "dashboard": {
    "title": "KBV2 Monitoring",
    "panels": [
      {
        "title": "Documents Processed",
        "targets": [{"expr": "kb_documents_processed_total"}]
      },
      {
        "title": "Extraction Quality",
        "targets": [{"expr": "kb_extraction_quality_avg"}]
      },
      {
        "title": "Experience Bank Hit Rate",
        "targets": [{"expr": "kb_experience_bank_hit_rate"}]
      }
    ]
  }
}
```

---

## Maintenance

### Daily Checks

```bash
# Check health
curl http://localhost:8765/api/v2/health

# Check stats
curl http://localhost:8765/api/v2/stats

# Check database connections
psql -d knowledge_base -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'knowledge_base';"
```

### Weekly Tasks

```bash
# Database maintenance
psql -d knowledge_base -c "VACUUM ANALYZE;"

# Check Experience Bank growth
psql -d knowledge_base -c "SELECT domain, COUNT(*) FROM extraction_experiences GROUP BY domain;"

# Review logs
tail -f /tmp/kbv2_ingestion.log
```

### Monthly Tasks

```bash
# Check table sizes
psql -d knowledge_base -c "
  SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Backup database
pg_dump -h localhost -U agentzero knowledge_base | gzip > kbv2_backup_$(date +%Y%m%d).sql.gz

# Check index usage
psql -d knowledge_base -c "
  SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
  FROM pg_stat_user_indexes
  ORDER BY idx_scan ASC;
"
```

---

## Troubleshooting

### Issue: Migration Failed

**Solution:**
```bash
# Check current version
alembic current

# Fix version mismatch
psql -d knowledge_base -c "UPDATE alembic_version SET version_num = '0003_upgrade_to_1024_dimensions';"

# Retry migration
alembic upgrade head
```

### Issue: High Memory Usage

**Solution:**
- Reduce Experience Bank size in config
- Reduce concurrent document processing
- Enable garbage collection

### Issue: Low Experience Bank Hit Rate

**Solution:**
- Wait for more documents (need baseline experiences)
- Lower quality threshold temporarily
- Check domain distribution

### Issue: Embedding Failures

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve

# Verify bge-m3 model
ollama pull bge-m3
```

### Issue: LLM API Unavailable

**Solution:**
```bash
# Check API health
curl http://localhost:8087/v1/health

# Check available models
curl http://localhost:8087/v1/models

# Verify environment variable
echo $LLM_API_BASE
```

---

## Scaling

### Horizontal Scaling

Run multiple instances behind load balancer:

```bash
# Instance 1
PORT=8765 uv run python -m knowledge_base.production

# Instance 2
PORT=8766 uv run python -m knowledge_base.production

# Instance 3
PORT=8767 uv run python -m knowledge_base.production
```

### Database Read Replicas

Configure read replicas for query offload:

```bash
# In .env
DATABASE_URL=postgresql://agentzero@primary:5432/knowledge_base
DATABASE_READ_URL=postgresql://agentzero@replica:5432/knowledge_base
```

### Caching

Consider Redis for:
- Session caching
- Query result caching
- Experience Bank caching

---

## Security

### Secrets Management

Never commit `.env` file. Use secret management:
- HashiCorp Vault
- AWS Secrets Manager
- Kubernetes Secrets

### Network Security

```bash
# Firewall rules
sudo ufw allow 8765/tcp  # API port
sudo ufw allow 5432/tcp  # PostgreSQL (if remote)
sudo ufw deny 11434/tcp  # Ollama (local only)
sudo ufw deny 8087/tcp  # LLM API (local only)
```

### Authentication

Enable API authentication in production:

```python
# In production.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    if token.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401)
    return token

@app.get("/api/v2/secure-endpoint", dependencies=[Depends(verify_token)])
async def secure_endpoint():
    return {"status": "authenticated"}
```

---

## Backup and Recovery

### Backup Strategy

```bash
# Daily automated backup
0 2 * * * pg_dump -h localhost -U agentzero knowledge_base | gzip > /backup/kbv2_$(date +\%Y\%m\%d).sql.gz

# Keep last 30 days
find /backup -name "kbv2_*.sql.gz" -mtime +30 -delete
```

### Recovery

```bash
# Stop application
sudo systemctl stop kbv2

# Drop and recreate database
psql -U postgres -c "DROP DATABASE knowledge_base;"
psql -U postgres -c "CREATE DATABASE knowledge_base;"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE knowledge_base TO agentzero;"

# Restore from backup
gunzip -c /backup/kbv2_20260208.sql.gz | psql -h localhost -U agentzero knowledge_base

# Restart application
sudo systemctl start kbv2
```

---

## Performance Tuning

### PostgreSQL Configuration

```ini
# postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
work_mem = 256MB
max_connections = 100
```

### Connection Pooling

Use PgBouncer for connection pooling:

```ini
# pgbouncer.ini
[databases]
knowledge_base = host=localhost port=5432 dbname=knowledge_base

[pgbouncer]
pool_mode = transaction
max_client_conn = 100
default_pool_size = 25
```

---

## Related Documentation

- [Ingestion Guide](ingestion.md)
- [Self-Improvement Features](self_improvement.md)
- [Setup Guide](operations/setup.md)
- [Operations Runbook](operations/runbook.md)

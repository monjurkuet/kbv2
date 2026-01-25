# Setup and Operations Guide

## Prerequisites

1. **PostgreSQL 16+** with pgvector extension
2. **Python 3.12+**
3. **uv** package manager

## Database Setup

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
# Start PostgreSQL service
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
hnsw.ef_search = 100
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

### Initialize Database Schema

```bash
python scripts/setup_db.py
```

## Environment Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `DATABASE_URL`: PostgreSQL connection string
- `LLM_GATEWAY_URL`: Your local LLM API endpoint
- `GOOGLE_API_KEY`: Google Embeddings API key

## Development

### Install Dependencies

```bash
uv sync
```

### Run the System

```bash
uv run knowledge-base
```

### Testing and Quality

```bash
# Run tests
uv run pytest tests/ -v

# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/
```

## Architecture Overview

The system follows Google Engineering Standards:

- **AIP-121**: Resource-oriented API design
- **AIP-122**: Hierarchical naming conventions
- **AIP-191**: Strict `src/` directory layout

### Components

1. **Ingestion Plane**: Document partitioning, adaptive gleaning extraction, embedding
2. **Persistence Plane**: pgvector with HNSW indexes
3. **Intelligence Plane**: Entity resolution, Leiden clustering, recursive summarization
4. **Observability**: Logfire-based SRE-lite monitoring

## Production Deployment

### Database Configuration

1. **Connection Pooling**: Use PgBouncer for connection pooling
2. **Performance Tuning**:
   - Set `max_connections` based on application server count
   - Configure `work_mem` appropriately for your workload
   - Enable `shared_buffers` at 25% of system RAM
3. **High Availability**:
   - Set up streaming replication to standby servers
   - Configure automatic failover with repmgr or Patroni

### Application Deployment

#### Using Gunicorn

```bash
# Install production dependencies
uv sync --frozen

# Run with Gunicorn and Uvicorn workers
uv run gunicorn knowledge_base.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

#### Systemd Service

Create `/etc/systemd/system/knowledge-base.service`:

```ini
[Unit]
Description=Knowledge Base API
After=network.target

[Service]
Type=notify
User=agentzero
Group=agentzero
WorkingDirectory=/opt/kbv2
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/uv run gunicorn knowledge_base.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable knowledge-base
sudo systemctl start knowledge-base
```

### Operations

#### Monitoring

- **Metrics**: Token throughput, latency, success rates via Logfire dashboard
- **Health Endpoint**: `/healthz` for load balancer health checks
- **Structured Logging**: JSON-formatted logs for centralized logging systems

#### Backup Strategy

```bash
# Daily PostgreSQL backup
pg_dump -h localhost -U agentzero knowledge_base | gzip > /backup/knowledge_base_$(date +%Y%m%d).sql.gz

# Document backup (if storing original files)
rsync -av /data/knowledge_base/documents/ /backup/documents/
```

#### Scaling

- **Horizontal Scaling**: Run multiple API instances behind a load balancer
- **Database Read Replicas**: Offload read queries to replicas
- **Caching**: Consider Redis for frequently accessed entity data

#### Maintenance

```bash
# Weekly database maintenance
psql -U agentzero -d knowledge_base -c "VACUUM ANALYZE;"

# Monthly index rebuilds
psql -U agentzero -d knowledge_base -c "REINDEX DATABASE knowledge_base;"

# Monitor pgvector indexes
psql -U agentzero -d knowledge_base -c "SELECT * FROM pg_vector_index_info();"
```

#### Incident Response

1. **High Latency**: Check database connection pool, query execution plans
2. **Embedding Failures**: Verify Google API quota and connectivity
3. **Memory Issues**: Monitor worker memory usage and restart if needed
4. **Disk Space**: Monitor PostgreSQL WAL growth and log files
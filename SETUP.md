# Setup Guide

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

### Run Setup Script

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

## Run the Application

```bash
# Install dependencies
uv sync

# Run the system
uv run knowledge-base
```

## Testing

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
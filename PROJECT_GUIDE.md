# KBV2 Project Guide

This guide provides everything you need to start working on the KBV2 (Knowledge Base Version 2) system.

## 1. Project Overview

**KBV2 is a high-fidelity information extraction engine that transforms unstructured documents into a structured, temporally-aware knowledge graph using adaptive AI extraction techniques.**

### Key Capabilities
- **Adaptive Multi-Pass Extraction**: 2-pass density-aware knowledge gleaning
- **Verbatim-Grounded Entity Resolution**: Hybrid matching with mandatory grounding quotes
- **Hierarchical Leiden Clustering**: Macro and micro community detection
- **Temporal Information Extraction**: ISO-8601 normalized temporal claims
- **Natural Language Query Interface**: Text-to-SQL translation powered by LLMs
- **Human Review Queue**: Flag low-confidence resolutions for manual review
- **Domain Tagging**: Propagate domain context throughout the pipeline
- **Vector Search**: Semantic search over entities and document chunks

### Architecture Stack
- **Backend**: Python 3.12+, FastAPI, SQLAlchemy, PostgreSQL + pgvector
- **Frontend**: TypeScript, SolidJS, Vite, TailwindCSS, Sigma.js (graph visualization)
- **AI/ML**: Multiple LLM providers (OpenAI, Google), vector embeddings, Leiden clustering
- **Infrastructure**: Docker-ready, uv package manager, comprehensive testing

## 2. Directory Structure

```
kbv2/
├── src/knowledge_base/           # Python backend source code
│   ├── orchestrator.py           # Main ReAct ingestion pipeline
│   ├── query_api.py              # FastAPI query endpoints (/api/v1/query)
│   ├── review_api.py             # FastAPI review endpoints (/api/v1/review)
│   ├── text_to_sql_agent.py      # Natural language to SQL translator
│   ├── mcp_server.py             # Model Context Protocol WebSocket server
│   ├── review_service.py         # Human review queue management
│   ├── common/                   # Shared utilities
│   │   ├── gateway.py            # LLM API client
│   │   ├── temporal_utils.py     # Temporal normalization
│   │   └── resilient_gateway/    # Circuit breaker & retry logic
│   ├── ingestion/v1/             # Document ingestion pipeline
│   │   ├── partitioning_service.py  # Document parsing & chunking
│   │   ├── gleaning_service.py      # 2-pass adaptive extraction
│   │   └── embedding_client.py      # Vector embedding generation
│   ├── persistence/v1/           # Database layer
│   │   ├── schema.py             # SQLAlchemy models
│   │   └── vector_store.py       # PostgreSQL/pgvector wrapper
│   └── intelligence/v1/          # Analysis layer
│       ├── resolution_agent.py   # Entity deduplication
│       ├── clustering_service.py # Hierarchical Leiden clustering
│       └── synthesis_agent.py    # Map-reduce summarization
│
├── frontend/                     # TypeScript/SolidJS frontend
│   ├── src/                      # Source code
│   │   ├── api/                  # API client & types (auto-generated)
│   │   ├── components/           # React components
│   │   ├── hooks/                # Custom hooks
│   │   └── stores/               # State management
│   ├── tests/e2e/                # Playwright end-to-end tests
│   └── openapi-schema.json       # Auto-generated API schema
│
├── tests/                        # Python test suite
│   ├── unit/                     # Unit tests
│   │   ├── test_api/             # API endpoint tests
│   │   ├── test_orchestrator/    # Orchestrator tests
│   │   └── test_services/        # Service tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test fixtures
│
├── docs/                         # Comprehensive documentation
│   ├── architecture/             # System architecture diagrams
│   ├── database/                 # Database schema & relationships
│   ├── development/              # Developer guides
│   ├── operations/               # Deployment & operations
│   └── api/                      # API documentation
│
├── scripts/                      # Utility scripts
│   ├── setup_db.py              # Database initialization
│   ├── generate_openapi.py      # OpenAPI schema generation
│   └── final_verification.py    # System verification
│
├── logs/                         # Application logs
├── build/                        # Build artifacts
├── .env & .env.example          # Environment configuration
├── pyproject.toml               # Python dependencies (uv)
└── uv.lock                      # Dependency lock file
```

## 3. Quick Start

### Prerequisites
- Python 3.12+ with [uv](https://github.com/astral-sh/uv) installed
- Node.js 18+ with npm
- PostgreSQL 14+ with pgvector extension
- Access to LLM APIs (OpenAI, Google, or custom gateway)

### Backend Setup

```bash
# Clone and enter repository
cd /home/muham/development/kbv2

# Install Python dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your credentials (see Configuration section)

# Initialize database
python scripts/setup_db.py

# Run the backend server
uv run knowledge-base
# API available at: http://localhost:8000
# WebSocket at: ws://localhost:8000/ws
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Generate API client from OpenAPI schema
npm run api:generate

# Run development server
npm run dev
# Available at: http://localhost:3000

# Run e2e tests
npm run test:e2e
```

### Verify Installation

```bash
# Run backend tests
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/test_real_world_pipeline.py -v

# Run frontend e2e tests
cd frontend && npm run test:e2e
```

## 4. Development Workflow

### Starting the Full Stack

```bash
# Terminal 1: Backend
uv run knowledge-base

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: API client watch mode (optional)
cd frontend && npm run api:watch
```

### Common Development Tasks

#### Processing a Document

```bash
# Using API endpoint
curl -X POST http://localhost:8000/api/v1/query/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'

# Or via WebSocket
# Connect to ws://localhost:8000/ws
# Send: {"method": "kbv2/ingest_document", "params": {"file_path": "..."}}
```

#### Querying the Knowledge Graph

```bash
# Natural language query
curl -X POST http://localhost:8000/api/v1/query/text_to_sql \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all companies mentioned in Q4 2024 documents"}'

# Vector search entities
curl -X POST http://localhost:8000/api/v1/query/search_entities \
  -H "Content-Type: application/json" \
  -d '{"query": "technology companies"}'
```

#### Working with the Review Queue

```bash
# Get pending reviews
curl http://localhost:8000/api/v1/review/pending

# Submit review decision
curl -X POST http://localhost:8000/api/v1/review/submit \
  -H "Content-Type: application/json" \
  -d '{
    "entity_ids": [1, 2, 3],
    "resolution": "merge",
    "reviewer_notes": "These refer to the same company"
  }'
```

### Code Changes Workflow

1. **Backend changes**: Edit Python files in `src/knowledge_base/`
2. **Frontend changes**: Edit TypeScript/SolidJS files in `frontend/src/`
3. **API changes**: Modify FastAPI endpoints → regenerate OpenAPI schema → regenerate client
4. **Database changes**: Modify `schema.py` → create migration script → run migration

## 5. Testing

### Backend Tests

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/ -v              # Unit tests only
uv run pytest tests/integration/ -v       # Integration tests only

# Run with coverage
uv run pytest --cov=src/knowledge_base --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_orchestrator/test_orchestrator.py -v

# Run tests matching pattern
uv run pytest -k "test_ingest" -v
```

### Frontend Tests

```bash
cd frontend

# Run all e2e tests
npm run test:e2e

# Run specific test phases
npm run test:phases

# Run specific test file
npx playwright test tests/e2e/ingestion.spec.ts

# Run with UI mode
npx playwright test --ui

# Generate test report
npx playwright show-report
```

### Test Data

Test documents are located in:
- `tests/test_data/` - Backend test documents
- `frontend/test-data/` - Frontend test documents
- `frontend/tests/e2e/test-data/` - E2E test fixtures

### Adding New Tests

**Backend unit test** (`tests/unit/test_api/test_query_api.py`):
```python
import pytest
from fastapi.testclient import TestClient
from knowledge_base.query_api import app

@pytest.mark.asyncio
async def test_ingest_document(client):
    response = client.post("/api/v1/query/ingest", json={
        "file_path": "/path/to/test.pdf"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
```

**Frontend e2e test** (`frontend/tests/e2e/ingestion.spec.ts`):
```typescript
import { test, expect } from '@playwright/test';

test('document ingestion workflow', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('File Path').fill('/path/to/test.pdf');
  await page.getByRole('button', { name: 'Ingest' }).click();
  await expect(page.getByText('Processing complete')).toBeVisible();
});
```

## 6. Key Commands

### Backend Development

```bash
# Run the server
uv run knowledge-base

# Linting
uv run ruff check src/
uv run ruff check src/ --fix  # Auto-fix issues

# Formatting
uv run ruff format src/

# Type checking
uv run mypy src/

# Install new dependency
uv add package-name
uv add --dev package-name  # Dev dependency

# Sync dependencies
uv sync

# Run script directly
uv run python scripts/setup_db.py

# Interactive Python console
uv run python -i src/knowledge_base/orchestrator.py
```

### Frontend Development

```bash
cd frontend

# Development
npm run dev              # Start dev server
npm run build           # Production build
npm run preview         # Preview production build

# API Client
npm run api:generate    # Generate from OpenAPI schema
npm run api:watch       # Watch mode

# Testing
npm run test:e2e        # Run all e2e tests
npx playwright test     # Direct playwright command
npx playwright codegen  # Generate tests interactively

# Code quality
npm run lint            # (if configured)
npx prettier --write .  # Format code
```

### Database

```bash
# Initialize database
python scripts/setup_db.py

# Reset database (WARNING: deletes all data)
python scripts/setup_db.py --reset

# Manual PostgreSQL commands
psql -U agentzero -d knowledge_base

# Check pgvector extension
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### System Operations

```bash
# Check system logs
tail -f logs/kbv2.log

# Run verification script
python scripts/final_verification.py

# Generate OpenAPI schema (automatic on startup)
# Schema saved to: frontend/openapi-schema.json
```

## 7. Documentation

### Core Documentation

- **Project Guide** ← You are here
- **`docs/README.md`**: Main documentation index with ASCII architecture diagrams
- **`docs/development/folder_structure.md`**: Complete directory structure breakdown
- **`docs/architecture/system_overview.md`**: 9-stage ingestion pipeline visualization
- **`docs/architecture/data_flow.md`**: Step-by-step data transformation
- **`docs/database/schema.md`**: Database schema and relationships
- **`docs/api/endpoints.md`**: Complete API endpoint reference
- **`docs/configuration/environment.md`**: Configuration flow and settings

### Architecture Documents

- **`DESIGN_DOC.md`**: High-level system design and goals
- **`docs/technical/ENTITY_PROCESSING_PIPELINE.md`**: Entity processing specification
- **`docs/operations/runbook.md`**: Operations and deployment guide

### API Documentation

- **Interactive API Docs**: http://localhost:8000/docs (when server is running)
- **`docs/api/endpoints.md`**: Detailed endpoint documentation
- **`frontend/openapi-schema.json`**: Machine-readable API schema
- **`frontend/src/api/types.ts`**: Auto-generated TypeScript types

### Implementation Plans

- **`docs/archive/implementation-plan.md`**: Enhancement roadmap
- **`docs/archive/api_planning/`**: API design and planning documents

### Testing Documentation

- **`tests/test_data/TEST_DATA_DOCUMENTATION.md`**: Test data structure
- **`docs/reports/validation-summary.md`**: System validation results
- **`docs/archive/api_planning/KBV2 End-to-End Testing Plan.md`**: E2E testing strategy

## 8. Configuration

### Environment Variables (`.env`)

**Database Configuration**:
```bash
DATABASE_URL=postgresql://agentzero@localhost:5432/knowledge_base
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_base
DB_USER=agentzero
DB_PASSWORD=
```

**LLM Gateway**:
```bash
LLM_GATEWAY_URL=http://localhost:8317/v1/
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

**Google Embeddings**:
```bash
GOOGLE_API_KEY=your_google_key_here
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

**Observability**:
```bash
LOGFIRE_PROJECT=knowledge-base
LOGFIRE_SEND_TO_LOGFIRE=false  # Set true for cloud logging
```

**Ingestion Settings**:
```bash
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_DENSITY_THRESHOLD=0.8      # Second pass trigger
MIN_DENSITY_THRESHOLD=0.3      # Minimum entity density
```

**Clustering Configuration**:
```bash
LEIDEN_RESOLUTION_MACRO=0.8    # Macro community detection
LEIDEN_RESOLUTION_MICRO=1.2    # Micro community detection
LEIDEN_ITERATIONS=10
```

**Entity Resolution**:
```bash
RESOLUTION_CONFIDENCE_THRESHOLD=0.7
RESOLUTION_SIMILARITY_THRESHOLD=0.85
```

**Vector Search (HNSW)**:
```bash
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=100
```

### Configuration Files

**`pyproject.toml`**: Python project configuration
- Dependencies and package metadata
- Development tools (pytest, ruff, mypy)
- Entry point definition

**`frontend/package.json`**: Node.js project configuration
- Dependencies and scripts
- Vite and TailwindCSS settings

**`frontend/vite.config.ts`**: Build configuration
- Development server settings
- Plugin configuration

**`frontend/playwright.config.ts`**: E2E test configuration
- Test directory and browser settings
- Base URL and timeout configuration

**`frontend/openapi-ts.config.ts`**: API client generation
- OpenAPI schema location
- Output directory and settings

### Database Configuration

The system uses **PostgreSQL with pgvector extension** for vector search capabilities.

**Required Extensions**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Key Tables**:
- `documents` - Source document metadata
- `chunks` - Text chunks from partitioning
- `entities` - Extracted entities (companies, people, locations, etc.)
- `edges` - Relationships between entities
- `communities` - Entity clusters (hierarchical)
- `chunk_entity` - Many-to-many junction table
- `review_queue` - Pending human reviews

**Vector Columns**:
- `entities.embedding` - 768-dimensional entity embeddings
- `chunks.embedding` - Document chunk embeddings

### Configuration Changes

**Important**: KBV2 does not support hot reload. Configuration changes require a restart.

1. Modify `.env` file
2. Restart the backend server (`uv run knowledge-base`)
3. For database changes: run migration or reset database

## Getting Help

- **General Questions**: Check `docs/README.md` and related documentation
- **API Issues**: Interactive docs at http://localhost:8000/docs
- **Frontend Issues**: Check browser console and `logs/kbv2.log`
- **Database Issues**: Verify PostgreSQL is running and pgvector is installed
- **LLM Issues**: Check gateway connectivity and API keys

## Architecture Summary

KBV2 follows a **clean architecture** pattern with clear separation:

1. **Core Layer**: `orchestrator.py` - Main business logic and ReAct loop
2. **API Layer**: `query_api.py`, `review_api.py` - HTTP/WebSocket endpoints
3. **Service Layer**: `*_service.py`, `*_agent.py` - Domain-specific logic
4. **Ingestion Layer**: `ingestion/v1/` - Document processing pipeline
5. **Intelligence Layer**: `intelligence/v1/` - Entity resolution, clustering, synthesis
6. **Persistence Layer**: `persistence/v1/` - Database access and vector operations

The system uses **versioned modules** (`v1/`) to support future parallel implementations and backward compatibility.
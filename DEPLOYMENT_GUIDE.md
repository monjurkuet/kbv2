# KBv2 Crypto Knowledgebase - Production Deployment Guide

## Overview

This guide covers production deployment of the self-improving cryptocurrency knowledgebase system, including:
- Environment setup and configuration
- Migration and orchestrator switch
- Monitoring and metrics
- Data pipeline integration

## Prerequisites

- PostgreSQL 15+ with pgvector extension
- Python 3.12+
- LLM API endpoint (OpenAI-compatible, e.g., localhost:8087)
- Ollama for embeddings (localhost:11434)
- Optional: Redis for caching

## Step 1: Database Migration

### 1.1 Run Alembic Migration

```bash
cd /home/muham/development/kbv2

# Activate virtual environment
source .venv/bin/activate

# Run migration
alembic upgrade experience_bank_001
```

### 1.2 Verify Migration

```bash
# Check table exists
psql -d knowledge_base -c "\dt extraction_experiences"

# Check indexes
psql -d knowledge_base -c "SELECT indexname FROM pg_indexes WHERE tablename = 'extraction_experiences';"
```

**Expected Output:**
- Table: `extraction_experiences` created
- Indexes: 6 indexes including domain/quality composite index

## Step 2: Environment Configuration

### 2.1 Create Production Environment File

```bash
cat > .env.production << EOF
# Database
DATABASE_URL=postgresql://agentzero@localhost/knowledge_base

# LLM Configuration
LLM_API_BASE=http://localhost:8087/v1
LLM_MODEL=default
LLM_TIMEOUT=120

# Embedding Configuration
EMBEDDING_API_BASE=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSIONS=1024

# Self-Improvement Features
ENABLE_EXPERIENCE_BANK=true
ENABLE_PROMPT_EVOLUTION=true
ENABLE_ONTOLOGY_VALIDATION=true

# Experience Bank Settings
EXPERIENCE_BANK_MIN_QUALITY=0.90
EXPERIENCE_BANK_MAX_SIZE=10000

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080

# Data Pipeline Integration (Optional)
DATA_PIPELINE_WEBHOOK_URL=
DATA_PIPELINE_API_KEY=
ENABLE_REALTIME_DATA=false

# Performance
MAX_CONCURRENT_DOCUMENTS=5
EOF
```

### 2.2 Load Environment

```bash
export $(cat .env.production | xargs)
```

## Step 3: Switch to Self-Improving Orchestrator

### 3.1 Update Application Entry Point

**Before (original):**
```python
from knowledge_base.orchestrator import IngestionOrchestrator

orchestrator = IngestionOrchestrator()
```

**After (self-improving):**
```python
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

orchestrator = SelfImprovingOrchestrator(
    enable_experience_bank=True,
    enable_prompt_evolution=True,
)
```

### 3.2 Update Main Application File

Edit your main application file (e.g., `main.py` or `app.py`):

```python
#!/usr/bin/env python3
"""KBv2 Crypto Knowledgebase - Production Application."""

import asyncio
import os
from pathlib import Path

from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator
from knowledge_base.config.production import get_config, apply_preset
from knowledge_base.monitoring.metrics import create_monitoring_app, metrics_collector, health_checker
import uvicorn


async def main():
    # Apply production preset
    config = apply_preset("production")
    print(f"Configuration: {config.to_dict()}")
    
    # Initialize orchestrator
    orchestrator = SelfImprovingOrchestrator(
        enable_experience_bank=config.enable_experience_bank,
        enable_prompt_evolution=config.enable_prompt_evolution,
    )
    await orchestrator.initialize()
    
    # Register health checks
    health_checker.register_check("database", lambda: True)  # Add actual DB check
    health_checker.register_check("llm_gateway", lambda: True)  # Add actual LLM check
    
    print("✅ KBv2 Crypto Knowledgebase initialized successfully")
    print(f"   Experience Bank: {'Enabled' if config.enable_experience_bank else 'Disabled'}")
    print(f"   Prompt Evolution: {'Enabled' if config.enable_prompt_evolution else 'Disabled'}")
    print(f"   Ontology Validation: {'Enabled' if config.enable_ontology_validation else 'Disabled'}")
    
    # Example: Process a document
    # document = await orchestrator.process_document(
    #     file_path="crypto_report.pdf",
    #     domain="BITCOIN"
    # )
    
    # Keep running for API/monitoring
    if config.enable_metrics:
        monitoring_app = create_monitoring_app()
        uvicorn.run(
            monitoring_app,
            host="0.0.0.0",
            port=config.health_check_port,
        )


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 4: Start Monitoring

### 4.1 Metrics Endpoint

The monitoring app provides three endpoints:

- **Health Check:** `GET http://localhost:8080/health`
- **Prometheus Metrics:** `GET http://localhost:8080/metrics`
- **Statistics:** `GET http://localhost:8080/stats`

### 4.2 Prometheus Integration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'kbv2'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### 4.3 Grafana Dashboard

Import this dashboard JSON (save as `kbv2-dashboard.json`):

```json
{
  "dashboard": {
    "title": "KBv2 Crypto Knowledgebase",
    "panels": [
      {
        "title": "Documents Processed",
        "targets": [{
          "expr": "kb_documents_processed_total"
        }]
      },
      {
        "title": "Extraction Quality",
        "targets": [{
          "expr": "kb_extraction_quality_avg"
        }]
      },
      {
        "title": "Experience Bank Hit Rate",
        "targets": [{
          "expr": "kb_experience_bank_hit_rate"
        }]
      },
      {
        "title": "Documents by Domain",
        "targets": [{
          "expr": "kb_documents_by_domain"
        }]
      }
    ]
  }
}
```

## Step 5: Verify Deployment

### 5.1 Test Basic Functionality

```python
import asyncio
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

async def test():
    orchestrator = SelfImprovingOrchestrator()
    await orchestrator.initialize()
    
    # Check Experience Bank
    stats = await orchestrator.get_experience_bank_stats()
    print(f"Experience Bank: {stats}")
    
    # Check health
    from knowledge_base.monitoring.metrics import health_checker
    health = await health_checker.check_health()
    print(f"Health: {health.status}")

asyncio.run(test())
```

### 5.2 Expected Output

```
Experience Bank: {
    'total_experiences': 0,
    'domain_distribution': {},
    'config': {...}
}
Health: healthy
```

## Step 6: Data Pipeline Integration

### 6.1 Provide Connector to Data Engineering Team

The connector interface is at:
```
src/knowledge_base/data_pipeline/connector.py
```

Share these files with your data engineering team:
1. `connector.py` - Data connector interface
2. Integration instructions (run `python connector.py` to see instructions)

### 6.2 Setup Webhook Endpoint (Optional)

If using webhooks for real-time data:

```python
from fastapi import FastAPI, Request
from knowledge_base.data_pipeline.connector import DataPipelineWebhookHandler

app = FastAPI()
webhook_handler = DataPipelineWebhookHandler(secret="your-webhook-secret")

@app.post("/webhook/data")
async def receive_data(request: Request):
    payload = await request.json()
    signature = request.headers.get("X-Webhook-Signature")
    
    response = await webhook_handler.handle_webhook(payload, signature)
    return response.dict()
```

### 6.3 API Endpoints for Data Ingestion

Add to your FastAPI application:

```python
from fastapi import FastAPI
from knowledge_base.data_pipeline.connector import (
    DataIngestionRequest,
    DataIngestionResponse,
    ETFFlowData,
    OnChainMetricData,
    DeFiProtocolData,
)

app = FastAPI()

@app.post("/api/v1/data/ingest", response_model=DataIngestionResponse)
async def ingest_data(request: DataIngestionRequest):
    """Ingest external data into knowledgebase."""
    # Implementation here
    pass

@app.post("/api/v1/data/etf-flows", response_model=DataIngestionResponse)
async def ingest_etf_flows(flows: List[ETFFlowData]):
    """Ingest ETF flow data."""
    pass

@app.post("/api/v1/data/onchain-metrics", response_model=DataIngestionResponse)
async def ingest_onchain_metrics(metrics: List[OnChainMetricData]):
    """Ingest on-chain metrics."""
    pass

@app.post("/api/v1/data/defi", response_model=DataIngestionResponse)
async def ingest_defi(protocols: List[DeFiProtocolData]):
    """Ingest DeFi protocol data."""
    pass
```

## Step 7: Production Monitoring

### 7.1 Key Metrics to Watch

**Critical Metrics:**
- `kb_documents_processed_total` - Documents processed
- `kb_documents_failed_total` - Failed documents
- `kb_extraction_quality_avg` - Average extraction quality
- `kb_validation_score_avg` - Ontology validation score

**Experience Bank Metrics:**
- `kb_experience_bank_stored_total` - Experiences stored
- `kb_experience_bank_hit_rate` - Cache hit rate (target: >60%)

**Performance Metrics:**
- `kb_processing_time_ms` - Document processing time
- `kb_documents_by_domain` - Domain distribution

### 7.2 Alerting Rules

Create `alerts.yml` for Prometheus Alertmanager:

```yaml
groups:
  - name: kbv2
    rules:
      - alert: HighFailureRate
        expr: rate(kb_documents_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "KBv2 document failure rate is high"
      
      - alert: LowExperienceBankHitRate
        expr: kb_experience_bank_hit_rate < 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Experience Bank hit rate is low"
      
      - alert: LowExtractionQuality
        expr: kb_extraction_quality_avg < 0.7
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Extraction quality has dropped"
```

## Step 8: Maintenance

### 8.1 Daily Checks

```bash
# Check health
curl http://localhost:8080/health

# Check metrics
curl http://localhost:8080/stats

# Check database
psql -d knowledge_base -c "SELECT COUNT(*) FROM extraction_experiences;"
```

### 8.2 Weekly Tasks

1. **Review Experience Bank Growth**
   ```python
   stats = await orchestrator.get_experience_bank_stats()
   print(f"Total experiences: {stats['total_experiences']}")
   print(f"Domain distribution: {stats['domain_distribution']}")
   ```

2. **Evolve Prompts**
   ```python
   for domain in ["BITCOIN", "DEFI", "INSTITUTIONAL_CRYPTO"]:
       result = await orchestrator.evolve_prompts(domain, test_docs)
       print(f"{domain}: {result['statistics']['best_variant']}")
   ```

3. **Clean Up Low-Quality Experiences**
   ```python
   await experience_bank.cleanup_low_quality(threshold=0.80)
   ```

### 8.3 Monthly Tasks

1. **Review Metrics Dashboard**
   - Check extraction quality trends
   - Monitor domain coverage
   - Review error rates

2. **Update Ontology Rules**
   - Add new entity types as needed
   - Refine validation rules

3. **Backup Experience Bank**
   ```bash
   pg_dump -t extraction_experiences knowledge_base > experiences_backup.sql
   ```

## Troubleshooting

### Issue: Migration Failed

**Solution:**
```bash
# Check current version
alembic current

# Fix version mismatch
psql -d knowledge_base -c "UPDATE alembic_version SET version_num = '0003_upgrade_to_1024_dimensions';"

# Retry migration
alembic upgrade experience_bank_001
```

### Issue: High Memory Usage

**Solution:**
- Reduce `experience_bank_max_size` (default 10000)
- Reduce `max_concurrent_documents` (default 5)
- Enable garbage collection

### Issue: Low Experience Bank Hit Rate

**Solution:**
- Wait for more documents to be processed (need baseline experiences)
- Lower `min_quality_threshold` temporarily
- Check domain distribution

### Issue: Slow Extraction

**Solution:**
- Check LLM API latency
- Reduce number of few-shot examples (k=3 default)
- Disable ontology validation temporarily

## Summary

Your KBv2 Crypto Knowledgebase is now production-ready with:

✅ **Self-Improvement Features:**
- Experience Bank for few-shot learning
- Prompt Evolution for automated optimization
- Ontology Validation for quality assurance

✅ **Monitoring:**
- Prometheus metrics endpoint
- Health checks
- Grafana dashboard

✅ **Data Pipeline Integration:**
- Connector interface for external pipelines
- Webhook support
- REST API endpoints

✅ **Production Configuration:**
- Environment-based presets
- Performance tuning
- Security settings

## Next Steps

1. **Start processing documents** - Experience Bank will populate automatically
2. **Monitor metrics** - Watch hit rates and quality scores
3. **Connect data pipeline** - Provide connector to data engineering team
4. **Scale up** - Add more workers for parallel processing

## Support

For issues:
1. Check logs: `tail -f logs/kbv2.log`
2. Verify health: `curl http://localhost:8080/health`
3. Review metrics: `curl http://localhost:8080/stats`
4. Check database: Connect to PostgreSQL and verify tables

---

## Appendix A: Production Entry Point (New)

### Using the Production Application

A production-ready entry point is provided at `src/knowledge_base/production.py`:

```bash
# Start the production server
uv run python -m knowledge_base.production

# Or with explicit host/port
PORT=8765 HOST=0.0.0.0 uv run python -m knowledge_base.production
```

### Production App Features

- **Extended Health Check** (`/api/v2/health`): Includes Experience Bank status
- **Statistics Endpoint** (`/api/v2/stats`): Comprehensive metrics with self-improvement data
- **Document Processing** (`/api/v2/documents/process`): Process with self-improving orchestrator
- **Metrics** (`/metrics`): Prometheus-compatible metrics

---

## Appendix B: Deployment Verification (New)

### Automated Verification

Run the comprehensive deployment verification script:

```bash
# Verify all components are ready
uv run python verify_deployment.py
```

This checks:
- Environment variables configured
- All modules importable
- Database connectivity and table structure
- External services (LLM, Embedding APIs)
- Production configuration
- File structure completeness
- Component initialization

Expected output:
```
🎉 ALL CHECKS PASSED - SYSTEM IS PRODUCTION READY!
Checks Passed: 41/41
```

---

## Appendix C: Quick Deployment Checklist

### Pre-Deployment (Before production)

```bash
# 1. Run verification
uv run python verify_deployment.py

# 2. Check all services
./deployment_checklist.sh

# 3. Run tests
uv run pytest tests/ -xvs

# 4. Verify database migrations
alembic current
```

### Deployment Steps

```bash
# 2. Pull latest code
cd /home/muham/development/kbv2
git pull

# 3. Run migrations
uv run alembic upgrade head

# 4. Verify deployment
uv run python verify_deployment.py

# 6. Verify running
curl http://localhost:8765/api/v2/health
```

### Post-Deployment Verification

```bash
# Check all endpoints
curl http://localhost:8765/health
curl http://localhost:8765/api/v2/health
curl http://localhost:8765/metrics
curl http://localhost:8765/api/v2/stats

```

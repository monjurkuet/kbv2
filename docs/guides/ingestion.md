# KBV2 Ingestion Guide

## Overview

KBV2 provides multiple methods for ingesting documents into the knowledge base. This guide covers all ingestion options, features, and best practices.

---

## Quick Start

### Method 1: Direct CLI (Recommended - No Server Needed)

```bash
# Basic ingestion
./ingest_cli.py /path/to/document.md --domain BITCOIN

# With verbose output
./ingest_cli.py /path/to/document.md --domain DEFI --verbose

# Simple script
./ingest.py /path/to/document.md BITCOIN
```

### Method 2: Via Production Server

```bash
# Start server
uv run python -m knowledge_base.production

# Ingest via CLI in server mode
./ingest_cli.py /path/to/document.md --domain BITCOIN --server

# Or via HTTP API
curl -X POST "http://localhost:8765/api/v2/documents/process" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.md", "domain": "BITCOIN"}'
```

### Method 3: Python API

```python
import asyncio
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

async def ingest(file_path: str, domain: str):
    orchestrator = SelfImprovingOrchestrator()
    await orchestrator.initialize()
    
    try:
        document = await orchestrator.process_document(
            file_path=file_path,
            document_name="My Document",
            domain=domain,
        )
        print(f"Ingested: {document.id}")
        print(f"Status: {document.status}")
    finally:
        await orchestrator.close()

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

### Legacy Domains (Backwards Compatibility)
- `TECHNOLOGY` - Software, APIs, ML/AI (non-crypto)
- `FINANCIAL` - Financial reports, markets
- `MEDICAL` - Medical, clinical, research
- `LEGAL` - Law, contracts, litigation
- `SCIENTIFIC` - Academic research, experiments
- `GENERAL` - Mixed content

**Auto-Detection:** Omit `--domain` parameter for automatic domain detection.

---

## Ingestion Pipeline Stages

### 1. Document Creation
- Initialize document record with PENDING status
- Store metadata (file path, MIME type, size)

### 2. Domain Detection
- Keyword-based screening for crypto indicators
- LLM analysis for domain classification
- Two-phase detection: crypto vs non-crypto

### 3. Document Partitioning
- Semantic chunking (default 1536 tokens, 25% overlap)
- Extract titles and structure
- Create chunk records

### 4. Adaptive Analysis
- LLM analyzes document complexity
- Recommends processing strategy:
  - **Simple**: Gleaning mode (~3-5 LLM calls)
  - **Moderate**: Enhanced gleaning (~12-15 LLM calls)
  - **Complex**: Full multi-agent (~25-30 LLM calls)

### 5. Embedding Generation
- Generate 1024-dim vectors using bge-m3 (Ollama)
- Batch processing for performance
- Store in pgvector with IVFFlat indexes

### 6. Entity Extraction

#### Multi-Agent Extractor (GraphMaster-style)
- **Perception Agent**: Initial entity extraction
- **Enhancement Agent**: Improve entity quality
- **Evaluation Agent**: Quality scoring

#### Experience Bank Integration
- Retrieve similar high-quality examples
- Use few-shot prompting
- Store successful extractions (quality ≥ 0.75)

#### Prompt Evolution
- Domain-optimized prompts
- A/B testing framework
- Automatic best variant selection

### 7. Entity Processing
- **Resolution**: Merge duplicate entities with verbatim grounding
- **Typing**: Domain-aware classification
- **Clustering**: Hierarchical Leiden (macro + micro)

### 8. Quality Assurance
- Ontology validation (15+ crypto rules)
- Hallucination detection (LLM-as-Judge)
- Schema validation

### 9. Storage
- Document metadata → PostgreSQL
- Entities/Edges → PostgreSQL
- Embeddings → pgvector (1024 dims)
- High-quality extractions → Experience Bank

---

## Configuration

### Environment Variables

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
```

### Experience Bank Config

```python
ExperienceBankConfig(
    min_quality_threshold=0.75,  # Store extractions ≥ 0.75
    max_storage_size=10000,
    similarity_top_k=3,
    enable_pattern_extraction=True,
)
```

### Prompt Evolution Config

```python
PromptEvolutionConfig(
    num_variants_per_generation=5,
    max_generations=10,
    mutation_temperature=0.7,
    crypto_domains=[
        "BITCOIN", "DEFI", "INSTITUTIONAL_CRYPTO",
        "STABLECOINS", "CRYPTO_REGULATION"
    ],
)
```

---

## Performance

### Processing Time
- **Simple documents**: ~3-5 minutes (news, blogs)
- **Moderate documents**: ~8-12 minutes (reports)
- **Complex documents**: ~15-20 minutes (research papers)

### LLM Calls
- **Gleaning mode**: 3-5 calls per document
- **Enhanced mode**: 12-15 calls per document
- **Multi-agent mode**: 25-30 calls per document

### Expected Output
- **Entities**: 20-50 per document
- **Edges**: 30-80 per document
- **Quality Score**: 0.6-0.95 (target ≥ 0.75)

---

## Supported File Types

- `.md` - Markdown files
- `.txt` - Plain text
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.html` - HTML files
- `.json` - JSON documents

---

## Troubleshooting

### Issue: High LLM Failure Rate

**Symptoms:** Frequent model switching, slow processing

**Solution:**
```bash
# Check available models
curl http://localhost:8087/v1/models

# Verify API connectivity
curl http://localhost:8087/v1/health
```

### Issue: Experience Bank Not Storing

**Symptoms:** No extractions stored in `extraction_experiences` table

**Solution:**
- Quality threshold is 0.75 (not 0.85)
- Check extraction quality scores in logs
- Verify database migration: `alembic current`

### Issue: Embedding Failures

**Symptoms:** Embedding generation errors

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Verify bge-m3 model is loaded
curl http://localhost:11434/api/show?name=bge-m3
```

### Issue: Domain Detection Wrong

**Symptoms:** Documents classified into wrong domain

**Solution:**
- Auto-detection uses keyword screening + LLM
- Manually specify `--domain` for critical documents
- Check domain feedback table for patterns

---

## Querying Ingested Data

```bash
# Check document count
psql -d knowledge_base -c "SELECT COUNT(*) FROM documents;"

# List recent documents
psql -d knowledge_base -c "SELECT id, name, domain, status, created_at FROM documents ORDER BY created_at DESC LIMIT 5;"

# Check entities
psql -d knowledge_base -c "SELECT COUNT(*) FROM entities;"

# Experience Bank stats
psql -d knowledge_base -c "SELECT COUNT(*) FROM extraction_experiences;"
```

---

## Best Practices

1. **Start with auto-detection** - Let the system determine domain
2. **Monitor first few documents** - Review quality scores and extracted entities
3. **Use appropriate domains** - Match content to correct domain
4. **Batch process similar documents** - Builds Experience Bank faster
5. **Regular maintenance** - Run `VACUUM ANALYZE` weekly

---

## Next Steps

1. Ingest your documents using any method above
2. Monitor the Experience Bank population
3. Query the knowledge base via API
4. Review extracted entities for quality
5. Evolve prompts for domain optimization

---

## Related Documentation

- [Deployment Guide](deployment.md)
- [Self-Improvement Features](self_improvement.md)
- [Available Features](available_features.md)
- [API Endpoints](../api/endpoints.md)

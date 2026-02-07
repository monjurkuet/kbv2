# KBv2 Ingestion Pipeline - Feature Report & Usage Guide

## 📊 Feature Status Report

### ✅ ACTIVE Features (Fully Integrated)

| Feature | File | Status | Description |
|---------|------|--------|-------------|
| **Multi-Agent Extractor** | `intelligence/v1/multi_agent_extractor.py` | ✅ ACTIVE | GraphMaster-style 3-phase extraction (Perception→Enhancement→Evaluation) with quality scoring |
| **Gleaning Service** | `ingestion/v1/gleaning_service.py` | ✅ ACTIVE | Adaptive 2-pass fallback extraction for complex documents |
| **Adaptive Ingestion Engine** | `intelligence/v1/adaptive_ingestion_engine.py` | ✅ ACTIVE | Analyzes documents and configures optimal pipeline parameters |
| **Entity Resolution** | `intelligence/v1/resolution_agent.py` | ✅ ACTIVE | Verbatim-grounded entity merging with confidence scoring |
| **Entity Typing** | `intelligence/v1/entity_typing_service.py` | ✅ ACTIVE | Domain-aware classification into 7 entity types |
| **Clustering Service** | `intelligence/v1/clustering_service.py` | ✅ ACTIVE | Leiden algorithm for entity community detection |
| **Hallucination Detection** | `intelligence/v1/hallucination_detector.py` | ✅ ACTIVE | LLM-as-Judge validation with risk classification |
| **Experience Bank** | `intelligence/v1/self_improvement/experience_bank.py` | ✅ ACTIVE | Few-shot learning from high-quality extractions (quality threshold: 0.85) |
| **Prompt Evolution** | `intelligence/v1/self_improvement/prompt_evolution.py` | ✅ ACTIVE | Automated prompt optimization for 5 crypto domains |
| **Ontology Validation** | `intelligence/v1/self_improvement/ontology_validator.py` | ✅ ACTIVE | 15+ crypto-specific validation rules |
| **Domain Detection** | `orchestration/domain_detection_service.py` | ✅ ACTIVE | Automatic domain classification |
| **Document Pipeline** | `orchestration/document_pipeline_service.py` | ✅ ACTIVE | Chunking, embedding, and document storage |
| **Entity Pipeline** | `orchestration/entity_pipeline_service.py` | ✅ ACTIVE | Full entity extraction workflow orchestration |
| **Quality Assurance** | `orchestration/quality_assurance_service.py` | ✅ ACTIVE | Schema validation and review queue management |

### 📦 AVAILABLE but NOT Used in Ingestion

| Feature | File | Status | Notes |
|---------|------|--------|-------|
| **Reranking Pipeline** | `reranking/reranking_pipeline.py` | 📦 AVAILABLE | Search-time feature, not used during ingestion |
| **Cross-Encoder Reranker** | `reranking/cross_encoder.py` | 📦 AVAILABLE | For search result reranking |
| **RRF Fuser** | `reranking/rrf_fuser.py` | 📦 AVAILABLE | Reciprocal Rank Fusion for multi-query |
| **Community Summarizer** | `summaries/community_summaries.py` | 📦 AVAILABLE | Referenced in QA but not actively called |
| **Guided Extractor** | `extraction/guided_extractor.py` | 📦 AVAILABLE | Available but adaptive approach preferred |

### 🔧 Architecture Components

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| **IngestionOrchestrator** | `orchestrator.py` | ✅ ACTIVE | Base ReAct loop orchestrator |
| **SelfImprovingOrchestrator** | `orchestrator_self_improving.py` | ✅ ACTIVE | Extended with self-improvement features |
| **Production App** | `production.py` | ✅ ACTIVE | FastAPI server with all endpoints |
| **MCP Server** | `mcp_server.py` | ✅ ACTIVE | WebSocket protocol server |

---

## 🚀 How to Ingest Documents

### Method 1: Direct CLI (Recommended - No Server Needed)

```bash
cd /home/muham/development/kbv2

# Basic usage
uv run python ingest_cli.py /path/to/document.md --domain BITCOIN

# With verbose output
uv run python ingest_cli.py /path/to/document.md --domain DEFI --verbose

# See all options
uv run python ingest_cli.py --help
```

**Supported Domains:**
- `BITCOIN` - Bitcoin ETFs, mining, treasuries, network metrics
- `DEFI` - DeFi protocols, liquidity pools, yield strategies
- `INSTITUTIONAL_CRYPTO` - ETF issuers, custody, corporate adoption
- `STABLECOINS` - USDC, USDT, backing mechanisms
- `CRYPTO_REGULATION` - SEC, legislation, compliance
- `GENERAL` - Generic entity extraction

### Method 2: Simple Direct Script

```bash
uv run python ingest.py /path/to/document.md BITCOIN
```

### Method 3: Via WebSocket Server (Requires Server Running)

```bash
# Terminal 1: Start server
uv run python -m knowledge_base.production

# Terminal 2: Ingest via WebSocket
uv run python ingest_cli.py /path/to/document.md --domain BITCOIN --server

# Or use old CLI
uv run python -m knowledge_base.clients.cli ingest /path/to/document.md --domain BITCOIN
```

### Method 4: HTTP API (Requires Server Running)

```bash
# Start server first
uv run python -m knowledge_base.production

# Then ingest
curl -X POST "http://localhost:8765/api/v2/documents/process" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.md", "domain": "BITCOIN"}'
```

### Method 5: Python API (Programmatic)

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

## 🔄 Ingestion Pipeline Flow

```
1. Document Input
   ↓
2. Domain Detection (keyword-based classification)
   ↓
3. Adaptive Analysis (LLM determines complexity & approach)
   ↓
4. Document Processing
   - Partitioning into semantic chunks
   - Embedding generation (bge-m3 via Ollama)
   ↓
5. Entity Extraction
   - Primary: Multi-Agent Extractor (GraphMaster-style)
   - Fallback: Gleaning Service (2-pass adaptive)
   - Self-Improvement: Experience Bank few-shot examples
   - Prompt Evolution: Domain-optimized prompts
   ↓
6. Entity Processing
   - Resolution (merge duplicates)
   - Typing (domain-aware classification)
   - Clustering (Leiden communities)
   ↓
7. Quality Assurance
   - Ontology Validation (15+ crypto rules)
   - Hallucination Detection (LLM-as-Judge)
   ↓
8. Storage
   - Document metadata → PostgreSQL
   - Entities/Edges → PostgreSQL
   - Embeddings → pgvector
   - High-quality extractions → Experience Bank
```

---

## 🎯 Supported File Types

- `.md` - Markdown files
- `.txt` - Plain text
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.html` - HTML files
- `.json` - JSON documents

---

## 📈 Recent Fixes Applied

1. ✅ **Filtered gemini-3 models** - No more 429 rate limit errors
2. ✅ **Fixed embedding client** - Now uses Ollama native `/api/embeddings` endpoint
3. ✅ **Improved prompt evolution** - Shorter prompts to avoid token limits
4. ✅ **Better JSON parsing** - Handles markdown code blocks and edge cases
5. ✅ **Unified CLI** - `ingest_cli.py` works in both direct and server modes

---

## 🔍 Querying Ingested Data

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

## 🚀 Next Steps

1. **Ingest your documents** using any of the methods above
2. **Monitor the Experience Bank** - it will improve extraction quality over time
3. **Query the knowledge base** using the query API
4. **Review extracted entities** via the review API for low-confidence items

All self-improvement features are ENABLED and working!

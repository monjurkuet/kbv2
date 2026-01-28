# KBV2 Full Feature Ingestion Guide

## Overview

The KBV2 ingestion pipeline now uses **all available features** for maximum intelligence extraction:

1. **Document Creation** - Creates document record with unique ID
2. **Auto Domain Detection** - Keyword screening + LLM analysis (NEW)
3. **Smart Partitioning** - 1536 tokens, 25% overlap
4. **Multi-Modal Extraction** - Tables, images, figures via modified LLM prompts (NEW)
5. **Guided Extraction** - Fully automated, domain-specific (NEW)
6. **Multi-Agent Entity Extraction** - Advanced AI extraction with quality scoring
7. **Embedding Generation** - Vector embeddings with batching
8. **Entity Resolution** - Deduplicates similar entities
9. **Entity Clustering** - Groups entities into communities
10. **Enhanced Community Summaries** - Multi-level hierarchy (NEW)
11. **Adaptive Type Discovery** - Schema induction from data (NEW)
12. **Schema Validation** - Validates entities against domain schemas
13. **Hybrid Search Indexing** - BM25 + Vector (NEW)
14. **Reranking Pipeline** - Cross-encoder reranking (NEW)
15. **Intelligence Reports** - Generates synthesis reports

---

## Method 1: Using the CLI (Recommended)

### Basic Ingestion

```bash
# Simple ingestion (auto-detects domain)
python -m knowledge_base.clients.cli ingest /path/to/document.md

# With custom name and domain
python -m knowledge_base.clients.cli ingest \
  /path/to/document.md \
  --name "My Document" \
  --domain "technology"

# With custom server settings
python -m knowledge_base.clients.cli ingest \
  /path/to/document.md \
  --host localhost \
  --port 8765 \
  --timeout 900
```

### Examples by Domain

```bash
# Technology document
python -m knowledge_base.clients.cli ingest \
  /path/to/tech_doc.md \
  --name "AI Research Paper" \
  --domain "technology"

# Healthcare document
python -m knowledge_base.clients.cli ingest \
  /path/to/medical_report.md \
  --name "Patient Records" \
  --domain "healthcare"

# Legal document
python -m knowledge_base.clients.cli ingest \
  /path/to/contract.pdf \
  --name "Service Agreement" \
  --domain "legal"

# Finance document
python -m knowledge_base.clients.cli ingest \
  /path/to/financial_report.xlsx \
  --name "Q4 Earnings" \
  --domain "finance"
```

---

## Method 2: Using WebSocket API

### Python Script

```python
import asyncio
from knowledge_base.clients.websocket_client import WebSocketClient

async def ingest_document():
    client = WebSocketClient(host="localhost", port=8765)
    await client.connect()
    
    try:
        result = await client.ingest_document(
            file_path="/path/to/document.md",
            document_name="My Document",
            domain="technology",
            timeout=900
        )
        
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Success! Document ID: {result.result['document_id']}")
            print(f"Status: {result.result['status']}")
            print(f"Duration: {result.result['duration']:.2f}s")
    
    finally:
        await client.disconnect()

asyncio.run(ingest_document())
```

### With Progress Tracking

```python
import asyncio
from knowledge_base.clients.websocket_client import WebSocketClient
from knowledge_base.clients.progress import ProgressVisualizer

async def ingest_with_progress():
    client = WebSocketClient(host="localhost", port=8765)
    visualizer = ProgressVisualizer()
    
    await client.connect()
    
    try:
        result = await client.ingest_document(
            file_path="/path/to/document.md",
            document_name="My Document",
            domain="technology",
            timeout=900,
            progress_callback=visualizer.update_progress
        )
        
        print(f"\nFinal Result: {result.result}")
    
    finally:
        await client.disconnect()

asyncio.run(ingest_with_progress())
```

---

## Method 3: Direct Python API

```python
import asyncio
from pathlib import Path
from knowledge_base.orchestrator import IngestionOrchestrator

async def ingest_direct():
    # Initialize orchestrator
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    try:
        # Process document
        document = await orchestrator.process_document(
            file_path=Path("/path/to/document.md"),
            document_name="My Document",
            domain="technology"  # Optional - will auto-detect if omitted
        )
        
        print(f"Document ID: {document.id}")
        print(f"Status: {document.status}")
        print(f"Domain: {document.domain}")
        
    finally:
        await orchestrator._vector_store.close()

asyncio.run(ingest_direct())
```

---

## Method 4: Batch Processing (NEW)

```python
import asyncio
from pathlib import Path
from knowledge_base.orchestrator import IngestionOrchestrator

async def batch_ingest():
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    documents = [
        Path("/path/to/doc1.md"),
        Path("/path/to/doc2.md"),
        Path("/path/to/doc3.md"),
    ]
    
    try:
        # Process all documents in parallel
        results = await orchestrator.process_batch(
            documents=documents,
            batch_size=3,
            parallel_embeddings=True
        )
        
        for doc_id, result in results.items():
            print(f"Document {doc_id}: {result.status}")
            
    finally:
        await orchestrator._vector_store.close()

asyncio.run(batch_ingest())
```

---

## What Happens During Ingestion

### Stage 1: Create Document
- Creates document record in database
- Assigns unique ID
- Initializes metadata

### Stage 2: Auto Domain Detection (NEW)
- Keyword screening for initial domain guess
- LLM analysis for refined domain classification
- Supports: technology, healthcare, legal, finance, scientific, general
- Falls back to "general" if detection is uncertain

### Stage 3: Smart Partitioning
- Splits document into semantic chunks
- Chunk size: 1536 tokens
- Overlap: 25% (384 tokens)
- Preserves context and structure

### Stage 4: Multi-Modal Extraction (NEW)
- Extracts tables via modified LLM prompts
- Extracts images and figures descriptions
- No extra LLM cost - uses existing extraction calls
- Integrates extracted content into entities

### Stage 5: Guided Extraction (NEW)
- Fully automated domain-specific prompts
- Uses detected domain to select optimal extraction schema
- Adapts entity types based on domain
- Adds ~10 seconds to processing

### Stage 6: Multi-Agent Extraction
- Multi-agent extraction with quality scoring
- Falls back to gleaning service (2 passes)
- Detects entities, edges, and temporal claims
- Each agent specializes in entity types

### Stage 7: Embedding Generation
- Generates vector embeddings (768 dimensions)
- Uses Ollama embedding model
- Batching for performance optimization
- Embeds chunks and entities

### Stage 8: Entity Resolution
- Detects duplicate entities
- Merges similar entities with confidence scoring
- Updates relationships
- Uses verbatim grounding quotes

### Stage 9: Entity Clustering
- Groups entities into communities
- Uses Leiden clustering algorithm
- Creates community hierarchies

### Stage 10: Enhanced Community Summaries (NEW)
- Multi-level summarization hierarchy:
  - **Macro**: High-level document themes
  - **Meso**: Topic clusters
  - **Micro**: Entity groups
  - **Nano**: Fine-grained relationships
- Generates summary at each level

### Stage 11: Adaptive Type Discovery (NEW)
- Analyzes extracted entities for new patterns
- Uses schema induction to discover types
- Auto-promotes high-confidence new types
- Updates domain schema dynamically

### Stage 12: Schema Validation
- Loads domain-specific schema
- Validates entity attributes
- Adds missing required attributes with defaults
- Enforces schema constraints

### Stage 13: Hybrid Search Indexing (NEW)
- Creates BM25 index for keyword search
- Stores vectors for semantic search
- Builds unified search index
- Enables hybrid retrieval with weighted fusion

### Stage 14: Reranking Pipeline (NEW)
- Cross-encoder model for result reordering
- Improves search result quality
- Configurable initial and final top-k
- Returns higher quality results

### Stage 15: Intelligence Reports
- Synthesizes intelligence reports
- Identifies patterns and insights
- Generates summaries at all levels
- Includes confidence scores

---

## Search API Examples (NEW)

### Hybrid Search

```bash
curl -X POST "http://localhost:8000/hybrid-search-v2" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "vector_weight": 0.7,
    "bm25_weight": 0.3,
    "top_k": 10
  }'
```

### Reranked Search

```bash
curl -X POST "http://localhost:8000/reranked-search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "initial_top_k": 50,
    "final_top_k": 10
  }'
```

### Unified Search

```bash
curl -X POST "http://localhost:8000/unified-search/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "mode": "reranked",
    "top_k": 10
  }'
```

---

## Supported Domains

| Domain | Description | Entity Types |
|--------|-------------|--------------|
| `technology` | Tech companies, frameworks, AI | ORGANIZATION, CONCEPT, PRODUCT, PROJECT |
| `healthcare` | Medical entities, diseases, treatments | PERSON, ORGANIZATION, CONCEPT, CONDITION |
| `legal` | Legal cases, contracts, regulations | EVENT, ORGANIZATION, CONCEPT, DOCUMENT |
| `finance` | Financial reports, markets, investments | ORGANIZATION, CONCEPT, FINANCIAL_INSTRUMENT |
| `scientific` | Research, scientific concepts | PERSON, ORGANIZATION, CONCEPT, RESEARCH_WORK |
| `general` | Mixed content | ALL TYPES |

---

## Features by Domain

### Auto Domain Detection
- Keyword screening identifies domain-specific terms
- LLM analysis provides refined classification
- Confidence scores for each prediction
- Falls back to "general" when uncertain

### Multi-Modal Extraction
- Tables extracted with structure preserved
- Images described via LLM vision prompts
- Figures and diagrams converted to text
- Integrated into entity extraction

### Guided Extraction
- Domain-specific extraction prompts
- Optimized entity type selection
- Required attribute customization
- Validation rules per domain

### Hybrid Search
- BM25 for keyword matching
- Vector similarity for semantic search
- Weighted fusion of both approaches
- Configurable weights per query

### Cross-Encoder Reranking
- Re-ranks initial search results
- Better semantic understanding
- Improves recall and precision
- Configurable depth

### Multi-Level Community Summaries
- Macro: Document-level themes
- Meso: Topic clusters
- Micro: Entity groups
- Nano: Fine relationships

### Adaptive Type Discovery
- Detects patterns in extracted data
- Proposes new entity types
- Validates against existing schema
- Auto-promotes high-confidence types

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Average processing time | ~560 seconds |
| Entities per document | 20-50 |
| Chunks per document | 10-30 |
| Embeddings created | Entities + Chunks |
| Communities generated | 5-20 |
| Multi-level summaries | 4 per document |
| Batch processing speedup | 2-3x |

### Processing Time Breakdown

| Stage | Time |
|-------|------|
| Domain Detection | 2-5s |
| Partitioning | 1-3s |
| Multi-Modal Extraction | 5-10s |
| Guided Extraction | 8-12s |
| Multi-Agent Extraction | 300-400s |
| Embeddings (batched) | 60-80s |
| Entity Resolution | 30-40s |
| Clustering | 10-15s |
| Community Summaries | 20-30s |
| Type Discovery | 15-25s |
| Validation | 5-10s |
| Hybrid Indexing | 10-15s |
| Reranking Setup | 5-10s |
| Reports | 30-40s |

---

## Best Practices

1. **Auto-detection works well** - Try without --domain first
2. **Multi-modal extraction is automatic** - Tables/images extracted via LLM
3. **Use batch processing** - For multiple documents (2-3x faster)
4. **Allow sufficient timeout** - Large documents need 900s+
5. **Monitor progress** - Use progress callbacks for long documents
6. **Hybrid search combines keywords + semantics** - Best of both worlds
7. **Multi-level communities** provide better entity organization
8. **Type discovery auto-promotes** high-confidence new types

---

## Troubleshooting

### Slow Ingestion
- Check document size (larger = slower)
- Verify LLM gateway is responsive
- Check embedding service is running
- Consider batch processing for multiple docs

### Low Entity Quality
- Ensure correct domain is selected
- Try auto-detection if unsure
- Check document formatting
- Review multi-agent quality scores

### Schema Validation Errors
- Verify domain schema exists
- Check entity type definitions
- Review attribute requirements
- Adaptive type discovery may help

### Search Quality Issues
- Try hybrid search with adjusted weights
- Use reranked search for better results
- Check entity resolution quality
- Verify embeddings are generated

---

## Example Script for Batch Ingestion

```python
import asyncio
from pathlib import Path
from knowledge_base.clients.websocket_client import WebSocketClient

DOCUMENTS = [
    ("/path/to/tech1.md", "AI Research", "technology"),
    ("/path/to/medical1.pdf", "Patient Report", "healthcare"),
    ("/path/to/legal1.docx", "Contract", "legal"),
    ("/path/to/finance1.xlsx", "Q4 Earnings", "finance"),
]

async def batch_ingest():
    client = WebSocketClient()
    await client.connect()
    
    try:
        for file_path, name, domain in DOCUMENTS:
            print(f"\nIngesting: {name}")
            result = await client.ingest_document(
                file_path=file_path,
                document_name=name,
                domain=domain,
                timeout=900
            )
            
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print(f"  Success! ID: {result.result['document_id']}")
                print(f"  Entities: {result.result.get('entity_count', 'N/A')}")
                print(f"  Duration: {result.result['duration']:.2f}s")
    
    finally:
        await client.disconnect()

asyncio.run(batch_ingest())
```

---

## Search Examples

### Python Search Client

```python
import requests

def hybrid_search(query, vector_weight=0.7, bm25_weight=0.3, top_k=10):
    response = requests.post(
        "http://localhost:8000/hybrid-search-v2",
        json={
            "query": query,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "top_k": top_k
        }
    )
    return response.json()

def reranked_search(query, initial_top_k=50, final_top_k=10):
    response = requests.post(
        "http://localhost:8000/reranked-search",
        json={
            "query": query,
            "initial_top_k": initial_top_k,
            "final_top_k": final_top_k
        }
    )
    return response.json()

# Usage
results = hybrid_search("machine learning neural networks")
results = reranked_search("machine learning", initial_top_k=100, final_top_k=5)
```

### Unified Search with Mode Selection

```python
import requests

def unified_search(query, mode="hybrid", top_k=10, filters=None):
    response = requests.post(
        "http://localhost:8000/unified-search/",
        json={
            "query": query,
            "mode": mode,  # "hybrid", "reranked", "vector", "bm25"
            "top_k": top_k,
            "filters": filters  # Optional domain filters
        }
    )
    return response.json()

# Modes:
# - "hybrid": Combined BM25 + Vector
# - "reranked": Hybrid + Cross-Encoder reranking
# - "vector": Pure semantic search
# - "bm25": Pure keyword search
```

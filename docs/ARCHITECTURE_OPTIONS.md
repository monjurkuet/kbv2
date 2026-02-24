# Architecture Options for External Data Pipeline Integration

## Executive Summary

This document analyzes several architectural approaches for integrating external data pipelines with KBV2 Knowledge Base. Additionally, we evaluate whether a Knowledge Base is the optimal architecture for a Bitcoin trading and analytics system, and propose alternatives.

---

## Part 1: Is Knowledge Base Optimal?

### Research Findings

Based on 2025-2026 research, here's the assessment:

| Approach | Best For | KB Fit |
|----------|----------|--------|
| **Knowledge Graph + Vector** | Complex relationships, semantic search | ✅ Good |
| **Time-Series Database** | OHLCV, price data, metrics | ❌ Poor - not designed for this |
| **Data Warehouse** | Historical analytics, reporting | ❌ Poor - too rigid |
| **Data Lake** | Raw data storage | ❌ Poor - needs structure |
| **Hybrid (KG + Time-Series)** | **Trading & Analytics** | ✅ **Best Fit** |

### The Problem with Single-Approach

A pure knowledge base has limitations for trading systems:

1. **Time-Series Data**: KBV2's PostgreSQL is not optimized for time-series queries (OHLCV, order book)
2. **High-Frequency Data**: Knowledge graphs aren't designed for millisecond-level updates
3. **Analytical Queries**: Aggregation queries (sums, averages, indicators) are inefficient in graph DBs
4. **Freshness**: Market data needs real-time; KB is designed for batch processing

### The Hybrid Approach (Recommended)

Research from 2025 shows **hybrid architectures** are optimal for financial systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                           │
├────────────────────┬─────────────────────┬──────────────────────┤
│  Knowledge Graph   │  Vector Store       │  Time-Series DB     │
│  (Entities,       │  (Semantic Search, │  (OHLCV, OrderBook, │
│   Relationships)  │   Similarity)       │   On-chain Metrics) │
└────────────────────┴─────────────────────┴──────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Unified Query  │
                    │  Layer          │
                    └─────────────────┘
```

**This aligns with KBV2's current design** - you're already building:
- ✅ Graph for entities/relationships (PostgreSQL)
- ✅ Vector for semantic search (pgvector)
- ❌ Missing: Time-Series layer for market data

---

## Part 2: Current Stack Evaluation

### KBV2 Current Architecture

| Component | Technology | Assessment | Recommendation |
|-----------|------------|------------|----------------|
| **Database** | PostgreSQL 16 + pgvector | Good for vectors, OK for graph | Keep |
| **Graph Layer** | Custom (Leiden clustering) | Good | Keep |
| **Vector Search** | pgvector (IVFFlat) | Good for moderate scale | Keep |
| **Embeddings** | bge-m3 (Ollama) | Good, but 2025 alternatives better | Consider upgrade |
| **LLM** | OpenAI-compatible API | Good flexibility | Keep |

### 2025-2026 Stack Improvements

| Component | Current | 2025-2026 Better Alternative | Reason |
|-----------|---------|-------------------------------|--------|
| **Embeddings** | bge-m3 (MTEB: 63.0) | **Qwen3-Embedding-8B** (MTEB: 70.58) - free, open-weight | 12% better MTEB |
| **Embeddings** | bge-m3 | **Voyage-3-large** (MTEB: 66.8) | Domain tuning |
| **Vector Index** | pgvector IVFFlat | **Qdrant** (dedicated) | 10x faster at scale |
| **Graph DB** | PostgreSQL + custom | **ArangoDB** (native multi-model) | Better graph queries |
| **Time-Series** | Not implemented | **TimescaleDB** or **QuestDB** | Purpose-built |

### Recommendation: Keep Most Current Stack

The current stack is **production-ready** for the knowledge base use case. Key findings:

1. **PostgreSQL + pgvector**: Still excellent for <10M vectors
2. **bge-m3**: Still a solid choice - free, self-hosted, good performance
3. **Custom Graph Layer**: Good control, keeps dependencies low

**Only consider changes if:**
- Scaling beyond 10M vectors → Add Qdrant
- Need better graph queries → Add ArangoDB  
- Need time-series → Add TimescaleDB

---

## Part 3: Architecture Options for Pipeline Integration

### Option A: Direct API (Current Baseline)

```
┌─────────────┐         ┌─────────────┐
│  Pipeline 1 │         │  Pipeline 2 │
│  (YouTube)  │         │  (Market)   │
└──────┬──────┘         └──────┬──────┘
       │   POST /api/v1/documents/external
       ▼                        ▼
┌─────────────────────────────────────────┐
│              KBV2 (Port 8000)            │
└─────────────────────────────────────────┘
```

**Best For:** Small number of pipelines, simple integrations

---

### Option B: Message Queue (Event-Driven)

```
┌─────────────┐         ┌─────────────┐
│  Pipeline 1 │         │  Pipeline 2 │
└──────┬──────┘         └──────┬──────┘
       │   Publish Event       │
       ▼                       ▼
┌─────────────────────────────────────┐
│         Redis Stream / RabbitMQ       │
└─────────────────────────────────────┘
              │ Subscribe
              ▼
┌─────────────────────────────────────┐
│              KBV2 Worker             │
└─────────────────────────────────────┘
```

**Best For:** Multiple pipelines, production, reliability needed

---

### Option C: Hybrid Storage (Recommended)

Pipelines own raw data; export standardized entities to KBV2.

```
┌──────────────────────────────────────────────────────┐
│              Pipeline Repositories (Separate)         │
├──────────────────┬──────────────────┬────────────────┤
│  youtube-        │  market-data/    │  trading-books/│
│  pipeline/       │  pipeline/       │  pipeline/     │
│  - MongoDB      │  - TimescaleDB   │  - PostgreSQL  │
└────────┬─────────┴────────┬─────────┴───────┬────────┘
         │                 │                 │
         │    Standardized Export (Entities)        │
         ▼                 ▼                 ▼
┌──────────────────────────────────────────────────────┐
│              KBV2 (Port 8000)                       │
│  - Entities (normalized)                            │
│  - Relationships                                    │
│  - Vector Search                                   │
└──────────────────────────────────────────────────────┘
```

**Best For:** Complex pipelines, independence, production systems

---

## Part 4: Updated Recommended Architecture

### The Complete System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    YOUR COMPLETE TRADING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               KNOWLEDGE BASE (KBV2) - Port 8000                 │   │
│  │  - Documents, Chunks, Entities, Relationships                    │   │
│  │  - Vector Search (pgvector)                                      │   │
│  │  - Graph Analytics (Leiden communities)                         │   │
│  │  - Query API (RAG + Graph queries)                              │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │               TIME-SERIES LAYER - Port 8002 (NEW)               │   │
│  │  - TimescaleDB / QuestDB                                        │   │
│  │  - OHLCV, Order Book, Funding Rates                            │   │
│  │  - On-Chain Metrics (MVRV, RHODL, etc.)                       │   │
│  │  - Market Data API                                             │   │
│  └───────────────────────────┬─────────────────────────────────────┘   │
│                              │                                          │
│  ┌───────────────────────────▼─────────────────────────────────────┐   │
│  │               PIPELINE LAYER (Separate Repos)                   │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │   │
│  │  │ youtube-pipeline│  │market-data-pipe │  │trading-books   │   │   │
│  │  │    (Port 8001) │  │    (Port 8003) │  │   (Port 8004) │   │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │   │
│  │           │                    │                    │             │   │
│  │           │    Unified Export (PipelineContract)    │             │   │
│  │           ▼                    ▼                    ▼             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Repository Structure

| Repo | Purpose | Port |
|------|---------|------|
| **kbv2** | Knowledge Base (entities + graph + vectors) | 8000 |
| **market-data** | Time-series DB + Market APIs | 8002 |
| **youtube-pipeline** | Video analysis pipeline | 8001 |
| **trading-books-pipeline** | Book processing pipeline | 8004 |

### Integration Contract

All pipelines export this standardized format to KBV2:

```python
class PipelineEntity:
    name: str
    entity_type: str      # From KBV2 schema
    confidence: float
    source_pipeline: str  # "youtube", "market_data", "trading_books"
    source_id: str        # Original ID from pipeline
    text_for_embedding: str
    properties: Dict[str, Any]
```

---

## Part 5: Summary

### Stack Decision

| Component | Current | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Database** | PostgreSQL + pgvector | **Keep** | Excellent for current scale |
| **Graph** | Custom + Leiden | **Keep** | Good control |
| **Embeddings** | bge-m3 | **Keep** (or upgrade to Qwen3-Embedding) | Still solid |
| **LLM** | OpenAI-compatible | **Keep** | Flexibility |
| **Time-Series** | Not implemented | **Add TimescaleDB** | Critical for market data |
| **Pipeline Integration** | Direct API | **Option C: Hybrid** | Clean separation |

### Key Takeaways

1. **Knowledge Base IS appropriate** for:
   - Semantic search over content
   - Entity relationships
   - Graph analytics
   - RAG-style queries

2. **Knowledge Base is NOT sufficient** for:
   - Time-series market data (need TimescaleDB)
   - High-frequency updates
   - Heavy analytical queries

3. **The hybrid approach** (KB + Time-Series + Pipelines) is optimal

---

## Files Created

- **`/home/muham/development/kbv2/docs/ARCHITECTURE_OPTIONS.md`** - Full analysis with research citations

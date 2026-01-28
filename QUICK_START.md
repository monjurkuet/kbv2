# KBV2 Quick Start - Full Feature Ingestion

## 🚀 Quick Ingestion Commands

```bash
# Fastest way - CLI with all features enabled
python -m knowledge_base.clients.cli ingest /path/to/document.md \
  --name "My Document" \
  --domain "technology"
```

## 📋 All 15 Pipeline Stages (Auto-Enabled)

1. ✅ Document creation
2. ✅ **Auto Domain Detection** (NEW - keyword + LLM analysis)
3. ✅ Smart partitioning (1536 tokens, 25% overlap)
4. ✅ **Multi-Modal Extraction** (NEW - tables, images, figures via modified LLM prompts)
5. ✅ **Guided Extraction** (NEW - fully automated, domain-specific)
6. ✅ Multi-agent extraction
7. ✅ Embedding generation (with batching for performance)
8. ✅ Entity resolution
9. ✅ Entity clustering
10. ✅ **Enhanced Community Summaries** (NEW - multi-level hierarchy)
11. ✅ **Adaptive Type Discovery** (NEW - schema induction)
12. ✅ Schema validation
13. ✅ Hybrid Search Indexing (NEW - BM25 + Vector)
14. ✅ **Reranking Pipeline** (NEW - cross-encoder)
15. ✅ Intelligence reports

## 🎯 Choose Your Domain (Auto-Detection Available)

```bash
--domain "technology"   # Tech companies, AI, frameworks
--domain "healthcare"   # Medical entities, diseases
--domain "legal"        # Contracts, cases, regulations
--domain "finance"      # Financial reports, markets
--domain "scientific"   # Research, scientific concepts
--domain "general"      # Mixed content

# Or omit --domain for auto-detection
python -m knowledge_base.clients.cli ingest /path/to/document.md
```

## 🔥 What You Get

- **20-50 entities** extracted per document
- **Multi-modal extraction** (tables, images, figures) - NO extra LLM cost
- **Auto domain detection** using keyword screening + LLM analysis
- **Guided extraction** fully automated based on detected domain
- **Cross-domain relationships** detected automatically
- **Refined entity types** with confidence scores
- **Schema-validated** entities with required attributes
- **Vector embeddings** for semantic search
- **BM25 keyword search** for hybrid retrieval
- **Cross-encoder reranking** for improved results
- **Deduplicated** entities
- **Multi-level community clusters** (macro → meso → micro → nano)
- **Adaptive type discovery** with schema induction
- **Intelligence reports** with insights

## ⚡ Performance

- **Time**: ~560 seconds per document
- **Timeout**: Use `--timeout 900` for large docs
- **Progress**: Enable with `--verbose` flag

## 🔍 Search Capabilities

```bash
# Hybrid search (BM25 + Vector)
curl -X POST "http://localhost:8000/hybrid-search-v2" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "vector_weight": 0.5, "bm25_weight": 0.5, "top_k": 10}'

# Reranked search (hybrid + cross-encoder)
curl -X POST "http://localhost:8000/reranked-search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "initial_top_k": 50, "final_top_k": 10}'

# Unified search (all modes)
curl -X POST "http://localhost:8000/unified-search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "mode": "reranked", "top_k": 10}'
```

## 📁 Example Scripts

See `INGESTION_GUIDE.md` for:
- WebSocket API usage
- Progress tracking
- Batch ingestion
- Direct Python API
- Search API examples

## 🆕 New Features in v2

| Feature | Description | Impact |
|---------|-------------|--------|
| **Auto Domain Detection** | Keyword screening + LLM analysis | Better extraction accuracy |
| **Multi-Modal Extraction** | Tables, images, figures via modified prompts | No extra cost |
| **Guided Extraction** | Fully automated, domain-specific prompts | +10s processing |
| **Hybrid Search** | BM25 + Vector with weighted fusion | Better retrieval |
| **Cross-Encoder Reranking** | Improved result ranking | Higher quality results |
| **Multi-Level Summaries** | Macro → Meso → Micro → Nano hierarchies | Better organization |
| **Adaptive Type Discovery** | Schema induction from data | Auto schema evolution |
| **Batch Processing** | Parallel LLM/embedding calls | Performance optimization |

## 💡 Tips

1. **Auto-detection works well** - try without --domain first
2. **Multi-modal extraction is automatic** - tables/images extracted via LLM
3. **Hybrid search combines keywords + semantics** - best of both worlds
4. **Multi-level communities** provide better entity organization
5. **Type discovery auto-promotes** high-confidence new types

## 📖 Full Documentation

See `INGESTION_GUIDE.md` for complete details!

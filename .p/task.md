# KBV2 Knowledge Base System - Comprehensive Update Plan
**Date:** 2026-01-28  
**Version:** 2.1 (Updated - Clarified Guided Extraction)  
**Status:** Ready for Review  

---

## Executive Summary

This document provides a unified, prioritized implementation roadmap for KBV2 that integrates:
1. **Infrastructure improvements** from codebase exploration (chunking, search, performance)
2. **Advanced features** from research analysis (domain detection, type discovery, guided extraction)

### Key Metrics
- **Current System:** 50 Python files, ~18,000 LOC, ~510 seconds processing time
- **Target System:** Enhanced hybrid search, automated features, ~560 seconds processing time
- **Implementation Timeline:** 11 weeks, 36 files modified, 17 new files

### ⚠️ Important Clarification: Guided Extraction

**Guided Extraction is FULLY AUTOMATED by default.** The system:
- Auto-detects domain from document content
- Selects appropriate extraction goals automatically
- Requires NO user input for standard use cases
- User goals are OPTIONAL - only needed for specific, known use cases

---

## Part 1: Current State Analysis

### ✅ Working Components (No Changes Needed)
| Component | Status | Location |
|-----------|--------|----------|
| Document ingestion | ✅ Working | `src/knowledge_base/ingestion/` |
| Partitioning | ✅ Working | `src/knowledge_base/partitioning/` |
| 2-pass gleaning | ✅ Working | `src/knowledge_base/gleaning/` |
| Multi-agent extraction | ✅ Working | `src/knowledge_base/agents/` |
| Vector storage | ✅ Working | `src/knowledge_base/storage/vector.py` |
| Entity resolution | ✅ Working | `src/knowledge_base/entity_resolution/` |
| Hierarchical clustering | ✅ Working | `src/knowledge_base/clustering/` |
| Domain schemas | ✅ Working | `src/knowledge_base/schemas/` |

### ⚠️ Partially Working Components (Needs Fixes)
| Component | Issue | Priority |
|-----------|-------|----------|
| Document search API | Incomplete implementation | HIGH |
| Reciprocal Rank Fusion | Partial implementation | MEDIUM |

### ❌ Missing Components (Need Implementation)
| Component | Priority | Impact |
|-----------|----------|--------|
| Keyword/BM25 search | HIGH | Enables hybrid retrieval |
| Reranking pipeline | HIGH | Improves result quality |
| Batch processing | MEDIUM | Performance optimization |
| Caching layer | MEDIUM | Performance optimization |
| Auto domain detection | HIGH | Research-backed feature |
| Guided extraction | HIGH | Research-backed feature (fully automated) |
| Adaptive type discovery | MEDIUM | Research-backed feature |
| Enhanced community summaries | MEDIUM | Research-backed feature |

---

## Part 2: 2025-2026 Best Practices Gap Analysis

### Best Practice Comparison
| Best Practice | Current State | Target State | Gap |
|--------------|---------------|--------------|-----|
| Chunk size | 512 tokens | 1024-2048 tokens | Upgrade chunking |
| Overlap | Minimal (50) | 20-30% | Increase overlap |
| Embedding dimensions | nomic-embed-text (768) | 1024-3072 dim | Upgrade embeddings |
| Vector index type | IVFFlat | HNSW | Change index type |
| Search type | Vector only | Hybrid (vector + BM25) | Add BM25 |
| Reranking | None | Cross-encoder | Add reranking |
| Table extraction | unstructured basic | pdfplumber | Add pdfplumber |
| Image OCR | None | Tesseract/OCR | Add OCR |
| Processing | Sequential | Batched | Add batching |
| Caching | None | Redis/memory | Add caching |
| Domain detection | Keyword-based | LLM zero-shot | Upgrade detection |
| Type discovery | Manual | Auto-discovery | Add discovery |

### Research-Backed Features (from task.md v1)
1. **GraphMaster Architecture** (arXiv:2504.00711) - Already implemented ✅
2. **LLM-as-Judge Quality** (arXiv:2411.17388) - Already implemented ✅
3. **BANER Boundary-Aware NER** (COLING 2025) - Already implemented ✅

---

## Part 3: Integrated Implementation Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-2)
**Focus:** Core infrastructure fixes and search foundation

#### 1.1 Enhanced Chunking Pipeline
- **Priority:** HIGH | **Effort:** 1 week | **Files:** 3 modified, 1 new
- **Changes:**
  - Increase chunk size from 512 to 1024-2048 tokens
  - Add 20-30% overlap between chunks
  - Implement semantic-aware chunking
  - Create `semantic_chunker.py`
- **Validation:** Unit tests for chunk size, overlap, semantic boundaries

#### 1.2 Hybrid Search Infrastructure (BM25 + Vector)
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 4 modified, 2 new
- **Changes:**
  - Create `bm25_index.py` (using rank-bm25)
  - Create `hybrid_search.py` (weighted fusion)
  - Add HNSW index option to vector storage
  - Update search API
- **Dependencies:** Phase 1.1
- **Validation:** BM25 precision/recall, hybrid fusion tests

#### 1.3 Caching Layer
- **Priority:** MEDIUM | **Effort:** 1 week | **Files:** 2 modified, 1 new
- **Changes:**
  - Create `cache.py` (Redis/memory backend)
  - Cache embedding results (24h TTL)
  - Cache LLM responses (1h TTL)
- **Validation:** Cache hit rate tests, memory usage tests

---

### Phase 2: Search Quality Enhancement (Weeks 3-4)
**Focus:** Improve retrieval quality

#### 2.1 Cross-Encoder Reranking Pipeline
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Changes:**
  - Create `cross_encoder.py` (cross-encoder/ms-marco-MiniLM)
  - Create `reranking_pipeline.py`
  - Enhance Reciprocal Rank Fusion
  - Update search API
- **Dependencies:** Phase 1.2
- **Validation:** NDCG, MAP benchmarks, latency <500ms

#### 2.2 Document Processing Enhancement
- **Priority:** MEDIUM | **Effort:** 2 weeks | **Files:** 4 modified, 1 new
- **Changes:**
  - Create `table_extractor.py` (pdfplumber)
  - Create `ocr_processor.py` (Tesseract)
  - Modify document processor for multi-modal content
- **Validation:** Table extraction accuracy, OCR quality tests

---

### Phase 3: Advanced Features (Weeks 5-7)
**Focus:** Research-backed automation features

#### 3.1 Auto Domain Detection
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Implementation:**
  - Create `domain/detection.py` (LLM zero-shot classification)
  - Create `domain/domain_models.py`
  - Create `domain/ontology_snippets.py`
  - Multi-stage: keyword screening → LLM analysis → confidence calibration
- **Integration:** Between Stage 1 (Create Document) and Stage 2 (Partition)
- **Time Added:** +15 seconds
- **Validation:** Single-domain >95%, multi-domain detection tests

#### 3.2 Guided Extraction Instructions ⚠️ UPDATED
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new

**🔑 KEY POINT: Fully Automated by Default - No User Input Required**

**How It Works:**
```
Document → Auto-Detect Domain → Select Default Goals → Dynamic Prompts → Extraction
```

**Default Goals by Domain:**
```python
DEFAULT_GOALS = {
    "TECHNOLOGY": "Extract software, APIs, frameworks, architectures, dependencies, versions, technical specifications",
    "FINANCIAL": "Extract companies, revenue, investments, market data, financial metrics, transactions, stakeholders",
    "MEDICAL": "Extract diseases, treatments, drugs, symptoms, clinical trials, medical procedures, patient data",
    "LEGAL": "Extract contracts, parties, obligations, clauses, jurisdictions, legal terms, compliance requirements",
    "HEALTHCARE": "Extract patient information, diagnoses, medications, procedures, healthcare providers, insurance",
    "ACADEMIC": "Extract research papers, authors, methodologies, findings, citations, academic institutions",
    "SCIENTIFIC": "Extract experiments, hypotheses, results, data sets, variables, scientific parameters, theories",
    "GENERAL": "Extract people, organizations, events, locations, concepts, relationships, timeline of events"
}
```

**Implementation:**
- Create `extraction/guided_extractor.py` (auto mode + optional user mode)
- Create `extraction/template_registry.py` (domain-specific templates)
- System automatically selects goals based on detected domain
- User goals parameter is OPTIONAL - system works without it

**Integration:** Between Stage 3 (Partition) and Stage 4 (Extract)
**Time Added:** +10 seconds (automated, no user action needed)

**User Scenarios:**
| Scenario | User Action | System Behavior |
|----------|-------------|-----------------|
| Normal processing | NONE | Auto-detect domain, use default goals |
| Specific focus | Provide goals | Override defaults with user goals |
| Mixed document | NONE | Use multi-domain default goals |

**Validation:** Goal interpretation accuracy (when provided), extraction quality tests

#### 3.3 Enhanced Community Summaries
- **Priority:** MEDIUM | **Effort:** 1.5 weeks | **Files:** 2 modified, 1 new
- **Implementation:**
  - Multi-level hierarchy (macro → meso → micro → nano)
  - LLM-generated community names
  - Community embeddings for similarity search
  - Parallel summary generation
- **Integration:** Modifies Stage 7-8 (Clustering + Summaries)
- **Time Added:** +30 seconds (more efficient than task.md)
- **Validation:** Multi-level summary quality, embedding similarity tests

#### 3.4 Adaptive Type Discovery
- **Priority:** MEDIUM | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Implementation:**
  - Create `types/type_discovery.py`
  - Create `types/schema_inducer.py`
  - Create `types/validation_layer.py`
  - LLM proposes types → LLM validates → auto-promote (>0.9) or flag (0.75-0.9)
- **Integration:** Between Stage 9 (Community Summaries) and Stage 10 (Validate)
- **Time Added:** +45 seconds
- **Validation:** Type proposal precision >80%, validation agreement >85%

---

### Phase 4: Performance Optimization (Weeks 8-9)
**Focus:** Speed and efficiency improvements

#### 4.1 Batch Processing Pipeline
- **Priority:** MEDIUM | **Effort:** 1.5 weeks | **Files:** 4 modified, 1 new
- **Changes:**
  - Create `batch_processor.py`
  - Batch LLM calls (5-10x speedup)
  - Batch embedding calls
  - Adaptive batch sizing
- **Validation:** Throughput tests, memory usage, concurrency safety

#### 4.2 Embedding Model Upgrade
- **Priority:** MEDIUM | **Effort:** 1 week | **Files:** 2 modified, 0 new
- **Changes:**
  - Support higher-dimension models (1024-3072 dim)
  - Update embedding pipeline configuration
  - Consider nomic-embed-text-v1.5 or BAAI/bge-large
- **Validation:** Embedding quality tests, storage compatibility

---

### Phase 5: Integration and Testing (Weeks 10-11)
**Focus:** Final integration and quality assurance

#### 5.1 Processing Flow Integration

**Updated 15-Stage Processing Flow:**

```
1. Create Document
         ↓
2. Auto-Detect Domain (NEW - +15s)
   └─ LLM analyzes content → domain + confidence
         ↓
3. Partition Document
         ↓
4. Guided Extraction (NEW - +10s) ⚠️ FULLY AUTOMATED
   └─ System auto-selects goals based on domain
   └─ User input is OPTIONAL
         ↓
5. Embed Content (Enhanced batching)
         ↓
6. Resolve Entities
         ↓
7. Enhanced Clustering (Enhanced - adaptive depth + LLM names + embeddings)
   └─ Multi-level hierarchy
         ↓
8. Community Summaries (Enhanced - +30s)
   └─ Multi-level parallel generation
         ↓
9. Adaptive Type Discovery (NEW - +45s)
   └─ Auto-discover, validate, promote types
         ↓
10. Validate Against Schema
         ↓
11. Hybrid Search Indexing (NEW)
    └─ Vector + BM25 + HNSW
         ↓
12. Reranking Pipeline Setup (NEW)
    └─ Cross-encoder configuration
         ↓
13. Generate Reports
         ↓
14. Update Domain + Finalize
         ↓
15. Cache Updates (NEW)
```

**User Interaction Points:**
```
Stage 1-3: NO USER ACTION (automated)
Stage 4:  NO USER ACTION (system auto-selects goals) ← ⚠️ KEY POINT
Stage 5-15: NO USER ACTION (automated)

User only provides goals if they have SPECIFIC KNOWN REQUIREMENTS
```

#### 5.2 Updated Time Estimates

| Stage | Original | New | Change |
|-------|----------|-----|--------|
| 1-3. Document Creation + Partitioning | ~30s | ~30s | 0s |
| 4. Auto-Detect Domain | - | +15s | NEW |
| 5. Guided Extraction | - | +10s | NEW (fully automated) |
| 5. Embed Content | ~60s | ~45s | -15s (batching) |
| 6. Entity Resolution | ~30s | ~30s | 0s |
| 7. Clustering | ~45s | ~45s | 0s |
| 8. Community Summaries | ~90s | ~120s | +30s |
| 9. Type Discovery | - | +45s | NEW |
| 10. Schema Validation | ~30s | ~30s | 0s |
| 11. Report Generation | ~60s | ~60s | 0s |
| 12. Finalization | ~30s | ~30s | 0s |
| Indexing | ~60s | ~60s | 0s |
| **TOTAL** | **~510s** | **~560s** | **+50s** |

---

## Part 4: Technical Implementation Details

### File Modification Summary

| Phase | Files Modified | New Files | Total |
|-------|----------------|-----------|-------|
| 1. Foundation | 9 | 4 | 13 |
| 2. Search Quality | 7 | 4 | 11 |
| 3. Advanced Features | 11 | 7 | 18 |
| 4. Performance | 6 | 1 | 7 |
| 5. Integration | 3 | 1 | 4 |
| **TOTAL** | **36** | **17** | **53** |

### New File Structure

```
src/knowledge_base/
├── partitioning/
│   └── semantic_chunker.py (Phase 1.1)
├── storage/
│   ├── bm25_index.py (Phase 1.2)
│   └── hybrid_search.py (Phase 1.2)
├── core/
│   └── cache.py (Phase 1.3)
├── reranking/
│   ├── cross_encoder.py (Phase 2.1)
│   └── reranking_pipeline.py (Phase 2.1)
├── processing/
│   ├── table_extractor.py (Phase 2.2)
│   ├── ocr_processor.py (Phase 2.2)
│   └── batch_processor.py (Phase 4.1)
├── domain/
│   ├── detection.py (Phase 3.1)
│   ├── domain_models.py (Phase 3.1)
│   └── ontology_snippets.py (Phase 3.1)
├── extraction/
│   ├── guided_extractor.py (Phase 3.2) ⚠️ AUTO MODE + OPTIONAL USER MODE
│   └── template_registry.py (Phase 3.2)
├── types/
│   ├── type_discovery.py (Phase 3.4)
│   ├── schema_inducer.py (Phase 3.4)
│   └── validation_layer.py (Phase 3.4)
└── api/
    └── unified_search_api.py (Phase 5.2)
```

---

## Part 5: Dependency Graph

```
Phase 1.1 (Chunking) 
    ↓
Phase 1.2 (BM25) ───────────────┐
    ↓                           ↓
Phase 2.1 (Reranking) ←────────┘
    ↓
Phase 3.1 (Domain Detection) ──┐
    ↓                          ↓
Phase 3.2 (Guided Extraction) ←┤ ← FULLY AUTOMATED ⚠️
    ↓                          │
Phase 3.3 (Community Summaries)┤
    ↓                          ↓
Phase 3.4 (Type Discovery) ────┘
    ↓
Phase 4.1 (Batch Processing)
    ↓
Phase 5 (Integration)
```

---

## Part 6: Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM API costs exceed budget | Medium | High | Caching, batching, budget alerts |
| Type discovery generates noise | Medium | High | Strict thresholds, review queue |
| BM25 implementation quality | Low | Medium | Use rank-bm25 library |
| Cross-encoder latency | High | Medium | Batch processing, model optimization |
| Multi-domain detection accuracy | Medium | Medium | Ensemble with keyword screening |
| Chunk size change affects quality | Low | High | A/B testing, gradual rollout |
| Schema evolution breaking changes | Low | High | Versioning, backward compatibility |

---

## Part 7: Key Decision Points

Before proceeding, please confirm:

### 1. Type Discovery Auto-Promotion
**Recommended:** YES - Auto-promote types with >0.9 confidence
- Confidence thresholds:
  - >0.9: Auto-promote to schema
  - 0.75-0.9: Flag for human review
  - <0.75: Ignore (noise)

### 2. Community Hierarchy Depth
**Recommended:** CAP at 4 levels (macro → meso → micro → nano)
- Adaptive based on data complexity
- Avoid unbounded depth growth

### 3. Multi-Domain Handling
**Recommended:** Use all detected domains with proportions
- Primary domain: highest confidence
- Secondary domains: proportional weighting
- Handle mixed-content documents

### 4. Processing Time Tolerance
**Recommended:** Accept 560 seconds (~9.3 minutes)
- Original: 510s
- New features: +100s
- Optimizations: -50s (batching)
- Net: +50s

### 5. Budget Constraints
**Recommended:** Set LLM budget at $500-1000/month
- Additional for production usage
- Implement usage monitoring
- Set per-document cost limits

### 6. Guided Extraction Mode
**Confirmed:** ✅ FULLY AUTOMATED BY DEFAULT
- User provides goals ONLY if specific requirements exist
- System auto-selects goals based on domain
- No manual intervention required for standard use

---

## Part 8: Automation Level Summary

| Feature | Automation Level | User Action Required |
|---------|------------------|---------------------|
| Auto Domain Detection | 100% | NONE |
| Guided Extraction | 100% (default) | NONE |
| Adaptive Type Discovery | 100% | NONE |
| Community Summaries | 100% | NONE |
| Batch Processing | 100% | NONE |
| Caching | 100% | NONE |
| Hybrid Search | 100% | NONE |
| Reranking | 100% | NONE |

**Total Automation Level: 100%** - All features work without human interaction

---

## Part 9: Validation Strategy

### Unit Testing Requirements
- **Chunking:** Size consistency, overlap correctness, semantic boundaries
- **Search:** BM25 precision/recall, hybrid fusion, reranking NDCG/MAP
- **Domain Detection:** Single-domain >95%, multi-domain accuracy
- **Type Discovery:** Proposal precision >80%, validation agreement >85%
- **Guided Extraction:** Auto-goal selection quality, user override functionality

### Integration Testing Requirements
- **E2E Pipeline:** Full document processing, time/resource monitoring
- **API Integration:** Response times, error handling, edge cases
- **Auto-Mode Tests:** Verify system works without user goals provided

### Performance Testing Requirements
- **Latency:** Document <10min, Search <500ms (P95), Reranking <100ms (P95)
- **Throughput:** Documents/hour, concurrent processing, batch efficiency
- **Resources:** Memory usage, API call reduction, storage growth

---

## Part 10: References

### Academic Sources
1. **GraphMaster:** arXiv:2504.00711 - Multi-Agent LLM Orchestration for KG Synthesis
2. **LLM-as-Judge:** arXiv:2411.17388 - KG Quality Evaluation
3. **BANER:** COLING 2025 - Boundary-Aware LLMs for Few-Shot NER
4. **NER Advances:** arXiv:2401.10825v3 - Recent Advances in Named Entity Recognition

### Implementation References
- **unstructured.io** - Document preprocessing
- **rank-bm25** - BM25 implementation
- **sentence-transformers** - Cross-encoder models
- **redis-py** - Caching backend
- **pdfplumber** - Table extraction
- **pytesseract** - OCR processing

---

## Part 11: Approval Required

### Sign-off Checklist

- [ ] Phase 1: Foundation Infrastructure approved
- [ ] Phase 2: Search Quality Enhancement approved
- [ ] Phase 3: Advanced Features approved (including fully automated Guided Extraction)
- [ ] Phase 4: Performance Optimization approved
- [ ] Phase 5: Integration and Testing approved
- [ ] Key Decision Points confirmed
- [ ] Budget constraints acknowledged
- [ ] Timeline (11 weeks) accepted
- [ ] **Confirmed: Guided Extraction is fully automated by default**

### Review Notes
_______________________________________________
_______________________________________________
_______________________________________________

---

**Document Version:** 2.1  
**Generated:** 2026-01-28  
**Status:** Awaiting Approval  
**Next Step:** Begin Phase 1 implementation upon approval

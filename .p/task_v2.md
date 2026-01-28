# KBV2 Knowledge Base System - Comprehensive Update Plan v2
**Date:** 2026-01-28  
**Version:** 3.0 (Updated - No Extra LLM Calls, Modify Existing Prompts)  
**Status:** Ready for Review  

---

## Executive Summary

This document provides a unified, prioritized implementation roadmap for KBV2 that integrates:
1. **Infrastructure improvements** from codebase exploration (chunking, search, performance)
2. **Advanced features** from research analysis (domain detection, type discovery, guided extraction)

### Key Changes from v1 to v2
- ❌ **Removed Caching Layer** - Not needed for this use case
- ❌ **Removed Table/OCR Extraction Libraries** - Handled by LLM via modified prompts
- ✅ **LLM-based Multi-Modal Extraction** - Tables and images extracted via MODIFIED existing LLM calls
- ✅ **NO Extra LLM Calls** - Modify `gleaning_service.py` prompts only
- ✅ **Simplified Architecture** - Fewer components, cleaner design

### Key Metrics
- **Current System:** 50 Python files, ~18,000 LOC, ~510 seconds processing time
- **Target System:** Enhanced hybrid search, automated features, ~560 seconds processing time
- **Implementation Timeline:** 9 weeks, 27 files modified, 12 new files

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
| 2-pass gleaning | ✅ Working | `src/knowledge_base/ingestion/v1/gleaning_service.py` |
| Multi-agent extraction | ✅ Working | `src/knowledge_base/intelligence/v1/multi_agent_extractor.py` |
| Vector storage | ✅ Working | `src/knowledge_base/storage/vector.py` |
| Entity resolution | ✅ Working | `src/knowledge_base/entity_resolution/` |
| Hierarchical clustering | ✅ Working | `src/knowledge_base/clustering/` |
| Domain schemas | ✅ Working | `src/knowledge_base/schemas/` |

### ⚠️ Partially Working Components (Needs Fixes)
| Component | Issue | Priority |
|-----------|-------|----------|
| Document search API | Incomplete implementation | HIGH |
| Reciprocal Rank Fusion | Partial implementation | MEDIUM |

### ❌ Removed from v2 (Not Needed)
| Component | Reason |
|-----------|--------|
| Caching Layer | Not needed for this use case |
| Table Extractor (pdfplumber) | Handled by LLM via modified prompts |
| OCR Processor (Tesseract) | Handled by LLM via modified prompts |
| Multi-Modal Extractor Class | Not needed - modify existing prompts |

### ❌ Missing Components (Need Implementation)
| Component | Priority | Impact |
|-----------|----------|--------|
| Keyword/BM25 search | HIGH | Enables hybrid retrieval |
| Reranking pipeline | HIGH | Improves result quality |
| Batch processing | MEDIUM | Performance optimization |
| Auto domain detection | HIGH | Research-backed feature |
| Guided extraction | HIGH | Research-backed feature (fully automated) |
| Adaptive type discovery | MEDIUM | Research-backed feature |
| Enhanced community summaries | MEDIUM | Research-backed feature |
| Multi-modal extraction in prompts | HIGH | Tables/images via existing LLM calls |

---

## Part 2: Key Architecture Decision - NO Extra LLM Calls

### ⚠️ Critical: Modify Existing LLM Calls, Don't Add New Ones

**Existing LLM Call Location:**
```
File: src/knowledge_base/ingestion/v1/gleaning_service.py
Method: _extract_pass()
Lines: 335-340
Call: await self._gateway.generate_text()
```

This LLM call is made **1-2 times per document** (Pass 1 + optional Pass 2).

**Solution: Modify the system prompt and JSON schema in existing code:**

### Where to Modify in `gleaning_service.py`:

| Location | What to Change |
|----------|----------------|
| Line 349-394 | `_get_discovery_prompt()` - Add table/image extraction instructions |
| Line 360-386 | JSON Schema - Add `tables` and `images_with_text` fields |
| Line 414-441 | `_get_gleaning_prompt()` - Add multi-modal focus |
| Line 472-650 | `_parse_extraction_result()` - Parse new fields |

### Updated JSON Schema for Multi-Modal:

```json
{
  "entities": [...],
  "edges": [...],
  "temporal_claims": [...],
  "tables": [
    {
      "content": "| Header1 | Header2 |...",
      "page_number": 1,
      "description": "Table description"
    }
  ],
  "images_with_text": [
    {
      "description": "Dashboard screenshot",
      "embedded_text": "Metrics visible in image",
      "page_number": 2
    }
  ],
  "information_density": 0.7
}
```

### Impact:
- **Extra LLM calls:** 0 (modify existing prompts)
- **Extra cost:** $0
- **Extra time:** 0 seconds
- **Files to modify:** 1 file (`gleaning_service.py`)

---

## Part 3: 2025-2026 Best Practices Gap Analysis

### Best Practice Comparison
| Best Practice | Current State | Target State | Gap |
|--------------|---------------|--------------|-----|
| Chunk size | 512 tokens | 1024-2048 tokens | Upgrade chunking |
| Overlap | Minimal (50) | 20-30% | Increase overlap |
| Embedding dimensions | nomic-embed-text (768) | 1024-3072 dim | Upgrade embeddings |
| Vector index type | IVFFlat | HNSW | Change index type |
| Search type | Vector only | Hybrid (vector + BM25) | Add BM25 |
| Reranking | None | Cross-encoder | Add reranking |
| Table extraction | unstructured basic | Modified LLM prompts | Update schema |
| Image extraction | unstructured basic | Modified LLM prompts | Update schema |
| Processing | Sequential | Batched | Add batching |
| Domain detection | Keyword-based | LLM zero-shot | Upgrade detection |
| Type discovery | Manual | Auto-discovery | Add discovery |

---

## Part 4: Integrated Implementation Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-2)
**Focus:** Core infrastructure fixes and search foundation

#### 1.1 Enhanced Chunking Pipeline
- **Priority:** HIGH | **Effort:** 1 week | **Files:** 2 modified, 1 new
- **Changes:**
  - Increase chunk size from 512 to 1024-2048 tokens
  - Add 20-30% overlap between chunks
  - Implement semantic-aware chunking
  - Create `semantic_chunker.py`
- **Files Modified:** `chunker.py`, `document_processor.py`
- **New File:** `semantic_chunker.py`

#### 1.2 Hybrid Search Infrastructure (BM25 + Vector)
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 4 modified, 2 new
- **Changes:**
  - Create `bm25_index.py` (using rank-bm25)
  - Create `hybrid_search.py` (weighted fusion)
  - Add HNSW index option to vector storage
  - Update search API
- **Dependencies:** Phase 1.1

#### 1.3 Multi-Modal Extraction via Modified Prompts ⚠️ UPDATED
- **Priority:** HIGH | **Effort:** 0.5 week | **Files:** 1 modified, 0 new
- **Changes:**
  - **NO new files needed**
  - Modify `gleaning_service.py` only
  
**Implementation:**

1. **Modify `_get_discovery_prompt()` (Line 349-394):**
   ```python
   # Add to "Focus on" section:
   Focus on:
   1. Clearly named entities (people, organizations, locations, concepts)
   2. Explicit relationships between entities
   3. Temporal information (dates, times, durations)
   4. TABLES: Extract all tables in markdown format with headers and rows
   5. IMAGES: Describe images and extract any visible text
   6. FIGURES: Describe charts, diagrams, graphs and their data
   ```

2. **Update JSON Schema (Line 360-386):**
   ```json
   {
     "entities": [...],
     "edges": [...],
     "temporal_claims": [...],
     "tables": [{"content": "...", "page_number": 1, "description": "..."}],
     "images_with_text": [{"description": "...", "embedded_text": "...", "page_number": 2}],
     "information_density": 0.7
   }
   ```

3. **Modify `_parse_extraction_result()` (Line 472-650):**
   - Parse `tables` and `images_with_text` from JSON response
   - Add to `ExtractionResult` model

**Impact:**
- ⏱️ Time added: 0 seconds (same LLM call)
- 💰 Cost added: $0 (no new LLM calls)
- 📁 New files: 0

---

### Phase 2: Search Quality Enhancement (Weeks 3-4)

#### 2.1 Cross-Encoder Reranking Pipeline
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Changes:**
  - Create `cross_encoder.py` (cross-encoder/ms-marco-MiniLM)
  - Create `reranking_pipeline.py`
  - Enhance Reciprocal Rank Fusion
  - Update search API
- **Dependencies:** Phase 1.2

---

### Phase 3: Advanced Features (Weeks 5-7)

#### 3.1 Auto Domain Detection
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Implementation:**
  - Create `domain/detection.py` (LLM zero-shot classification)
  - Create `domain/domain_models.py`
  - Create `domain/ontology_snippets.py`
- **Integration:** Between Stage 1 (Create Document) and Stage 2 (Partition)

#### 3.2 Guided Extraction Instructions
- **Priority:** HIGH | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **🔑 KEY POINT: Fully Automated by Default - No User Input Required**
- **Implementation:**
  - Create `extraction/guided_extractor.py` (auto mode + optional user mode)
  - Create `extraction/template_registry.py` (domain-specific templates)
- **Integration:** Between Stage 3 (Partition) and Stage 4 (Extract)

#### 3.3 Enhanced Community Summaries
- **Priority:** MEDIUM | **Effort:** 1.5 weeks | **Files:** 2 modified, 1 new
- **Implementation:**
  - Multi-level hierarchy (macro → meso → micro → nano)
  - LLM-generated community names
  - Community embeddings for similarity search

#### 3.4 Adaptive Type Discovery
- **Priority:** MEDIUM | **Effort:** 2 weeks | **Files:** 3 modified, 2 new
- **Implementation:**
  - Create `types/type_discovery.py`
  - Create `types/schema_inducer.py`
  - Create `types/validation_layer.py`
  - LLM proposes types → LLM validates → auto-promote (>0.9) or flag (0.75-0.9)

---

### Phase 4: Performance Optimization (Weeks 8-9)

#### 4.1 Batch Processing Pipeline
- **Priority:** MEDIUM | **Effort:** 1.5 weeks | **Files:** 4 modified, 1 new
- **Changes:**
  - Create `batch_processor.py`
  - Batch LLM calls (5-10x speedup)
  - Batch embedding calls

#### 4.2 Embedding Model Upgrade
- **Priority:** MEDIUM | **Effort:** 1 week | **Files:** 2 modified, 0 new
- **Changes:**
  - Support higher-dimension models (1024-3072 dim)
  - Update embedding pipeline configuration

---

### Phase 5: Integration and Testing (Weeks 10-11)

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
4. Multi-Modal Extraction (MODIFIED - +0s) ⚠️ NO EXTRA LLM CALL
   └─ Modify gleaning_service.py prompts
   └─ Extract tables/images via existing LLM call
         ↓
5. Guided Extraction (NEW - +10s) ⚠️ FULLY AUTOMATED
   └─ System auto-selects goals based on domain
   └─ User input is OPTIONAL
         ↓
6. Embed Content (Enhanced batching)
         ↓
7. Resolve Entities
         ↓
8. Enhanced Clustering (Enhanced - +30s)
   └─ Multi-level hierarchy
         ↓
9. Community Summaries (Enhanced - +30s)
         ↓
10. Adaptive Type Discovery (NEW - +45s)
         ↓
11. Validate Against Schema
         ↓
12. Hybrid Search Indexing (NEW)
         ↓
13. Reranking Pipeline Setup (NEW)
         ↓
14. Generate Reports
         ↓
15. Update Domain + Finalize
```

#### 5.2 Updated Time Estimates

| Stage | Original | New | Change |
|-------|----------|-----|--------|
| 1-3. Document Creation + Partitioning | ~30s | ~30s | 0s |
| 4. Auto-Detect Domain | - | +15s | NEW |
| 4. Multi-Modal Extraction | - | +0s | MODIFIED (no extra cost) |
| 5. Guided Extraction | - | +10s | NEW |
| 6. Embed Content | ~60s | ~45s | -15s (batching) |
| 7. Entity Resolution | ~30s | ~30s | 0s |
| 8. Clustering | ~45s | ~45s | 0s |
| 9. Community Summaries | ~90s | ~120s | +30s |
| 10. Type Discovery | - | +45s | NEW |
| 11-15. Other stages | ~210s | ~210s | 0s |
| **TOTAL** | **~510s** | **~560s** | **+50s** |

---

## Part 5: Technical Implementation Details

### File Modification Summary (v2)

| Phase | Files Modified | New Files | Change from v1 |
|-------|----------------|-----------|----------------|
| 1. Foundation | 7 | 3 | -2 files |
| 2. Search Quality | 5 | 2 | -2 files |
| 3. Advanced Features | 11 | 7 | 0 |
| 4. Performance | 6 | 1 | 0 |
| 5. Integration | 3 | 1 | 0 |
| **TOTAL** | **32** | **14** | **-5 files, -3 new** |

### New File Structure (v2)

```
src/knowledge_base/
├── partitioning/
│   └── semantic_chunker.py (Phase 1.1)
├── storage/
│   ├── bm25_index.py (Phase 1.2)
│   └── hybrid_search.py (Phase 1.2)
├── extraction/
│   ├── guided_extractor.py (Phase 3.2)
│   └── template_registry.py (Phase 3.2)
├── reranking/
│   ├── cross_encoder.py (Phase 2.1)
│   └── reranking_pipeline.py (Phase 2.1)
├── domain/
│   ├── detection.py (Phase 3.1)
│   ├── domain_models.py (Phase 3.1)
│   └── ontology_snippets.py (Phase 3.1)
├── types/
│   ├── type_discovery.py (Phase 3.4)
│   ├── schema_inducer.py (Phase 3.4)
│   └── validation_layer.py (Phase 3.4)
├── processing/
│   └── batch_processor.py (Phase 4.1)
└── api/
    └── unified_search_api.py (Phase 5.2)
```

### Files MODIFIED (Not Added)
```
src/knowledge_base/ingestion/v1/gleaning_service.py
  - Line 349: _get_discovery_prompt() - add multi-modal instructions
  - Line 360: JSON schema - add tables/images fields
  - Line 472: _parse_extraction_result() - parse new fields
```

---

## Part 6: Key Decision Points

### 1. Multi-Modal Extraction Strategy
**Confirmed:** ✅ Modify existing `gleaning_service.py` prompts
- NO new files for multi-modal extraction
- NO extra LLM calls
- NO extra cost
- NO extra time

### 2. Type Discovery Auto-Promotion
**Recommended:** YES - Auto-promote types with >0.9 confidence

### 3. Community Hierarchy Depth
**Recommended:** CAP at 4 levels (macro → meso → micro → nano)

### 4. Processing Time Tolerance
**Recommended:** Accept 560 seconds (~9.3 minutes)

### 5. Budget Constraints
**Recommended:** Set LLM budget at $300-600/month

---

## Part 7: Summary Comparison

| Metric | v1 | v2 |
|--------|----|----|
| Caching Layer | Phase 1.3 | ❌ Removed |
| Table Extraction | pdfplumber | ✅ Modified prompts |
| OCR Processing | Tesseract | ✅ Modified prompts |
| Multi-Modal Files | 3 new | 0 new (modify 1 file) |
| Extra LLM Calls | 1-2 per doc | 0 (modify existing) |
| Files Modified | 36 | 32 |
| New Files | 17 | 14 |
| Processing Time | ~560s | ~560s |
| Timeline | 11 weeks | 9 weeks |
| Budget | $500-1000/mo | $300-600/mo |

---

## Part 8: Approval Required

### Sign-off Checklist

- [ ] Phase 1: Foundation Infrastructure approved (multi-modal via modified prompts)
- [ ] Phase 2: Search Quality Enhancement approved
- [ ] Phase 3: Advanced Features approved
- [ ] Phase 4: Performance Optimization approved
- [ ] Phase 5: Integration and Testing approved
- [ ] **Confirmed: NO extra LLM calls for multi-modal extraction**
- [ ] **Confirmed: Modify gleaning_service.py only**
- [ ] **Confirmed: Guided Extraction is fully automated**
- [ ] Key Decision Points confirmed
- [ ] Budget constraints acknowledged

---

**Document Version:** 3.0  
**Generated:** 2026-01-28  
**Status:** Awaiting Approval  
**Next Step:** Begin Phase 1 implementation upon approval

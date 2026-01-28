# KBV2 Comprehensive Update Plan v2
## Executive Summary

This document provides a unified, prioritized implementation roadmap for KBV2 that integrates:
1. **Infrastructure improvements** from codebase exploration (chunking, search, performance)
2. **Advanced features** from task.md (domain detection, type discovery, guided extraction)

### Key Changes from v1 to v2
- ❌ **Removed Caching Layer** - Not needed for this use case
- ❌ **Removed Table/OCR Extraction Libraries** - Handled by LLM via MODIFIED existing prompts
- ✅ **LLM-based Multi-Modal Extraction** - Tables and images extracted via EXISTING LLM calls
- ✅ **NO Extra LLM Calls** - Modify `gleaning_service.py` prompts only
- ✅ **Simplified Architecture** - Fewer components, cleaner design

**Current System State:**
- 50 Python files, ~18,000 LOC
- Working: Document ingestion, partitioning, 2-pass gleaning, multi-agent extraction, vector storage, entity resolution, hierarchical clustering, domain schemas
- Partially working: Document search API, Reciprocal Rank Fusion
- Not implemented: Keyword/BM25 search, reranking pipeline, batch processing
- Current processing time: ~510 seconds per document

**Target State After Implementation:**
- Enhanced hybrid search (vector + BM25 + reranking)
- Optimized document processing pipeline
- Automated domain detection and type discovery
- Multi-modal extraction via MODIFIED existing LLM calls (NO extra cost)
- Dynamic extraction guidance (fully automated)
- Estimated processing time: ~560 seconds

**⚠️ Critical Architecture Decision: NO Extra LLM Calls**

Multi-modal extraction (tables, images, figures) is handled by **modifying existing prompts** in `gleaning_service.py`. No new LLM calls, no new files, no extra cost, no extra time.

---

## Current State Analysis

### 1. Working Components (No Changes Needed)

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

### 2. Partially Working Components (Needs Fixes)

| Component | Issue | Priority |
|-----------|-------|----------|
| Document search API | Incomplete implementation | HIGH |
| Reciprocal Rank Fusion | Partial implementation | MEDIUM |

### 3. Missing Components (Need Implementation)

| Component | Priority | Impact |
|-----------|----------|--------|
| Keyword/BM25 search | HIGH | Enables hybrid retrieval |
| Reranking pipeline | HIGH | Improves result quality |
| Batch processing | MEDIUM | Performance optimization |
| Auto domain detection | HIGH | Research-backed feature |
| Guided extraction | HIGH | Research-backed feature (fully automated) |
| Adaptive type discovery | MEDIUM | Research-backed feature |
| Enhanced community summaries | MEDIUM | Research-backed feature |
| Multi-modal extraction | HIGH | Tables/images via MODIFIED existing LLM calls |

### 4. Removed from v2 (Not Needed)

| Component | Reason |
|-----------|--------|
| Caching Layer | Not needed for this use case |
| Table Extractor (pdfplumber) | Handled by LLM via modified prompts |
| OCR Processor (Tesseract) | Handled by LLM via modified prompts |
| Multi-Modal Extractor Class | Not needed - modify existing prompts |

---

## Key Architecture Decision: Modify Existing LLM Calls

### ⚠️ Critical: NO Extra LLM Calls for Multi-Modal Extraction

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

### Implementation Details:

#### 1. Modify `_get_discovery_prompt()` (Line 349)

```python
def _get_discovery_prompt(self) -> str:
    """Get discovery pass system prompt."""
    return """You are an expert information extraction system. Your task is to extract entities and relationships from the provided text.

Focus on:
1. Clearly named entities (people, organizations, locations, concepts)
2. Explicit relationships between entities
3. Temporal information (dates, times, durations)
4. TABLES: Extract all tables in markdown format with headers and rows
5. IMAGES: Describe images and extract any visible text (OCR via LLM analysis)
6. FIGURES: Describe charts, diagrams, graphs and their data

CRITICAL: You must respond with valid JSON only.

Output in the following JSON schema:
{
  "entities": [...],
  "edges": [...],
  "temporal_claims": [...],
  "tables": [
    {
      "content": "| Header1 | Header2 |\\n| --- | --- |\\n| Cell1 | Cell2 |",
      "page_number": 1,
      "description": "Sales data for Q1 2024"
    }
  ],
  "images_with_text": [
    {
      "description": "Dashboard screenshot showing key metrics",
      "embedded_text": "Total Users: 1,234\\nRevenue: $56,789\\nGrowth: 15%",
      "page_number": 2
    }
  ],
  "figures": [
    {
      "type": "bar_chart",
      "description": "Monthly revenue trend for 2024",
      "data_points": [{"month": "Jan", "value": 10000}]
    }
  ],
  "information_density": 0.7
}

Be precise and factual. Only extract information explicitly stated in the text."""
```

#### 2. Update JSON Schema (Line 360)

```json
{
  "entities": [
    {
      "name": "string (entity name)",
      "type": "string (entity type)",
      "description": "string (optional description)",
      "confidence": 0.9
    }
  ],
  "edges": [
    {
      "source": "string (source entity name)",
      "target": "string (target entity name)",
      "type": "string (relationship type)",
      "confidence": 0.9
    }
  ],
  "temporal_claims": [
    {
      "text": "string (temporal text)",
      "type": "atemporal|static|dynamic",
      "date": "string (optional date)"
    }
  ],
  "tables": [
    {
      "content": "| Header1 | Header2 |... (markdown format)",
      "page_number": 1,
      "description": "Brief description of table"
    }
  ],
  "images_with_text": [
    {
      "description": "Description of image content",
      "embedded_text": "All text visible in the image",
      "page_number": 2
    }
  ],
  "figures": [
    {
      "type": "bar_chart|line_graph|pie_chart|diagram|other",
      "description": "Description of figure",
      "data_points": [{"label": "string", "value": 0}]
    }
  ],
  "information_density": 0.7
}
```

#### 3. Modify `_parse_extraction_result()` (Line 472)

```python
class ExtractionResult(BaseModel):
    """Extraction result from a pass."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    temporal_claims: list[TemporalClaim] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)  # NEW
    images_with_text: list[ExtractedImage] = Field(default_factory=list)  # NEW
    figures: list[ExtractedFigure] = Field(default_factory=list)  # NEW
    information_density: float = Field(default=0.0)

# Parse new fields in _parse_extraction_result:
def _parse_extraction_result(self, response: str, text: str) -> ExtractionResult:
    # ... existing code ...
    
    tables = []
    for table_data in data.get("tables", []):
        tables.append(ExtractedTable(
            content=table_data.get("content", ""),
            page_number=table_data.get("page_number"),
            description=table_data.get("description", "")
        ))
    
    images_with_text = []
    for img_data in data.get("images_with_text", []):
        images_with_text.append(ExtractedImage(
            description=img_data.get("description", ""),
            embedded_text=img_data.get("embedded_text", ""),
            page_number=img_data.get("page_number")
        ))
    
    figures = []
    for fig_data in data.get("figures", []):
        figures.append(ExtractedFigure(
            type=fig_data.get("type", "other"),
            description=fig_data.get("description", ""),
            data_points=fig_data.get("data_points", [])
        ))
    
    return ExtractionResult(
        entities=entities,
        edges=edges,
        temporal_claims=temporal_claims,
        tables=tables,
        images_with_text=images_with_text,
        figures=figures,
        information_density=information_density
    )
```

### Impact Summary:
| Metric | Value |
|--------|-------|
| Extra LLM calls | 0 |
| Extra cost | $0 |
| Extra time | 0 seconds |
| Files modified | 1 (`gleaning_service.py`) |
| New files | 0 |

---

## Integrated Feature Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-2)

#### 1.1 Enhanced Chunking Pipeline
**Priority: HIGH | Effort: 1 week | Files to Modify: 2 | New Files: 1**

**Current Implementation:**
- `src/knowledge_base/partitioning/chunker.py` - 512 token chunks

**Required Changes:**

1. **Modify `src/knowledge_base/partitioning/chunker.py`:**
   - Increase default chunk size from 512 to 1024-2048 tokens
   - Add 20-30% overlap between chunks
   - Implement semantic-aware chunking
   - Add chunk metadata

2. **Create `src/knowledge_base/partitioning/semantic_chunker.py`:**
   ```python
   class SemanticChunker:
       def __init__(self, chunk_size: int = 1536, overlap_ratio: float = 0.25):
           self.chunk_size = chunk_size
           self.overlap = int(chunk_size * overlap_ratio)
       
       def chunk(self, document: Document) -> List[Chunk]:
           # Implement semantic-aware chunking
           pass
   ```

**Validation:**
- Unit tests for chunk size consistency
- Overlap verification tests

#### 1.2 Hybrid Search Infrastructure (BM25 + Vector)
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 4 | New Files: 2**

**Required Changes:**

1. **Create `src/knowledge_base/storage/bm25_index.py`:**
   ```python
   class BM25Index:
       def __init__(self, k1: float = 1.5, b: float = 0.75):
           self.k1 = k1
           self.b = b
           self.index = {}
       
       def index_documents(self, documents: List[IndexedDocument]):
           # Build BM25 index
           pass
       
       def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
           # Execute BM25 search
           pass
   ```

2. **Create `src/knowledge_base/storage/hybrid_search.py`:**
   ```python
   class HybridSearchEngine:
       def __init__(self, vector_store, bm25_index):
           self.vector = vector_store
           self.bm25 = bm25_index
       
       def search(
           self,
           query: str,
           vector_weight: float = 0.5,
           bm25_weight: float = 0.5,
           top_k: int = 10
       ) -> List[SearchResult]:
           # Execute parallel vector + BM25 search
           pass
   ```

3. **Modify `src/knowledge_base/api/search_api.py`:**
   - Add hybrid search endpoint
   - Support weight tuning

4. **Modify `src/knowledge_base/storage/vector.py`:**
   - Add HNSW index option

**Dependencies:**
- Requires: Phase 1.1 (enhanced chunking)

#### 1.3 Multi-Modal Extraction via Modified Prompts ⚠️ UPDATED
**Priority: HIGH | Effort: 0.5 week | Files to Modify: 1 | New Files: 0**

**Required Changes (All in `gleaning_service.py`):**

1. **Modify `_get_discovery_prompt()` (Line 349-394):**
   - Add table/image/figure extraction to system prompt
   - Add new fields to JSON schema

2. **Modify `_get_gleaning_prompt()` (Line 414-441):**
   - Add multi-modal focus for second pass

3. **Modify `_parse_extraction_result()` (Line 472-650):**
   - Parse `tables`, `images_with_text`, `figures` from JSON
   - Add to `ExtractionResult` model

**No new files needed. Modify existing code only.**

---

### Phase 2: Search Quality Enhancement (Weeks 3-4)

#### 2.1 Cross-Encoder Reranking Pipeline
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**Required Changes:**

1. **Create `src/knowledge_base/reranking/cross_encoder.py`:**
   ```python
   class CrossEncoderReranker:
       def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM"):
           self.model = load_cross_encoder(model_name)
       
       def rerank(
           self,
           query: str,
           candidates: List[SearchResult],
           top_k: int = 5
       ) -> List[SearchResult]:
           pass
   ```

2. **Create `src/knowledge_base/reranking/reranking_pipeline.py`:**
   ```python
   class RerankingPipeline:
       def __init__(self, hybrid_search, cross_encoder, rr_fuser):
           self.hybrid = hybrid_search
           self.cross_encoder = cross_encoder
           self.rr_fuser = rr_fuser
       
       def search(
           self,
           query: str,
           initial_top_k: int = 50,
           final_top_k: int = 10
       ) -> List[SearchResult]:
           # Step 1: Hybrid search
           candidates = self.hybrid.search(query, top_k=initial_top_k)
           # Step 2: Cross-encoder reranking
           reranked = self.cross_encoder.rerank(query, candidates, top_k=final_top_k)
           return reranked
   ```

3. **Modify `src/knowledge_base/api/search_api.py`:**
   - Update search endpoint to use reranking pipeline

**Dependencies:**
- Requires: Phase 1.2 (hybrid search)

---

### Phase 3: Advanced Features (Weeks 5-7)

#### 3.1 Auto Domain Detection
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**Implementation:**

1. **Create `src/knowledge_base/domain/detection.py`:**
   ```python
   class DomainDetector:
       def __init__(self, llm_client, ontology_snippets: Dict[str, str]):
           self.llm = llm_client
           self.ontology = ontology_snippets
       
       async def detect_domain(
           self,
           document: Document,
           top_k: int = 3
       ) -> List[DomainPrediction]:
           # Stage 1: Fast keyword screening
           keyword_scores = self._keyword_screening(document)
           # Stage 2: Deep LLM analysis
           llm_analysis = await self._llm_analysis(document, candidates=...)
           # Stage 3: Confidence calibration
           predictions = self._calibrate_confidence(llm_analysis)
           return predictions
   ```

2. **Create `src/knowledge_base/domain/domain_models.py`:**
   ```python
   class DomainPrediction(BaseModel):
       domain: str
       confidence: float
       supporting_evidence: List[str]
   
   class DomainDetectionResult(BaseModel):
       primary_domain: str
       all_domains: List[DomainPrediction]
       is_multi_domain: bool
   ```

3. **Modify `src/knowledge_base/ingestion/document_processor.py`:**
   - Add domain detection step

**Integration with Processing Flow:**
- Between Stage 1 (Create Document) and Stage 2 (Partition)
- Estimated time: +15 seconds

#### 3.2 Guided Extraction Instructions
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**🔑 KEY POINT: Fully Automated by Default**

**Implementation:**

1. **Create `src/knowledge_base/extraction/guided_extractor.py`:**
   ```python
   class GuidedExtractor:
       def __init__(self, llm_client, template_registry):
           self.llm = llm_client
           self.templates = template_registry
       
       async def generate_extraction_prompts(
           self,
           document: Document,
           user_goals: Optional[List[str]] = None,  # Optional!
           domain: Optional[str] = None
       ) -> ExtractionPrompts:
           
           # AUTO MODE: Use domain-based default goals
           if user_goals is None:
               detected_domain = domain or self._detect_domain(document)
               goals = self._get_default_goals(detected_domain)
           else:
               goals = await self._interpret_goals(user_goals)
           
           return self._generate_prompts(goals)
   ```

2. **Create `src/knowledge_base/extraction/template_registry.py`:**
   ```python
   DEFAULT_GOALS = {
       "TECHNOLOGY": "Extract software, APIs, frameworks, architectures...",
       "FINANCIAL": "Extract companies, revenue, investments...",
       "MEDICAL": "Extract diseases, treatments, drugs...",
       "GENERAL": "Extract people, organizations, events...",
   }
   ```

**Integration:** Between Stage 3 (Partition) and Stage 4 (Extract)
**Time Added:** +10 seconds (fully automated)

#### 3.3 Enhanced Community Summaries
**Priority: MEDIUM | Effort: 1.5 weeks | Files to Modify: 2 | New Files: 1**

**Implementation:**
- Multi-level hierarchy (macro → meso → micro → nano)
- LLM-generated community names
- Community embeddings for similarity search

#### 3.4 Adaptive Type Discovery
**Priority: MEDIUM | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**Implementation:**
- Create `types/type_discovery.py`
- Create `types/schema_inducer.py`
- Create `types/validation_layer.py`
- Auto-promote types with >0.9 confidence

---

### Phase 4: Performance Optimization (Weeks 8-9)

#### 4.1 Batch Processing Pipeline
**Priority: MEDIUM | Effort: 1.5 weeks | Files to Modify: 4 | New Files: 1**

**Implementation:**
- Create `batch_processor.py`
- Batch LLM calls (5-10x speedup)
- Batch embedding calls

#### 4.2 Embedding Model Upgrade
**Priority: MEDIUM | Effort: 1 week | Files to Modify: 2 | New Files: 0**

**Implementation:**
- Support higher-dimension models (1024-3072 dim)
- Update embedding pipeline configuration

---

### Phase 5: Integration and Testing (Weeks 10-11)

#### 5.1 Processing Flow Integration

**Updated Processing Flow:**

```
1. Create Document
         ↓
2. Auto-Detect Domain (NEW - +15s)
         ↓
3. Partition Document
         ↓
4. Multi-Modal Extraction (MODIFIED - +0s) ⚠️ NO EXTRA LLM CALL
   └─ Modify gleaning_service.py prompts only
         ↓
5. Guided Extraction (NEW - +10s) ⚠️ FULLY AUTOMATED
         ↓
6. Embed Content (Batching: -15s)
         ↓
7. Resolve Entities
         ↓
8. Enhanced Clustering (+30s)
         ↓
9. Community Summaries (+30s)
         ↓
10. Adaptive Type Discovery (+45s)
         ↓
11. Validate Against Schema
         ↓
12. Hybrid Search Indexing
         ↓
13. Reranking Pipeline
         ↓
14. Generate Reports
         ↓
15. Update Domain + Finalize
```

**Time Estimates:**

| Stage | Original | New | Change |
|-------|----------|-----|--------|
| 1-3. Document + Domain + Partition | ~30s | ~45s | +15s |
| 4. Multi-Modal Extraction | - | +0s | 0 |
| 5. Guided Extraction | - | +10s | NEW |
| 6. Embed Content | ~60s | ~45s | -15s |
| 7-9. Clustering + Summaries + Types | ~180s | ~255s | +75s |
| 10-15. Other stages | ~240s | ~240s | 0s |
| **TOTAL** | **~510s** | **~560s** | **+50s** |

#### 5.2 API Integration
- Create unified search API
- Update document API
- Update health checks

---

## Technical Implementation Details

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
  - Line 360: JSON schema - add tables/images/figures fields
  - Line 414: _get_gleaning_prompt() - add multi-modal focus
  - Line 472: _parse_extraction_result() - parse new fields
```

---

## Dependencies and Risks

### Dependency Graph (v2)

```
Phase 1.1 (Chunking) 
    ↓
Phase 1.2 (BM25) ───────────┐
    ↓                        ↓
Phase 2.1 (Reranking) ←─────┤
    ↓                        ↓
Phase 1.3 (Multi-Modal) ←───┤ ← MODIFY existing prompts ⚠️
    ↓                        ↓
Phase 3.1 (Domain Detection) ─┤
    ↓                         ↓
Phase 3.2 (Guided Extraction) ←┤ ← FULLY AUTOMATED ⚠️
    ↓                         │
Phase 3.3 (Community Summaries)┤
    ↓                         ↓
Phase 3.4 (Type Discovery) ────┘
    ↓
Phase 4.1 (Batch Processing)
    ↓
Phase 5 (Integration)
```

### Risk Assessment (v2)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Multi-modal prompt complexity | Low | Medium | Keep prompts simple, test incrementally |
| Type discovery generates noise | Medium | High | Strict thresholds, review queue |
| BM25 implementation quality | Low | Medium | Use rank-bm25 library |
| Cross-encoder latency | High | Medium | Batch processing |
| Chunk size change affects quality | Low | High | A/B testing |

**Changes from v1:**
- ✅ Removed "LLM API costs" risks (no extra calls)
- ✅ Removed "OCR/table extraction" risks (handled by LLM)

---

## Time Estimates Summary (v2)

| Phase | Duration | Effort | Files Modified | New Files |
|-------|----------|--------|----------------|-----------|
| 1. Foundation Infrastructure | 2 weeks | 3 weeks | 7 | 3 |
| 2. Search Quality Enhancement | 2 weeks | 3 weeks | 5 | 2 |
| 3. Advanced Features | 3 weeks | 6 weeks | 11 | 7 |
| 4. Performance Optimization | 2 weeks | 3 weeks | 6 | 1 |
| 5. Integration and Testing | 2 weeks | 2 weeks | 3 | 1 |
| **TOTAL** | **9 weeks** | **17 weeks** | **32** | **14** |

**Resource Requirements:**
- 1 Senior Engineer (full-time)
- LLM API budget: $300-600/month (lower than v1)
- Additional storage: ~30GB for BM25 indexes

---

## Validation Strategy

### Unit Testing Requirements

1. **Chunking Tests:**
   - Verify chunk size consistency
   - Verify overlap correctness

2. **Search Tests:**
   - BM25 precision/recall benchmarks
   - Hybrid search fusion accuracy
   - Reranking quality (NDCG, MAP)

3. **Multi-Modal Extraction Tests:**
   - Table structure validation
   - Image text extraction accuracy
   - Verify same LLM response structure

4. **Domain Detection Tests:**
   - Single-domain >95%
   - Multi-domain detection accuracy

### Integration Testing Requirements

1. **End-to-End Processing:**
   - Full document processing pipeline
   - Verify no extra LLM calls
   - Output quality verification

2. **API Integration:**
   - Search API response times
   - Domain detection in API responses

---

## Automation Level Summary (v2)

| Feature | Automation Level | User Action Required |
|---------|------------------|---------------------|
| Multi-Modal Extraction | 100% | NONE (modify existing prompts) |
| Auto Domain Detection | 100% | NONE |
| Guided Extraction | 100% (default) | NONE |
| Adaptive Type Discovery | 100% | NONE |
| Community Summaries | 100% | NONE |
| Batch Processing | 100% | NONE |
| Hybrid Search | 100% | NONE |
| Reranking | 100% | NONE |

**Total Automation Level: 100%** - All features work without human interaction

---

## References

### Academic Sources

1. "GraphMaster: Multi-Agent LLM Orchestration for KG Synthesis" - arXiv:2504.00711
2. "LLM-as-Judge for KG Quality" - arXiv:2411.17388
3. "BANER: Boundary-Aware LLMs for Few-Shot NER" - COLING 2025

### Implementation References

1. unstructured.io - Document preprocessing (minimal use)
2. rank-bm25 - BM25 implementation
3. sentence-transformers - Cross-encoder models

---

*Document Generated: 2026-01-28*
*Version: 2.0 (Updated - NO Extra LLM Calls, Modify Existing Prompts)*
*Status: Implementation Ready*

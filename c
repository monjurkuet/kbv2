# KBV2 Comprehensive Update Plan
## Executive Summary

This document provides a unified, prioritized implementation roadmap for KBV2 that merges:
1. **Infrastructure improvements** from codebase exploration (chunking, search, performance)
2. **Advanced features** from task.md (domain detection, type discovery, guided extraction)

The combined plan addresses critical gaps in the current system while introducing research-backed advanced capabilities, targeting a balanced approach between foundational improvements and innovative features.

**Current System State:**
- 50 Python files, ~18,000 LOC
- Working: Document ingestion, partitioning, 2-pass gleaning, multi-agent extraction, vector storage, entity resolution, hierarchical clustering, domain schemas
- Partially working: Document search API, Reciprocal Rank Fusion
- Not implemented: Keyword/BM25 search, reranking pipeline, batch processing, caching
- Current processing time: ~510 seconds per document

**Target State After Implementation:**
- Enhanced hybrid search (vector + BM25 + reranking)
- Optimized document processing pipeline
- Automated domain detection and type discovery
- Dynamic extraction guidance (fully automated)
- Estimated processing time: ~580-620 seconds (with significant quality improvements)

**⚠️ Important Clarification: Guided Extraction**
Guided Extraction is **FULLY AUTOMATED by default.** The system:
- Auto-detects domain from document content
- Selects appropriate extraction goals automatically
- Requires NO user input for standard use cases
- User goals are OPTIONAL - only needed for specific, known use cases

---

## Current State Analysis

### 1. Working Components (No Changes Needed)

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
| Caching layer | MEDIUM | Performance optimization |
| Auto domain detection | HIGH | Research-backed feature |
| Guided extraction | HIGH | Research-backed feature (fully automated) |
| Adaptive type discovery | MEDIUM | Research-backed feature |
| Enhanced community summaries | MEDIUM | Research-backed feature |

### 4. 2025-2026 Best Practices Gap Analysis

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

---

## Integrated Feature Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-2)

#### 1.1 Enhanced Chunking Pipeline
**Priority: HIGH | Effort: 1 week | Files to Modify: 3 | New Files: 1**

**Current Implementation:**
- `src/knowledge_base/partitioning/chunker.py` - 512 token chunks

**Required Changes:**

1. **Modify `src/knowledge_base/partitioning/chunker.py`:**
   - Increase default chunk size from 512 to 1024-2048 tokens
   - Add 20-30% overlap between chunks
   - Implement semantic-aware chunking (respect paragraph/section boundaries)
   - Add chunk metadata: source_position, chunk_index, overlap_info

2. **Create `src/knowledge_base/partitioning/semantic_chunker.py`:**
   ```python
   class SemanticChunker:
       def __init__(self, chunk_size: int = 1536, overlap_ratio: float = 0.25):
           self.chunk_size = chunk_size
           self.overlap = int(chunk_size * overlap_ratio)
       
       def chunk(self, document: Document) -> List[Chunk]:
           # Implement semantic-aware chunking
           # Respect document structure
           # Apply overlap
           pass
   ```

3. **Modify `src/knowledge_base/ingestion/document_processor.py`:**
   - Update to use new chunking configuration
   - Pass chunk metadata to embedding pipeline

**Validation:**
- Unit tests for chunk size consistency
- Overlap verification tests
- Semantic boundary tests

#### 1.2 Hybrid Search Infrastructure (BM25 + Vector)
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 4 | New Files: 2**

**Current State:**
- Vector search only via `src/knowledge_base/storage/vector.py`
- BM25 not implemented

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
           # Fuse results using weighted scoring
           pass
   ```

3. **Modify `src/knowledge_base/api/search_api.py`:**
   - Add hybrid search endpoint
   - Support weight tuning via query parameters
   - Add search analytics logging

4. **Modify `src/knowledge_base/storage/vector.py`:**
   - Add HNSW index option (current: IVFFlat)
   - Add index type configuration

**Dependencies:**
- Requires: Phase 1.1 (enhanced chunking)
- Enables: Phase 2.2 (reranking pipeline)

**Validation:**
- BM25 retrieval tests (precision/recall)
- Hybrid search fusion tests
- HNSW vs IVFFlat comparison tests

#### 1.3 Caching Layer
**Priority: MEDIUM | Effort: 1 week | Files to Modify: 2 | New Files: 1**

**Required Changes:**

1. **Create `src/knowledge_base/core/cache.py`:**
   ```python
   from functools import lru_cache
   from redis import Redis
   
   class CacheManager:
       def __init__(self, backend: str = "memory", ttl: int = 3600):
           self.backend = backend
           self.ttl = ttl
           self._redis = None if backend == "memory" else Redis()
       
       def get(self, key: str) -> Optional[Any]:
           pass
       
       def set(self, key: str, value: Any):
           pass
   ```

2. **Modify embedding pipeline to cache results:**
   - Cache: `embedding(text)` → `vector`
   - Key: `hash(text) + model_name`
   - TTL: 24 hours

3. **Modify LLM calls to cache responses:**
   - Cache: `llm_response(prompt)` → `response`
   - Key: `hash(prompt + temperature + model)`
   - TTL: 1 hour

**Validation:**
- Cache hit rate tests
- Memory usage tests
- TTL expiration tests

---

### Phase 2: Search Quality Enhancement (Weeks 3-4)

#### 2.1 Cross-Encoder Reranking Pipeline
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**Current State:**
- Reciprocal Rank Fusion partially implemented

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
           # Score query-document pairs
           # Re-rank by cross-encoder scores
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
           # Step 1: Hybrid search (vector + BM25)
           candidates = self.hybrid.search(query, top_k=initial_top_k)
           
           # Step 2: Cross-encoder reranking
           reranked = self.cross_encoder.rerank(query, candidates, top_k=final_top_k)
           
           return reranked
   ```

3. **Modify `src/knowledge_base/api/search_api.py`:**
   - Update search endpoint to use reranking pipeline
   - Add configuration for reranking parameters
   - Add latency tracking for reranking step

4. **Enhance Reciprocal Rank Fusion:**
   - Fix partial implementation
   - Fuse cross-encoder scores with RRF

**Dependencies:**
- Requires: Phase 1.2 (hybrid search)
- Enables: Phase 3.1 (guided extraction)

**Validation:**
- Reranking quality tests (NDCG, MAP)
- Latency tests (target: <500ms for 100 candidates)
- Model comparison tests (different cross-encoders)

#### 2.2 Document Processing Enhancement
**Priority: MEDIUM | Effort: 2 weeks | Files to Modify: 4 | New Files: 1**

**Current State:**
- Uses `unstructured` library for PDF parsing (basic)

**Required Changes:**

1. **Create `src/knowledge_base/processing/table_extractor.py`:**
   ```python
   class TableExtractor:
       def __init__(self):
           self.pdfplumber_client = PdfPlumberExtractor()
       
       def extract_tables(self, document: Document) -> List[ExtractedTable]:
           # Detect tables in PDF pages
           # Extract table structure (rows, columns)
           # Convert to markdown/text representation
           pass
   ```

2. **Modify `src/knowledge_base/ingestion/document_processor.py`:**
   - Add table extraction pipeline
   - Add image OCR pipeline
   - Route different content types to appropriate processors

3. **Create `src/knowledge_base/processing/ocr_processor.py`:**
   ```python
   class OCRProcessor:
       def __init__(self):
           self.tesseract = TesseractOCR()
       
       def extract_text_from_images(self, document: Document) -> str:
           # Detect images in document
           # Apply OCR to each image
           # Aggregate extracted text
           pass
   ```

4. **Modify partitioning to handle multi-modal content:**
   - Preserve table structures in chunks
   - Add image text to relevant chunks
   - Track content type metadata

**Dependencies:**
- None (standalone enhancement)

**Validation:**
- Table extraction accuracy tests
- OCR accuracy tests (comparison with ground truth)
- Processing time tests

---

### Phase 3: Advanced Features (Weeks 5-7)

#### 3.1 Auto Domain Detection
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**From task.md - Implementation Details:**

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
           
           # Stage 2: Deep LLM analysis of top candidates
           llm_analysis = await self._llm_analysis(
               document,
               candidates=list(keyword_scores.keys())[:top_k]
           )
           
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
       sub_domains: Dict[str, float] = {}
   
   class DomainDetectionResult(BaseModel):
       primary_domain: str
       all_domains: List[DomainPrediction]
       is_multi_domain: bool
       domain_proportions: Dict[str, float]
   ```

3. **Modify `src/knowledge_base/ingestion/document_processor.py`:**
   - Add domain detection step after document creation
   - Pass domain information to extraction pipeline

4. **Create `src/knowledge_base/domain/ontology_snippets.py`:**
   - Domain-specific prompt snippets for better accuracy
   - Support mixed-domain detection

**Integration with Processing Flow:**
- Inserts between Stage 1 (Create Document) and Stage 2 (Partition)
- Estimated time: +15 seconds

**Validation:**
- Domain accuracy tests (comparison with labeled data)
- Multi-domain detection tests
- Confidence calibration tests

#### 3.2 Guided Extraction Instructions ⚠️ UPDATED
**Priority: HIGH | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

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
               detected_domain = domain or self._detect_domain_from_document(document)
               goals = self._get_default_goals_for_domain(detected_domain)
           else:
               # USER MODE: Use provided goals (override)
               goals = await self._interpret_goals(user_goals, domain)
           
           # Generate dynamic prompts from goals
           prompts = self._generate_prompts(goals, document)
           
           return ExtractionPrompts(
               main_prompt=prompts["main"],
               perspective_prompts=prompts["perspectives"],
               validation_prompt=prompts["validation"]
           )
   ```

2. **Create `src/knowledge_base/extraction/template_registry.py`:**
   ```python
   class ExtractionTemplateRegistry:
       def __init__(self):
           self.templates = {
               "companies_products": Template(...),
               "timeline_events": Template(...),
               "technical_concepts": Template(...),
               "people_relationships": Template(...),
               # Domain-specific default templates
               "TECHNOLOGY_default": Template(...),
               "FINANCIAL_default": Template(...),
               "MEDICAL_default": Template(...),
               "GENERAL_default": Template(...),
           }
       
       def get_template(self, objective: str, domain: str) -> Template:
           pass
       
       def get_default_template(self, domain: str) -> Template:
           """Get default template for domain (no user goals)"""
           return self.templates.get(f"{domain}_default", self.templates["GENERAL_default"])
   ```

3. **Modify extraction pipeline:**
   - Update `src/knowledge_base/extraction/entity_extractor.py`
   - Add guided extraction mode (auto + optional user)
   - Support multi-perspective extraction

**User Scenarios:**
| Scenario | User Action | System Behavior |
|----------|-------------|-----------------|
| Normal processing | NONE | Auto-detect domain, use default goals |
| Specific focus | Provide goals | Override defaults with user goals |
| Mixed document | NONE | Use multi-domain default goals |

**Integration with Processing Flow:**
- Inserts between Stage 3 (Partition) and Stage 4 (Extract)
- Estimated time: +10 seconds (fully automated)

**Dependencies:**
- Requires: Phase 3.1 (Auto Domain Detection)

**Validation:**
- Goal interpretation accuracy tests (when provided)
- Auto-goal selection quality tests
- Multi-perspective extraction tests

#### 3.3 Enhanced Community Summaries
**Priority: MEDIUM | Effort: 1.5 weeks | Files to Modify: 2 | New Files: 1**

**From task.md - Implementation Details:**

1. **Enhance `src/knowledge_base/clustering/community_summaries.py`:**
   ```python
   class EnhancedCommunitySummaries:
       def __init__(self, llm_client):
           self.llm = llm_client
       
       async def generate_multi_level_summaries(
           self,
           communities: List[Community],
           max_depth: int = 4
       ) -> HierarchicalSummary:
           # Generate summaries at multiple levels:
           # Level 0: Macro (all communities)
           # Level 1: Meso (community groups)
           # Level 2: Micro (individual communities)
           # Level 3: Nano (sub-communities, adaptive)
           
           # Parallel generation for efficiency
           pass
       
       async def generate_community_embeddings(
           self,
           summaries: List[CommunitySummary]
       ) -> List[CommunityEmbedding]:
           # Generate embeddings for similarity search
           pass
   ```

2. **Modify `src/knowledge_base/clustering/hierarchical_clustering.py`:**
   - Add adaptive depth detection
   - Add LLM-generated community names
   - Add embedding generation for communities

3. **Enhance similarity search:**
   - Add community-level retrieval
   - Support hierarchical query expansion

**Integration with Processing Flow:**
- Modifies Stage 7 (Enhanced Clustering) and Stage 8 (Community Summaries)
- Estimated time: +30 seconds (from +30s in task.md, more efficient parallelization)

**Dependencies:**
- None (standalone enhancement)

**Validation:**
- Multi-level summary quality tests
- Embedding similarity tests
- Hierarchical retrieval tests

#### 3.4 Adaptive Type Discovery
**Priority: MEDIUM | Effort: 2 weeks | Files to Modify: 3 | New Files: 2**

**From task.md - Implementation Details:**

1. **Create `src/knowledge_base/types/type_discovery.py`:**
   ```python
   class AdaptiveTypeDiscovery:
       def __init__(self, llm_client, schema_manager):
           self.llm = llm_client
           self.schema = schema_manager
       
       async def discover_new_types(
           self,
           entities: List[Entity],
           schema: DomainSchema
       ) -> TypeDiscoveryResult:
           # Stage 1: Identify uncertain entities
           uncertain = self._find_uncertain_entities(entities)
           
           # Stage 2: LLM proposes new types from patterns
           proposals = await self._propose_types(uncertain, schema)
           
           # Stage 3: Independent LLM validation
           validated = await self._llm_to_llm_validation(proposals)
           
           # Stage 4: Auto-promote or flag based on confidence
           result = self._apply_thresholds(validated)
           
           return result
   ```

2. **Create `src/knowledge_base/types/schema_inducer.py`:**
   ```python
   class SchemaInducer:
       async def induce_schema_from_entities(
           self,
           entities: List[Entity]
       ) -> InducedSchema:
           # Analyze entity patterns
           # Propose type hierarchy
           # Suggest attribute definitions
           pass
   ```

3. **Modify schema management:**
   - Update `src/knowledge_base/schemas/schema_manager.py`
   - Add type promotion logic
   - Add confidence threshold configuration

4. **Create `src/knowledge_base/types/validation_layer.py`:**
   - LLM-to-LLM validation logic
   - Confidence calibration
   - Review queue management

**Confidence Thresholds:**
- >0.9: Auto-promote to schema
- 0.75-0.9: Flag for later review
- <0.75: Ignore (noise)

**Integration with Processing Flow:**
- Inserts between Stage 9 (Community Summaries) and Stage 10 (Validate)
- Estimated time: +45 seconds

**Dependencies:**
- None (standalone enhancement)

**Validation:**
- Type proposal accuracy tests
- Validation agreement tests
- Auto-promotion precision tests

---

### Phase 4: Performance Optimization (Weeks 8-9)

#### 4.1 Batch Processing Pipeline
**Priority: MEDIUM | Effort: 1.5 weeks | Files to Modify: 4 | New Files: 1**

**Required Changes:**

1. **Create `src/knowledge_base/processing/batch_processor.py`:**
   ```python
   class BatchProcessor:
       def __init__(self, max_batch_size: int = 10, max_concurrent: int = 5):
           self.max_batch_size = max_batch_size
           self.semaphore = asyncio.Semaphore(max_concurrent)
       
       async def process_batch(
           self,
           documents: List[Document],
           processing_fn: Callable
       ) -> List[ProcessingResult]:
           # Batch LLM calls
           # Batch embedding calls
           # Parallel processing with concurrency limits
           pass
   ```

2. **Modify embedding pipeline:**
   - Batch embed calls (current: sequential)
   - Target: 5-10x speedup for large documents

3. **Modify extraction pipeline:**
   - Batch entity extraction
   - Parallel agent execution

4. **Add batch size configuration:**
   - Environment variables for tuning
   - Adaptive batch sizing based on document complexity

**Dependencies:**
- None (standalone optimization)

**Validation:**
- Batch processing throughput tests
- Memory usage tests
- Concurrency safety tests

#### 4.2 Embedding Model Upgrade
**Priority: MEDIUM | Effort: 1 week | Files to Modify: 2 | New Files: 0**

**Required Changes:**

1. **Modify `src/knowledge_base/embeddings/embedding_pipeline.py`:**
   - Support higher-dimension embedding models (1024-3072 dim)
   - Add model configuration options
   - Support nomic-embed-text and alternatives

2. **Update `config/embedding_config.yaml`:**
   ```yaml
   embedding:
     primary_model: "nomic-embed-text"
     dimensions: 768  # Current
     # Target configurations:
     # - nomic-embed-text-v1.5: 768 dim
     # - BAAI/bge-large-en-v1.5: 1024 dim
     # - sentence-transformers/all-mpnet-base-v2: 768 dim
     # - voyage-large-2: 1024 dim
   ```

3. **Modify vector storage:**
   - Support variable embedding dimensions
   - Update index configuration for higher dimensions

**Dependencies:**
- None (standalone enhancement)

**Validation:**
- Embedding quality tests
- Storage compatibility tests
- Performance tests

---

### Phase 5: Integration and Testing (Weeks 10-11)

#### 5.1 Processing Flow Integration
**Priority: HIGH | Effort: 1 week | Files to Modify: 1 | New Files: 0**

**Updated Processing Flow (15 stages):**

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

**Updated Time Estimates:**

| Stage | Original Time | New Time | Change |
|-------|---------------|----------|--------|
| 1-3. Document Creation + Partitioning | ~30s | ~30s | 0s |
| 4. Auto-Detect Domain | +15s | +15s | NEW |
| 5. Guided Extraction | +10s | +10s | NEW (fully automated) |
| 5. Embed Content | ~60s | ~45s | -15s (batching) |
| 6. Entity Resolution | ~30s | ~30s | 0s |
| 7. Clustering | ~45s | ~45s | 0s |
| 8. Community Summaries | ~90s | ~120s | +30s |
| 9. Type Discovery | +45s | +45s | NEW |
| 10. Schema Validation | ~30s | ~30s | 0s |
| 11. Report Generation | ~60s | ~60s | 0s |
| 12. Finalization | ~30s | ~30s | 0s |
| Indexing | ~60s | ~60s | 0s |
| **TOTAL** | **~510s** | **~560s** | **+50s** |

#### 5.2 API Integration
**Priority: HIGH | Effort: 1 week | Files to Modify: 3 | New Files: 1**

**Required Changes:**

1. **Create unified search API:**
   - Combine vector, BM25, and reranking
   - Support all configuration options
   - Add search analytics

2. **Update document API:**
   - Add domain detection results
   - Add type discovery results
   - Add extraction guidance configuration

3. **Update health checks:**
   - Monitor all new components
   - Add performance metrics

---

## Technical Implementation Details

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

### Configuration Changes

**New `config/embedding_config.yaml`:**
```yaml
embedding:
  model: "nomic-embed-text-v1.5"
  dimensions: 768
  batch_size: 32
  normalize: true

reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  batch_size: 64
  device: "auto"

cache:
  enabled: true
  backend: "memory"  # or "redis"
  ttl:
    embeddings: 86400  # 24 hours
    llm_responses: 3600  # 1 hour
    search_results: 300  # 5 minutes

processing:
  chunk_size: 1536
  chunk_overlap: 0.25
  max_concurrent_embeddings: 5
  max_concurrent_llm: 3
```

---

## Dependencies and Risks

### Dependency Graph

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

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLM API costs exceed budget | Medium | High | Implement caching, batch processing, budget alerts |
| Type discovery generates noise | Medium | High | Strict confidence thresholds, review queue |
| BM25 implementation quality | Low | Medium | Test against established libraries (Rank-BM25) |
| Cross-encoder latency | High | Medium | Batch processing, model optimization |
| Multi-domain detection accuracy | Medium | Medium | Ensemble with keyword screening |
| Chunk size change affects quality | Low | High | A/B testing, gradual rollout |
| Schema evolution breaking changes | Low | High | Versioning, backward compatibility |

### Mitigation Strategies

1. **Cost Control:**
   - Implement usage monitoring and alerts
   - Set per-document cost limits
   - Use caching to reduce LLM calls

2. **Quality Control:**
   - Strict confidence thresholds for auto-actions
   - Human review queue for borderline cases
   - A/B testing for major changes

3. **Performance Control:**
   - Progressive rollout of batch processing
   - Monitor latency metrics
   - Implement circuit breakers

---

## Time Estimates Summary

| Phase | Duration | Effort | Files Modified | New Files |
|-------|----------|--------|----------------|-----------|
| 1. Foundation Infrastructure | 2 weeks | 4 weeks | 9 | 4 |
| 2. Search Quality Enhancement | 2 weeks | 4 weeks | 7 | 4 |
| 3. Advanced Features | 3 weeks | 6 weeks | 11 | 7 |
| 4. Performance Optimization | 2 weeks | 3 weeks | 6 | 1 |
| 5. Integration and Testing | 2 weeks | 2 weeks | 3 | 1 |
| **TOTAL** | **11 weeks** | **19 weeks** | **36** | **17** |

**Resource Requirements:**
- 1 Senior Engineer (full-time)
- LLM API budget: $500-1000/month (production)
- Additional storage: ~50GB for BM25 indexes and cache
- Redis instance for production caching (optional)

---

## Validation Strategy

### Unit Testing Requirements

1. **Chunking Tests:**
   - Verify chunk size consistency
   - Verify overlap correctness
   - Test semantic boundary detection

2. **Search Tests:**
   - BM25 precision/recall benchmarks
   - Hybrid search fusion accuracy
   - Reranking quality (NDCG, MAP)

3. **Domain Detection Tests:**
   - Single-domain >95%
   - Multi-domain detection accuracy
   - Confidence calibration tests

4. **Type Discovery Tests:**
   - Type proposal precision >80%
   - Validation agreement >85%
   - Auto-promotion precision >95%

5. **Guided Extraction Tests:**
   - Auto-goal selection quality
   - User override functionality
   - Multi-perspective extraction

### Integration Testing Requirements

1. **End-to-End Processing:**
   - Full document processing pipeline
   - Time and resource monitoring
   - Output quality verification

2. **API Integration:**
   - Search API response times
   - Domain detection in API responses
   - Error handling and edge cases

3. **Auto-Mode Tests:**
   - Verify system works without user goals provided
   - Test domain-based default goal selection

### Performance Testing Requirements

1. **Latency Benchmarks:**
   - Document processing: <10 minutes
   - Search queries: <500ms (P95)
   - Reranking: <100ms (P95)

2. **Throughput Tests:**
   - Documents/hour
   - Concurrent document processing
   - Batch processing efficiency

3. **Resource Usage:**
   - Memory usage monitoring
   - LLM API call reduction (caching)
   - Storage growth tracking

---

## Automation Level Summary

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

## References

### Academic Sources

1. "GraphMaster: Multi-Agent LLM Orchestration for KG Synthesis" - arXiv:2504.00711
2. "LLM-as-Judge for KG Quality" - arXiv:2411.17388
3. "BANER: Boundary-Aware LLMs for Few-Shot NER" - COLING 2025
4. "Recent Advances in Named Entity Recognition" - arXiv:2401.10825v3

### Implementation References

1. unstructured.io - Document preprocessing
2. rank-bm25 - BM25 implementation
3. sentence-transformers - Cross-encoder models
4. redis-py - Caching backend
5. pdfplumber - Table extraction
6. pytesseract - OCR processing

---

*Document Generated: 2026-01-28*
*Version: 2.0 (Updated - Clarified Guided Extraction)*
*Status: Implementation Ready*

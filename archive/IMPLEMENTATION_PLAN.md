# KBV2 Implementation Plan (2026)

**Version:** 1.0  
**Last Updated:** February 4, 2026  
**Timeline:** 6 weeks

---

## Executive Summary

This is the **single source of truth** for KBV2 development. After consolidating all previous planning documents and analyzing the current codebase, this plan reflects the **actual state** of the system (70% of infrastructure already complete) and focuses on completing the remaining work.

### Current System State

| Metric | Value |
|--------|-------|
| **Total Tests** | 1,013 collected |
| **Test Pass Rate** | ~99% (9 failing) |
| **Orchestrator Size** | 1,276 lines |
| **Services Implemented** | 5 (of 5) |
| **LLM Clients** | 1 unified + 4 legacy |
| **Domain Support** | 8 domains + CRYPTO_TRADING |
| **Search** | Hybrid (Vector + BM25 + Reranking) |

### 6-Week Roadmap

| Week | Phase | Focus | Target |
|------|-------|-------|--------|
| 1 | Fix Tests | 9 failing tests | 100% passing |
| 2 | Modernize | main.py lifespan | Zero warnings |
| 3-4 | Consolidate | Orchestrator cleanup | <500 lines |
| 5 | Migrate | LLM client usage | Unified only |
| 6 | Deprecate | Legacy code | Clear path forward |

---

## Phase 1: Fix Failing Tests (Week 1)

### Goal
Get all 1,013 tests passing. Currently 9 tests are failing.

### Failing Tests

```
tests/integration/test_real_world_pipeline.py
├── test_complex_natural_language_queries_sql
├── test_mcp_protocol_concurrent_requests
├── test_entity_resolution_deduplication_complex
├── test_performance_load_multiple_operations
├── test_temporal_knowledge_graph_features
├── test_domain_tagging_complex_relationships
├── test_resilient_gateway_under_stress
├── test_end_to_end_knowledge_ingestion_query_cycle
└── test_entity_mention_duplication_handling

tests/integration/test_rotation_integration.py
├── test_resilient_gateway_rotates_on_429
└── test_resilient_gateway_retries_429_with_exponential_backoff
```

### Day 1-2: Diagnosis

```bash
# Run individual failing tests to identify patterns
uv run pytest tests/integration/test_real_world_pipeline.py::TestRealWorldKBV2System::test_end_to_end_knowledge_ingestion_query_cycle -v --tb=short

uv run pytest tests/integration/test_rotation_integration.py::test_resilient_gateway_rotates_on_429 -v --tb=short
```

### Day 3-4: Fix Issues

**Common Patterns Expected:**
- Mock setup issues
- Async test isolation
- Database session management
- LLM client configuration

### Day 5: Verification

```bash
# Full test suite
uv run pytest tests/ -v --tb=short 2>&1 | tail -20

# Expected output:
# ================== 1007 passed, 6 skipped ==================
```

### Success Criteria
- ✅ 100% tests passing
- ✅ No regression in existing tests
- ✅ Test execution time < 5 minutes

---

## Phase 2: Modernize main.py (Week 2)

### Goal
Remove deprecation warnings by migrating from `on_event` to `lifespan` pattern.

### Current Issues

```
src/knowledge_base/main.py:182: DeprecationWarning
  @app.on_event("startup") is deprecated, use lifespan event handlers instead.

src/knowledge_base/main.py:551: DeprecationWarning  
  @app.on_event("shutdown") is deprecated, use lifespan event handlers instead.
```

### Implementation

**File:** `src/knowledge_base/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup logic
    await startup_event()
    yield
    # Shutdown logic
    await shutdown_event()

app = FastAPI(lifespan=lifespan)

async def startup_event():
    """Initialize services on startup."""
    # Move code from @app.on_event("startup")
    pass

async def shutdown_event():
    """Cleanup services on shutdown."""
    # Move code from @app.on_event("shutdown")  
    pass
```

### Files to Update

| File | Change |
|------|--------|
| `main.py:1-50` | Add lifespan import |
| `main.py:180-200` | Replace startup event |
| `main.py:540-560` | Replace shutdown event |
| All imports | Update if services change |

### Verification

```bash
# Run with warnings as errors
uv run python -W error src/knowledge_base/main.py

# Check for warnings
uv run pytest tests/ -v --disable-warnings 2>&1 | grep -i deprecation
# Should return empty
```

### Success Criteria
- ✅ Zero deprecation warnings
- ✅ Lifespan pattern working
- ✅ All tests passing

---

## Phase 3: Consolidate Orchestrator (Weeks 3-4)

### Goal
Reduce orchestrator.py from 1,276 lines to <500 lines by removing duplicate methods.

### Current State

```
src/knowledge_base/orchestrator.py (1,276 lines)
├── async def initialize() - 80 lines
├── async def _partition_document() - OLD, delegates to service
├── async def _extract_knowledge() - OLD, delegates to service  
├── async def _resolve_entities() - OLD, delegates to service
├── async def _refine_entity_types() - OLD, delegates to service
├── async def _cluster_entities() - OLD, delegates to service
├── async def _embed_content() - OLD, delegates to service
├── async def process_document() - 200 lines, main entry point
└── Other methods - 200 lines
```

### Problem
The orchestrator has BOTH:
1. Old methods that implement logic directly
2. New services that implement the same logic

This creates confusion and maintenance burden.

### Solution: Pure Coordinator Pattern

```python
class IngestionOrchestrator:
    """Pure coordinator - only delegates to services."""

    async def initialize(self) -> None:
        """Initialize all services."""
        self._document_service = DocumentPipelineService()
        self._entity_service = EntityPipelineService()
        self._quality_service = QualityAssuranceService()
        self._domain_service = DomainDetectionService()
        
        await asyncio.gather(
            self._document_service.initialize(),
            self._entity_service.initialize(),
            self._quality_service.initialize(),
            self._domain_service.initialize(),
        )

    async def process_document(
        self,
        file_path: str,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Document:
        """Process document - pure delegation."""
        # Stage 1: Detect domain
        if not domain:
            domain = await self._domain_service.detect_domain(file_path)
        
        # Stage 2: Partition and embed
        document = await self._document_service.process(
            file_path=file_path,
            document_name=document_name,
            domain=domain,
        )
        
        # Stage 3: Extract entities
        entities, edges = await self._entity_service.extract(document)
        
        # Stage 4: Resolve and cluster
        await self._entity_service.resolve_and_cluster(document, entities)
        
        # Stage 5: Quality assurance
        await self._quality_service.validate(document, entities, edges)
        
        return document

    # REMOVE ALL OLD METHODS:
    # - _partition_document()
    # - _extract_knowledge() 
    # - _resolve_entities()
    # - _refine_entity_types()
    # - _cluster_entities()
    # - _embed_content()
```

### Week 3: Extract Methods

| Method | Lines | Replace With |
|--------|-------|--------------|
| `_partition_document()` | ~50 | `self._document_service.partition()` |
| `_extract_entities_multi_agent()` | ~194 | `self._entity_service.extract()` |
| `_resolve_entities()` | ~102 | `self._entity_service.resolve()` |
| `_refine_entity_types()` | ~60 | `self._entity_service.refine_types()` |
| `_cluster_entities()` | ~45 | `self._entity_service.cluster()` |
| `_embed_content()` | ~89 | `self._document_service.embed()` |

### Week 4: Refactor process_document()

```python
async def process_document(
    self,
    file_path: str,
    document_name: str | None = None,
    domain: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Document:
    """Process document through full pipeline."""
    
    # Progress reporting
    if progress_callback:
        progress_callback({"stage": "domain_detection", "status": "started"})
    
    # Detect domain
    if not domain:
        domain = await self._domain_service.detect_domain(file_path)
    
    if progress_callback:
        progress_callback({"stage": "domain_detection", "status": "completed", "domain": domain})
    
    # Partition and embed
    if progress_callback:
        progress_callback({"stage": "partitioning", "status": "started"})
    
    document = await self._document_service.process(
        file_path=file_path,
        document_name=document_name,
        domain=domain,
    )
    
    if progress_callback:
        progress_callback({"stage": "partitioning", "status": "completed"})
    
    # Extract entities
    if progress_callback:
        progress_callback({"stage": "entity_extraction", "status": "started"})
    
    entities, edges = await self._entity_service.extract(document)
    
    if progress_callback:
        progress_callback({
            "stage": "entity_extraction", 
            "status": "completed",
            "entities_found": len(entities)
        })
    
    # Resolve and cluster
    if progress_callback:
        progress_callback({"stage": "resolution", "status": "started"})
    
    await self._entity_service.resolve_and_cluster(document, entities)
    
    if progress_callback:
        progress_callback({"stage": "resolution", "status": "completed"})
    
    # Quality assurance
    if progress_callback:
        progress_callback({"stage": "quality_assurance", "status": "started"})
    
    await self._quality_service.validate(document, entities, edges)
    
    if progress_callback:
        progress_callback({"stage": "quality_assurance", "status": "completed"})
    
    return document
```

### Success Criteria
- ✅ orchestrator.py < 500 lines
- ✅ No duplicate logic
- ✅ All tests passing
- ✅ No regression in functionality

---

## Phase 4: Migrate LLM Clients (Week 5)

### Goal
Use only `UnifiedLLMClient` throughout the codebase.

### Current State

| File | Lines | Status |
|------|-------|--------|
| `unified_llm_client.py` | 96 | ✅ Exists |
| `llm_client.py` | 707 | ⚠️ Legacy |
| `gateway.py` | 503 | ⚠️ Legacy |
| `rotating_llm_client.py` | 309 | ⚠️ Legacy |
| `rotation_manager.py` | 428 | ⚠️ Legacy |

### Migration Plan

**Step 1: Find all imports**

```bash
grep -r "from knowledge_base.clients.llm_client import" src/
grep -r "from knowledge_base.clients.gateway import" src/
grep -r "from knowledge_base.clients.rotating_llm_client import" src/
```

**Step 2: Update each file**

```python
# BEFORE
from knowledge_base.clients.llm_client import LLMClient

# AFTER  
from knowledge_base.clients.unified_llm_client import UnifiedLLMClient
```

**Step 3: Update usage patterns**

```python
# BEFORE
client = LLMClient(api_key="...", model="...")
response = await client.complete(prompt="...")

# AFTER
client = UnifiedLLMClient(api_url="http://localhost:8087/v1")
response = await client.chat_completion(messages=[...])
```

### Files to Update

| File | Import |
|------|--------|
| `main.py` | Update gateway initialization |
| `orchestrator.py` | Update `_gateway` type |
| `ingestion/v1/gleaning_service.py` | Update LLM calls |
| `intelligence/v1/*_agent.py` | Update LLM calls |

### Verification

```bash
# Ensure no old imports remain
grep -r "from knowledge_base.clients.llm_client import" src/
grep -r "from knowledge_base.clients.gateway import" src/
# Both should return empty
```

### Success Criteria
- ✅ UnifiedLLMClient used everywhere
- ✅ All LLM functionality working
- ✅ Tests passing

---

## Phase 5: Deprecate Legacy Code (Week 6)

### Goal
Mark old code as deprecated and provide clear migration path.

### Deprecation Strategy

**File:** `src/knowledge_base/clients/llm_client.py`

```python
import warnings

class LLMClient:
    """Legacy LLM client - use UnifiedLLMClient instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LLMClient is deprecated since v0.6.0. "
            "Use UnifiedLLMClient from unified_llm_client instead.",
            DeprecationWarning,
            stacklevel=2,
            source=__file__
        )
        super().__init__(*args, **kwargs)
```

**File:** `src/knowledge_base/clients/gateway.py`

```python
import warnings

class GatewayClient:
    """Legacy gateway client - use UnifiedLLMClient instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "GatewayClient is deprecated since v0.6.0. "
            "Use UnifiedLLMClient from unified_llm_client instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

### Removal Timeline

| Version | Action |
|---------|--------|
| v0.6.0 (current) | Add deprecation warnings |
| v0.7.0 | Remove old client files |
| v1.0.0 | Legacy code gone |

### Success Criteria
- ✅ Deprecation warnings visible
- ✅ Migration path documented
- ✅ No new code uses legacy clients

---

## Technical Implementation Details

### Architecture Overview

```
src/knowledge_base/
├── orchestration/                    # Service layer (1,164 lines total)
│   ├── __init__.py
│   ├── base_service.py              # Abstract base (21 lines)
│   ├── document_pipeline_service.py # Document processing (117 lines)
│   ├── domain_detection_service.py  # Domain classification (236 lines)
│   ├── entity_pipeline_service.py   # Entity extraction (577 lines)
│   └── quality_assurance_service.py # Validation (208 lines)
├── clients/                         # LLM clients
│   ├── unified_llm_client.py        # ✅ Use this (96 lines)
│   ├── middleware/                  # Middleware pattern
│   │   ├── retry_middleware.py
│   │   ├── rotation_middleware.py
│   │   └── circuit_breaker.py
│   └── models.py                    # Shared models
├── config/
│   └── constants.py                 # Centralized constants (86 lines)
├── common/
│   ├── exceptions.py                # Exception hierarchy
│   └── types.py                     # Type aliases
├── storage/                         # Search layer
│   ├── bm25_index.py               # BM25 implementation
│   └── hybrid_search.py            # Vector + BM25
├── reranking/                       # Reranking layer
│   ├── cross_encoder.py
│   ├── reranking_pipeline.py
│   └── rrf_fuser.py
├── domain/                         # Domain system
│   ├── detection.py                 # Domain detection
│   ├── domain_models.py
│   └── ontology_snippets.py        # Domain keywords
└── orchestrator.py                 # Coordinator (<500 lines target)
```

### Dependency Graph

```
main.py
    ↓
orchestrator.py (IngestionOrchestrator)
    ├── document_pipeline_service.py
    │   ├── partitioning/semantic_chunker.py
    │   └── ingestion/v1/embedding_client.py
    ├── entity_pipeline_service.py
    │   ├── intelligence/v1/multi_agent_extractor.py
    │   ├── intelligence/v1/resolution_agent.py
    │   ├── intelligence/v1/hallucination_detector.py
    │   └── ingestion/v1/gleaning_service.py
    ├── domain_detection_service.py
    │   └── domain/detection.py
    └── quality_assurance_service.py
        └── intelligence/v1/clustering_service.py
```

### Configuration

All magic numbers centralized in `config/constants.py`:

```python
# Network
LLM_GATEWAY_URL: Final[str] = "http://localhost:8087/v1"
WEBSOCKET_PORT: Final[int] = 8765

# Timeouts
DEFAULT_LLM_TIMEOUT: Final[float] = 120.0
DEFAULT_HTTP_TIMEOUT: Final[float] = 60.0

# Chunking
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 50

# Quality Thresholds
MIN_EXTRACTION_QUALITY_SCORE: Final[float] = 0.5
ENTITY_SIMILARITY_THRESHOLD: Final[float] = 0.85

# Search
VECTOR_SEARCH_WEIGHT: Final[float] = 0.6
GRAPH_SEARCH_WEIGHT: Final[float] = 0.4
```

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/
│   ├── test_api/                    # API endpoint tests
│   ├── test_orchestration/         # Service tests
│   ├── test_clients/               # LLM client tests
│   └── test_common/                 # Utility tests
├── integration/
│   ├── test_enhanced_pipeline.py   # Full pipeline tests
│   ├── test_real_world_pipeline.py # E2E tests
│   └── test_rotation_integration.py # Rotation tests
└── conftest.py                     # Pytest fixtures
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=knowledge_base --cov-report=html

# Specific test file
uv run pytest tests/integration/test_real_world_pipeline.py -v

# With verbose output
uv run pytest tests/ -vv --tb=short
```

### Test Fixtures

```python
@pytest.fixture
async def llm_client():
    """Create LLM client for testing."""
    return UnifiedLLMClient()

@pytest.fixture
async def sample_document():
    """Sample document for testing."""
    return Document(
        id=uuid4(),
        title="Test Document",
        content="Sample content...",
    )
```

---

## Performance Targets

### Current vs Target

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Test Suite | 5+ min | <3 min | 40% faster |
| Orchestrator | 1,276 lines | <500 lines | 60% reduction |
| LLM Clients | 4 files | 1 file | 75% reduction |
| Deprecation Warnings | 2 | 0 | 100% fixed |

### Optimization Strategies

1. **Parallel Service Initialization**
```python
await asyncio.gather(
    self._document_service.initialize(),
    self._entity_service.initialize(),
    self._quality_service.initialize(),
)
```

2. **Batch LLM Calls**
```python
# Instead of individual calls
for entity in entities:
    await typer.type_entity(entity)

# Use batch
await typer.type_entities_batch(entities)
```

3. **Connection Pooling**
```python
# Ensure connection reuse
engine = create_engine(DATABASE_URL, pool_size=10)
```

---

## Migration Checklist

### Week 1: Fix Tests
- [ ] Run failing tests individually
- [ ] Identify common failure patterns
- [ ] Fix setup issues
- [ ] Verify all tests pass

### Week 2: Modernize main.py
- [ ] Add lifespan context manager
- [ ] Move startup logic
- [ ] Move shutdown logic
- [ ] Remove deprecation warnings

### Week 3: Consolidate Orchestrator
- [ ] Identify duplicate methods
- [ ] Create delegation methods
- [ ] Remove old implementations
- [ ] Update process_document()

### Week 4: Complete Consolidation
- [ ] Refactor process_document()
- [ ] Remove unused imports
- [ ] Verify functionality
- [ ] Update tests if needed

### Week 5: Migrate LLM Clients
- [ ] Find all legacy imports
- [ ] Update to UnifiedLLMClient
- [ ] Update usage patterns
- [ ] Verify functionality

### Week 6: Deprecate Legacy
- [ ] Add deprecation warnings
- [ ] Document migration path
- [ ] Remove old files
- [ ] Final verification

---

## Known Issues & Workarounds

### 1. Circular Imports
**Issue:** Services importing each other  
**Workaround:** Use lazy imports in methods

### 2. Async Test Isolation
**Issue:** Tests affecting each other  
**Workaround:** Use unique database sessions per test

### 3. LLM Rate Limiting
**Issue:** Rate limit errors during tests  
**Workaround:** Use mock LLM responses in unit tests

### 4. Database Schema Evolution
**Issue:** Schema changes causing test failures  
**Workaround:** Use migrations, not schema recreation

---

## References

### Code Files
- `src/knowledge_base/orchestrator.py` - Main orchestrator
- `src/knowledge_base/orchestration/` - Service layer
- `src/knowledge_base/clients/unified_llm_client.py` - LLM client

### Configuration
- `pyproject.toml` - Project configuration
- `.env` - Environment variables

### Documentation
- `README.md` - Project overview
- `docs/` - Architecture documentation

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-04 | Initial plan |

---

**Status:** Ready for execution  
**Confidence:** High (based on codebase analysis)  
**Risk:** Medium (9 failing tests need investigation)

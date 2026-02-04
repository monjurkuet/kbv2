# KBV2 Knowledge Base - Comprehensive Refactoring Plan

## Executive Summary

After analyzing the KBV2 codebase (~29,055 lines) and reviewing four different refactoring proposals, I've synthesized a **unified, pragmatic refactoring plan** that addresses the core architectural issues while minimizing risk and maximizing maintainability improvements.

### Current State Assessment

| Metric | Current | Target |
|--------|---------|--------|
| **orchestrator.py** | 2,031 lines (god class) | <300 lines (coordinator only) |
| **LLM Client Variants** | 4+ implementations | 1 unified client |
| **Magic Numbers** | 50+ scattered | Centralized constants |
| **Type Safety** | ~500 mypy errors | 0 errors (strict mode) |
| **Test Coverage** | <30% | >80% |
| **Duplicate Code** | High | <5% |

### Critical Issues Identified

1. **God Class Anti-Pattern**: `orchestrator.py` at 2,031 lines violates Single Responsibility Principle
2. **LLM Client Proliferation**: 4+ implementations with overlapping functionality:
   - `llm_client.py` (707 lines) - Base client with prompting strategies
   - `gateway.py` (503 lines) - Gateway client with duplicate models
   - `resilient_gateway/` (770 lines) - Resilient wrapper
   - `rotating_llm_client.py` (309 lines) + `rotation_manager.py` (428 lines) - Duplicate rotation logic
3. **Configuration Chaos**: Magic numbers, hardcoded URLs, scattered thresholds
4. **Error Handling Inconsistency**: Empty except blocks, broad exception catching

---

## My Recommended Refactoring Strategy

### Guiding Principles

1. **Strangler Fig Pattern**: Gradual migration, not big-bang rewrite
2. **Backward Compatibility**: All external APIs remain unchanged
3. **Test-Driven**: Each phase has verification steps
4. **Risk-First**: Address highest-risk items first

---

## Phase 1: Foundation & Constants (Week 1) - LOW RISK

### Goal
Create centralized configuration system to eliminate magic numbers.

### New Files

```
src/knowledge_base/config/
├── __init__.py
├── constants.py          # All magic numbers
├── settings.py           # Pydantic settings
└── domain_keywords.py    # Extracted from orchestrator
```

### constants.py
```python
"""Centralized constants for KBV2."""

# Network
DEFAULT_LLM_GATEWAY_PORT = 8087
DEFAULT_WEBSOCKET_PORT = 8765
DEFAULT_DATABASE_PORT = 5432
LLM_GATEWAY_BASE_URL = "http://localhost:8087/v1/"

# Timeouts
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_HTTP_TIMEOUT = 60.0
ROTATION_DELAY = 5.0
INGESTION_TIMEOUT = 3600.0
CONNECTION_TIMEOUT = 60.0

# Rate Limiting
RATE_LIMIT_STATUS_CODES = [429, 503, 529]
RATE_LIMIT_MESSAGES = [
    "too many requests",
    "rate limit",
    "quota exceeded",
    "try again later",
]

# Embedding
EMBEDDING_DIMENSIONS_BGE_M3 = 1024
EMBEDDING_MAX_TOKENS = 8191

# Pagination
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 1000

# Quality Thresholds
MIN_EXTRACTION_QUALITY_SCORE = 0.5
ENTITY_SIMILARITY_THRESHOLD = 0.85
DOMAIN_CONFIDENCE_THRESHOLD = 0.6
HALLUCINATION_THRESHOLD = 0.3

# Chunking
DEFAULT_CHUNK_SIZE = 1536
DEFAULT_CHUNK_OVERLAP = 0.25
MIN_CHUNK_SIZE = 256

# Extraction
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_ENTITY_COUNT = 20

# Search Weights
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5
WEIGHT_TOLERANCE = 1e-6

# Verification
VERIFICATION_TEMP = 0.1
VERIFICATION_MAX_TOKENS = 1024
BATCH_SIZE = 10
```

### Files to Modify

| File | Changes |
|------|---------|
| `orchestrator.py` | Remove DOMAIN_KEYWORDS dict (lines 27118-27249) |
| `clients/cli.py` | Replace hardcoded port 8765 |
| `clients/websocket_client.py` | Replace hardcoded port 8765 |
| `clients/gateway.py` | Replace hardcoded URL/timeout |
| `clients/llm_client.py` | Replace hardcoded URL/timeout |
| `clients/rotation_manager.py` | Replace hardcoded thresholds |
| `clients/rotating_llm_client.py` | Replace hardcoded thresholds |

### Verification
```bash
# Ensure no hardcoded values remain
grep -r "8087\|8765\|120.0\|0.5" src/knowledge_base --include="*.py" | grep -v constants.py
```

---

## Phase 2: Dead Code & Debug Cleanup (Week 1-2) - LOW RISK

### Goal
Remove debug artifacts and clean up production code.

### Actions

#### 2.1 Remove print() Statements

| File | Lines | Action |
|------|-------|--------|
| `orchestrator.py` | 2022-2023 | Replace with logger.debug() |
| `persistence/v1/vector_store.py` | 83, 118, 120 | Replace with logger.info() |
| `clients/cli.py` | 244-277 | Keep for CLI (acceptable) |
| `ingestion/v1/embedding_client.py` | 267-277 | Remove debug prints |
| `common/resilient_gateway/example.py` | 45-161 | Convert to logger or mark as example |

#### 2.2 Fix Empty except Blocks (12 instances)

| File | Line | Action |
|------|------|--------|
| `clients/llm_client.py` | 521 | Add logging context |
| `orchestrator.py` | 27404 | Add logging context |
| `storage/hybrid_search.py` | 464 | Add logging context |
| `mcp_server.py` | 129 | Add logging or remove |
| `intelligence/v1/extraction_logging.py` | 341 | Add logging context |
| `intelligence/v1/resolution_agent.py` | 252 | Add logging context |
| `intelligence/v1/cross_domain_detector.py` | 535 | Add logging context |
| `common/gateway.py` | 93 | Add logging context |
| `common/offset_service.py` | 267 | Add logging context |
| `ingestion/v1/gleaning_service.py` | 754, 770 | Add logging context |
| `summaries/community_summaries.py` | 372 | Add logging context |

#### 2.3 Remove Duplicate Imports

| File | Lines | Action |
|------|-------|--------|
| `orchestrator.py` | 27064-27075 | Remove duplicate HallucinationDetector import |

### Verification
```bash
# Check for remaining print statements
grep -r "print(" src/knowledge_base --include="*.py" | grep -v "__pycache__" | grep -v example.py

# Check for empty except blocks
grep -r "except.*:.*$" src/knowledge_base --include="*.py" -A1 | grep -E "except.*:|^\s*pass\s*$"
```

---

## Phase 3: LLM Client Consolidation (Week 2-3) - MEDIUM RISK

### Goal
Consolidate 4+ LLM client implementations into 1 unified client.

### Current State Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Client Chaos                         │
├─────────────────────────────────────────────────────────────┤
│  llm_client.py (707 lines)                                  │
│    ├── ChatMessage, ChatCompletionRequest, LLMResponse      │
│    ├── LLMClient with retry logic                           │
│    └── Prompting strategies (few-shot, CoT, CoD)            │
├─────────────────────────────────────────────────────────────┤
│  gateway.py (503 lines)                                     │
│    ├── DUPLICATE ChatMessage, ChatCompletionRequest         │
│    ├── GatewayClient (similar to LLMClient)                 │
│    └── EnhancedGateway (adds rotation)                      │
├─────────────────────────────────────────────────────────────┤
│  resilient_gateway/ (770 lines)                             │
│    ├── ResilientGatewayClient                               │
│    ├── Continuous rotation logic                            │
│    └── Circuit breaker pattern                              │
├─────────────────────────────────────────────────────────────┤
│  rotating_llm_client.py (309 lines)                         │
│    ├── ModelRotationConfig                                  │
│    └── RotatingLLMClient (extends LLMClient)                │
├─────────────────────────────────────────────────────────────┤
│  rotation_manager.py (428 lines)                            │
│    ├── DUPLICATE ModelRotationConfig                        │
│    └── ModelRotationManager (wraps GatewayClient)           │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Unified LLM Client Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           UnifiedLLMClient (facade)                 │   │
│  │  - Single interface for all LLM operations          │   │
│  │  - Automatic model rotation                         │   │
│  │  - Retry logic with exponential backoff             │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Retry     │ │  Rotation   │ │   Circuit   │           │
│  │  Handler    │ │   Manager   │ │   Breaker   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│         │               │               │                   │
│         └───────────────┼───────────────┘                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              httpx.AsyncClient                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### New Files

```
src/knowledge_base/clients/
├── __init__.py
├── unified_llm_client.py      # NEW: Single LLM client
├── middleware/
│   ├── __init__.py
│   ├── retry_middleware.py    # NEW: Retry logic
│   ├── rotation_middleware.py # NEW: Model rotation
│   └── circuit_breaker.py     # NEW: Circuit breaker
└── models.py                  # NEW: Shared models (moved from llm_client.py)
```

### unified_llm_client.py (Core Interface)

```python
"""Unified LLM Client - Single interface for all LLM operations."""

from typing import Any, Callable
import httpx
from pydantic import BaseModel, Field

from knowledge_base.config.constants import (
    DEFAULT_LLM_TIMEOUT,
    LLM_GATEWAY_BASE_URL,
    RATE_LIMIT_STATUS_CODES,
)


class ChatMessage(BaseModel):
    """Unified chat message."""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    """Unified LLM response."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    usage: dict[str, int] | None = Field(None, description="Token usage")
    success: bool = Field(default=True, description="Whether call succeeded")
    attempts: int = Field(default=1, description="Number of attempts made")


class UnifiedLLMClient:
    """Unified LLM client with retry, rotation, and circuit breaker.
    
    This is the ONLY LLM client that should be used throughout the codebase.
    It combines:
    - Base LLM operations from llm_client.py
    - Gateway functionality from gateway.py
    - Rotation from rotating_llm_client.py + rotation_manager.py
    - Resilience from resilient_gateway/
    """
    
    def __init__(
        self,
        base_url: str = LLM_GATEWAY_BASE_URL,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        enable_rotation: bool = True,
        enable_retry: bool = True,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._enable_rotation = enable_rotation
        self._enable_retry = enable_retry
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._rotation_manager: RotationManager | None = None
        
    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Execute chat completion with automatic retry and rotation."""
        # Implementation combines best of all existing clients
        pass
    
    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Simple completion interface."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        
        response = await self.chat_completion(messages, **kwargs)
        return response.content
    
    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Completion with JSON output."""
        response = await self.complete(
            prompt,
            system_prompt,
            response_format={"type": "json_object"},
            **kwargs,
        )
        import json
        return json.loads(response)
```

### Migration Strategy

1. **Week 2**: Create unified client alongside existing ones
2. **Week 3**: Migrate one service at a time to unified client
3. **Week 4**: Deprecate old clients (add deprecation warnings)
4. **Week 6**: Remove old clients after verification

### Files to Deprecate

| File | Replacement | Timeline |
|------|-------------|----------|
| `clients/llm_client.py` | `unified_llm_client.py` | Week 4 (deprecate) |
| `clients/gateway.py` | `unified_llm_client.py` | Week 4 (deprecate) |
| `clients/rotating_llm_client.py` | `unified_llm_client.py` | Week 4 (deprecate) |
| `clients/rotation_manager.py` | `unified_llm_client.py` | Week 4 (deprecate) |
| `common/resilient_gateway/` | `unified_llm_client.py` | Week 4 (deprecate) |

---

## Phase 4: God Class Decomposition (Week 3-6) - HIGH RISK

### Goal
Break down 2,031-line orchestrator.py into focused, single-responsibility services.

### Current Orchestrator Structure

```python
class IngestionOrchestrator:
    # 131 lines: DOMAIN_KEYWORDS dictionary
    # 82 lines: __init__ with 15+ service dependencies
    # 133 lines: _determine_domain() - domain detection
    # 49 lines: _partition_document() - document chunking
    # 288 lines: _extract_knowledge() - entity extraction
    # 102 lines: _resolve_entities() - entity resolution
    # 60 lines: _refine_entity_types() - entity typing
    # 81 lines: _validate_entities_against_schema() - validation
    # 45 lines: _cluster_entities() - clustering
    # 89 lines: _embed_content() - embedding generation
    # 156 lines: _generate_reports() - reporting
    # 67 lines: _add_to_review_queue() - review queue
    # 45 lines: _merge_entities() - entity merging
    # 78 lines: _route_to_review() - review routing
    # 245 lines: process_document() - main entry point
    # 95 lines: process_document_stream() - streaming
    # 142 lines: deduplicate_all_entities() - global dedup
    # 194 lines: _extract_entities_multi_agent() - multi-agent
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Refactored Orchestration Layer                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │       IngestionOrchestrator (~250 lines)            │   │
│  │  - Pure coordinator, no business logic              │   │
│  │  - Delegates to specialized services                │   │
│  │  - Manages pipeline flow only                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                         │                                   │
│    ┌────────────────────┼────────────────────┐              │
│    │                    │                    │              │
│    ▼                    ▼                    ▼              │
│ ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│ │ Document │     │ Entity   │     │ Quality  │             │
│ │ Pipeline │────▶│ Pipeline │────▶│ Assurance│             │
│ │ Service  │     │ Service  │     │ Service  │             │
│ └──────────┘     └──────────┘     └──────────┘             │
│       │                │                │                   │
│       ▼                ▼                ▼                   │
│ ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│ │ Partition│     │ Extract  │     │ Validate │             │
│ │ Service  │     │ Service  │     │ Service  │             │
│ └──────────┘     └──────────┘     └──────────┘             │
│       │                │                │                   │
│       ▼                ▼                ▼                   │
│ ┌──────────┐     ┌──────────┐     ┌──────────┐             │
│ │ Embed    │     │ Resolve  │     │ Review   │             │
│ │ Service  │     │ Service  │     │ Service  │             │
│ └──────────┘     └──────────┘     └──────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### New Service Structure

```
src/knowledge_base/orchestration/
├── __init__.py
├── orchestrator.py              # Modified: ~250 lines coordinator
├── base_service.py              # NEW: Abstract base for all services
├── document_pipeline_service.py # NEW: Document processing pipeline
├── entity_pipeline_service.py   # NEW: Entity extraction pipeline
├── quality_assurance_service.py # NEW: Validation & review
├── domain_detection_service.py  # NEW: Domain classification
├── embedding_service.py         # NEW: Embedding generation
└── clustering_service.py        # NEW: Entity clustering
```

### orchestrator.py (Refactored - 250 lines)

```python
"""IngestionOrchestrator - Pure coordinator for document ingestion.

This module contains ONLY coordination logic. All business logic has been
extracted to specialized services in the orchestration package.
"""

from knowledge_base.orchestration.document_pipeline_service import DocumentPipelineService
from knowledge_base.orchestration.entity_pipeline_service import EntityPipelineService
from knowledge_base.orchestration.quality_assurance_service import QualityAssuranceService
from knowledge_base.orchestration.domain_detection_service import DomainDetectionService


class IngestionOrchestrator:
    """Main orchestrator - coordinates ingestion pipeline stages.
    
    This class is intentionally thin. It only:
    1. Manages service lifecycle
    2. Coordinates pipeline flow
    3. Handles progress callbacks
    
    All business logic is delegated to specialized services.
    """
    
    def __init__(self, progress_callback=None, log_broadcast=None):
        self._progress_callback = progress_callback
        self._log_broadcast = log_broadcast
        
        # Services are injected for testability
        self._document_service: DocumentPipelineService | None = None
        self._entity_service: EntityPipelineService | None = None
        self._quality_service: QualityAssuranceService | None = None
        self._domain_service: DomainDetectionService | None = None
        
    async def initialize(self) -> None:
        """Initialize all services."""
        # Initialize services in dependency order
        self._domain_service = DomainDetectionService()
        self._document_service = DocumentPipelineService()
        self._entity_service = EntityPipelineService()
        self._quality_service = QualityAssuranceService()
        
        await asyncio.gather(
            self._domain_service.initialize(),
            self._document_service.initialize(),
            self._entity_service.initialize(),
            self._quality_service.initialize(),
        )
        
    async def process_document(
        self,
        file_path: str,
        document_name: str | None = None,
        domain: str | None = None,
    ) -> Document:
        """Process a document through the full pipeline.
        
        Pipeline stages:
        1. Domain detection (if not provided)
        2. Document partitioning
        3. Entity extraction
        4. Entity resolution
        5. Quality validation
        6. Embedding generation
        7. Clustering
        """
        await self._send_progress({"stage": 0, "status": "started"})
        
        # Stage 1: Detect domain if not provided
        if not domain:
            domain = await self._domain_service.detect_domain(file_path)
        
        # Stage 2: Partition document
        document = await self._document_service.partition(
            file_path=file_path,
            document_name=document_name,
            domain=domain,
        )
        
        # Stage 3: Extract entities
        entities, edges = await self._entity_service.extract(document)
        
        # Stage 4: Resolve entities
        await self._entity_service.resolve(document, entities)
        
        # Stage 5: Quality assurance
        await self._quality_service.validate(document, entities, edges)
        
        # Stage 6: Generate embeddings
        await self._document_service.embed(document)
        
        # Stage 7: Cluster entities
        await self._entity_service.cluster(document)
        
        return document
```

### Migration Strategy

1. **Week 3-4**: Extract services one at a time, keeping orchestrator functional
2. **Week 5**: Migrate all internal calls to new services
3. **Week 6**: Remove old methods from orchestrator, add delegation only

### Service Extraction Plan

| Service | Source Lines | Target Lines | Priority |
|---------|--------------|--------------|----------|
| DomainDetectionService | 133 | ~150 | Week 3 |
| DocumentPipelineService | 133 + 89 | ~200 | Week 3 |
| EntityPipelineService | 288 + 102 + 60 + 194 | ~400 | Week 4 |
| QualityAssuranceService | 81 + 67 + 45 + 78 | ~200 | Week 5 |
| EmbeddingService | 89 | ~100 | Week 5 |
| ClusteringService | 45 | ~80 | Week 5 |

---

## Phase 5: Type Safety & Error Handling (Week 5-7) - MEDIUM RISK

### Goal
Achieve mypy strict mode compliance and implement proper exception hierarchy.

### 5.1 Exception Hierarchy

```python
# src/knowledge_base/common/exceptions.py

class KBV2BaseException(Exception):
    """Base exception for all KBV2 errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}


class IngestionError(KBV2BaseException):
    """Document ingestion failed."""
    pass


class ExtractionError(KBV2BaseException):
    """Entity extraction failed."""
    pass


class ResolutionError(KBV2BaseException):
    """Entity resolution failed."""
    pass


class ConfigurationError(KBV2BaseException):
    """Invalid or missing configuration."""
    pass


class LLMClientError(KBV2BaseException):
    """LLM client operation failed."""
    pass


class ValidationError(KBV2BaseException):
    """Data validation failed."""
    pass
```

### 5.2 Type Safety Improvements

| File | Current Issues | Target |
|------|----------------|--------|
| `observability.py` | 8 `Any` types | Specific types |
| `rotation_manager.py` | 2 `Any` types | Specific types |
| `gateway.py` | Many `Any` types | Specific types |
| `orchestrator.py` | Missing return types | Full type hints |

### 5.3 Early Returns & Guard Clauses

Convert deeply nested code to early return pattern:

```python
# Before (deep nesting)
def process(self, doc):
    if doc:
        try:
            result = self.extract(doc)
            if result.is_valid():
                if result.confidence > 0.8:
                    return self.save(result)
        except Exception as e:
            logger.error(e)
    return None

# After (early returns)
def process(self, doc: Document | None) -> Result | None:
    if not doc:
        return None
        
    result = self._safe_extract(doc)
    if result is None:
        return None
        
    if not result.is_valid():
        return None
        
    if result.confidence <= 0.8:
        return None
        
    return self.save(result)
```

---

## Phase 6: Testing & Verification (Week 6-8) - LOW RISK

### Goal
Achieve >80% test coverage with proper test pyramid.

### Test Structure

```
tests/
├── unit/
│   ├── orchestration/         # Service unit tests
│   ├── clients/               # LLM client tests
│   └── common/                # Utility tests
├── integration/
│   ├── test_pipeline.py       # Pipeline integration
│   ├── test_api.py            # API integration
│   └── fixtures/              # Test data
├── e2e/
│   └── test_ingestion.py      # End-to-end tests
└── conftest.py                # pytest configuration
```

### Coverage Requirements

| Component | Target Coverage |
|-----------|-----------------|
| orchestration/* | 90% |
| clients/* | 85% |
| common/* | 80% |
| API endpoints | 100% |

## Timeline Summary

| Phase | Duration | Risk | Key Deliverables |
|-------|----------|------|------------------|
| 1. Foundation | Week 1 | Low | constants.py, settings.py |
| 2. Cleanup | Week 1-2 | Low | No print(), no empty excepts |
| 3. LLM Consolidation | Week 2-3 | Medium | unified_llm_client.py |
| 4. Orchestrator Decomposition | Week 3-6 | High | 6 new services |
| 5. Type Safety | Week 5-7 | Medium | exceptions.py, type hints |
| 6. Testing | Week 6-8 | Low | >80% coverage |

**Total: 8 weeks**

---

## Success Criteria

1. ✅ `orchestrator.py` < 300 lines
2. ✅ Single LLM client for all operations
3. ✅ Zero mypy errors with `--strict`
4. ✅ >80% test coverage
5. ✅ All print() statements removed from production code
6. ✅ Proper exception hierarchy implemented
7. ✅ No duplicate code patterns
8. ✅ All external APIs unchanged

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes | Strangler fig pattern, feature flags |
| Test failures | Parallel test development |
| Performance regression | Benchmark before/after |
| Team disruption | Incremental migration |

---

## Next Steps

1. **Immediate**: Create feature branch `refactor/phase-1-foundation`
2. **Week 1**: Implement constants.py and settings.py
3. **Week 1**: Remove print() statements
4. **Week 2**: Begin LLM client consolidation design
5. **Week 3**: Start orchestrator decomposition

---
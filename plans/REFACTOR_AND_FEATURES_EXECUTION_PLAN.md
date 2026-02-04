# KBV2 Refactoring & Bitcoin Trading Features - Session Handoff Plan

**Date:** 2026-01-31  
**Status:** Planning Complete - Ready for Execution  
**Related Documents:** 
- `kbv2_comprehensive_refactoring_plan_kimi.md` (comprehensive 8-week plan)
- `btc_trading_kb_implementation_plan_claude_website.md` (Bitcoin features spec)

---

## Executive Summary

This document provides a **hybrid execution plan** that delivers Bitcoin trading knowledge base features within **1 week** while maintaining a clean foundation for the full 8-week refactoring. The strategy prioritizes:

1. **Quick wins first** - Bitcoin features deliverable in ~1 week
2. **Clean foundation** - Phase 1-2 refactoring done before features
3. **Zero technical debt** - New features use proper patterns from the start
4. **Incremental improvement** - Full refactor continues after feature delivery

---

## Strategic Decision Log

| Decision | Rationale |
|----------|-----------|
| **Do Phase 1-2 BEFORE features** | Clean foundation prevents technical debt in new code |
| **Do features BEFORE Phase 3-6** | Quick win (~1 week) vs 8-week full refactor; maintains momentum |
| **8-week timeline** | More realistic than 12-16 week alternatives |
| **1 unified LLM client** | Better than 2-client compromise; removes 2,000+ duplicate lines |
| **orchestrator.py → 250 lines** | Aggressive but achievable with proper service extraction |
| **Strangler fig pattern** | Gradual migration safer than big-bang for production system |

---

## Codebase Assessment Summary

**Current State (from kbv2_repofull.txt analysis):**
- **Total LOC:** ~29,055 lines
- **orchestrator.py:** 2,031 lines (god class with 15+ responsibilities)
- **LLM Clients:** 4 implementations (llm_client.py, gateway.py, rotating_llm_client.py, rotation_manager.py) with ~2,000 duplicate lines
- **Magic Numbers:** 50+ scattered values (ports, timeouts, thresholds)
- **Debug Artifacts:** 30+ print() statements in production code
- **Type Issues:** 500+ mypy errors, excessive `Any` usage
- **Error Handling:** 12 empty except blocks, 70+ broad exception handlers

**Critical Files:**
1. `src/knowledge_base/orchestrator.py` - 2,031 lines (lines 27036-29067 in repofull.txt)
2. `src/knowledge_base/clients/llm_client.py` - 707 lines
3. `src/knowledge_base/clients/gateway.py` - 503 lines
4. `src/knowledge_base/common/resilient_gateway/` - 770 lines
5. `src/knowledge_base/domain/ontology_snippets.py` - Needs CRYPTO_TRADING domain

---

## Execution Roadmap

### Phase 0: Pre-Flight (1-2 days) - CRITICAL
**Goal:** Establish baseline before any changes

**Actions:**
```bash
# 1. Create baseline tag
git tag pre-refactor-baseline

# 2. Capture current state
uv run pytest --cov=knowledge_base --cov-report=html -v > baseline_tests.txt
uv run mypy src/knowledge_base --ignore-missing-imports > baseline_mypy.txt
uv run ruff check src/knowledge_base --output-format=json > baseline_lint.json

# 3. Document critical paths
# - Which API endpoints are used in production?
# - Which features are business-critical?
# - What are acceptable downtime windows?
```

**Deliverables:**
- ✅ `baseline_tests.txt` - Current test pass rate
- ✅ `baseline_mypy.txt` - Current type errors
- ✅ `baseline_lint.json` - Current lint errors
- ✅ Git tag: `pre-refactor-baseline`

**Gate:** All existing tests must pass before proceeding

---

### Phase 1: Foundation (2-3 days) - LOW RISK
**Goal:** Extract all magic numbers into centralized constants

**New Files to Create:**

**File 1:** `src/knowledge_base/config/constants.py`
```python
"""Centralized constants for KBV2."""
from typing import Final

# Network
LLM_GATEWAY_PORT: Final[int] = 8087
WEBSOCKET_PORT: Final[int] = 8765
DATABASE_PORT: Final[int] = 5432
LLM_GATEWAY_BASE_URL: Final[str] = "http://localhost:8087/v1/"

# Timeouts
DEFAULT_LLM_TIMEOUT: Final[float] = 120.0
DEFAULT_HTTP_TIMEOUT: Final[float] = 60.0
ROTATION_DELAY: Final[float] = 5.0
INGESTION_TIMEOUT: Final[float] = 3600.0
CONNECTION_TIMEOUT: Final[float] = 60.0

# Rate Limiting
RATE_LIMIT_STATUS_CODES: Final[list[int]] = [429, 503, 529]

# Embedding
EMBEDDING_DIMENSIONS_BGE_M3: Final[int] = 1024
EMBEDDING_MAX_TOKENS: Final[int] = 8191

# Pagination
DEFAULT_PAGE_LIMIT: Final[int] = 50
MAX_PAGE_LIMIT: Final[int] = 1000

# Quality Thresholds
MIN_EXTRACTION_QUALITY_SCORE: Final[float] = 0.5
ENTITY_SIMILARITY_THRESHOLD: Final[float] = 0.85
DOMAIN_CONFIDENCE_THRESHOLD: Final[float] = 0.6
HALLUCINATION_THRESHOLD: Final[float] = 0.3

# Chunking
DEFAULT_CHUNK_SIZE: Final[int] = 1536
DEFAULT_CHUNK_OVERLAP: Final[float] = 0.25
MIN_CHUNK_SIZE: Final[int] = 256

# Extraction
DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.7
DEFAULT_MAX_ITERATIONS: Final[int] = 2
DEFAULT_ENTITY_COUNT: Final[int] = 20

# Search Weights
DEFAULT_VECTOR_WEIGHT: Final[float] = 0.5
DEFAULT_BM25_WEIGHT: Final[float] = 0.5
WEIGHT_TOLERANCE: Final[float] = 1e-6
```

**File 2:** `src/knowledge_base/config/enums.py`
```python
"""Type-safe enumerations."""
from enum import Enum

class DomainType(str, Enum):
    """Domain classifications."""
    TECHNOLOGY = "technology"
    FINANCIAL = "financial"
    CRYPTO_TRADING = "crypto_trading"
    GENERAL = "general"

class ProcessingStatus(str, Enum):
    """Pipeline status."""
    PENDING = "pending"
    PARTITIONING = "partitioning"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    COMPLETED = "completed"
    FAILED = "failed"
```

**Files to Update:**

| File | Lines | Changes |
|------|-------|---------|
| `orchestrator.py` | 27118-27249 | Move DOMAIN_KEYWORDS to new file |
| `clients/cli.py` | Default port 8765 | Replace with `constants.WEBSOCKET_PORT` |
| `clients/websocket_client.py` | Hardcoded ports | Replace with constants |
| `clients/llm_client.py` | Hardcoded URLs/timeouts | Replace with constants |
| `clients/gateway.py` | Hardcoded values | Replace with constants |

**Verification:**
```bash
# Ensure no hardcoded values remain
grep -r "8087\|8765\|120.0\|0.5" src/knowledge_base --include="*.py" | grep -v constants.py
# Should return empty
```

**Gate:** All tests still pass, no new type errors

---

### Phase 2: Cleanup (1-2 days) - LOW RISK
**Goal:** Remove debug artifacts from production code

**Action 2.1: Remove print() Statements**

| File | Lines | Action |
|------|-------|--------|
| `orchestrator.py` | 2022-2023 | Replace with `logger.debug()` |
| `persistence/v1/vector_store.py` | 83, 118, 120 | Replace with `logger.info()` |
| `ingestion/v1/embedding_client.py` | 267-277 | Remove debug prints |

**Action 2.2: Fix Empty Except Blocks (12 instances)**

Pattern to find and fix:
```python
# BEFORE:
try:
    result = await operation()
except:
    pass

# AFTER:
try:
    result = await operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    # Handle appropriately (re-raise, return default, etc.)
```

**Files with empty except blocks:**
- `orchestrator.py:367` - Add logging context
- `storage/hybrid_search.py:464` - Add logging context  
- `gleaning_service.py:754,770` - Add logging context
- `mcp_server.py:129` - Add logging or remove
- `common/gateway.py:93` - Add logging context
- `intelligence/v1/extraction_logging.py:341` - Add logging context

**Action 2.3: Remove Duplicate Imports**

File: `orchestrator.py` lines 27064-27075
```python
# REMOVE duplicate import:
from knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,  # Duplicate - already imported at line 27064
    EntityVerification,     # Duplicate
    RiskLevel,             # Duplicate
)
```

**Verification:**
```bash
# Check for remaining print statements
rg "^\s*print\(" src/knowledge_base --type py | grep -v -E "(cli\.py|example\.py|progress\.py)"
# Should return empty
```

**Gate:** No debug statements in production code, all tests pass

---

### Phase 3: Bitcoin Trading Features (3-5 days) - FEATURE DELIVERY
**Goal:** Implement Bitcoin trading knowledge base features

**Rationale:** Now that codebase is clean (constants extracted, debug removed), new features won't introduce technical debt.

**Feature 3.1: Domain Ontology (Priority: CRITICAL, 45 min)**

**File:** `src/knowledge_base/domain/ontology_snippets.py`

**Action:** Add CRYPTO_TRADING domain (see full spec in btc_trading_kb_implementation_plan_claude_website.md Phase 1.1)

```python
"CRYPTO_TRADING": {
    "keywords": [
        # Core Bitcoin/Crypto
        "bitcoin", "btc", "satoshi", "sats", "cryptocurrency", "crypto",
        "blockchain", "halving", "mining", "hash rate", ...
        # Technical Analysis - Indicators
        "moving average", "ma", "sma", "ema", "rsi", "macd", ...
        # Chart Patterns
        "head and shoulders", "double top", "triangle", "flag", ...
        # Trading Strategies
        "dca", "dollar cost averaging", "hodl", "swing trading", ...
        # Market Structure
        "higher high", "hh", "higher low", "hl", "order block", ...
        # On-Chain Metrics
        "mvrv", "nupl", "sopr", "hodl waves", ...
    ],
    "entity_types": [
        "Cryptocurrency", "Exchange", "Wallet", "BlockchainNetwork",
        "TradingStrategy", "TradingPlan", "EntrySetup", "ExitStrategy",
        "TechnicalIndicator", "ChartPattern", "PriceLevel", "Timeframe",
        "MarketStructure", "LiquidityZone", "OnChainMetric", "MarketCycle",
        "Trader", "TradingBook", "TradingVideo",
    ],
}
```

**Feature 3.2: Extraction Templates (Priority: MEDIUM, 30 min)**

**File:** `src/knowledge_base/extraction/template_registry.py`

**Action:** Add Bitcoin-specific extraction goals to `DEFAULT_GOALS` (see btc_trading_kb_implementation_plan_claude_website.md Phase 1.2)

Goals to add:
1. `technical_indicators` - RSI, MACD, Moving Averages (Priority: 1)
2. `chart_patterns` - Head & Shoulders, Triangles, Flags (Priority: 1)
3. `trading_strategies` - Complete systems with entry/exit (Priority: 2)
4. `market_structure` - Smart money, liquidity zones (Priority: 2)
5. `on_chain_metrics` - MVRV, NUPL, SOPR (Priority: 3)
6. `market_cycles` - Halvings, cycle phases (Priority: 3)
7. `price_levels` - Support/resistance (Priority: 1)
8. `risk_management` - Position sizing, stop losses (Priority: 4)

**Feature 3.3: Batch Ingestion Script (Priority: CRITICAL, 45 min)**

**File:** `scripts/ingest_trading_library.py` (NEW)

**Features:**
- Recursive directory scanning
- Progress tracking with rich console output
- Resume capability (state file)
- Error logging with detailed reports
- File type statistics
- Dry-run mode

**See full implementation:** btc_trading_kb_implementation_plan_claude_website.md Phase 3.1

**Feature 3.4: YouTube Transcript Preprocessor (Priority: LOW, 20 min)**

**File:** `scripts/preprocess_transcript.py` (NEW)

**Features:**
- YAML frontmatter generation
- Filler word removal (um, uh, like, you know)
- Timestamp preservation option
- Sentence segmentation
- Batch processing support

**See full implementation:** btc_trading_kb_implementation_plan_claude_website.md Phase 4

**Feature 3.5: Type Discovery Configuration (Priority: LOW, 15 min)**

**File:** `src/knowledge_base/types/type_discovery.py`

**Action:** Add domain-specific config for CRYPTO_TRADING:
```python
"CRYPTO_TRADING": {
    "min_frequency": 3,           # Lower threshold for trading terms
    "promotion_threshold": 0.82,  # Slightly lower for specialized terms
    "max_new_types": 40,          # Allow more crypto-specific types
    "similarity_threshold": 0.90,  # Stricter similarity
}
```

**Feature 3.6: Query Preprocessing (Priority: MEDIUM, 30 min)**

**File:** `src/knowledge_base/query_api.py`

**Action:** Add trading query preprocessing:
```python
def preprocess_trading_query(query: str, domain: str = "CRYPTO_TRADING") -> Dict:
    """Preprocess trading queries to optimize retrieval."""
    # Detect intent: definition, how_to, comparison, temporal, ranking
    # Extract entity types mentioned
    # Suggest filters based on query type
```

**Feature 3.7: Documentation (Priority: MEDIUM, 45 min)**

**File:** `docs/BITCOIN_TRADING_KB_GUIDE.md` (NEW)

**Contents:**
- Quick start guide
- Ingesting content (books, videos, transcripts)
- Query examples for different research goals
- Directory structure best practices
- Common use cases

**See template:** btc_trading_kb_implementation_plan_claude_website.md Phase 7

**Gate:** All 7 features implemented and tested

---

### Phase 4: LLM Client Consolidation (3-4 days) - MEDIUM RISK
**Goal:** Consolidate 4+ LLM client implementations into 1 unified client

**Current State:**
- `llm_client.py` (707 lines) - Base client with prompting strategies
- `gateway.py` (503 lines) - Gateway client with duplicate models
- `resilient_gateway/` (770 lines) - Resilient wrapper
- `rotating_llm_client.py` (309 lines) - Model rotation
- `rotation_manager.py` (428 lines) - DUPLICATE rotation logic

**Target Architecture:**
```
UnifiedLLMClient (single interface)
├── RetryMiddleware
├── RotationMiddleware
└── CircuitBreakerMiddleware
```

**New Files:**
1. `src/knowledge_base/clients/unified_llm_client.py` (300-400 lines)
2. `src/knowledge_base/clients/middleware/retry_middleware.py`
3. `src/knowledge_base/clients/middleware/rotation_middleware.py`
4. `src/knowledge_base/clients/middleware/circuit_breaker.py`

**Migration Strategy:**
1. Week 1: Create unified client alongside existing ones
2. Week 2: Migrate one service at a time
3. Week 3: Add deprecation warnings to old clients
4. Week 4: Remove old clients after validation

**Files to Deprecate:**
- `clients/llm_client.py` → `unified_llm_client.py`
- `clients/gateway.py` → `unified_llm_client.py`
- `clients/rotating_llm_client.py` → `unified_llm_client.py`
- `clients/rotation_manager.py` → `unified_llm_client.py`
- `common/resilient_gateway/` → `unified_llm_client.py`

**Gate:** All LLM tests pass with new client

---

### Phase 5: God Class Decomposition (5-7 days) - HIGH RISK
**Goal:** Break down 2,031-line orchestrator.py into focused services

**Current Structure (orchestrator.py lines 27036-29067):**
```
IngestionOrchestrator
├── DOMAIN_KEYWORDS (131 lines) - MOVED in Phase 1
├── __init__ (82 lines)
├── _determine_domain() (133 lines)
├── _partition_document() (49 lines)
├── _extract_knowledge() (288 lines)
├── _resolve_entities() (102 lines)
├── _refine_entity_types() (60 lines)
├── _validate_entities_against_schema() (81 lines)
├── _cluster_entities() (45 lines)
├── _embed_content() (89 lines)
├── _generate_reports() (156 lines)
├── _add_to_review_queue() (67 lines)
├── _route_to_review() (78 lines)
├── process_document() (245 lines)
└── _extract_entities_multi_agent() (194 lines)
```

**Target Architecture:**
```
IngestionOrchestrator (~250 lines) - Pure coordinator
├── DomainDetectionService
├── DocumentPipelineService
├── EntityPipelineService
├── QualityAssuranceService
├── EmbeddingService
└── ClusteringService
```

**New Directory Structure:**
```
src/knowledge_base/orchestration/
├── __init__.py
├── orchestrator.py              # Modified: ~250 lines coordinator
├── base_service.py              # Abstract base for all services
├── document_pipeline_service.py # Document processing pipeline
├── entity_pipeline_service.py   # Entity extraction pipeline
├── quality_assurance_service.py # Validation & review
├── domain_detection_service.py  # Domain classification
├── embedding_service.py         # Embedding generation
└── clustering_service.py        # Entity clustering
```

**Service Extraction Plan:**

| Service | Source Lines | Target Lines | Priority |
|---------|--------------|--------------|----------|
| DomainDetectionService | 133 | ~150 | Week 1 |
| DocumentPipelineService | 133 + 89 | ~200 | Week 1 |
| EntityPipelineService | 288 + 102 + 60 + 194 | ~400 | Week 2 |
| QualityAssuranceService | 81 + 67 + 45 + 78 | ~200 | Week 3 |
| EmbeddingService | 89 | ~100 | Week 3 |
| ClusteringService | 45 | ~80 | Week 3 |

**Migration Strategy (CRITICAL):**
1. Keep old orchestrator as `orchestrator_legacy.py`
2. Extract services one at a time
3. Run both orchestrators in parallel with feature flag
4. Compare outputs, log discrepancies
5. Switch default to new orchestrator after 1 week
6. Remove legacy after 2 weeks

**Gate:** New orchestrator produces identical results, no performance regression

---

### Phase 6: Type Safety & Error Handling (3-4 days) - MEDIUM RISK
**Goal:** Achieve mypy strict mode compliance and proper exception hierarchy

**Action 6.1: Exception Hierarchy**

**New File:** `src/knowledge_base/common/exceptions.py`
```python
class KBV2BaseException(Exception):
    """Base exception for all KB errors."""
    def __init__(self, message: str, error_code: str | None = None, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}

class IngestionError(KBV2BaseException): pass
class ExtractionError(KBV2BaseException): pass
class EmbeddingError(KBV2BaseException): pass
class LLMClientError(KBV2BaseException): pass
class ValidationError(KBV2BaseException): pass
class ResolutionError(KBV2BaseException): pass
class ConfigurationError(KBV2BaseException): pass
```

**Action 6.2: Replace Broad Exception Handlers (70+ instances)**

**Priority Files:**
1. `orchestrator.py` - Core ingestion logic
2. `query_api.py` - User-facing API
3. `graph_api.py` - User-facing API
4. `clients/websocket_client.py` - Network communication

**Pattern:**
```python
# Before:
except Exception as e:
    logger.error(f"Failed: {e}")
    return {"status": "error"}

# After:
except DocumentProcessingError as e:
    logger.error(f"Document processing failed: {e}", exc_info=True)
    raise HTTPException(status_code=400, detail=str(e))
except LLMClientError as e:
    logger.error(f"LLM client failed: {e}", exc_info=True)
    raise HTTPException(status_code=503, detail="AI service unavailable")
```

**Action 6.3: Add Type Hints (25+ locations)**

**New File:** `src/knowledge_base/common/types.py`
```python
"""Common type aliases."""
from typing import Callable, Dict, Any, List, Union
from uuid import UUID

ProgressCallback = Callable[[Dict[str, Any]], None]
LogBroadcast = Callable[[str], Any]
EntityId = Union[str, UUID]
DocumentId = Union[str, UUID]
EdgeId = Union[str, UUID]
EntityDict = Dict[str, Any]
EdgeDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
```

**Files to Update:**
- `observability.py` - Replace 8x `Any` types
- `rotation_manager.py` - Replace 2x `Any` types
- `graph_api.py` - Add missing return types
- `entity_typing_service.py` - Move misplaced `import asyncio` to top

**Gate:** <100 mypy errors with `--strict` flag

---

### Phase 7: Testing & Optimization (2-3 days) - LOW RISK
**Goal:** Achieve >80% test coverage and optimize performance

**Action 7.1: Test Structure**
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

**Action 7.2: Performance Optimization**
- Add connection pooling verification
- Check for N+1 database queries
- Add caching for LLM responses (optional)
- Profile ingestion pipeline

**Gate:** >80% test coverage, no performance regression

---

## Implementation Order Summary

| Phase | Duration | Risk | Key Deliverables |
|-------|----------|------|------------------|
| 0. Pre-Flight | 1-2 days | None | Baseline metrics, git tag |
| 1. Foundation | 2-3 days | Low | constants.py, settings.py |
| 2. Cleanup | 1-2 days | Low | No print(), no empty excepts |
| **3. Bitcoin Features** | **3-5 days** | **Low** | **Complete feature delivery** |
| 4. LLM Consolidation | 3-4 days | Medium | unified_llm_client.py |
| 5. Orchestrator | 5-7 days | **High** | 6 new services |
| 6. Type Safety | 3-4 days | Medium | exceptions.py, type hints |
| 7. Testing | 2-3 days | Low | >80% coverage |
| **Total** | **21-29 days** | - | **~6 weeks calendar time** |

**Quick Win Milestone:** After Phase 3 (~1 week), Bitcoin trading features are **fully functional and production-ready**.

---

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Breaking changes | Feature flags, strangler fig pattern, parallel old/new code |
| Test failures | Baseline capture, incremental validation at each gate |
| Performance regression | Benchmark before/after each phase |
| Team disruption | Incremental migration, clear phase boundaries |
| Scope creep | Strict phase gates, no mixing phases |

**Rollback Procedure:**
```bash
# Emergency rollback
git checkout pre-refactor-baseline
git checkout -b rollback-emergency
./deploy_rollback.sh
```

---

## Success Criteria

**After Phase 3 (Quick Win):**
- ✅ Bitcoin trading domain ontology implemented
- ✅ Batch ingestion script working
- ✅ YouTube transcript preprocessor functional
- ✅ All features tested with sample data

**After Phase 7 (Full Refactor):**
- ✅ orchestrator.py < 300 lines
- ✅ 1 unified LLM client (4→1)
- ✅ Zero magic numbers in code
- ✅ Zero print() in production code
- ✅ <100 mypy errors with `--strict`
- ✅ >80% test coverage
- ✅ All tests passing
- ✅ No API breaking changes
- ✅ Performance within 5% of baseline

---

## Command Reference

**Testing:**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=knowledge_base --cov-report=html

# Run specific module
uv run pytest tests/unit/test_orchestrator.py -v
```

**Type Checking:**
```bash
# Check all files
uv run mypy src/knowledge_base --ignore-missing-imports

# Check specific file
uv run mypy src/knowledge_base/orchestrator.py --strict
```

**Linting:**
```bash
# Check all files
uv run ruff check src/knowledge_base

# Fix auto-fixable issues
uv run ruff check --fix src/knowledge_base

# Format code
uv run ruff format src/knowledge_base
```

**Pre-commit (recommended):**
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Dependencies

**Required Tools (add to pyproject.toml):**
```toml
[tool.uv.dev-dependencies]
ruff = "^0.1.0"
mypy = "^1.7.0"
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"
pre-commit = "^3.5.0"
```

**Install:**
```bash
uv add --dev ruff mypy pytest pytest-cov pytest-asyncio pre-commit
```

---

## Next Session Instructions

**When starting next session:**

1. **Read this document first** - Sets context and decisions
2. **Verify baseline exists** - Check for `pre-refactor-baseline` git tag
3. **Start with Phase 0** - If baseline not captured, do it first
4. **Execute Phase 1-2** - Foundation and cleanup (~3-5 days)
5. **Deliver Phase 3** - Bitcoin features (~3-5 days)
6. **Report progress** - Update this document with actual results

**Do NOT:**
- Skip Phase 0 (baseline capture is critical for rollback)
- Mix phases (complete each gate before proceeding)
- Add new features during refactor (violates strangler fig pattern)
- Delete old code immediately (use deprecation warnings first)

---

## Reference Documents

1. **kbv2_comprehensive_refactoring_plan_kimi.md** - Complete 8-week refactoring plan with detailed code examples
2. **btc_trading_kb_implementation_plan_claude_website.md** - Bitcoin trading features specification
3. **kbv2_repofull.txt** - Full codebase snapshot (lines 1-29,055)
4. **plan_comparison_summary_kimi.md** - Analysis of alternative refactoring approaches

---

## Notes for Future Sessions

**Key Decisions Made:**
- Hybrid approach: Phase 1-2 → Features → Phase 3-7
- 8-week timeline (realistic, not 12-16 weeks)
- 1 unified LLM client (not 2)
- orchestrator.py → 250 lines (aggressive but achievable)
- Strangler fig pattern for safe migration
- Phase gates mandatory (no skipping)

**Known Challenges:**
- orchestrator.py decomposition is HIGH RISK - use parallel execution
- LLM client consolidation affects many files - migrate gradually
- Type safety improvements touch 25+ files - batch by module
- Bitcoin features depend on clean foundation - Phase 1-2 first

**Recommended Team:**
- 1 developer: 6-7 weeks
- 2 developers: 3-4 weeks  
- 3 developers: 2-3 weeks

---

**Document Version:** 1.0  
**Last Updated:** 2026-01-31  
**Status:** Ready for Execution  
**Confidence Level:** 90% success if followed rigorously

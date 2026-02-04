# Step-by-Step Implementation Plan: Bitcoin Trading Knowledge Base

## Current State Analysis

### ✅ Already Implemented
1. **CRYPTO_TRADING domain in ontology_snippets.py** - Comprehensive keywords (~150+) and entity types already exist
2. **ingest_trading_library.py** - Basic batch ingestion script exists
3. **preprocess_transcript.py** - Basic transcript preprocessing exists
4. **GleaningService** - Full adaptive extraction with guided prompts
5. **TypeDiscovery** - Basic adaptive type discovery

### ❌ Missing / Needs Enhancement
1. **CRYPTO_TRADING extraction goals** - Domain exists but no extraction goals in template_registry.py
2. **Domain detection for CRYPTO_TRADING** - Not in guided_extractor.py detection
3. **Domain-specific type discovery config** - No crypto-specific thresholds
4. **Enhanced transcript preprocessing** - Basic, needs enhancement
5. **Enhanced batch ingestion** - Basic, needs resume/progress features
6. **User documentation** - Missing

---

## Implementation Steps

### Step 1: Add CRYPTO_TRADING Extraction Goals
**File**: `src/knowledge_base/extraction/template_registry.py`

Add extraction goals for CRYPTO_TRADING domain. This is **CRITICAL** - without goals, guided extraction won't work properly.

**Changes needed**:
- Add `CRYPTO_TRADING` key to `DEFAULT_GOALS` dictionary
- Add goals for: technical_indicators, chart_patterns, trading_strategies, market_structure, on_chain_metrics, price_levels, risk_management

**Estimated time**: 20 minutes

**Verification**:
```bash
uv run python -c "
from knowledge_base.extraction.template_registry import get_default_goals
goals = get_default_goals('CRYPTO_TRADING')
print(f'Found {len(goals)} extraction goals for CRYPTO_TRADING')
for goal in goals:
    print(f'  - {goal.name}: {goal.description}')
"
```

---

### Step 2: Add CRYPTO_TRADING to Domain Detection
**File**: `src/knowledge_base/extraction/guided_extractor.py`

Update `_detect_domain()` method to recognize CRYPTO_TRADING indicators.

**Changes needed**:
- Add CRYPTO_TRADING indicators to `domain_indicators` dictionary
- Keywords: bitcoin, btc, cryptocurrency, crypto, rsi, macd, trading, etc.

**Estimated time**: 10 minutes

**Verification**:
```bash
uv run python -c "
from knowledge_base.extraction.guided_extractor import GuidedExtractor
from knowledge_base.common.gateway import GatewayClient

# Mock test - will need actual gateway for full test
ge = GuidedExtractor.__new__(GuidedExtractor)
ge.templates = None
import asyncio
text = 'Bitcoin RSI shows overbought conditions at 70. The head and shoulders pattern indicates a reversal.'
result = asyncio.run(ge._detect_domain(text))
print(f'Detected domain: {result}')
"
```

---

### Step 3: Add Domain-Specific Type Discovery Config
**File**: `src/knowledge_base/types/type_discovery.py`

Add CRYPTO_TRADING-specific configuration for type discovery with lower thresholds for trading terminology.

**Changes needed**:
- Add `DOMAIN_SPECIFIC_CONFIGS` dictionary
- Add `get_config_for_domain()` function
- Lower `min_frequency` and `promotion_threshold` for crypto terms

**Estimated time**: 10 minutes

**Verification**:
```bash
uv run python -c "
from knowledge_base.types.type_discovery import get_config_for_domain
config = get_config_for_domain('CRYPTO_TRADING')
print(f'CRYPTO_TRADING config: {config}')
"
```

---

### Step 4: Enhance Transcript Preprocessor
**File**: `src/knowledge_base/scripts/preprocess_transcript.py`

Enhance the basic preprocessor with:
- YAML frontmatter support
- Speaker/date/URL metadata
- Filler word removal
- Better text segmentation
- Batch processing mode

**Estimated time**: 20 minutes

**Verification**:
```bash
uv run python -m knowledge_base.scripts.preprocess_transcript --help
```

---

### Step 5: Enhance Batch Ingestion Script
**File**: `src/knowledge_base/scripts/ingest_trading_library.py`

Enhance with:
- Progress bars (rich library)
- Resume capability with state file
- Error logging
- File type statistics
- Dry-run mode

**Estimated time**: 30 minutes

**Verification**:
```bash
uv run python -m knowledge_base.scripts.ingest_trading_library --help
```

---

### Step 6: Create User Documentation
**File**: `docs/BITCOIN_TRADING_KB_GUIDE.md`

Comprehensive user guide including:
- Quick start guide
- Ingestion examples (single file, directory, YouTube)
- Query examples
- Best practices
- Troubleshooting

**Estimated time**: 30 minutes

---

## Implementation Order (Priority)

| Step | Priority | Effort | Description |
|------|----------|--------|-------------|
| 1 | 🔴 Critical | 20 min | Add CRYPTO_TRADING extraction goals |
| 2 | 🔴 Critical | 10 min | Add domain detection for crypto |
| 3 | 🟡 Medium | 10 min | Domain-specific type discovery |
| 4 | 🟡 Medium | 20 min | Enhanced transcript preprocessor |
| 5 | 🟡 Medium | 30 min | Enhanced batch ingestion |
| 6 | 🟢 Low | 30 min | User documentation |

**Total estimated time**: ~2 hours

---

## Verification Commands

After each step, run the verification command. Full system test:

```bash
# 1. Test extraction goals
uv run python -c "
from knowledge_base.extraction.template_registry import get_default_goals
goals = get_default_goals('CRYPTO_TRADING')
assert len(goals) >= 5, 'Need at least 5 extraction goals'
print('✓ CRYPTO_TRADING goals loaded')
"

# 2. Test domain detection
uv run python -c "
from knowledge_base.extraction.guided_extractor import GuidedExtractor
ge = GuidedExtractor.__new__(GuidedExtractor)
ge.templates = None
import asyncio
text = 'Bitcoin shows RSI overbought at 70 with head and shoulders pattern'
result = asyncio.run(ge._detect_domain(text))
assert result == 'CRYPTO_TRADING', f'Expected CRYPTO_TRADING, got {result}'
print('✓ Domain detection works')
"

# 3. Test type discovery config
uv run python -c "
from knowledge_base.types.type_discovery import get_config_for_domain
config = get_config_for_domain('CRYPTO_TRADING')
assert config.min_frequency < 5, 'Crypto should have lower frequency threshold'
print('✓ Type discovery config loaded')
"

# 4. End-to-end test
uv run python -c "
from knowledge_base.orchestrator import IngestionOrchestrator
import asyncio

async def test():
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    # Create test content
    test_text = '''
    # Bitcoin Technical Analysis
    
    Bitcoin (BTC) is showing a classic head and shoulders pattern on the daily chart.
    The RSI indicator is at 65, approaching overbought territory.
    Key support is at \$42,000 with resistance at \$48,000.
    
    The EMA 20 crossed above EMA 50, indicating bullish momentum.
    Stop loss should be placed below the neckline at \$40,000.
    '''
    
    doc = await orchestrator.process_document(
        file_path=None,
        document_name='test_crypto_doc',
        content=test_text,
        domain='CRYPTO_TRADING'
    )
    
    print(f'✓ Processed document: {doc.id}')
    print(f'  Entities: {len(doc.entities)}')
    print(f'  Relationships: {len(doc.relationships)}')
    
    await orchestrator.shutdown()

asyncio.run(test())
"
```

---

## Rollback Plan

If any step causes issues:
1. Each file modification can be reverted with `git checkout <file>`
2. The system will fallback to existing FINANCIAL domain if CRYPTO_TRADING is broken
3. Ontology keywords still work even without extraction goals (general extraction applies)

---

## Dependencies

No new packages needed - everything uses existing dependencies:
- `rich` - Already in requirements
- `pydantic` - Already in requirements
- `pytest` - Already in requirements

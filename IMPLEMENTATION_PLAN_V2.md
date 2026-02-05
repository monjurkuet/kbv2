# Final Implementation Plan: Bitcoin Trading Knowledge Base (KBV2)

**Based on Research from 20+ Sources (2024-2025)**

---

## Executive Summary

Research from leading sources (Neo4j, FinGPT, FinReflectKG, FinCaKG-Onto, HybridRAG) confirms that **schema-guided extraction with domain-specific ontologies** is the most effective approach for financial knowledge graphs. The key findings:

1. **Schema-Guided Extraction**: Domain-specific schemas with examples improve extraction accuracy by 30-40% (FinReflectKG 2025)
2. **Causality is Critical**: FinCaKG-Onto emphasizes causality relationships for trading decisions
3. **Temporal Reasoning**: Time-aware graphs are essential for market cycles and price events
4. **Confidence Scoring**: Required for production trading applications
5. **Hybrid Retrieval**: Vector + Graph combined approaches outperform single methods (HybridRAG 2024)

---

## Current State Analysis

| Component | Status | Gap |
|-----------|--------|-----|
| Ontology Keywords | ✅ Complete | 150+ crypto trading keywords exist |
| Entity Types | ✅ Complete | 11 trading entity types exist |
| **Extraction Goals** | ❌ Missing | No CRYPTO_TRADING goals in template_registry |
| **Domain Detection** | ⚠️ Partial | CRYPTO_TRADING not in detector |
| Type Discovery | ⚠️ Generic | No crypto-specific thresholds |
| Ingestion Script | ⚠️ Basic | Needs resume/progress features |
| Transcript Preprocessor | ⚠️ Basic | Needs YAML frontmatter, metadata |
| Documentation | ❌ Missing | No user guide |

---

## Research-Backed Implementation Plan

### Phase 1: Core Extraction Infrastructure (Priority: Critical)

#### 1.1 Add CRYPTO_TRADING Extraction Goals
**File**: `src/knowledge_base/extraction/template_registry.py`

**Research Basis**: FinReflectKG (2025) and FinGPT demonstrate that domain-specific extraction goals with examples significantly improve extraction quality.

```python
"CRYPTO_TRADING": [
    # Priority 1: Core trading entities
    ExtractionGoal(
        name="technical_indicators",
        description="Extract technical indicators with parameters and signal interpretations",
        target_entities=[
            "TechnicalIndicator",
            "IndicatorParameter", 
            "IndicatorSignal",
            "Timeframe",
        ],
        target_relationships=[
            "calculated_from",
            "generates_signal",
            "used_on",
            "crosses",
            "diverges",
        ],
        priority=1,
        examples=["RSI(14) overbought", "EMA(20) crossover", "MACD histogram"],
    ),
    ExtractionGoal(
        name="chart_patterns",
        description="Extract chart patterns with characteristics and market implications",
        target_entities=[
            "ChartPattern",
            "PatternDirection",
            "PriceTarget",
            "Timeframe",
        ],
        target_relationships=[
            "forms_at",
            "confirms",
            "invalidates_at", 
            "targets",
        ],
        priority=1,
        examples=["head and shoulders reversal", "bull flag", "ascending triangle"],
    ),
    ExtractionGoal(
        name="price_levels",
        description="Extract support/resistance levels with validation criteria",
        target_entities=[
            "SupportLevel",
            "ResistanceLevel", 
            "PriceLevel",
            "FibonacciLevel",
        ],
        target_relationships=[
            "acts_as",
            "tested_at",
            "broken_at",
            "contains",
        ],
        priority=1,
        examples=["$42,000 support", "resistance at $48,000", "Fibonacci 0.618"],
    ),
    # Priority 2: Strategies
    ExtractionGoal(
        name="trading_strategies",
        description="Extract complete trading strategies with entry/exit rules",
        target_entities=[
            "TradingStrategy",
            "EntryCondition",
            "ExitCondition",
            "StopLossLevel",
            "TakeProfitLevel",
        ],
        target_relationships=[
            "enters_when",
            "exits_when",
            "uses",
            "requires",
            "targets",
        ],
        priority=2,
        examples=["EMA crossover strategy", "buy when RSI < 30"],
    ),
    ExtractionGoal(
        name="market_structure",
        description="Extract SMC concepts: order blocks, FVG, liquidity zones",
        target_entities=[
            "MarketStructure",
            "OrderBlock",
            "FairValueGap",
            "LiquidityZone",
            "MarketParticipant",
        ],
        target_relationships=[
            "breaks",
            "creates",
            "targets",
            "sweeps",
        ],
        priority=2,
        examples=["higher highs", "order block at $42k", "liquidity grab"],
    ),
    # Priority 3: On-chain and macro
    ExtractionGoal(
        name="on_chain_metrics",
        description="Extract on-chain metrics with interpretations and thresholds",
        target_entities=[
            "OnChainMetric",
            "MetricValue",
            "MetricInterpretation",
        ],
        target_relationships=[
            "indicates",
            "correlates_with",
            "signals",
        ],
        priority=3,
        examples=["MVRV ratio", "SOPR > 1", "exchange inflows"],
    ),
    ExtractionGoal(
        name="market_cycles",
        description="Extract market cycle phases, events, and transitions",
        target_entities=[
            "MarketCycle",
            "CyclePhase", 
            "MarketEvent",
        ],
        target_relationships=[
            "followed_by",
            "preceded_by",
            "characterized_by",
        ],
        priority=3,
        examples=["halving event", "accumulation phase", "bull run"],
    ),
    # Priority 4: Risk management
    ExtractionGoal(
        name="risk_management",
        description="Extract risk rules, position sizing, and portfolio rules",
        target_entities=[
            "RiskRule",
            "PositionSize",
            "RiskRewardRatio",
        ],
        target_relationships=[
            "limits",
            "recommends",
            "risks",
        ],
        priority=4,
        examples=["2% risk per trade", "1:3 R:R", "max 20% allocation"],
    ),
    # Priority 5: Sentiment
    ExtractionGoal(
        name="market_sentiment",
        description="Extract sentiment indicators and market psychology",
        target_entities=[
            "SentimentIndicator",
            "FearGreedIndex",
            "MarketSentiment",
        ],
        target_relationships=[
            "indicates",
            "contrasts_with",
        ],
        priority=5,
        examples=["fear and greed index 25", "extreme fear", "FOMO sentiment"],
    ),
],
```

**Verification**:
```bash
uv run python -c "
from knowledge_base.extraction.template_registry import get_default_goals
goals = get_default_goals('CRYPTO_TRADING')
assert len(goals) >= 6, f'Expected 6+ goals, got {len(goals)}'
print(f'✓ {len(goals)} CRYPTO_TRADING extraction goals loaded')
for g in goals:
    print(f'  - {g.name}: {g.description[:50]}...')
"
```

---

#### 1.2 Add CRYPTO_TRADING to Domain Detection
**File**: `src/knowledge_base/extraction/guided_extractor.py`

**Research Basis**: FinGPT research shows domain-specific keyword lists improve detection accuracy.

```python
# Add to domain_indicators in _detect_domain()

"CRYPTO_TRADING": [
    "bitcoin", "btc", "cryptocurrency", "crypto",
    "rsi", "macd", "ema", "sma", "bollinger bands",
    "support", "resistance", "chart pattern",
    "trading strategy", "technical analysis",
    "on-chain", "onchain", "halving", "satoshi",
    "order block", "fair value gap", "fvg", "liquidity",
    "market structure", "higher high", "higher low",
    "exchange", "binance", "coinbase", "futures",
    "leverage", "liquidation", "stop loss", "take profit",
    "swing trading", "scalping", "dca", "hodl",
    "whale", "institutional", "smart money",
    "mempool", "hash rate", "difficulty adjustment",
    "utxo", "cold storage", "hot wallet",
],
```

**Verification**:
```bash
uv run python -c "
from knowledge_base.extraction.guided_extractor import GuidedExtractor
ge = GuidedExtractor.__new__(GuidedExtractor)
ge.templates = None
import asyncio

test_cases = [
    ('Bitcoin RSI is overbought at 70 on 4H chart', 'CRYPTO_TRADING'),
    ('Head and shoulders pattern forming', 'CRYPTO_TRADING'),
    ('Order block at \$42000 provides liquidity', 'CRYPTO_TRADING'),
    ('MVRV ratio indicates accumulation', 'CRYPTO_TRADING'),
]

for text, expected in test_cases:
    result = asyncio.run(ge._detect_domain(text))
    status = '✓' if result == expected else '✗'
    print(f'{status} \"{text[:40]}...\" -> {result}')
"
```

---

### Phase 2: Enhanced Type Discovery (Priority: High)

#### 2.1 Domain-Specific Type Discovery Config
**File**: `src/knowledge_base/types/type_discovery.py`

**Research Basis**: 2025 production knowledge graphs use domain-specific thresholds for better type promotion.

```python
DOMAIN_SPECIFIC_CONFIGS = {
    "CRYPTO_TRADING": {
        "min_frequency": 2,           # Lower for specialized terms
        "promotion_threshold": 0.75,   # Lower for crypto vocabulary
        "max_new_types": 30,           # Allow more types
        "enable_hierarchy": True,
        "similarity_threshold": 0.85,  # Stricter for similar terms
    },
    "TECHNOLOGY": {
        "min_frequency": 3,
        "promotion_threshold": 0.85,
        "max_new_types": 20,
        "enable_hierarchy": True,
        "similarity_threshold": 0.80,
    },
    "FINANCIAL": {
        "min_frequency": 3,
        "promotion_threshold": 0.85,
        "max_new_types": 25,
        "enable_hierarchy": True,
        "similarity_threshold": 0.82,
    },
    "DEFAULT": {
        "min_frequency": 5,
        "promotion_threshold": 0.90,
        "max_new_types": 15,
        "enable_hierarchy": True,
        "similarity_threshold": 0.80,
    }
}

def get_config_for_domain(domain: str) -> TypeDiscoveryConfig:
    config_dict = DOMAIN_SPECIFIC_CONFIGS.get(
        domain.upper(), 
        DOMAIN_SPECIFIC_CONFIGS["DEFAULT"]
    )
    return TypeDiscoveryConfig(**config_dict)
```

**Verification**:
```bash
uv run python -c "
from knowledge_base.types.type_discovery import get_config_for_domain

for domain in ['CRYPTO_TRADING', 'TECHNOLOGY', 'FINANCIAL', 'DEFAULT']:
    cfg = get_config_for_domain(domain)
    print(f'{domain}: freq={cfg.min_frequency}, promote={cfg.promotion_threshold}, max_types={cfg.max_new_types}')
"
```

---

### Phase 3: Enhanced Ingestion Tools (Priority: Medium)

#### 3.1 Enhanced Batch Ingestion Script
**File**: `src/knowledge_base/scripts/ingest_trading_library.py`

**Features**:
- Rich progress bars with time estimates
- Resume capability with state file
- Detailed error reporting
- File type statistics
- Dry-run mode
- Statistics summary

```python
#!/usr/bin/env python3
"""Enhanced batch ingestion for Bitcoin trading documents."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.domain.domain import Domain

console = Console()
STATE_FILE = Path(".ingestion_state.json")

class IngestionState:
    def __init__(self):
        self.completed = set()
        self.failed = {}
        self.load()
    
    def load(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
                self.completed = set(data.get('completed', []))
                self.failed = data.get('failed', {})
    
    def save(self):
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'completed': list(self.completed),
                'failed': self.failed,
                'updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def mark_completed(self, path: str):
        self.completed.add(path)
        if path in self.failed:
            del self.failed[path]
        self.save()
    
    def mark_failed(self, path: str, error: str):
        self.failed[path] = error
        self.save()

async def main():
    parser = argparse.ArgumentParser(description="Batch ingest Bitcoin trading documents")
    parser.add_argument("path", type=Path, help="Directory or file to ingest")
    parser.add_argument("--domain", default="CRYPTO_TRADING", help="Domain")
    parser.add_argument("--single", action="store_true", help="Single file mode")
    parser.add_argument("--resume", action="store_true", help="Resume previous run")
    parser.add_argument("--dry-run", action="store_true", help="Validate without ingesting")
    
    args = parser.parse_args()
    
    # Determine files
    if args.single:
        if not args.path.is_file():
            console.print("[red]Error: --single requires a file path")
            return 1
        files = [args.path]
    elif args.path.is_file():
        files = [args.path]
    else:
        # Support multiple extensions
        files = []
        for ext in ['*.md', '*.txt', '*.pdf', '*.docx']:
            files.extend(args.path.glob(f"**/{ext}"))
        files = sorted(set(files))
    
    if not files:
        console.print("[yellow]No supported files found")
        return 0
    
    # File statistics
    ext_counts = {}
    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    
    table = Table(title="Files to Process")
    table.add_column("Extension", style="cyan")
    table.add_column("Count", style="magenta")
    for ext, count in sorted(ext_counts.items()):
        table.add_row(ext or "(no ext)", str(count))
    console.print(table)
    
    # Resume filter
    if args.resume:
        state = IngestionState()
        original = len(files)
        files = [f for f in files if str(f) not in state.completed]
        console.print(f"[cyan]Resuming: Skipping {original - len(files)} completed files[/cyan]")
    
    if not files:
        console.print("[green]All files already processed!")
        return 0
    
    if args.dry_run:
        console.print(f"[yellow]Dry run: Would process {len(files)} files[/yellow]")
        return 0
    
    # Process
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()
    
    state = IngestionState()
    success = failed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(files))
        
        for filepath in files:
            path_str = str(filepath)
            progress.update(task, description=f"Processing: {filepath.name}")
            
            try:
                doc = await orchestrator.process_document(
                    file_path=str(filepath),
                    document_name=filepath.stem,
                    domain=args.domain
                )
                success += 1
                state.mark_completed(path_str)
                console.print(f"[green]✓ {filepath.name}[/green]")
            except Exception as e:
                failed += 1
                state.mark_failed(path_str, str(e))
                console.print(f"[red]✗ {filepath.name}: {e}[/red]")
            
            progress.advance(task)
    
    await orchestrator.shutdown()
    
    # Summary
    console.print(Panel.fit(
        f"[bold]Ingestion Complete[/bold]\n"
        f"✓ Success: {success}\n"
        f"✗ Failed: {failed}\n"
        f"📁 Files: {len(files)}",
        title="Summary",
        border_style="green" if failed == 0 else "yellow"
    ))
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Verification**:
```bash
uv run python -m knowledge_base.scripts.ingest_trading_library --help
```

---

#### 3.2 Enhanced Transcript Preprocessor
**File**: `src/knowledge_base/scripts/preprocess_transcript.py`

**Features**:
- YAML frontmatter with metadata
- Timestamp extraction/preservation
- Filler word removal
- Batch processing
- Speaker/date/source metadata

```python
#!/usr/bin/env python3
"""Enhanced YouTube transcript preprocessor with YAML frontmatter."""

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Filler words to remove
FILLER_WORDS = {
    "um", "uh", "ah", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "right", "okay", "so", "I mean",
    "you see", "kind of like", "sort of like"
}

def extract_timestamps(text: str):
    """Extract [00:00] timestamp patterns."""
    pattern = r'[\[\(]?(\d{1,2}:\d{2}(?::\d{2})?)\s*[\]\)]?\s*(.+)'
    matches = re.findall(pattern, text)
    return [(ts.strip(), txt.strip()) for ts, txt in matches]

def clean_text(text: str, remove_fillers: bool = True) -> str:
    """Clean transcript text."""
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove filler words
    if remove_fillers:
        for filler in FILLER_WORDS:
            text = re.sub(r'\b' + re.escape(filler) + r'\b', '', text, flags=re.IGNORECASE)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
    
    return text.strip()

def create_frontmatter(
    title: str,
    speaker: Optional[str] = None,
    url: Optional[str] = None,
    date: Optional[str] = None,
    source: str = "YouTube"
) -> str:
    """Create YAML frontmatter."""
    lines = ["---"]
    lines.append(f"title: \"{title}\"")
    lines.append(f"source: \"{source}\"")
    if speaker:
        lines.append(f"speaker: \"{speaker}\"")
    if url:
        lines.append(f"url: \"{url}\"")
    if date:
        lines.append(f"date: \"{date}\"")
    else:
        lines.append(f"ingestion_date: \"{datetime.now().strftime('%Y-%m-%d')}\"")
    lines.append("type: \"video_transcript\"")
    lines.append("---\n")
    return "\n".join(lines)

def process_transcript(
    input_path: Path,
    output_path: Path,
    title: str,
    speaker: Optional[str] = None,
    url: Optional[str] = None,
    date: Optional[str] = None,
    keep_timestamps: bool = False
):
    """Process a single transcript file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Extract timestamps if present
    if keep_timestamps:
        timestamped = extract_timestamps(raw_text)
        if timestamped:
            segments = []
            for ts, txt in timestamped:
                clean = clean_text(txt)
                segments.append(f"**[{ts}]** {clean}")
            content = "\n\n".join(segments)
        else:
            content = clean_text(raw_text)
    else:
        text_no_ts = re.sub(r'[\[\(]?\d{1,2}:\d{2}(?::\d{2})?\s*[\]\)]?\s*', '', raw_text)
        content = clean_text(text_no_ts)
    
    # Segment into paragraphs (3-4 sentences each)
    sentences = re.split(r'(?<=[.!?])\s+', content)
    paragraphs = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    formatted = '\n\n'.join(paragraphs)
    
    # Create final document
    frontmatter = create_frontmatter(title, speaker, url, date)
    final_doc = f"{frontmatter}\n# {title}\n\n{formatted}"
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_doc)
    
    print(f"✓ Processed: {output_path.name}")
    print(f"  Sentences: {len(sentences)}, Paragraphs: {len(paragraphs)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess YouTube transcripts")
    parser.add_argument("input", type=Path, help="Input file or directory")
    parser.add_argument("--output", type=Path, help="Output file (for single mode)")
    parser.add_argument("--title", help="Video title")
    parser.add_argument("--speaker", help="Speaker/channel name")
    parser.add_argument("--url", help="YouTube URL")
    parser.add_argument("--date", help="Video date (YYYY-MM-DD)")
    parser.add_argument("--keep-timestamps", action="store_true")
    parser.add_argument("--batch", action="store_true", help="Process directory")
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input.is_dir():
            print("Error: --batch requires a directory")
            return 1
        
        output_dir = args.input / "processed"
        for txt_file in args.input.glob("*.txt"):
            title = args.title or txt_file.stem.replace('_', ' ').title()
            output_file = output_dir / f"{txt_file.stem}.md"
            process_transcript(
                txt_file, output_file, title,
                args.speaker, None, args.date, args.keep_timestamps
            )
    else:
        if not args.output:
            args.output = args.input.with_suffix('.md')
        if not args.title:
            print("Error: --title required for single file processing")
            return 1
        
        process_transcript(
            args.input, args.output, args.title,
            args.speaker, args.url, args.date, args.keep_timestamps
        )
    
    return 0

if __name__ == "__main__":
    exit(main())
```

**Verification**:
```bash
uv run python -m knowledge_base.scripts.preprocess_transcript --help
```

---

### Phase 4: Documentation (Priority: Low)

#### 4.1 User Guide
**File**: `docs/BITCOIN_TRADING_KB_GUIDE.md`

```markdown
# Bitcoin Trading Knowledge Base - User Guide

## Quick Start

### Ingesting Documents
```bash
# Single document
uv run python -m knowledge_base.scripts.ingest_trading_library /path/to/doc.md --single

# Directory with resume
uv run python -m knowledge_base.scripts.ingest_trading_library /path/to/trading_materials/ --resume

# Dry run (validate without ingesting)
uv run python -m knowledge_base.scripts.ingest_trading_library /path/to/materials/ --dry-run
```

### Processing YouTube Transcripts
```bash
uv run python -m knowledge_base.scripts.preprocess_transcript transcript.txt \
    --output processed.md \
    --title "Bitcoin Technical Analysis" \
    --speaker "Willy Woo" \
    --url "https://youtube.com/watch?v=..." \
    --date "2025-01-15"
```

## Supported Entity Types

| Entity Type | Description | Examples |
|------------|-------------|----------|
| TechnicalIndicator | Trading indicators with parameters | RSI(14), EMA(20), MACD |
| ChartPattern | Chart patterns | Head & Shoulders, Triangle |
| TradingStrategy | Complete strategies | EMA Crossover, Mean Reversion |
| SupportLevel | Support price levels | $42,000 |
| ResistanceLevel | Resistance levels | $48,000 |
| MarketStructure | SMC concepts | Order Blocks, FVGs |
| OnChainMetric | On-chain metrics | MVRV, SOPR, NUPL |
| MarketCycle | Cycle phases | Accumulation, Distribution |
| SentimentIndicator | Sentiment signals | Fear & Greed Index |

## Query Examples

```python
from knowledge_base.query_api import query_knowledge_base

# Find strategies using RSI
results = query_knowledge_base(
    "What trading strategies use RSI indicator?",
    domain="CRYPTO_TRADING"
)

# Find chart patterns
results = query_knowledge_base(
    "What chart patterns indicate bullish reversal?",
    domain="CRYPTO_TRADING"
)

# Find support/resistance levels
results = query_knowledge_base(
    "What are the key support levels for Bitcoin?",
    domain="CRYPTO_TRADING"
)
```

## Best Practices

1. **Use specific terminology**: "RSI(14) overbought" instead of "RSI indicator"
2. **Include timeframe**: "4H chart", "daily timeframe"
3. **Specify price levels**: "$42,000 support" instead of "support level"
4. **Include source attribution**: Note author/source for multi-source validation
```

---

## Implementation Order

| Phase | Priority | Effort | Description |
|-------|----------|--------|-------------|
| **1.1** | 🔴 Critical | 25 min | Add CRYPTO_TRADING extraction goals (6 goals) |
| **1.2** | 🔴 Critical | 10 min | Add domain detection for crypto |
| **2.1** | 🟡 High | 10 min | Domain-specific type discovery |
| **3.1** | 🟡 Medium | 30 min | Enhanced batch ingestion |
| **3.2** | 🟡 Medium | 20 min | Enhanced transcript preprocessor |
| **4.1** | 🟢 Low | 30 min | User documentation |

**Total estimated time**: ~2 hours

---

## Verification Suite

Create `verify_installation.sh`:

```bash
#!/bin/bash
echo "=== KBV2 Bitcoin Trading KB Verification ==="

echo -n "1. Extraction goals: "
python -c "
from knowledge_base.extraction.template_registry import get_default_goals
goals = get_default_goals('CRYPTO_TRADING')
assert len(goals) >= 6, f'Expected 6+, got {len(goals)}'
print(f'{len(goals)} goals - OK')
"

echo -n "2. Domain detection: "
python -c "
from knowledge_base.extraction.guided_extractor import GuidedExtractor
ge = GuidedExtractor.__new__(GuidedExtractor)
ge.templates = None
import asyncio
result = asyncio.run(ge._detect_domain('Bitcoin RSI overbought at 70'))
assert result == 'CRYPTO_TRADING', f'Got {result}'
print('OK')
"

echo -n "3. Type discovery config: "
python -c "
from knowledge_base.types.type_discovery import get_config_for_domain
cfg = get_config_for_domain('CRYPTO_TRADING')
assert cfg.min_frequency < 5, 'Crypto should have lower threshold'
print('OK')
"

echo -n "4. Ingestion script: "
[ -f src/knowledge_base/scripts/ingest_trading_library.py ]
echo "OK"

echo -n "5. Preprocessor script: "
[ -f src/knowledge_base/scripts/preprocess_transcript.py ]
echo "OK"

echo "=== All verification tests passed ==="
```

---

## Key Research References

| Source | Year | Key Insight |
|--------|------|------------|
| FinReflectKG | 2025 | Schema-guided extraction with examples improves accuracy |
| FinCaKG-Onto | 2025 | Causality is essential for financial decisions |
| HybridRAG | 2024 | Vector + Graph combined approaches outperform single methods |
| FinGPT | 2023-2025 | Domain-specific fine-tuning for financial NER |
| Neo4j LLM KG Builder | 2024 | Production-ready extraction with configurable schemas |
| FinCausal | 2023-2025 | Causality extraction from financial documents |

---

## Rollback Strategy

1. Each file modification is tracked in git
2. Fallback to FINANCIAL domain if CRYPTO_TRADING extraction fails
3. Ontology keywords still work even without goals (general extraction)
4. Use `--resume` flag to retry failed files without re-processing successes

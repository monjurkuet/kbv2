# Trading Intelligence System - Complete Refactor Plan

## Executive Summary

Build an LLM-powered system that:
1. Processes YouTube videos from Chart Champions & ECKrown
2. Extracts trading levels, patterns, signals, and market outlook
3. Monitors real-time market conditions against extracted triggers
4. Processes course materials (PDFs + videos)
5. Alerts when conditions are met

---

## Part 1: System Overview

### Data Sources

| Channel | Content Type | Update Frequency | Key Data to Extract |
|---------|-------------|-----------------|---------------------|
| **Chart Champions** | Daily analysis, Trade setups | Daily | Price levels, patterns, entry/exit, risk management |
| **ECKrown** | Market outlook, Weekly recaps | Weekly | Long-term outlook, macro views, BTC levels |
| **Chart Champions Course** | PDFs + Videos | Static | Strategies, patterns, educational content |

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRADING INTELLIGENCE SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              DATA INGESTION LAYER                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │   │
│  │  │ Chart Champions │  │    ECKrown     │  │ Course Materials│   │   │
│  │  │  YouTube API   │  │   YouTube API  │  │  PDF/Video      │   │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘   │   │
│  │           │                    │                    │              │   │
│  └──────────┼────────────────────┼────────────────────┼──────────────┘   │
│              │                    │                    │                  │
│              ▼                    ▼                    ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              EXTRACTION LAYER (LLM)                              │   │
│  │  - Transcript Processing                                          │   │
│  │  - Entity Extraction (Levels, Patterns, Signals)                │   │
│  │  - Sentiment Analysis                                            │   │
│  │  - Structured Data Extraction                                    │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐   │
│  │              KNOWLEDGE BASE (KBV2)                              │   │
│  │  - Entities: Trading Levels, Patterns, Signals                 │   │
│  │  - Relationships: Causes, Validates, Targets                    │   │
│  │  - Vector Search: Semantic Similarity                           │   │
│  │  - Graph: Community Detection                                    │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐   │
│  │              TIME-SERIES LAYER (TimescaleDB)                    │   │
│  │  - OHLCV Data (Hyperliquid)                                     │   │
│  │  - On-Chain Metrics (CBBI, Coin Metrics)                        │   │
│  │  - Extracted Levels (Stored for Comparison)                     │   │
│  └──────────────────────────┬──────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────▼──────────────────────────────────────┐   │
│  │              MONITORING & ALERTING ENGINE                        │   │
│  │  - Level Watching (Price reaches extracted levels)             │   │
│  │  - Pattern Recognition (Chart patterns from videos)             │   │
│  │  - Signal Matching (Conditions met)                             │   │
│  │  - Alert Delivery (Telegram, Email, Webhook)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Data Extraction Schema

### What to Extract from Videos

Based on the channel content, here's what the LLM should extract:

```python
# Core Trading Entities
class TradingLevel(BaseModel):
    """Price level mentioned in video"""
    level: float              # e.g., 42000, 100000
    asset: str               # BTC, ETH, etc.
    type: str               # support, resistance, target, invalidation
    timeframe: str          # weekly, monthly, yearly
    confidence: float       # 0.0 - 1.0
    context: str           # Why this level matters
    source_timestamp: str  # Video timestamp where mentioned

class TradingPattern(BaseModel):
    """Chart pattern mentioned"""
    name: str               # harmonic, bull flag, head & shoulders, etc.
    asset: str              # BTC, etc.
    direction: str          # bullish, bearish, neutral
    target: Optional[float] # Price target
    invalidation: float   # Invalidation level
    confidence: float
    source_timestamp: str

class TradeSignal(BaseModel):
    """Specific trade setup"""
    id: str
    asset: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    timeframe: str         # intraday, swing, position
    risk_reward: Optional[float]
    confidence: float
    status: str            # active, triggered, expired
    source_video_id: str
    source_timestamp: str

class MarketOutlook(BaseModel):
    """Market outlook/analysis"""
    asset: str
    outlook: str            # bullish, bearish, neutral
    timeframe: str         # short, medium, long
    key_levels: List[float]
    key_conditions: List[str]  # "BTC above 100k", "ETH above 3k"
    sentiment_score: float     # -1.0 to 1.0
    source_video_id: str
    analysis_summary: str

class PatternSetup(BaseModel):
    """Educational pattern from course"""
    pattern_name: str
    category: str          # harmonic, price action, indicators
    description: str
    entry_criteria: List[str]
    exit_criteria: List[str]
    risk_management: str
    timeframe: str
    asset_class: str       # crypto, stocks, forex
```

---

## Part 3: YouTube Video Processing Pipeline

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              YOUTUBE VIDEO PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. VIDEO DISCOVERY                                             │
│     ┌─────────────────┐  ┌─────────────────┐                   │
│     │ Chart Champions │  │    ECKrown     │                   │
│     │  Channel RSS   │  │  Channel RSS   │                   │
│     └────────┬────────┘  └────────┬────────┘                   │
│              │                    │                              │
│              ▼                    ▼                              │
│     ┌─────────────────────────────────────────┐                 │
│     │     New Video Detector                  │                 │
│     │  - Poll every X minutes                 │                 │
│     │  - Store video metadata                 │                 │
│     └──────────────────┬──────────────────────┘                 │
│                        │                                          │
│  2. TRANSCRIPT EXTRACTION                                       │
│     ┌──────────────────▼──────────────────────┐                 │
│     │  - YouTube Transcript API              │                 │
│     │  - Store with timestamps                │                 │
│     │  - Language detection                   │                 │
│     └──────────────────┬──────────────────────┘                 │
│                        │                                          │
│  3. LLM PROCESSING                                               │
│     ┌──────────────────▼──────────────────────┐                 │
│     │  - Chunk transcript                     │                 │
│     │  - Extract entities (LLM)               │                 │
│     │  - Validate & normalize                 │                 │
│     │  - Generate embeddings                  │                 │
│     └──────────────────┬──────────────────────┘                 │
│                        │                                          │
│  4. STORAGE                                                       │
│     ┌──────────────────▼──────────────────────┐                 │
│     │  - KBV2: Entities + Graph              │                 │
│     │  - Time-Series: Levels history         │                 │
│     │  - Document Store: Transcripts          │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Video Processing Flow

```python
# Step 1: Discover new videos
async def check_for_new_videos(channel_id: str) -> List[Video]:
    """Poll channel for new videos"""
    # Use YouTube Data API or RSS feed
    # Compare with already-processed videos
    # Return new videos

# Step 2: Get transcript
async def get_transcript(video_id: str) -> Transcript:
    """Get video transcript with timestamps"""
    # Use YouTube Transcript API
    # Return list of {text, start, duration}

# Step 3: Process with LLM
async def extract_trading_data(transcript: Transcript, video_metadata: dict) -> ExtractionResult:
    """Use LLM to extract trading entities"""
    
    system_prompt = """You are a trading expert analyzing video transcripts.
    Extract all trading information including:
    
    1. PRICE LEVELS: Support, resistance, targets, invalidation points
    2. CHART PATTERNS: Harmonic, technical patterns mentioned
    3. TRADE SETUPS: Entry, stop loss, take profit levels
    4. MARKET OUTLOOK: Bullish/bearish sentiment and reasoning
    5. CONDITIONS: "If BTC above X", "when price reaches Y"
    
    Always include:
    - The exact price levels mentioned
    - The context/reasoning
    - Timestamp in video where mentioned
    - Confidence level
    """
    
    # Process transcript in chunks
    # Extract entities
    # Return structured data

# Step 4: Store in KBV2
async def store_extracted_data(extraction: ExtractionResult):
    """Store extracted data in knowledge base"""
    # Store entities
    # Create relationships
    # Generate embeddings
    # Update graph
```

---

## Part 4: Real-Time Monitoring Engine

### Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              MONITORING ENGINE                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EXTRACTED TRIGGERS                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Level(price=95000, type=resistance, asset=BTC)      │    │
│  │ Pattern(name=bull_flag, target=102000)                 │    │
│  │ Condition(if_BTC_above=100000)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│              │                                                   │
│              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           LEVEL WATCHER                                  │    │
│  │  - Monitor price vs extracted levels                    │    │
│  │  - Calculate distance to level                          │    │
│  │  - Trigger when within threshold                        │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌───────────────────────▼────────────────────────────────┐    │
│  │           PATTERN MATCHER                                │    │
│  │  - Detect chart patterns from price data                │    │
│  │  - Match with extracted patterns                        │    │
│  │  - Notify when pattern completes                        │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  ┌───────────────────────▼────────────────────────────────┐    │
│  │           CONDITION EVALUATOR                            │    │
│  │  - Parse extracted conditions                           │    │
│  │  - Evaluate against current market data                 │    │
│  │  - Trigger when conditions met                         │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           ALERT DISPATCHER                               │    │
│  │  - Telegram bot                                         │    │
│  │  - Email                                                │    │
│  │  - Webhook                                              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Trigger Types

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class TriggerType(str, Enum):
    PRICE_LEVEL = "price_level"           # Price reaches level
    PATTERN_COMPLETE = "pattern_complete" # Pattern completes
    CONDITION_MET = "condition_met"       # "If BTC above X"
    LEVEL_BREAK = "level_break"          # Support/resistance breaks

class TradingTrigger(BaseModel):
    """A trigger extracted from video"""
    id: str
    type: TriggerType
    
    # For price level triggers
    asset: str             # BTC, ETH
    level: Optional[float]
    level_type: Optional[str]  # support, resistance, target
    
    # For pattern triggers
    pattern_name: Optional[str]
    pattern_direction: Optional[str]
    
    # For condition triggers
    condition: Optional[str]  # "BTC > 100000"
    
    # Metadata
    source_video_id: str
    source_channel: str     # ChartChampions, ECKrown
    extracted_timestamp: str
    confidence: float
    
    # Alert settings
    alert_threshold: float = 0.01  # 1% for price triggers
    notified: bool = False
```

### Level Monitoring Logic

```python
async def check_price_levels(current_price: float, triggers: List[TradingTrigger]):
    """Check if any price level triggers are hit"""
    
    for trigger in triggers:
        if trigger.type != TriggerType.PRICE_LEVEL:
            continue
            
        level = trigger.level
        threshold = level * trigger.alert_threshold
        
        # Check if price is within threshold of level
        if abs(current_price - level) <= threshold:
            await send_alert(
                title=f"🔔 Price Level Alert: {trigger.asset}",
                message=f"Price approaching {trigger.level_type}: ${level:,.0f}\n"
                        f"Current: ${current_price:,.0f}\n"
                        f"Distance: {abs(current_price - level)/level*100:.2f}%\n"
                        f"Source: {trigger.source_channel}\n"
                        f"Video: {trigger.source_video_id}",
                priority="high"
            )
            
        # Check if price broke through level
        if trigger.level_type == "resistance":
            if current_price > level:
                await send_alert(
                    title=f"🚀 Resistance Broken: {trigger.asset}",
                    message=f"${level:,.0f} resistance broken!\n"
                            f"Now at: ${current_price:,.0f}",
                    priority="critical"
                )
```

---

## Part 5: Course Materials Processing

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│              COURSE MATERIALS PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PDF PROCESSING                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │ Pattern Guide   │  │ Strategy PDF   │  │ Cheat Sheets   ││
│  │  (Text + Tables)│  │  (Full content)│  │ (Key points)   ││
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘│
│           │                    │                    │          │
│           ▼                    ▼                    ▼          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  PDF Extractor                                          │    │
│  │  - Text extraction (pdfplumber)                        │    │
│  │  - Table extraction (camelot)                         │    │
│  │  - Image extraction (for charts)                      │    │
│  │  - Metadata (title, author, pages)                    │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  VIDEO LESSON PROCESSING                                        │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │ Course Videos   │  │ Step-by-Step   │                      │
│  │ (Educational)   │  │ (Walkthroughs) │                      │
│  └────────┬────────┘  └────────┬────────┘                      │
│           │                    │                                │
│           ▼                    ▼                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Video Analyzer                                          │    │
│  │  - Transcript extraction                                 │    │
│  │  - Step/phase identification                            │    │
│  │  - Key concept extraction                               │    │
│  │  - Code/indicator extraction (if applicable)            │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           │                                      │
│  LLM PROCESSING                                               │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  - Extract strategies                                    │   │
│  │  - Extract pattern definitions                          │   │
│  │  - Extract entry/exit rules                             │   │
│  │  - Create structured knowledge                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  STORAGE                                                       │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  KBV2: Entities + Relationships + Vector Search        │   │
│  │  - Strategy entities                                    │   │
│  │  - Pattern entities                                     │   │
│  │  - Course relationship to strategies                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 6: Data Models

### Complete Entity Schema

```python
# ============== ENTITIES ==============

class YouTubeVideo(BaseModel):
    """YouTube video entity"""
    video_id: str
    channel_name: str          # ChartChampions, ECKrown
    title: str
    description: str
    published_at: datetime
    duration_seconds: int
    url: str
    transcript_status: str     # pending, processed, failed

class TradingLevelEntity(BaseModel):
    """Price level from analysis"""
    name: str                  # "BTC 95k Resistance"
    level_type: str           # support, resistance, target, invalidation
    asset: str                # BTC, ETH
    price: float
    timeframe: str           # daily, weekly, monthly
    source_video_id: str
    source_channel: str
    confidence: float
    context: str             # Why this level matters

class TradingPatternEntity(BaseModel):
    """Chart pattern"""
    name: str                 # "Bullish Harmonic"
    pattern_type: str        # harmonic, flag, wedge, etc.
    asset: str
    direction: str           # bullish, bearish
    target_price: Optional[float]
    stop_loss: Optional[float]
    source_video_id: str
    confidence: float

class TradeSetupEntity(BaseModel):
    """Complete trade setup"""
    name: str
    asset: str
    entry_zone: str          # "95,000 - 96,000"
    stop_loss: float
    take_profits: List[float]
    timeframe: str
    risk_reward: Optional[float]
    source_video_id: str
    status: str              # active, triggered, expired

class MarketOutlookEntity(BaseModel):
    """Market outlook summary"""
    asset: str
    outlook: str             # bullish, bearish, neutral
    timeframe: str          # short, medium, long
    key_levels: List[float]
    conditions: List[str]    # ["BTC > 100k", "ETH > 3k"]
    sentiment: float        # -1.0 to 1.0
    summary: str
    source_video_id: str

class StrategyEntity(BaseModel):
    """Trading strategy from course"""
    name: str
    category: str           # harmonic, price action, etc.
    description: str
    entry_rules: List[str]
    exit_rules: List[str]
    risk_management: str
    timeframe: str
    asset_class: str
    source_material: str    # Course name
    source_page: Optional[int]

# ============== RELATIONSHIPS ==============

# Level → Asset (HAS_LEVEL)
# Pattern → Asset (FORMS_PATTERN)
# Setup → Asset (SETUP_FOR)
# Outlook → Asset (OUTLOOK_FOR)
# Strategy → Pattern (BASED_ON)
# Video → Level (MENTIONS_LEVEL)
# Video → Outlook (CONTAINS_OUTLOOK)
# Course → Strategy (TEACHES)
```

---

## Part 7: API Endpoints

### New Endpoints to Add

```python
# Video Management
GET    /api/v1/videos                          # List all processed videos
GET    /api/v1/videos/{video_id}              # Get video details
GET    /api/v1/videos/{video_id}/transcript   # Get full transcript
GET    /api/v1/videos/{video_id}/analysis     # Get extracted analysis

# Trading Data
GET    /api/v1/trading/levels                  # Get all price levels
GET    /api/v1/trading/levels/{asset}          # Get levels for asset
GET    /api/v1/trading/patterns                # Get extracted patterns
GET    /api/v1/trading/setups                   # Get trade setups
GET    /api/v1/trading/outlooks               # Get market outlooks
GET    /api/v1/trading/strategies             # Get course strategies

# Active Triggers
GET    /api/v1/triggers                        # List active triggers
GET    /api/v1/triggers/active                # Currently relevant triggers
GET    /api/v1/triggers/levels/{asset}        # Levels being watched

# Monitoring
GET    /api/v1/monitoring/status              # Monitoring status
POST   /api/v1/monitoring/check              # Manual check
GET    /api/v1/monitoring/history            # Alert history

# Alerts
POST   /api/v1/alerts/subscribe              # Subscribe to alerts
GET    /api/v1/alerts/channels               # List alert channels
POST   /api/v1/alerts/test                   # Test alert

# Query
POST   /api/v1/query/trading                 # Query trading knowledge
POST   /api/v1/query/course                  # Query course materials
```

---

## Part 8: Implementation Phases

### Phase 1: YouTube Integration (Weeks 1-2)

- [ ] Set up YouTube Data API access
- [ ] Create channel polling mechanism
- [ ] Implement transcript extraction
- [ ] Build video metadata storage

### Phase 2: LLM Extraction (Weeks 3-4)

- [ ] Design extraction prompts for trading data
- [ ] Implement entity extraction pipeline
- [ ] Add validation and normalization
- [ ] Generate embeddings for semantic search

### Phase 3: Knowledge Base (Weeks 5-6)

- [ ] Create entity models in KBV2
- [ ] Build relationship mappings
- [ ] Implement vector search for similarity
- [ ] Add graph analytics for communities

### Phase 4: Time-Series Integration (Weeks 7-8)

- [ ] Set up TimescaleDB
- [ ] Integrate Hyperliquid API for OHLCV
- [ ] Add CBBI/Coin Metrics for on-chain
- [ ] Store extracted levels in time-series

### Phase 5: Monitoring Engine (Weeks 9-10)

- [ ] Build level watcher service
- [ ] Implement pattern matcher
- [ ] Create condition evaluator
- [ ] Add alert dispatcher (Telegram, etc.)

### Phase 6: Course Materials (Weeks 11-12)

- [ ] Build PDF extraction pipeline
- [ ] Process video lessons
- [ ] Extract strategies and patterns
- [ ] Link to source materials

---

## Part 9: Repository Structure

```
trading-intelligence/
├── kbv2/                         # Knowledge Base
│   ├── src/knowledge_base/
│   │   ├── api/                 # REST API
│   │   ├── models/              # Data models (updated)
│   │   └── extraction/          # Trading entity extraction
│   └── docs/
│
├── youtube-pipeline/             # Video processing
│   ├── src/
│   │   ├── clients/             # YouTube API client
│   │   ├── extractors/          # Transcript extraction
│   │   ├── processors/          # LLM processing
│   │   └── export/              # KBV2 export
│   └── tests/
│
├── market-data-pipeline/         # Time-series data
│   ├── src/
│   │   ├── clients/             # Hyperliquid, CBBI, CoinMetrics
│   │   ├── storage/              # TimescaleDB integration
│   │   └── api/                 # Market data API
│   └── tests/
│
├── monitoring-engine/            # Alert system
│   ├── src/
│   │   ├── watchers/            # Level, pattern, condition watchers
│   │   ├── alerts/              # Alert dispatchers
│   │   └── scheduler/          # Monitoring scheduler
│   └── tests/
│
└── course-pipeline/             # Course materials
    ├── src/
    │   ├── pdf/                 # PDF extraction
    │   ├── video/               # Video lesson processing
    │   └── export/              # KBV2 export
    └── tests/
```

---

## Part 10: Summary

### What This System Does

1. **Ingest Videos** - Automatically process new videos from Chart Champions & ECKrown
2. **Extract Knowledge** - Use LLM to extract trading levels, patterns, signals, outlooks
3. **Store Intelligence** - Build knowledge graph in KBV2
4. **Monitor Markets** - Watch extracted levels/conditions in real-time
5. **Alert You** - Notify when price hits levels or conditions are met
6. **Process Courses** - Extract strategies from Chart Champions course materials

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Video Pipeline | YouTube API | Extract transcripts, metadata |
| LLM Extraction | GPT-4o / Claude | Extract trading entities |
| Knowledge Base | KBV2 | Store entities, relationships, vectors |
| Time-Series | TimescaleDB | Store OHLCV, track levels |
| Monitoring | Custom | Watch triggers, send alerts |
| Alerting | Telegram/Email | Notify when conditions met |

### Next Steps

1. Confirm architecture
2. Start Phase 1: YouTube integration
3. Build extraction prompts
4. Deploy monitoring engine

# Trading Intelligence System - UPDATED Plan

## Executive Summary

After analyzing both repositories, here's the updated plan:

---

## What We HAVE (Already Built)

### ✅ YouTube Content Pipeline (Production Ready)

| Component | Status |
|-----------|--------|
| Video fetching | ✅ Ready |
| Transcript extraction (YouTube API + Whisper fallback) | ✅ Ready |
| 3-Agent LLM Pipeline | ✅ Ready |
| Frame extraction + Vision analysis | ✅ Ready |
| **Trading entity extraction** | ✅ Ready |
| KBV2 formatter | ✅ Ready |
| External ingestion to KBV2 | ✅ Ready |

### What's Already Extracted from Videos

```python
# Already being extracted by YouTube Pipeline:
- TradingLevel (price, type, confidence, timestamp)
- TradingSignal (asset, direction, entry, target, stop_loss)
- Technical indicators (RSI, MACD, etc.)
- Chart patterns (harmonic, flag, etc.)
- Market context (bullish/bearish/neutral)
- Executive summaries
- Frame analysis with timestamps
```

### ✅ KBV2 (Already Ready)

| Component | Status |
|-----------|--------|
| External ingestion API | ✅ Ready |
| Domain detection (6 crypto domains) | ✅ Ready |
| Knowledge graph (entities, edges) | ✅ Ready |
| Hybrid search (vector + BM25) | ✅ Ready |
| Self-improvement system | ✅ Ready |

---

## What We NEED to Build

### 🚨 Phase 1: Register TRADING Domain Schema (Week 1)

**Why:** KBV2 needs to recognize the entity types from YouTube pipeline.

| Action | Description |
|--------|-------------|
| Register entity types | Add: `price_level_support`, `price_level_resistance`, `trading_signal_long`, `trading_signal_short`, `technical_indicator`, `price_pattern` |
| Add extraction goals | Add trading extraction goals to template_registry |

**Files to modify:**
- `main.py` - Register entity types
- `template_registry.py` - Add extraction goals

---

### 🚨 Phase 2: Add ECKrown to YouTube Pipeline (Week 2)

**What's needed:**
- Add ECKrown channel ID to YouTube pipeline
- Create channel polling for both channels

```python
# In youtube-pipeline config:
CHANNELS = {
    "chart_champions": "UC5u_618hRZFhX3iVJ8J5pLQ",
    "eckrown": "UCnwxzpFZQNtLH8NgTeAROFA"
}
```

---

### ⚠️ Phase 3: Market Data + Level Tracking (Weeks 3-4)

**Why:** Need to monitor extracted price levels against real-time market.

| Component | Purpose | New File |
|-----------|---------|----------|
| TimescaleDB | Store OHLCV data | market-data-pipeline |
| Level Tracker | Track extracted levels | market-data-pipeline |
| Market API Client | Fetch prices (Hyperliquid) | market-data-pipeline |

**Database Schema:**
```sql
-- Tracked Price Levels (from videos)
CREATE TABLE tracked_levels (
    id UUID PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    price_level DECIMAL(24, 8) NOT NULL,
    level_type VARCHAR(20),  -- support, resistance, target
    source_video_id VARCHAR(50),
    source_channel VARCHAR(50),
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    alert_threshold DECIMAL(5, 4) DEFAULT 0.01,
    status VARCHAR(20) DEFAULT 'active'
);

-- Alert History
CREATE TABLE alert_history (
    id UUID PRIMARY KEY,
    alert_type VARCHAR(50),
    asset VARCHAR(20),
    message TEXT,
    triggered_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

### ⚠️ Phase 4: Monitoring Engine (Weeks 5-6)

**Why:** Alert when price reaches extracted levels.

| Component | Purpose |
|-----------|---------|
| Level Watcher | Monitor price vs extracted levels |
| Alert Dispatcher | Send Telegram/Email notifications |
| Web Dashboard | View active triggers |

**Logic:**
```
For each extracted price level:
  - Fetch current price from market API
  - If price within threshold (e.g., 1%):
    - Send "approaching level" alert
  - If price breaks through:
    - Send "level broken" alert (high priority)
```

---

### 📌 Phase 5: Course Materials (Week 7-8)

**Why:** Process Chart Champions course PDFs + videos.

| Component | Purpose |
|-----------|---------|
| PDF Extractor | Extract text, tables from PDFs |
| Strategy Extractor | LLM extracts entry/exit rules |
| Link to Source | Link strategies to course materials |

---

## Updated Implementation Timeline

```
Week 1: Phase 1 - TRADING Domain Schema
├── Register entity types in KBV2
└── Add extraction goals

Week 2: Phase 2 - ECKrown Channel
├── Add ECKrown to YouTube pipeline
└── Test video processing

Weeks 3-4: Phase 3 - Market Data
├── Set up TimescaleDB
├── Integrate Hyperliquid API
└── Build level tracking

Weeks 5-6: Phase 4 - Monitoring
├── Build level watcher
├── Implement alert dispatcher
└── Add Telegram notifications

Weeks 7-8: Phase 5 - Course Materials
├── Build PDF extractor
├── Extract strategies
└── Store in KBV2
```

---

## What to Build vs What Already Exists

### ✅ Already Exists (No Work Needed)

| Component | Location |
|-----------|----------|
| YouTube video fetching | youtube-content-pipeline |
| Transcript extraction | youtube-content-pipeline |
| 3-Agent LLM pipeline | youtube-content-pipeline |
| Trading entity extraction | youtube-content-pipeline |
| KBV2 integration | youtube-content-pipeline → kbv2 |
| External ingestion API | kbv2 |
| Domain detection | kbv2 |
| Knowledge graph | kbv2 |
| Hybrid search | kbv2 |

### 🚨 Need to Build

| Component | Effort | Priority |
|-----------|--------|----------|
| TRADING domain schema | 1 day | Critical |
| ECKrown channel integration | 1 day | Critical |
| Market data (TimescaleDB + API) | 1 week | High |
| Level tracking in KBV2 | 2 days | High |
| Monitoring + alerting | 1 week | Medium |
| Course processing | 1 week | Low |

---

## Architecture (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│              YOUTUBE CONTENT PIPELINE (EXISTING)                 │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │ Chart Champions │  │    ECKrown     │  ← ADD ECKrown     │
│  │   Channel      │  │   Channel      │                    │
│  └────────┬────────┘  └────────┬────────┘                    │
│           │                    │                                │
│           ▼                    ▼                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3-Agent LLM Pipeline                                  │   │
│  │  - Transcript Intelligence                             │   │
│  │  - Frame Analysis                                     │   │
│  │  - Synthesis                                          │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  KBV2 External Ingestion                              │   │
│  │  (Already exists!)                                    │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              KBV2 (ENHANCED)                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  + TRADING Domain Schema (NEW)                        │   │
│  │  - price_level_support, price_level_resistance       │   │
│  │  - trading_signal_long, trading_signal_short         │   │
│  │  - technical_indicator, price_pattern                 │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼──────────────────────────────┐   │
│  │  + Level Tracking Table (NEW)                        │   │
│  │  - Store extracted price levels                      │   │
│  │  - Track alert status                                │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MARKET DATA PIPELINE (NEW)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - TimescaleDB (OHLCV data)                           │   │
│  │  - Hyperliquid API (prices)                            │   │
│  │  - CBBI + Coin Metrics (on-chain)                     │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌─────────────────────────▼──────────────────────────────┐   │
│  │  MONITORING ENGINE (NEW)                               │   │
│  │  - Watch price levels                                  │   │
│  │  - Alert when triggered                                │   │
│  │  - Telegram/Email notifications                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

**Should I start with:**

1. **Phase 1: Register TRADING domain schema** - Add entity types to KBV2
2. **Phase 2: Add ECKrown** - Update YouTube pipeline config

Which would you like me to start with?

# KBV2 Operation Scripts - Summary

## 📁 Script Directory Structure

```
scripts/
├── start_kbv2_server.sh      # Main server start script with colored output
├── quick_reference_card.sh    # On-screen command reference
└── README.md                  # Comprehensive operations guide
```

## 🚀 Quick Start

### Starting the Server
```bash
cd /home/muham/development/kbv2/scripts
./start_kbv2_server.sh
```

This will:
- Run pre-flight checks (verify uv, project directory, logs)
- Stop any existing KBV2 processes
- Backup old logs with timestamps
- Start fresh server with consolidated logging
- Display all monitoring commands

### Monitoring Logs
```bash
# Terminal 1: Watch ingestion logs in real-time
tail -f /tmp/kbv2_ingestion.log

# Terminal 2: Watch server logs in real-time
tail -f /tmp/kbv2_server.log
```

## 📊 What Gets Logged

The consolidated log at `/tmp/kbv2_ingestion.log` contains:

1. **Adaptive Analysis**
   - Document complexity assessment
   - Processing strategy recommendations
   - Expected entity counts

2. **Stage Progress**
   - Perception (6 steps)
   - Enhancement (3 steps)
   - Evaluation (scoring)

3. **LLM Calls**
   - Call number, agent name, model used
   - Token counts (prompt + response)
   - Duration and status

4. **Entity Extraction**
   - Entity count per chunk
   - Entity types extracted
   - Confidence scores

5. **Quality Scores**
   - Overall extraction quality (0-1)
   - Quality level (low/medium/high)
   - Feedback summary

6. **Errors and Timeouts**
   - Failed LLM calls
   - Model rotation events
   - Circuit breaker activations

## 🔑 Key Improvements

### Before (Non-Adaptive)
- 29 minutes for 5KB document
- 18-28 LLM calls per chunk
- Fixed processing pipeline
- Single model usage

### After (Smart Adaptive)
- LLM analyzes document first
- Chooses optimal strategy (gleaning/multi-agent)
- Random model selection per call
- Expected 20-80% fewer LLM calls
- Better quality through intelligent processing

### Logging Consolidation
- **All logs in one file**: `/tmp/kbv2_ingestion.log`
- Server logs included via ingestion_logger
- No more hunting across multiple files
- Comprehensive context for debugging

## 📈 Performance Metrics to Watch

### During Ingestion
1. **Adaptive Analysis Time**: Should complete in 10-30s
2. **Entities Extracted**: Track per-chunk count
3. **LLM Call Duration**: Should be 2-5s per call (through proxy)
4. **Model Distribution**: Check for even rotation

### After Ingestion
1. **Total Time**: Compare to document size
2. **Total LLM Calls**: Lower is better (adaptive optimization)
3. **Quality Score**: Target >0.7 for good extraction
4. **Entity Count**: Compare to expected from adaptive analysis

## 🔧 Customization

### Adjust Adaptive Engine Thresholds
Edit: `src/knowledge_base/intelligence/v1/adaptive_ingestion_engine.py`

```python
COMPLEXITY_THRESHOLDS = {
    "simple": 3000,      # < 3KB = simple (use gleaning)
    "moderate": 15000,   # 3KB-15KB = moderate (balanced)
    "complex": float("inf"),  # > 15KB = complex (full multi-agent)
}
```

### Add/Remove Models
Edit: `src/knowledge_base/common/resilient_gateway/gateway.py`

```python
FALLBACK_MODELS = [
    "claude-opus-4.5-20251101",
    "claude-3.7-sonnet-20250219",
    "gemini-1.5-pro",
    "gemini-2-flash",
]
```

## 🐛 Troubleshooting Commands

```bash
# Check for errors during ingestion
grep "ERROR" /tmp/kbv2_ingestion.log

# Count LLM failures
grep -c "LLM ERROR" /tmp/kbv2_ingestion.log

# View recent quality scores
grep "QUALITY SCORE" /tmp/kbv2_ingestion.log | tail -10

# Calculate average stage duration
grep "STAGE COMPLETE" /tmp/kbv2_ingestion.log | grep -o "elapsed: [0-9.]*s"

# Check model distribution
grep "->" /tmp/kbv2_ingestion.log | grep "🤖" | awk '{print $6}' | sort | uniq -c
```

## 📚 Next Steps

1. **Start the server**: `./start_kbv2_server.sh`
2. **Monitor logs**: `tail -f /tmp/kbv2_ingestion.log`
3. **Run test ingestion**: See README.md for detailed examples
4. **Analyze results**: Use grep commands to extract metrics
5. **Tune parameters**: Adjust thresholds based on results

## 🎯 Success Criteria

✅ Server starts cleanly with no errors
✅ Logs appear in `/tmp/kbv2_ingestion.log`
✅ Adaptive analysis completes (🔍 ADAPTIVE ANALYSIS START)
✅ Entities extracted (🎯 ENTITIES EXTRACTED)
✅ Quality scores generated (🏆 QUALITY SCORE)
✅ Summary logged (📊 EXTRACTION SUMMARY)
✅ Different models used per LLM call
✅ Processing completes faster than before (target: <15 min for 5KB doc)

---

**Location**: `/home/muham/development/kbv2/scripts/`
**Logs**: `/tmp/kbv2_ingestion.log`
**Server**: http://localhost:8765

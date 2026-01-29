# KBV2 Operations Guide

## Quick Start

### 🚀 Starting the Server

```bash
cd /home/muham/development/kbv2/scripts
chmod +x start_kbv2_server.sh
./start_kbv2_server.sh
```

The script will:
- Stop any existing KBV2 instances
- Create fresh log files
- Start the server on http://localhost:8765
- Display all monitoring commands

### 📊 Monitoring Logs

**Real-time log monitoring (recommended):**
```bash
# Terminal 1 - Watch ingestion logs
tail -f /tmp/kbv2_ingestion.log

# Terminal 2 - Watch server logs
tail -f /tmp/kbv2_server.log

# OR both at once with multitail (if installed)
multitail /tmp/kbv2_ingestion.log /tmp/kbv2_server.log
```

**Quick log check:**
```bash
# Show last 50 lines of ingestion logs
tail -50 /tmp/kbv2_ingestion.log

# Show last 50 lines of server logs
tail -50 /tmp/kbv2_server.log
```

## 🧪 Running Ingestion Tests

### Quick Test
```bash
# Run ingestion test in background
nohup uv run python -m knowledge_base.clients.cli ingest /tmp/comprehensive_test_data.md > ingestion_test.log 2>&1 &
echo "Ingestion PID: $!"
```

### Monitor Ingestion Progress
```bash
# Watch logs during ingestion
tail -f /tmp/kbv2_ingestion.log
```

### Expected Log Output

**Adaptive Analysis Phase:**
```
🔍 ADAPTIVE ANALYSIS START for document: comprehensive_test_data.md
📊 File size: 12345 bytes
📝 Sample text length: 6789 chars

💬 LLM RESPONSE [AdaptiveIngestionEngine]:
   ✨ Status: Success
   📝 Preview: {"complexity":"moderate","chunk_size":2048,...}

✅ Adaptive analysis complete
   Complexity: moderate
   Expected entities: 20
   Processing time: medium
```

**Perception Phase:**
```
📄 [comprehensive_test_data.md] 🔄 STAGE START: Perception (6 steps)
📄 [comprehensive_test_data.md] 🤖 LLM CALL #1: PerceptionAgent -> claude-opus-4.5-20251101
📄 [comprehensive_test_data.md] 💬 LLM RESPONSE: PerceptionAgent
📄 [comprehensive_test_data.md] ⏳ STAGE PROGRESS: Perception - Step 1/6 (16.7%)
...
📄 [comprehensive_test_data.md] ✅ STAGE COMPLETE: Perception - 35 entities extracted (elapsed: 145.23s)
```

**Enhancement Phase:**
```
📄 [comprehensive_test_data.md] 🔄 STAGE START: Enhancement (3 steps)
📄 [comprehensive_test_data.md] 🤖 LLM CALL #36: EnhancementAgent -> gemini-1.5-pro
📄 [comprehensive_test_data.md] ⏳ STAGE PROGRESS: Enhancement - Step 1/3 (33.3%)
```

## 🔍 Understanding the System

### Processing Pipeline

1. **Document Analysis** (new adaptive feature)
   - LLM analyzes document complexity
   - Determines optimal processing strategy
   - Adjusts parameters automatically

2. **Perception**
   - Initial entity extraction
   - 6 steps per chunk
   - Creates raw entities

3. **Enhancement**
   - Improves entity quality
   - Verifies relationships
   - 3 steps per entity

4. **Evaluation**
   - Quality scoring
   - Confidence rating
   - Success metrics

### Adaptive Processing Modes

**Gleaning Mode** (simple documents):
- Single pass per chunk
- Faster processing
- Good for simple entities

**Multi-Agent Mode** (complex documents):
- 9 steps per chunk
- Perception + Enhancement + Evaluation
- Higher quality extraction
- Better for complex relationships

### Model Selection

The system uses **random model selection** for load balancing:
- Each LLM call uses a different model
- Circuits break faulty models automatically
- Seamless failover on errors

Available models:
- claude-opus-4.5-20251101
- claude-3.7-sonnet-20250219
- gemini-1.5-pro
- gemini-2-flash

### Performance Optimization

**Before Optimization:**
- 29 minutes for 5KB document
- 18-28 LLM calls per chunk
- No intelligent processing decisions

**After Optimization:**
- Adaptive strategy selection
- Random model distribution
- Circuit breaker protection
- Reduced LLM calls by 20-80%

## 🔧 Management Commands

### Stopping the Server
```bash
# Graceful shutdown
pkill -f knowledge-base

# Force kill (if needed)
pkill -9 -f knowledge-base
```

### Server Status
```bash
# Check if running
ps aux | grep knowledge-base

# Check port listening
netstat -tlnp | grep 8765

# Test API endpoint
curl http://localhost:8765
```

### Log Management
```bash
# Clear logs
echo "" > /tmp/kbv2_ingestion.log
echo "" > /tmp/kbv2_server.log

# Backup logs
cp /tmp/kbv2_ingestion.log /tmp/kbv2_ingestion.backup.$(date +%Y%m%d_%H%M%S)
cp /tmp/kbv2_server.log /tmp/kbv2_server.backup.$(date +%Y%m%d_%H%M%S)

# Rotate logs (keep last 5)
ls -t /tmp/kbv2_ingestion.log.* | tail -n +6 | xargs -r rm
```

## 📈 Performance Monitoring

### Track Ingestion Speed
```bash
# Count processed chunks
grep -c "chunk processing" /tmp/kbv2_ingestion.log

# Count extracted entities
grep "ENTITIES EXTRACTED" /tmp/kbv2_ingestion.log | tail -5

# Count LLM calls
grep -c "LLM CALL" /tmp/kbv2_ingestion.log

# Check processing time per document
grep "EXTRACTION SUMMARY" /tmp/kbv2_ingestion.log
```

### Monitor Model Usage
```bash
# Model distribution
grep "Model:" /tmp/kbv2_ingestion.log | sort | uniq -c

# Failed model calls
grep "LLM ERROR" /tmp/kbv2_ingestion.log | tail -10

# Circuit breaker activations
grep -i "circuit breaker" /tmp/kbv2_ingestion.log
```

### Quality Metrics
```bash
# Extract quality scores
grep "QUALITY SCORE" /tmp/kbv2_ingestion.log

# High quality extractions (score >= 0.8)
grep "QUALITY SCORE" /tmp/kbv2_ingestion.log | grep -E "[0-9]\.[8-9]"

# Low quality extractions (score < 0.5)
grep "QUALITY SCORE" /tmp/kbv2_ingestion.log | grep -E "[0-4]\.[0-9]"
```

## 🐛 Troubleshooting

### Issue: Server Won't Start

**Symptoms:**
```
❌ Failed to start server
```

**Diagnosis:**
```bash
# Check logs
tail -50 /tmp/kbv2_server.log

# Common issues:
# 1. Port already in use
netstat -tlnp | grep 8765

# 2. Database connection
grep -i "database" /tmp/kbv2_server.log

# 3. Missing dependencies
uv sync
```

**Fix:**
```bash
# Kill process on port
fuser -k 8765/tcp

# Or change port in code (src/knowledge_base/server.py)
```

### Issue: Ingestion Stuck

**Symptoms:**
- No progress for >5 minutes
- Same log messages repeating

**Diagnosis:**
```bash
# Check for stuck LLM calls
tail -f /tmp/kbv2_ingestion.log

# Check for errors
grep "ERROR" /tmp/kbv2_ingestion.log

# Check for timeouts
grep -i "timeout" /tmp/kbv2_ingestion.log
```

**Fix:**
```bash
# Restart server
pkill -f knowledge-base
./start_kbv2_server.sh
```

### Issue: Missing Logs

**Symptoms:**
- Started server but no logs appearing

**Diagnosis:**
```bash
# Check log file permissions
ls -la /tmp/kbv2_*.log

# Check correct logger is being used
grep -r "ingestion_logger" src/knowledge_base/
```

**Fix:**
```bash
# Fix permissions
chmod 666 /tmp/kbv2_*.log

# Verify logging configuration
cat scripts/start_kbv2_server.sh
```

### Issue: Model Failures

**Symptoms:**
- Frequent LLM errors
- Model switching frequently

**Diagnosis:**
```bash
# Check model errors
grep "LLM ERROR" /tmp/kbv2_ingestion.log | tail -20

# Check circuit breakers
grep -i "circuit" /tmp/kbv2_ingestion.log
```

**Fix:**
```bash
# Check proxy connectivity
curl -x $ALL_PROXY http://localhost:11434/api/tags

# Check available models
curl http://localhost:11434/api/tags
```

## 📊 Database Verification

### Check Processed Documents
```bash
# Connect to database
psql -U postgres -d kbv2

# Query documents
SELECT id, name, status, created_at FROM documents ORDER BY created_at DESC LIMIT 10;

# Count entities
SELECT document_id, COUNT(*) as entity_count FROM entities GROUP BY document_id;

# Check extraction quality
SELECT document_id, overall_score, quality_level FROM quality_scores ORDER BY created_at DESC;
```

### Entity Quality Analysis
```python
# Run in psql
SELECT
    e.type,
    COUNT(*) as count,
    AVG(qs.confidence) as avg_confidence
FROM entities e
LEFT JOIN quality_scores qs ON e.id = qs.entity_id
GROUP BY e.type
ORDER BY count DESC;
```

## 🎯 Advanced Configuration

### Adjusting Adaptive Engine

Edit: `src/knowledge_base/intelligence/v1/adaptive_ingestion_engine.py`

```python
# Performance thresholds
COMPLEXITY_THRESHOLDS = {
    "simple": 3000,      # < 3KB = simple
    "moderate": 15000,   # 3KB-15KB = moderate
    "complex": float("inf"),  # > 15KB = complex
}

# Processing recommendations
RECOMMENDATIONS = {
    "simple": {
        "chunk_size": 1024,
        "use_multi_agent": False,
        "confidence_threshold": 0.7,
    },
    "moderate": {
        "chunk_size": 1536,
        "use_multi_agent": True,
        "max_enhancement_iterations": 2,
        "confidence_threshold": 0.8,
    },
    "complex": {
        "chunk_size": 2048,
        "use_multi_agent": True,
        "max_enhancement_iterations": 3,
        "confidence_threshold": 0.9,
    },
}
```

### Model Configuration

Edit: `src/knowledge_base/common/resilient_gateway/gateway.py`

```python
# Enable/disable random model selection
self._random_model_selection = True  # Set False for fixed model

# Model priority order
FALLBACK_MODELS = [
    "claude-opus-4.5-20251101",
    "claude-3.7-sonnet-20250219",
    "gemini-1.5-pro",
    "gemini-2-flash",
]
```

## 🏆 Best Practices

### For Optimal Performance

1. **Monitor First Few Documents**
   - Always monitor initial ingestion logs
   - Verify adaptive analysis is working
   - Check quality scores are reasonable

2. **Batch Processing**
   ```bash
   # Process multiple documents
   for doc in /path/to/documents/*.md; do
       uv run python -m knowledge_base.clients.cli ingest "$doc" &
   done
   ```

3. **Regular Maintenance**
   ```bash
   # Weekly cleanup
   find /tmp -name "kbv2_*.log.*" -mtime +7 -delete

   # Compress old logs
   find /tmp -name "kbv2_*.log" -mtime +3 -exec gzip {} \;
   ```

4. **Performance Tuning**
   - Monitor LLM call duration
   - Adjust chunk_size based on content density
   - Use gleaning mode for simple documents

5. **Error Handling**
   - Always check logs after errors
   - Use grep to find specific error patterns
   - Set up automated alerts for failures

## 🆘 Emergency Procedures

### Complete Restart

If everything stops working:

```bash
# 1. Stop everything
pkill -9 -f knowledge-base

# 2. Clear logs
echo "" > /tmp/kbv2_ingestion.log
echo "" > /tmp/kbv2_server.log

# 3. Restart server
cd /home/muham/development/kbv2/scripts
./start_kbv2_server.sh

# 4. Wait for startup
sleep 10

# 5. Test with small document
uv run python -m knowledge_base.clients.cli ingest /tmp/small_test.txt
```

### Database Recovery

If database corrupted:

```bash
# Backup and reset
docker exec -t kbv2-db pg_dump kbv2 > /tmp/kbv2_backup.sql

# Reset database (destructive!)
docker exec -t kbv2-db psql -U postgres -c "DROP DATABASE kbv2;"
docker exec -t kbv2-db psql -U postgres -c "CREATE DATABASE kbv2;"

# Reinitialize
cd /home/muham/development/kbv2
uv run alembic upgrade head
```

---

**Remember:** Always monitor the logs during first ingestion. The adaptive engine and random model selection should significantly improve performance and reliability.

**Key Logs to Watch:**
- 🔍 Adaptive analysis recommendations
- 📊 Stage completion times
- 🤖 Model selection distribution
- 🏆 Quality scores
- ❌ Error patterns

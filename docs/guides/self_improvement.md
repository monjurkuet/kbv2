# KBV2 Self-Improvement Features

## Overview

KBV2 includes comprehensive self-improvement capabilities that enable the system to learn from processed documents and improve extraction quality over time.

---

## Components

### 1. Experience Bank

Stores and retrieves high-quality extraction examples for few-shot prompting.

**Purpose:**
- Learn from successful extractions
- Provide few-shot examples for new documents
- Improve extraction accuracy over time

**Configuration:**
```python
ExperienceBankConfig(
    min_quality_threshold=0.75,  # Store extractions ≥ 0.75
    max_storage_size=10000,
    similarity_top_k=3,
    enable_pattern_extraction=True,
)
```

**Quality Calculation:**
The system calculates extraction quality using multiple factors:
- Entity count (20% weight)
- Edge count (20% weight)
- Average entity confidence (30% weight)
- Extraction pipeline quality (30% weight)

**Storage:**
High-quality extractions (quality ≥ 0.75) are automatically stored for each chunk, with entities distributed across chunks to provide diverse examples.

**Retrieval:**
When processing new documents, the system retrieves similar examples based on:
- Vector similarity (text embeddings)
- Domain matching
- Entity type overlap

### 2. Prompt Evolution

Automatically optimizes extraction prompts for different domains.

**Purpose:**
- Generate prompt variants for each domain
- Test variants against evaluation documents
- Select best performing prompts

**Supported Domains:**
- BITCOIN
- DEFI
- INSTITUTIONAL_CRYPTO
- STABLECOINS
- CRYPTO_REGULATION

**Configuration:**
```python
PromptEvolutionConfig(
    num_variants_per_generation=5,
    max_generations=10,
    mutation_temperature=0.7,
    min_evaluation_samples=10,
    selection_threshold=0.75,
    crypto_domains=[
        "BITCOIN", "DEFI", "INSTITUTIONAL_CRYPTO",
        "STABLECOINS", "CRYPTO_REGULATION"
    ],
)
```

**Process:**
1. Initialize domain with base prompts
2. Generate variants via mutation
3. Test against evaluation documents
4. Select best performing variant
5. Repeat for specified generations

### 3. Ontology Validator

Validates extractions against domain-specific rules and detects contradictions.

**Purpose:**
- Ensure extracted entities follow domain ontology
- Detect semantic contradictions
- Validate required properties
- Identify missing attributes

**Validation Rules:**
- 15+ crypto-specific rules
- Required property checking
- Property type validation
- Min/max value constraints
- Allowed values validation
- Cardinality constraints

**Semantic Validation:**
- Contradiction detection
- Relationship consistency checking
- Entity reference validation

### 4. Domain Detection Feedback

Learns from domain classification accuracy and improves detection over time.

**Purpose:**
- Record all domain classifications
- Track accuracy over time
- Identify problematic domains
- Suggest improvements

**Learning Sources:**
1. **User Feedback** - Explicit corrections from users
2. **Quality Correlation** - Low extraction quality indicates wrong domain
3. **Auto-Tracking** - All detections recorded with metadata

**Metrics Tracked:**
- Total classifications per domain
- Confirmed accurate/inaccurate counts
- Average confidence per domain
- Average extraction quality per domain
- Accuracy rate calculation

---

## Usage

### Initialize Self-Improving Orchestrator

```python
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

orchestrator = SelfImprovingOrchestrator(
    enable_experience_bank=True,
    enable_prompt_evolution=True,
    enable_ontology_validation=True,
)
await orchestrator.initialize()
```

### Process Document (Automatic Learning)

```python
document = await orchestrator.process_document(
    file_path="bitcoin_report.pdf",
    domain="BITCOIN"
)

# System automatically:
# 1. Retrieves similar examples from Experience Bank
# 2. Uses evolved prompts for extraction
# 3. Validates extraction against ontology
# 4. Stores high-quality results in Experience Bank
```

### Check Experience Bank Stats

```python
stats = await orchestrator.get_experience_bank_stats()
print(f"Total experiences: {stats['total_experiences']}")
print(f"Domain distribution: {stats['domain_distribution']}")
print(f"Hit rate: {stats['hit_rate']}")
```

### Check Domain Detection Accuracy

```python
from knowledge_base.intelligence.v1.self_improvement import DomainDetectionSelfImprovement

async with session.begin():
    feedback = DomainDetectionSelfImprovement(session)
    
    # Get accuracy stats
    stats = await feedback.get_domain_accuracy_stats(days=30)
    for stat in stats:
        print(f"{stat.domain}: {stat.accuracy_rate:.1%} accuracy")
    
    # Get improvement suggestions
    suggestions = await feedback.get_improvement_suggestions()
    for sug in suggestions:
        print(f"{sug.domain}: {sug.issue} → {sug.suggestion}")
```

### Record User Correction

```python
# When user manually corrects domain
feedback_id = await feedback.record_detection(
    document_id=doc.id,
    detected_domain="GENERAL",  # What system thought
    confidence=0.8,
    detection_method="llm",
    crypto_indicators=["bitcoin", "mining"],
    domain_scores={"BITCOIN": 0.7, "GENERAL": 0.3},
    user_correction="BITCOIN",  # What user said
)
```

### Manually Evolve Prompts

```python
result = await orchestrator.evolve_prompts(
    domain="DEFI",
    test_documents=test_docs
)

print(f"Best variant: {result['best_variant']['name']}")
print(f"Quality score: {result['best_variant']['quality_score']}")
print(f"Statistics: {result['statistics']}")
```

---

## Database Schema

### extraction_experiences Table

```sql
CREATE TABLE extraction_experiences (
    id UUID PRIMARY KEY,
    text_snippet TEXT NOT NULL,
    text_embedding_id VARCHAR,
    entities JSONB DEFAULT '[]',
    relationships JSONB DEFAULT '[]',
    extraction_patterns JSONB DEFAULT '{}',
    domain VARCHAR NOT NULL,
    entity_types JSONB DEFAULT '[]',
    quality_score FLOAT NOT NULL,
    extraction_method VARCHAR,
    document_id UUID,
    chunk_id UUID,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX ix_experiences_domain_quality ON extraction_experiences(domain, quality_score);
CREATE INDEX ix_experiences_entity_types ON extraction_experiences USING GIN(entity_types);
CREATE INDEX ix_experiences_text_embedding ON extraction_experiences(text_embedding_id);
```

### domain_detection_feedback Table

```sql
CREATE TABLE domain_detection_feedback (
    id UUID PRIMARY KEY,
    document_id UUID,
    detected_domain VARCHAR,
    confidence FLOAT,
    detection_method VARCHAR,
    crypto_indicators JSONB,
    domain_scores JSONB,
    user_correction VARCHAR,
    feedback_source VARCHAR,
    extraction_quality FLOAT,
    entity_count INTEGER,
    was_accurate VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Indexes
CREATE INDEX ix_feedback_domain_accuracy ON domain_detection_feedback(detected_domain, was_accurate);
CREATE INDEX ix_feedback_confidence ON domain_detection_feedback(confidence);
CREATE INDEX ix_feedback_created_at ON domain_detection_feedback(created_at);
```

---

## Expected Improvements

### Experience Bank
- **10-15%** improvement in extraction accuracy via few-shot examples
- **Faster adaptation** to new crypto concepts as experiences accumulate
- **Better entity coverage** for domain-specific terms

### Prompt Evolution
- **10-20%** improvement in extraction quality over time
- **Domain-optimized prompts** that evolve with your content
- **Automatic optimization** without manual tuning

### Combined Effect
- **20-35%** total improvement in extraction quality
- **Reduced hallucinations** via consistency with past extractions
- **Better crypto entity recognition** (ETFs, protocols, treasuries)

---

## Monitoring

### Experience Bank Metrics

```bash
# Query database
psql -d knowledge_base -c "
SELECT 
    domain,
    COUNT(*) as examples,
    AVG(quality_score) as avg_quality
FROM extraction_experiences 
GROUP BY domain;
"

# Check retrieval patterns
psql -d knowledge_base -c "
SELECT 
    domain,
    AVG(retrieval_count) as avg_retrievals,
    SUM(retrieval_count) as total_retrievals
FROM extraction_experiences
GROUP BY domain;
"
```

### Domain Detection Metrics

```bash
# Accuracy by domain
psql -d knowledge_base -c "
SELECT 
    detected_domain,
    COUNT(*) as total,
    SUM(CASE WHEN was_accurate = 'yes' THEN 1 ELSE 0 END) as accurate,
    ROUND(
        100.0 * SUM(CASE WHEN was_accurate = 'yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) as accuracy_pct
FROM domain_detection_feedback
WHERE was_accurate IS NOT NULL
GROUP BY detected_domain;
"
```

### Quality Trends

```bash
# Average quality over time
psql -d knowledge_base -c "
SELECT 
    DATE(created_at) as date,
    COUNT(*) as count,
    AVG(quality_score) as avg_quality
FROM extraction_experiences
GROUP BY DATE(created_at)
ORDER BY date DESC
LIMIT 30;
"
```

---

## Troubleshooting

### Experience Bank Not Storing

**Symptoms:** No extractions stored in `extraction_experiences` table

**Possible Causes:**
1. Quality threshold too high (0.75 is correct, not 0.85)
2. Database migration not run
3. Experience Bank disabled in config

**Solution:**
```bash
# Check migration
alembic current

# Verify threshold in code
# Should be 0.75 in orchestrator_self_improving.py

# Check logs for storage attempts
grep "Stored.*extraction experiences" /tmp/kbv2_ingestion.log
```

### Prompt Evolution Not Improving

**Symptoms:** Quality scores not improving over generations

**Possible Causes:**
1. Insufficient test documents (< 10)
2. LLM API issues
3. Mutation temperature too low/high

**Solution:**
```python
# Increase test documents
test_documents = [...]  # Need at least 10-20

# Adjust mutation temperature
config = PromptEvolutionConfig(mutation_temperature=0.8)

# Verify LLM gateway
curl http://localhost:8087/v1/health
```

### Domain Detection Accuracy Low

**Symptoms:** Documents frequently misclassified

**Possible Causes:**
1. Insufficient training data
2. Domain overlap
3. Keyword thresholds too strict

**Solution:**
```python
# Check accuracy stats
stats = await feedback.get_domain_accuracy_stats()

# Get improvement suggestions
suggestions = await feedback.get_improvement_suggestions()

# Record user corrections to improve
await feedback.record_detection(..., user_correction="CORRECT_DOMAIN")
```

### Ontology Validation Too Strict

**Symptoms:** Many entities rejected as invalid

**Possible Causes:**
1. Rules too restrictive
2. Schema outdated
3. Domain mismatch

**Solution:**
```python
# Check validation report
report = await validator.validate_extraction(entities, relationships)

# Review violations
for violation in report.violations:
    print(f"{violation.violation_type}: {violation.message}")

# Adjust validator rules if needed
# Edit: src/knowledge_base/intelligence/v1/self_improvement/ontology_validator.py
```

---

## Best Practices

1. **Monitor First Few Documents** - Review quality scores and stored experiences
2. **Provide User Feedback** - Record domain corrections to improve detection
3. **Regular Evolution** - Run prompt evolution weekly for active domains
4. **Review Validation Reports** - Check for systematic rule violations
5. **Monitor Hit Rates** - Track Experience Bank retrieval effectiveness

---

## Next Steps

1. **Process Documents** - Ingest documents to populate Experience Bank
2. **Monitor Learning** - Track accuracy and quality improvements
3. **Evolve Prompts** - Run prompt evolution for domains with many documents
4. **Review Feedback** - Analyze domain detection patterns
5. **Tune Thresholds** - Adjust quality thresholds based on observed behavior

---

## Related Documentation

- [Ingestion Guide](ingestion.md)
- [Deployment Guide](deployment.md)
- [Available Features](available_features.md)

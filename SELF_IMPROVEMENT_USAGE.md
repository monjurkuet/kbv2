# Self-Improving Crypto Knowledgebase - Implementation Summary

## Overview

This implementation adds **Tier 1 Self-Improvement** capabilities to KBv2 with a focus on cryptocurrency domain. The system now includes:

1. **Experience Bank** - Stores and retrieves high-quality extraction examples
2. **Prompt Evolution** - Automatically optimizes extraction prompts
3. **Self-Improving Orchestrator** - Integrates everything into the ingestion pipeline

## Files Created

### Core Self-Improvement Module
```
src/knowledge_base/intelligence/v1/self_improvement/
├── __init__.py                    # Module exports
├── experience_bank.py             # ExperienceBank implementation
├── extraction_integration.py      # Integration layer
└── prompt_evolution.py            # Automated prompt evolution
```

### Enhanced Orchestrator
```
src/knowledge_base/orchestrator_self_improving.py  # Extended orchestrator
```

### Database Migration
```
migrations/versions/experience_bank_001.py         # Creates extraction_experiences table
```

## Features Implemented

### 1. Experience Bank

**Capabilities:**
- Stores high-quality extractions (quality > 0.85)
- Retrieves similar examples for few-shot prompting
- Tracks usage statistics and domain distribution
- Pattern extraction from successful extractions

**Usage:**
```python
from knowledge_base.intelligence.v1.self_improvement import (
    ExperienceBank, ExperienceBankConfig
)

# Initialize
config = ExperienceBankConfig(min_quality_threshold=0.85)
bank = ExperienceBank(session=db_session, config=config)

# Store successful extraction
experience_id = await bank.store_experience(
    text=document_text,
    entities=extracted_entities,
    relationships=extracted_relationships,
    quality_score=0.92,
    domain="DEFI"
)

# Retrieve similar examples
examples = await bank.retrieve_similar_examples(
    text=new_document_text,
    domain="DEFI",
    k=3
)
```

### 2. Prompt Evolution

**Capabilities:**
- Generates prompt mutations for each crypto domain
- A/B tests prompt variants on validation documents
- Selects best performing prompts automatically
- Tracks performance metrics per variant

**Crypto Domains Supported:**
- `BITCOIN` - Bitcoin protocol, mining, ETFs
- `DEFI` - Protocols, TVL, yields
- `INSTITUTIONAL_CRYPTO` - ETFs, custody, treasuries
- `STABLECOINS` - USDC, USDT, post-GENIUS Act
- `CRYPTO_REGULATION` - SEC, CFTC, compliance

**Usage:**
```python
from knowledge_base.intelligence.v1.self_improvement import (
    PromptEvolutionEngine, PromptEvolutionConfig
)

# Initialize
evolution = PromptEvolutionEngine(gateway=llm_gateway)

# Initialize domain variants
variants = await evolution.initialize_domain("DEFI")

# Evolve to next generation
best_variant = await evolution.evolve_generation(
    domain="DEFI",
    test_documents=test_docs
)

# Get best prompt
prompt = await evolution.get_best_prompt("DEFI")
```

### 3. Self-Improving Orchestrator

**Integration Points:**
1. **Before Extraction:** Retrieves similar examples from Experience Bank
2. **During Extraction:** Uses evolved prompts for better entity recognition
3. **After Extraction:** Stores high-quality results back to Experience Bank

**Usage:**
```python
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

# Initialize
orchestrator = SelfImprovingOrchestrator(
    enable_experience_bank=True,
    enable_prompt_evolution=True,
)
await orchestrator.initialize()

# Process document (automatically uses self-improvement)
document = await orchestrator.process_document(
    file_path="crypto_report.pdf",
    domain="DEFI"
)

# Check stats
stats = await orchestrator.get_experience_bank_stats()
print(f"Stored experiences: {stats['total_experiences']}")

# Manually evolve prompts
result = await orchestrator.evolve_prompts(
    domain="DEFI",
    test_documents=test_docs
)
```

## Crypto Domain Ontologies

### Updated Domain Models

```python
class Domain(str, Enum):
    BITCOIN = "BITCOIN"
    DIGITAL_ASSETS = "DIGITAL_ASSETS"
    STABLECOINS = "STABLECOINS"
    BLOCKCHAIN_INFRA = "BLOCKCHAIN_INFRA"
    DEFI = "DEFI"
    CRYPTO_MARKETS = "CRYPTO_MARKETS"
    INSTITUTIONAL_CRYPTO = "INSTITUTIONAL_CRYPTO"
    CRYPTO_REGULATION = "CRYPTO_REGULATION"
    CRYPTO_AI = "CRYPTO_AI"
    TOKENIZATION = "TOKENIZATION"
    GENERAL = "GENERAL"
```

### Entity Types Added

**Bitcoin Domain:**
- `BitcoinETF` - IBIT, GBTC, FBTC, etc.
- `MiningPool` - Foundry USA, Antpool, F2Pool
- `DigitalAssetTreasury` - Strategy, Metaplanet, Tesla
- `LightningNode` - Lightning Network infrastructure
- `BitcoinUpgrade` - BIPs, soft forks

**DeFi Domain:**
- `DeFiProtocol` - Aave, Compound, Uniswap
- `DEX` - Decentralized exchanges
- `LendingProtocol` - Lending platforms
- `LiquidityPool` - AMM pools
- `YieldStrategy` - Vaults, farming

**Institutional Domain:**
- `ETFIssuer` - BlackRock, Grayscale, Fidelity
- `CryptoCustodian` - Coinbase, BitGo
- `CryptoFund` - Pantera, a16z crypto

**Regulatory Domain:**
- `RegulatoryBody` - SEC, CFTC, FINMA
- `Regulation` - GENIUS Act, MiCA
- `LegalCase` - SEC v Ripple, etc.

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
    last_retrieved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX ix_experiences_domain_quality ON extraction_experiences(domain, quality_score);
CREATE INDEX ix_experiences_entity_types ON extraction_experiences USING GIN(entity_types);
```

## Expected Improvements

### With Experience Bank
- **10-15%** improvement in extraction accuracy via few-shot examples
- **Faster adaptation** to new crypto concepts as experiences accumulate
- **Better entity coverage** for domain-specific terms

### With Prompt Evolution
- **10-20%** improvement in extraction quality over time
- **Domain-optimized prompts** that evolve with your content
- **Automatic optimization** without manual tuning

### Combined Effect
- **20-35%** total improvement in extraction quality
- **Reduced hallucinations** via consistency with past high-quality extractions
- **Better crypto entity recognition** (ETFs, protocols, treasuries)

## Migration Steps

### 1. Run Database Migration
```bash
alembic upgrade experience_bank_001
```

### 2. Update Dependencies
The implementation uses existing dependencies:
- SQLAlchemy (already used)
- Pydantic (already used)
- Asyncio (already used)

### 3. Switch to Self-Improving Orchestrator

**Before:**
```python
from knowledge_base.orchestrator import IngestionOrchestrator

orchestrator = IngestionOrchestrator()
```

**After:**
```python
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

orchestrator = SelfImprovingOrchestrator(
    enable_experience_bank=True,
    enable_prompt_evolution=True,
)
```

### 4. Monitor Progress

```python
# Check experience bank growth
stats = await orchestrator.get_experience_bank_stats()
print(f"Total experiences: {stats['total_experiences']}")
print(f"Domain distribution: {stats['domain_distribution']}")

# Check prompt evolution
for domain in ["BITCOIN", "DEFI", "INSTITUTIONAL_CRYPTO"]:
    result = await orchestrator.evolve_prompts(domain, test_docs)
    print(f"{domain}: Best quality = {result['statistics']['best_variant']['quality_score']}")
```

## Configuration Options

### Experience Bank Config
```python
ExperienceBankConfig(
    min_quality_threshold=0.85,  # Minimum quality to store
    max_storage_size=10000,      # Max experiences to keep
    similarity_top_k=3,          # Examples to retrieve
    max_text_length=2000,        # Text snippet length
    enable_pattern_extraction=True,
)
```

### Prompt Evolution Config
```python
PromptEvolutionConfig(
    num_variants_per_generation=5,  # Variants per evolution cycle
    max_generations=10,             # Max evolution generations
    mutation_temperature=0.7,       # LLM creativity for mutations
    min_evaluation_samples=10,      # Min docs to evaluate
    selection_threshold=0.75,       # Min quality to keep variant
    crypto_domains=["BITCOIN", "DEFI", "INSTITUTIONAL_CRYPTO", ...],
)
```

## Testing

### Test Experience Bank
```python
async def test_experience_bank():
    # Create test extraction
    entities = [
        {"name": "Aave", "entity_type": "DeFiProtocol", "confidence": 0.95},
        {"name": "$12.5B TVL", "entity_type": "Metric", "confidence": 0.90},
    ]
    
    # Store
    exp_id = await bank.store_experience(
        text="Aave V3 has $12.5B TVL with 3.2% APY",
        entities=entities,
        relationships=[],
        quality_score=0.92,
        domain="DEFI"
    )
    
    # Retrieve
    examples = await bank.retrieve_similar_examples(
        text="Compound has $8B in deposits",
        domain="DEFI",
        k=3
    )
    
    assert len(examples) > 0
    assert examples[0].quality_score >= 0.85
```

### Test Prompt Evolution
```python
async def test_prompt_evolution():
    # Initialize
    engine = PromptEvolutionEngine(gateway)
    variants = await engine.initialize_domain("BITCOIN")
    
    # Check variants created
    assert len(variants) >= 5  # base + 4 mutations
    
    # Get best prompt
    prompt = await engine.get_best_prompt("BITCOIN")
    assert "Bitcoin" in prompt or "BTC" in prompt
```

## Next Steps (Tier 2)

### 1. Ontology Consistency Validator
Implement rule-based validation to catch contradictions:
- "Bitcoin is deflationary" vs "Bitcoin has infinite supply"
- Stablecoin backing type consistency

### 2. Bidirectional Co-Induction
Extract triples and induce schema simultaneously for faster adaptation

### 3. Self-Distillation
Fine-tune local LLM on accumulated high-quality extractions

## Troubleshooting

### Experience Bank not storing
- Check quality_score >= min_quality_threshold (default 0.85)
- Verify database migration ran successfully
- Check logs for storage errors

### Prompt evolution not improving
- Ensure sufficient test documents (min 10)
- Check LLM gateway is responding
- Review mutation generation logs

### Performance issues
- Reduce similarity_top_k (default 3)
- Limit max_storage_size (default 10000)
- Disable pattern extraction if not needed

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify database schema with migration
3. Test individual components (Experience Bank, Prompt Evolution)
4. Review integration points in SelfImprovingOrchestrator

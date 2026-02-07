# KBv2 Crypto-Focused Self-Improving Knowledgebase
## Integrated Implementation Plan v2.0

**Date**: 2026-02-07  
**Status**: Updated with self-improvement analysis & 2026 research insights

---

## Executive Summary

This plan transforms KBv2 into a **self-improving cryptocurrency knowledgebase** that:
1. **Focuses exclusively** on crypto/Bitcoin domain
2. **Automatically evolves** its understanding through LLM-driven self-improvement
3. **Learns from experience** to continuously improve extraction quality

**Key Insight**: KBv2 already has sophisticated foundations (type discovery, multi-agent evaluation). We need to add **feedback loops** and **prompt evolution** to make it truly self-improving.

---

## Phase 0: Foundation (Week 1)

### Goal: Audit current state & establish baseline

**Tasks**:
1. **Run extraction quality audit** on 50 crypto documents
   - Measure entity extraction F1
   - Identify top 10 failure modes
   - Document hallucination patterns

2. **Inventory existing self-improvement features**:
   - ✅ Type Discovery (cross-document learning)
   - ✅ Schema Induction (schema evolution)
   - ✅ Multi-Agent Extractor (LLM-as-Judge)
   - ✅ Adaptive Ingestion Engine (per-document optimization)
   - ❌ Feedback loops to future extractions
   - ❌ Prompt evolution
   - ❌ Ontology consistency checking

3. **Establish success metrics**:
   - Entity Extraction F1 (baseline)
   - Hallucination rate (baseline)
   - Schema coverage (current types)
   - Human review rate (current)

**Deliverable**: `docs/baseline_report.md`

---

## Phase 1: Tier 1 Self-Improvement (Weeks 2-4)

### Goal: Implement high-impact, quick wins

### 1.1 Experience Bank (Week 2)

**Purpose**: Store and reuse successful extraction patterns

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/experience_bank.py

class ExperienceBank:
    """Store high-quality extractions as reusable examples."""
    
    async def store_experience(
        self, 
        text: str, 
        extraction: ExtractionResult,
        quality_score: float
    ):
        """Store extraction if quality > 0.9."""
        if quality_score > 0.9:
            await self.vector_store.add_document({
                "text_snippet": text[:1000],
                "entities": extraction.entities,
                "patterns": self._extract_patterns(extraction),
                "quality_score": quality_score,
                "domain": "CRYPTO",
                "timestamp": datetime.utcnow()
            })
    
    async def retrieve_similar_examples(
        self, 
        text: str, 
        k: int = 3
    ) -> List[ExtractionExample]:
        """Retrieve similar successful extractions."""
        return await self.vector_store.similarity_search(
            query=text,
            filter={"quality_score": "> 0.9", "domain": "CRYPTO"},
            k=k
        )
```

**Integration Points**:
- Hook into `MultiAgentExtractor` to save high-quality extractions
- Retrieve examples in `PerceptionAgent` for few-shot prompting

**Success Metric**: 
- Store 100+ high-quality crypto extractions
- 15% improvement in extraction accuracy via better few-shot examples

---

### 1.2 Prompt Evolution System (Weeks 3-4)

**Purpose**: Automatically evolve crypto extraction prompts

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/prompt_evolution.py

class PromptEvolution:
    """Evolve extraction prompts via mutation and selection."""
    
    async def evolve_crypto_prompts(self):
        """Evolve prompts specifically for crypto domain."""
        
        # 1. Generate variants
        variants = await self._generate_prompt_variants(
            domain="CRYPTO",
            n_variants=5
        )
        
        # 2. A/B test on validation set
        results = []
        for variant in variants:
            score = await self._evaluate_prompt(
                prompt=variant,
                test_docs=self.crypto_validation_set
            )
            results.append((variant, score))
        
        # 3. Deploy top performer
        best_prompt = max(results, key=lambda x: x[1])
        await self._deploy_prompt("CRYPTO", best_prompt)
    
    async def _generate_prompt_variants(self, domain: str, n_variants: int) -> List[str]:
        """Use LLM to generate prompt variations."""
        base_prompt = self.template_registry.get_prompt(domain)
        
        mutation_prompt = f"""
        You are an expert prompt engineer for cryptocurrency entity extraction.
        
        Generate {n_variants} variations of this extraction prompt.
        Each variation should emphasize a different aspect:
        - Variant 1: Focus on precision (reduce false positives)
        - Variant 2: Focus on recall (catch more entities)
        - Variant 3: Focus on relationships (extract more edges)
        - Variant 4: Focus on Bitcoin-specific entities
        - Variant 5: Focus on DeFi protocol details
        
        Current prompt:
        {base_prompt}
        
        Generate variations that are structurally different but preserve core intent.
        """
        
        response = await self.gateway.complete(mutation_prompt)
        return self._parse_variants(response)
```

**Crypto-Specific Prompt Variants**:

**Variant A: Bitcoin Focus**:
```
Extract Bitcoin-specific entities: addresses, transactions, mining pools, 
halving events, Lightning Network nodes, ETFs (IBIT, GBTC, etc.), 
Digital Asset Treasuries (Strategy, Metaplanet), and nation-state adoption.
```

**Variant B: DeFi Focus**:
```
Extract DeFi entities: protocols (Aave, Uniswap, Compound), 
TVL figures, APY/APR rates, liquidity pools, governance tokens, 
yield strategies, and smart contract addresses.
```

**Variant C: Institutional Focus**:
```
Extract institutional crypto entities: ETFs, custodians (Coinbase, BitGo), 
wirehouses (Morgan Stanley, Merrill Lynch), 401(k) providers, 
regulatory bodies (SEC, CFTC), and compliance frameworks.
```

**Success Metric**:
- 10-20% improvement in extraction quality
- Prompt evolution cycle runs weekly

---

### 1.3 Dynamic Few-Shot Enhancement (Week 4)

**Purpose**: Use Experience Bank for retrieval-augmented few-shot prompting

**Implementation**:
```python
# Enhanced MultiAgentExtractor

class PerceptionAgent:
    """Enhanced with dynamic few-shot examples."""
    
    async def extract(
        self, 
        text: str,
        experience_bank: ExperienceBank
    ) -> List[EntityCandidate]:
        
        # Retrieve similar successful extractions
        examples = await experience_bank.retrieve_similar_examples(text, k=3)
        
        # Build few-shot prompt
        prompt = self._build_few_shot_prompt(text, examples)
        
        # Extract with examples
        return await self._extract_with_prompt(prompt)
```

**Deliverables**:
- ✅ Experience Bank operational
- ✅ Prompt Evolution system running
- ✅ Dynamic few-shot integration
- 📊 Baseline vs Tier 1 comparison report

---

## Phase 2: Crypto Domain Foundation (Weeks 5-7)

### Goal: Establish comprehensive crypto domain

### 2.1 Domain Model Refactoring (Week 5)

**Update** `src/knowledge_base/domain/domain_models.py`:

```python
class Domain(str, Enum):
    """Cryptocurrency-focused domains only."""
    
    BITCOIN = "BITCOIN"                    # Bitcoin-specific
    DIGITAL_ASSETS = "DIGITAL_ASSETS"      # General crypto assets
    STABLECOINS = "STABLECOINS"           # Stablecoin-specific
    BLOCKCHAIN_INFRA = "BLOCKCHAIN_INFRA" # L1/L2, consensus
    DEFI = "DEFI"                         # DeFi protocols
    CRYPTO_MARKETS = "CRYPTO_MARKETS"     # Trading, on-chain metrics
    INSTITUTIONAL_CRYPTO = "INSTITUTIONAL_CRYPTO"  # ETFs, custody
    CRYPTO_REGULATION = "CRYPTO_REGULATION"        # SEC, CFTC
    CRYPTO_AI = "CRYPTO_AI"               # AI-blockchain convergence
    TOKENIZATION = "TOKENIZATION"         # RWA, asset tokenization
    
    GENERAL = "GENERAL"  # Fallback only
```

### 2.2 Ontology Expansion (Week 6)

**Update** `src/knowledge_base/domain/ontology_snippets.py`:

Add comprehensive crypto ontologies (from original plan):
- 500+ keywords per domain
- 50+ entity types
- Comprehensive relationship types

**Key Entity Types**:
```python
CRYPTO_ENTITY_TYPES = {
    # Assets
    "Cryptocurrency", "Stablecoin", "GovernanceToken",
    
    # Bitcoin-Specific
    "BitcoinUpgrade", "MiningPool", "BitcoinETF", "DigitalAssetTreasury",
    
    # Infrastructure
    "Blockchain", "Layer2", "SmartContract", "ConsensusMechanism",
    
    # DeFi
    "DeFiProtocol", "DEX", "LendingProtocol", "YieldStrategy", "LiquidityPool",
    
    # Market
    "CryptoExchange", "MarketIndicator", "TradingPair", "ETFIssuer",
    
    # Institutional
    "CryptoCustodian", "DigitalAssetTreasury", "CryptoFund",
    
    # Regulatory
    "RegulatoryBody", "Regulation", "ComplianceFramework",
}
```

### 2.3 Enhanced Extraction Prompts (Week 7)

**Create** `src/knowledge_base/extraction/crypto_prompts.py`:

```python
BITCOIN_EXTRACTION_PROMPT = """
You are a Bitcoin knowledge extraction specialist.

Extract entities with focus on:
1. Bitcoin protocol entities (addresses, transactions, blocks)
2. Mining entities (pools, difficulty, hash rate, halving)
3. Scaling solutions (Lightning Network, sidechains, L2s)
4. Institutional products (ETFs: IBIT, GBTC, FBTC, ARKB)
5. Treasury holdings (Strategy, Metaplanet, nation states)
6. Market metrics (MVRV, NUPL, HODL waves, realized price)

Extract quantities:
- Bitcoin amounts (e.g., "holds 100,000 BTC")
- Hash rates (e.g., "500 EH/s")
- ETF flows (e.g., "$500M inflow")

Output: JSON with entities and relationships.
"""

DEFI_EXTRACTION_PROMPT = """
You are a DeFi protocol extraction specialist.

Extract:
1. Protocols (Aave, Compound, Uniswap, etc.)
2. TVL figures and changes
3. APY/APR rates for lending/yield
4. Liquidity pools and token pairs
5. Governance tokens and proposals
6. Smart contract addresses

Metrics to capture:
- TVL: $5.2B
- APY: 8.5%
- Market Cap: $1.2B
- FDV: $2.1B

Output: JSON with entities and financial metrics.
"""

INSTITUTIONAL_EXTRACTION_PROMPT = """
You are an institutional crypto adoption specialist.

Extract:
1. ETF issuers (BlackRock, Grayscale, Fidelity)
2. ETF products (IBIT, GBTC, spot vs futures)
3. Custody providers (Coinbase, BitGo, BNY Mellon)
4. Wirehouses (Morgan Stanley, Merrill Lynch, UBS)
5. Corporate treasuries (Strategy, Tesla, Block)
6. Regulatory entities (SEC, CFTC, legislation)

Key metrics:
- AUM (Assets Under Management)
- Expense ratios
- Flows (inflows/outflows)
- Cost basis

Output: JSON with entities and institutional metrics.
"""
```

**Deliverables**:
- ✅ 10 crypto domains defined
- ✅ Comprehensive ontologies (500+ keywords)
- ✅ 50+ entity types
- ✅ Domain-specific extraction prompts

---

## Phase 3: Tier 2 Self-Improvement (Weeks 8-11)

### Goal: Advanced quality and consistency

### 3.1 Ontology Consistency Validator (Weeks 8-9)

**Purpose**: Validate extractions against crypto ontology rules

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/ontology_validator.py

class OntologyValidator:
    """Validate crypto extractions against ontology rules."""
    
    def __init__(self):
        self.rules = self._load_crypto_rules()
    
    def _load_crypto_rules(self) -> List[OntologyRule]:
        return [
            # Rule 1: BitcoinETF must have issuer
            OntologyRule(
                entity_type="BitcoinETF",
                required_properties=["issuer", "ticker"],
                severity="ERROR"
            ),
            
            # Rule 2: DeFiProtocol must have TVL or MarketCap
            OntologyRule(
                entity_type="DeFiProtocol",
                required_properties=["tvl", "market_cap"],
                at_least_one=True,
                severity="WARNING"
            ),
            
            # Rule 3: MiningPool must have hashrate
            OntologyRule(
                entity_type="MiningPool",
                required_properties=["hash_rate", "blocks_mined_24h"],
                at_least_one=True,
                severity="WARNING"
            ),
            
            # Rule 4: Stablecoin must have backing_type
            OntologyRule(
                entity_type="Stablecoin",
                required_properties=["backing_type", "collateral_ratio"],
                severity="ERROR"
            ),
        ]
    
    async def validate_extraction(
        self, 
        entities: List[Entity],
        edges: List[Edge]
    ) -> ValidationReport:
        """Validate extraction against ontology rules."""
        violations = []
        
        # Check type-property rules
        for entity in entities:
            rule = self.rules.get(entity.entity_type)
            if rule:
                missing = self._check_required_properties(entity, rule)
                if missing:
                    violations.append(PropertyViolation(
                        entity=entity,
                        missing_properties=missing,
                        severity=rule.severity
                    ))
        
        # Semantic consistency check (via LLM)
        semantic_issues = await self._llm_semantic_check(entities, edges)
        
        return ValidationReport(violations, semantic_issues)
    
    async def _llm_semantic_check(
        self, 
        entities: List[Entity],
        edges: List[Edge]
    ) -> List[SemanticIssue]:
        """Check for semantic contradictions."""
        prompt = f"""
        Check these cryptocurrency extractions for semantic contradictions:
        
        Entities: {self._format_entities(entities)}
        Relationships: {self._format_edges(edges)}
        
        Look for contradictions like:
        - "Bitcoin is deflationary" AND "Bitcoin has infinite supply"
        - "USDC is fiat-backed" AND "USDC is algorithmic"
        - "IBIT is a spot ETF" AND "IBIT is a futures ETF"
        
        Return any contradictions found.
        """
        
        response = await self.gateway.complete(prompt)
        return self._parse_semantic_issues(response)
```

**Success Metric**:
- 20-30% reduction in hallucinations
- Auto-correction of 50% of violations

---

### 3.2 Bidirectional Schema-Triple Co-Induction (Weeks 10-11)

**Purpose**: Extract triples AND induce schema simultaneously

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/co_induction.py

class CoInductionEngine:
    """Simultaneous schema extraction and triple extraction."""
    
    async def co_induce(
        self, 
        text: str, 
        current_schema: Schema
    ) -> CoInductionResult:
        """
        Extract triples and update schema in a loop.
        """
        iteration = 0
        max_iterations = 3
        
        extraction = None
        schema = current_schema
        
        while iteration < max_iterations:
            # 1. Extract with current schema
            extraction = await self._extract_with_schema(text, schema)
            
            # 2. Discover new types from extraction
            new_types = await self._discover_new_types(extraction)
            
            # 3. If new types found, update schema and re-extract
            if new_types:
                schema = await self._update_schema(schema, new_types)
                iteration += 1
            else:
                # No new types, we're done
                break
        
        return CoInductionResult(extraction, schema, iteration)
```

**Success Metric**:
- Schema updates in real-time
- 20% more entity types discovered
- Faster adaptation to new crypto concepts

---

## Phase 4: Data Integration (Weeks 12-14)

### Goal: Connect to real-time crypto data sources

### 4.1 Data Source Connectors

**Create** `src/knowledge_base/ingestion/crypto_data/`:

```python
# ETF Flow Connector
class ETFFlowConnector:
    """Ingest ETF flow data from Bloomberg/ETF.com."""
    
    async def fetch_flows(
        self,
        etfs: List[str] = ["IBIT", "GBTC", "FBTC", "ARKB"],
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[ETFFlow]:
        pass

# On-Chain Metrics Connector  
class OnChainConnector:
    """Fetch on-chain metrics from Amberdata/Glassnode."""
    
    async def fetch_metrics(
        self,
        metrics: List[str] = ["mvrv", "nupl", "hodl_waves"],
        timeframe: str = "1d"
    ) -> List[OnChainMetric]:
        pass

# DeFi TVL Connector
class DeFiConnector:
    """Fetch DeFi data from DeFiLlama."""
    
    async def fetch_tvl(
        self,
        protocols: List[str] = None
    ) -> List[ProtocolTVL]:
        pass
```

### 4.2 Document Processors

**Processors for**:
- Research reports (Grayscale, Pantera, Coinbase)
- Protocol documentation (BIPs, EIPs)
- Regulatory filings (SEC, CFTC)
- Institutional communications (ETF prospectuses)

---

## Phase 5: Tier 3 Self-Improvement (Weeks 15-20)

### Goal: Cutting-edge capabilities

### 5.1 Self-Distillation Training Pipeline (Weeks 15-17)

**Purpose**: Fine-tune local LLM on accumulated crypto extractions

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/self_distillation.py

class SelfDistillationPipeline:
    """Generate training data and fine-tune model."""
    
    async def generate_training_dataset(
        self,
        min_quality: float = 0.95,
        min_samples: int = 1000
    ) -> Dataset:
        """Generate instruction-following dataset from extractions."""
        
        # 1. Select high-quality extractions
        extractions = await self._select_high_quality(
            min_quality=min_quality,
            min_samples=min_samples
        )
        
        # 2. Convert to instruction format
        instructions = []
        for extraction in extractions:
            instruction = {
                "instruction": f"Extract entities from this crypto text: {extraction.text[:500]}",
                "input": extraction.text,
                "output": json.dumps({
                    "entities": extraction.entities,
                    "relationships": extraction.edges
                })
            }
            instructions.append(instruction)
        
        return Dataset(instructions)
    
    async def fine_tune_model(self, dataset: Dataset):
        """Fine-tune via SFT + DPO."""
        # Stage 1: Supervised Fine-Tuning
        sft_model = await self._sft_train(dataset)
        
        # Stage 2: Direct Preference Optimization
        # Generate preference pairs from quality scores
        preference_pairs = self._create_preference_pairs(dataset)
        dpo_model = await self._dpo_train(sft_model, preference_pairs)
        
        return dpo_model
```

**Success Metric**:
- 30-50% long-term extraction quality improvement
- Model specialized for crypto domain

---

### 5.2 Conceptualization Layer (Weeks 18-20)

**Purpose**: Abstract crypto instances into semantic categories

**Implementation**:
```python
# src/knowledge_base/intelligence/v1/self_improvement/conceptualizer.py

class CryptoConceptualizer:
    """Abstract crypto entities into investment themes."""
    
    async def conceptualize(self, entities: List[Entity]) -> ConceptHierarchy:
        """
        Abstract specific instances into concepts.
        
        Examples:
        - IBIT, GBTC, FBTC → SpotBitcoinETF
        - Aave, Compound, Morpho → LendingProtocol
        - Strategy, Metaplanet, Tesla → BitcoinTreasury
        """
        
        # 1. Cluster similar entities
        clusters = await self._cluster_by_embedding(entities)
        
        # 2. Generate concept names
        concepts = []
        for cluster in clusters:
            concept_name = await self._generate_concept_name(cluster)
            concepts.append(Concept(
                name=concept_name,
                instances=cluster,
                properties=self._infer_common_properties(cluster)
            ))
        
        # 3. Build hierarchy
        hierarchy = await self._build_concept_hierarchy(concepts)
        
        return hierarchy
```

**Success Metric**:
- Auto-discover 10+ investment themes
- Enable semantic reasoning ("treasury adoption trend")

---

## Success Metrics & Milestones

### Phase 0 (Week 1):
- [ ] Baseline metrics established
- [ ] 50 crypto documents audited

### Phase 1 (Weeks 2-4):
- [ ] Experience Bank with 100+ high-quality extractions
- [ ] Prompt Evolution running weekly
- [ ] 10-20% extraction quality improvement

### Phase 2 (Weeks 5-7):
- [ ] 10 crypto domains defined
- [ ] Comprehensive ontologies (500+ keywords)
- [ ] Domain-specific prompts deployed

### Phase 3 (Weeks 8-11):
- [ ] Ontology Validator operational
- [ ] 20-30% hallucination reduction
- [ ] Co-Induction Engine running

### Phase 4 (Weeks 12-14):
- [ ] Real-time data connectors operational
- [ ] 5+ document processors created

### Phase 5 (Weeks 15-20):
- [ ] Self-distillation pipeline trained
- [ ] 30-50% long-term improvement
- [ ] Conceptualization layer operational

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| **0: Baseline** | Week 1 | Baseline metrics report |
| **1: Tier 1** | Weeks 2-4 | Experience Bank + Prompt Evolution |
| **2: Domain** | Weeks 5-7 | Crypto domains + ontologies |
| **3: Tier 2** | Weeks 8-11 | Ontology Validator + Co-Induction |
| **4: Data** | Weeks 12-14 | Real-time data connectors |
| **5: Tier 3** | Weeks 15-20 | Self-distillation + Conceptualization |

**Total**: 20 weeks to production-grade self-improving crypto knowledgebase

---

## Immediate Next Steps

### This Week:
1. **Run baseline audit** on 50 crypto documents
2. **Design Experience Bank schema**
3. **Create crypto validation set** (100 documents)

### Next Week:
4. **Implement Experience Bank**
5. **Integrate with MultiAgentExtractor**
6. **Test retrieval-augmented few-shot**

---

## Files to Create/Modify

### New Files:
```
src/knowledge_base/intelligence/v1/self_improvement/
├── __init__.py
├── experience_bank.py          # Store/retrieve high-quality extractions
├── prompt_evolution.py         # Automated prompt optimization
├── ontology_validator.py       # Consistency checking
├── co_induction.py            # Bidirectional schema-text induction
├── self_distillation.py       # Training data generation
├── conceptualizer.py          # Abstraction layer
└── crypto_rules.py            # Crypto ontology rules

src/knowledge_base/ingestion/crypto_data/
├── __init__.py
├── etf_flow_connector.py
├── onchain_connector.py
├── defi_connector.py
└── price_connector.py
```

### Modified Files:
```
src/knowledge_base/domain/domain_models.py        # Add crypto domains
src/knowledge_base/domain/ontology_snippets.py    # Add crypto ontologies
src/knowledge_base/extraction/
├── crypto_prompts.py                            # Domain-specific prompts
└── guided_extractor.py                          # Use evolved prompts

src/knowledge_base/intelligence/v1/
├── multi_agent_extractor.py                     # Integrate Experience Bank
└── template_registry.py                         # Add crypto templates
```

---

*Plan Version: 2.0*  
*Updated: 2026-02-07*  
*Incorporates: Self-improvement analysis + 2026 research insights*

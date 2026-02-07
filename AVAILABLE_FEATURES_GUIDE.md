# 📦 Available Features - Usage Guide

This guide explains the features that are **built but not integrated** into the main ingestion pipeline, and how to use them.

---

## 1️⃣ Reranking Pipeline (Search-Time Feature)

### What It Does
Improves search results by:
1. Running hybrid search (vector + BM25) to get initial candidates
2. Reranking results using a cross-encoder model (ms-marco-MiniLM-L-6-v2)
3. Optionally fusing multiple query results with Reciprocal Rank Fusion (RRF)

### When to Use
- **Searching** your knowledge base (not during ingestion)
- When you want more accurate search results
- For multi-query searches that need result fusion

### How to Use

```python
from knowledge_base.storage.hybrid_search import HybridSearchEngine
from knowledge_base.reranking.cross_encoder import CrossEncoderReranker
from knowledge_base.reranking.reranking_pipeline import RerankingPipeline

# Initialize
hybrid_search = HybridSearchEngine()
cross_encoder = CrossEncoderReranker()
pipeline = RerankingPipeline(hybrid_search, cross_encoder)

# Search with reranking
results = await pipeline.search(
    query="Bitcoin ETF approval",
    initial_top_k=50,  # Get 50 candidates
    final_top_k=10,    # Return top 10 after reranking
)

# Each result has:
# - vector_score: Initial vector similarity
# - bm25_score: BM25 text score
# - cross_encoder_score: Reranker relevance score
# - reranked_score: Final combined score
```

### Multi-Query Search with RRF
```python
from knowledge_base.reranking.rrf_fuser import ReciprocalRankFuser

rrf = ReciprocalRankFuser()
pipeline = RerankingPipeline(hybrid_search, cross_encoder, rrf)

# Search multiple queries and fuse results
results = await pipeline.multi_query_search(
    queries=[
        "Bitcoin price prediction",
        "BTC market analysis",
        "cryptocurrency trading"
    ],
    initial_top_k=30,
    final_top_k=10,
)
```

### API Endpoints (if using server)
```bash
# Standard hybrid search
POST /api/v1/queries/hybrid-search

# Reranked search
POST /api/v1/queries/reranked-search

# With explanation
POST /api/v1/queries/reranked-search-explain
```

---

## 2️⃣ Community Summarizer

### What It Does
Generates hierarchical summaries of entity communities at 4 levels:
- **MACRO**: Broad themes (e.g., "Cryptocurrency Markets")
- **MESO**: Sub-topics (e.g., "Bitcoin ETFs", "DeFi Protocols")
- **MICRO**: Specific entities (e.g., "BlackRock IBIT")
- **NANO**: Atomic entities (e.g., individual transactions)

### When to Use
- After entity extraction to understand document structure
- For generating document overviews
- To identify key themes and topics
- For knowledge graph navigation

### How to Use

```python
from knowledge_base.summaries.community_summaries import CommunitySummarizer

summarizer = CommunitySummarizer()

# Generate multi-level summary
summary = await summarizer.generate_multi_level_summary(
    document_id="doc-123",
    entities=[
        {"name": "Bitcoin", "type": "Cryptocurrency", "community_id": "macro_1"},
        {"name": "BlackRock", "type": "Company", "community_id": "macro_2"},
        {"name": "IBIT", "type": "ETF", "community_id": "macro_2"},
    ],
    relationships=[
        {"source": "BlackRock", "target": "IBIT", "type": "issues"},
    ],
)

# Access different levels
print(f"Macro communities: {len(summary.macro_communities)}")
print(f"Meso communities: {len(summary.meso_communities)}")
print(f"Hierarchy tree: {summary.hierarchy_tree}")
```

### LLM-Based Community Naming
```python
# Use LLM to intelligently name communities
result = await summarizer.name_community(
    entities=["Bitcoin", "Ethereum", "Solana"],
    relationships=["Bitcoin is the first cryptocurrency"],
)

print(f"Name: {result.suggested_name}")
print(f"Description: {result.description}")
print(f"Themes: {result.key_themes}")
```

### Integration with Ingestion
To add this to the ingestion pipeline:

```python
# In entity_pipeline_service.py, after clustering:
from knowledge_base.summaries.community_summaries import CommunitySummarizer

async def process_with_summaries(self, document, entities, relationships):
    # Existing processing...
    clusters = await self.clustering_service.cluster(entities)
    
    # NEW: Generate community summaries
    summarizer = CommunitySummarizer()
    summary = await summarizer.generate_multi_level_summary(
        document_id=str(document.id),
        entities=entities,
        relationships=relationships,
    )
    
    # Store summary in database
    await self.store_community_summary(document.id, summary)
```

---

## 3️⃣ Guided Extractor

### What It Does
Extracts entities based on **user-defined goals** rather than automatic detection:
- Automated mode: Detects domain and uses default goals
- Guided mode: Uses specific user goals (e.g., "find all ETF issuers")
- Iterative mode: Builds on previous extractions

### When to Use
- When you have specific extraction requirements
- For targeted entity extraction
- When automatic extraction misses important entities
- For domain-specific extraction tasks

### How to Use

#### Automated Mode (No User Goals)
```python
from knowledge_base.extraction.guided_extractor import GuidedExtractor
from knowledge_base.clients.llm import AsyncLLMClient

llm_client = AsyncLLMClient()
extractor = GuidedExtractor(llm_client)

# Automated extraction
prompts = await extractor.generate_extraction_prompts(
    document_text="Your document text here...",
    user_goals=None,  # Automated mode
)

# System detects domain and generates appropriate prompts
print(f"Detected domain: {prompts.domain}")
print(f"Number of prompts: {len(prompts.prompts)}")
```

#### Guided Mode (With User Goals)
```python
# User provides specific extraction goals
user_goals = [
    "Find all companies holding Bitcoin in their treasury",
    "Extract the amount of BTC each company holds",
    "Identify the cost basis for each holding",
]

prompts = await extractor.generate_extraction_prompts(
    document_text=document_text,
    user_goals=user_goals,
    domain="INSTITUTIONAL_CRYPTO",  # Optional domain hint
)

# Prompts are tailored to user goals
for prompt in prompts.prompts:
    print(f"Goal: {prompt.goal_name}")
    print(f"Target entities: {prompt.target_entities}")
```

#### Iterative Mode
```python
# First pass
pass1 = await extractor.generate_extraction_prompts(
    document_text=text,
    user_goals=["Find ETF issuers"],
)

# Second pass - build on first
pass2 = await extractor.generate_extraction_prompts(
    document_text=text,
    user_goals=["Find AUM data"],
    previous_extraction=pass1,  # Don't duplicate ETF issuers
)
```

### Template Registry
The extractor uses domain-specific templates:

```python
from knowledge_base.extraction.template_registry import TemplateRegistry

registry = TemplateRegistry()

# List available domains
domains = registry.list_domains()
print(f"Available domains: {domains}")

# Get default goals for a domain
goals = registry.get_default_goals("BITCOIN")
for goal in goals:
    print(f"Goal: {goal.name}")
    print(f"Description: {goal.description}")
    print(f"Entity types: {goal.entity_types}")
```

### Integration with Ingestion
To add this to the ingestion pipeline:

```python
# In orchestrator.py, modify process_document():
async def process_document(
    self,
    file_path: str,
    document_name: Optional[str] = None,
    domain: str = "GENERAL",
    user_goals: Optional[List[str]] = None,  # NEW parameter
):
    # ... existing code ...
    
    # Pass user_goals to entity pipeline
    entity_results = await self._entity_service.process(
        document=document,
        chunks=chunks,
        domain=domain,
        user_goals=user_goals,  # NEW
    )
```

Then in `entity_pipeline_service.py`:
```python
async def process(
    self,
    document,
    chunks,
    domain,
    user_goals=None,  # NEW parameter
):
    # Choose extractor based on user_goals
    if user_goals:
        from knowledge_base.extraction.guided_extractor import GuidedExtractor
        extractor = GuidedExtractor(self._llm_client)
        prompts = await extractor.generate_extraction_prompts(
            document_text="\n".join([c.text for c in chunks]),
            user_goals=user_goals,
            domain=domain,
        )
        # Use prompts for extraction
    else:
        # Use default MultiAgentExtractor
        extractor = MultiAgentExtractor(...)
```

---

## Summary Table

| Feature | Purpose | When to Use | Integration Effort |
|---------|---------|-------------|-------------------|
| **Reranking Pipeline** | Improve search results | Querying the knowledge base | Already available via API |
| **Community Summarizer** | Generate hierarchical summaries | After extraction for overview | Medium - add to EntityPipelineService |
| **Guided Extractor** | User-controlled extraction | Specific extraction needs | Medium - add user_goals parameter |

---

## Next Steps

1. **Reranking Pipeline**: Already available - use when querying
2. **Community Summarizer**: Add to EntityPipelineService for automatic summary generation
3. **Guided Extractor**: Add `user_goals` parameter to enable targeted extraction

All features are **tested and working** - they just need to be wired into the main pipeline if you want automatic execution during ingestion.

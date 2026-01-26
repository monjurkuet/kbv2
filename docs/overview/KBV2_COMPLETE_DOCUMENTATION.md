# KBV2 Knowledge Base System - Complete Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Components](#core-components)
4. [New LLM-Powered Services (2025-2026)](#new-llm-powered-services-2025-2026)
5. [Data Flow and Pipelines](#data-flow-and-pipelines)
6. [API Endpoints](#api-endpoints)
7. [Configuration and Environment](#configuration-and-environment)
8. [Testing Framework](#testing-framework)
9. [Research Foundation](#research-foundation)
10. [Quick Start Guide](#quick-start-guide)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)

---

## Executive Summary

KBV2 is a high-fidelity Knowledge Base system designed for advanced entity extraction, multi-domain knowledge management, and intelligent query processing. The system leverages Large Language Models (LLMs) for entity typing, multi-agent orchestration for complex extraction tasks, and hybrid retrieval mechanisms combining vector and graph-based approaches.

### Key Capabilities

- **LLM-Powered Entity Extraction**: Multi-agent system with Manager, Perception, Enhancement, and Evaluation agents
- **Entity Typing**: Few-shot prompting with domain-aware classification
- **Hallucination Detection**: LLM-as-Judge verification layer
- **Multi-Domain Management**: Domain schemas with inheritance
- **Cross-Domain Relationships**: Detection of relationships across different knowledge domains
- **Federated Query Routing**: Intelligent query routing across multiple domains
- **Hybrid Retrieval**: Combines vector similarity with graph traversal

### Technology Stack

- **Backend**: FastAPI (Python 3.12)
- **Database**: PostgreSQL with async SQLAlchemy
- **LLM Integration**: OpenAI-compatible API gateway (localhost:8087/v1)
- **Embeddings**: Configurable embedding provider
- **Testing**: Pytest with async support

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KBV2 Knowledge Base                       │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                            │
│  ├── Query API     ├── Review API     ├── Graph API             │
│  ├── Document API  └── MCP Server (WebSocket)                   │
├─────────────────────────────────────────────────────────────────┤
│  Intelligence Layer (LLM-Powered Services)                      │
│  ├── Entity Extraction Manager (Multi-Agent)                    │
│  ├── Entity Typing Service                                      │
│  ├── Hallucination Detector (LLM-as-Judge)                      │
│  ├── Hybrid Retriever (Vector + Graph)                          │
│  ├── Domain Schema Registry                                     │
│  ├── Cross-Domain Detector                                      │
│  └── Federated Query Router                                     │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion Layer                                                │
│  ├── Partitioning Service    ├── Gleaning Service               │
│  └── Embedding Client                                            │
├─────────────────────────────────────────────────────────────────┤
│  Persistence Layer                                              │
│  ├── Vector Store           ├── Graph Store                     │
│  └── Schema (SQLAlchemy)                                        │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure                                                 │
│  ├── LLM Gateway (localhost:8087/v1)   ├── PostgreSQL           │
│  └── Observability (Logfire)                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
/home/muham/development/kbv2/
├── src/knowledge_base/
│   ├── main.py                    # FastAPI application entry point
│   ├── clients/
│   │   └── llm_client.py          # LLM client for API gateway
│   ├── common/
│   │   ├── api_models.py          # Pydantic models for API
│   │   ├── dependencies.py        # Dependency injection
│   │   ├── error_handlers.py      # Error handling middleware
│   │   ├── gateway.py             # Resilient gateway client
│   │   └── pagination.py          # Pagination utilities
│   ├── intelligence/
│   │   └── v1/
│   │       ├── entity_typing_service.py      # Entity classification
│   │       ├── multi_agent_extractor.py      # GraphMaster-style
│   │       ├── hallucination_detector.py     # LLM-as-Judge
│   │       ├── hybrid_retriever.py           # Vector + Graph
│   │       ├── domain_schema_service.py      # Domain schemas
│   │       ├── cross_domain_detector.py      # Cross-domain links
│   │       └── federated_query_router.py     # Query routing
│   ├── ingestion/
│   │   └── v1/
│   │       ├── embedding_client.py   # Embedding generation
│   │       ├── gleaning_service.py   # Information extraction
│   │       └── partitioning_service.py # Document chunking
│   ├── persistence/
│   │   └── v1/
│   │       ├── vector_store.py       # Vector similarity search
│   │       ├── graph_store.py        # Knowledge graph storage
│   │       └── schema.py             # SQLAlchemy models
│   ├── document_api.py              # Document management API
│   ├── graph_api.py                 # Graph operations API
│   ├── query_api.py                 # Query processing API
│   ├── review_api.py                # Human review workflow API
│   ├── orchestrator.py              # Main orchestration logic
│   └── mcp_server.py                # MCP protocol server
├── tests/
│   ├── unit/                        # Unit tests
│   └── integration/                 # Integration tests
├── docs/
│   └── overview/                    # Documentation
├── plan.md                          # Research plan
└── .env                             # Configuration
```

---

## Core Components

### 1. FastAPI Application (main.py)

The main FastAPI application provides RESTful APIs and WebSocket connections.

**Key Features:**
- Request ID tracking for observability
- CORS middleware
- AIP-193 compliant error responses
- WebSocket support for real-time updates
- Health and readiness endpoints

**Entry Point:**
```python
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8765,
    reload=True
)
```

### 2. Database Schema

The system uses PostgreSQL with the following core models:

#### Document
```python
class Document(BaseModel):
    id: UUID
    title: str
    content: str
    status: DocumentStatus
    metadata: JSON
    chunks: List[DocumentChunk]
```

#### Entity
```python
class Entity(BaseModel):
    id: UUID
    name: str
    entity_type: str
    description: Optional[str]
    properties: Dict[str, Any]
    domain: str
    embedding: Optional[List[float]]
```

#### Relationship
```python
class Relationship(BaseModel):
    id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float
```

---

## New LLM-Powered Services (2025-2026)

### 1. LLM Client (llm_client.py)

**Purpose:** Unified interface to LLM API gateway

**Location:** `src/knowledge_base/clients/llm_client.py`

**Key Features:**
- OpenAI-compatible API format
- Sync and async methods
- Multiple prompting strategies
- Structured JSON output
- Retry logic with exponential backoff

**Configuration:**
```python
LLM_GATEWAY_URL=http://localhost:8087/v1/
LLM_API_KEY=dev_api_key
LLM_MODEL=gemini-2.5-flash-lite
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

**Usage Examples:**

```python
from knowledge_base.clients import create_llm_client

# Initialize
client = create_llm_client()

# Standard completion
response = await client.complete(
    prompt="Classify this entity: {entity}",
    strategy="standard"
)

# Few-shot prompting
response = await client.complete(
    prompt="Classify: Apple is a",
    strategy="few_shot",
    examples=[
        FewShotExample(
            input="Google is a",
            output="ORGANIZATION"
        )
    ]
)

# Chain-of-Thought reasoning
response, steps = await client.complete_with_cot_steps(
    "Identify all entities in this text with reasoning"
)

# Chain-of-Draft (token-efficient)
response, steps = await client.complete_with_cod_steps(
    "Extract entities concisely"
)

# Structured JSON output
response = await client.complete_json(
    prompt="Extract entities as JSON",
    schema=EntitySchema
)
```

**Supported Strategies:**
- `STANDARD` - Basic completion
- `FEW_SHOT` - With examples
- `CHAIN_OF_THOUGHT` - With reasoning steps
- `CHAIN_OF_DRAFT` - Token-efficient reasoning
- `JSON` - Structured output

### 2. Entity Typing Service (entity_typing_service.py)

**Purpose:** Classify entities into semantic categories using LLM

**Location:** `src/knowledge_base/intelligence/v1/entity_typing_service.py`

**Entity Types:**
```python
class EntityType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"
```

**Domain Types:**
```python
class DomainType(Enum):
    GENERAL = "GENERAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    FINANCIAL = "FINANCIAL"
    TECHNOLOGY = "TECHNOLOGY"
    ACADEMIC = "ACADEMIC"
    SCIENTIFIC = "SCIENTIFIC"
    GOVERNMENT = "GOVERNMENT"
```

**Core Classes:**

```python
class EntityTyper:
    """Main entity typing service"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[EntityTypingConfig] = None
    ):
        self.llm = llm_client
        self.config = config or EntityTypingConfig()
        self.taxonomy = self._build_taxonomy()
        self.prompt_registry = PromptTemplateRegistry()
        self.example_bank = FewShotExampleBank()
    
    async def type_entities(
        self,
        text: str,
        entities: List[str],
        domain: DomainType = DomainType.GENERAL
    ) -> EntityTypingResult:
        """Type a batch of entities from text"""
        
    async def type_single_entity(
        self,
        text: str,
        entity: str,
        domain: DomainType
    ) -> TypedEntity:
        """Type a single entity"""
```

**Usage:**

```python
from knowledge_base.intelligence import EntityTyper, EntityType, DomainType

typer = EntityTyper()

# Type entities
result = await typer.type_entities(
    text="Apple Inc. was founded by Steve Jobs in Los Altos, California.",
    entities=["Apple Inc.", "Steve Jobs", "Los Altos", "California"],
    domain=DomainType.TECHNOLOGY
)

# Access results
for entity in result.typed_entities:
    print(f"{entity.text}: {entity.entity_type}")
    print(f"  Confidence: {entity.confidence}")
    print(f"  Needs Review: {entity.human_review_required}")
```

**Prompt Templates:**

The service uses domain-specific prompt templates:

```python
# Default template
template = PromptTemplate(
    name="default_entity_typing",
    system_prompt="You are an expert entity classifier.",
    user_template="""
    Classify the following entity from the text.

    Text: {text}
    Entity: {entity}

    Entity Types: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, OTHER

    Return JSON:
    {
        "entity": "...",
        "type": "...",
        "confidence": 0.0-1.0,
        "reasoning": "..."
    }
    """
)
```

### 3. Multi-Agent Entity Extractor (multi_agent_extractor.py)

**Purpose:** GraphMaster-style multi-agent orchestration for complex entity extraction

**Location:** `src/knowledge_base/intelligence/v1/multi_agent_extractor.py`

**Based on Research:**
- GraphMaster (arXiv:2504.00711) - Multi-agent LLM orchestration
- BANER (COLING 2025) - Boundary-aware entity recognition
- GPT-NER (NAACL 2025) - Sequence-to-generation approach

**Agent Architecture:**

```python
class EntityExtractionManager:
    """Main orchestrator for multi-agent extraction"""
    
    def __init__(self, config: MultiAgentConfig = None):
        self.config = config or MultiAgentConfig()
        self.manager_agent = ManagerAgent(self.config)
        self.perception_agent = PerceptionAgent(self.config)
        self.enhancement_agent = EnhancementAgent(self.config)
        self.evaluation_agent = EvaluationAgent(self.config)
    
    async def extract_entities(
        self,
        text: str,
        domain: str,
        entity_types: Optional[List[str]] = None
    ) -> ExtractionResult:
        """Execute full extraction workflow"""
```

**Agent Details:**

```python
class ManagerAgent:
    """Coordinates extraction workflow across all phases"""
    
    async def create_plan(
        self,
        text: str,
        domain: str
    ) -> ExtractionPlan:
        """Analyze text and create extraction plan"""
        
    async def execute_phase(
        self,
        phase: ExtractionPhase,
        context: Dict
    ) -> Dict:
        """Execute a specific extraction phase"""
```

```python
class PerceptionAgent:
    """BANER-style boundary-aware entity extraction"""
    
    async def extract(
        self,
        text: str,
        domain: str,
        entity_types: Optional[List[str]] = None
    ) -> List[EntityCandidate]:
        """
        Extract entity candidates with boundary awareness.
        Handles overlapping and nested entities.
        """
        
    async def classify_boundary(
        self,
        boundary: str
    ) -> EntityBoundaryType:
        """Classify entity boundary type"""
```

```python
class EnhancementAgent:
    """Refine and link entities using KG context"""
    
    async def enhance(
        self,
        entities: List[EntityCandidate],
        context: EnhancementContext
    ) -> List[ExtractedEntity]:
        """
        1. Cross-reference with existing knowledge graph
        2. Resolve entity linking (same entity mention)
        3. Add domain-specific attributes
        4. Infer implicit relationships
        """
```

```python
class EvaluationAgent:
    """LLM-as-Judge quality validation"""
    
    async def evaluate(
        self,
        entities: List[ExtractedEntity]
    ) -> ExtractionQualityScore:
        """
        Assess extraction quality using LLM-as-Judge:
        - Entity accuracy
        - Type correctness
        - Completeness
        - Coherence
        """
```

**Extraction Phases:**

```python
class ExtractionPhase(Enum):
    PERCEPTION = "perception"      # Initial extraction
    ENHANCEMENT = "enhancement"    # Refinement
    EVALUATION = "evaluation"      # Quality check
    COMPLETED = "completed"
```

**Usage:**

```python
from knowledge_base.intelligence import EntityExtractionManager

manager = EntityExtractionManager()

# Extract entities
result = await manager.extract_entities(
    text="""
    Tesla, Inc. was founded by Elon Musk and is headquartered
    in California. The company manufactures electric vehicles
    and solar panels.
    """,
    domain="TECHNOLOGY",
    entity_types=["ORGANIZATION", "PERSON", "LOCATION", "PRODUCT"]
)

# Access results
print(f"Quality Score: {result.quality_score.overall_score}")
print(f"Quality Level: {result.quality_score.quality_level}")
print(f"Entities Extracted: {len(result.entities)}")

for entity in result.entities:
    print(f"  {entity.text} ({entity.entity_type})")
    print(f"    Boundary: {entity.boundary_type}")
    print(f"    Phase: {entity.phase}")
```

**Quality Score:**

```python
class ExtractionQualityScore:
    overall_score: float           # 0.0 - 1.0
    entity_accuracy: float
    type_correctness: float
    completeness: float
    coherence: float
    quality_level: QualityLevel    # EXCELLENT, GOOD, ACCEPTABLE, NEEDS_REVIEW
    
class QualityLevel(Enum):
    EXCELLENT = "excellent"        # score >= 0.9
    GOOD = "good"                  # score >= 0.75
    ACCEPTABLE = "acceptable"      # score >= 0.6
    NEEDS_REVIEW = "needs_review"  # score < 0.6
```

### 4. Hallucination Detector (hallucination_detector.py)

**Purpose:** LLM-as-Judge verification layer to detect fabricated entity attributes

**Location:** `src/knowledge_base/intelligence/v1/hallucination_detector.py`

**Based on Research:**
- LLM-as-Judge for KG Quality (arXiv:2411.17388)

**Core Classes:**

```python
class HallucinationDetector:
    """LLM-as-Judge verification for entity quality"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[HallucinationDetectorConfig] = None
    ):
        self.llm = llm_client
        self.config = config or HallucinationDetectorConfig()
    
    async def verify_entity(
        self,
        entity: Entity,
        context: str,
        source_text: str
    ) -> EntityVerification:
        """Verify a single entity for hallucinations"""
        
    async def verify_entity_batch(
        self,
        entities: List[Entity],
        context: str,
        source_text: str,
        batch_size: int = 10
    ) -> BatchVerificationResult:
        """Verify multiple entities efficiently"""
```

**Verification Models:**

```python
class AttributeVerification:
    attribute_name: str
    attribute_value: Any
    is_supported: bool
    confidence: float
    evidence: Optional[str]

class EntityVerification:
    entity: Entity
    attributes: List[AttributeVerification]
    overall_confidence: float
    is_hallucinated: bool
    risk_level: RiskLevel
    supported_count: int
    unsupported_count: int
    unsupported_attributes: List[str]

class RiskLevel(Enum):
    LOW = "low"          # < 20% unsupported attributes
    MEDIUM = "medium"    # 20-40% unsupported
    HIGH = "high"        # 40-60% unsupported
    CRITICAL = "critical" # > 60% unsupported
```

**Usage:**

```python
from knowledge_base.intelligence import HallucinationDetector

detector = HallucinationDetector()

# Single entity verification
verification = await detector.verify_entity(
    entity=extracted_entity,
    context="Previous sentences about the company...",
    source_text="Original document text..."
)

print(f"Entity: {verification.entity.name}")
print(f"Hallucinated: {verification.is_hallucinated}")
print(f"Risk Level: {verification.risk_level.value}")
print(f"Confidence: {verification.overall_confidence}")

for attr in verification.attributes:
    status = "✓" if attr.is_supported else "✗"
    print(f"  {status} {attr.attribute_name}: {attr.confidence:.2%}")

# Batch verification
batch_result = await detector.verify_entity_batch(
    entities=[e1, e2, e3, e4, e5],
    context="Document context...",
    source_text="Full source text...",
    batch_size=5
)

print(f"\nBatch Summary:")
print(f"  Total: {batch_result.total_entities}")
print(f"  Hallucinated: {batch_result.hallucinated_count}")
print(f"  Hallucination Rate: {batch_result.hallucination_rate:.2%}")
print(f"  Overall Risk: {batch_result.overall_risk_level.value}")
```

### 5. Hybrid Retriever (hybrid_retriever.py)

**Purpose:** Combines vector similarity search with graph traversal

**Location:** `src/knowledge_base/intelligence/v1/hybrid_retriever.py`

**Core Classes:**

```python
class HybridEntityRetriever:
    """Retrieves entities using combined vector + graph approach"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        min_confidence: float = 0.5
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.min_confidence = min_confidence
    
    async def retrieve_context(
        self,
        query: str,
        entity_candidates: List[EntityCandidate],
        domain: Optional[str] = None
    ) -> RetrievalContext:
        """Retrieve context using hybrid approach"""
```

**Retrieval Process:**

1. **Vector Search**: Similarity search on entity embeddings
2. **Graph Expansion**: Traverse knowledge graph relationships
3. **Weighted Fusion**: Combine scores from both sources
4. **Ranking**: Sort by final combined score

**Usage:**

```python
from knowledge_base.intelligence import HybridEntityRetriever

retriever = HybridEntityRetriever(
    vector_store=vector_store,
    graph_store=graph_store,
    vector_weight=0.6,
    graph_weight=0.4
)

# Retrieve context
result = await retriever.retrieve_context(
    query="artificial intelligence companies",
    entity_candidates=[candidate1, candidate2],
    domain="TECHNOLOGY"
)

print(f"Total Results: {len(result.entities)}")
print(f"Vector-only: {len([e for e in result.entities if e.vector_score > 0])}")
print(f"Graph-only: {len([e for e in result.entities if e.graph_score > 0])}")
print(f"Hybrid: {len([e for e in result.entities if e.vector_score > 0 and e.graph_score > 0])}")

for entity in result.entities:
    print(f"\n{entity.text}")
    print(f"  Vector Score: {entity.vector_score:.3f}")
    print(f"  Graph Score: {entity.graph_score:.3f}")
    print(f"  Final Score: {entity.final_score:.3f}")
    print(f"  Sources: {entity.sources}")
```

### 6. Domain Schema Service (domain_schema_service.py)

**Purpose:** Manages domain-specific entity schemas with inheritance

**Location:** `src/knowledge_base/intelligence/v1/domain_schema_service.py`

**Key Concepts:**

```python
class DomainLevel(Enum):
    ROOT = "root"           # Base domain (GENERAL)
    PRIMARY = "primary"     # Top-level domain
    SECONDARY = "secondary" # Subdomain
    TERTIARY = "tertiary"   # Deep subdomain

class InheritanceType(Enum):
    EXTENDS = "extends"     # Add to parent attributes
    OVERRIDES = "overrides" # Replace parent attributes
    COMPOSES = "composes"   # Combine with parent
```

**Core Classes:**

```python
class SchemaRegistry:
    """Registry for managing domain schemas"""
    
    async def register_schema(
        self,
        name: str,
        entity_types: List[EntityTypeDef],
        parent_domain: Optional[str] = None,
        inheritance_type: InheritanceType = InheritanceType.EXTENDS
    ) -> DomainSchema:
        """Register a new domain schema"""
        
    async def get_schema(self, name: str) -> Optional[DomainSchema]:
        """Get schema by name"""
        
    async def get_with_inheritance(self, name: str) -> DomainSchema:
        """Get schema with parent attributes applied"""
```

**Usage:**

```python
from knowledge_base.intelligence import SchemaRegistry, EntityTypeDef, InheritanceType

registry = SchemaRegistry()

# Register parent domain (ROOT)
await registry.register_schema(
    name="GENERAL",
    entity_types=[
        EntityTypeDef(
            name="NamedEntity",
            base_type="OTHER",
            domain_specific_attrs={
                "name": "str",
                "description": "Optional[str]"
            }
        )
    ]
)

# Register child domain (inherits from GENERAL)
await registry.register_schema(
    name="TECHNOLOGY",
    entity_types=[
        EntityTypeDef(
            name="Startup",
            base_type="ORGANIZATION",
            domain_specific_attrs={
                "funding_stage": "str",
                "valuation": "float",
                "founded_year": "int"
            },
            parent="NamedEntity"
        ),
        EntityTypeDef(
            name="TechProduct",
            base_type="PRODUCT",
            domain_specific_attrs={
                "version": "str",
                "release_date": "datetime",
                "platform": "List[str]"
            }
        )
    ],
    parent_domain="GENERAL",
    inheritance_type=InheritanceType.EXTENDS
)

# Get schema with inheritance
schema = await registry.get_with_inheritance("TECHNOLOGY")

print(f"Domain: {schema.name}")
print(f"Level: {schema.level.value}")
print(f"Inherited Types: {len(schema.inherited_entity_types)}")

for entity_type in schema.inherited_entity_types:
    print(f"\n{entity_type.name} (base: {entity_type.base_type})")
    for attr_name, attr_type in entity_type.domain_specific_attrs.items():
        print(f"  {attr_name}: {attr_type}")
```

### 7. Cross-Domain Detector (cross_domain_detector.py)

**Purpose:** Detects relationships between entities across different domains

**Location:** `src/knowledge_base/intelligence/v1/cross_domain_detector.py`

**Relationship Types:**

```python
class RelationshipType(Enum):
    # Hierarchical
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    SUBSET_OF = "SUBSET_OF"
    SUPERSET_OF = "SUPERSET_OF"
    
    # Ownership
    OWNS = "OWNS"
    OWNED_BY = "OWNED_BY"
    ACQUIRED = "ACQUIRED"
    
    # Location
    LOCATED_IN = "LOCATED_IN"
    HEADQUARTERED_IN = "HEADQUARTERED_IN"
    OPERATES_IN = "OPERATES_IN"
    
    # Employment
    WORKS_FOR = "WORKS_FOR"
    EMPLOYEES = "EMPLOYEES"
    FOUNDED_BY = "FOUNDED_BY"
    CO_FOUNDED_BY = "CO_FOUNDED_BY"
    
    # Product/Service
    PRODUCES = "PRODUCES"
    USES = "USES"
    COMPETES_WITH = "COMPETES_WITH"
    
    # Financial
    INVESTS_IN = "INVESTS_IN"
    FUNDED_BY = "FUNDED_BY"
    MERGED_WITH = "MERGED_WITH"
    
    # Temporal
    PRECEDES = "PRECEDES"
    FOLLOWS = "FOLLOWS"
    SIMULTANEOUS_WITH = "SIMULTANEOUS_WITH"
    
    # Generic
    RELATED_TO = "RELATED_TO"
    CONNECTED_TO = "CONNECTED_TO"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    DEPENDS_ON = "DEPENDS_ON"
    ENABLES = "ENABLES"
```

**Domain Types:**

```python
class DomainType(Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    CONCEPT = "CONCEPT"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    FINANCIAL = "FINANCIAL"
    TECHNOLOGY = "TECHNOLOGY"
    ACADEMIC = "ACADEMIC"
    GOVERNMENT = "GOVERNMENT"
```

**Core Classes:**

```python
class CrossDomainDetector:
    """Detects relationships across domain boundaries"""
    
    def __init__(self, config: CrossDomainDetectorConfig = None):
        self.config = config or CrossDomainDetectorConfig()
        self.taxonomy = DomainPairTaxonomy()
        self.patterns: Dict[str, List[RelationshipPattern]] = {}
    
    async def detect_relationships(
        self,
        entities: List[Entity],
        min_confidence: float = 0.7
    ) -> CrossDomainRelationshipResult:
        """Detect all cross-domain relationships"""
        
    async def detect_entity_relationships(
        self,
        source_entity: Entity,
        target_entities: List[Entity],
        domain: str
    ) -> List[CrossDomainRelationship]:
        """Detect relationships from source to targets"""
```

**Usage:**

```python
from knowledge_base.intelligence import CrossDomainDetector

detector = CrossDomainDetector()

# Entities from different domains
entities = [
    Entity(name="Elon Musk", domain="PERSON"),
    Entity(name="Tesla", domain="ORGANIZATION"),
    Entity(name="California", domain="LOCATION"),
    Entity(name="SpaceX", domain="ORGANIZATION")
]

# Detect cross-domain relationships
result = await detector.detect_relationships(
    entities=entities,
    min_confidence=0.7
)

print(f"Total Relationships: {result.total_relationships}")
print(f"Cross-Domain: {result.cross_domain_count}")
print(f"Same-Domain: {result.same_domain_count}")

for rel in result.cross_domain:
    print(f"\n{rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
    print(f"  Source Domain: {rel.source_domain}")
    print(f"  Target Domain: {rel.target_domain}")
    print(f"  Confidence: {rel.confidence:.2%}")
    print(f"  Bidirectional: {rel.bidirectional}")
    print(f"  Evidence: {rel.evidence}")

# Statistics
stats = detector.get_statistics(result.relationships)
print(f"\nStatistics:")
print(f"  Most Common Type: {stats.most_common_type}")
print(f"  Avg Confidence: {stats.avg_confidence:.2%}")
print(f"  Bidirectional Ratio: {stats.bidirectional_ratio:.2%}")
```

### 8. Federated Query Router (federated_query_router.py)

**Purpose:** Routes queries across multiple knowledge domains

**Location:** `src/knowledge_base/intelligence/v1/federated_query_router.py`

**Query Domains:**

```python
class QueryDomain(Enum):
    GENERAL = "GENERAL"
    TECHNICAL = "TECHNICAL"
    BUSINESS = "BUSINESS"
    DOCUMENTATION = "DOCUMENTATION"
    RESEARCH = "RESEARCH"
    ANALYTICS = "ANALYTICS"
```

**Execution Strategies:**

```python
class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"    # Execute domains one at a time
    PARALLEL = "parallel"        # Execute all domains simultaneously
    PRIORITY = "priority"        # Execute by domain priority
```

**Core Classes:**

```python
class FederatedQueryRouter:
    """Routes queries across multiple domains"""
    
    def __init__(
        self,
        retriever: Optional[HybridEntityRetriever] = None,
        config: FederatedQueryRouterConfig = None
    ):
        self.retriever = retriever
        self.config = config or FederatedQueryRouterConfig()
        self.domain_detector = DomainDetector()
        self.subquery_builder = SubQueryBuilder()
        self.result_aggregator = ResultAggregator()
    
    async def create_plan(
        self,
        query: str,
        max_domains: int = 3,
        strategy: ExecutionStrategy = ExecutionStrategy.PRIORITY
    ) -> FederatedQueryPlan:
        """Create query execution plan"""
        
    async def execute_plan(
        self,
        plan: FederatedQueryPlan,
        retriever: Optional[HybridEntityRetriever] = None
    ) -> FederatedQueryResult:
        """Execute query plan"""
```

**Usage:**

```python
from knowledge_base.intelligence import FederatedQueryRouter

router = FederatedQueryRouter()

# Create execution plan
plan = await router.create_plan(
    query="How does machine learning affect financial trading algorithms?",
    max_domains=3,
    strategy=ExecutionStrategy.PARALLEL
)

print(f"Query: {plan.original_query}")
print(f"Detected Domains:")
for domain_det in plan.detected_domains:
    print(f"  {domain_det.domain.value} (confidence: {domain_det.confidence:.2%})")

print(f"Strategy: {plan.strategy.value}")
print(f"Sub-queries: {len(plan.sub_queries)}")

for subq in plan.sub_queries:
    print(f"\n{subq.domain.value}: {subq.query}")

# Execute plan
result = await router.execute_plan(
    plan=plan,
    retriever=hybrid_retriever
)

print(f"\nResults:")
print(f"  Total Entities: {result.total_entities}")
print(f"  Domain Results: {len(result.domain_results)}")

for domain_result in result.domain_results:
    print(f"\n  {domain_result.domain.value}:")
    print(f"    Entities: {len(domain_result.entities)}")
    print(f"    Confidence: {domain_result.avg_confidence:.2%}")
    
    for entity in domain_result.entities[:3]:
        print(f"    - {entity.text} ({entity.entity_type})")

# Get combined results
combined = result.get_combined_results(merge_strategy="confidence_weighted")
print(f"\nCombined Results: {len(combined)} entities")
```

---

## Data Flow and Pipelines

### End-to-End Entity Extraction Pipeline

```
1. Document Ingestion
   └── Raw text → Partitioning → Chunking → Embedding

2. Multi-Agent Entity Extraction
   ├── ManagerAgent: Create extraction plan
   ├── PerceptionAgent: Extract candidate entities (BANER-style)
   ├── EnhancementAgent: Link with existing KG
   └── EvaluationAgent: Quality assessment (LLM-as-Judge)

3. Entity Typing
   └── LLM-based classification with few-shot prompting

4. Hallucination Detection
   └── Verify attributes against source text

5. Cross-Domain Linking
   └── Detect relationships across domains

6. Knowledge Graph Update
   └── Store entities and relationships

7. Query Processing
   └── Hybrid retrieval + federated routing
```

### Detailed Flow

```python
async def process_document(document: Document) -> ProcessingResult:
    """
    Complete document processing pipeline:
    1. Partition document into chunks
    2. Extract entities using multi-agent system
    3. Classify entity types
    4. Verify for hallucinations
    5. Detect cross-domain relationships
    6. Update knowledge graph
    7. Index for retrieval
    """
    
    # Step 1: Partition
    chunks = await partitioning_service.partition(document.content)
    
    # Step 2: Extract entities
    extractor = EntityExtractionManager()
    extraction = await extractor.extract_entities(
        text=document.content,
        domain=document.domain
    )
    
    # Step 3: Type entities
    typer = EntityTyper()
    typed = await typer.type_entities(
        text=document.content,
        entities=[e.text for e in extraction.entities],
        domain=document.domain
    )
    
    # Step 4: Verify for hallucinations
    detector = HallucinationDetector()
    verified = await detector.verify_entity_batch(
        entities=extraction.entities,
        context=document.content,
        source_text=document.content
    )
    
    # Step 5: Detect cross-domain relationships
    cross_domain = CrossDomainDetector()
    relationships = await cross_domain.detect_relationships(
        entities=extraction.entities
    )
    
    # Step 6: Update knowledge graph
    for entity in verified_verified_entities:
        await graph_store.upsert_entity(entity)
    
    for rel in relationships:
        await graph_store.add_relationship(rel)
    
    # Step 7: Index for retrieval
    for entity in extraction.entities:
        embedding = await embedding_client.get_embedding(entity.text)
        await vector_store.upsert(entity.id, embedding)
    
    return ProcessingResult(
        entities=extraction.entities,
        relationships=relationships,
        quality_score=extraction.quality_score
    )
```

---

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check |
| `/api/v1/documents` | GET/POST | Document management |
| `/api/v1/documents/{id}` | GET/PUT/DELETE | Single document |
| `/api/v1/documents/{id}/search` | POST | Vector search |
| `/api/v1/graphs` | GET | Graph operations |
| `/api/v1/graphs/{id}/neighborhood` | GET | Entity neighborhood |
| `/api/v1/graphs/path` | POST | Find path between entities |
| `/api/v1/query` | POST | Natural language query |
| `/api/v1/review` | GET | Human review queue |
| `/api/v1/review/{id}` | PUT | Approve/reject review |
| `/ws` | WebSocket | Real-time updates |

### WebSocket Protocol

```python
# Client sends:
{
    "action": "process",
    "document_id": "uuid",
    "options": {
        "extract_entities": True,
        "detect_relationships": True,
        "verify_hallucinations": True
    }
}

# Server responds:
{
    "action": "progress",
    "stage": "extraction",
    "progress": 0.5,
    "entities_found": 25
}

{
    "action": "complete",
    "result": {
        "entities": [...],
        "relationships": [...],
        "quality_score": 0.85
    }
}
```

---

## Configuration and Environment

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://agentzero@localhost:5432/knowledge_base
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_base
DB_USER=agentzero
DB_PASSWORD=dev_password

# LLM Gateway
LLM_GATEWAY_URL=http://localhost:8087/v1/
LLM_API_KEY=dev_api_key
LLM_MODEL=gemini-2.5-flash-lite
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# Embeddings
GOOGLE_API_KEY=AIzaSy...
GOOGLE_EMBEDDING_URL=https://generativelanguage.googleapis.com
GOOGLE_EMBEDDING_MODEL=embedding-001

# Observability
LOGFIRE_PROJECT=knowledge-base
LOGFIRE_SEND_TO_LOGFIRE=false

# Ingestion
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MAX_DENSITY_THRESHOLD=0.8
MIN_DENSITY_THRESHOLD=0.3

# Clustering
LEIDEN_RESOLUTION_MACRO=0.8
LEIDEN_RESOLUTION_MICRO=1.2
LEIDEN_ITERATIONS=10

# Resolution
RESOLUTION_CONFIDENCE_THRESHOLD=0.7
RESOLUTION_SIMILARITY_THRESHOLD=0.85

# HNSW Index
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=100
```

### Configuration Classes

```python
# LLM Client Configuration
@dataclass
class LLMClientConfig:
    base_url: str = "http://localhost:8087/v1/"
    api_key: str = "dev_api_key"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    max_retries: int = 3

# Entity Typing Configuration
@dataclass
class EntityTypingConfig:
    min_confidence: float = 0.6
    max_alternatives: int = 3
    require_reasoning: bool = False
    enable_human_review: bool = True
    review_threshold: float = 0.7

# Multi-Agent Configuration
@dataclass
class MultiAgentConfig:
    enable_parallel_extraction: bool = True
    max_concurrent_agents: int = 4
    quality_threshold: float = 0.6
    enable_evaluation: bool = True
```

---

## Testing Framework

### Test Structure

```
tests/
├── unit/
│   ├── test_llm_client.py           # 34 tests
│   ├── test_entity_typing_service.py # 35 tests
│   ├── test_multi_agent_extractor.py # 38 tests
│   ├── test_hallucination_detector.py # 26 tests
│   ├── test_hybrid_retriever.py      # 21 tests
│   ├── test_domain_schema_service.py # 32 tests
│   ├── test_federated_query_router.py # 36 tests
│   ├── test_cross_domain_detector.py # 53 tests
│   └── test_api/                      # API tests
├── integration/
│   └── test_real_world_pipeline.py   # E2E tests
└── conftest.py                        # Pytest fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_entity_typing_service.py -v

# Run with coverage
pytest --cov=src/knowledge_base/intelligence/v1/

# Run integration tests
pytest tests/integration/ -v

# Run a specific test
pytest tests/unit/test_entity_typing_service.py::TestEntityTyper::test_type_single_entity -v
```

### Test Fixtures

```python
@pytest.fixture
async def llm_client():
    """Create LLM client for testing"""
    return create_llm_client()

@pytest.fixture
async def sample_entities():
    """Sample entities for testing"""
    return [
        Entity(name="Apple", domain="ORGANIZATION"),
        Entity(name="Tim Cook", domain="PERSON"),
        Entity(name="California", domain="LOCATION")
    ]

@pytest.fixture
async def sample_text():
    """Sample text for entity extraction"""
    return """
    Apple Inc. was founded by Steve Jobs and Steve Wozniak
    in Cupertino, California. The company is known for the
    iPhone, iPad, and Mac computers.
    """
```

---

## Research Foundation

### Papers Implemented

| Paper | Venue | Implementation |
|-------|-------|----------------|
| BANER: Boundary-Aware LLMs for Few-Shot Named Entity Recognition | COLING 2025 | PerceptionAgent boundary detection |
| GPT-NER: Transforming Named Entity Recognition via Generative Pretraining | NAACL 2025 | Entity typing pipeline |
| GraphMaster: Multi-Agent LLM Orchestration for KG Synthesis | arXiv:2504.00711 | Multi-agent orchestration |
| LLM-as-Judge for KG Quality | arXiv:2411.17388 | Hallucination detection layer |

### Key Research Directions (2025-2026)

1. **LLM-Based Entity Linking**: Zero-shot entity linking using contextual augmentation
2. **Boundary-Aware NER**: Handling ambiguous and overlapping entity boundaries
3. **Multi-Agent Orchestration**: Specialized agents for different extraction phases
4. **Hallucination Mitigation**: Verification layers for LLM-generated content
5. **Hybrid Retrieval**: Combining vector and graph-based approaches
6. **Chain-of-Draft**: Token-efficient reasoning for extraction tasks

---

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd /home/muham/development/kbv2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Environment Setup

```bash
# Copy environment file
cp .env.example .env

# Configure environment (edit .env)
# Ensure LLM_GATEWAY_URL points to your LLM API
```

### 3. Start Services

```bash
# Start PostgreSQL
# Start LLM Gateway (localhost:8087/v1)

# Start KBV2 API
python -m knowledge_base.main
# Or: uvicorn knowledge_base.main:app --host 0.0.0.0 --port 8765
```

### 4. Quick Example

```python
import asyncio
from knowledge_base.intelligence import (
    EntityExtractionManager,
    HallucinationDetector
)

async def main():
    # Initialize services
    extractor = EntityExtractionManager()
    detector = HallucinationDetector()
    
    # Process text
    text = """
    OpenAI was founded by Sam Altman, Greg Brockman, and Ilya Sutskever.
    The company is headquartered in San Francisco and created ChatGPT.
    """
    
    # Extract entities
    extraction = await extractor.extract_entities(
        text=text,
        domain="TECHNOLOGY"
    )
    
    print(f"Extracted {len(extraction.entities)} entities")
    print(f"Quality Score: {extraction.quality_score.overall_score:.2%}")
    
    # Verify for hallucinations
    verification = await detector.verify_entity_batch(
        entities=extraction.entities,
        context=text,
        source_text=text
    )
    
    print(f"Hallucination Rate: {verification.hallucination_rate:.2%}")
    
    # Return only verified entities
    verified = [
        e for e, v in zip(extraction.entities, verification.verifications)
        if v.risk_level.value in ["LOW", "MEDIUM"]
    ]
    
    return verified

if __name__ == "__main__":
    result = asyncio.run(main())
```

---

## Advanced Usage

### Custom Prompt Templates

```python
from knowledge_base.intelligence import PromptTemplate, PromptTemplateRegistry

custom_template = PromptTemplate(
    name="legal_entity_typing",
    system_prompt="You are a legal document expert.",
    user_template="""
    Extract and classify legal entities from this document.

    Document: {text}
    Entities to classify: {entities}

    Legal Entity Types: PARTY, COURT, JUDGE, ATTORNEY, LAW, CONTRACT, CASE

    Return JSON array:
    [
        {"entity": "...", "type": "...", "confidence": 0.0-1.0}
    ]
    """
)

registry = PromptTemplateRegistry()
registry.register(custom_template, domain="LEGAL")

# Use custom template
typer = EntityTyper()
result = await typer.type_entities(
    text="...",
    entities=["Court of Appeals", "Judge Smith"],
    domain="LEGAL"
)
```

### Custom Domain Schemas

```python
from knowledge_base.intelligence import (
    SchemaRegistry,
    EntityTypeDef,
    DomainAttribute,
    InheritanceType
)

registry = SchemaRegistry()

# Register financial domain schema
await registry.register_schema(
    name="FINANCIAL",
    entity_types=[
        EntityTypeDef(
            name="PublicCompany",
            base_type="ORGANIZATION",
            domain_specific_attrs={
                "ticker_symbol": DomainAttribute(
                    name="ticker_symbol",
                    type="str",
                    required=True,
                    validation={"pattern": "^[A-Z]{1,5}$"}
                ),
                "market_cap": DomainAttribute(
                    name="market_cap",
                    type="float",
                    required=False,
                    unit="USD"
                ),
                "stock_exchange": DomainAttribute(
                    name="stock_exchange",
                    type="str",
                    required=True,
                    validation={
                        "enum": ["NYSE", "NASDAQ", "LSE", "TSE"]
                    }
                )
            },
            parent="Organization"
        ),
        EntityTypeDef(
            name="FinancialInstrument",
            base_type="PRODUCT",
            domain_specific_attrs={
                "isin": DomainAttribute(
                    name="isin",
                    type="str",
                    required=True,
                    validation={"pattern": "^[A-Z]{2}[A-Z0-9]{9}[0-9]$"}
                ),
                "currency": DomainAttribute(
                    name="currency",
                    type="str",
                    required=True
                )
            }
        )
    ],
    parent_domain="GENERAL",
    inheritance_type=InheritanceType.EXTENDS
)
```

### Custom Relationship Patterns

```python
from knowledge_base.intelligence import (
    CrossDomainDetector,
    RelationshipPattern
)

detector = CrossDomainDetector()

# Register custom relationship pattern
pattern = RelationshipPattern(
    name="legal_representation",
    source_domains=["PERSON"],
    target_domains=["ORGANIZATION"],
    relationship_types=["REPRESENTS", "WORKS_FOR"],
    pattern_template=r"{person}.*(?:attorney|lawyer|representative).*{organization}",
    examples=[
        ("John Smith", "Davis & Partners", "REPRESENTS"),
        ("Jane Doe", "Legal Corp", "WORKS_FOR")
    ]
)

detector.register_pattern("legal_representation", pattern)

# Use custom pattern
relationships = await detector.detect_relationships(entities)
```

### Parallel Query Execution

```python
from knowledge_base.intelligence import FederatedQueryRouter, ExecutionStrategy

router = FederatedQueryRouter()

# Create plan with parallel execution
plan = await router.create_plan(
    query="Compare AI regulations in US vs EU with major tech companies",
    max_domains=4,
    strategy=ExecutionStrategy.PARALLEL
)

# Execute with custom retriever
result = await router.execute_plan(
    plan=plan,
    retriever=my_custom_retriever
)

# Process results
for domain_result in result.domain_results:
    print(f"\n{domain_result.domain.value}:")
    for entity in domain_result.entities:
        print(f"  - {entity.text}")
```

---

## Troubleshooting

### Common Issues

#### 1. LLM Connection Errors

```python
# Error: Connection refused to localhost:8087

# Check if LLM gateway is running
curl http://localhost:8087/v1/models

# Verify environment variables
import os
print(os.getenv("LLM_GATEWAY_URL"))
```

#### 2. Database Connection Issues

```python
# Error: Could not connect to PostgreSQL

# Verify PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string
DATABASE_URL=postgresql://agentzero@localhost:5432/knowledge_base
```

#### 3. Entity Extraction Timeout

```python
# Increase timeout in config
config = EntityExtractionConfig(timeout=120)

# Or reduce text size for extraction
chunks = text.split("\n\n")  # Process in smaller chunks
```

#### 4. Hallucination Detection False Positives

```python
# Adjust confidence threshold
detector = HallucinationDetector(
    config=HallucinationDetectorConfig(
        min_confidence_for_positive=0.3  # Lower threshold
    )
)
```

#### 5. Low Quality Scores

```python
# Enable human review for low quality
result = await extractor.extract_entities(
    text=text,
    domain=domain,
    options=ExtractionOptions(
        enable_evaluation=True,
        quality_threshold=0.5,
        route_low_quality_to_review=True
    )
)
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logger = logging.getLogger("knowledge_base.intelligence")
logger.setLevel(logging.DEBUG)
```

### Performance Optimization

```python
# Use parallel extraction
config = MultiAgentConfig(
    enable_parallel_extraction=True,
    max_concurrent_agents=8
)

# Batch entity verification
batch_result = await detector.verify_entity_batch(
    entities=all_entities,
    batch_size=20  # Increase batch size
)

# Use caching for domain schemas
registry = SchemaRegistry(cache_ttl=3600)  # 1 hour cache
```

---

## Appendices

### A. Entity Type Taxonomy

```
PERSON
├── Individual
│   ├── Politician
│   ├── Athlete
│   ├── Artist
│   └── Scientist
└── Group
    └── Team

ORGANIZATION
├── Company
│   ├── Startup
│   ├── Corporation
│   └── Subsidiary
├── Government
│   ├── Agency
│   └── Department
└── Non-Profit
    ├── Foundation
    └── Association

LOCATION
├── Geographic
│   ├── Country
│   ├── State/Province
│   └── City
├── Facility
│   ├── Building
│   └── Infrastructure
└── Region
    └── Historical

EVENT
├── Natural
│   ├── Disaster
│   └── Weather
└── Human
    ├── Conference
    ├── Conflict
    └── Ceremony

CONCEPT
├── Abstract Idea
├── Theory
└── Methodology

PRODUCT
├── Technology
│   ├── Software
│   └── Hardware
├── Media
│   ├── Book
│   └── Film
└── Service
```

### B. Relationship Taxonomy

```
Hierarchical
├── PART_OF / CONTAINS
├── SUBSET_OF / SUPERSET_OF
└── CATEGORY_OF / INSTANCE_OF

Ownership
├── OWNS / OWNED_BY
├── ACQUIRED / ACQUIREE
└── SPUN_OFF / MERGED_WITH

Spatial
├── LOCATED_IN
├── BORDERS_ON
└── CONNECTED_TO

Causal
├── CAUSES / CAUSED_BY
├── ENABLES / ENABLED_BY
└── PREVENTS / PREVENTED_BY

Temporal
├── PRECEDES / FOLLOWS
├── OVERLAPS_WITH
└── SIMULTANEOUS_WITH

Associative
├── RELATED_TO
├── ASSOCIATED_WITH
├── COMPETES_WITH
└── PARTNERS_WITH
```

### C. Domain Taxonomy

```
ROOT
├── GENERAL
├── TECHNOLOGY
│   ├── SOFTWARE
│   ├── HARDWARE
│   └── AI_ML
├── BUSINESS
│   ├── FINANCE
│   ├── MARKETING
│   └── OPERATIONS
├── LEGAL
│   ├── CONTRACT
│   ├── LITIGATION
│   └── REGULATORY
├── MEDICAL
│   ├── CLINICAL
│   ├── PHARMACEUTICAL
│   └── RESEARCH
├── ACADEMIC
│   ├── RESEARCH
│   └── EDUCATION
├── GOVERNMENT
│   ├── FEDERAL
│   ├── STATE
│   └── LOCAL
└── MEDIA
    ├── NEWS
    ├── ENTERTAINMENT
    └── PUBLISHING
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-27 | Initial implementation of all LLM-powered services |

---

*Document Generated: January 27, 2026*
*KBV2 Knowledge Base System - Complete Documentation*

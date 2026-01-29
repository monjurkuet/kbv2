# Entity Processing Pipeline - Complete Guide

## Overview

The entity processing pipeline transforms unstructured text into a structured, searchable knowledge graph through multiple stages. Here's the complete flow:

```
Raw Document → Partitioning → Extraction → Embedding → Resolution → Clustering → Synthesis
```

---

## Stage 1: Document Partitioning

**Location**: `src/knowledge_base/ingestion/v1/partitioning_service.py`

**Purpose**: Break documents into manageable chunks for LLM processing

**Process**:
1. Read the document file (PDF, TXT, etc.)
2. Split into chunks based on token count (default: 512 tokens)
3. Add overlap between chunks (default: 50 tokens) to preserve context
4. Extract metadata (page numbers, headers, etc.)

**Output**: List of `Chunk` objects
```python
Chunk(
    id=UUID,
    document_id=UUID,
    text=str,              # Chunk text content
    chunk_index=int,       # Position in document
    page_number=int,       # Page number (if applicable)
    token_count=int,       # Approximate token count
    metadata=dict          # Additional metadata
)
```

---

## Stage 2: Adaptive Gleaning Extraction

**Location**: `src/knowledge_base/ingestion/v1/gleaning_service.py`

**Purpose**: Extract entities, relationships, and temporal claims from each chunk

### 2a: Pass 1 - Discovery Pass

**What it does**:
- Extract obvious, clearly named entities
- Extract explicit relationships
- Extract temporal information (dates, timelines)
- Calculate `information_density` (0.0 - 1.0)

**LLM Prompt** (discovery pass):
```
You are an expert information extraction system.

Focus on:
1. Clearly named entities (people, organizations, locations, concepts)
2. Explicit relationships between entities
3. Temporal information (dates, times, durations)

Output JSON with:
- entities: List of extracted entities
- edges: List of relationships
- temporal_claims: List of temporal information
- information_density: Float (0.0-1.0) indicating remaining info density
```

**Extracted Entities**:
```python
ExtractedEntity(
    name="Project Phoenix",
    entity_type="Project",
    description="Quantum computing initiative",
    properties={"budget": "$89M"},
    confidence=0.95
)
```

**Extracted Edges**:
```python
ExtractedEdge(
    source="Dr. Sarah Chen",
    target="Project Phoenix",
    edge_type="leads",
    properties={"since": "2024-01-15"},
    confidence=0.95
)
```

### 2b: Adaptive Decision

**The Critical Adaptive Logic**:
```python
for pass_num in range(1, max_passes + 1):
    pass_result = await self._extract_pass(...)
    results.append(pass_result)

    # Adaptive pass control based on information density
    if pass_num < max_passes:
        if pass_result.information_density < min_density_threshold:
            break  # Skip second pass
```

**Decision Matrix**:
| Pass 1 Density | Pass 2 Executed? | Reason |
|---------------|------------------|---------|
| < 0.3 | NO | Low density, skip gleaning |
| >= 0.3 | YES | High density, run gleaning |

**Cost Savings**: 50% reduction in LLM calls for low-density content

### 2c: Pass 2 - Gleaning Pass (if triggered)

**What it does**:
- Find subtle, nested, or implicit relationships
- Identify hierarchical structures
- Discover technical or domain-specific connections
- Find additional temporal relationships

**LLM Prompt** (gleaning pass):
```
You are performing a second pass analysis.

Focus on:
1. Implicit relationships
2. Nested or hierarchical structures
3. Technical or domain-specific connections
4. Temporal relationships and dependencies

Previous extraction found:
- X entities
- Y relationships
- Information density: Z.ZZ

Identify ADDITIONAL entities and relationships that were missed.
```

**Value Added**: Pass 2 typically finds 20-50% additional entities and relationships

### 2d: Result Merging

**Process**:
1. Deduplicate entities by name
2. Merge edges (keep highest confidence)
3. Combine temporal claims
4. Average information density

---

## Stage 3: Entity Creation

**Location**: `src/knowledge_base/orchestrator.py:521-545`

**Purpose**: Convert extracted entities into database entities

**Process**:
```python
def _create_entities_from_extraction(extraction, chunk_id):
    return [
        Entity(
            id=uuid4(),                          # Unique ID
            name=entity.name,                    # Entity name
            entity_type=entity.entity_type,      # Type (Person, Org, etc.)
            description=entity.description,      # Description
            properties=entity.properties,        # Key-value properties
            confidence=entity.confidence         # Confidence score
        )
        for entity in extraction.entities
    ]
```

**Database Storage**:
- Entities stored in `entities` table
- Each entity gets a unique UUID
- Linked to source chunk via `chunk_entities` junction table

---

## Stage 4: Edge Creation

**Location**: `src/knowledge_base/orchestrator.py:547-655`

**Purpose**: Convert extracted relationships into database edges

**Process**:
1. Map entity names to entity IDs
2. Create edges between entities
3. Special handling for temporal claims (creates timeline edges)

**Edge Types**:
```python
class EdgeType(str, Enum):
    UNKNOWN = "unknown"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    CONTAINS = "contains"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    KNOWS = "knows"
    OWNS = "owns"
    INVALIDATED_BY = "invalidated_by"
```

**Temporal Edge Special Handling**:
```python
# Extract temporal claims with dates
for temporal_claim in extraction.temporal_claims:
    date_str = temporal_claim.iso8601_date.split("T")[0]  # YYYY-MM-DD

    # Create edge with temporal properties
    edge = Edge(
        source_id=person_entity.id,
        target_id=project_entity.id,
        edge_type=EdgeType.WORKS_FOR,
        properties={
            "status": "Active",  # or "Failed", "Success"
            "date": date_str
        },
        confidence=1.0
    )
```

---

## Stage 5: Embedding Generation

**Location**: `src/knowledge_base/ingestion/v1/embedding_client.py`

**Purpose**: Generate vector embeddings for semantic search

### 5a: Chunk Embeddings

**Process**:
```python
for chunk in chunks:
    embedding = await embedding_client.embed_text(chunk.text)
    await vector_store.update_chunk_embedding(str(chunk.id), embedding)
```

**Embedding Model**: `nomic-embed-text` (via Ollama)
**Dimensions**: 768
**Purpose**: Enable semantic search across document chunks

### 5b: Entity Embeddings

**Process**:
```python
for entity in entities:
    text = f"{entity.name}. {entity.description or ''}"
    embedding = await embedding_client.embed_text(text)
    await vector_store.update_entity_embedding(str(entity.id), embedding)
```

**Embedding Text**: Entity name + description
**Purpose**: Enable semantic similarity search for entity resolution

**Storage**: Embeddings stored in `pgvector` column (768-dimensional vectors)

---

## Stage 6: Global Entity Resolution

**Location**: `src/knowledge_base/intelligence/v1/resolution_agent.py` and `src/knowledge_base/orchestrator.py`

**Purpose**: Merge duplicate entities globally across all documents in the database (e.g., "Dr. Chen", "Sarah Chen", "Dr. Sarah Chen")

### 6a: Candidate Search

**Process**:
```python
for entity in entities:
    # Find similar entities using vector similarity
    similar = await vector_store.search_similar_entities(
        entity.embedding,
        limit=5,
        similarity_threshold=0.85
    )

    # Selection is GLOBAL - we search all entities in the database
    candidates = [
        e for e in all_database_entities
        if e.id != entity.id
        and str(e.id) in [s["id"] for s in similar]
    ]
```

**Global vs Local**: Historically, ER was document-local. The system now searches the entire vector store for candidates, enabling cross-document knowledge unification.

**Similarity Threshold**: 0.85 (85% similarity)

### 6b: LLM-Based Resolution

**Process**:
```python
resolution = await resolution_agent.resolve_entity(
    entity,
    candidates,
    source_text  # Original text for grounding
)
```

**Resolution Prompt**:
```
You are an expert entity resolution system.

Rules:
1. Consider semantic similarity of names
2. Consider entity type consistency
3. Consider contextual information
4. ALWAYS provide a verbatim quote from the source text
5. Assign a confidence score (0.0-1.0)
6. MERGING IS PROHIBITED WITHOUT A VERBATIM CITATION

Output JSON:
{
  "decision": "keep|merge",
  "target_entity_id": "<uuid>",
  "merged_entity_ids": ["<uuid>", ...],
  "grounding_quote": "verbatim excerpt from source text",
  "confidence_score": 0.85,
  "reasoning": "explanation of decision"
}
```

**Critical Feature**: Rule 6 prohibits merging without verbatim citation

### 6c: Merge Execution

**Process**:
```python
if resolution.merged_entity_ids:
    await self._merge_entities(resolution)
```

**Merge Operations**:
1. Update all `Edge` records (source/target) to point to the survivor entity.
2. Update all `ChunkEntity` links to point to the survivor entity (preserving evidence).
3. Handle potential duplicate `ChunkEntity` links (avoiding constraint violations).
4. Delete merged entities from the database.
5. Preserve confidence scores (max).

**Knowledge Health CLI**:
A dedicated command is available to perform a global deduplication sweep:
```bash
uv run python -m knowledge_base.clients.cli dedupe
```

**Resolution Model**:
```python
class EntityResolution(BaseModel):
    entity_id: UUID                    # Kept entity ID
    merged_entity_ids: list[UUID]     # Merged entity IDs
    grounding_quote: str              # Verbatim quote
    confidence_score: float           # 0.0-1.0
    human_review_required: bool       # Low confidence flag
```

---

## Stage 7: Hierarchical Clustering

**Location**: `src/knowledge_base/intelligence/v1/clustering_service.py`

**Purpose**: Group entities into communities (micro and macro)

### 7a: Build Graph

**Process**:
```python
# Create igraph from entities and edges
g = ig.Graph()
g.add_vertices(len(entities))

# Add edges with weights
for edge in edges:
    g.add_edges([(source_idx, target_idx)])
g.es["weight"] = [1.0] * len(edges)
```

### 7b: Leiden Clustering

**Algorithm**: Leiden algorithm with modularity optimization

**Macro Communities** (Resolution: 0.8):
```python
partition = la.find_partition(
    g,
    la.ModularityVertexPartition,
    n_iterations=10
)
```

**Micro Communities** (Resolution: 1.2):
- Higher resolution = more, smaller communities
- Only created if macro communities > 3

**Community Properties**:
```python
Community(
    id=UUID,
    name="Macro Community 0: Project Phoenix, Dr. Sarah Chen...",
    level=0 or 1,                    # 0=micro, 1=macro
    resolution=float,                # 0.8 or 1.2
    entity_count=int,                # Number of entities
    parent_id=UUID,                  # Parent community (for micro)
    summary=str                      # Generated by synthesis agent
)
```

### 7c: Hierarchy Building

**Structure**:
```
Level 1 (Macro): Broad themes
├── Level 0 (Micro): Specific clusters
│   └── Entities
└── Level 0 (Micro): Specific clusters
    └── Entities
```

**Example**:
```
Macro Community "Executive Leadership"
├── Micro Community "CEO Office"
│   ├── Dr. Sarah Chen
│   ├── Jennifer Park
│   └── ...
└── Micro Community "Engineering Team"
    ├── Dr. Marcus Webb
    ├── James Liu
    └── ...
```

---

## Stage 8: Synthesis and Reporting

**Location**: `src/knowledge_base/intelligence/v1/synthesis_agent.py`

**Purpose**: Generate intelligence reports for each community

### 8a: Micro Reports (Level 0)

**Process**:
```python
for community in leaf_communities:
    community_entities = get_entities_in_community(community)
    community_edges = get_edges_in_community(community)

    report = await synthesis_agent.generate_micro_report(
        community,
        community_entities,
        community_edges
    )
```

**Micro Report Prompt**:
```
Generate a detailed, factual micro-report for this community.

Guidelines:
1. Be comprehensive and factually accurate
2. Include all key entities and their roles
3. Detail specific relationships with confidence scores
4. Focus on what is explicitly known from the data
5. Maintain fidelity to raw relationship data
6. Preserve temporal and contextual information
7. Include all specific technical details, values, dates, and specifications

Output JSON:
{
  "summary": "detailed factual summary",
  "key_entities": ["entity1", "entity2"],
  "key_relationships": ["relationship descriptions"],
  "thematic_focus": ["theme1", "theme2"]
}
```

**Output**:
```python
MicroReport(
    community_id=UUID,
    summary="Dr. Sarah Chen serves as Chief Quantum Architect for Project Phoenix...",
    key_entities=["Dr. Sarah Chen", "Project Phoenix", "QPU-Alpha"],
    entity_count=5,
    key_relationships=["leads", "manages", "part_of"]
)
```

### 8b: Macro Reports (Level 1)

**Process**:
```python
for community in parent_communities:
    child_reports = get_micro_reports_for_children(community)
    cross_edges = get_edges_between_children(community)

    report = await synthesis_agent.generate_macro_report(
        community,
        child_reports,
        cross_edges
    )
```

**Macro Report Prompt**:
```
Generate a strategic macro-report synthesizing child community reports.

Guidelines:
1. Identify cross-cutting themes and patterns
2. Synthesize relationships between communities
3. Preserve high-confidence edge information
4. Provide strategic insights while maintaining factual accuracy
5. Do not smooth over or omit specific technical details
6. Preserve all specific technical details from child reports including exact values, dates, version numbers, and technical specifications

Output JSON:
{
  "summary": "strategic synthesis",
  "thematic_focus": ["theme1", "theme2", "theme3"],
  "key_insights": ["insight1", "insight2"]
}
```

**Output**:
```python
MacroReport(
    community_id=UUID,
    summary="The Executive Leadership community oversees quantum computing initiatives...",
    child_reports=[UUID1, UUID2, UUID3],
    thematic_focus=["Leadership", "Strategy", "Oversight"]
)
```

### 8c: Report Storage

**Process**:
```python
for community in communities:
    if community.id in reports:
        report = reports[community.id]
        community.summary = report.summary
        await session.commit()
```

---

## Complete Data Flow Example

```
Input Document:
"The Quantum Computing Infrastructure consists of three QPUs.
Dr. Sarah Chen is the Chief Quantum Architect.
Project Phoenix uses IBM Quantum System One."

↓

Stage 1: Partitioning
Chunk 1: "The Quantum Computing Infrastructure..."

↓

Stage 2: Extraction (Pass 1)
Entities:
  - "Quantum Computing Infrastructure" (System)
  - "QPU" (Component)
  - "Dr. Sarah Chen" (Person)
  - "Chief Quantum Architect" (Role)
  - "Project Phoenix" (Project)
  - "IBM Quantum System One" (Hardware)

Edges:
  - Dr. Sarah Chen --[is]--> Chief Quantum Architect
  - Project Phoenix --[uses]--> IBM Quantum System One

Information Density: 0.75 (high)

↓

Stage 2: Extraction (Pass 2 - Gleaning)
Additional Entities:
  - "Chief Architect" (Role - synonym)
Additional Edges:
  - Dr. Sarah Chen --[leads]--> Project Phoenix

↓

Stage 3: Entity Creation
Entity 1: UUID-001, "Quantum Computing Infrastructure"
Entity 2: UUID-002, "Dr. Sarah Chen"
Entity 3: UUID-003, "Project Phoenix"
...

↓

Stage 4: Edge Creation
Edge 1: UUID-002 --[is]--> UUID-001
Edge 2: UUID-003 --[uses]--> UUID-001
...

↓

Stage 5: Embedding
Entity "Dr. Sarah Chen" → [0.12, -0.45, 0.78, ...]  # 768-dim vector

↓

Stage 6: Resolution
"Dr. Sarah Chen" + "Chief Architect" → Merge to "Dr. Sarah Chen"
  (Grounding quote: "Dr. Sarah Chen is the Chief Quantum Architect")

↓

Stage 7: Clustering
Macro Community "Executive Team":
  Micro Community "Leadership":
    - Dr. Sarah Chen
    - Project Phoenix
    - IBM Quantum System One

↓

Stage 8: Synthesis
Micro Report:
  "Dr. Sarah Chen serves as Chief Quantum Architect for Project Phoenix,
   which uses IBM Quantum System One. Confidence: 0.95"

Macro Report:
  "The Executive Team leads quantum computing initiatives including
   Project Phoenix and its IBM Quantum System One infrastructure."
```

---

## Key Features

### 1. Adaptive Processing
- Low-density content: 1 LLM pass (50% cost savings)
- High-density content: 2 LLM passes (comprehensive extraction)

### 2. Grounded Resolution
- Entity merging requires verbatim citations
- Confidence scores track uncertainty
- Human review flag for low-confidence decisions

### 3. Hierarchical Organization
- Micro communities: Specific clusters (resolution 1.2)
- Macro communities: Broad themes (resolution 0.8)
- Nested hierarchy for multi-level insights

### 4. Temporal Awareness
- ISO-8601 normalized dates
- Timeline tracking for status changes
- Relative date support ("yesterday", "last month")

### 5. Fidelity Preservation
- Technical details preserved in synthesis
- No smoothing over minor details
- Exact values, dates, versions retained

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Chunk Size | 512 tokens |
| Embedding Dimensions | 768 |
| Similarity Threshold | 0.85 |
| Clustering Resolution (Micro) | 1.2 |
| Clustering Resolution (Macro) | 0.8 |
| Density Threshold (Low) | < 0.3 |
| Density Threshold (High) | >= 0.3 |

---

## Monitoring with Logfire

**Metrics Tracked**:
- `gleaning_pass_count`: Passes per document
- `information_density_pass_1`: Density after pass 1
- `information_density_pass_2`: Density after pass 2
- `entities_extracted_pass_1`: Entities found in pass 1
- `entities_extracted_pass_2`: Entities found in pass 2
- `resolution_confidence_score`: Average resolution confidence
- `clustering_modularity`: Community modularity score
- `clustering_community_count`: Number of communities

**Spans Traced**:
- `document_ingestion`: Full ingestion pipeline
- `partition_document`: Chunk splitting
- `extract_knowledge`: Entity extraction
- `embed_content`: Vector generation
- `resolve_entities`: Duplicate resolution
- `cluster_entities`: Community detection
- `generate_reports`: Synthesis

---

## Summary

The entity processing pipeline transforms raw documents into a rich, structured knowledge graph through:

1. **Intelligent Extraction**: Adaptive 2-pass gleaning based on content density
2. **Semantic Understanding**: Vector embeddings for similarity search
3. **Grounded Resolution**: Citation-based entity merging
4. **Hierarchical Organization**: Multi-level community clustering
5. **Comprehensive Synthesis**: Fidelity-preserving intelligence reports

This pipeline enables powerful knowledge discovery, semantic search, and intelligent querying across document collections.
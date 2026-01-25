# KBV2 Data Logic Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA TRANSFORMATION FLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT DATA (Unstructured Document)
│
├─ PDF File
│  └─ Binary data → Text extraction → Structured elements
│
├─ DOCX File
│  └─ Binary data → Text extraction → Structured elements
│
└─ Text File
   └─ Raw text → Structured elements

        │
        ▼
PARTITIONING DATA
│
├─ Document Record
│  ├─ id: UUID
│  ├─ name: "project_report.pdf"
│  ├─ source_uri: "/path/to/file.pdf"
│  ├─ mime_type: "application/pdf"
│  ├─ status: PENDING → PARTITIONED
│  └─ domain: null
│
└─ Chunk Records (many)
   ├─ chunk[0]: "Project Nova was initiated in August 2021..."
   ├─ chunk[1]: "Elena Vance was appointed as the lead..."
   ├─ chunk[2]: "The project encountered technical challenges..."
   └─ chunk[N]: "...final report submitted"
       │
       ▼
EXTRACTION DATA (GleaningService)
│
├─ Pass 1 Output (Discovery)
│  ├─ Entities: [Project Nova, Elena Vance, Technical Team]
│  ├─ Edges: [(Elena Vance, WORKS_FOR, Project Nova)]
│  ├─ Temporal Claims: [
│  │    {text: "August 2021: Project initiated", type: STATIC, date: 2021-08-01}
│  │  ]
│  └─ Information Density: 0.7
│
├─ Pass 2 Output (Gleaning) - if density > 0.3
│  ├─ Entities: [Dr. Elena Vance, Technical Challenges]
│  ├─ Edges: [(Dr. Elena Vance, MANAGES, Technical Team)]
│  ├─ Temporal Claims: [
│  │    {text: "May 2023: Technical challenges, status Failed", type: STATIC, date: 2023-05-01}
│  │  ]
│  └─ Information Density: 0.2
│
└─ Merged Result
   ├─ Entities (deduplicated): [Project Nova, Elena Vance, Technical Team, Technical Challenges]
   ├─ Edges (merged): [
   │    (Elena Vance, WORKS_FOR, Project Nova),
   │    (Elena Vance, MANAGES, Technical Team)
   │  ]
   └─ Temporal Claims (unique): [...]
       │
       ▼
ENTITY CREATION DATA
│
├─ Entity Records
│  ├─ Entity[0]:
│  │  ├─ id: UUID-1
│  │  ├─ name: "Project Nova"
│  │  ├─ entity_type: "Project"
│  │  ├─ description: "A software development project initiated in 2021"
│  │  ├─ uri: "entity:project_nova"
│  │  ├─ confidence: 0.95
│  │  └─ embedding: null → [0.123, 0.456, ..., 0.789] (768-dim)
│  │
│  ├─ Entity[1]:
│  │  ├─ id: UUID-2
│  │  ├─ name: "Elena Vance"
│  │  ├─ entity_type: "Person"
│  │  ├─ description: "Project lead appointed in 2021"
│  │  ├─ uri: "entity:elena_vance"
│  │  ├─ confidence: 0.90
│  │  └─ embedding: null → [0.234, 0.567, ..., 0.890] (768-dim)
│  │
│  └─ Entity[N]: ...
│
├─ Edge Records
│  ├─ Edge[0]:
│  │  ├─ id: UUID-E1
│  │  ├─ source_id: UUID-2 (Elena Vance)
│  │  ├─ target_id: UUID-1 (Project Nova)
│  │  ├─ edge_type: WORKS_FOR
│  │  ├─ confidence: 0.95
│  │  ├─ temporal_validity_start: 2021-08-01
│  │  ├─ temporal_validity_end: null
│  │  ├─ source_text: "Elena Vance was appointed as the lead"
│  │  └─ domain: null
│  │
│  └─ Edge[N]: ...
│
└─ ChunkEntity Junction Records (many-to-many)
   ├─ (chunk[0].id, Entity[0].id, grounding_quote: "Project Nova was initiated...")
   ├─ (chunk[0].id, Entity[1].id, grounding_quote: "Elena Vance was appointed...")
   └─ (chunk[N].id, Entity[M].id, grounding_quote: "...")
       │
       ▼
ENTITY RESOLUTION DATA
│
├─ Vector Search Result
│  └─ Similar entities to UUID-2 (Elena Vance):
│      ├─ UUID-3: "Dr. Elena Vance" (similarity: 0.92)
│      └─ UUID-4: "E. Vance" (similarity: 0.88)
│
├─ LLM Resolution
│  ├─ Decision: "merge"
│  ├─ Target: UUID-2 (keep)
│  ├─ Merged: [UUID-3, UUID-4]
│  ├─ Grounding Quote: "Dr. Elena Vance, who was appointed as the lead"
│  ├─ Confidence: 0.85
│  └─ Human Review: false (confidence >= 0.7)
│
└─ Merge Action
   ├─ Update all edges with source_id = UUID-3 → UUID-2
   ├─ Update all edges with target_id = UUID-3 → UUID-2
   ├─ Delete entity UUID-3
   └─ Repeat for UUID-4
       │
       ▼
CLUSTERING DATA
│
├─ Graph Construction
│  ├─ Nodes: All entities (UUID-1, UUID-2, UUID-5, ...)
│  └─ Edges: All relationships (weighted by confidence)
│
├─ Level 0 Clustering (Macro)
│  ├─ Community A (id: UUID-C1, level: 0, resolution: 0.8)
│  │  ├─ entities: [UUID-1, UUID-2, UUID-5]
│  │  └─ entity_count: 3
│  │
│  ├─ Community B (id: UUID-C2, level: 0, resolution: 0.8)
│  │  ├─ entities: [UUID-6, UUID-7]
│  │  └─ entity_count: 2
│  │
│  └─ Community C (id: UUID-C3, level: 0, resolution: 0.8)
│      ├─ entities: [UUID-8, UUID-9, UUID-10]
│      └─ entity_count: 3
│
└─ Level 1 Clustering (Micro)
   ├─ Micro A1 (id: UUID-C4, level: 1, resolution: 1.2, parent: UUID-C1)
   │  ├─ entities: [UUID-1, UUID-2]
   │  └─ entity_count: 2
   │
   ├─ Micro A2 (id: UUID-C5, level: 1, resolution: 1.2, parent: UUID-C1)
   │  ├─ entities: [UUID-5]
   │  └─ entity_count: 1
   │
   └─ Micro B1 (id: UUID-C6, level: 1, resolution: 1.2, parent: UUID-C2)
      ├─ entities: [UUID-6, UUID-7]
      └─ entity_count: 2
       │
       ▼
SYNTHESIS DATA
│
├─ Micro Reports (Level 1)
│  ├─ Micro A1 Summary:
│  │  "This community contains Project Nova and Elena Vance. Key relationships
│  │   include Elena Vance working for Project Nova (confidence 0.95, from
│  │   August 2021). The project was initiated in August 2021 with status
│  │   Active."
│  │
│  └─ Micro A2 Summary:
│     "This community contains the Technical Team entity."
│
├─ Macro Reports (Level 0)
│  └─ Community A Summary:
│     "This community focuses on Project Nova and its key personnel. Elena
│      Vance leads the project, which also includes technical team members.
│      The project was initiated in August 2021 and has progressed through
│      multiple phases. Key relationships: WORKS_FOR, MANAGES."
│
└─ Community.summary updated for all communities
       │
       ▼
DOMAIN PROPAGATION DATA
│
├─ Domain Determination
│  └─ Heuristic: "project_report.pdf" → "technology"
│
└─ Domain Assignment
   ├─ Document.domain = "technology"
   ├─ Entity[0].domain = "technology"
   ├─ Entity[1].domain = "technology"
   ├─ Edge[0].domain = "technology"
   └─ All entities/edges from this document = "technology"
       │
       ▼
FINAL DATA STATE
│
├─ Document: COMPLETED, domain = "technology"
├─ Chunks: N chunks, all embedded
├─ Entities: M entities, all embedded, resolved, clustered, with domain
├─ Edges: P edges, with temporal validity, provenance, domain
├─ Communities: Q communities (macro + micro), with summaries
└─ Observability: All metrics logged
```

## Data Transformation Stages

### 1. Input Processing
- **PDF/DOCX**: Binary → Structured elements via `unstructured` library
- **Text**: Raw text → Structured elements
- **Output**: Document record + Chunk records

### 2. Knowledge Extraction
- **Pass 1 (Discovery)**: Extract obvious entities, relationships, temporal claims
- **Pass 2 (Gleaning)**: Find implicit, nested, technical relationships
- **Merge**: Deduplicate entities and edges
- **Output**: Entity records, Edge records, Temporal claims

### 3. Entity Creation
- **URI Generation**: RDF-style URIs (e.g., `entity:project_nova`)
- **Confidence**: 0.0-1.0 based on extraction certainty
- **Properties**: Flexible JSON storage for entity attributes
- **Output**: Entity records with null embeddings

### 4. Embedding Generation
- **Model**: Ollama `nomic-embed-text`
- **Dimensions**: 768-dim vectors
- **Input**: Entity text (name + description), Chunk text
- **Output**: Updated entity.embedding and chunk.embedding

### 5. Entity Resolution
- **Vector Search**: Find similar entities (threshold 0.85)
- **LLM Reasoning**: Requires verbatim quote from source
- **Decision**: Merge (confidence >= 0.7) or ReviewQueue (confidence < 0.7)
- **Output**: Merged entities, updated edges

### 6. Clustering
- **Level 0**: Macro communities (resolution 0.8)
- **Level 1**: Micro communities (resolution 1.2)
- **Hierarchy**: Micro communities have macro parents
- **Output**: Community records + entity.community_id updates

### 7. Synthesis
- **Micro Reports**: Detailed summaries of leaf communities
- **Macro Reports**: Strategic synthesis of child reports
- **Edge Fidelity**: Preserve raw relationships to prevent information loss
- **Output**: Community.summary updates

### 8. Domain Propagation
- **Determination**: User-provided → Metadata → Heuristic
- **Propagation**: Document → Entities → Edges
- **Domains**: technology, healthcare, finance, general
- **Output**: domain field populated for all records

### 9. Final State
- **Document**: Status = COMPLETED
- **Chunks**: Embedded, linked to entities
- **Entities**: Embedded, resolved, clustered, with domain
- **Edges**: With temporal validity, provenance, domain
- **Communities**: With summaries
- **Observability**: All metrics logged
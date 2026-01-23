# KBV2 Database Schema Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABASE SCHEMA RELATIONSHIPS                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│   DOCUMENT      │
│  ─────────────  │
│ id (PK, UUID)   │
│ name            │
│ source_uri      │
│ mime_type       │
│ status          │─── PENDING | PARTITIONED | EXTRACTED | EMBEDDED | COMPLETED | FAILED
│ metadata (JSON) │
│ domain          │
│ created_at      │
│ updated_at      │
└────────┬────────┘
         │ 1
         │
         │ N
         ▼
┌─────────────────┐
│     CHUNK       │
│  ─────────────  │
│ id (PK, UUID)   │
│ document_id (FK)│◀──────────────────┐
│ text            │                   │
│ chunk_index     │                   │
│ page_number     │                   │
│ token_count     │                   │
│ metadata (JSON) │                   │
│ embedding (vec) │                   │  N
│ created_at      │                   │
└────────┬────────┘                   │
         │                           │
         │ M                         │
         │                           │
         │ N                         │
         │                           │
         ▼                           │
┌─────────────────┐                   │
│  CHUNK_ENTITY   │◀──────────────────┘
│   (JUNCTION)    │
│  ─────────────  │
│ id (PK, UUID)   │
│ chunk_id (FK)   │
│ entity_id (FK)  │◀──────────────────┐
│ grounding_quote │                   │
│ confidence      │                   │
│ created_at      │                   │
└─────────────────┘                   │
                                     │ M
                                     │
                                     │ N
                                     │
                                     ▼
                          ┌─────────────────┐
                          │     ENTITY      │
                          │  ─────────────  │
                          │ id (PK, UUID)   │
                          │ name            │
                          │ entity_type     │
                          │ description     │
                          │ properties (JSON)│
                          │ confidence      │
                          │ embedding (vec) │◀─────┐
                          │ uri (unique)    │      │
                          │ source_text     │      │
                          │ domain          │      │
                          │ community_id(FK)│      │
                          │ created_at      │      │
                          │ updated_at      │      │
                          └────────┬────────┘      │
                                   │               │
                                   │ M             │
                                   │               │
                                   │ N             │
                                   │               │
                ┌──────────────────┼───────────────┼──────────────────┐
                │                  │               │                  │
                │ N                │ N             │ N                │ N
                │                  │               │                  │
                ▼                  ▼               ▼                  ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ EDGE (source)│  │ EDGE (target)│  │ ChunkEntity  │  │   COMMUNITY  │
        │   (FK)       │  │   (FK)       │  │   (FK)       │  │   (FK)       │
        └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
                                                    │                    │
                                                    │                    │ M
                                                    │                    │
                                                    │ 1                  │
                                                    │                    │
                                                    ▼                    ▼
                                         ┌─────────────────┐  ┌─────────────────┐
                                         │     CHUNK       │  │   COMMUNITY    │
                                         │  (FK)           │  │  ─────────────  │
                                         └─────────────────┘  │ id (PK, UUID)   │
                                                              │ name            │
                                                              │ level           │
                                                              │ resolution      │
                                                              │ summary         │
                                                              │ entity_count    │
                                                              │ parent_id (FK)  │◀─────┐
                                                              │ created_at      │      │
                                                              │ updated_at      │      │
                                                              └─────────────────┘      │
                                                                                        │ M
                                                                                        │
                                                                                        │ N
                                                                                        │
                                                                                        │
                                                                                        ▼
                                                                         ┌─────────────────┐
                                                                         │   COMMUNITY    │
                                                                         │   (parent)     │
                                                                         │   (FK)         │
                                                                         └─────────────────┘

┌─────────────────┐
│      EDGE       │
│  ─────────────  │
│ id (PK, UUID)   │
│ source_id (FK)  │◀─────────────────────┐
│ target_id (FK)  │◀─────────────────────┤
│ edge_type       │─── 30+ relation types│
│ properties (JSON)│                    │
│ confidence      │                    │
│ temporal_validity_start             │
│ temporal_validity_end               │
│ provenance      │                    │
│ source_text     │                    │
│ is_directed     │                    │
│ domain          │                    │
│ created_at      │                    │
└─────────────────┘                    │
                                        │
                                        │ M
                                        │
                                        │ N
                                        │
                                        ▼
                              ┌─────────────────┐
                              │     ENTITY      │
                              │  (source/target)│
                              └─────────────────┘

┌─────────────────┐
│   REVIEW_QUEUE  │
│  ─────────────  │
│ id (PK, UUID)   │
│ item_type       │─── entity_resolution | edge_validation
│ entity_id (FK)  │◀─────────────────────┐
│ edge_id (FK)    │◀─────────────────────┤
│ document_id (FK)│◀─────────────────────┤
│ merged_entity_ids│                    │
│ confidence_score │                   │
│ grounding_quote  │                    │
│ source_text     │                    │
│ status          │─── PENDING | APPROVED | REJECTED
│ priority        │─── 1-10
│ reviewer_notes  │
│ reviewed_by     │
│ reviewed_at     │
│ created_at      │
└─────────────────┘                    │
                                        │
                                        │ M
                                        │
                                        │ N
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ ENTITY / EDGE   │
                              │   / DOCUMENT    │
                              └─────────────────┘
```

## Table Descriptions

### Document
Stores metadata about processed documents.
- **status**: Processing state (PENDING → PARTITIONED → EXTRACTED → EMBEDDED → COMPLETED/FAILED)
- **domain**: Domain classification (technology, healthcare, finance, general)
- **metadata**: Flexible JSON storage for document-specific metadata

### Chunk
Represents a semantic chunk of a document.
- **embedding**: 768-dim vector for similarity search
- **chunk_index**: Sequential index within document
- **page_number**: Original page number (if applicable)

### Entity
Represents a knowledge graph entity (person, organization, concept, etc.).
- **uri**: RDF-style unique identifier (e.g., `entity:project_nova`)
- **embedding**: 768-dim vector for entity resolution
- **community_id**: Links to community cluster
- **source_text**: Original text that established this entity

### Edge
Represents a relationship between two entities.
- **edge_type**: One of 30+ relation types (WORKS_FOR, CAUSES, PART_OF, etc.)
- **temporal_validity_start/end**: Temporal bounds for the relationship
- **provenance**: Source of the relationship
- **source_text**: Original text that established this relationship
- **is_directed**: Whether the relationship is directed

### ChunkEntity (Junction Table)
Many-to-many relationship between chunks and entities.
- **grounding_quote**: Verbatim quote linking entity to chunk
- **confidence**: Confidence score for the entity-chunk link

### Community
Represents a cluster of related entities (Leiden clustering).
- **level**: Hierarchy level (0 = macro, 1 = micro)
- **resolution**: Clustering resolution parameter
- **parent_id**: Self-referential for hierarchical structure
- **summary**: AI-generated summary of the community

### ReviewQueue
Tracks items requiring human review.
- **item_type**: Type of review (entity_resolution, edge_validation)
- **status**: Review state (PENDING, APPROVED, REJECTED)
- **priority**: 1-10 scale for review prioritization
- **confidence_score**: Original confidence score that triggered review

## Key Relationships

### One-to-Many
- **Document → Chunk**: One document has many chunks
- **Community → Entity**: One community contains many entities
- **Community → Community**: One community can have many child communities

### Many-to-Many
- **Chunk ↔ Entity**: Via ChunkEntity junction table
- **Entity ↔ Edge**: As source and target

### Self-Referential
- **Community → Community**: Parent-child hierarchy

## Edge Types (30+)

### Hierarchical
- PART_OF, SUBCLASS_OF, INSTANCE_OF, CONTAINS

### Causal
- CAUSES, CAUSED_BY, INFLUENCES, INFLUENCED_BY

### Temporal
- PRECEDES, FOLLOWS, CO_OCCURS_WITH

### Spatial
- LOCATED_IN, LOCATED_NEAR

### Social/Organizational
- WORKS_FOR, WORKS_WITH, REPORTS_TO, KNOWS, COLLEAGUE_OF

### Ownership
- OWNS, MANAGES, OPERATES

### Activity
- PARTICIPATES_IN, PERFORMS, TARGETS, AFFECTS

### Validity
- INVALIDATED_BY, REPLACED_BY, SUCCEEDED_BY, PREDECESSOR_OF

### Long-tail
- UNKNOWN, NOTA (None-of-the-Above), HYPOTHETICAL

### Document-Specific
- RELATED_TO, MENTIONS, REFERENCES, DISCUSSES
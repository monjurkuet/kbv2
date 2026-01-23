# KBV2 System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KBV2 KNOWLEDGE BASE SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   USER INPUT │
│  (Document)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INGESTION ORCHESTRATOR                               │
│                           (ReAct Loop)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ├────────────────────────────────────────────────────────────────────┐
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CREATE DOCUMENT                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Document Record (PENDING)                                            │ │
│  │  - id, name, source_uri, mime_type, status, metadata, domain        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: PARTITION DOCUMENT                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  PartitioningService                                                  │ │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐                │ │
│  │  │  PDF/DOCX  │───▶│  Extract   │───▶│  Chunks    │                │ │
│  │  │  Elements  │    │   Title    │    │  (512 tok) │                │ │
│  │  └────────────┘    └────────────┘    └────────────┘                │ │
│  │                                                                      │ │
│  │  Chunk Records (PARTITIONED)                                         │ │
│  │  - id, document_id, text, chunk_index, page_number, metadata        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: EXTRACT KNOWLEDGE (Adaptive Gleaning)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  GleaningService                                                      │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  PASS 1: DISCOVERY                                           │  │ │
│  │  │  ├─ Extract obvious entities                                │  │ │
│  │  │  ├─ Extract explicit relationships                          │  │ │
│  │  │  ├─ Extract temporal claims                                 │  │ │
│  │  │  └─ Calculate information density                           │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼ (if density > 0.3 & stable < 90%)     │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  PASS 2: GLEANING (Optional)                                 │  │ │
│  │  │  ├─ Find implicit relationships                             │  │ │
│  │  │  ├─ Find nested/hierarchical structures                      │  │ │
│  │  │  ├─ Find technical connections                               │  │ │
│  │  │  └─ Handle long-tail relations (NOTA, HYPOTHETICAL)          │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  MERGE RESULTS                                                │  │ │
│  │  │  ├─ Deduplicate entities                                     │  │ │
│  │  │  ├─ Merge edges                                              │  │ │
│  │  │  └─ Combine temporal claims                                  │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  Entity Records                                                      │ │
│  │  - id, name, entity_type, description, properties, confidence,     │ │
│  │    uri, source_text, domain                                         │ │
│  │                                                                      │ │
│  │  Edge Records                                                        │ │
│  │  - id, source_id, target_id, edge_type, properties, confidence,     │ │
│  │    temporal_validity_start/end, provenance, source_text, domain     │ │
│  │                                                                      │ │
│  │  Temporal Claims (normalized to ISO-8601)                           │ │
│  │  - text, type (atemporal/static/dynamic), iso8601_date              │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: EMBED CONTENT                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  EmbeddingClient (Ollama - nomic-embed-text)                         │ │
│  │                                                                      │ │
│  │  Chunk Text ──▶ [768-dim vector] ──▶ chunk.embedding                 │ │
│  │  Entity Text ──▶ [768-dim vector] ──▶ entity.embedding               │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Document Status: EMBEDDED                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: RESOLVE ENTITIES (Verbatim-Grounded)                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  ResolutionAgent                                                      │ │
│  │                                                                      │ │
│  │  For each entity:                                                    │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  1. Vector Search (similarity > 0.85)                        │  │ │
│  │  │     entity.embedding ──▶ similar entities                    │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  2. LLM Reasoning (requires verbatim quote)                  │  │ │
│  │  │     ├─ Compare semantic similarity                           │  │ │
│  │  │     ├─ Check entity type consistency                         │  │ │
│  │  │     ├─ Validate quote against source text                    │  │ │
│  │  │     └─ Assign confidence score (0.0-1.0)                     │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  3. Decision                                                  │  │ │
│  │  │     ├─ Confidence >= 0.7 → Merge entities                    │  │ │
│  │  │     ├─ Confidence < 0.7 → Add to ReviewQueue                 │  │ │
│  │  │     └─ No quote → Keep separate (conservative)               │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  EntityResolution Result:                                           │ │
│  │  - entity_id (kept), merged_entity_ids, grounding_quote,           │ │
│  │    confidence_score, human_review_required                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: CLUSTER ENTITIES (Hierarchical Leiden)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  ClusteringService (igraph + leidenalg)                             │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  Build Graph                                                  │  │ │
│  │  │  Entities + Edges ──▶ Weighted undirected graph              │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  LEVEL 0: Macro Clustering (resolution = 0.8)                 │  │ │
│  │  │  ─────────────────────────────────────────────────────        │  │ │
│  │  │  Community A ──┬── Entity 1, Entity 2, Entity 3              │  │ │
│  │  │  Community B ──┼── Entity 4, Entity 5                         │  │ │
│  │  │  Community C ──┴── Entity 6, Entity 7, Entity 8              │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  LEVEL 1: Micro Clustering (resolution = 1.2)                │  │ │
│  │  │  ─────────────────────────────────────────────────────        │  │ │
│  │  │  Micro A1 ──┬── Entity 1, Entity 2      (parent: Community A) │  │ │
│  │  │  Micro A2 ──┴── Entity 3               (parent: Community A) │  │ │
│  │  │  Micro B1 ──── Entity 4, Entity 5      (parent: Community B) │  │ │
│  │  │  Micro C1 ──┬── Entity 6, Entity 7      (parent: Community C) │  │ │
│  │  │  Micro C2 ──┴── Entity 8               (parent: Community C) │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  Community Records:                                                  │ │
│  │  - id, name, level, resolution, summary, entity_count, parent_id    │ │
│  │                                                                      │ │
│  │  Entity.community_id updated                                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: GENERATE REPORTS (Map-Reduce Summarization)                       │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  SynthesisAgent                                                      │ │
│  │                                                                      │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  MICRO-REPORTS (Level 1 - Leaf Communities)                  │  │ │
│  │  │  ─────────────────────────────────────────────────────        │  │ │
│  │  │  For each micro community:                                   │  │ │
│  │  │  ├─ List all entities                                        │  │ │
│  │  │  ├─ List all edges (raw relationships)                      │  │ │
│  │  │  ├─ Extract key themes                                       │  │ │
│  │  │  └─ Generate detailed summary (max 2000 tokens)             │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                        │ │
│  │                              ▼                                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐  │ │
│  │  │  MACRO-REPORTS (Level 0 - Parent Communities)                │  │ │
│  │  │  ─────────────────────────────────────────────────────        │  │ │
│  │  │  For each macro community:                                   │  │ │
│  │  │  ├─ Synthesize child micro-reports                           │  │ │
│  │  │  ├─ Preserve edge fidelity (raw relationships)               │  │ │
│  │  │  ├─ Identify strategic themes                                │  │ │
│  │  │  └─ Generate strategic overview (max 2000 tokens)            │  │ │
│  │  └──────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  Community.summary updated                                           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 8: UPDATE DOMAIN                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Domain Propagation                                                   │ │
│  │                                                                      │ │
│  │  document.domain ──▶ entities.domain ──▶ edges.domain                │ │
│  │                                                                      │ │
│  │  Domain determination:                                               │ │
│  │  ├─ User-provided domain (if specified)                             │ │
│  │  ├─ Metadata domain (if available)                                  │ │
│  │  └─ Heuristic classification (technology/healthcare/finance/general)│ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 9: COMPLETE                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Document Status: COMPLETED                                           │ │
│  │                                                                      │ │
│  │  Observability Metrics Logged:                                       │ │
│  │  - chunks_created, entities_extracted, edges_extracted               │ │
│  │  - entities_embedded, chunks_embedded, communities_created           │ │
│  │  - Token usage, operation duration, success rate                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages Summary

1. **CREATE DOCUMENT**: Initialize document record with PENDING status
2. **PARTITION DOCUMENT**: Split into semantic chunks (512 tokens, 50 overlap)
3. **EXTRACT KNOWLEDGE**: 2-pass adaptive extraction (Discovery + Gleaning)
4. **EMBED CONTENT**: Generate 768-dim vectors using Ollama
5. **RESOLVE ENTITIES**: Verbatim-grounded deduplication with LLM reasoning
6. **CLUSTER ENTITIES**: Hierarchical Leiden clustering (Macro + Micro)
7. **GENERATE REPORTS**: Map-reduce summarization with edge fidelity
8. **UPDATE DOMAIN**: Propagate domain to entities and edges
9. **COMPLETE**: Final status with observability metrics
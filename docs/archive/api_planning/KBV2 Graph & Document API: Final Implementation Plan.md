This is the **Final Implementation Plan** for the KBV2 Graph & Document API.

This plan synthesizes the architectural constraints from the `database_schema.md` and `system_architecture_flow.md` with the requirements for frontend visualization (Sigma.js) and Google AIP compliance.

---

# KBV2 Graph & Document API: Final Implementation Plan

## 1. Architectural Strategy

We will extend the existing FastAPI application with two new high-level domains: **Graph** (visualization) and **Documents** (evidence). These will sit alongside the existing `query` and `review` APIs, sharing a standardized `common` infrastructure layer.

### Module Structure Updates

```text
src/knowledge_base/
├── common/
│   ├── api_models.py       # [NEW] Standardized AIP-193 Responses
│   ├── pagination.py       # [NEW] AIP-158 Pagination Logic
│   └── error_handlers.py   # [NEW] Global Exception Handling
├── graph_api.py            # [NEW] Graph Visualization Endpoints
├── document_api.py         # [NEW] Document Evidence Endpoints
└── main.py                 # [UPDATE] Router inclusion & Middleware

```

---

## 2. Core Infrastructure (Phase 1)

Before implementing specific endpoints, we establish the standard communication patterns.

### 2.1 Standard Response Wrapper (AIP-193)

**File:** `src/knowledge_base/common/api_models.py`
We will move away from raw Dict responses to a strict Generic wrapper.

```python
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional, Dict, Any

T = TypeVar('T')

class APIError(BaseModel):
    code: str = Field(..., description="Canonical error code (e.g., NOT_FOUND)")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = None

class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[APIError] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

```

### 2.2 Pagination Standard (AIP-158)

**File:** `src/knowledge_base/common/pagination.py`
Standardizes `limit` and `offset` handling across all list endpoints.

```python
class PageParams(BaseModel):
    limit: int = Field(50, ge=1, le=1000, description="Max items to return")
    offset: int = Field(0, ge=0, description="Index to start from")

class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    has_more: bool

```

---

## 3. Graph API Implementation (Phase 2)

**Objective:** Power Sigma.js visualization with hierarchical and temporal graph data.
**File:** `src/knowledge_base/graph_api.py`

### 3.1 Endpoint: Macro Graph Summary

**`GET /api/v1/graph/summary`**
Returns the "Level 0" view: Communities as nodes and inter-community relationships as edges.

* **Logic:**
1. Fetch `Community` records where `level=0`.
2. **Node Generation:** Map Community ID → Node ID. Size = `entity_count`.
3. **Edge Aggregation:**
* SQL Query: Join `Edge` → `Entity` (Source) → `Community` (Source) AND `Entity` (Target) → `Community` (Target).
* Group by Source/Target Community IDs to count weight.




* **Sigma.js Layout:** Backend assigns random initial `x, y` or reads from `Community.metadata` if pre-calculated.

### 3.2 Endpoint: Neighborhood Drill-Down

**`GET /api/v1/graph/neighborhood`**
Returns immediate neighbors for a specific entity ID (on-click expansion).

* **Logic:**
1. **Direct Edges:** SQL Query on `Edge` table where `source_id = {id}` OR `target_id = {id}`.
2. **Filter:** Apply `confidence > 0.7` and `edge_type` filters.
3. **Fetch Nodes:** Retrieve `Entity` details for all discovered neighbor IDs.
4. **Format:** Return pure Node/Edge lists. Frontend (Sigma/Graphology) handles the force-directed layout adjustment.



### 3.3 Endpoint: Temporal Trajectory (2026 Requirement)

**`GET /api/v1/graph/trajectory`**
Visualizes how an entity's relationships evolve over time.

* **Logic:**
1. Input: `entity_ids`, `start_date`, `end_date`.
2. Query `Edge` table checking `temporal_validity_start` and `end`.
3. **Bucketing:** Group valid edges into time buckets (e.g., Monthly).
4. **Events:** Identify "New Edge" or "Invalidated Edge" events within buckets.
5. **Response:** A timeline object containing graph snapshots (deltas) for the frontend time-slider.



---

## 4. Document API Implementation (Phase 3)

**Objective:** Provide "Verbatim Grounding" by linking entities back to exact text spans in source documents.
**File:** `src/knowledge_base/document_api.py`

### 4.1 Endpoint: Document Content & Structure

**`GET /api/v1/documents/{document_id}/content`**

* **Logic:**
1. Verify `Document.status == 'COMPLETED'`.
2. Fetch raw text from `Chunk` table (ordered by `chunk_index`).
3. **Concatenation:** Reassemble full text (if not stored as a blob).
4. **Metadata:** Return domain, extraction stats, and MIME type.



### 4.2 Endpoint: Verbatim Spans (Highlighting)

**`GET /api/v1/documents/{document_id}/spans`**
The critical endpoint for "Click entity -> See text" functionality.

* **Logic:**
1. **Join:** `ChunkEntity` ↔ `Chunk` ↔ `Entity`.
2. **Locate Quote:**
* The `ChunkEntity` table stores `grounding_quote`.
* *Implementation Detail:* Python backend must search for `grounding_quote` inside `Chunk.text` to calculate the exact `start_offset` and `end_offset` relative to the chunk, then offset by the chunk's position in the document.


3. **Response:** List of `TextSpan` objects: `{ start: 105, end: 120, entity_id: "uuid...", type: "Person" }`.



### 4.3 Endpoint: Document Search

**`POST /api/v1/documents/search`**

* **Logic:**
1. **Hybrid Search:**
* **Vector:** Embed query using `Ollama` → Cosine similarity on `Chunk.embedding`.
* **Keyword:** ILIKE query on `Entity.name`.


2. **Rank:** Weighted average of Vector Score (0.7) and Keyword Score (0.3).
3. **Snippet:** Return the `Chunk.text` surrounding the match.



---

## 5. Database Integration Details

Specific SQLAlchemy strategies to support the above.

### Schema Requirements (Verification)

* **Coordinates:** The `Community` table in `database_schema.md` does *not* currently have `x, y` columns. We will store visual coordinates in the `metadata` JSON field or generate them on the fly.
* **Offsets:** `ChunkEntity` has `grounding_quote` but no integer offsets. The **Offset Calculation Service** must be implemented in `document_api.py` to find these quotes at runtime.

### Key SQL Queries

**Aggregating Inter-Community Edges (Graph Summary):**

```sql
SELECT 
    s.community_id as source, 
    t.community_id as target, 
    COUNT(*) as weight
FROM edges e
JOIN entities s ON e.source_id = s.id
JOIN entities t ON e.target_id = t.id
WHERE s.community_id != t.community_id
GROUP BY s.community_id, t.community_id

```

**Verbatim Grounding Lookup:**

```sql
SELECT 
    ce.grounding_quote, 
    c.text as chunk_text, 
    c.chunk_index,
    e.id as entity_id,
    e.entity_type
FROM chunk_entity ce
JOIN chunks c ON ce.chunk_id = c.id
JOIN entities e ON ce.entity_id = e.id
WHERE c.document_id = :doc_id

```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1)

1. Create `src/knowledge_base/common/api_models.py` & `pagination.py`.
2. Implement `GlobalExceptionHandler` in `main.py`.
3. Update `requirements.txt` if specific Pydantic V2 features are needed.

### Phase 2: Visualization (Week 1.5)

1. Implement `graph_api.py`.
2. Build the "Community Aggregation" SQL query.
3. Test with a mocked Sigma.js frontend payload.

### Phase 3: Evidence (Week 2)

1. Implement `document_api.py`.
2. **Crucial:** Implement the text search algorithm to convert `grounding_quote` + `chunk_text` into integer offsets (`start`, `end`) for frontend highlighting.
3. Implement Vector Search integration (connect to `VectorStore` class).

### Phase 4: Integration (Week 2.5)

1. Register routers in `main.py`.
2. Add OpenAPI tags: `["Graph"]`, `["Documents"]`.
3. Generate `openapi.json` and verify strict typing for TypeScript client generation.
This high-level logic guide breaks down the implementation into algorithmic steps. It ensures the code strictly follows the architectural patterns defined in `KBV2 Graph & Document API - Implementation Plan.md` and the database structure in `database_schema.md`.

### 1. Common Infrastructure Logic

**Goal:** Enforce AIP-193 (Standard Responses) and AIP-158 (Pagination) automatically, minimizing boilerplate in individual endpoints.

#### A. The Response Interceptor (Middleware/Decorator)

Instead of manually wrapping every return statement, use a pattern that automatically envelopes data.

* **Logic:**
1. **Input:** Any Pydantic model returned by a router function (e.g., `GraphResponse`).
2. **Process:**
* Check if the result is already an `APIResponse` (skip if so).
* If it's an exception/error, format as `APIError`.
* Otherwise, wrap in `APIResponse(success=True, data=result)`.


3. **Output:** JSON serialized standard response.


* *Benefit:* Keeps endpoint logic clean.



#### B. The Pagination Calculator

**File:** `src/knowledge_base/common/pagination.py`

* **Input:** `query_set` (SQLAlchemy query), `page_params` (limit, offset).
* **Logic:**
1. **Count Total:** Execute `select(func.count()).select_from(...)` matching the filters.
2. **Slice:** Apply `query.limit(limit).offset(offset)`.
3. **Has More?** Simple check: `(offset + len(current_batch)) < total_count`.
4. **Output:** `PaginatedResponse[T]` containing items + metadata.



---

### 2. Graph API Logic (`graph_api.py`)

**Goal:** Efficiently transform relational table rows into graph nodes/edges for visualization.

#### A. Macro Summary (Community Aggregation)

**Endpoint:** `GET /api/v1/graph/summary`
This requires a "Map-Reduce" style operation performed in SQL to avoid fetching thousands of edges.

* **Step 1: Fetch Nodes (Communities)**
* Query `Community` table where `level = 0`.
* *Logic:* Map `community.id` → `Node.key`. Use `entity_count` for `Node.attributes.size`.


* **Step 2: Aggregate Edges (The "Inter-Community" Logic)**
* *Concept:* An edge exists between Community A and Community B if *any* entity in A is connected to *any* entity in B.
* *Query Logic:*
1. **Join:** `Edge` table to `Entity` (as Source) and `Entity` (as Target).
2. **Join:** Source Entity to `Community` (SourceComm) and Target Entity to `Community` (TargetComm).
3. **Filter:** Where `SourceComm.id != TargetComm.id`.
4. **Group By:** `SourceComm.id`, `TargetComm.id`.
5. **Calculate Weight:** `COUNT(*)` of the raw edges.


* *Result:* A lightweight list of weighted links between communities.



#### B. Neighborhood Expansion (Drill-Down)

**Endpoint:** `GET /api/v1/graph/neighborhood`
**Input:** `entity_id`

* **Step 1: Fetch Center Node**
* Get `Entity` by ID. Convert to `GraphNode`.


* **Step 2: Fetch Raw Edges**
* Query `Edge` table where `source_id == entity_id` OR `target_id == entity_id`.
* *Filter:* Apply `confidence_threshold` (default 0.7) to filter out weak links.


* **Step 3: Fetch Neighbors**
* Collect all unique `source_id` and `target_id` values from the edges found in Step 2 (excluding the center ID).
* Batch query `Entity` table for these IDs.


* **Step 4: Layout Preparation (Backend)**
* *Optimization:* While frontend handles physics, the backend should assign an initial `(x, y)` to neighbors in a circle around the center node to prevent visual "explosion" on load.



---

### 3. Document API Logic (`document_api.py`)

**Goal:** Link entities to their exact text evidence ("Verbatim Grounding").

#### A. Content & Span Calculation (The "Offset Service")

**Endpoint:** `GET /api/v1/documents/{doc_id}/spans`
The database stores the `grounding_quote` string, but the UI needs integer offsets (`start: 102`, `end: 115`).

* **Step 1: Load Document Structure**
* Fetch all `Chunk` records for the document, ordered by `chunk_index`.
* Fetch all `ChunkEntity` records joined with `Entity`.


* **Step 2: Reconstruct Text Stream (Virtual)**
* Maintain a running `global_offset` counter.
* Iterate through chunks sequentially.


* **Step 3: Exact String Matching (The core logic)**
* For each Chunk:
* Get `chunk.text`.
* Get associated `grounding_quotes` from `ChunkEntity`.
* *Algorithm:*
1. `local_start = chunk.text.find(quote)`
2. If found:
* `absolute_start = global_offset + local_start`
* `absolute_end = absolute_start + len(quote)`
* Create `TextSpan` object.


3. If not found (data drift/encoding issue):
* Log warning.
* Fallback: Fuzzy match or skip.




* Increment `global_offset += len(chunk.text)`.




* **Step 4: Response**
* Return list of `TextSpan` objects sorted by `start_offset`.



#### B. Hybrid Search Logic

**Endpoint:** `POST /api/v1/documents/search`

* **Step 1: Vector Search (Semantic)**
* Input: `query` string.
* Action: Call `Ollama` to get embedding vector.
* Query: `VectorStore` (pgvector) on `Chunk` table for nearest neighbors (L2 distance/Cosine).
* *Weight:* 0.7 importance.


* **Step 2: Keyword Search (Exact)**
* Query: `Entity` table `WHERE name ILIKE %query%`.
* *Weight:* 0.3 importance.


* **Step 3: Merge & Rank**
* Normalize scores from both sources.
* If a Chunk matches via Vector, it's a hit.
* If an Entity matches via Keyword, find its most relevant `Chunk` (highest confidence in `ChunkEntity`) and use that as the hit.
* Return unified `SearchResult` list.



---

### 4. Database Interaction Logic (Persistence Layer)

#### Repository Pattern Extension

Extend `src/knowledge_base/persistence/v1/vector_store.py` (or create `graph_store.py`) to encapsulate the complex joins.

* **New Method: `get_community_topology(level: int)**`
* Encapsulates the raw SQL for the "Map-Reduce" edge aggregation described in section 2A.
* This prevents leaking raw SQL into the API router layer.


* **New Method: `get_entity_grounding(doc_id: UUID)**`
* Optimized join of `Chunk` + `ChunkEntity` + `Entity` to fetch all data needed for the Offset Service in one round-trip.



### 5. Summary of Dependencies

* **Frontend Requirements:** Sigma.js needs nodes/edges; Evidence viewer needs integer offsets.
* **Database:** `Community` table provides grouping; `ChunkEntity` provides the text quotes.
* **API Standards:** All endpoints return `APIResponse` and use `PaginatedResponse` where lists are involved.
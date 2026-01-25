This complete knowledge base unifies **Google’s API Improvement Proposals (AIPs)** with advanced **Graph Theory** and **Document Intelligence** patterns. It serves as a Single Source of Truth (SSOT) for architects and developers building complex, evidence-based systems.

---

# API Design Patterns: The Resource-Oriented Graph Standard

**Version:** 1.0.0 | **Philosophy:** Resource-Oriented, Graph-Aware, Evidence-Based

## 1. Core Architectural Standards (Google AIPs)

The foundation of this architecture is **Resource-Oriented Design**, where the API is modeled as a hierarchy of resources (nouns) manipulated by a small set of standard methods (verbs).

### 1.1 Resource Names & Hierarchy

Every entity is a resource with a unique, globally addressable name.

* **Format:** `collections/{collection_id}/resources/{resource_id}`
* **Pattern:**
* **Nodes:** `graphs/{graph}/nodes/{node}`
* **Edges:** `graphs/{graph}/edges/{edge}`
* **Documents:** `publishers/{publisher}/documents/{document}`



> **Critical Rule:** Do not use database IDs (integers) as the primary public identifier if possible. Use string-based, client-assigned IDs where feasible to ensure idempotency.

### 1.2 Standard Methods (CRUD)

All resources must implement the standard lifecycle methods.

| Method | HTTP Verb | URL Pattern | Body | Success Status | AIP Ref |
| --- | --- | --- | --- | --- | --- |
| **List** | `GET` | `/v1/{parent}/{collection}` | None | 200 OK | AIP-132 |
| **Get** | `GET` | `/v1/{name}` | None | 200 OK | AIP-131 |
| **Create** | `POST` | `/v1/{parent}/{collection}` | Resource JSON | 201 Created | AIP-133 |
| **Update** | `PATCH` | `/v1/{name}` | Resource + `update_mask` | 200 OK | AIP-134 |
| **Delete** | `DELETE` | `/v1/{name}` | None | 204 No Content | AIP-135 |

### 1.3 Custom Methods

For actions that do not map to CRUD (e.g., graph traversal, document analysis), use Custom Methods with the `:verb` syntax.

* **Pattern:** `POST /v1/{resource}:customVerb`
* **Examples:**
* `POST /v1/graphs/123:export` (Export graph)
* `POST /v1/documents/abc:analyze` (Run OCR/NLP)



---

## 2. Graph API Patterns

Bridging REST constraints with Graph Theory requirements (traversal, depth, connectivity).

### 2.1 The Neighborhood Pattern (Traversal)

Avoid "chatty" APIs where clients must fetch a node, then fetch its edges, then fetch the target nodes. Use a **Custom Method** to retrieve a subgraph in one request.

**Endpoint:** `GET /v1/graphs/{graph}/nodes/{node}:expand`

**Query Parameters:**

* `depth` (int): Number of hops (default: 1).
* `filter` (string): AIP-160 filter for edge/node inclusion (e.g., `edge.type="cites"`).

**Response (Adjacency List + Node Map):**

```json
{
  "central_node": "graphs/1/nodes/A",
  "nodes": [
    {"name": "graphs/1/nodes/B", "attributes": {"label": "Witness"}},
    {"name": "graphs/1/nodes/C", "attributes": {"label": "Suspect"}}
  ],
  "edges": [
    {"source": "graphs/1/nodes/A", "target": "graphs/1/nodes/B", "type": "interviews"},
    {"source": "graphs/1/nodes/B", "target": "graphs/1/nodes/C", "type": "identifies"}
  ]
}

```

### 2.2 Path Finding (Analytics)

Complex algorithms run on the server to avoid moving massive datasets to the client.

**Endpoint:** `POST /v1/graphs/{graph}:findPath`

**Request Body:**

```json
{
  "source": "nodes/123",
  "target": "nodes/456",
  "algorithm": "SHORTEST_PATH", // or ALL_SIMPLE_PATHS
  "max_hops": 5,
  "cost_property": "weight"
}

```

### 2.3 Bitemporal Graph Data

Handle both **Valid Time** (when the event happened in real life) and **Transaction Time** (when the data was recorded).

* **Pattern:** Append time dimensions to the resource schema.
* **Querying:** `GET /v1/graphs/1/nodes?valid_time=2024-01-01&view=AS_OF_RECORDED_TIME`

**Resource Schema:**

```json
{
  "name": "graphs/1/nodes/A",
  "valid_interval": {
    "start_time": "2023-01-01T00:00:00Z",
    "end_time": "2023-12-31T23:59:59Z"
  },
  "transaction_time": "2023-01-05T10:00:00Z"
}

```

---

## 3. Document Intelligence & Evidence Patterns

Patterns for connecting raw unstructured documents to structured graph nodes.

### 3.1 Hierarchical Extraction

**Resource Path:** `documents/{document}/pages/{page}/blocks/{block}`

### 3.2 W3C Web Annotation Compliance

To anchor evidence (excerpts) reliably, use the **W3C Web Annotation Data Model**. This supports both text offsets (for reliable text) and visual bounding boxes (for scanned PDFs).

**Resource:** `documents/{document}/excerpts/{excerpt}`

**Schema:**

```json
{
  "name": "documents/doc-1/excerpts/ex-1",
  "target": {
    "source": "documents/doc-1",
    "selector": [
      {
        "type": "TextQuoteSelector",
        "exact": "The suspect entered the building.",
        "prefix": "At 10:00 AM, ",
        "suffix": " He was wearing a red hat."
      },
      {
        "type": "TextPositionSelector",
        "start": 1450,
        "end": 1481
      }
    ]
  },
  "body": {
    "label": "Evidence",
    "confidence": 0.98
  }
}

```

---

## 4. Frontend Integration: Sigma.js & Graphology

To visualize the graph, the API must support the **Graphology** JSON specification.

### 4.1 The Export Endpoint

**Endpoint:** `GET /v1/graphs/{graph}:export?format=graphology`

**Response Structure (Strict Graphology Compliance):**

```json
{
  "attributes": {"name": "Investigation A"},
  "options": {"type": "directed", "multi": true},
  "nodes": [
    {
      "key": "n1",
      "attributes": {
        "label": "Person A",
        "x": 0, "y": 0, "size": 10, "color": "#FF0000", // Visual attributes
        "community": 1
      }
    }
  ],
  "edges": [
    {
      "key": "e1",
      "source": "n1",
      "target": "n2",
      "attributes": {
        "size": 2,
        "label": "KNOWS"
      }
    }
  ]
}

```

### 4.2 Server-Side Layouts

For large graphs (>1000 nodes), calculating layout (ForceAtlas2) on the client is slow.

* **Strategy:** Compute `x, y` coordinates on the server (using Python `networkx` or C++ libraries).
* **API Flag:** `?include_layout=true` in the export or expand requests.

---

## 5. Advanced Querying (AIP-160)

Do not invent a new filtering language. Use the standard AIP-160 syntax.

**Syntax:** `field operator value AND/OR field operator value`

| Operator | Meaning | Example |
| --- | --- | --- |
| `=` | Equals | `type = "person"` |
| `!=` | Not Equals | `status != "archived"` |
| `>` / `<` | Comparison | `confidence_score > 0.8` |
| `:` | Has / Contains | `title : "report"` (matches "Annual Report") |
| `AND` / `OR` | Logic | `type="person" AND age > 21` |
| `.` | Traversal | `metadata.source = "FBI"` |

**Example URL:**
`GET /v1/graphs/1/nodes?filter="type='person' AND metadata.risk_score > 80"`

---

## 6. Operational Excellence

### 6.1 Concurrency Control (AIP-154)

Graph edits are prone to race conditions.

* **Response Header:** `ETag: "v1-strong-hash"`
* **Update Request Header:** `If-Match: "v1-strong-hash"`
* **Error:** If hashes mismatch, return `412 Precondition Failed`.

### 6.2 Error Handling (AIP-193)

Return structured errors, not just text.

```json
{
  "error": {
    "code": 404,
    "message": "Node 'n1' not found in Graph 'g1'",
    "status": "NOT_FOUND",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "RESOURCE_NOT_FOUND",
        "domain": "graph.service",
        "metadata": {"resource": "n1", "parent": "g1"}
      }
    ]
  }
}

```

---

## 7. Implementation Checklist (Next Steps)

1. **Define Open API Spec:** Write the Swagger/OpenAPI 3.0 definition for `graphs`, `nodes`, and `edges` using the patterns above.
2. **Select Layout Engine:** Decide if server-side layout (e.g., Graphviz, Gephi toolkit) is required for the `:export` endpoint.
3. **Implement AIP-160 Parser:** Choose a library (like `google-api-core` or a simple ANTLR grammar) to safely parse filter strings.
4. **Validate W3C Selectors:** Ensure the frontend text selection tool produces valid W3C `TextQuoteSelector` JSON.

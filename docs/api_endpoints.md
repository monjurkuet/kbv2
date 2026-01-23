# KBV2 API Endpoints Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            API ENDPOINTS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

FastAPI Application
│
├─ Query API (query_api.py)
│  └─ Base Path: /api/v1/query
│     ├─ POST /translate
│     │  └─ Request: {query: string}
│     │  └─ Response: {sql: string, warnings: string[]}
│     │
│     ├─ POST /execute
│     │  └─ Request: {query: string}
│     │  └─ Response: {sql: string, results: any[], warnings: string[]}
│     │
│     └─ GET /schema
│        └─ Response: {tables: TableInfo[]}
│
├─ Review API (review_api.py)
│  └─ Base Path: /api/v1/review
│     ├─ GET /pending
│     │  └─ Query: ?limit=10&offset=0
│     │  └─ Response: {reviews: ReviewItem[]}
│     │
│     ├─ GET /{review_id}
│     │  └─ Response: ReviewItem
│     │
│     ├─ POST /{review_id}/approve
│     │  └─ Request: {notes: string}
│     │  └─ Response: {success: boolean}
│     │
│     └─ POST /{review_id}/reject
│        └─ Request: {corrections: any[]}
│        └─ Response: {success: boolean}
│
└─ MCP Server (mcp_server.py)
   └─ WebSocket: /ws
      └─ JSON-RPC Methods:
         ├─ kbv2/ingest_document
         │  └─ Request: {file_path: string, domain?: string}
         │  └─ Response: {document_id: UUID, status: string}
         │
         ├─ kbv2/query_text_to_sql
         │  └─ Request: {query: string}
         │  └─ Response: {sql: string, results: any[]}
         │
         ├─ kbv2/search_entities
         │  └─ Request: {query: string, limit?: number}
         │  └─ Response: {entities: Entity[]}
         │
         ├─ kbv2/search_chunks
         │  └─ Request: {query: string, limit?: number}
         │  └─ Response: {chunks: Chunk[]}
         │
         └─ kbv2/get_document_status
            └─ Request: {document_id: UUID}
            └─ Response: {status: string, progress: number}
```

## Query API Details

### POST /api/v1/query/translate
Translates natural language query to SQL without execution.

**Request:**
```json
{
  "query": "Show me all entities in the technology domain"
}
```

**Response:**
```json
{
  "sql": "SELECT * FROM entities WHERE domain = 'technology' LIMIT 1000",
  "warnings": ["Query limited to 1000 rows for security"]
}
```

**Security Features:**
- SQL injection prevention
- Schema validation
- Statement timeout (5s)
- Result limit (1000 rows)

### POST /api/v1/query/execute
Translates and executes natural language query.

**Request:**
```json
{
  "query": "How many entities are in the technology domain?"
}
```

**Response:**
```json
{
  "sql": "SELECT COUNT(*) FROM entities WHERE domain = 'technology'",
  "results": [{"count": 42}],
  "warnings": []
}
```

**Security Features:**
- All translate security features
- Execution-time validation
- Row limit enforcement

### GET /api/v1/query/schema
Returns database schema information.

**Response:**
```json
{
  "tables": [
    {
      "name": "documents",
      "columns": [
        {"name": "id", "type": "UUID"},
        {"name": "name", "type": "VARCHAR(500)"},
        {"name": "status", "type": "VARCHAR(50)"}
      ]
    },
    {
      "name": "entities",
      "columns": [
        {"name": "id", "type": "UUID"},
        {"name": "name", "type": "VARCHAR(500)"},
        {"name": "entity_type", "type": "VARCHAR(100)"}
      ]
    }
  ]
}
```

## Review API Details

### GET /api/v1/review/pending
Get pending review items ordered by priority.

**Query Parameters:**
- `limit`: Number of items to return (default: 10)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "reviews": [
    {
      "id": "uuid-1",
      "item_type": "entity_resolution",
      "entity_id": "uuid-2",
      "confidence_score": 0.65,
      "grounding_quote": "Dr. Elena Vance...",
      "status": "PENDING",
      "priority": 8,
      "created_at": "2024-01-23T00:00:00Z"
    }
  ]
}
```

### GET /api/v1/review/{review_id}
Get specific review item details.

**Response:**
```json
{
  "id": "uuid-1",
  "item_type": "entity_resolution",
  "entity_id": "uuid-2",
  "merged_entity_ids": ["uuid-3", "uuid-4"],
  "confidence_score": 0.65,
  "grounding_quote": "Dr. Elena Vance, who was appointed as the lead",
  "source_text": "Full source text...",
  "status": "PENDING",
  "priority": 8,
  "created_at": "2024-01-23T00:00:00Z"
}
```

### POST /api/v1/review/{review_id}/approve
Approve a review item.

**Request:**
```json
{
  "notes": "Confirmed merge based on quote"
}
```

**Response:**
```json
{
  "success": true,
  "review_id": "uuid-1",
  "status": "APPROVED"
}
```

**Action:**
- Marks review as APPROVED
- Applies entity merge (if applicable)
- Records reviewer notes

### POST /api/v1/review/{review_id}/reject
Reject a review item with corrections.

**Request:**
```json
{
  "corrections": [
    {
      "entity_id": "uuid-2",
      "action": "keep_separate",
      "reason": "Insufficient evidence for merge"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "review_id": "uuid-1",
  "status": "REJECTED"
}
```

**Action:**
- Marks review as REJECTED
- Applies corrections
- Records rejection reason

## MCP Server Details

### WebSocket Endpoint: /ws

The MCP server uses JSON-RPC over WebSocket for bidirectional communication.

### kbv2/ingest_document
Ingest a document through the orchestrator.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "kbv2/ingest_document",
  "params": {
    "file_path": "/path/to/document.pdf",
    "domain": "technology"
  },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "document_id": "uuid-123",
    "status": "PENDING",
    "message": "Document ingestion started"
  },
  "id": 1
}
```

### kbv2/query_text_to_sql
Execute a text-to-SQL query.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "kbv2/query_text_to_sql",
  "params": {
    "query": "Show me all entities"
  },
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "sql": "SELECT * FROM entities LIMIT 1000",
    "results": [
      {"id": "uuid-1", "name": "Entity 1", "entity_type": "Person"},
      {"id": "uuid-2", "name": "Entity 2", "entity_type": "Organization"}
    ],
    "warnings": []
  },
  "id": 2
}
```

### kbv2/search_entities
Search for entities using vector similarity.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "kbv2/search_entities",
  "params": {
    "query": "project lead",
    "limit": 5
  },
  "id": 3
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "entities": [
      {
        "id": "uuid-1",
        "name": "Elena Vance",
        "entity_type": "Person",
        "description": "Project lead",
        "similarity": 0.92
      }
    ]
  },
  "id": 3
}
```

### kbv2/search_chunks
Search for document chunks using vector similarity.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "kbv2/search_chunks",
  "params": {
    "query": "project initiation",
    "limit": 3
  },
  "id": 4
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "chunks": [
      {
        "id": "uuid-10",
        "document_id": "uuid-100",
        "text": "Project Nova was initiated in August 2021...",
        "similarity": 0.88
      }
    ]
  },
  "id": 4
}
```

### kbv2/get_document_status
Get processing status of a document.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "kbv2/get_document_status",
  "params": {
    "document_id": "uuid-100"
  },
  "id": 5
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "document_id": "uuid-100",
    "status": "COMPLETED",
    "progress": 100,
    "stages": {
      "partition": "completed",
      "extract": "completed",
      "embed": "completed",
      "resolve": "completed",
      "cluster": "completed",
      "report": "completed"
    }
  },
  "id": 5
}
```

## Security Considerations

### Query API
- SQL injection prevention via pattern matching
- Schema validation for table/column names
- Query timeout (5 seconds)
- Result limits (1000 rows max)
- Blocks dangerous SQL keywords (DROP, DELETE, etc.)

### Review API
- Authentication required (if configured)
- Authorization checks for review actions
- Audit logging for all review operations

### MCP Server
- WebSocket authentication (if configured)
- Rate limiting per connection
- Input validation for all parameters
- Error handling without exposing internal details

## Error Responses

All endpoints follow standard error response format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid query format",
    "details": "Query must be a non-empty string"
  }
}
```

Common error codes:
- `INVALID_REQUEST`: Malformed request
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `INTERNAL_ERROR`: Server error
- `SQL_INJECTION_DETECTED`: Potentially dangerous SQL
# KBV2 Enhancement Implementation Plan

Implementation plan for the three identified gaps: Natural Language Query Interface, Domain Tagging, and Human Review UI.

---

## Proposed Changes

### Feature 1: Natural Language Query Interface

A Text-to-SQL agent that allows users to query the knowledge graph using natural language.

---

#### [NEW] [text_to_sql_agent.py](file:///home/administrator/dev/kbv2/src/knowledge_base/query/v1/text_to_sql_agent.py)

**Purpose**: Translate natural language questions into SQL queries and execute against PostgreSQL.

**Architecture**:
```
User Query → Schema Context Builder → LLM (SQL Generation) → SQL Validation → Execution → Result Formatting
```

**Key Classes**:
```python
class QueryConfig(BaseSettings):
    max_results: int = 100
    timeout_seconds: float = 30.0
    allow_writes: bool = False  # Safety: read-only by default

class TextToSQLAgent:
    def __init__(self, gateway: GatewayClient, vector_store: VectorStore)
    
    async def query(self, natural_language: str, domain: str | None = None) -> QueryResult:
        """Main entry point for NL queries."""
        
    async def _build_schema_context(self) -> str:
        """Generate schema description for LLM context."""
        
    async def _generate_sql(self, question: str, schema_context: str) -> str:
        """Use LLM to generate SQL from question."""
        
    async def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Validate SQL is safe (no writes, injections)."""
        
    async def _execute_query(self, sql: str) -> list[dict]:
        """Execute validated SQL against PostgreSQL."""
        
    async def _format_results(self, results: list[dict], question: str) -> str:
        """Optionally use LLM to format results as natural language."""
```

**LLM System Prompt**:
```
You are a SQL expert. Given the database schema and a user question, 
generate a PostgreSQL query to answer it.

Rules:
1. Use only SELECT statements (no INSERT, UPDATE, DELETE)
2. Use explicit column names, not SELECT *
3. Include LIMIT clause (max 100)
4. Use proper JOINs for relationships
5. Handle entity/edge traversal using source_id/target_id
```

---

#### [NEW] [query_api.py](file:///home/administrator/dev/kbv2/src/knowledge_base/api/v1/query_api.py)

**Purpose**: FastAPI endpoints for the query interface.

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/query", tags=["query"])

class QueryRequest(BaseModel):
    question: str
    domain: str | None = None
    format: Literal["raw", "natural"] = "natural"

class QueryResponse(BaseModel):
    question: str
    sql: str
    results: list[dict]
    formatted_answer: str | None
    execution_time_ms: float

@router.post("/", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Query the knowledge base using natural language."""
```

---

### Feature 2: Domain Tagging System

Add explicit domain classification to enable multi-tenant or multi-domain knowledge bases.

---

#### [MODIFY] [schema.py](file:///home/administrator/dev/kbv2/src/knowledge_base/persistence/v1/schema.py)

**Changes**:
```diff
class Entity(Base):
    __tablename__ = "entities"
    # ... existing fields ...
+   domain = Column(String(100), nullable=True, index=True)

class Edge(Base):
    __tablename__ = "edges"
    # ... existing fields ...
+   domain = Column(String(100), nullable=True, index=True)

class Document(Base):
    __tablename__ = "documents"
    # ... existing fields ...
+   domain = Column(String(100), nullable=True, index=True)
```

---

#### [NEW] [migrations/add_domain_columns.sql](file:///home/administrator/dev/kbv2/scripts/migrations/add_domain_columns.sql)

```sql
-- Add domain columns to existing tables
ALTER TABLE documents ADD COLUMN IF NOT EXISTS domain VARCHAR(100);
ALTER TABLE entities ADD COLUMN IF NOT EXISTS domain VARCHAR(100);
ALTER TABLE edges ADD COLUMN IF NOT EXISTS domain VARCHAR(100);

-- Create indexes for domain filtering
CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_entities_domain ON entities(domain);
CREATE INDEX IF NOT EXISTS idx_edges_domain ON edges(domain);
```

---

#### [MODIFY] [orchestrator.py](file:///home/administrator/dev/kbv2/src/knowledge_base/orchestrator.py)

**Changes to [process_document](file:///home/administrator/dev/kbv2/src/knowledge_base/orchestrator.py#66-124)**:
```diff
async def process_document(
    self,
    file_path: str | Path,
    document_name: str | None = None,
+   domain: str | None = None,
) -> Document:
```

**Propagate domain to entities/edges during creation**.

---

### Feature 3: Human Review Interface

API and workflow for reviewing low-confidence entity resolutions.

---

#### [NEW] [review_service.py](file:///home/administrator/dev/kbv2/src/knowledge_base/intelligence/v1/review_service.py)

**Purpose**: Manage the queue of items needing human review.

```python
class ReviewItem(BaseModel):
    id: UUID
    item_type: Literal["entity_resolution", "edge_validation"]
    entity_id: UUID
    merged_entity_ids: list[UUID]
    confidence_score: float
    grounding_quote: str
    source_text: str
    created_at: datetime
    status: Literal["pending", "approved", "rejected"]

class ReviewService:
    async def get_pending_reviews(
        self, 
        limit: int = 50,
        min_confidence: float = 0.0,
        max_confidence: float = 0.7,
    ) -> list[ReviewItem]:
        """Get items flagged for human review."""
        
    async def approve_resolution(self, review_id: UUID) -> bool:
        """Approve an entity merge."""
        
    async def reject_resolution(self, review_id: UUID) -> bool:
        """Reject and undo an entity merge."""
        
    async def get_review_stats(self) -> dict:
        """Get summary stats on review queue."""
```

---

#### [NEW] [review_api.py](file:///home/administrator/dev/kbv2/src/knowledge_base/api/v1/review_api.py)

```python
router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])

@router.get("/pending")
async def get_pending_reviews(limit: int = 50) -> list[ReviewItem]:
    """Get pending review items."""

@router.post("/{review_id}/approve")
async def approve_review(review_id: UUID) -> dict:
    """Approve a resolution."""

@router.post("/{review_id}/reject")  
async def reject_review(review_id: UUID) -> dict:
    """Reject a resolution."""

@router.get("/stats")
async def get_review_stats() -> dict:
    """Get review queue statistics."""
```

---

#### [MODIFY] [schema.py](file:///home/administrator/dev/kbv2/src/knowledge_base/persistence/v1/schema.py)

**Add review tracking table**:
```python
class ReviewQueue(Base):
    __tablename__ = "review_queue"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    item_type = Column(String(50), nullable=False)
    entity_id = Column(PG_UUID(as_uuid=True), ForeignKey("entities.id"))
    merged_entity_ids = Column(JSON)
    confidence_score = Column(Float)
    grounding_quote = Column(Text)
    source_text = Column(Text)
    status = Column(String(20), default="pending")
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

### Feature 4: Optional MCP Protocol Layer

Standard MCP interface for external tool integrations.

---

#### [NEW] [mcp_server.py](file:///home/administrator/dev/kbv2/src/knowledge_base/mcp/server.py)

**MCP Tool Definitions**:
```python
MCP_TOOLS = [
    {
        "name": "query_knowledge_base",
        "description": "Query the knowledge graph using natural language",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "domain": {"type": "string"}
            },
            "required": ["question"]
        }
    },
    {
        "name": "get_entity",
        "description": "Get entity details by name or ID",
        "input_schema": {...}
    },
    {
        "name": "get_related_entities",
        "description": "Get entities related to a given entity",
        "input_schema": {...}
    },
    {
        "name": "search_entities",
        "description": "Semantic search for entities",
        "input_schema": {...}
    }
]
```

---

## Directory Structure After Implementation

```
src/knowledge_base/
├── api/                          # [NEW] FastAPI layer
│   └── v1/
│       ├── __init__.py
│       ├── query_api.py          # NL query endpoints
│       └── review_api.py         # Human review endpoints
├── query/                        # [NEW] Query layer
│   └── v1/
│       ├── __init__.py
│       └── text_to_sql_agent.py  # Text-to-SQL agent
├── mcp/                          # [NEW] Optional MCP
│   ├── __init__.py
│   └── server.py                 # MCP server
├── intelligence/
│   └── v1/
│       ├── review_service.py     # [NEW] Review workflow
│       └── ... (existing)
├── persistence/
│   └── v1/
│       ├── schema.py             # [MODIFIED] + ReviewQueue, domain
│       └── ... (existing)
└── orchestrator.py               # [MODIFIED] + domain param
```

---

## Verification Plan

### Automated Tests

```bash
# Unit tests for Text-to-SQL
uv run pytest tests/test_text_to_sql.py -v

# Integration tests for query API
uv run pytest tests/test_query_api.py -v

# Review workflow tests
uv run pytest tests/test_review_service.py -v
```

### Manual Verification

1. **Query Interface**: Test queries like "What entities are related to Project Phoenix?"
2. **Domain Filtering**: Ingest documents with domain tags, verify filtering
3. **Review UI**: Create low-confidence resolutions, verify they appear in queue

---

## Implementation Timeline

| Phase | Feature | Effort | Priority |
|-------|---------|--------|----------|
| 1 | Domain Tagging | 2-3 days | High (foundation) |
| 2 | Text-to-SQL Agent | 1 week | High (main value) |
| 3 | Human Review API | 3-4 days | Medium |
| 4 | MCP Protocol | 1 week | Low (optional) |

**Total**: ~3 weeks for core features, +1 week for MCP if desired.

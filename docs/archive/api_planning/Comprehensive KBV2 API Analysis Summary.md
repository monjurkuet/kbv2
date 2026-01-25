Comprehensive KBV2 API Analysis Summary
Based on my exploration of the KBV2 codebase, here's a detailed analysis of existing patterns and structures:
1. Current API Structure
Query API (src/knowledge_base/query_api.py)
- Endpoints: /translate, /execute, /schema
- Response Pattern: response_model=Dict (primitive type)
- Error Handling: Generic try/except returning error strings in response
- Design: Simple functional API without structured response models
- Prefix: /api/v1/query
- Tags: ["query"]
Review API (src/knowledge_base/review_api.py) 
- Endpoints: /pending, /{review_id}, /{review_id}/approve, /{review_id}/reject
- Response Pattern: Proper Pydantic models (ReviewItem, ReviewApproval, ReviewRejection)
- Error Handling: HTTPException with proper status codes
- Design: Full RESTful patterns with CRUD operations
- Prefix: /api/v1/review
- Tags: ["review"]
- Pagination: limit and offset parameters with default values
- Filtering: Status-based filtering (pending reviews only)
MCP Server (src/knowledge_base/mcp_server.py)
- Protocol: WebSocket-based Model Context Protocol
- Response Pattern: Custom MCPRequest/MCPResponse models
- Error Handling: Structured error field in response objects
- Design: RPC-style protocol over WebSocket
- Prefix: /ws (WebSocket endpoint)
- Tag: Not applicable (not REST)
2. Schema Models Analysis
Core Entities (src/knowledge_base/persistence/v1/schema.py)
# Primary Models
- Document: UUID, name, source_uri, status, metadata, domain
- Chunk: UUID, document_id, text, chunk_index, embedding (Vector)
- Entity: UUID, name, type, description, properties JSON, confidence, embedding
- Edge: UUID, source_id, target_id, edge_type, properties, temporal_validity
- Community: UUID, name, level, resolution, summary, entity_count, hierarchy
- ReviewQueue: UUID, item_type, entity_id/edge_id, status, priority, confidence_score
Key Characteristics
- Primary Keys: All use UUID with uuid4 default
- Timestamps: created_at (auto), updated_at (auto + onupdate)
- Indexes: Strategic indexes on foreign keys and queryable fields
- Embeddings: 768-dimension vectors using custom Vector type
- JSON Fields: Flexible property storage via PostgreSQL JSON
- Relationships: Proper SQLAlchemy relationships with back_populates
- Domains: Optional domain field for multi-tenant/data isolation
3. Common API Patterns
Request/Response Patterns
# Inconsistent patterns found:
1. Primitive Dict responses (query_api)
2. Pydantic model responses (review_api)
3. Boolean success/failure flags
4. Custom MCP protocol responses
5. List responses without pagination wrappers
Parameter Patterns
# Standard query parameters:
- limit: int = 50          # Pagination
- offset: int = 0          # Pagination offset
- domain: Optional[str]    # Domain filtering
- Query parameters in MCP: params.get("key", default)
Error Handling
# Mixed patterns:
1. HTTPException(status_code, detail) - Proper REST errors
2. Return {"error": "message", ...} - Soft errors in response body
3. Exception catching with generic error messages
4. No global exception handlers found
Database Patterns
# Consistent patterns:
1. Dependency injection with get_db() function
2. Session management with try/finally blocks
3. Proper commit/refresh patterns
4. SQLAlchemy 2.0+ select() syntax usage
4. Technology Stack
Core Framework
- FastAPI: v0.104.1 - v0.128.0 (from requirements)
- SQLAlchemy: v2.0.23+ (modern async/sync support)
- Pydantic: v2.5.0+ (v2 features available)
- Uvicorn: ASGI server
- PostgreSQL: With pgvector extension
Notable Features
- Type Hints: Comprehensive throughout codebase
- Async Support: Mixed async/await and synchronous code
- Dependencies: Well-structured dependency injection
- Settings: Uses pydantic-settings for configuration
- Testing: pytest with async support
Missing Standardizations
- ❌ No CORS configuration found
- ❌ No middleware implementations
- ❌ No global exception handlers
- ❌ No OpenAPI/Swagger customization
- ❌ No TypeScript client generation
- ❌ No unified response wrappers
- ❌ No pagination helper classes
5. Opportunities for Standardization
Response Structure Alignment
# Current Inconsistencies:
- query_api.py: {"sql": "...", "warnings": [], "error": None}
- review_api.py: Proper Pydantic models
- mcp_server.py: {"result": ..., "error": None, "id": "..."}
# Recommendation: Standardize on Pydantic models
Error Response Standardization
# Create standardized error response:
class APIError(BaseModel):
    detail: str
    status_code: int
    error_code: Optional[str] = None
# And success wrapper:
class APIResponse(BaseModel):
    data: Optional[T] = None
    error: Optional[APIError] = None
    metadata: Optional[Dict] = None
Pagination Patterns
# Currently inconsistent - create standard:
class PageParams(BaseModel):
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
class PageResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool
6. Recommendations for Graph/Document APIs
Align with Review API Patterns
- ✅ Use Pydantic models for request/response (like ReviewItem)
- ✅ HTTPException for error handling (consistent with review_api)
- ✅  /api/v1 prefix with resource-specific segments
- ✅ Proper tags for OpenAPI grouping
- ✅  Limit/offset pagination with sensible defaults
Follow Schema Conventions
- ✅ UUID primary keys with uuid4 defaults
- ✅ created_at/updated_at timestamp patterns
- ✅ Flexible JSON properties for extensibility
- ✅ Optional domain fields for multi-tenancy
- ✅ Confidence scores on extracted entities/edges
Consistent Endpoints Structure
# Recommended structure:
GET    /api/v1/{resource}              # List with pagination
GET    /api/v1/{resource}/{id}         # Get by ID
POST   /api/v1/{resource}              # Create
PUT    /api/v1/{resource}/{id}         # Update
DELETE /api/v1/{resource}/{id}         # Delete
POST   /api/v1/{resource}/search       # Semantic search
Response Consistency
# Standard response format for list endpoints:
{
  "items": [...],
  "total": 100,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
# Standard error format:
{
  "error": {
    "detail": "Resource not found",
    "status_code": 404,
    "error_code": "NOT_FOUND"
  }
}
The codebase shows a clear evolution from simple functional endpoints to proper REST API patterns. The graph and document APIs should follow the review_api.py patterns as the most mature and consistent implementation while adopting the database schema conventions from persistence/v1/schema.py.
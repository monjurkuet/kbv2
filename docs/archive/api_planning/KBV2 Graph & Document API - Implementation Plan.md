KBV2 Graph & Document API - Implementation Plan
Executive Summary
This plan expands the KBV2 system with two new API routers (graph_api.py and document_api.py) to support frontend graph visualization (Sigma.js) and document evidence viewing, following Google API Improvement Proposals (AIPs) and maintaining consistency with existing KBV2 patterns.
1. API Standards & Conventions
1.1 Google AIP Compliance
Following Google Cloud API Design Guide:
- AIP-121: Resource-oriented design
- AIP-122: Hierarchical resource names
- AIP-130-135: Standard methods (Get, List, Create, Update, Delete)
- AIP-136: Custom methods with :verb syntax
- AIP-148: Standard fields (name, display_name, create_time)
- AIP-158: Pagination with page_size and page_token
- AIP-160: Filtering syntax
- AIP-193: Standardized error responses
1.2 KBV2 Pattern Consistency
- Response Models: Pydantic-based (like review_api.py)
- Error Handling: HTTPException with proper status codes
- Pagination: limit/offset pattern (consistent with review API)
- Tags: OpenAPI tags for documentation grouping
- Path Structure: /api/v1/{resource}/...
1.3 URL Structure
/api/v1
├── /graph
│   ├── /summary
│   ├── /neighborhood
│   ├── /trajectory
│   └── /communities
├── /documents
│   ├── /{document_id}
│   │   ├── /content
│   │   ├── /spans
│   │   └── /entities
│   ├── /search
│   └── /batch
└── /query (existing)
└── /review (existing)
2. Graph Visualization API (graph_api.py)
2.1 Resource Model
# src/knowledge_base/common/api_models.py (NEW)
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List, Optional, Dict, Any
from datetime import datetime
T = TypeVar('T')
class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper following Google AIP-193 patterns."""
    success: bool = Field(..., description="Indicates if request succeeded")
    data: Optional[T] = Field(None, description="Response payload")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Pagination, timing, etc.")
class GraphNode(BaseModel):
    """Sigma.js compatible node representation."""
    key: str = Field(..., description="Unique node identifier (UUID)")
    label: str = Field(..., description="Display label")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sigma.js attributes: x, y, size, color, community, etc."
    )
class GraphEdge(BaseModel):
    """Sigma.js compatible edge representation."""
    key: str = Field(..., description="Unique edge identifier (UUID)")
    source: str = Field(..., description="Source node key")
    target: str = Field(..., description="Target node key")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Edge attributes: label, weight, type, color, etc."
    )
class GraphResponse(BaseModel):
    """Complete graph data for Sigma.js."""
    nodes: List[GraphNode] = Field(..., description="Graph nodes")
    edges: List[GraphEdge] = Field(..., description="Graph edges")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Graph-level metadata: bounds, stats, timestamps"
    )
class NeighborhoodQuery(BaseModel):
    """Parameters for neighborhood traversal."""
    entity_id: str = Field(..., description="Center entity UUID")
    depth: int = Field(1, ge=1, le=3, description="Traversal depth")
    direction: str = Field(
        "bidirectional",
        pattern="^(outgoing|incoming|bidirectional)$",
        description="Edge direction to follow"
    )
    node_types: Optional[List[str]] = Field(
        None,
        description="Filter by entity types"
    )
    edge_types: Optional[List[str]] = Field(
        None,
        description="Filter by relationship types"
    )
    min_confidence: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
2.2 Endpoints Specification
A. GET /api/v1/graph/summary
Purpose: High-level graph overview for initial "zoomed out" view (Macro communities only)
Query Parameters:
domain: Optional[str] = None  # Filter by domain
min_community_size: int = 3    # Minimum entities per community
level: int = 0                  # Community hierarchy level (0=macro, 1=micro)
include_metrics: bool = False   # Include centrality, density metrics
Response:
class GraphSummaryResponse(BaseModel):
    communities: List[GraphNode]    # Community nodes
    inter_community_edges: List[GraphEdge]  # Connections between communities
    stats: Dict[str, Any]           # Graph statistics
    timestamp: datetime             # Generation time
# Returns: ApiResponse[GraphSummaryResponse]
Implementation Logic:
1. Query Community table for level=0 (macro) communities
2. Calculate community positions (force-directed layout)
3. Aggregate inter-community edges (edges between entities in different communities)
4. Return compressed representation
Google AIP Compliance: AIP-132 (List) + custom aggregation
B. GET /api/v1/graph/neighborhood
Purpose: Dynamic node expansion when user clicks a node in Sigma.js
Query Parameters:
entity_id: str = Field(..., description="Center entity UUID")
depth: int = Field(1, ge=1, le=3)
direction: str = "bidirectional"
node_types: Optional[List[str]] = None
edge_types: Optional[List[str]] = None
min_confidence: float = 0.7
max_nodes: int = Field(100, le=1000)  # Safety limit
Response:
class NeighborhoodResponse(BaseModel):
    center_node: GraphNode
    nodes: List[GraphNode]      # Direct neighbors + center
    edges: List[GraphEdge]      # Connecting edges
    path: List[str]             # Traversal path
    expanded_count: int         # Number of new nodes
# Returns: ApiResponse[NeighborhoodResponse]
Implementation Logic:
1. Validate entity_id exists
2. Recursive traversal based on depth parameter
3. Apply filters (node_types, edge_types, min_confidence)
4. Limit results to max_nodes (AIP-158 safety)
5. Calculate node positions relative to center
6. Return subgraph
Google AIP Compliance: Custom method :neighborhood following AIP-136
C. GET /api/v1/graph/trajectory
Purpose: Time-based graph evolution for time slider (2026 Standard)
Query Parameters:
entity_ids: List[str] = Field(..., description="Entities to track")
start_date: datetime = Field(..., description="Start timestamp")
end_date: datetime = Field(..., description="End timestamp")
time_step: str = Field("month", pattern="^(day|week|month|year)$")
include_snapshots: bool = False
Response:
class TemporalSnapshot(BaseModel):
    timestamp: datetime
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metrics: Dict[str, float]
class TrajectoryResponse(BaseModel):
    entities: List[str]
    start_date: datetime
    end_date: datetime
    snapshots: List[TemporalSnapshot]
    timeline: List[Dict[str, Any]]  # Events, changes
# Returns: ApiResponse[TrajectoryResponse]
Implementation Logic:
1. Query temporal_validity_start/end from Edge table
2. Generate time series based on time_step
3. For each time point, filter edges valid at that time
4. Capture node/edge lifecycle events
5. Return timeline with graph snapshots
Google AIP Compliance: Custom method :trajectory following AIP-136 + AIP-160 filtering
D. GET /api/v1/graph/communities
Purpose: List all communities with metadata
Query Parameters:
level: Optional[int] = None        # Filter by hierarchy level
domain: Optional[str] = None       # Filter by domain
min_entity_count: int = 1
limit: int = 50
offset: int = 0
sort_by: str = "entity_count"      # Sort field
sort_order: str = "desc"           # asc/desc
entity_id: Optional[str] = None    # Find communities containing entity
Response:
class CommunitySummary(BaseModel):
    id: str
    name: str
    level: int
    entity_count: int
    summary: Optional[str]
    domain: Optional[str]
    created_at: datetime
    stats: Dict[str, Any]
class CommunitiesResponse(BaseModel):
    communities: List[CommunitySummary]
    total: int
    limit: int
    offset: int
    has_more: bool
# Returns: ApiResponse[CommunitiesResponse]
Implementation Logic:
1. Query Community table with filters
2. Apply pagination (AIP-158)
3. Include entity counts and statistics
4. Support reverse lookup (communities containing specific entity)
Google AIP Compliance: AIP-132 (List)
3. Document Evidence API (document_api.py)
3.1 Resource Model
# src/knowledge_base/common/api_models.py (CONTINUED)
class DocumentContentResponse(BaseModel):
    """Document content with extraction information."""
    document_id: str
    content: str                       # Full text content
    mime_type: str
    metadata: Dict[str, Any]
    extraction_state: Dict[str, Any]   # What was extracted
class TextSpan(BaseModel):
    """Text span with entity grounding information."""
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=0)
    text: str                          # Extracted text snippet
    entity_id: Optional[str] = None    # Linked entity UUID
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    grounding_quote: Optional[str] = None  # Verbatim quote
class DocumentSpansResponse(BaseModel):
    """All text spans with entity references for highlighting."""
    document_id: str
    spans: List[TextSpan]              # Sorted by start_offset
    total_entities: int
    verified_spans: int                # High confidence spans
class DocumentEntity(BaseModel):
    """Entity extracted from document with location info."""
    entity_id: str
    name: str
    entity_type: str
    description: Optional[str]
    spans: List[TextSpan]              # All occurrences in document
    confidence: float
    properties: Dict[str, Any]
class DocumentEntitiesResponse(BaseModel):
    """All entities extracted from document."""
    document_id: str
    entities: List[DocumentEntity]
    total: int
    domain: Optional[str]
3.2 Endpoints Specification
A. GET /api/v1/documents/{document_id}/content
Purpose: Retrieve raw document content (text or PDF URL)
Path Parameters:
document_id: str = Path(..., description="Document UUID")
Query Parameters:
format: str = Field("text", pattern="^(text|pdf|original)$")
signed_url_ttl: int = Field(3600, ge=60, le=86400)  # For PDF URLs
Response:
class DocumentContentResponse(BaseModel):
    document_id: str
    content: str                          # Text content or signed URL
    mime_type: str
    metadata: Dict[str, Any]
    extraction_state: Dict[str, Any]      # Processing summary
    signed_url: Optional[str] = None      # If format=pdf and available
# Returns: ApiResponse[DocumentContentResponse]
Implementation Logic:
1. Validate document exists and status=COMPLETED
2. Retrieve content from storage (file system or object storage)
3. For PDF: generate signed URL with TTL
4. Return extraction metadata (entities count, edges count, etc.)
Google AIP Compliance: AIP-131 (Get) with custom content format
B. GET /api/v1/documents/{document_id}/spans
Purpose: Get all text spans with entity references for exact highlighting (Verbatim Grounding)
Path Parameters:
document_id: str = Path(..., description="Document UUID")
Query Parameters:
confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
entity_types: Optional[List[str]] = None
verified_only: bool = False          # Only spans with grounding quotes
sort_by: str = "start_offset"        # Sorting field
Response:
# Returns: ApiResponse[DocumentSpansResponse]
Implementation Logic:
1. Query ChunkEntity junction table for document
2. Join with Entity and Chunk tables
3. Extract text spans with offsets
4. Include grounding_quote from extraction
5. Sort by start_offset for sequential highlighting
6. Support entity type filtering
Google AIP Compliance: Custom method :spans following AIP-136
C. GET /api/v1/documents/{document_id}/entities
Purpose: Get all entities extracted from document with their locations
Path Parameters:
document_id: str = Path(..., description="Document UUID")
Query Parameters:
entity_types: Optional[List[str]] = None
min_confidence: float = Field(0.7, ge=0.0, le=1.0)
sort_by: str = "confidence"
sort_order: str = "desc"
include_spans: bool = True           # Include text locations
Response:
# Returns: ApiResponse[DocumentEntitiesResponse]
Implementation Logic:
1. Query entities through ChunkEntity → Chunk → Document
2. Group spans by entity (multiple occurrences)
3. Calculate aggregate confidence
4. Support filtering and sorting
5. Include all text spans if requested
Google AIP Compliance: AIP-132 (List) for nested resource
D. POST /api/v1/documents/search
Purpose: Search across documents by content or metadata
Request Body:
class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    search_type: str = Field(
        "content",
        pattern="^(content|entities|metadata)$"
    )
    domains: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(20, ge=1, le=100)
Response:
class SearchResult(BaseModel):
    document_id: str
    score: float
    highlights: List[TextSpan]      # Matching spans
    snippet: str                    # Preview text
    metadata: Dict[str, Any]
class DocumentSearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    query: str
    took_ms: int
# Returns: ApiResponse[DocumentSearchResponse]
Implementation Logic:
1. Vector similarity search on Chunk.embeddings
2. Keyword search on entity names
3. Metadata filtering
4. Highlight matching regions
5. Return scored results with snippets
Google AIP Compliance: AIP-136 custom :search method
4. Common Components
4.1 API Configuration
# src/knowledge_base/common/api_config.py (NEW)
from fastapi import APIRouter
from typing import List, Optional
def create_api_router(prefix: str, tags: List[str]) -> APIRouter:
    """Factory for creating consistent API routers."""
    return APIRouter(
        prefix=prefix,
        tags=tags,
        responses={
            400: {"description": "Bad Request"},
            401: {"description": "Unauthorized"},
            403: {"description": "Forbidden"},
            404: {"description": "Not Found"},
            500: {"description": "Internal Server Error"},
        }
    )
4.2 Pagination Helper
# src/knowledge_base/common/pagination.py (NEW)
from pydantic import BaseModel
from typing import Generic, TypeVar, List
T = TypeVar('T')
class PaginationParams(BaseModel):
    """Standard pagination parameters following AIP-158."""
    limit: int = 50
    offset: int = 0
class PaginatedResponse(BaseModel, Generic[T]):
    """Standard paginated response wrapper."""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool
def paginate_query(results: List[T], total: int, limit: int, offset: int) -> PaginatedResponse[T]:
    """Helper to create paginated responses."""
    return PaginatedResponse(
        items=results,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(results)) < total
    )
4.3 Error Handling Middleware
# src/knowledge_base/common/error_handlers.py (NEW)
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi import status
async def api_exception_handler(request: Request, exc: Exception):
    """Global API exception handler following AIP-193."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "detail": exc.detail,
                    "status_code": exc.status_code,
                    "error_code": exc.__class__.__name__.upper()
                },
                "data": None
            }
        )
    
    # Log unexpected errors
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "detail": "Internal server error",
                "status_code": 500,
                "error_code": "INTERNAL_ERROR"
            },
            "data": None
        }
    )
5. TypeScript Client Generation
5.1 FastAPI Configuration
# src/knowledge_base/main.py (UPDATED)
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
app = FastAPI(
    title="KBV2 Knowledge Base API",
    description="High-fidelity information extraction and graph visualization API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json"
)
# Add new routers
from knowledge_base import graph_api, document_api
app.include_router(graph_api.router)
app.include_router(document_api.router)
# Add middleware
from knowledge_base.common.error_handlers import api_exception_handler
app.add_exception_handler(Exception, api_exception_handler)
def custom_openapi():
    """Customize OpenAPI schema for TypeScript generation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="KBV2 API",
        version="1.0.0",
        description="KBV2 Knowledge Base System API",
        routes=app.routes,
    )
    
    # Add custom extensions for TypeScript client generation
    openapi_schema["x-typescript"] = {
        "client_package": "@kbv2/typescript-client",
        "base_url_env": "NEXT_PUBLIC_KBV2_API_URL"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi
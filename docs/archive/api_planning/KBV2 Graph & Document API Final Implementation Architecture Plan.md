KBV2 Graph & Document API: Final Implementation Architecture Plan
Executive Summary
Based on comprehensive analysis of the codebase, Google AIP standards (v2024-2025), and graph/document intelligence best practices, I present a production-ready implementation plan that establishes KBV2 as a fully Google AIP-compliant, resource-oriented API with graph visualization and evidence-linking capabilities.
---
1. Phase 1: Foundation Infrastructure (Week 1)
1.1 Create Main Application (src/knowledge_base/main.py)
Purpose: Central FastAPI application to orchestrate all routers
Implementation:
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from knowledge_base.common.error_handlers import global_exception_handler
from knowledge_base.common.aip193_middleware import AIP193ResponseMiddleware
from knowledge_base.query_api import router as query_router
from knowledge_base.review_api import router as review_router
from knowledge_base.graph_api import router as graph_router
from knowledge_base.document_api import router as document_router
app = FastAPI(
    title="KBV2 Knowledge Base API",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)
# AIP-193 Response Middleware (wraps all responses)
app.add_middleware(AIP193ResponseMiddleware)
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Register Routers
app.include_router(query_router, prefix="/api/v1/query", tags=["query"])
app.include_router(review_api.router, prefix="/api/v1/review", tags=["review"])
app.include_router(graph_router, prefix="/api/v1/graphs", tags=["graph"])
app.include_router(document_router, prefix="/api/v1/documents", tags=["documents"])
# Global Exception Handler
app.add_exception_handler(Exception, global_exception_handler)
Justification: Follows established FastAPI patterns while introducing centralized AIP-193 compliance.
---
1.2 Standard Response Infrastructure (src/knowledge_base/common/)
A. AIP-193 Response Models (api_models.py)
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional, List, Any, Dict
T = TypeVar('T')
class ErrorInfo(BaseModel):
    """Google.rpc.ErrorInfo per AIP-193 specification"""
    reason: str = Field(..., description="Error reason code")
    domain: str = Field(default="kb.v2.api", description="Error domain")
    metadata: Dict[str, str] = Field(default_factory=dict)
class AIPError(BaseModel):
    """AIP-193 compliant error structure"""
    code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Human-readable error message")
    status: str = Field(..., description="google.rpc.Code enum value")
    details: List[Dict[str, Any]] = Field(default_factory=list)
class APIResponse(BaseModel, Generic[T]):
    """Standard wrapper for all API responses (AIP-193 compliance)"""
    success: bool
    data: Optional[T] = None
    error: Optional[AIPError] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
class PaginatedResponse(BaseModel, Generic[T]):
    """AIP-158 compliant pagination wrapper"""
    items: List[T]
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., ge=1, le=1000)
    offset: int = Field(..., ge=0)
    has_more: bool
B. AIP-193 Middleware (aip193_middleware.py)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json
class AIP193ResponseMiddleware(BaseHTTPMiddleware):
    """Automatically wraps all endpoint responses in AIP-193 structure"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Skip for non-JSON responses or health checks
        if "application/json" not in response.headers.get("content-type", ""):
            return response
            
        # Read response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        try:
            original_data = json.loads(body)
            
            # Already wrapped or is error response
            if "success" in original_data or "error" in original_data:
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            
            # Wrap in AIP-193 success format
            wrapped = {
                "success": response.status_code < 400,
                "data": original_data,
                "error": None,
                "metadata": {"request_id": request.headers.get("x-request-id")}
            }
            
            return Response(
                content=json.dumps(wrapped),
                status_code=200,  # AIP-193 always returns 200 for JSON
                headers=response.headers
            )
            
        except json.JSONDecodeError:
            return Response(content=body, status_code=response.status_code)
Justification: Ensures 100% AIP-193 compliance across all endpoints without boilerplate.
---
C. AIP-158 Pagination (pagination.py)
from pydantic import BaseModel, Field, validator
from typing import Optional
class PageParams(BaseModel):
    """AIP-158 compliant pagination parameters"""
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of items to return"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip"
    )
    order_by: Optional[str] = Field(
        default=None,
        description="AIP-132 ordering expression (e.g., 'name desc, create_time')"
    )
def paginate_query(query, page_params: PageParams) -> tuple[list, int]:
    """
    Generic pagination for SQLAlchemy queries
    
    Args:
        query: SQLAlchemy query object
        page_params: Pagination parameters
    
    Returns:
        tuple of (items, total_count)
    """
    # Get total count before pagination
    total = query.count()
    
    # Apply pagination
    items = (
        query
        .offset(page_params.offset)
        .limit(page_params.limit)
        .all()
    )
    
    return items, total
Justification: Standardizes pagination across all list endpoints per AIP-158.
---
D. Error Handlers (error_handlers.py)
from fastapi import Request
from fastapi.responses import JSONResponse
from knowledge_base.common.api_models import AIPError, ErrorInfo
def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler converting to AIP-193 error format"""
    
    if isinstance(exc, ValueError):
        error = AIPError(
            code=400,
            message=str(exc),
            status="INVALID_ARGUMENT",
            details=[{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "INVALID_ARGUMENT",
                "domain": "kb.v2.api",
                "metadata": {"path": str(request.url.path)}
            }]
        )
        status_code = 200  # AIP-193 returns 200 for JSON errors
    elif isinstance(exc, ResourceNotFoundError):
        error = AIPError(
            code=404,
            message="Resource not found",
            status="NOT_FOUND",
            details=[{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "RESOURCE_NOT_FOUND",
                "domain": "kb.v2.api",
                "metadata": {"resource_id": getattr(exc, "resource_id", "unknown")}
            }]
        )
        status_code = 200
    else:
        # Internal server error
        error = AIPError(
            code=500,
            message="Internal server error",
            status="INTERNAL",
            details=[{
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "INTERNAL_ERROR",
                "domain": "kb.v2.api"
            }]
        )
        status_code = 200
    
    return JSONResponse(
        content={"success": False, "error": error.dict(), "data": None},
        status_code=status_code
    )
Justification: Provides centralized error handling with AIP-193 compliance.
---
E. AIP-160 Filter Parser (filter_parser.py)
import re
from typing import Any, Dict, List, Optional
from sqlalchemy import and_, or_, not_
class FilterParser:
    """
    AIP-160 compliant filter parser for SQLAlchemy
    Converts filter expressions to SQLAlchemy WHERE clauses
    """
    
    OPERATORS = {
        "=": lambda col, val: col == val,
        "!=": lambda col, val: col != val,
        ">": lambda col, val: col > val,
        "<": lambda col, val: col < val,
        ">=": lambda col, val: col >= val,
        "<=": lambda col, val: col <= val,
        ":": lambda col, val: col.contains(val),
    }
    
    @staticmethod
    def parse(filter_str: str, model_class: Any) -> Any:
        """Parse AIP-160 filter string into SQLAlchemy expression"""
        # Implementation would use an EBNF parser library
        # For MVP, we'll support simple expressions: field operator value AND field operator value
        pass
---
2. Phase 2: Graph API Implementation (src/knowledge_base/graph_api.py)
2.1 Endpoint Architecture
from fastapi import APIRouter, Depends, Query
from typing import Optional
from uuid import UUID
router = APIRouter()
@router.get(
    "/{graph_id}/:summary",
    response_model=APIResponse[GraphSummaryResponse]
)
async def get_graph_summary(
    graph_id: UUID,
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/graphs/{graph_id}:summary
    
    Returns Level 0 community view with inter-community edges
    AIP-136 Custom Method pattern
    """
    pass
@router.get(
    "/{graph_id}/nodes/{node_id}/:neighborhood",
    response_model=APIResponse[NeighborhoodResponse]
)
async def get_neighborhood(
    graph_id: UUID,
    node_id: UUID,
    depth: int = Query(1, ge=1, le=3),
    confidence_threshold: float = Query(0.7),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/graphs/{graph_id}/nodes/{node_id}:neighborhood
    
    Expand node neighborhood for Sigma.js drill-down
    """
    pass
@router.post(
    "/{graph_id}/:findPath",
    response_model=APIResponse[PathResponse]
)
async def find_path(
    graph_id: UUID,
    request: PathFindingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    POST /api/v1/graphs/{graph_id}:findPath
    
    Complex pathfinding algorithm on server
    """
    pass
@router.get(
    "/{graph_id}/:export",
    response_model=APIResponse[GraphologyExport]
)
async def export_graph(
    graph_id: UUID,
    format: str = Query("graphology", enum=["graphology", "gexf", "graphml"]),
    include_layout: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/graphs/{graph_id}:export
    
    Export graph in Sigma.js/Graphology format
    """
    pass
---
2.2 Response Models
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
class GraphNode(BaseModel):
    """Sigma.js compatible node representation"""
    key: str = Field(..., description="Unique node identifier")
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node attributes (label, x, y, size, color, community)"
    )
class GraphEdge(BaseModel):
    """Sigma.js compatible edge representation"""
    key: str
    source: str
    target: str
    attributes: Dict[str, Any]
class GraphSummaryResponse(BaseModel):
    """Community-level graph view"""
    nodes: List[GraphNode] = Field(..., description="Community nodes")
    edges: List[GraphEdge] = Field(..., description="Inter-community edges")
    total_nodes: int
    total_edges: int
class NeighborhoodResponse(BaseModel):
    """Node neighborhood expansion"""
    center_node: GraphNode
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    max_confidence: float
    min_confidence: float
class GraphologyExport(BaseModel):
    """Strict Graphology JSON specification"""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "directed", "multi": True}
    )
    nodes: List[GraphNode]
    edges: List[GraphEdge]
---
2.3 Database Layer (graph_store.py)
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Tuple
class GraphStore:
    """Repository pattern for graph operations"""
    
    @staticmethod
    async def get_community_topology(
        db: AsyncSession,
        level: int = 0
    ) -> Tuple[List[Entity], List[EdgeAggregation]]:
        """
        Map-Reduce style edge aggregation for community views
        Returns: (communities, inter-community_edges)
        """
        # Community nodes query
        community_query = select(Entity).where(
            Entity.community_id.isnot(None),
            Community.level == level
        )
        
        # Inter-community edge aggregation
        edge_agg_query = """
        SELECT 
            s.community_id as source, 
            t.community_id as target, 
            COUNT(*) as weight,
            AVG(e.confidence) as avg_confidence
        FROM edges e
        JOIN entities s ON e.source_id = s.id
        JOIN entities t ON e.target_id = t.id
        WHERE s.community_id != t.community_id
        GROUP BY s.community_id, t.community_id
        """
        
        return await db.execute(community_query), await db.execute(edge_agg_query)
    
    @staticmethod
    async def get_entity_neighborhood(
        db: AsyncSession,
        entity_id: UUID,
        depth: int = 1,
        min_confidence: float = 0.7
    ) -> Tuple[List[Entity], List[Edge]]:
        """
        Retrieve immediate neighbors with edge filtering
        """
        # Edges from/to entity
        edge_query = select(Edge).where(
            or_(
                Edge.source_id == entity_id,
                Edge.target_id == entity_id
            ),
            Edge.confidence >= min_confidence
        )
        
        # Fetch all related entities
        neighbor_ids = set()
        async for edge in db.stream(edge_query):
            neighbor_ids.add(edge.source_id)
            neighbor_ids.add(edge.target_id)
        
        neighbor_ids.discard(entity_id)  # Remove center
        
        # Fetch neighbor entities
        entity_query = select(Entity).where(Entity.id.in_(neighbor_ids))
        
        return await db.execute(entity_query), await db.execute(edge_query)
---
3. Phase 3: Document API Implementation (src/knowledge_base/document_api.py)
3.1 Endpoint Architecture
from fastapi import APIRouter, Depends, Path, Body
from uuid import UUID
router = APIRouter()
@router.get(
    "/{document_id}",
    response_model=APIResponse[DocumentResponse]
)
async def get_document(
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/documents/{document_id}
    
    AIP-131 Get method for document metadata
    """
    pass
@router.get(
    "/{document_id}/chunks",
    response_model=APIResponse[PaginatedResponse[ChunkResponse]]
)
async def list_document_chunks(
    document_id: UUID,
    page_params: PageParams = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/documents/{document_id}/chunks
    
    AIP-132 List method for document chunks with pagination
    """
    pass
@router.get(
    "/{document_id}/spans",
    response_model=APIResponse[List[TextSpan]]
)
async def get_document_spans(
    document_id: UUID,
    entity_id: Optional[UUID] = None,
    confidence_threshold: float = Query(0.5),
    db: AsyncSession = Depends(get_db)
):
    """
    GET /api/v1/documents/{document_id}/spans
    
    Returns integer offsets for entity grounding
    """
    pass
@router.post(
    "/:search",
    response_model=APIResponse[PaginatedResponse[SearchResult]]
)
async def search_documents(
    request: SearchRequest,
    page_params: PageParams = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    POST /api/v1/documents:search
    
    Hybrid search combining vector + keyword
    AIP-136 Custom Method
    """
    pass
@router.post(
    "/{document_id}/annotations",
    response_model=APIResponse[AnnotationResponse]
)
async def create_annotation(
    document_id: UUID,
    annotation: W3CAnnotation,
    db: AsyncSession = Depends(get_db)
):
    """
    POST /api/v1/documents/{document_id}/annotations
    
    W3C Web Annotation Data Model compliance
    """
    pass
---
3.2 Response Models (W3C Web Annotation Compliant)
from pydantic import BaseModel, Field, conint
from typing import List, Dict, Any, Optional
class TextPositionSelector(BaseModel):
    """W3C TextPositionSelector for exact offsets"""
    type: str = Field(default="TextPositionSelector")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
class TextQuoteSelector(BaseModel):
    """W3C TextQuoteSelector for fuzzy matching"""
    type: str = Field(default="TextQuoteSelector")
    exact: str = Field(..., description="Exact text to match")
    prefix: Optional[str] = None
    suffix: Optional[str] = None
class Selector(BaseModel):
    """Unified selector for text spans"""
    type: str
    start: Optional[int] = None
    end: Optional[int] = None
    exact: Optional[str] = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
class TextSpan(BaseModel):
    """Entity grounding with offsets (frontend highlighting)"""
    start_offset: int = Field(..., description="Absolute start position")
    end_offset: int = Field(..., description="Absolute end position")
    entity_id: UUID
    entity_type: str
    entity_name: str
    quote: str = Field(..., description="Exact text from document")
    confidence: float = Field(..., ge=0.0, le=1.0)
    selectors: List[Selector] = Field(
        default_factory=list,
        description="W3C compliant selectors"
    )
class W3CAnnotation(BaseModel):
    """W3C Web Annotation Data Model"""
    id: Optional[str] = None
    type: str = Field(default="Annotation")
    target: Dict[str, Any] = Field(..., description="Target with selectors")
    body: Dict[str, Any] = Field(..., description="Annotation body")
    created: Optional[datetime] = None
    creator: Optional[str] = None
class SearchResult(BaseModel):
    """Hybrid search result"""
    document_id: UUID
    chunk_id: UUID
    chunk_text: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    match_type: str = Field(..., enum=["vector", "keyword", "hybrid"])
    entity_matches: Optional[List[Dict[str, Any]]] = None
---
3.3 Text Offset Calculation Service (offset_service.py)
class OffsetCalculationService:
    """
    Converts grounding_quote + chunk_text -> absolute integer offsets
    Critical for frontend text highlighting
    """
    
    @staticmethod
    def calculate_absolute_offsets(
        chunks: List[Chunk],
        chunk_entities: List[ChunkEntity]
    ) -> List[TextSpan]:
        """
        Core logic from high-level logic guide
        
        Args:
            chunks: Ordered list of document chunks
            chunk_entities: Entity mentions with grounding quotes
        
        Returns:
            List of TextSpan objects with absolute offsets
        """
        global_offset = 0
        text_spans = []
        
        for chunk in sorted(chunks, key=lambda c: c.chunk_index):
            for ce in chunk_entities:
                if ce.chunk_id != chunk.id:
                    continue
                
                # Find quote in chunk text
                local_start = chunk.text.find(ce.grounding_quote)
                
                if local_start == -1:
                    logging.warning(
                        f"Quote not found in chunk: {ce.grounding_quote[:50]}..."
                    )
                    # Fallback: fuzzy matching or skip
                    continue
                
                absolute_start = global_offset + local_start
                absolute_end = absolute_start + len(ce.grounding_quote)
                
                text_span = TextSpan(
                    start_offset=absolute_start,
                    end_offset=absolute_end,
                    entity_id=ce.entity_id,
                    entity_type=ce.entity.entity_type,
                    entity_name=ce.entity.name,
                    quote=ce.grounding_quote,
                    confidence=ce.confidence_score,
                    selectors=[
                        Selector(
                            type="TextPositionSelector",
                            start=absolute_start,
                            end=absolute_end
                        ),
                        Selector(
                            type="TextQuoteSelector",
                            exact=ce.grounding_quote,
                            prefix=chunk.text[max(0, local_start-50):local_start],
                            suffix=chunk.text[local_start+len(ce.grounding_quote):local_start+len(ce.grounding_quote)+50]
                        )
                    ]
                )
                text_spans.append(text_span)
            
            global_offset += len(chunk.text)
        
        return sorted(text_spans, key=lambda s: (s.start_offset, s.end_offset))
---
3.4 Hybrid Search Service (hybrid_search.py)
from knowledge_base.persistence.v1.vector_store import VectorStore
from sqlalchemy import select, or_, and_, func
class HybridSearchService:
    """
    Combines vector similarity (0.7 weight) with keyword search (0.3 weight)
    Uses Reciprocal Rank Fusion (RRF) for result merging
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    async def search(
        self,
        query: str,
        document_ids: Optional[List[UUID]] = None,
        limit: int = 50
    ) -> List[SearchResult]:
        """
        Execute hybrid search with weighted fusion
        
        Args:
            query: Search query string
            document_ids: Optional filter to specific documents
            limit: Maximum results to return
        
        Returns:
            Ranked list of SearchResult objects
        """
        # 1. Vector search (semantic similarity)
        vector_results = await self._vector_search(query, document_ids, limit)
        
        # 2. Keyword search (entity names)
        keyword_results = await self._keyword_search(query, document_ids, limit)
        
        # 3. Reciprocal Rank Fusion
        return self._fuse_results(vector_results, keyword_results)
    
    async def _vector_search(
        self,
        query: str,
        document_ids: Optional[List[UUID]],
        limit: int
    ) -> List[Dict]:
        """Query vector store for similar chunks"""
        query_embedding = await self._embed_query(query)
        
        search_results = await self.vector_store.search_similar_chunks(
            query_embedding,
            limit=limit * 2,  # Get more for reranking
            threshold=0.6
        )
        
        return [
            {
                "chunk_id": result.id,
                "score": result.score,
                "chunk": result.chunk,
                "rank": idx + 1
            }
            for idx, result in enumerate(search_results)
        ]
    
    async def _keyword_search(
        self,
        query: str,
        document_ids: Optional[List[UUID]],
        limit: int
    ) -> List[Dict]:
        """ILike search on Entity names"""
        db_query = select(Entity, Chunk).join(
            ChunkEntity, Entity.id == ChunkEntity.entity_id
        ).join(
            Chunk, ChunkEntity.chunk_id == Chunk.id
        ).where(
            Entity.name.ilike(f"%{query}%")
        )
        
        if document_ids:
            db_query = db_query.where(Chunk.document_id.in_(document_ids))
        
        results = await db.execute(db_query.limit(limit * 2))
        
        return [
            {
                "entity_id": row.Entity.id,
                "entity_name": row.Entity.name,
                "entity_type": row.Entity.entity_type,
                "chunk_id": row.Chunk.id,
                "chunk_text": row.Chunk.text,
                "rank": idx + 1
            }
            for idx, row in enumerate(results)
        ]
    
    def _fuse_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60  # RRF constant
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion algorithm
        
        Score = Σ(1 / (rank + k)) across both result sets
        """
        fused_scores = {}
        
        # Score vector results
        for result in vector_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            fused_scores[chunk_id] = {
                "score": 1.0 / (rank + k),
                "chunk_text": result["chunk"]["text"],
                "match_type": "vector"
            }
        
        # Score and combine keyword results
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            
            if chunk_id in fused_scores:
                # Combined match (higher weight)
                fused_scores[chunk_id]["score"] += 1.0 / (rank + k)
                fused_scores[chunk_id]["match_type"] = "hybrid"
                fused_scores[chunk_id]["entity_matches"] = [{
                    "entity_id": result["entity_id"],
                    "entity_name": result["entity_name"],
                    "entity_type": result["entity_type"]
                }]
            else:
                # Keyword-only match
                fused_scores[chunk_id] = {
                    "score": 1.0 / (rank + k),
                    "chunk_text": result["chunk_text"],
                    "match_type": "keyword"
                }
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        return [
            SearchResult(
                document_id=chunk.chunk.document_id,
                chunk_id=chunk_id,
                chunk_text=data["chunk_text"],
                relevance_score=min(data["score"], 1.0),
                match_type=data["match_type"],
                entity_matches=data.get("entity_matches")
            )
            for chunk_id, data in sorted_results[:limit]
        ]
---
4. Phase 4: Configuration & Integration (Week 2)
4.1 Updated Configuration (config.py)
from pydantic import BaseSettings, Field
from typing import Optional
class GraphAPIConfig(BaseSettings):
    """Graph API specific configuration"""
    default_depth: int = Field(default=1, ge=1, le=5)
    default_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_nodes_export: int = Field(default=10000, ge=100)
    enable_server_layout: bool = Field(default=True)
class DocumentAPIConfig(BaseSettings):
    """Document API specific configuration"""
    max_chunk_size: int = Field(default=512)
    hybrid_search_vector_weight: float = Field(default=0.7)
    hybrid_search_keyword_weight: float = Field(default=0.3)
    rrf_constant: int = Field(default=60)
class APIConfig(BaseSettings):
    """Root API configuration"""
    graph: GraphAPIConfig = Field(default_factory=GraphAPIConfig)
    documents: DocumentAPIConfig = Field(default_factory=DocumentAPIConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
---
4.2 Database Extensions (schema.py)
Required Schema Additions:
# In entities table - ensure indexes
Index("idx_entity_community", Entity.community_id)
Index("idx_entity_type", Entity.entity_type)
# In edges table - ensure indexes  
Index("idx_edge_source", Edge.source_id)
Index("idx_edge_target", Edge.target_id)
Index("idx_edge_confidence", Edge.confidence)
# In chunk_entities table - ensure indexes
Index("idx_chunk_entity_entity", ChunkEntity.entity_id)
Index("idx_chunk_entity_chunk", ChunkEntity.chunk_id)
Index("idx_chunk_entity_confidence", ChunkEntity.confidence_score)
---
4.3 Vector Store Integration
Enhance existing VectorStore class:
class EnhancedVectorStore(VectorStore):
    """Extended VectorStore with graph/document search capabilities"""
    
    async def search_graph_entities(
        self,
        query_embedding: List[float],
        community_id: Optional[UUID] = None,
        limit: int = 10
    ) -> List[SimilarityResult]:
        """Search within community context"""
        if community_id:
            # Filter by community
            query = select(Entity).where(Entity.community_id == community_id)
        else:
            query = select(Entity)
        
        # Add vector similarity condition
        query = query.order_by(
            Entity.embedding.cosine_distance(query_embedding)
        ).limit(limit)
        
        results = await self.session.execute(query)
        return results.scalars().all()
---
5. Phase 5: Testing & Validation (Week 2.5)
5.1 Unit Tests
# tests/unit/test_graph_api.py
@pytest.mark.asyncio
async def test_graph_summary_endpoint():
    """Test AIP-136 :summary custom method"""
    # Setup mock data
    # Call endpoint
    # Assert AIP-193 response structure
    assert response.status_code == 200
    assert "success" in response.json()
    assert "data" in response.json()
# tests/unit/test_offset_calculation.py
def test_text_span_calculation():
    """Test W3C TextSpan offset calculation"""
    chunks = [...]
    chunk_entities = [...]
    spans = OffsetCalculationService.calculate_absolute_offsets(chunks, chunk_entities)
    assert len(spans) > 0
    assert all(hasattr(span, "start_offset") for span in spans)
---
5.2 Integration Tests
# tests/integration/test_graph_flow.py
@pytest.mark.asyncio
async def test_full_graph_workflow():
    """End-to-end graph API workflow"""
    # 1. Get graph summary
    summary = await client.get("/api/v1/graphs/{graph_id}:summary")
    
    # 2. Expand neighborhood
    neighborhood = await client.get(
        "/api/v1/graphs/{graph_id}/nodes/{node_id}:neighborhood"
    )
    
    # 3. Verify response structure
    assert summary.json()["success"] is True
    assert "nodes" in summary.json()["data"]
    assert "edges" in summary.json()["data"]
# tests/integration/test_document_workflow.py
@pytest.mark.asyncio
async def test_evidence_workflow():
    """End-to-end evidence linking workflow"""
    # 1. Get document spans
    spans = await client.get("/api/v1/documents/{doc_id}/spans")
    
    # 2. Verify W3C format
    span = spans.json()["data"][0]
    assert "selectors" in span
    assert any(s["type"] == "TextPositionSelector" for s in span["selectors"])
---
5.3 AIP Compliance Testing
# tests/compliance/test_aip193.py
def test_all_responses_wrapped():
    """Verify every endpoint returns AIP-193 structure"""
    endpoints = get_all_endpoints()
    for endpoint in endpoints:
        response = client.request(endpoint.method, endpoint.path)
        
        # AIP-193: All JSON responses must have "success" field
        assert "success" in response.json()
        assert "data" in response.json() or "error" in response.json()
# tests/compliance/test_aip158.py
@pytest.mark.parametrize("endpoint", list_endpoints())
async def test_pagination_compliance(endpoint):
    """Verify pagination follows AIP-158"""
    if endpoint.supports_pagination:
        response = await client.get(endpoint.path, params={"limit": 50})
        data = response.json()["data"]
        
        # AIP-158 required fields
        assert "items" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "has_more" in data
---
6. Frontend Integration Specifications
6.1 Sigma.js Integration
API Client:
// frontend/lib/api-client.ts
interface GraphAPI {
  getSummary(graphId: string): Promise<GraphSummaryResponse>;
  getNeighborhood(graphId: string, nodeId: string): Promise<NeighborhoodResponse>;
  exportGraph(graphId: string, format: 'graphology'): Promise<GraphologyExport>;
}
// Usage
const graphData = await api.graphs.getSummary(graphId);
const renderer = new Sigma(graphData, container);
---
6.2 Evidence Viewer Integration
interface EvidenceAPI {
  getDocumentSpans(docId: string): Promise<TextSpan[]>;
  searchDocuments(query: string): Promise<SearchResult[]>;
}
// Text highlighting implementation
function highlightSpans(text: string, spans: TextSpan[]): JSX.Element {
  return spans
    .sort((a, b) => a.start_offset - b.start_offset)
    .map(span => (
      <mark 
        key={span.entity_id}
        data-entity-id={span.entity_id}
        data-start={span.start_offset}
        data-end={span.end_offset}
      >
        {text. From my research, I can provide a comprehensive report on the current state of the codebase structure and how it aligns with the proposed API implementation.
## **Key Findings:**
### **Current State:**
- **Well-architected modular design** with separate `query_api.py` and `review_api.py` routers
- **Comprehensive database schema** with all necessary tables (Entity, Edge, Community, Chunk, ChunkEntity)
- **Existing VectorStore infrastructure** that we can extend
- **Strong type safety** with Pydantic models throughout
- **No main application entry point** - routers exist but aren't assembled
- **Missing common infrastructure** for AIP-193 responses and AIP-158 pagination
### **Architecture Gaps:**
1. **No `main.py`** - FastAPI app instance doesn't exist
2. **No AIP-193 compliance** - endpoints return raw dicts, not standardized responses
3. **No AIP-158 pagination** - pagination is ad-hoc in review_api
4. **Missing graph_api.py and document_api.py** - planned but not implemented
5. **No error handling middleware** - each endpoint handles errors independently
### **Recommended Patterns:**
Following existing router patterns in `query_api.py` and `review_api.py`, the implementation should:
1. Create individual `APIRouter` instances in each module
2. Use Pydantic models consistently for request/response schemas
3. Leverage existing `VectorStore` and database session patterns
4. Add centralized middleware for AIP compliance rather than per-endpoint wrappers
### **Implementation Strategy:**
The codebase is **ready for extension** - we should:
1. Start with Phase 1: Create `main.py` and common AIP-compliant infrastructure
2. Follow existing patterns from `query_api.py` when implementing Graph and Document APIs
3. Extend `VectorStore` class instead of creating new database abstractions
4. Maintain backward compatibility with existing endpoints
**File Structure to Follow:**
- Each API module creates its own `APIRouter`
- Database access through dependency injection (`Depends(get_db)`)
- Configuration via existing Pydantic Settings pattern
- Observability through `@trace_operation` decorators already in place
---
**Critical Note**: I notice a discrepancy between the documentation and codebase:
The docs reference `src/knowledge_base/common/` but the codebase uses `src/knowledge_base/persistence/v1/` and `src/knowledge_base/common/gateway.py`. **Which directory structure should we follow?**
Should I:
1. Use the existing structure (`src/knowledge_base/common/gateway.py`, `src/knowledge_base/persistence/v1/`)
2. Create new structure per docs (`src/knowledge_base/common/api_models.py`, `src/knowledge_base/common/pagination.py`)
3. Hybrid approach - new files in existing `/common/` directory
Please clarify this before I proceed with the final architecture plan. \ Provide warnings about any aspects of this plan you may want us to reconsider! Provide warnings about any aspects of this plan you may want us to reconsider!
We are working in our first planning iteration, so we won't do anything beyond plan. Users often forget to mention important details early. When you call Task subagents to provide additional context for planning and as constraints, add targeted "reminders" at the end of the prompts we provide, to ensure the subagent attempts to surface any conflict or issue we should be aware of before generating our final plan. \ Provide warnings about any aspects of this plan you may want us to reconsider! I'll surface potential conflicts and concerns for your review.
## **⚠️ Potential Issues & Conflicts**
### **1. Directory Structure Conflict**
**Problem**: Documentation specifies `src/knowledge_base/common/` but codebase uses `src/knowledge_base/persistence/v1/` and `src/knowledge_base/common/gateway.py`
**Impact**: This creates ambiguity about where to place new files
- **Option A**: Follow docs structure, create new files under `/common/`
- **Option B**: Follow existing structure, use `/persistence/v1/` for database files
**Recommendation Needed**: Which structure should take precedence?
---
### **2. Response Format Backward Compatibility**
**Problem**: Introducing AIP-193 response wrappers will change ALL existing API responses
**Current Query API Response**:
```json
{
  "sql": "SELECT * FROM entities",
  "results": [...]
}
New AIP-193 Response:
{
  success: true,
  data: {
    sql: SELECT * FROM entities,
    results: [...]
  },
  error: null,
  metadata: {}
}
Impact: 
- Breaks existing frontend/clients using Query and Review APIs
- MCP WebSocket protocol may need updates
- Breaking change requires version bump (v1 → v2) or opt-in mechanism
Options:
1. Major version bump (/api/v2/) to maintain backward compatibility
2. Opt-in header (X-AIP-193: true) for gradual migration  
3. Break existing APIs and update all clients immediately
4. Leave existing APIs unchanged, only apply to new Graph/Document APIs
---
3. Pagination Token vs Offset
Problem: AIP-158 specifies opaque page_token (string), but current Review API uses limit+offset
Current Review API:
# GET /api/v1/review/pending?limit=50&offset=100
AIP-158 Standard:
# GET /api/v1/review/pending?page_size=50&page_token=ChRjYUoJfSFdEh0KCVpPo67QGAUo8MDc  # opaque token
Impact:
- Changing to opaque tokens breaks existing Review API
- More

"""
Document Evidence API for verbatim grounding and W3C Web Annotation compliance.

This module provides endpoints for linking entities back to exact text evidence
in source documents, supporting Sigma.js click-through to text evidence viewer.

Endpoints:
- GET /api/v1/documents/{document_id} - Get document metadata
- GET /api/v1/documents/{document_id}/content - Get document content
- GET /api/v1/documents/{document_id}/spans - Get entity text spans with offsets
- GET /api/v1/documents/{document_id}/entities - Get entities extracted from document
- POST /api/v1/documents:search - Search across documents (hybrid search)
- POST /api/v1/documents/{document_id}/annotations - Create W3C annotations
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base.common.api_models import APIResponse, PaginatedResponse
from knowledge_base.common.pagination import PageParams, create_paginated_response
from knowledge_base.persistence.v1.schema import Document, Chunk, ChunkEntity, Entity
from knowledge_base.common.offset_service import OffsetCalculationService, TextSpan


router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


# Response Models
class DocumentResponse(BaseModel):
    """Document metadata response."""

    id: str
    name: str
    source_uri: Optional[str]
    status: str
    domain: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    chunk_count: int
    entity_count: int
    processing_time_ms: Optional[int]


class DocumentContentResponse(BaseModel):
    """Document content with text and extraction information."""

    document_id: str
    content: str
    mime_type: str
    metadata: Dict[str, Any]
    extraction_state: Dict[str, Any]
    total_length: int
    chunk_count: int


class DocumentSpansResponse(BaseModel):
    """All text spans with entity references for highlighting."""

    document_id: str
    spans: List[Dict[str, Any]]  # TextSpan objects as dictionaries
    total_spans: int
    verified_spans: int
    entities_found: int
    coverage_percentage: float


class DocumentEntityResponse(BaseModel):
    """Entity extracted from document with location information."""

    entity_id: str
    name: str
    entity_type: str
    description: Optional[str]
    confidence: float
    properties: Dict[str, Any]
    spans: List[Dict[str, Any]]  # All occurrences in this document
    first_mentioned: Optional[datetime]


class DocumentEntitiesResponse(BaseModel):
    """All entities extracted from document."""

    document_id: str
    entities: List[DocumentEntityResponse]
    total: int
    domain: Optional[str]


class DocumentSearchRequest(BaseModel):
    """Request for hybrid search across documents."""

    query: str = Field(..., min_length=1, description="Search query text")
    search_type: str = Field(
        default="hybrid",
        pattern="^(vector|keyword|hybrid)$",
        description="Search type to use",
    )
    domains: Optional[List[str]] = Field(None, description="Filter by domains")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    entity_types: Optional[List[str]] = Field(
        None, description="Filter by entity types"
    )
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence")
    limit: int = Field(50, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Result offset")


class SearchResult(BaseModel):
    """Single search result with relevance information."""

    document_id: str
    chunk_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    snippet: str = Field(..., description="Matching text snippet")
    highlights: List[Dict[str, Any]] = Field(..., description="Highlight regions")
    entities: List[Dict[str, Any]] = Field(..., description="Matched entities")
    metadata: Dict[str, Any]
    match_type: str = Field(..., pattern="^(vector|keyword|hybrid)$")


class DocumentSearchResponse(BaseModel):
    """Hybrid search response with ranked results."""

    results: List[SearchResult]
    total: int
    query: str
    took_ms: int
    search_type: str


# Pydantic models for W3C Web Annotation compliance
class TextPositionSelectorModel(BaseModel):
    """W3C TextPositionSelector model."""

    type: str = Field(default="TextPositionSelector")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class TextQuoteSelectorModel(BaseModel):
    """W3C TextQuoteSelector model."""

    type: str = Field(default="TextQuoteSelector")
    exact: str = Field(..., description="Exact text to match")
    prefix: Optional[str] = None
    suffix: Optional[str] = None


class W3CAnnotation(BaseModel):
    """W3C Web Annotation Data Model."""

    id: Optional[str] = None
    type: str = Field(default="Annotation")
    target: Dict[str, Any] = Field(..., description="Target with selectors")
    body: Dict[str, Any] = Field(..., description="Annotation body/content")
    created: Optional[datetime] = None
    creator: Optional[str] = None
    motivation: Optional[str] = None


# Database dependency
async def get_db() -> AsyncSession:
    """Get async database session."""
    from knowledge_base.common.dependencies import get_async_session

    async for session in get_async_session():
        yield session


@router.get("/{document_id}", response_model=APIResponse[DocumentResponse])
async def get_document(
    document_id: UUID = Path(..., description="Document UUID"),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/v1/documents/{document_id}

    Get document metadata and extraction statistics.
    AIP-131 Get method for retrieving resource by ID.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Count chunks
    chunk_count_result = await db.execute(
        select(func.count(Chunk.id)).where(Chunk.document_id == document_id)
    )
    chunk_count = chunk_count_result.scalar()

    # Count distinct entities
    entity_count_result = await db.execute(
        select(func.count(Entity.id.distinct()))
        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id)
    )
    entity_count = entity_count_result.scalar()

    response = DocumentResponse(
        id=str(document.id),
        name=document.name,
        source_uri=document.source_uri,
        status=document.status.value
        if hasattr(document.status, "value")
        else str(document.status),
        domain=document.domain,
        metadata=document.metadata or {},
        created_at=document.created_at,
        updated_at=document.updated_at,
        chunk_count=chunk_count,
        entity_count=entity_count or 0,
        processing_time_ms=document.metadata.get("processing_time_ms")
        if document.metadata
        else None,
    )

    return APIResponse(
        success=True,
        data=response,
        error=None,
        metadata={"document_id": str(document_id)},
    )


@router.get(
    "/{document_id}/content", response_model=APIResponse[DocumentContentResponse]
)
async def get_document_content(
    document_id: UUID = Path(..., description="Document UUID"),
    format: str = Query("text", pattern="^(text|json)$", description="Response format"),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/v1/documents/{document_id}/content

    Get document content with all chunks concatenated in order.
    Supports content extraction state and metadata.
    """
    # Verify document exists and is completed
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Fetch all chunks in order
    chunk_result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    chunks = chunk_result.scalars().all()

    if not chunks:
        raise HTTPException(
            status_code=404, detail=f"No content found for document {document_id}"
        )

    # Concatenate all chunks
    full_text = ""
    chunk_metadata = []

    for chunk in chunks:
        full_text += chunk.text
        chunk_metadata.append(
            {
                "chunk_id": str(chunk.id),
                "index": chunk.chunk_index,
                "length": len(chunk.text),
            }
        )

    response = DocumentContentResponse(
        document_id=str(document_id),
        content=full_text,
        mime_type=document.metadata.get("mime_type", "text/plain")
        if document.metadata
        else "text/plain",
        metadata=document.metadata or {},
        extraction_state={
            "status": document.status.value
            if hasattr(document.status, "value")
            else str(document.status),
            "chunk_count": len(chunks),
            "total_length": len(full_text),
            "chunks": chunk_metadata,
        },
        total_length=len(full_text),
        chunk_count=len(chunks),
    )

    return APIResponse(
        success=True,
        data=response,
        error=None,
        metadata={"document_id": str(document_id)},
    )


@router.get("/{document_id}/spans", response_model=APIResponse[DocumentSpansResponse])
async def get_document_spans(
    document_id: UUID = Path(..., description="Document UUID"),
    confidence_threshold: float = Query(
        0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    entity_types: Optional[List[str]] = Query(
        None, description="Filter by entity types"
    ),
    verified_only: bool = Query(
        False, description="Only return spans with grounding quotes"
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/v1/documents/{document_id}/spans

    Get all text spans with entity references for exact text highlighting.
    Calculates absolute character offsets using the OffsetCalculationService.

    Critical for "Click entity -> See text evidence" functionality.
    Implements W3C Web Annotation Data Model compliance.
    """
    # Fetch document with chunks
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Fetch all chunks ordered by index
    chunk_result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    chunks = chunk_result.scalars().all()

    if not chunks:
        return APIResponse(
            success=True,
            data=DocumentSpansResponse(
                document_id=str(document_id),
                spans=[],
                total_spans=0,
                verified_spans=0,
                entities_found=0,
                coverage_percentage=0.0,
            ),
            error=None,
            metadata={"document_id": str(document_id)},
        )

    # Build query for chunk entities with optional filters
    entity_query = (
        select(ChunkEntity, Entity)
        .join(Entity, ChunkEntity.entity_id == Entity.id)
        .where(ChunkEntity.chunk_id.in_([chunk.id for chunk in chunks]))
    )

    if confidence_threshold > 0:
        entity_query = entity_query.where(
            ChunkEntity.confidence_score >= confidence_threshold
        )

    if entity_types:
        entity_query = entity_query.where(Entity.entity_type.in_(entity_types))

    if verified_only:
        entity_query = entity_query.where(ChunkEntity.grounding_quote.isnot(None))

    # Execute query and get results
    entity_result = await db.execute(entity_query)
    chunk_entities = entity_result.all()

    if not chunk_entities:
        return APIResponse(
            success=True,
            data=DocumentSpansResponse(
                document_id=str(document_id),
                spans=[],
                total_spans=0,
                verified_spans=0,
                entities_found=0,
                coverage_percentage=0.0,
            ),
            error=None,
            metadata={"document_id": str(document_id)},
        )

    # Calculate absolute offsets using the offset calculation service
    text_spans = OffsetCalculationService.calculate_absolute_offsets(
        chunks=chunks, chunk_entities=[(ce, e) for ce, e in chunk_entities]
    )

    # Filter by confidence threshold (already filtered at DB level, but redundant is safe)
    if confidence_threshold > 0:
        text_spans = [s for s in text_spans if s.confidence >= confidence_threshold]

    # Count verified spans (those with grounding quotes)
    verified_spans = len([s for s in text_spans if s.grounding_quote])

    # Unique entities found
    unique_entities = len(set(s.entity_id for s in text_spans if s.entity_id))

    # Calculate coverage percentage
    if chunks:
        total_text_length = sum(len(c.text) for c in chunks)
        covered_length = sum(s.end_offset - s.start_offset for s in text_spans)
        coverage_percentage = (
            (covered_length / total_text_length * 100) if total_text_length > 0 else 0.0
        )
    else:
        coverage_percentage = 0.0

    response = DocumentSpansResponse(
        document_id=str(document_id),
        spans=[span.to_dict() for span in text_spans],
        total_spans=len(text_spans),
        verified_spans=verified_spans,
        entities_found=unique_entities,
        coverage_percentage=round(coverage_percentage, 2),
    )

    return APIResponse(
        success=True,
        data=response,
        error=None,
        metadata={
            "document_id": str(document_id),
            "chunk_count": len(chunks),
            "entity_count": unique_entities,
        },
    )


@router.get(
    "/{document_id}/entities", response_model=APIResponse[DocumentEntitiesResponse]
)
async def get_document_entities(
    document_id: UUID = Path(..., description="Document UUID"),
    entity_types: Optional[List[str]] = Query(
        None, description="Filter by entity types"
    ),
    min_confidence: float = Query(
        0.5, ge=0.0, le=1.0, description="Minimum confidence"
    ),
    include_spans: bool = Query(True, description="Include text locations"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/v1/documents/{document_id}/entities

    Get all entities extracted from document with their locations and confidence scores.
    Groups multiple occurrences of the same entity.
    """
    # Build entity query with document join
    entity_query = (
        select(
            Entity.id,
            Entity.name,
            Entity.entity_type,
            Entity.description,
            Entity.confidence,
            Entity.properties,
            Entity.created_at,
            func.count(ChunkEntity.id).label("mention_count"),
            func.max(ChunkEntity.confidence_score).label("max_confidence"),
        )
        .join(ChunkEntity, Entity.id == ChunkEntity.entity_id)
        .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
        .where(Chunk.document_id == document_id)
    )

    # Apply filters
    if entity_types:
        entity_query = entity_query.where(Entity.entity_type.in_(entity_types))

    if min_confidence > 0:
        entity_query = entity_query.where(
            or_(
                Entity.confidence >= min_confidence,
                ChunkEntity.confidence_score >= min_confidence,
            )
        )

    # Group by entity
    entity_query = entity_query.group_by(
        Entity.id,
        Entity.name,
        Entity.entity_type,
        Entity.description,
        Entity.confidence,
        Entity.properties,
        Entity.created_at,
    )

    # Get total count
    count_query = select(func.count()).select_from(entity_query.subquery())
    count_result = await db.execute(count_query)
    total = count_result.scalar()

    # Apply pagination
    entity_query = entity_query.limit(limit).offset(offset)

    # Execute query
    result = await db.execute(entity_query)
    entities = result.all()

    entity_responses = []

    for entity_row in entities:
        # Fetch spans for this entity if requested
        spans = []
        if include_spans:
            span_result = await db.execute(
                select(ChunkEntity, Chunk)
                .join(Chunk, ChunkEntity.chunk_id == Chunk.id)
                .where(
                    ChunkEntity.entity_id == entity_row.id,
                    Chunk.document_id == document_id,
                )
                .order_by(Chunk.chunk_index)
            )

            span_rows = span_result.all()

            for chunk_entity, chunk in span_rows:
                if chunk_entity.grounding_quote:
                    # Find position in chunk
                    local_start = chunk.text.find(chunk_entity.grounding_quote)

                    if local_start != -1:
                        spans.append(
                            {
                                "start_offset": local_start,
                                "end_offset": local_start
                                + len(chunk_entity.grounding_quote),
                                "text": chunk.text[
                                    local_start : local_start
                                    + len(chunk_entity.grounding_quote)
                                ],
                                "chunk_index": chunk.chunk_index,
                                "confidence": chunk_entity.confidence_score or 0.0,
                                "grounding_quote": chunk_entity.grounding_quote,
                            }
                        )

        entity_response = DocumentEntityResponse(
            entity_id=str(entity_row.id),
            name=entity_row.name,
            entity_type=entity_row.entity_type,
            description=entity_row.description,
            confidence=float(entity_row.confidence or 0.0),
            properties=entity_row.properties or {},
            spans=spans,
            first_mentioned=entity_row.created_at,
        )

        entity_responses.append(entity_response)

    response = DocumentEntitiesResponse(
        document_id=str(document_id),
        entities=entity_responses,
        total=total,
        domain=document.domain if document else None,
    )

    return APIResponse(
        success=True,
        data=response,
        error=None,
        metadata={
            "document_id": str(document_id),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        },
    )


@router.post("/:search", response_model=APIResponse[DocumentSearchResponse])
async def search_documents(
    request: DocumentSearchRequest = Body(...), db: AsyncSession = Depends(get_db)
):
    """
    POST /api/v1/documents:search

    Hybrid search combining vector similarity (semantic) and keyword search.
    Uses Reciprocal Rank Fusion for result merging with weights:
    - Vector search: 0.7 weight (semantic meaning)
    - Keyword search: 0.3 weight (exact matches)

    AIP-136 Custom Method for search operations.
    """
    import time

    start_time = time.time()

    # This is a placeholder implementation
    # Real implementation would integrate with VectorStore and implement RRF

    results = []
    total_results = 0

    if request.search_type in ["vector", "hybrid"]:
        # Vector search would go here - using embeddings from VectorStore
        pass

    if request.search_type in ["keyword", "hybrid"]:
        # Keyword search on entity names would go here
        pass

    took_ms = int((time.time() - start_time) * 1000)

    response = DocumentSearchResponse(
        results=results,
        total=total_results,
        query=request.query,
        took_ms=took_ms,
        search_type=request.search_type,
    )

    return APIResponse(
        success=True,
        data=response,
        error=None,
        metadata={
            "search_type": request.search_type,
            "filters_applied": bool(
                request.domains
                or request.entity_types
                or request.date_from
                or request.date_to
            ),
        },
    )


@router.post("/{document_id}/annotations", response_model=APIResponse[W3CAnnotation])
async def create_annotation(
    document_id: UUID = Path(..., description="Document UUID"),
    annotation: W3CAnnotation = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/v1/documents/{document_id}/annotations

    Create a W3C Web Annotation for a document excerpt.
    Supports both TextPositionSelector and TextQuoteSelector.

    AIP-136 Custom Method for creating annotations.
    """
    # Verify document exists
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    document = doc_result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Validate annotation structure
    if not annotation.target or "selector" not in annotation.target:
        raise HTTPException(
            status_code=400, detail="Annotation must have target with selector"
        )

    # In production, this would save the annotation to database
    # For now, return the annotation as-is

    return APIResponse(
        success=True,
        data=annotation,
        error=None,
        metadata={
            "document_id": str(document_id),
            "annotation_id": annotation.id or "generated-id",
            "created": datetime.utcnow().isoformat(),
        },
    )

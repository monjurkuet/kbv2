"""Unified Search API providing a single endpoint for all search modes.

This module implements a comprehensive search API that supports multiple
search strategies including vector search, BM25 keyword search, hybrid
search combining both methods, and reranked search with cross-encoder
re-ranking for improved relevance.
"""

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """Search mode enumeration.

    Attributes:
        VECTOR: Pure semantic similarity search using embeddings.
        BM25: Keyword-based search using BM25 algorithm.
        HYBRID: Combined vector and BM25 search with weighted fusion.
        RERANKED: Hybrid search with cross-encoder re-ranking.
    """

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class UnifiedSearchRequest(BaseModel):
    """Request model for unified search endpoint.

    Attributes:
        query: The search query string.
        mode: Search mode to use (default: hybrid).
        top_k: Maximum number of results to return.
        filters: Optional metadata filters to apply.
        domain: Optional domain restriction.
        vector_weight: Weight for vector scores in fusion (0.0-1.0).
        bm25_weight: Weight for BM25 scores in fusion (0.0-1.0).
        use_reranking: Whether to apply cross-encoder re-ranking.
        initial_top_k: Number of candidates to retrieve before re-ranking.
    """

    query: str = Field(..., description="Search query string", min_length=1)
    mode: SearchMode = Field(
        default=SearchMode.HYBRID, description="Search mode to use"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Max results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters"
    )
    domain: Optional[str] = Field(default=None, description="Domain restriction")
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Vector score weight"
    )
    bm25_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="BM25 score weight"
    )
    use_reranking: bool = Field(
        default=True, description="Apply cross-encoder re-ranking"
    )
    initial_top_k: int = Field(
        default=50, ge=1, le=200, description="Candidates before re-ranking"
    )


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search endpoint.

    Attributes:
        query: Original search query.
        mode: Search mode used.
        total_results: Total number of results found.
        results: List of search results.
        processing_time_ms: Processing time in milliseconds.
        metadata: Additional response metadata.
    """

    query: str = Field(..., description="Original search query")
    mode: SearchMode = Field(..., description="Search mode used")
    total_results: int = Field(..., description="Total results found")
    results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Search results"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


router = APIRouter(prefix="/unified-search", tags=["Unified Search"])


def get_search_services() -> Dict[str, Any]:
    """Get search service instances.

    Returns:
        Dictionary containing initialized search service instances.
    """
    from knowledge_base.persistence.v1.vector_store import VectorStore
    from knowledge_base.storage.bm25_index import BM25Index
    from knowledge_base.storage.hybrid_search import HybridSearchEngine
    from knowledge_base.reranking import CrossEncoderReranker, RerankingPipeline

    vector_store = VectorStore()
    bm25_index = BM25Index()
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )
    cross_encoder = CrossEncoderReranker()
    reranking_pipeline = RerankingPipeline(
        hybrid_search=hybrid_engine,
        cross_encoder=cross_encoder,
    )

    return {
        "vector_store": vector_store,
        "bm25_index": bm25_index,
        "hybrid_engine": hybrid_engine,
        "reranking_pipeline": reranking_pipeline,
    }


@router.post("/", response_model=UnifiedSearchResponse)
async def unified_search(
    request: UnifiedSearchRequest,
) -> UnifiedSearchResponse:
    """Execute unified search across all search modes.

    This endpoint provides a single interface for executing searches using
    different strategies. It routes the request to the appropriate search
    implementation based on the specified mode.

    Args:
        request: UnifiedSearchRequest containing search parameters.

    Returns:
        UnifiedSearchResponse with search results and metadata.
    """
    start_time = time.time()
    results: List[Dict[str, Any]] = []

    try:
        services = get_search_services()

        if request.mode == SearchMode.VECTOR:
            results = await _vector_search(request, services)
        elif request.mode == SearchMode.BM25:
            results = await _bm25_search(request, services)
        elif request.mode == SearchMode.RERANKED:
            results = await _reranked_search(request, services)
        else:
            results = await _hybrid_search(request, services)

    except Exception as e:
        logger.error(f"Unified search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )

    processing_time = (time.time() - start_time) * 1000

    return UnifiedSearchResponse(
        query=request.query,
        mode=request.mode,
        total_results=len(results),
        results=results,
        processing_time_ms=processing_time,
        metadata={
            "top_k": request.top_k,
            "filters": request.filters,
            "domain": request.domain,
            "vector_weight": request.vector_weight,
            "bm25_weight": request.bm25_weight,
        },
    )


async def _vector_search(
    request: UnifiedSearchRequest,
    services: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute pure vector similarity search.

    Args:
        request: Search request parameters.
        services: Dictionary of search service instances.

    Returns:
        List of search results from vector search.
    """
    from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient

    vector_store = services["vector_store"]
    embed_client = EmbeddingClient()

    query_embedding = await embed_client.embed_texts([request.query])
    if not query_embedding or not query_embedding[0]:
        return []

    vector_results = await vector_store.search_similar_chunks(
        query_embedding=query_embedding[0],
        limit=request.top_k,
        similarity_threshold=0.0,
    )

    return [
        {
            "id": str(r.get("id", "")),
            "text": r.get("text", ""),
            "score": r.get("similarity", 0.0),
            "metadata": r.get("metadata"),
            "source": "vector",
        }
        for r in vector_results
    ]


async def _bm25_search(
    request: UnifiedSearchRequest,
    services: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute BM25 keyword search.

    Args:
        request: Search request parameters.
        services: Dictionary of search service instances.

    Returns:
        List of search results from BM25 search.
    """
    bm25_index = services["bm25_index"]

    bm25_results = await bm25_index.search(
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )

    return [
        {
            "id": r.id,
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "source": "bm25",
        }
        for r in bm25_results
    ]


async def _hybrid_search(
    request: UnifiedSearchRequest,
    services: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute hybrid search combining vector and BM25.

    Args:
        request: Search request parameters.
        services: Dictionary of search service instances.

    Returns:
        List of search results from hybrid search.
    """
    hybrid_engine = services["hybrid_engine"]

    search_results = await hybrid_engine.search(
        query=request.query,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight,
        top_k=request.top_k,
        filters=request.filters,
        domain=request.domain,
    )

    return [
        {
            "id": r.id,
            "text": r.text,
            "vector_score": r.vector_score,
            "bm25_score": r.bm25_score,
            "final_score": r.final_score,
            "metadata": r.metadata,
            "source": r.source,
        }
        for r in search_results
    ]


async def _reranked_search(
    request: UnifiedSearchRequest,
    services: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute re-ranked search with cross-encoder.

    Args:
        request: Search request parameters.
        services: Dictionary of search service instances.

    Returns:
        List of re-ranked search results.
    """
    reranking_pipeline = services["reranking_pipeline"]

    if not request.use_reranking:
        return await _hybrid_search(request, services)

    reranked_results = await reranking_pipeline.search(
        query=request.query,
        initial_top_k=request.initial_top_k,
        final_top_k=request.top_k,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight,
        filters=request.filters,
        domain=request.domain,
    )

    return [
        {
            "id": r.id,
            "text": r.text,
            "reranked_score": r.reranked_score,
            "cross_encoder_score": r.cross_encoder_score,
            "metadata": r.metadata,
            "source": "reranked",
        }
        for r in reranked_results
    ]


@router.get("/health", response_model=Dict[str, Any])
async def search_health() -> Dict[str, Any]:
    """Health check endpoint for search services.

    Returns:
        Dictionary containing health status of each search component.
    """
    try:
        services = get_search_services()

        health_status = {
            "status": "healthy",
            "services": {
                "vector": "available",
                "bm25": "available",
                "hybrid": "available",
                "reranking": "available",
            },
        }

        vector_store = services["vector_store"]
        bm25_index = services["bm25_index"]

        try:
            bm25_stats = await bm25_index.get_stats()
            health_status["services"]["bm25"] = (
                "healthy" if bm25_stats.get("indexed", False) else "empty"
            )
        except Exception as e:
            health_status["services"]["bm25"] = f"error: {str(e)}"

        return health_status

    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return {
            "status": "unhealthy",
            "services": {
                "vector": "unavailable",
                "bm25": "unavailable",
                "hybrid": "unavailable",
                "reranking": "unavailable",
            },
            "error": str(e),
        }


@router.get("/modes", response_model=List[Dict[str, Any]])
async def get_search_modes() -> List[Dict[str, Any]]:
    """Get available search modes and their descriptions.

    Returns:
        List of search mode configurations.
    """
    return [
        {
            "mode": SearchMode.VECTOR.value,
            "description": "Pure semantic similarity search using vector embeddings",
            "parameters": ["query", "top_k", "filters", "domain"],
        },
        {
            "mode": SearchMode.BM25.value,
            "description": "Keyword-based search using BM25 algorithm",
            "parameters": ["query", "top_k", "filters"],
        },
        {
            "mode": SearchMode.HYBRID.value,
            "description": "Combined vector and BM25 search with weighted fusion",
            "parameters": [
                "query",
                "top_k",
                "filters",
                "domain",
                "vector_weight",
                "bm25_weight",
            ],
        },
        {
            "mode": SearchMode.RERANKED.value,
            "description": "Hybrid search with cross-encoder re-ranking",
            "parameters": [
                "query",
                "top_k",
                "filters",
                "domain",
                "vector_weight",
                "bm25_weight",
                "use_reranking",
                "initial_top_k",
            ],
        },
    ]

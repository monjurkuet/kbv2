"""Search endpoints for KBV2."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from knowledge_base.routes.dependencies import get_dependencies


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["search"])


# Request/Response models
class SearchRequest(BaseModel):
    """Search request."""

    query: str
    limit: int = 10
    domain: Optional[str] = None
    use_reranking: bool = True


@router.post("/search")
async def search(request: SearchRequest):
    """Search for documents using hybrid search with optional reranking.

    By default, uses cross-encoder reranking for improved relevance.
    Set use_reranking=false for faster basic hybrid search.
    """
    deps = get_dependencies()
    if not deps.hybrid_search:
        raise RuntimeError("Search engine not initialized")

    # Generate query embedding for vector search
    query_embedding = None
    if deps.embedding:
        try:
            query_embedding = await deps.embedding.embed_query(request.query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")

    # Use reranking pipeline if available and requested
    if request.use_reranking and deps.reranking and query_embedding:
        try:
            results = await deps.reranking.search(
                query=request.query,
                initial_top_k=max(request.limit * 2, 20),
                final_top_k=request.limit,
                query_embedding=query_embedding,
            )

            return {
                "results": [
                    {
                        "chunk_id": r.id,
                        "document_id": r.metadata.get("document_id", "") if r.metadata else "",
                        "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                        "score": r.reranked_score,
                        "cross_encoder_score": r.cross_encoder_score,
                        "bm25_score": r.bm25_score,
                        "vector_score": r.vector_score,
                    }
                    for r in results
                ],
                "query": request.query,
                "count": len(results),
                "reranking_used": True,
            }
        except Exception as e:
            logger.warning(f"Reranking failed, falling back to basic search: {e}")

    # Fallback to basic hybrid search
    results = await deps.hybrid_search.search(
        query=request.query,
        query_embedding=query_embedding,
        limit=request.limit,
    )

    return {
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.score,
                "bm25_score": r.bm25_score,
                "vector_score": r.vector_score,
                "document_name": r.document_name,
            }
            for r in results
        ],
        "query": request.query,
        "count": len(results),
        "reranking_used": False,
    }


@router.post("/search/reranked")
async def search_reranked(request: SearchRequest):
    """Search with cross-encoder reranking for maximum relevance."""
    deps = get_dependencies()
    if not deps.reranking or not deps.embedding:
        raise HTTPException(
            status_code=503, detail="Reranking pipeline or embedding client not available"
        )

    try:
        query_embedding = await deps.embedding.embed_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate embedding: {e}")

    results = await deps.reranking.search(
        query=request.query,
        initial_top_k=max(request.limit * 2, 20),
        final_top_k=request.limit,
        query_embedding=query_embedding,
    )

    return {
        "results": [
            {
                "chunk_id": r.id,
                "document_id": r.metadata.get("document_id", "") if r.metadata else "",
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.reranked_score,
                "cross_encoder_score": r.cross_encoder_score,
                "bm25_score": r.bm25_score,
                "vector_score": r.vector_score,
            }
            for r in results
        ],
        "query": request.query,
        "count": len(results),
    }

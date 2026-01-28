"""
FastAPI endpoints for the Natural Language Query Interface.
Implements Google Python style guide with type hints and comprehensive docstrings.
"""

from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, Field
from .text_to_sql_agent import TextToSQLAgent
import os
from dotenv import load_dotenv

from knowledge_base.intelligence import (
    FederatedQueryRouter,
    HybridEntityRetriever,
    QueryDomain,
    ExecutionStrategy,
    FederatedQueryPlan,
    FederatedQueryResult,
)
from knowledge_base.persistence.v1.vector_store import VectorStore
from knowledge_base.persistence.v1.graph_store import GraphStore
from knowledge_base.storage.bm25_index import BM25Index
from knowledge_base.storage.hybrid_search import HybridSearchEngine
from knowledge_base.reranking import (
    CrossEncoderReranker,
    RerankingPipeline,
    RerankedSearchResult,
    RerankedSearchResultWithExplanation,
)

# Load environment variables
load_dotenv()

# Database configuration
database_url = os.getenv("DATABASE_URL", "sqlite:///./knowledge_base.db")
engine = create_engine(database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create router
router = APIRouter(
    prefix="/api/v1/query",
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


def get_text_to_sql_agent() -> TextToSQLAgent:
    """Dependency to get TextToSQLAgent instance.

    Returns:
        TextToSQLAgent: Initialized agent for translating NL to SQL.
    """
    return TextToSQLAgent(engine)


@router.post("/translate", response_model=Dict)
async def translate_query(
    nl_query: str, agent: TextToSQLAgent = Depends(get_text_to_sql_agent)
) -> Dict:
    """Translate natural language query to SQL without executing it.

    Args:
        nl_query: Natural language query string.
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing:
            - sql: The generated SQL statement
            - warnings: List of validation warnings
            - error: Error message (if any)
    """
    try:
        sql, warnings = agent.translate(nl_query)
        return {"sql": sql, "warnings": warnings, "error": None}
    except Exception as e:
        return {"sql": "", "warnings": [], "error": str(e)}


@router.post("/execute", response_model=Dict)
async def execute_query(
    nl_query: str, agent: TextToSQLAgent = Depends(get_text_to_sql_agent)
) -> Dict:
    """Translate and execute natural language query.

    Args:
        nl_query: Natural language query string.
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing:
            - sql: The generated SQL statement
            - results: Query results (if successful)
            - warnings: List of validation warnings
            - error: Error message (if any)
    """
    result = agent.execute_query(nl_query)
    return result


@router.get("/schema", response_model=Dict)
async def get_schema(agent: TextToSQLAgent = Depends(get_text_to_sql_agent)) -> Dict:
    """Get database schema information.

    Args:
        agent: TextToSQLAgent instance (injected by dependency).

    Returns:
        Dictionary containing table names and their columns with data types.
    """
    return agent.schema_cache


class ExecutionStrategyEnum(str, Enum):
    """Execution strategy enum for federated queries."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY = "priority"


class FederatedQueryRequest(BaseModel):
    """Request model for federated query."""

    query: str = Field(..., description="Natural language query")
    max_domains: int = Field(
        default=3, ge=1, le=6, description="Maximum domains to query"
    )
    strategy: ExecutionStrategyEnum = Field(
        default=ExecutionStrategyEnum.PARALLEL, description="Execution strategy"
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class FederatedQueryResponse(BaseModel):
    """Response model for federated query."""

    query: str
    detected_domains: List[Dict[str, Any]]
    strategy: str
    total_results: int
    domain_results: List[Dict[str, Any]]


class RerankedSearchResponse(BaseModel):
    """Response model for reranked search.

    Attributes:
        query: Original search query.
        total_results: Total number of results found.
        results: List of reranked search results.
        initial_top_k: Number of candidates retrieved initially.
        final_top_k: Number of results after reranking.
        reranking_model: Cross-encoder model used for reranking.
    """

    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total results found")
    results: List[RerankedSearchResult] = Field(
        ..., description="Reranked search results"
    )
    initial_top_k: int = Field(
        ..., description="Number of candidates retrieved initially"
    )
    final_top_k: int = Field(..., description="Number of results after reranking")
    reranking_model: str = Field(..., description="Cross-encoder model used")


class RerankedSearchRequest(BaseModel):
    """Request model for reranked search.

    Attributes:
        query: Search query string.
        initial_top_k: Number of candidates to retrieve from hybrid search.
        final_top_k: Number of results to return after reranking.
        vector_weight: Weight for vector similarity in fusion.
        bm25_weight: Weight for BM25 in fusion.
        filters: Optional metadata filters.
        domain: Optional domain filter.
    """

    query: str = Field(..., description="Search query")
    initial_top_k: int = Field(
        default=50, ge=1, le=200, description="Initial candidates to retrieve"
    )
    final_top_k: int = Field(
        default=10, ge=1, le=50, description="Final results after reranking"
    )
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    domain: Optional[str] = Field(None, description="Domain filter")


class HybridSearchRequestV2(BaseModel):
    """Request model for hybrid search v2 (BM25 + Vector + Reranking).

    This is the enhanced hybrid search request that combines BM25 keyword
    search with vector similarity search for comprehensive document retrieval.

    Attributes:
        query: Search query string.
        domain: Optional domain filter.
        limit: Maximum number of results.
        vector_weight: Weight for vector similarity scores.
        bm25_weight: Weight for BM25 scores.
        enable_reranking: Whether to enable cross-encoder reranking.
        filters: Optional metadata filters.
    """

    query: str = Field(..., description="Search query")
    domain: Optional[str] = Field(None, description="Filter by domain")
    limit: int = Field(default=20, ge=1, le=100)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_reranking: bool = Field(
        default=False, description="Enable cross-encoder reranking"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")


class HybridSearchResultV2(BaseModel):
    """Individual result from hybrid search v2.

    Attributes:
        id: Document or chunk ID.
        text: Text content.
        vector_score: Normalized vector similarity score.
        bm25_score: Normalized BM25 score.
        final_score: Combined fusion score.
        metadata: Optional metadata.
        source: Result source (vector, bm25, or hybrid).
    """

    id: str = Field(..., description="Document or chunk ID")
    text: str = Field(..., description="Text content")
    vector_score: float = Field(default=0.0, description="Normalized vector similarity")
    bm25_score: float = Field(default=0.0, description="Normalized BM25 score")
    final_score: float = Field(..., description="Combined fusion score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )
    source: str = Field(
        default="hybrid", description="Result source: vector, bm25, or hybrid"
    )


class HybridSearchResponseV2(BaseModel):
    """Response model for hybrid search v2.

    Attributes:
        query: Original search query.
        total_results: Total number of results found.
        results: List of hybrid search results.
        weights: Weights used for fusion.
        reranked: Whether reranking was applied.
    """

    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total results found")
    results: List[HybridSearchResultV2] = Field(
        ..., description="Hybrid search results"
    )
    weights: Dict[str, float] = Field(..., description="Weights used for fusion")
    reranked: bool = Field(default=False, description="Whether reranking was applied")


async def get_query_services() -> Tuple[FederatedQueryRouter, HybridEntityRetriever]:
    """Dependency to get query services.

    Returns:
        Tuple of FederatedQueryRouter and HybridEntityRetriever.
    """
    vector_store = VectorStore()
    graph_store = GraphStore()
    retriever = HybridEntityRetriever(
        vector_store=vector_store, graph_store=graph_store
    )
    router = FederatedQueryRouter(retriever=retriever)
    return router, retriever


@router.post("/federated", response_model=FederatedQueryResponse)
async def execute_federated_query(
    request: FederatedQueryRequest,
    services: Tuple[FederatedQueryRouter, HybridEntityRetriever] = Depends(
        get_query_services
    ),
) -> FederatedQueryResponse:
    """Execute query across multiple knowledge domains.

    Args:
        request: FederatedQueryRequest containing query details.
        services: Tuple of FederatedQueryRouter and HybridEntityRetriever.

    Returns:
        FederatedQueryResponse with aggregated results from all domains.
    """
    router, _ = services

    try:
        strategy_map = {
            ExecutionStrategyEnum.SEQUENTIAL: ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategyEnum.PARALLEL: ExecutionStrategy.PARALLEL,
            ExecutionStrategyEnum.PRIORITY: ExecutionStrategy.PRIORITY,
        }

        domain_matches = router._domain_detector.detect(
            request.query,
            max_domains=request.max_domains,
            min_confidence=request.min_confidence,
        )

        detected_domains = [
            {
                "domain": m.domain.value,
                "confidence": m.confidence,
                "keywords": m.keywords,
            }
            for m in domain_matches
        ]

        result = await router.route_and_execute(
            query=request.query,
            strategy=strategy_map.get(request.strategy, ExecutionStrategy.PARALLEL),
        )

        domain_results = []
        for domain, domain_data in result.results.items():
            for item in domain_data:
                domain_results.append({**item, "_domain": domain.value})

        return FederatedQueryResponse(
            query=request.query,
            detected_domains=detected_domains,
            strategy=request.strategy.value,
            total_results=result.total_results,
            domain_results=domain_results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute federated query: {str(e)}",
        )


def get_hybrid_search_engine() -> HybridSearchEngine:
    """Dependency to get HybridSearchEngine instance.

    Returns:
        HybridSearchEngine: Initialized hybrid search engine.
    """
    import os

    db_path = os.getenv("BM25_DB_PATH", "bm25_index.db")

    vector_store = VectorStore()
    bm25_index = BM25Index(db_path=db_path)

    return HybridSearchEngine(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )


@router.post("/hybrid-search-v2", response_model=HybridSearchResponseV2)
async def hybrid_search_v2(
    request: HybridSearchRequestV2,
    engine: HybridSearchEngine = Depends(get_hybrid_search_engine),
) -> HybridSearchResponseV2:
    """Hybrid search with BM25 + vector + optional reranking.

    This endpoint provides enhanced hybrid search combining BM25 keyword
    search with vector similarity search. Results are fused using weighted
    score combination with optional cross-encoder reranking.

    Args:
        request: HybridSearchRequestV2 containing search parameters.
        engine: HybridSearchEngine instance (injected by dependency).

    Returns:
        HybridSearchResponseV2 with combined BM25 and vector results.
    """
    try:
        if abs(request.vector_weight + request.bm25_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and bm25_weight must sum to 1.0",
            )

        if request.enable_reranking:
            results = await engine.search_with_reranking(
                query=request.query,
                initial_top_k=request.limit * 3,
                final_top_k=request.limit,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
                filters=request.filters,
                domain=request.domain,
            )
        else:
            results = await engine.search(
                query=request.query,
                vector_weight=request.vector_weight,
                bm25_weight=request.bm25_weight,
                top_k=request.limit,
                filters=request.filters,
                domain=request.domain,
            )

        search_results = [
            HybridSearchResultV2(
                id=r.id,
                text=r.text,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                final_score=r.final_score,
                metadata=r.metadata,
                source=r.source,
            )
            for r in results
        ]

        return HybridSearchResponseV2(
            query=request.query,
            total_results=len(search_results),
            results=search_results,
            weights={
                "vector": request.vector_weight,
                "bm25": request.bm25_weight,
            },
            reranked=request.enable_reranking,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute hybrid search v2: {str(e)}",
        )


def get_reranking_pipeline() -> RerankingPipeline:
    """Dependency to get RerankingPipeline instance.

    Returns:
        RerankingPipeline: Initialized reranking pipeline.
    """
    import os

    db_path = os.getenv("BM25_DB_PATH", "bm25_index.db")

    vector_store = VectorStore()
    bm25_index = BM25Index(db_path=db_path)
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        bm25_index=bm25_index,
    )
    cross_encoder = CrossEncoderReranker()

    return RerankingPipeline(
        hybrid_search=hybrid_engine,
        cross_encoder=cross_encoder,
    )


@router.post("/reranked-search", response_model=RerankedSearchResponse)
async def reranked_search(
    request: RerankedSearchRequest,
    pipeline: RerankingPipeline = Depends(get_reranking_pipeline),
) -> RerankedSearchResponse:
    """Reranked search using cross-encoder.

    This endpoint performs hybrid search followed by cross-encoder
    reranking for improved search result quality.

    Args:
        request: RerankedSearchRequest containing search parameters.
        pipeline: RerankingPipeline instance (injected by dependency).

    Returns:
        RerankedSearchResponse with cross-encoder scored results.
    """
    try:
        if abs(request.vector_weight + request.bm25_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and bm25_weight must sum to 1.0",
            )

        if request.initial_top_k < request.final_top_k:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="initial_top_k must be >= final_top_k",
            )

        results = await pipeline.search(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_k=request.final_top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight,
            filters=request.filters,
            domain=request.domain,
        )

        return RerankedSearchResponse(
            query=request.query,
            total_results=len(results),
            results=results,
            initial_top_k=request.initial_top_k,
            final_top_k=request.final_top_k,
            reranking_model=pipeline._cross_encoder.model_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute reranked search: {str(e)}",
        )


@router.post(
    "/reranked-search-explain", response_model=List[RerankedSearchResultWithExplanation]
)
async def reranked_search_with_explanation(
    request: RerankedSearchRequest,
    pipeline: RerankingPipeline = Depends(get_reranking_pipeline),
) -> List[RerankedSearchResultWithExplanation]:
    """Reranked search with detailed explanations.

    Returns results with human-readable explanations of why each
    document was ranked as it was.

    Args:
        request: RerankedSearchRequest containing search parameters.
        pipeline: RerankingPipeline instance.

    Returns:
        List of RerankedSearchResultWithExplanation objects.
    """
    try:
        if abs(request.vector_weight + request.bm25_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and bm25_weight must sum to 1.0",
            )

        results = await pipeline.search_with_explanation(
            query=request.query,
            initial_top_k=request.initial_top_k,
            final_top_k=request.final_top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight,
            filters=request.filters,
            domain=request.domain,
        )

        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute reranked search with explanation: {str(e)}",
        )

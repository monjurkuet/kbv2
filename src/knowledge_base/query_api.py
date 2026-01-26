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


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""

    query: str = Field(..., description="Search query")
    domain: Optional[str] = Field(None, description="Filter by domain")
    limit: int = Field(default=20, ge=1, le=100)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    graph_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class FederatedQueryResponse(BaseModel):
    """Response model for federated query."""

    query: str
    detected_domains: List[Dict[str, Any]]
    strategy: str
    total_results: int
    domain_results: List[Dict[str, Any]]


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""

    query: str
    total_results: int
    entities: List[Dict[str, Any]]


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


@router.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    services: Tuple[FederatedQueryRouter, HybridEntityRetriever] = Depends(
        get_query_services
    ),
) -> HybridSearchResponse:
    """Perform hybrid vector + graph search.

    Args:
        request: HybridSearchRequest containing search parameters.
        services: Tuple of FederatedQueryRouter and HybridEntityRetriever.

    Returns:
        HybridSearchResponse with combined vector and graph results.
    """
    _, retriever = services

    try:
        if abs(request.vector_weight + request.graph_weight - 1.0) > 1e-6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="vector_weight and graph_weight must sum to 1.0",
            )

        retriever._vector_weight = request.vector_weight
        retriever._graph_weight = request.graph_weight

        result = await retriever.retrieve(
            query=request.query,
            query_embedding=[],
            domain=request.domain,
            vector_limit=request.limit,
        )

        entities = [
            {
                "id": str(e.id),
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "confidence": e.confidence,
                "vector_score": e.vector_score,
                "graph_score": e.graph_score,
                "graph_hops": e.graph_hops,
                "final_score": e.final_score,
                "source": e.source,
            }
            for e in result.entities
        ]

        return HybridSearchResponse(
            query=request.query,
            total_results=len(entities),
            entities=entities,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute hybrid search: {str(e)}",
        )

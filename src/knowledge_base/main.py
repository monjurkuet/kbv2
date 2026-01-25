"""
Main FastAPI application entry point for KBV2 Knowledge Base API.

This module orchestrates all API routers (query, review, graph, document) and
configures global middleware and exception handlers for enterprise-grade API
functionality following Google AIP standards.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base.common.aip193_middleware import AIP193ResponseMiddleware
from knowledge_base.common.error_handlers import setup_exception_handlers
from knowledge_base import query_api, review_api, graph_api, document_api


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="KBV2 Knowledge Base API",
    description="High-fidelity information extraction and graph visualization API",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next: Any):
    """
    Middleware to generate and track request IDs for observability.

    Adds x-request-id header to all requests for tracing through the system.
    """
    import uuid

    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["x-request-id"] = request_id

    return response


app.add_middleware(AIP193ResponseMiddleware)


setup_exception_handlers(app)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    name: str


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        Basic service health information
    """
    return HealthResponse(
        status="healthy", version="1.0.0", name="KBV2 Knowledge Base API"
    )


@app.get("/ready", response_model=HealthResponse, tags=["health"])
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes deployments.

    Returns:
        Service readiness status
    """
    return HealthResponse(
        status="ready", version="1.0.0", name="KBV2 Knowledge Base API"
    )


app.include_router(query_api.router, tags=["query"])
app.include_router(review_api.router, tags=["review"])
app.include_router(graph_api.router, tags=["graphs"])
app.include_router(document_api.router, tags=["documents"])


@app.get("/api/v1/openapi")
async def get_openapi():
    """
    Alternative OpenAPI endpoint for better client generation.

    Returns:
        Complete OpenAPI specification
    """
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="KBV2 Knowledge Base API",
        version="1.0.0",
        description="High-fidelity information extraction and graph visualization API",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler for application initialization.

    Runs when the FastAPI application starts up.
    """
    logger.info("KBV2 Knowledge Base API starting up...")

    try:
        from knowledge_base.persistence.v1.schema import Base
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        import os

        database_url = os.getenv("DATABASE_URL")
        if database_url:
            engine = create_engine(database_url)
            Base.metadata.create_all(engine)
            logger.info("Database tables verified/created successfully")

            # Setup async session factory
            from knowledge_base.common.dependencies import set_session_factory

            async_engine = create_async_engine(
                database_url.replace("postgresql://", "postgresql+asyncpg://")
            )
            session_factory = async_sessionmaker(
                async_engine, class_=AsyncSession, expire_on_commit=False
            )
            set_session_factory(session_factory)
            logger.info("Async session factory initialized")
        else:
            logger.warning("DATABASE_URL not set, skipping table creation")

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)

    logger.info("KBV2 API startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler for graceful cleanup.

    Runs when the FastAPI application shuts down.
    """
    logger.info("KBV2 Knowledge Base API shutting down...")
    # Add cleanup logic here (database connections, etc.)
    logger.info("KBV2 API shutdown complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["."],
        log_config=None,  # Use our custom logging configuration
    )

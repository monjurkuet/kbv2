"""
Main FastAPI application entry point for KBV2 Knowledge Base API.

This module orchestrates all API routers (query, review, graph, document) and
configures global middleware and exception handlers for enterprise-grade API
functionality following Google AIP standards.
"""

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_base.common.aip193_middleware import AIP193ResponseMiddleware
from knowledge_base.common.error_handlers import setup_exception_handlers
from knowledge_base import (
    query_api,
    review_api,
    graph_api,
    document_api,
    mcp_server,
    schema_api,
)


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
app.include_router(schema_api.router, tags=["schemas"])

from knowledge_base.mcp_server import kbv2_protocol


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await kbv2_protocol.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            await kbv2_protocol.handle_message(websocket, message)
    except WebSocketDisconnect:
        kbv2_protocol.disconnect(websocket)


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


# Mount static files for the Windows XP WebSocket client
# Mount static files for the Windows XP WebSocket client
import os
dashboard_version = os.getenv("DASHBOARD_VERSION", "v1")
base_static_dir = Path(__file__).parent / "static"

if dashboard_version == "v2":
    static_dir = base_static_dir / "shizuku_xp" / "build"
else:
    static_dir = base_static_dir / "v1_legacy"

if not static_dir.exists():
    # Fallback or just log
    logger.warning(f"Static directory {static_dir} does not exist. Dashboard may not load.")
else:
    logger.info(f"Mounting dashboard version {dashboard_version} from {static_dir}")
    app.mount("/dashboard", StaticFiles(directory=str(static_dir), html=True), name="static")


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

        from knowledge_base.mcp_server import kbv2_protocol

        await kbv2_protocol.initialize()
        logger.info("MCP server initialized")

        from knowledge_base.intelligence import (
            SchemaRegistry,
            EntityTypeDef,
            DomainAttribute,
            InheritanceType,
        )
        from knowledge_base.common.dependencies import get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as db:
            registry = SchemaRegistry(db)
            existing = await registry.list_schemas()

            if not existing:
                logger.info("Initializing domain schemas...")

                DOMAIN_CONFIGS = [
                    {
                        "domain_name": "GENERAL",
                        "domain_display_name": "General Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="NamedEntity",
                                parent_type="OTHER",
                                attributes={},
                            )
                        ],
                        "parent_domain_name": None,
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "TECHNOLOGY",
                        "domain_display_name": "Technology Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Software",
                                parent_type="PRODUCT",
                                attributes={
                                    "version": DomainAttribute(
                                        name="version", attribute_type="str"
                                    ),
                                    "license": DomainAttribute(
                                        name="license", attribute_type="str"
                                    ),
                                    "programming_language": DomainAttribute(
                                        name="programming_language",
                                        attribute_type="List[str]",
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="API",
                                parent_type="CONCEPT",
                                attributes={
                                    "endpoint": DomainAttribute(
                                        name="endpoint", attribute_type="str"
                                    ),
                                    "method": DomainAttribute(
                                        name="method", attribute_type="str"
                                    ),
                                    "status": DomainAttribute(
                                        name="status", attribute_type="str"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="Framework",
                                parent_type="CONCEPT",
                                attributes={
                                    "language": DomainAttribute(
                                        name="language", attribute_type="str"
                                    ),
                                    "version": DomainAttribute(
                                        name="version", attribute_type="str"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "FINANCIAL",
                        "domain_display_name": "Financial Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Company",
                                parent_type="ORGANIZATION",
                                attributes={
                                    "ticker_symbol": DomainAttribute(
                                        name="ticker_symbol", attribute_type="str"
                                    ),
                                    "market_cap": DomainAttribute(
                                        name="market_cap", attribute_type="float"
                                    ),
                                    "stock_exchange": DomainAttribute(
                                        name="stock_exchange", attribute_type="str"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="FinancialInstrument",
                                parent_type="PRODUCT",
                                attributes={
                                    "isin": DomainAttribute(
                                        name="isin", attribute_type="str"
                                    ),
                                    "currency": DomainAttribute(
                                        name="currency", attribute_type="str"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "MEDICAL",
                        "domain_display_name": "Medical Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Drug",
                                parent_type="PRODUCT",
                                attributes={
                                    "active_ingredients": DomainAttribute(
                                        name="active_ingredients",
                                        attribute_type="List[str]",
                                    ),
                                    "dosage": DomainAttribute(
                                        name="dosage", attribute_type="str"
                                    ),
                                    "manufacturer": DomainAttribute(
                                        name="manufacturer", attribute_type="str"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="Procedure",
                                parent_type="EVENT",
                                attributes={
                                    "procedure_type": DomainAttribute(
                                        name="procedure_type", attribute_type="str"
                                    ),
                                    "outcome": DomainAttribute(
                                        name="outcome", attribute_type="str"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "LEGAL",
                        "domain_display_name": "Legal Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Contract",
                                parent_type="CONCEPT",
                                attributes={
                                    "parties": DomainAttribute(
                                        name="parties", attribute_type="List[str]"
                                    ),
                                    "effective_date": DomainAttribute(
                                        name="effective_date", attribute_type="datetime"
                                    ),
                                    "expiration_date": DomainAttribute(
                                        name="expiration_date",
                                        attribute_type="datetime",
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="Court",
                                parent_type="ORGANIZATION",
                                attributes={
                                    "jurisdiction": DomainAttribute(
                                        name="jurisdiction", attribute_type="str"
                                    ),
                                    "level": DomainAttribute(
                                        name="level", attribute_type="str"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "HEALTHCARE",
                        "domain_display_name": "Healthcare Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="HealthcareProvider",
                                parent_type="ORGANIZATION",
                                attributes={
                                    "specialty": DomainAttribute(
                                        name="specialty", attribute_type="str"
                                    ),
                                    "accreditation": DomainAttribute(
                                        name="accreditation", attribute_type="List[str]"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="InsurancePlan",
                                parent_type="PRODUCT",
                                attributes={
                                    "coverage_type": DomainAttribute(
                                        name="coverage_type", attribute_type="str"
                                    ),
                                    "premium": DomainAttribute(
                                        name="premium", attribute_type="float"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "ACADEMIC",
                        "domain_display_name": "Academic Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Publication",
                                parent_type="CONCEPT",
                                attributes={
                                    "title": DomainAttribute(
                                        name="title", attribute_type="str"
                                    ),
                                    "authors": DomainAttribute(
                                        name="authors", attribute_type="List[str]"
                                    ),
                                    "publication_date": DomainAttribute(
                                        name="publication_date",
                                        attribute_type="datetime",
                                    ),
                                    "venue": DomainAttribute(
                                        name="venue", attribute_type="str"
                                    ),
                                    "citation_count": DomainAttribute(
                                        name="citation_count", attribute_type="int"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="ResearchField",
                                parent_type="CONCEPT",
                                attributes={
                                    "discipline": DomainAttribute(
                                        name="discipline", attribute_type="str"
                                    ),
                                    "subdisciplines": DomainAttribute(
                                        name="subdisciplines",
                                        attribute_type="List[str]",
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                    {
                        "domain_name": "SCIENTIFIC",
                        "domain_display_name": "Scientific Domain",
                        "entity_types": [
                            EntityTypeDef(
                                type_name="Theory",
                                parent_type="CONCEPT",
                                attributes={
                                    "formulated_by": DomainAttribute(
                                        name="formulated_by", attribute_type="str"
                                    ),
                                    "year_proposed": DomainAttribute(
                                        name="year_proposed", attribute_type="int"
                                    ),
                                    "status": DomainAttribute(
                                        name="status", attribute_type="str"
                                    ),
                                },
                            ),
                            EntityTypeDef(
                                type_name="Experiment",
                                parent_type="EVENT",
                                attributes={
                                    "hypothesis": DomainAttribute(
                                        name="hypothesis", attribute_type="str"
                                    ),
                                    "methodology": DomainAttribute(
                                        name="methodology", attribute_type="str"
                                    ),
                                    "results": DomainAttribute(
                                        name="results", attribute_type="str"
                                    ),
                                },
                            ),
                        ],
                        "parent_domain_name": "GENERAL",
                        "inheritance_type": InheritanceType.EXTENDS,
                    },
                ]

                from knowledge_base.intelligence.v1.domain_schema_service import (
                    DomainSchemaCreate,
                    DomainLevel,
                )

                for config in DOMAIN_CONFIGS:
                    # Convert list of EntityTypeDef to dict with type_name as key
                    entity_types_dict = {et.type_name: et for et in config["entity_types"]}
                    schema = DomainSchemaCreate(
                        domain_name=config["domain_name"],
                        domain_display_name=config["domain_display_name"],
                        entity_types=entity_types_dict,
                        parent_domain_name=config["parent_domain_name"],
                        inheritance_type=config["inheritance_type"],
                        domain_level=DomainLevel.PRIMARY,
                    )
                    await registry.register(schema)

                logger.info(f"Initialized {len(DOMAIN_CONFIGS)} domain schemas")
            else:
                logger.info(f"Found {len(existing)} existing domain schemas")

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
        host="localhost",
        port=8765,
        reload=True,
        reload_dirs=["."],
        log_config=None,  # Use our custom logging configuration
    )

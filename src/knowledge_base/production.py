#!/usr/bin/env python3
"""
KBv2 Crypto Knowledgebase - Production Entry Point

This is the main production application that integrates:
- Self-improving orchestrator with Experience Bank
- Prompt Evolution for automated optimization
- Ontology Validation for quality assurance
- Monitoring and metrics endpoints
- Data pipeline integration
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import KBv2 components
from knowledge_base.main import (
    app as base_app,
    startup_event as base_startup,
    shutdown_event as base_shutdown,
)
from knowledge_base.monitoring.metrics import (
    metrics_collector,
    health_checker,
    create_monitoring_app,
)
from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator
from knowledge_base.config.production import apply_preset, get_config

# Global orchestrator instance
orchestrator: SelfImprovingOrchestrator | None = None


async def init_self_improving_orchestrator() -> SelfImprovingOrchestrator:
    """Initialize the self-improving orchestrator with production config."""
    global orchestrator

    config = get_config()
    logger.info(
        f"Initializing SelfImprovingOrchestrator with config: {config.to_dict()}"
    )

    orchestrator = SelfImprovingOrchestrator(
        enable_experience_bank=config.enable_experience_bank,
        enable_prompt_evolution=config.enable_prompt_evolution,
        enable_ontology_validation=config.enable_ontology_validation,
    )

    await orchestrator.initialize()
    logger.info("✅ SelfImprovingOrchestrator initialized successfully")

    return orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("KBv2 Crypto Knowledgebase - Production Startup")
    logger.info("=" * 60)

    # Apply production preset
    prod_config = apply_preset("production")
    logger.info(f"Applied preset: production")
    logger.info(f"  Experience Bank: {prod_config.enable_experience_bank}")
    logger.info(f"  Prompt Evolution: {prod_config.enable_prompt_evolution}")
    logger.info(f"  Ontology Validation: {prod_config.enable_ontology_validation}")

    # Run base startup
    await base_startup()

    # Initialize self-improving orchestrator
    try:
        await init_self_improving_orchestrator()

        # Register health checks
        health_checker.register_check(
            "self_improving_orchestrator", lambda: orchestrator is not None
        )
        health_checker.register_check(
            "experience_bank",
            lambda: orchestrator is not None
            and orchestrator._experience_bank is not None,
        )

        logger.info("✅ All components initialized successfully")

    except Exception as e:
        logger.error(
            f"❌ Failed to initialize self-improving orchestrator: {e}", exc_info=True
        )
        raise

    yield

    # Shutdown
    logger.info("KBv2 Crypto Knowledgebase - Shutdown initiated")

    if orchestrator:
        logger.info("Shutting down SelfImprovingOrchestrator...")
        # Note: orchestrator cleanup happens via garbage collection or explicit close if needed

    await base_shutdown()
    logger.info("✅ Shutdown complete")


# Create the production app
app = FastAPI(
    title="KBv2 Crypto Knowledgebase - Production",
    description="Self-improving cryptocurrency knowledgebase with Experience Bank, Prompt Evolution, and Ontology Validation",
    version="2.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include all routers from base app
for route in base_app.routes:
    if hasattr(route, "path"):
        app.routes.append(route)


# Additional production endpoints
@app.get("/api/v2/health", tags=["health"])
async def health_check_v2():
    """Extended health check with self-improving components."""
    health = await health_checker.check_health()

    # Add experience bank stats if available
    experience_stats = None
    if orchestrator:
        try:
            experience_stats = await orchestrator.get_experience_bank_stats()
        except Exception as e:
            logger.warning(f"Could not get experience bank stats: {e}")

    return {
        "status": health.status,
        "version": "2.0.0",
        "components": {
            "database": health.status == "healthy",
            "self_improving_orchestrator": orchestrator is not None,
            "experience_bank": experience_stats is not None,
        },
        "experience_bank": experience_stats,
        "timestamp": health.timestamp.isoformat(),
    }


@app.get("/api/v2/stats", tags=["monitoring"])
async def get_stats_v2():
    """Get comprehensive statistics including self-improvement metrics."""
    base_stats = metrics_collector.get_stats()

    # Add experience bank stats
    if orchestrator:
        try:
            exp_stats = await orchestrator.get_experience_bank_stats()
            base_stats["experience_bank_detail"] = exp_stats
        except Exception as e:
            logger.warning(f"Could not get experience bank stats: {e}")

    return base_stats


@app.post("/api/v2/documents/process", tags=["documents"])
async def process_document_v2(file_path: str, domain: str = "GENERAL"):
    """Process a document with self-improving orchestrator."""
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "Self-improving orchestrator not initialized"},
        )

    try:
        document = await orchestrator.process_document(
            file_path=file_path, domain=domain
        )
        return {
            "status": "success",
            "document_id": str(document.id),
            "filename": document.filename,
            "domain": domain,
        }
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/v2/prompts/evolve", tags=["self-improvement"])
async def evolve_prompts(domain: str, test_count: int = 10):
    """Trigger prompt evolution for a domain."""
    if not orchestrator:
        return JSONResponse(
            status_code=503,
            content={"error": "Self-improving orchestrator not initialized"},
        )

    try:
        # Get test documents from experience bank or database
        # For now, return placeholder
        return {
            "status": "not_implemented",
            "message": "Prompt evolution requires test documents. Use orchestrator.evolve_prompts() directly.",
            "domain": domain,
        }
    except Exception as e:
        logger.error(f"Error evolving prompts: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# Mount monitoring app
monitoring_app = create_monitoring_app()


# Combine monitoring routes
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    metrics_data = metrics_collector.get_prometheus_format()
    return PlainTextResponse(content=metrics_data)


if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", "8765"))
    host = os.getenv("HOST", "localhost")

    logger.info(f"Starting KBv2 Crypto Knowledgebase on {host}:{port}")

    # Always use the app object directly when running from __main__
    # This avoids import path issues
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,  # Never use reload in production
        log_config=None,
        access_log=True,
    )

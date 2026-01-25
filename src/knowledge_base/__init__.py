"""Knowledge Base package.

This package provides the core functionality for the Agentic Knowledge Ingestion
& Management System, including orchestration of document ingestion, entity
extraction, and knowledge graph construction.
"""

from knowledge_base.orchestrator import IngestionOrchestrator

def main() -> None:
    """Entry point for the Knowledge Base CLI.

    Starts the FastAPI application defined in the ``knowledge_base.main`` module.
    This function is primarily used as a script entry point for `uv run` and
    standardized enterprise deployments.

    Returns:
        None
    """
    import uvicorn

    # Run the FastAPI app with reload for development convenience.
    uvicorn.run(
        "knowledge_base.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

__all__ = ["IngestionOrchestrator", "main"]

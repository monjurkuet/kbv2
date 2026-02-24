"""Knowledge Base package.

This package provides a portable, self-contained knowledge base system using:
- SQLite + FTS5: Documents, full-text search
- ChromaDB: Vector similarity search
- Kuzu: Knowledge graph storage

All components are file-based and require no external services.
"""

__version__ = "0.2.0"


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
        host="localhost",
        port=8765,
        reload=True,
    )


__all__ = ["main", "__version__"]

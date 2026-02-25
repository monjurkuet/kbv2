"""Health and stats endpoints for KBV2."""

from fastapi import APIRouter

from knowledge_base.routes.dependencies import get_dependencies


# Create router
router = APIRouter(tags=["health"])


class HealthResponse:
    """Health check response."""

    def __init__(self, status: str, version: str, components: dict[str, bool]):
        self.status = status
        self.version = version
        self.components = components


@router.get("/health", response_model=dict, tags=["health"])
async def health_check():
    """Health check endpoint."""
    deps = get_dependencies()
    return {
        "status": "healthy",
        "version": "0.3.0",
        "components": {
            "sqlite": deps.sqlite is not None,
            "chromadb": deps.chroma is not None,
            "kuzu": deps.kuzu is not None,
            "vision_api": deps.vision is not None,
            "embedding": deps.embedding is not None,
            "domain_detector": deps.domain_detector is not None,
            "reranking": deps.reranking is not None,
        },
    }


@router.get("/stats")
async def get_stats():
    """Get storage statistics."""
    deps = get_dependencies()
    stats = {}

    if deps.sqlite:
        stats["sqlite"] = await deps.sqlite.get_stats()
    if deps.chroma:
        stats["chromadb"] = await deps.chroma.get_stats()
    if deps.kuzu:
        stats["kuzu"] = await deps.kuzu.get_stats()

    return stats

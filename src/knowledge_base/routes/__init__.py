"""Routes package for KBV2 API."""

from knowledge_base.routes.health import router as health_router
from knowledge_base.routes.documents import router as documents_router
from knowledge_base.routes.search import router as search_router
from knowledge_base.routes.ingestion import router as ingestion_router
from knowledge_base.routes.domain import router as domain_router
from knowledge_base.routes.graph import router as graph_router


__all__ = [
    "health_router",
    "documents_router",
    "search_router",
    "ingestion_router",
    "domain_router",
    "graph_router",
]

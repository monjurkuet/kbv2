"""Route dependencies for KBV2.

This module provides dependency injection for route modules.
"""

from typing import Optional

from knowledge_base.storage.portable.sqlite_store import SQLiteStore
from knowledge_base.storage.portable.chroma_store import ChromaStore
from knowledge_base.storage.portable.kuzu_store import KuzuGraphStore
from knowledge_base.storage.portable.hybrid_search import HybridSearchEngine
from knowledge_base.clients.embedding import EmbeddingClient
from knowledge_base.clients.llm import AsyncLLMClient
from knowledge_base.ingestion.document_processor import DocumentProcessor
from knowledge_base.ingestion.vision_client import VisionModelClient
from knowledge_base.domain.detection import DomainDetector
from knowledge_base.reranking.reranking_pipeline import RerankingPipeline
from knowledge_base.extraction.pipeline import ExtractionPipeline
from knowledge_base.summaries.community_summaries import CommunitySummarizer


class RouteDeps:
    """Dependency container for route modules.

    This class holds references to all the store and client instances
    that are needed by the route handlers.
    """

    def __init__(
        self,
        sqlite_store: Optional[SQLiteStore] = None,
        chroma_store: Optional[ChromaStore] = None,
        kuzu_store: Optional[KuzuGraphStore] = None,
        hybrid_search: Optional[HybridSearchEngine] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        llm_client: Optional[AsyncLLMClient] = None,
        doc_processor: Optional[DocumentProcessor] = None,
        vision_client: Optional[VisionModelClient] = None,
        domain_detector: Optional[DomainDetector] = None,
        reranking_pipeline: Optional[RerankingPipeline] = None,
        extraction_pipeline: Optional[ExtractionPipeline] = None,
        community_summarizer: Optional[CommunitySummarizer] = None,
    ):
        self.sqlite = sqlite_store
        self.chroma = chroma_store
        self.kuzu = kuzu_store
        self.hybrid_search = hybrid_search
        self.embedding = embedding_client
        self.llm = llm_client
        self.doc_processor = doc_processor
        self.vision = vision_client
        self.domain_detector = domain_detector
        self.reranking = reranking_pipeline
        self.extraction = extraction_pipeline
        self.community_summarizer = community_summarizer


# Global dependency container
_deps: Optional[RouteDeps] = None


def set_dependencies(deps: RouteDeps) -> None:
    """Set the global dependency container."""
    global _deps
    _deps = deps


def get_dependencies() -> RouteDeps:
    """Get the global dependency container."""
    if _deps is None:
        raise RuntimeError("Dependencies not initialized")
    return _deps

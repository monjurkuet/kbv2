"""
Main entry point for Portable Knowledge Base System.

This module provides:
- FastAPI application for REST API
- CLI commands for document ingestion and querying
- Lifespan management for storage components

Routes have been modularized into:
- routes/health.py - Health and stats endpoints
- routes/documents.py - Document CRUD endpoints
- routes/search.py - Search and reranked search endpoints
- routes/ingestion.py - Document ingestion endpoint
- routes/domain.py - Domain detection endpoint
- routes/graph.py - Knowledge graph endpoints
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from knowledge_base.config.loader import load_config
from knowledge_base.routes.dependencies import set_dependencies, RouteDeps
from knowledge_base.routes import (
    health_router,
    documents_router,
    search_router,
    ingestion_router,
    domain_router,
    graph_router,
)
from knowledge_base.storage.portable import (
    SQLiteStore,
    ChromaStore,
    KuzuGraphStore,
    HybridSearchEngine,
    PortableStorageConfig,
)
from knowledge_base.ingestion import VisionModelClient, DocumentProcessor
from knowledge_base.extraction import ExtractionPipeline
from knowledge_base.clients.embedding import EmbeddingClient
from knowledge_base.domain.detection import DomainDetector
from knowledge_base.domain.domain_models import DomainConfig
from knowledge_base.summaries.community_summaries import CommunitySummarizer
from knowledge_base.reranking.reranking_pipeline import RerankingPipeline
from knowledge_base.reranking.cross_encoder import CrossEncoderReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global storage components
_storage_config: Optional[PortableStorageConfig] = None
_sqlite_store: Optional[SQLiteStore] = None
_chroma_store: Optional[ChromaStore] = None
_kuzu_store: Optional[KuzuGraphStore] = None
_search_engine: Optional[HybridSearchEngine] = None
_vision_client: Optional[VisionModelClient] = None
_doc_processor: Optional[DocumentProcessor] = None
_extraction_pipeline: Optional[ExtractionPipeline] = None
_embedding_client: Optional[EmbeddingClient] = None
_domain_detector: Optional[DomainDetector] = None
_community_summarizer: Optional[CommunitySummarizer] = None
_reranking_pipeline: Optional[RerankingPipeline] = None
_cross_encoder: Optional[CrossEncoderReranker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _storage_config, _sqlite_store, _chroma_store, _kuzu_store
    global _search_engine, _vision_client, _doc_processor, _extraction_pipeline
    global _embedding_client, _domain_detector, _community_summarizer
    global _reranking_pipeline, _cross_encoder

    # Startup
    logger.info("Starting Portable Knowledge Base...")

    try:
        # Initialize configuration
        _storage_config = PortableStorageConfig()
        _storage_config.ensure_directories()

        # Initialize storage components
        _sqlite_store = SQLiteStore(_storage_config.sqlite)
        await _sqlite_store.initialize()
        logger.info("SQLite store initialized")

        _chroma_store = ChromaStore(_storage_config.chroma)
        await _chroma_store.initialize()
        logger.info("ChromaDB store initialized")

        _kuzu_store = KuzuGraphStore(_storage_config.kuzu)
        await _kuzu_store.initialize()
        logger.info("Kuzu graph store initialized")

        # Initialize search engine
        _search_engine = HybridSearchEngine(
            sqlite_store=_sqlite_store,
            chroma_store=_chroma_store,
            kuzu_store=_kuzu_store,
            config=_storage_config.hybrid_search,
        )
        logger.info("Hybrid search engine initialized")

        # Initialize embedding client
        try:
            _embedding_client = EmbeddingClient()
            await _embedding_client.initialize()
            logger.info(f"Embedding client initialized (model: {_embedding_client._model})")
        except Exception as e:
            logger.warning(f"Embedding client initialization failed: {e}")

        # Initialize vision client (optional)
        try:
            _vision_client = VisionModelClient()
            await _vision_client.initialize()

            # Check health
            health = await _vision_client.health_check()
            if health.get("status") == "healthy":
                logger.info(f"Vision model API healthy: {health.get('available_models', [])}")
                _doc_processor = DocumentProcessor(vision_client=_vision_client)
                _extraction_pipeline = ExtractionPipeline(_vision_client)

                # Initialize domain detector with LLM client
                _domain_detector = DomainDetector(
                    llm_client=_vision_client,
                    config=DomainConfig(
                        enable_keyword_screening=True,
                        enable_llm_analysis=True,
                        min_confidence=0.3,
                    ),
                )
                logger.info("Domain detector initialized")

                # Initialize community summarizer
                _community_summarizer = CommunitySummarizer(llm_client=_vision_client)
                logger.info("Community summarizer initialized")
            else:
                logger.warning(f"Vision model API not available: {health}")
        except Exception as e:
            logger.warning(f"Vision model initialization failed: {e}")

        # Initialize cross-encoder for reranking
        try:
            _cross_encoder = CrossEncoderReranker()
            await _cross_encoder.initialize()
            logger.info(f"Cross-encoder initialized: {_cross_encoder.model_name}")

            # Initialize reranking pipeline
            _reranking_pipeline = RerankingPipeline(
                hybrid_search=_search_engine,
                cross_encoder=_cross_encoder,
            )
            logger.info("Reranking pipeline initialized")
        except Exception as e:
            logger.warning(f"Cross-encoder/reranking initialization failed: {e}")

            logger.info("Portable Knowledge Base started successfully")

            # Initialize route dependencies
            set_dependencies(
                RouteDeps(
                    sqlite_store=_sqlite_store,
                    chroma_store=_chroma_store,
                    kuzu_store=_kuzu_store,
                    hybrid_search=_search_engine,
                    embedding_client=_embedding_client,
                    llm_client=_vision_client,
                    doc_processor=_doc_processor,
                    vision_client=_vision_client,
                    domain_detector=_domain_detector,
                    reranking_pipeline=_reranking_pipeline,
                    extraction_pipeline=_extraction_pipeline,
                    community_summarizer=_community_summarizer,
                )
            )
            logger.info("Route dependencies initialized")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Portable Knowledge Base...")

    if _cross_encoder:
        await _cross_encoder.shutdown()
    if _embedding_client:
        await _embedding_client.close()
    if _vision_client:
        await _vision_client.close()
    if _kuzu_store:
        await _kuzu_store.close()
    if _chroma_store:
        await _chroma_store.close()
    if _sqlite_store:
        await _sqlite_store.close()

    logger.info("Portable Knowledge Base shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Portable Knowledge Base API",
    description="Self-contained knowledge base with vector search, graph database, RAG, domain detection, and reranking",
    version="0.3.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# Load configuration
_config = load_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.cors.allow_origins,
    allow_credentials=_config.cors.allow_credentials,
    allow_methods=_config.cors.allow_methods,
    allow_headers=_config.cors.allow_headers,
)

# Register route modules
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(ingestion_router)
app.include_router(domain_router)
app.include_router(graph_router)

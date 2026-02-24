"""
Main entry point for Portable Knowledge Base System.

This module provides:
- FastAPI application for REST API
- CLI commands for document ingestion and querying
- Lifespan management for storage components
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from knowledge_base.storage.portable import (
    SQLiteStore,
    ChromaStore,
    KuzuGraphStore,
    HybridSearchEngine,
    PortableStorageConfig,
)
from knowledge_base.storage.portable.sqlite_store import Document, Chunk
from knowledge_base.ingestion import (
    VisionModelClient,
    DocumentProcessor,
    SemanticChunker,
)
from knowledge_base.extraction import ExtractionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _storage_config, _sqlite_store, _chroma_store, _kuzu_store
    global _search_engine, _vision_client, _doc_processor, _extraction_pipeline

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
            else:
                logger.warning(f"Vision model API not available: {health}")
        except Exception as e:
            logger.warning(f"Vision model initialization failed: {e}")

        logger.info("Portable Knowledge Base started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Portable Knowledge Base...")

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
    description="Self-contained knowledge base with vector search, graph database, and RAG",
    version="0.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
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


# ==================== Health Endpoints ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: dict[str, bool]


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        components={
            "sqlite": _sqlite_store is not None,
            "chromadb": _chroma_store is not None,
            "kuzu": _kuzu_store is not None,
            "vision_api": _vision_client is not None,
        }
    )


@app.get("/stats", tags=["health"])
async def get_stats():
    """Get storage statistics."""
    stats = {}

    if _sqlite_store:
        stats["sqlite"] = await _sqlite_store.get_stats()
    if _chroma_store:
        stats["chromadb"] = await _chroma_store.get_stats()
    if _kuzu_store:
        stats["kuzu"] = await _kuzu_store.get_stats()

    return stats


# ==================== Document Endpoints ====================

class DocumentCreate(BaseModel):
    """Document creation request."""
    name: str
    source_uri: Optional[str] = None
    content: Optional[str] = None
    domain: Optional[str] = None
    metadata: dict = {}


class DocumentResponse(BaseModel):
    """Document response."""
    id: str
    name: str
    status: str
    created_at: str


@app.post("/documents", response_model=DocumentResponse, tags=["documents"])
async def create_document(doc: DocumentCreate):
    """Create a new document."""
    if not _sqlite_store:
        raise RuntimeError("Storage not initialized")

    document = Document(
        name=doc.name,
        source_uri=doc.source_uri,
        content=doc.content,
        domain=doc.domain,
        metadata=doc.metadata,
    )

    doc_id = await _sqlite_store.add_document(document)

    return DocumentResponse(
        id=doc_id,
        name=doc.name,
        status="pending",
        created_at=document.created_at.isoformat(),
    )


@app.get("/documents", tags=["documents"])
async def list_documents(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List documents."""
    if not _sqlite_store:
        raise RuntimeError("Storage not initialized")

    docs = await _sqlite_store.list_documents(
        status=status,
        domain=domain,
        limit=limit,
        offset=offset,
    )

    return {
        "documents": [
            {
                "id": d.id,
                "name": d.name,
                "status": d.status,
                "domain": d.domain,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "count": len(docs),
    }


@app.get("/documents/{document_id}", tags=["documents"])
async def get_document(document_id: str):
    """Get a document by ID."""
    if not _sqlite_store:
        raise RuntimeError("Storage not initialized")

    doc = await _sqlite_store.get_document(document_id)
    if not doc:
        return {"error": "Document not found"}

    return {
        "id": doc.id,
        "name": doc.name,
        "source_uri": doc.source_uri,
        "content": doc.content,
        "status": doc.status,
        "domain": doc.domain,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat(),
    }


# ==================== Search Endpoints ====================

class SearchRequest(BaseModel):
    """Search request."""
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    """Search result."""
    chunk_id: str
    document_id: str
    text: str
    score: float
    document_name: Optional[str] = None


@app.post("/search", tags=["search"])
async def search(request: SearchRequest):
    """Search for documents."""
    if not _search_engine:
        raise RuntimeError("Search engine not initialized")

    results = await _search_engine.search(
        query=request.query,
        limit=request.limit,
    )

    return {
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.score,
                "document_name": r.document_name,
            }
            for r in results
        ],
        "query": request.query,
        "count": len(results),
    }


# ==================== Ingestion Endpoints ====================

class IngestRequest(BaseModel):
    """Ingestion request."""
    file_path: str
    domain: Optional[str] = None


class IngestResponse(BaseModel):
    """Ingestion response."""
    document_id: str
    name: str
    status: str
    chunk_count: int


@app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_document(request: IngestRequest):
    """Ingest a document from file path."""
    if not _doc_processor or not _sqlite_store:
        raise RuntimeError("Document processor not initialized")

    path = Path(request.file_path)
    if not path.exists():
        return {"error": f"File not found: {request.file_path}"}

    # Process document
    processed = await _doc_processor.process(path)

    # Create document in store
    document = Document(
        name=processed.name,
        source_uri=processed.source_path,
        content=processed.content,
        domain=request.domain,
        metadata=processed.metadata,
        status="processed",
    )

    doc_id = await _sqlite_store.add_document(document)

    # Chunk document
    chunker = SemanticChunker()
    chunks = chunker.chunk(processed.content, doc_id)

    # Store chunks
    chunk_ids = await _sqlite_store.add_chunks_batch(chunks)

    return IngestResponse(
        document_id=doc_id,
        name=processed.name,
        status="processed",
        chunk_count=len(chunk_ids),
    )


# ==================== Graph Endpoints ====================

@app.get("/graph/entities", tags=["graph"])
async def list_entities(
    name: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """List entities from knowledge graph."""
    if not _kuzu_store:
        raise RuntimeError("Graph store not initialized")

    entities = await _kuzu_store.search_entities(
        name=name,
        entity_type=entity_type,
        limit=limit,
    )

    return {
        "entities": [
            {
                "id": e.id,
                "name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "domain": e.domain,
            }
            for e in entities
        ],
        "count": len(entities),
    }


@app.get("/graph/entities/{entity_id}", tags=["graph"])
async def get_entity(entity_id: str):
    """Get an entity and its relationships."""
    if not _kuzu_store:
        raise RuntimeError("Graph store not initialized")

    entity = await _kuzu_store.get_entity(entity_id)
    if not entity:
        return {"error": "Entity not found"}

    relationships = await _kuzu_store.get_entity_relationships(entity_id)

    return {
        "entity": {
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "description": entity.description,
            "domain": entity.domain,
        },
        "relationships": relationships,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="localhost",
        port=8765,
        reload=True,
        log_config=None,
    )

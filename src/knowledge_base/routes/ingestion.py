"""Ingestion endpoints for KBV2."""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from knowledge_base.routes.dependencies import get_dependencies
from knowledge_base.storage.portable.sqlite_store import Document, Chunk
from knowledge_base.storage.portable.kuzu_store import Entity, Edge
from knowledge_base.ingestion.chunker import SemanticChunker


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["ingestion"])


# Request/Response models
class IngestRequest(BaseModel):
    """Ingest request."""

    file_path: str
    domain: Optional[str] = None
    auto_detect_domain: bool = True


class IngestResponse(BaseModel):
    """Ingest response."""

    document_id: str
    document_name: str
    status: str
    chunk_count: int
    embedding_count: int = 0
    entity_count: int = 0
    detected_domain: Optional[str] = None
    domain_confidence: Optional[float] = None


@router.post("/ingest", response_model=dict)
async def ingest_document(request: IngestRequest):
    """Ingest a document from file path.

    This endpoint:
    1. Processes the document (text extraction)
    2. Auto-detects domain (if enabled) or uses provided domain
    3. Chunks the content
    4. Stores chunks in SQLite
    5. Generates embeddings and stores in ChromaDB
    6. Extracts entities and stores in Kuzu (if extraction pipeline available)
    """
    deps = get_dependencies()
    if not deps.doc_processor or not deps.sqlite:
        raise RuntimeError("Document processor not initialized")

    path = Path(request.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    # Process document
    processed = await deps.doc_processor.process(path)

    # Auto-detect domain if enabled and no domain provided
    detected_domain = request.domain
    domain_confidence = None

    if (
        request.auto_detect_domain
        and not request.domain
        and deps.domain_detector
        and processed.content
    ):
        try:
            detection_result = await deps.domain_detector.detect_domain(processed.content, top_k=1)
            detected_domain = detection_result.primary_domain
            domain_confidence = (
                detection_result.all_domains[0].confidence if detection_result.all_domains else None
            )
            logger.info(
                f"Auto-detected domain: {detected_domain} (confidence: {domain_confidence})"
            )
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}")
            detected_domain = detected_domain or "GENERAL"

    detected_domain = detected_domain or "GENERAL"

    # Create document in store with detected domain
    document = Document(
        name=processed.name,
        source_uri=processed.source_path,
        content=processed.content,
        domain=detected_domain,
        metadata={
            **processed.metadata,
            "domain_confidence": domain_confidence,
            "auto_detected": request.auto_detect_domain and not request.domain,
        },
        status="processed",
    )

    doc_id = await deps.sqlite.add_document(document)

    # Chunk document
    chunker = SemanticChunker()
    chunks = chunker.chunk(processed.content, doc_id)

    # Store chunks in SQLite
    chunk_ids = await deps.sqlite.add_chunks_batch(chunks)

    # Generate and store embeddings in ChromaDB
    embedding_count = 0
    if deps.embedding and deps.chroma and chunks:
        try:
            # Get text from each chunk
            texts = [c.text for c in chunks]

            # Generate embeddings in batch
            embeddings = await deps.embedding.embed_batch(texts)

            # Prepare metadata for each chunk
            metadatas = [
                {
                    "document_id": doc_id,
                    "chunk_index": c.chunk_index,
                    "domain": detected_domain,
                }
                for c in chunks
            ]

            # Store in ChromaDB
            await deps.chroma.add_embeddings(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts,
            )
            embedding_count = len(chunk_ids)
            logger.info(f"Stored {embedding_count} embeddings for document {doc_id}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")

    # Extract entities (if extraction pipeline available)
    entity_count = 0
    if deps.extraction and deps.kuzu:
        try:
            analysis = await deps.extraction.extract_from_document(
                content=processed.content,
                document_id=doc_id,
                document_name=processed.name,
                document_type=detected_domain,
            )

            # Store entities in Kuzu (batch insert for performance)
            if analysis.entities:
                try:
                    entities = [
                        Entity(
                            id=extracted_entity.id,
                            name=extracted_entity.name,
                            entity_type=extracted_entity.entity_type,
                            description=extracted_entity.description,
                            confidence=extracted_entity.confidence,
                            domain=detected_domain,
                            source_text=extracted_entity.source_text,
                        )
                        for extracted_entity in analysis.entities
                    ]
                    await deps.kuzu.add_entities_batch(entities)
                    entity_count = len(entities)
                except Exception as e:
                    logger.warning(f"Failed to store entities: {e}")
                    entity_count = 0

            # Store relationships (batch insert for performance)
            if analysis.relationships:
                try:
                    edges = [
                        Edge(
                            source_id=rel.source_entity,
                            target_id=rel.target_entity,
                            relation_type=rel.relation_type,
                            properties={
                                "description": rel.description,
                                "confidence": rel.confidence,
                            }
                            if rel.description
                            else {},
                        )
                        for rel in analysis.relationships
                    ]
                    await deps.kuzu.add_edge_batch(edges)
                except Exception as e:
                    logger.debug(f"Failed to store relationships: {e}")

            logger.info(f"Extracted {entity_count} entities for document {doc_id}")
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

    return {
        "document_id": doc_id,
        "document_name": processed.name,
        "status": "processed",
        "chunk_count": len(chunk_ids),
        "embedding_count": embedding_count,
        "entity_count": entity_count,
        "detected_domain": detected_domain,
        "domain_confidence": domain_confidence,
    }

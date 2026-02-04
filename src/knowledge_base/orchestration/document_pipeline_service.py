"""Document pipeline service for processing documents."""

import logging
from typing import Optional, Tuple
from pathlib import Path
import mimetypes
from uuid import uuid4

from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.partitioning.semantic_chunker import SemanticChunker
from knowledge_base.ingestion.v1.embedding_client import (
    EmbeddingClient,
    EmbeddingConfig,
)
from knowledge_base.config.constants import (
    SEMANTIC_CHUNK_SIZE,
    OVERLAP_RATIO,
    EMBEDDING_URL,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from knowledge_base.persistence.v1.schema import Document, Chunk, Entity
from sqlalchemy import select


class DocumentPipelineService(BaseService):
    """Service for document processing including partitioning and embedding."""

    def __init__(
        self,
        chunker: Optional[SemanticChunker] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        super().__init__()
        self._chunker = chunker
        self._embedding_client = embedding_client

    async def initialize(self) -> None:
        """Initialize the service."""
        if self._chunker is None:
            self._chunker = SemanticChunker(
                chunk_size=SEMANTIC_CHUNK_SIZE,
                overlap_ratio=OVERLAP_RATIO,
            )

        if self._embedding_client is None:
            config = EmbeddingConfig(
                embedding_url=EMBEDDING_URL,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                dimensions=EMBEDDING_DIMENSION,
            )
            self._embedding_client = EmbeddingClient(config=config)

        self._logger.info("DocumentPipelineService initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._logger.info("DocumentPipelineService shutdown")

    async def partition(
        self,
        file_path: str,
        document_name: Optional[str] = None,
        domain: str = "GENERAL",
    ) -> Document:
        """Partition a document into chunks.

        Args:
            file_path: Path to the document file
            document_name: Optional name for the document
            domain: Domain of the document

        Returns:
            Document with chunks
        """
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Create document
        doc_name = document_name or Path(file_path).stem
        document = Document(
            name=doc_name,
            file_path=file_path,
            domain=domain,
            content=content,
        )

        # Partition into chunks
        chunks = self._chunker.chunk(content)

        # Create chunk objects
        document.chunks = [
            Chunk(
                document_id=document.id,
                text=chunk_text,
                chunk_index=i,
                metadata={"chunk_type": "semantic"},
            )
            for i, chunk_text in enumerate(chunks)
        ]

        self._logger.info(
            f"Partitioned document '{doc_name}' into {len(document.chunks)} chunks"
        )
        return document

    async def embed(self, chunks: list[Chunk]) -> None:
        """Generate embeddings for document chunks.

        Args:
            chunks: List of chunks to embed (modified in-place)
        """
        if not chunks:
            self._logger.warning("No chunks to embed")
            return

        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await self._embedding_client.embed_batch(chunk_texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        self._logger.info(f"Generated embeddings for {len(chunks)} chunks")

    async def embed_entities(self, entities: list[Entity]) -> None:
        """Generate embeddings for entities.

        Args:
            entities: List of entities to embed (modified in-place)
        """
        if not entities:
            return

        entity_texts = []
        for entity in entities:
            entity_name = (
                entity.name if isinstance(entity.name, str) else str(entity.name)
            )
            entity_desc = (
                entity.description
                if isinstance(entity.description, str)
                else str(entity.description or "")
            )
            entity_texts.append(f"{entity_name}. {entity_desc}")

        if entity_texts:
            embeddings = await self._embedding_client.embed_batch(entity_texts)
            for entity, embedding in zip(entities, embeddings):
                entity.embedding = embedding

        self._logger.info(f"Generated embeddings for {len(entities)} entities")

    def _get_mime_type(self, path: Path) -> str:
        """Get the MIME type of a file.

        Args:
            path: Path to the file.

        Returns:
            MIME type string.
        """
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"

    async def create_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        vector_store=None,
    ) -> Document:
        """Create a document record in the database.

        Args:
            file_path: Path to the document file.
            document_name: Optional document name override.
            vector_store: Vector store for database operations.

        Returns:
            Created document.
        """
        path = Path(file_path) if isinstance(file_path, str) else file_path
        name = document_name or path.name
        mime_type = self._get_mime_type(path)

        document = Document(
            id=uuid4(),
            name=name,
            source_uri=str(path),
            mime_type=mime_type,
            status="pending",
            doc_metadata={"source": str(path)},
        )

        if vector_store:
            async with vector_store.get_session() as session:
                session.add(document)
                await session.commit()
                await session.refresh(document)

        self._logger.info(f"Created document record: {document.name}")
        return document

    async def process(
        self,
        file_path: str | Path,
        document_name: str | None = None,
        domain: str = "GENERAL",
        vector_store=None,
    ) -> Document:
        """Process a document - partition, save to DB, and embed.

        Args:
            file_path: Path to the document file.
            document_name: Optional document name override.
            domain: Domain of the document.
            vector_store: Vector store for database operations.

        Returns:
            Processed document with chunks and embeddings.
        """
        if not vector_store:
            raise ValueError("vector_store is required for processing documents")

        path = Path(file_path) if isinstance(file_path, str) else file_path
        name = document_name or path.name

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        mime_type = self._get_mime_type(path)

        document = Document(
            id=uuid4(),
            name=name,
            source_uri=str(path),
            mime_type=mime_type,
            status="pending",
            doc_metadata={"source": str(path)},
            domain=domain,
        )

        chunks = self._chunker.chunk(content)
        # Convert Pydantic chunks to SQLAlchemy chunks
        # Import directly to avoid Pydantic Chunk shadowing
        from knowledge_base.persistence.v1.schema import Chunk as SchemaChunk

        chunk_objects = []
        for i, pydantic_chunk in enumerate(chunks):
            sql_chunk = SchemaChunk(
                id=uuid4(),
                document_id=document.id,
                text=pydantic_chunk.text,
                chunk_index=pydantic_chunk.chunk_index,
                page_number=pydantic_chunk.page_number,
                token_count=pydantic_chunk.token_count,
                chunk_metadata={"chunk_type": "semantic", **pydantic_chunk.metadata},
            )
            chunk_objects.append(sql_chunk)

        async with vector_store.get_session() as session:
            # Add document and chunks in a single transaction
            session.add(document)
            for chunk in chunk_objects:
                session.add(chunk)
            await session.commit()
            await session.refresh(document)

        await self.embed(chunk_objects)

        async with vector_store.get_session() as session:
            doc_to_update = await session.get(Document, document.id)
            if doc_to_update:
                doc_to_update.status = "embedded"
                await session.commit()

        self._logger.info(f"Processed document '{name}': {len(chunk_objects)} chunks")
        return document

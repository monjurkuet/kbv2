"""Document pipeline service for processing documents."""

import logging
from typing import Optional
from pathlib import Path

from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.partitioning.semantic_chunker import SemanticChunker
from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient
from knowledge_base.config.constants import (
    SEMANTIC_CHUNK_SIZE,
    OVERLAP_RATIO,
    EMBEDDING_URL,
    DEFAULT_EMBEDDING_MODEL,
)
from knowledge_base.persistence.v1.schema import Document, Chunk


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
            self._embedding_client = EmbeddingClient(
                model=DEFAULT_EMBEDDING_MODEL,
                api_url=EMBEDDING_URL,
            )

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
        chunks = self._chunker.partition(content)

        # Create chunk objects
        document.chunks = [
            Chunk(
                document_id=document.id,
                text=chunk_text.text,
                chunk_index=chunk_text.chunk_index,
                page_number=chunk_text.page_number,
                token_count=chunk_text.token_count,
                chunk_metadata={"chunk_type": "semantic"},
            )
            for chunk_text in chunks
        ]

        self._logger.info(
            f"Partitioned document '{doc_name}' into {len(document.chunks)} chunks"
        )
        return document

    async def embed(self, document: Document) -> None:
        """Generate embeddings for document chunks.

        Args:
            document: Document to embed (modified in-place)
        """
        if not document.chunks:
            self._logger.warning(f"No chunks to embed for document '{document.name}'")
            return

        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in document.chunks]
        embeddings = await self._embedding_client.embed_batch(chunk_texts)

        # Assign embeddings to chunks
        for chunk, embedding in zip(document.chunks, embeddings):
            chunk.embedding = embedding

        self._logger.info(f"Generated embeddings for {len(document.chunks)} chunks")

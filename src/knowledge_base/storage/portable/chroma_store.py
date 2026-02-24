"""ChromaDB vector store for embedding storage and similarity search.

ChromaDB provides:
- Persistent storage (single directory)
- HNSW indexing for fast similarity search
- Multiple embedding functions support
- Metadata filtering

This module wraps ChromaDB for use in the portable storage layer.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from knowledge_base.storage.portable.config import ChromaConfig

logger = logging.getLogger(__name__)

# Lazy import for ChromaDB
_chromadb = None


def _get_chromadb():
    """Lazy import of chromadb."""
    global _chromadb
    if _chromadb is None:
        import chromadb
        _chromadb = chromadb
    return _chromadb


class EmbeddingResult(BaseModel):
    """Result from embedding search."""

    chunk_id: str
    document_id: Optional[str] = None
    text: Optional[str] = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChromaStore:
    """ChromaDB-based vector store for embedding storage and search.

    This class provides a persistent vector store using ChromaDB with:
    - HNSW indexing for fast approximate nearest neighbor search
    - Support for multiple embedding functions
    - Metadata filtering capabilities
    - Single-directory persistence for portability

    Example:
        >>> store = ChromaStore()
        >>> await store.initialize()
        >>> await store.add_embeddings(chunk_ids, embeddings, metadatas)
        >>> results = await store.search(query_embedding, limit=10)
    """

    def __init__(self, config: Optional[ChromaConfig] = None) -> None:
        """Initialize ChromaDB store.

        Args:
            config: ChromaDB configuration. Uses defaults if not provided.
        """
        self._config = config or ChromaConfig()
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        # Ensure directory exists
        self._config.persist_directory.mkdir(parents=True, exist_ok=True)

        def _init():
            chromadb = _get_chromadb()

            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=self._config.persist_directory_str
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self._config.collection_name,
                metadata={
                    "hnsw:space": self._config.distance_metric,
                    "description": "Knowledge base chunk embeddings",
                }
            )

        await asyncio.get_event_loop().run_in_executor(None, _init)

        self._initialized = True
        logger.info(f"ChromaDB initialized at {self._config.persist_directory}")

    async def close(self) -> None:
        """Close ChromaDB client."""
        self._client = None
        self._collection = None
        self._initialized = False

    # ==================== Collection Management ====================

    async def create_collection(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a new collection.

        Args:
            name: Collection name.
            metadata: Collection metadata.
        """
        def _create():
            self._client.create_collection(
                name=name,
                metadata=metadata or {"hnsw:space": self._config.distance_metric}
            )

        await asyncio.get_event_loop().run_in_executor(None, _create)

    async def list_collections(self) -> list[str]:
        """List all collections.

        Returns:
            List of collection names.
        """
        def _list():
            return [c.name for c in self._client.list_collections()]

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Args:
            name: Collection name.
        """
        def _delete():
            self._client.delete_collection(name)

        await asyncio.get_event_loop().run_in_executor(None, _delete)

    # ==================== Embedding Operations ====================

    async def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float] | np.ndarray],
        metadatas: Optional[list[dict[str, Any]]] = None,
        documents: Optional[list[str]] = None,
    ) -> int:
        """Add embeddings to the collection.

        Args:
            ids: Unique IDs for each embedding.
            embeddings: List of embedding vectors.
            metadatas: Optional metadata for each embedding.
            documents: Optional document text for each embedding.

        Returns:
            Number of embeddings added.
        """
        if not ids:
            return 0

        # Convert numpy arrays to lists
        embeddings = [
            e.tolist() if isinstance(e, np.ndarray) else e
            for e in embeddings
        ]

        def _add():
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            return len(ids)

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def add_single_embedding(
        self,
        id: str,
        embedding: list[float] | np.ndarray,
        metadata: Optional[dict[str, Any]] = None,
        document: Optional[str] = None,
    ) -> None:
        """Add a single embedding.

        Args:
            id: Unique ID for the embedding.
            embedding: Embedding vector.
            metadata: Optional metadata.
            document: Optional document text.
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        def _add():
            self._collection.add(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata] if metadata else None,
                documents=[document] if document else None,
            )

        await asyncio.get_event_loop().run_in_executor(None, _add)

    async def update_embedding(
        self,
        id: str,
        embedding: Optional[list[float] | np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
        document: Optional[str] = None,
    ) -> None:
        """Update an existing embedding.

        Args:
            id: Embedding ID.
            embedding: New embedding vector (optional).
            metadata: New metadata (optional).
            document: New document text (optional).
        """
        update_data = {"ids": [id]}

        if embedding is not None:
            update_data["embeddings"] = [
                embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            ]
        if metadata is not None:
            update_data["metadatas"] = [metadata]
        if document is not None:
            update_data["documents"] = [document]

        def _update():
            self._collection.update(**update_data)

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def delete_embeddings(self, ids: list[str]) -> None:
        """Delete embeddings by IDs.

        Args:
            ids: List of embedding IDs to delete.
        """
        if not ids:
            return

        def _delete():
            self._collection.delete(ids=ids)

        await asyncio.get_event_loop().run_in_executor(None, _delete)

    async def delete_by_metadata(self, where: dict[str, Any]) -> int:
        """Delete embeddings matching metadata filter.

        Args:
            where: Metadata filter condition.

        Returns:
            Number of embeddings deleted.
        """
        def _delete():
            # Get count first
            result = self._collection.get(where=where)
            count = len(result["ids"]) if result["ids"] else 0

            if count > 0:
                self._collection.delete(where=where)

            return count

        return await asyncio.get_event_loop().run_in_executor(None, _delete)

    # ==================== Search Operations ====================

    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        limit: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
    ) -> list[EmbeddingResult]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            where: Metadata filter.
            where_document: Document content filter.

        Returns:
            List of search results.
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        def _search():
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
                where_document=where_document,
                include=["metadatas", "documents", "distances"]
            )

            # Convert to EmbeddingResult objects
            embedding_results = []
            if results["ids"] and results["ids"][0]:
                for i, id_ in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0
                    # Convert distance to similarity score
                    # For cosine: distance = 1 - similarity, so similarity = 1 - distance
                    similarity = 1 - distance if self._config.distance_metric == "cosine" else 1 / (1 + distance)

                    embedding_results.append(EmbeddingResult(
                        chunk_id=id_,
                        document_id=results["metadatas"][0][i].get("document_id") if results["metadatas"] else None,
                        text=results["documents"][0][i] if results["documents"] else None,
                        score=similarity,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    ))

            return embedding_results

        return await asyncio.get_event_loop().run_in_executor(None, _search)

    async def search_by_text(
        self,
        query_text: str,
        embedding_function: Optional[Any] = None,
        limit: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> list[EmbeddingResult]:
        """Search by text (requires embedding function).

        Args:
            query_text: Query text.
            embedding_function: Function to embed text.
            limit: Maximum results.
            where: Metadata filter.

        Returns:
            List of search results.
        """
        if embedding_function is None:
            raise ValueError("Embedding function required for text search")

        # Embed query text
        if hasattr(embedding_function, "embed_query"):
            query_embedding = await embedding_function.embed_query(query_text)
        elif hasattr(embedding_function, "__call__"):
            query_embedding = embedding_function([query_text])[0]
        else:
            raise ValueError("Invalid embedding function")

        return await self.search(query_embedding, limit, where)

    async def get_embedding(self, id: str) -> Optional[dict[str, Any]]:
        """Get a single embedding by ID.

        Args:
            id: Embedding ID.

        Returns:
            Dictionary with id, embedding, metadata, document if found.
        """
        def _get():
            result = self._collection.get(
                ids=[id],
                include=["embeddings", "metadatas", "documents"]
            )

            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "embedding": result["embeddings"][0] if result["embeddings"] else None,
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                    "document": result["documents"][0] if result["documents"] else None,
                }
            return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def get_embeddings(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Get multiple embeddings.

        Args:
            ids: List of embedding IDs.
            where: Metadata filter.
            limit: Maximum results.

        Returns:
            List of embedding dictionaries.
        """
        def _get():
            result = self._collection.get(
                ids=ids,
                where=where,
                limit=limit,
                include=["embeddings", "metadatas", "documents"]
            )

            results = []
            for i, id_ in enumerate(result["ids"]):
                results.append({
                    "id": id_,
                    "embedding": result["embeddings"][i] if result["embeddings"] else None,
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                    "document": result["documents"][i] if result["documents"] else None,
                })

            return results

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Statistics ====================

    async def count(self, where: Optional[dict[str, Any]] = None) -> int:
        """Count embeddings in collection.

        Args:
            where: Optional metadata filter (note: not supported in all ChromaDB versions).

        Returns:
            Number of embeddings.
        """
        def _count():
            # ChromaDB's count() doesn't accept 'where' in newer versions
            return self._collection.count()

        return await asyncio.get_event_loop().run_in_executor(None, _count)

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with statistics.
        """
        count = await self.count()

        # Get directory size
        dir_size = 0
        if self._config.persist_directory.exists():
            for f in self._config.persist_directory.rglob("*"):
                if f.is_file():
                    dir_size += f.stat().st_size

        return {
            "collection_name": self._config.collection_name,
            "embedding_count": count,
            "storage_size_mb": dir_size / (1024 * 1024),
            "distance_metric": self._config.distance_metric,
        }

    # ==================== Context Managers ====================

    async def __aenter__(self) -> "ChromaStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

"""BM25-based keyword search index implementation.

This module provides a BM25 (Okapi BM25) full-text search index for keyword-based
document retrieval. BM25 is a probabilistic ranking function used for information
retrieval that ranks documents based on the query terms appearing in each document.

The implementation uses the rank-bm25 library and stores indexed documents
in a SQLite database for persistence across application restarts.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import sqlite3
from uuid import uuid4

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


class IndexedDocument(BaseModel):
    """Document model for BM25 indexing.

    Attributes:
        id: Unique identifier for the document.
        text: Full text content to be indexed.
        metadata: Optional metadata dictionary for filtering.
        document_id: Reference to the source document.
    """

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique document ID"
    )
    text: str = Field(..., description="Text content to index")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )
    document_id: str = Field(..., description="Source document reference")


class SearchResult(BaseModel):
    """Search result from BM25 index.

    Attributes:
        id: Document ID.
        text: Document text content.
        score: BM25 relevance score.
        metadata: Document metadata.
    """

    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text content")
    score: float = Field(..., description="BM25 relevance score")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )


class BM25Index:
    """BM25-based keyword search index.

    This class implements a full-text search index using the BM25 (Okapi BM25)
    ranking function. It provides efficient keyword-based document retrieval
    with support for metadata filtering and persistent storage.

    The BM25 algorithm considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across all documents
    - Document length normalization: Prevents bias toward longer documents

    Example:
        >>> index = BM25Index()
        >>> await index.initialize()
        >>> documents = [
        ...     IndexedDocument(id="1", text="Python programming guide", document_id="doc1"),
        ...     IndexedDocument(id="2", text="JavaScript tutorials", document_id="doc1"),
        ... ]
        >>> await index.index_documents(documents)
        >>> results = await index.search("Python", top_k=10)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        db_path: str = "bm25_index.db",
    ) -> None:
        """Initialize the BM25 index.

        Args:
            k1: Term frequency saturation parameter. Controls how quickly
                term frequency saturates. Higher values give more weight
                to repeated terms. Typical range: 1.2-2.0.
            b: Length normalization parameter. Controls how much document
                length affects scoring. 1.0 means full normalization,
                0.0 means no normalization. Typical value: 0.75.
            epsilon: Negative IDF smoothing parameter. Prevents division by
                zero for terms not in the corpus. Typical value: 0.25.
            db_path: Path to SQLite database for persistence.
        """
        self._k1 = k1
        self._b = b
        self._epsilon = epsilon
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_texts: Dict[str, str] = {}
        self._doc_metadata: Dict[str, Dict[str, Any]] = {}
        self._document_ids: List[str] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the index and database tables.

        Creates the SQLite database and necessary tables for storing indexed
        documents. This method must be called before performing any search
        or indexing operations.

        Raises:
            RuntimeError: If database initialization fails.
        """
        if self._initialized:
            return

        try:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row

            cursor = self._conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bm25_documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_bm25_document_id
                ON bm25_documents(document_id)
            """)

            self._conn.commit()

            await self._reload_index()

            self._initialized = True

        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize BM25 index: {e}")

    async def _reload_index(self) -> None:
        """Reload the BM25 index from the database.

        Rebuilds the internal BM25 index from all documents stored in
        the database. Called during initialization and after updates.
        """
        if self._conn is None:
            return

        cursor = self._conn.cursor()
        cursor.execute("SELECT id, text, document_id, metadata FROM bm25_documents")

        self._doc_ids = []
        self._doc_texts = {}
        self._doc_metadata = {}
        self._document_ids = []

        for row in cursor.fetchall():
            doc_id = row["id"]
            text = row["text"]
            doc_id_ref = row["document_id"]
            metadata = row["metadata"]

            self._doc_ids.append(doc_id)
            self._doc_texts[doc_id] = text
            self._document_ids.append(doc_id_ref)

            if metadata:
                import json

                self._doc_metadata[doc_id] = json.loads(metadata)
            else:
                self._doc_metadata[doc_id] = {}

        if self._doc_ids:
            tokenized_corpus = [
                self._tokenize(text) for text in self._doc_texts.values()
            ]
            self._bm25 = BM25Okapi(
                tokenized_corpus,
                k1=self._k1,
                b=self._b,
                epsilon=self._epsilon,
            )
        else:
            self._bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into individual terms.

        Performs basic tokenization including lowercasing and removing
        special characters. Override this method for language-specific
        tokenization (e.g., stemming, stop word removal).

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokenized terms.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [token for token in tokens if token.strip()]

    async def index_documents(self, documents: List[IndexedDocument]) -> None:
        """Index documents for BM25 search.

        Adds documents to the index and updates the BM25 data structure.
        Existing documents with the same ID will be replaced.

        Args:
            documents: List of documents to index.

        Raises:
            RuntimeError: If index is not initialized.
            ValueError: If documents list is empty.
        """
        if not self._initialized:
            raise RuntimeError("BM25 index not initialized. Call initialize() first.")

        if not documents:
            return

        import json

        cursor = self._conn.cursor()
        if self._conn is None:
            raise RuntimeError("Database connection not established")

        for doc in documents:
            cursor.execute(
                """
                INSERT OR REPLACE INTO bm25_documents (id, text, document_id, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.text,
                    doc.document_id,
                    json.dumps(doc.metadata) if doc.metadata else None,
                ),
            )

            self._doc_texts[doc.id] = doc.text
            self._doc_metadata[doc.id] = doc.metadata or {}
            if doc.id not in self._document_ids:
                self._document_ids.append(doc.document_id)

        self._conn.commit()

        await self._reload_index()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Execute BM25 search.

        Searches the index for documents matching the query string.
        Results are ranked by BM25 relevance score and optionally filtered
        by metadata.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters. Only documents matching
                all filter criteria will be returned.

        Returns:
            List of SearchResult objects ranked by relevance score.

        Raises:
            RuntimeError: If index is not initialized.
            ValueError: If top_k is negative.
        """
        if not self._initialized:
            raise RuntimeError("BM25 index not initialized. Call initialize() first.")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not self._bm25 or not self._doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        results: List[Tuple[str, float]] = []
        for i, doc_id in enumerate(self._doc_ids):
            score = scores[i]
            if filters:
                metadata = self._doc_metadata.get(doc_id, {})
                if not self._matches_filters(metadata, filters):
                    continue
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            SearchResult(
                id=doc_id,
                text=self._doc_texts[doc_id],
                score=score,
                metadata=self._doc_metadata.get(doc_id),
            )
            for doc_id, score in results
        ]

    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict) -> bool:
        """Check if metadata matches all filter criteria.

        Args:
            metadata: Document metadata to check.
            filters: Filter criteria to match.

        Returns:
            True if all filters match, False otherwise.
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True

    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the index.

        Removes documents by their document_id reference. Note that this
        deletes all chunks belonging to the specified document IDs.

        Args:
            document_ids: List of document IDs to delete.

        Raises:
            RuntimeError: If index is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("BM25 index not initialized. Call initialize() first.")

        if not document_ids:
            return

        cursor = self._conn.cursor()
        if self._conn is None:
            raise RuntimeError("Database connection not established")

        placeholders = ",".join("?" * len(document_ids))
        cursor.execute(
            f"DELETE FROM bm25_documents WHERE document_id IN ({placeholders})",
            document_ids,
        )

        self._conn.commit()

        await self._reload_index()

    async def update_document(self, document: IndexedDocument) -> None:
        """Update a document in the index.

        Updates an existing document or inserts a new one. This is equivalent
        to calling delete followed by index for the specific document.

        Args:
            document: Updated document data.

        Raises:
            RuntimeError: If index is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("BM25 index not initialized. Call initialize() first.")

        await self.index_documents([document])

    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary containing index statistics including document count.
        """
        if self._conn is None:
            return {"document_count": 0, "indexed": False}

        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM bm25_documents")
        count = cursor.fetchone()[0]

        return {
            "document_count": count,
            "indexed": self._bm25 is not None,
            "k1": self._k1,
            "b": self._b,
            "epsilon": self._epsilon,
        }

    async def clear(self) -> None:
        """Clear all documents from the index.

        Removes all indexed documents and resets the BM25 data structure.
        """
        if self._conn is None:
            return

        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM bm25_documents")
        self._conn.commit()

        self._doc_ids = []
        self._doc_texts = {}
        self._doc_metadata = {}
        self._document_ids = []
        self._bm25 = None

    async def close(self) -> None:
        """Close the database connection.

        Closes the SQLite connection and cleans up resources.
        """
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False

    async def __aenter__(self) -> "BM25Index":
        """Enter async context manager.

        Returns:
            Self after initialization.
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()

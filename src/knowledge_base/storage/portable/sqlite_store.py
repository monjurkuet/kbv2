"""SQLite storage with FTS5 full-text search and sqlite-vec vector search.

This module provides a portable, file-based storage solution using SQLite
with extensions for:
- FTS5: Full-text search with BM25 ranking
- sqlite-vec: Vector similarity search using HNSW

All data is stored in a single .db file for maximum portability.
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from knowledge_base.storage.portable.config import SQLiteConfig

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Document model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source_uri: Optional[str] = None
    content: Optional[str] = None
    mime_type: Optional[str] = None
    status: str = "pending"
    domain: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_db_dict(self) -> dict:
        """Convert to database dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "source_uri": self.source_uri,
            "content": self.content,
            "mime_type": self.mime_type,
            "status": self.status,
            "domain": self.domain,
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Chunk(BaseModel):
    """Document chunk model."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    chunk_index: int
    token_count: Optional[int] = None
    page_number: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_db_dict(self) -> dict:
        """Convert to database dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "page_number": self.page_number,
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


class EntityMention(BaseModel):
    """Entity mention in a chunk."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    grounding_quote: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Search result model."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    document_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SQLiteStore:
    """SQLite-based storage with FTS5 and vector search.

    This class provides a fully portable storage solution that stores all
    data in a single SQLite database file with:
    - Documents table: Source documents with metadata
    - Chunks table: Chunked content for retrieval
    - FTS5 virtual table: Full-text search with BM25
    - Vector virtual table: Similarity search with sqlite-vec

    Example:
        >>> store = SQLiteStore()
        >>> await store.initialize()
        >>> await store.add_document(doc)
        >>> results = await store.search("bitcoin price target", limit=10)
    """

    def __init__(self, config: Optional[SQLiteConfig] = None) -> None:
        """Initialize SQLite store.

        Args:
            config: SQLite configuration. Uses defaults if not provided.
        """
        self._config = config or SQLiteConfig()
        self._pool: list[sqlite3.Connection] = []
        self._initialized = False
        self._vector_enabled = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection from the pool.

        Returns:
            SQLite connection with extensions loaded.
        """
        if self._pool:
            return self._pool.pop()

        conn = sqlite3.connect(self._config.db_path_str)
        conn.row_factory = sqlite3.Row

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        # Try to load sqlite-vec extension
        try:
            conn.enable_load_extension(True)
            # Try different extension names
            for ext_name in ["vec0", "sqlite_vec", "vec"]:
                try:
                    conn.load_extension(ext_name)
                    self._vector_enabled = True
                    logger.info(f"Loaded sqlite-vec extension: {ext_name}")
                    break
                except sqlite3.OperationalError:
                    continue
            conn.enable_load_extension(False)
        except Exception as e:
            logger.warning(f"Could not load sqlite-vec extension: {e}")
            logger.info("Vector search will be disabled. Install sqlite-vec for vector support.")

        return conn

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection to return.
        """
        if len(self._pool) < self._config.pool_size:
            self._pool.append(conn)
        else:
            conn.close()

    @contextmanager
    def _get_db(self):
        """Context manager for database connections.

        Yields:
            SQLite connection.
        """
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)

    async def initialize(self) -> None:
        """Initialize the database schema.

        Creates all required tables and indexes.
        """
        # Ensure directory exists
        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)

        def _init_schema():
            with self._get_db() as conn:
                # Documents table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        source_uri TEXT,
                        content TEXT,
                        mime_type TEXT,
                        status TEXT DEFAULT 'pending',
                        domain TEXT,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Chunks table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        token_count INTEGER,
                        page_number INTEGER,
                        metadata JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                    )
                """)

                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)")

                # FTS5 virtual table for full-text search
                if self._config.enable_fts:
                    conn.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                            chunk_id,
                            document_id,
                            text,
                            tokenize='porter unicode61',
                            content='chunks',
                            content_rowid='rowid'
                        )
                    """)

                    # Triggers to keep FTS index in sync
                    conn.execute("""
                        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                            INSERT INTO chunks_fts(rowid, chunk_id, document_id, text)
                            VALUES (new.rowid, new.id, new.document_id, new.text);
                        END
                    """)

                    conn.execute("""
                        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, document_id, text)
                            VALUES('delete', old.rowid, old.id, old.document_id, old.text);
                        END
                    """)

                    conn.execute("""
                        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                            INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, document_id, text)
                            VALUES('delete', old.rowid, old.id, old.document_id, old.text);
                            INSERT INTO chunks_fts(rowid, chunk_id, document_id, text)
                            VALUES (new.rowid, new.id, new.document_id, new.text);
                        END
                    """)

                # sqlite-vec virtual table for vector search
                if self._config.enable_vector and self._vector_enabled:
                    dim = self._config.embedding_dimension
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                            chunk_id TEXT PRIMARY KEY,
                            embedding FLOAT[{dim}]
                        )
                    """)

                # Entity mentions table (for linking chunks to extracted entities)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_mentions (
                        id TEXT PRIMARY KEY,
                        chunk_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        entity_name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        grounding_quote TEXT,
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                    )
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_chunk_id ON entity_mentions(chunk_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity_id ON entity_mentions(entity_id)")

                conn.commit()

        # Run schema creation in thread pool
        await asyncio.get_event_loop().run_in_executor(None, _init_schema)

        self._initialized = True
        logger.info(f"SQLite store initialized at {self._config.db_path}")

    async def close(self) -> None:
        """Close all database connections."""
        # Note: Connections may have been created in different threads via run_in_executor
        # We clear the pool but don't explicitly close connections to avoid thread errors
        self._pool.clear()
        self._initialized = False

    # ==================== Document Operations ====================

    async def add_document(self, document: Document) -> str:
        """Add a document to the store.

        Args:
            document: Document to add.

        Returns:
            Document ID.
        """
        def _add():
            with self._get_db() as conn:
                conn.execute("""
                    INSERT INTO documents (id, name, source_uri, content, mime_type, status, domain, metadata, created_at, updated_at)
                    VALUES (:id, :name, :source_uri, :content, :mime_type, :status, :domain, :metadata, :created_at, :updated_at)
                """, document.to_db_dict())
                conn.commit()
            return document.id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID.

        Args:
            document_id: Document ID.

        Returns:
            Document if found, None otherwise.
        """
        def _get():
            with self._get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                if row:
                    return Document(
                        id=row["id"],
                        name=row["name"],
                        source_uri=row["source_uri"],
                        content=row["content"],
                        mime_type=row["mime_type"],
                        status=row["status"],
                        domain=row["domain"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def list_documents(
        self,
        status: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents with optional filtering.

        Args:
            status: Filter by status.
            domain: Filter by domain.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of documents.
        """
        def _list():
            with self._get_db() as conn:
                query = "SELECT * FROM documents WHERE 1=1"
                params: list[Any] = []

                if status:
                    query += " AND status = ?"
                    params.append(status)
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)

                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                rows = conn.execute(query, params).fetchall()
                return [
                    Document(
                        id=row["id"],
                        name=row["name"],
                        source_uri=row["source_uri"],
                        content=row["content"],
                        mime_type=row["mime_type"],
                        status=row["status"],
                        domain=row["domain"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                    for row in rows
                ]

        return await asyncio.get_event_loop().run_in_executor(None, _list)

    async def update_document_status(self, document_id: str, status: str) -> None:
        """Update document processing status.

        Args:
            document_id: Document ID.
            status: New status.
        """
        def _update():
            with self._get_db() as conn:
                conn.execute("""
                    UPDATE documents
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, document_id))
                conn.commit()

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks.

        Args:
            document_id: Document ID.

        Returns:
            True if deleted, False if not found.
        """
        def _delete():
            with self._get_db() as conn:
                cursor = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                conn.commit()
                return cursor.rowcount > 0

        return await asyncio.get_event_loop().run_in_executor(None, _delete)

    # ==================== Chunk Operations ====================

    async def add_chunk(self, chunk: Chunk) -> str:
        """Add a chunk to the store.

        Args:
            chunk: Chunk to add.

        Returns:
            Chunk ID.
        """
        def _add():
            with self._get_db() as conn:
                conn.execute("""
                    INSERT INTO chunks (id, document_id, text, chunk_index, token_count, page_number, metadata, created_at)
                    VALUES (:id, :document_id, :text, :chunk_index, :token_count, :page_number, :metadata, :created_at)
                """, chunk.to_db_dict())
                conn.commit()
            return chunk.id

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def add_chunks_batch(self, chunks: list[Chunk]) -> list[str]:
        """Add multiple chunks in a batch.

        Args:
            chunks: List of chunks to add.

        Returns:
            List of chunk IDs.
        """
        def _add():
            with self._get_db() as conn:
                for chunk in chunks:
                    conn.execute("""
                        INSERT INTO chunks (id, document_id, text, chunk_index, token_count, page_number, metadata, created_at)
                        VALUES (:id, :document_id, :text, :chunk_index, :token_count, :page_number, :metadata, :created_at)
                    """, chunk.to_db_dict())
                conn.commit()
            return [c.id for c in chunks]

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def get_chunks_by_document(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document.

        Args:
            document_id: Document ID.

        Returns:
            List of chunks.
        """
        def _get():
            with self._get_db() as conn:
                rows = conn.execute("""
                    SELECT * FROM chunks
                    WHERE document_id = ?
                    ORDER BY chunk_index
                """, (document_id,)).fetchall()
                return [
                    Chunk(
                        id=row["id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        chunk_index=row["chunk_index"],
                        token_count=row["token_count"],
                        page_number=row["page_number"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                    for row in rows
                ]

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID.

        Args:
            chunk_id: Chunk ID.

        Returns:
            Chunk if found, None otherwise.
        """
        def _get():
            with self._get_db() as conn:
                row = conn.execute(
                    "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
                ).fetchone()
                if row:
                    return Chunk(
                        id=row["id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        chunk_index=row["chunk_index"],
                        token_count=row["token_count"],
                        page_number=row["page_number"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                return None

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Vector Operations ====================

    async def add_embedding(self, chunk_id: str, embedding: list[float] | np.ndarray) -> bool:
        """Add an embedding for a chunk.

        Args:
            chunk_id: Chunk ID.
            embedding: Embedding vector.

        Returns:
            True if successful, False if vector search not available.
        """
        if not self._vector_enabled:
            logger.warning("Vector search not enabled - skipping embedding storage")
            return False

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        def _add():
            with self._get_db() as conn:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, embedding))
                    conn.commit()
                    return True
                except sqlite3.OperationalError as e:
                    logger.error(f"Failed to add embedding: {e}")
                    return False

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    async def add_embeddings_batch(self, embeddings: list[tuple[str, list[float]]]) -> int:
        """Add multiple embeddings in a batch.

        Args:
            embeddings: List of (chunk_id, embedding) tuples.

        Returns:
            Number of embeddings added.
        """
        if not self._vector_enabled:
            logger.warning("Vector search not enabled - skipping batch embedding storage")
            return 0

        def _add():
            count = 0
            with self._get_db() as conn:
                for chunk_id, embedding in embeddings:
                    try:
                        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                        conn.execute("""
                            INSERT OR REPLACE INTO chunk_vectors (chunk_id, embedding)
                            VALUES (?, ?)
                        """, (chunk_id, emb_list))
                        count += 1
                    except sqlite3.OperationalError as e:
                        logger.error(f"Failed to add embedding for {chunk_id}: {e}")
                conn.commit()
            return count

        return await asyncio.get_event_loop().run_in_executor(None, _add)

    # ==================== Search Operations ====================

    async def search_fts(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchResult]:
        """Full-text search using FTS5 with BM25 ranking.

        Args:
            query: Search query.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of search results with BM25 scores.
        """
        def _search():
            with self._get_db() as conn:
                # Use BM25 for ranking with rowid-based join for FTS5
                rows = conn.execute("""
                    SELECT
                        c.id as chunk_id,
                        c.document_id,
                        c.text,
                        d.name as document_name,
                        c.metadata,
                        bm25(chunks_fts) as score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.rowid
                    JOIN documents d ON c.document_id = d.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ? OFFSET ?
                """, (query, limit, offset)).fetchall()

                # BM25 returns negative scores, so we negate for ranking
                # Normalize to 0-1 range using sigmoid-like transformation
                results = []
                for row in rows:
                    raw_score = -row["score"]  # Negate since BM25 returns negative for ranking
                    normalized_score = raw_score / (raw_score + 1) if raw_score > 0 else 0
                    results.append(SearchResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        text=row["text"],
                        score=normalized_score,
                        document_name=row["document_name"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    ))
                return results

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _search)
        except sqlite3.OperationalError as e:
            logger.error(f"FTS search failed: {e}")
            return []

    async def search_vector(
        self,
        query_embedding: list[float] | np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Vector similarity search using sqlite-vec.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            similarity_threshold: Minimum similarity score.

        Returns:
            List of search results with similarity scores.
        """
        if not self._vector_enabled:
            logger.warning("Vector search not enabled")
            return []

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        def _search():
            with self._get_db() as conn:
                try:
                    # Cosine distance search (lower is better)
                    rows = conn.execute("""
                        SELECT
                            v.chunk_id,
                            c.document_id,
                            c.text,
                            d.name as document_name,
                            c.metadata,
                            vec_distance_cosine(v.embedding, ?) as distance
                        FROM chunk_vectors v
                        JOIN chunks c ON v.chunk_id = c.id
                        JOIN documents d ON c.document_id = d.id
                        WHERE vec_distance_cosine(v.embedding, ?) >= ?
                        ORDER BY distance
                        LIMIT ?
                    """, (query_embedding, query_embedding, 1 - similarity_threshold, limit)).fetchall()

                    # Convert distance to similarity (1 - distance for cosine)
                    return [
                        SearchResult(
                            chunk_id=row["chunk_id"],
                            document_id=row["document_id"],
                            text=row["text"],
                            score=1 - row["distance"],
                            document_name=row["document_name"],
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        )
                        for row in rows
                    ]
                except sqlite3.OperationalError as e:
                    logger.error(f"Vector search failed: {e}")
                    return []

        return await asyncio.get_event_loop().run_in_executor(None, _search)

    # ==================== Statistics ====================

    async def get_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with statistics.
        """
        def _get():
            with self._get_db() as conn:
                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                vector_count = 0
                if self._vector_enabled:
                    try:
                        vector_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
                    except:
                        pass

                # Get file size
                db_size = self._config.db_path.stat().st_size if self._config.db_path.exists() else 0

                return {
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "vectors": vector_count,
                    "db_size_mb": db_size / (1024 * 1024),
                    "vector_enabled": self._vector_enabled,
                    "fts_enabled": self._config.enable_fts,
                }

        return await asyncio.get_event_loop().run_in_executor(None, _get)

    # ==================== Utility Methods ====================

    def compute_content_hash(self, content: str) -> str:
        """Compute content hash for deduplication.

        Args:
            content: Content to hash.

        Returns:
            SHA256 hash string.
        """
        return hashlib.sha256(content.encode()).hexdigest()

    async def document_exists_by_hash(self, content_hash: str) -> Optional[str]:
        """Check if a document with the given content hash exists.

        Args:
            content_hash: SHA256 hash of content.

        Returns:
            Document ID if found, None otherwise.
        """
        def _check():
            with self._get_db() as conn:
                # Check metadata JSON for content_hash
                row = conn.execute("""
                    SELECT id FROM documents
                    WHERE json_extract(metadata, '$.content_hash') = ?
                """, (content_hash,)).fetchone()
                return row["id"] if row else None

        return await asyncio.get_event_loop().run_in_executor(None, _check)

    async def __aenter__(self) -> "SQLiteStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

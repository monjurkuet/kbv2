"""Portable storage layer for Knowledge Base.

This module provides a fully portable, self-contained storage solution using:
- SQLite + FTS5: Documents, full-text search (BM25)
- sqlite-vec: Vector similarity search
- ChromaDB: Alternative vector store with HNSW indexing
- Kuzu: Embedded graph database with Cypher query support

All components are:
- 100% portable (single file/directory)
- Self-hosted (no external services required)
- Open source (MIT/Apache licensed)
"""

from knowledge_base.storage.portable.sqlite_store import SQLiteStore
from knowledge_base.storage.portable.chroma_store import ChromaStore
from knowledge_base.storage.portable.kuzu_store import KuzuGraphStore
from knowledge_base.storage.portable.hybrid_search import HybridSearchEngine
from knowledge_base.storage.portable.config import PortableStorageConfig

__all__ = [
    "SQLiteStore",
    "ChromaStore",
    "KuzuGraphStore",
    "HybridSearchEngine",
    "PortableStorageConfig",
]

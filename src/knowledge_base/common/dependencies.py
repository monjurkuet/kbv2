"""
Common dependency injection utilities for portable storage.

Provides singleton access to storage components (SQLite, ChromaDB, Kuzu)
without the need for PostgreSQL/SQLAlchemy session management.
"""

from typing import Optional

from knowledge_base.storage.portable import (
    SQLiteStore,
    ChromaStore,
    KuzuGraphStore,
    HybridSearchEngine,
    PortableStorageConfig,
)

# Global storage instances (singletons)
_sqlite_store: Optional[SQLiteStore] = None
_chroma_store: Optional[ChromaStore] = None
_kuzu_store: Optional[KuzuGraphStore] = None
_hybrid_search: Optional[HybridSearchEngine] = None
_config: Optional[PortableStorageConfig] = None


def get_config() -> PortableStorageConfig:
    """Get the global storage configuration.

    Returns:
        PortableStorageConfig instance (creates default if not set).
    """
    global _config
    if _config is None:
        _config = PortableStorageConfig()
    return _config


def set_config(config: PortableStorageConfig) -> None:
    """Set the global storage configuration.

    Args:
        config: Configuration to use.
    """
    global _config
    _config = config


def get_sqlite_store() -> SQLiteStore:
    """Get the SQLite store instance.

    Returns:
        SQLiteStore instance (creates new if not initialized).
    """
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore(config=get_config().sqlite)
    return _sqlite_store


def get_chroma_store() -> ChromaStore:
    """Get the ChromaDB store instance.

    Returns:
        ChromaStore instance (creates new if not initialized).
    """
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore(config=get_config().chroma)
    return _chroma_store


def get_kuzu_store() -> KuzuGraphStore:
    """Get the Kuzu graph store instance.

    Returns:
        KuzuGraphStore instance (creates new if not initialized).
    """
    global _kuzu_store
    if _kuzu_store is None:
        _kuzu_store = KuzuGraphStore(config=get_config().kuzu)
    return _kuzu_store


def get_hybrid_search() -> HybridSearchEngine:
    """Get the hybrid search engine instance.

    Returns:
        HybridSearchEngine instance (creates new if not initialized).
    """
    global _hybrid_search
    if _hybrid_search is None:
        config = get_config()
        _hybrid_search = HybridSearchEngine(
            sqlite_store=get_sqlite_store(),
            chroma_store=get_chroma_store(),
            config=config.hybrid_search,
        )
    return _hybrid_search


async def initialize_storage() -> None:
    """Initialize all storage components.

    This should be called at application startup.
    """
    config = get_config()
    config.ensure_directories()

    await get_sqlite_store().initialize()
    await get_chroma_store().initialize()
    await get_kuzu_store().initialize()


async def close_storage() -> None:
    """Close all storage connections.

    This should be called at application shutdown.
    """
    global _sqlite_store, _chroma_store, _kuzu_store, _hybrid_search

    if _sqlite_store:
        await _sqlite_store.close()
        _sqlite_store = None

    if _chroma_store:
        await _chroma_store.close()
        _chroma_store = None

    if _kuzu_store:
        await _kuzu_store.close()
        _kuzu_store = None

    _hybrid_search = None

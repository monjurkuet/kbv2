"""Configuration for portable storage."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SQLiteConfig(BaseModel):
    """SQLite database configuration."""

    db_path: Path = Field(default=Path("data/knowledge.db"), description="Path to SQLite database file")
    enable_fts: bool = Field(default=True, description="Enable FTS5 full-text search")
    enable_vector: bool = Field(default=True, description="Enable sqlite-vec for vector search")
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension")
    pool_size: int = Field(default=5, description="Connection pool size")

    @property
    def db_path_str(self) -> str:
        """Get database path as string."""
        return str(self.db_path)


class ChromaConfig(BaseModel):
    """ChromaDB vector store configuration."""

    persist_directory: Path = Field(default=Path("data/chroma"), description="ChromaDB persistence directory")
    collection_name: str = Field(default="knowledge_base", description="Default collection name")
    embedding_function: str = Field(default="sentence-transformers", description="Embedding function type")
    distance_metric: Literal["cosine", "l2", "ip"] = Field(default="cosine", description="Distance metric")
    hnsw_space: str = Field(default="hnsw:space", description="HNSW space parameter")

    @property
    def persist_directory_str(self) -> str:
        """Get persist directory as string."""
        return str(self.persist_directory)


class KuzuConfig(BaseModel):
    """Kuzu graph database configuration."""

    db_path: Path = Field(default=Path("data/knowledge_graph.kuzu"), description="Path to Kuzu database directory")
    buffer_pool_size: int = Field(default=256, description="Buffer pool size in MB")
    max_num_threads: int = Field(default=4, description="Maximum number of threads for query execution")
    enable_compression: bool = Field(default=True, description="Enable data compression")

    @property
    def db_path_str(self) -> str:
        """Get database path as string."""
        return str(self.db_path)


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""

    bm25_weight: float = Field(default=0.5, description="Weight for BM25 scores")
    vector_weight: float = Field(default=0.5, description="Weight for vector similarity scores")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant")
    default_limit: int = Field(default=10, description="Default number of results")
    min_score: float = Field(default=0.0, description="Minimum score threshold")


class PortableStorageConfig(BaseModel):
    """Complete portable storage configuration."""

    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    kuzu: KuzuConfig = Field(default_factory=KuzuConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)

    # Data directory for all storage components
    data_dir: Path = Field(default=Path("data"), description="Base data directory")

    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma.persist_directory.mkdir(parents=True, exist_ok=True)
        self.kuzu.db_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_prefix": "KB_",
        "env_nested_delimiter": "__",
    }

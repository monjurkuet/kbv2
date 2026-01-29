"""Vector store with pgvector/HNSW."""

from typing import Any

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import event, select

from knowledge_base.persistence.v1.schema import Base, Entity, Chunk


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "knowledge_base"
    user: str = "agentzero"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 10

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class HNSWConfig(BaseSettings):
    """HNSW index configuration."""

    model_config = SettingsConfigDict(env_prefix="HNSW_")

    m: int = 16
    ef_construction: int = 64
    ef_search: int = 100


class VectorStore:
    """Vector store with pgvector/HNSW."""

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        hnsw_config: HNSWConfig | None = None,
    ) -> None:
        """Initialize vector store.

        Args:
            config: Database configuration.
            hnsw_config: HNSW index configuration.
        """
        self._config = config or VectorStoreConfig()
        self._hnsw_config = hnsw_config or HNSWConfig()
        self._engine: Any = None
        self._session_factory: async_sessionmaker[Any] | None = None
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        # Create engine
        self._engine = create_async_engine(
            self._config.url,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            echo=False,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        await self._create_tables()
        await self._setup_pgvector()

        print("Vector store initialized with pgvector support")

    async def _create_tables(self) -> None:
        """Create database tables."""
        assert self._engine is not None
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def _setup_pgvector(self) -> None:
        """Setup pgvector extension and HNSW indexes."""
        dsn = (
            f"postgresql://{self._config.user}:{self._config.password}"
            f"@{self._config.host}:{self._config.port}/{self._config.name}"
        )

        self._pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)

        async with self._pool.acquire() as conn:
            # Register vector type first, before any other operations
            await register_vector(conn)

            # Then create extension and check columns
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Check if embedding column exists and has correct type
            result = await conn.fetch(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'embedding'"
            )
            if result:
                print(f"Chunk embedding column type: {result[0]['data_type']}")
            else:
                print("Warning: embedding column not found in chunks table")

    async def create_entity_embedding_index(self) -> None:
        """Create IVFFlat index for entity embeddings (HNSW limited to 2000 dims)."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_entity_embedding
                ON entities
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )

    async def create_chunk_embedding_index(self) -> None:
        """Create IVFFlat index for chunk embeddings (HNSW limited to 2000 dims)."""
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_embedding
                ON chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )

    def get_session(self) -> AsyncSession:
        """Get database session.

        Returns:
            Async database session.

        Raises:
            RuntimeError: If store not initialized.
        """
        if self._session_factory is None:
            raise RuntimeError("Vector store not initialized")

        return self._session_factory()

    async def search_similar_entities(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search for similar entities by embedding.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            similarity_threshold: Minimum cosine similarity.

        Returns:
            List of similar entities with similarity scores.
        """
        assert self._pool is not None
        query_embedding = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        )
        async with self._pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    name,
                    entity_type,
                    description,
                    properties,
                    confidence,
                    1 - (embedding <=> $1::vector) as similarity
                FROM entities
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> $1::vector) >= $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                similarity_threshold,
                limit,
            )

            return [dict(row) for row in rows]

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks by embedding.

        Args:
            query_embedding: Query vector.
            limit: Maximum results.
            similarity_threshold: Minimum cosine similarity.

        Returns:
            List of similar chunks with similarity scores.
        """
        assert self._pool is not None
        query_embedding = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        )
        async with self._pool.acquire() as conn:
            # Convert embedding to string format for pgvector
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
            rows = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.text,
                    c.chunk_index,
                    c.page_number,
                    d.name as document_name,
                    1 - (c.embedding <=> $1::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.embedding IS NOT NULL
                AND 1 - (c.embedding <=> $1::vector) >= $2
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                similarity_threshold,
                limit,
            )

            return [dict(row) for row in rows]

    async def update_entity_embedding(
        self,
        entity_id: str,
        embedding: list[float],
    ) -> None:
        """Update entity embedding.

        Args:
            entity_id: Entity UUID.
            embedding: Embedding vector.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(Entity).where(Entity.id == entity_id))
            entity = result.scalar_one_or_none()
            if entity:
                # Ensure embedding is a list of floats
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                entity.embedding = embedding
                await session.commit()

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: list[float],
    ) -> None:
        """Update chunk embedding.

        Args:
            chunk_id: Chunk UUID.
            embedding: Embedding vector.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(Chunk).where(Chunk.id == chunk_id))
            chunk = result.scalar_one_or_none()
            if chunk:
                chunk.embedding = (
                    embedding.tolist() if hasattr(embedding, "tolist") else embedding
                )
                await session.commit()

    async def close(self) -> None:
        """Close database connections."""
        if self._pool:
            await self._pool.close()
            self._pool = None

        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def __aenter__(self) -> "VectorStore":
        """Enter async context.

        Returns:
            Self.
        """
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()

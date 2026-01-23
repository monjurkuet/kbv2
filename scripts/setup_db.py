"""Database setup script."""

import asyncio

from sqlalchemy import text

from knowledge_base.persistence.v1.schema import Base
from knowledge_base.persistence.v1.vector_store import VectorStore


async def setup_database() -> None:
    """Setup database with schema and indexes."""
    vector_store = VectorStore()

    try:
        print("Initializing database connection...")
        await vector_store.initialize()

        print("Creating tables...")
        await vector_store._create_tables()

        print("Setting up pgvector extension...")
        await vector_store._setup_pgvector()

        print("Creating HNSW indexes...")
        await vector_store.create_entity_embedding_index()
        await vector_store.create_chunk_embedding_index()

        print("Database setup completed successfully!")

    except Exception as e:
        print(f"Database setup failed: {e}")
        raise
    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(setup_database())

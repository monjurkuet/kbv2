"""Tests for ChromaStore."""

import pytest

from knowledge_base.storage.portable.chroma_store import ChromaStore
from knowledge_base.storage.portable.config import ChromaConfig


@pytest.fixture
def chroma_store(tmp_path):
    """Create a Chroma store with temporary directory."""
    db_path = tmp_path / "test_chroma"
    config = ChromaConfig(db_path=str(db_path))
    store = ChromaStore(config)
    return store


class TestChromaStore:
    """Tests for ChromaStore class."""

    @pytest.mark.asyncio
    async def test_initialize(self, chroma_store):
        """Test store initialization."""
        await chroma_store.initialize()
        assert chroma_store._client is not None

    @pytest.mark.asyncio
    async def test_add_embeddings(self, chroma_store):
        """Test adding embeddings."""
        await chroma_store.initialize()

        ids = ["chunk1", "chunk2", "chunk3"]
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        metadatas = [
            {"document_id": "doc1", "chunk_index": 0},
            {"document_id": "doc1", "chunk_index": 1},
            {"document_id": "doc1", "chunk_index": 2},
        ]
        documents = ["Text 1", "Text 2", "Text 3"]

        await chroma_store.add_embeddings(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    @pytest.mark.asyncio
    async def test_search(self, chroma_store):
        """Test similarity search."""
        await chroma_store.initialize()

        # Add test data
        await chroma_store.add_embeddings(
            ids=["1", "2"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            metadatas=[{"text": "bitcoin"}, {"text": "ethereum"}],
            documents=["Bitcoin text", "Ethereum text"],
        )

        # Search
        results = await chroma_store.search(
            query_embedding=[1.0, 0.0, 0.0],
            n_results=1,
        )

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_delete_embeddings(self, chroma_store):
        """Test deleting embeddings."""
        await chroma_store.initialize()

        # Add embeddings
        await chroma_store.add_embeddings(
            ids=["to_delete"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{}],
            documents=["Test"],
        )

        # Delete
        await chroma_store.delete_embeddings(ids=["to_delete"])

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, chroma_store):
        """Test getting collection statistics."""
        await chroma_store.initialize()

        # Add some data
        await chroma_store.add_embeddings(
            ids=["1", "2", "3"],
            embeddings=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            metadatas=[{}, {}, {}],
            documents=["a", "b", "c"],
        )

        count = await chroma_store.get_collection_count()
        assert count == 3

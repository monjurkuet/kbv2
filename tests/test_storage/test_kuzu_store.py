"""Tests for KuzuGraphStore."""

import pytest
from pathlib import Path

from knowledge_base.storage.portable.kuzu_store import KuzuGraphStore, Entity, Edge
from knowledge_base.storage.portable.config import KuzuConfig


@pytest.fixture
def kuzu_store(tmp_path):
    """Create a Kuzu store with temporary database."""
    db_path = tmp_path / "test_kuzu"
    config = KuzuConfig(db_path=str(db_path))
    store = KuzuGraphStore(config)
    return store


class TestKuzuGraphStore:
    """Tests for KuzuGraphStore class."""

    @pytest.mark.asyncio
    async def test_initialize(self, kuzu_store):
        """Test database initialization."""
        await kuzu_store.initialize()
        assert kuzu_store._initialized is True

    @pytest.mark.asyncio
    async def test_add_entity(self, kuzu_store):
        """Test adding a single entity."""
        await kuzu_store.initialize()

        entity = Entity(
            name="Bitcoin",
            entity_type="CRYPTOCURRENCY",
            description="Digital currency",
            domain="BITCOIN",
            confidence=0.95,
        )

        entity_id = await kuzu_store.add_entity(entity)
        assert entity_id is not None
        assert entity_id == entity.id

    @pytest.mark.asyncio
    async def test_add_entities_batch(self, kuzu_store):
        """Test batch adding entities."""
        await kuzu_store.initialize()

        entities = [
            Entity(
                name="Bitcoin",
                entity_type="CRYPTOCURRENCY",
                description="Digital currency",
                domain="BITCOIN",
                confidence=0.95,
            ),
            Entity(
                name="Ethereum",
                entity_type="CRYPTOCURRENCY",
                description="Smart contract platform",
                domain="DEFI",
                confidence=0.90,
            ),
        ]

        entity_ids = await kuzu_store.add_entities_batch(entities)
        assert len(entity_ids) == 2

    @pytest.mark.asyncio
    async def test_add_edge(self, kuzu_store):
        """Test adding an edge between entities."""
        await kuzu_store.initialize()

        # Add source entity
        source = Entity(
            name="Satoshi",
            entity_type="PERSON",
            description="Bitcoin creator",
            domain="BITCOIN",
            confidence=1.0,
        )
        await kuzu_store.add_entity(source)

        # Add target entity
        target = Entity(
            name="Bitcoin",
            entity_type="CRYPTOCURRENCY",
            description="Digital currency",
            domain="BITCOIN",
            confidence=0.95,
        )
        await kuzu_store.add_entity(target)

        # Add edge
        edge = Edge(
            source_id=source.id,
            target_id=target.id,
            relation_type="CREATED",
            properties={"year": 2009},
        )

        edge_id = await kuzu_store.add_edge(edge)
        assert edge_id is not None

    @pytest.mark.asyncio
    async def test_search_entities(self, kuzu_store):
        """Test searching entities."""
        await kuzu_store.initialize()

        entity = Entity(
            name="Bitcoin ETF",
            entity_type="FINANCIAL_PRODUCT",
            description="Bitcoin exchange traded fund",
            domain="INSTITUTIONAL_CRYPTO",
            confidence=0.85,
        )
        await kuzu_store.add_entity(entity)

        results = await kuzu_store.search_entities(query="Bitcoin")
        assert len(results) > 0
        assert any(e.name == "Bitcoin ETF" for e in results)

    @pytest.mark.asyncio
    async def test_get_stats(self, kuzu_store):
        """Test getting graph statistics."""
        await kuzu_store.initialize()

        entity = Entity(
            name="Test",
            entity_type="TEST",
            description="Test entity",
            domain="GENERAL",
            confidence=1.0,
        )
        await kuzu_store.add_entity(entity)

        stats = await kuzu_store.get_stats()

        assert "entities" in stats
        assert stats["entities"] >= 1
        assert "storage_size_mb" in stats

"""Tests for common utilities."""

import pytest

from knowledge_base.common.utils import entity_to_dict, entities_to_dict_list
from knowledge_base.storage.portable.kuzu_store import Entity


class TestEntityUtils:
    """Tests for entity utility functions."""

    def test_entity_to_dict(self):
        """Test converting entity to dictionary."""
        entity = Entity(
            id="test-id",
            name="Bitcoin",
            entity_type="CRYPTOCURRENCY",
            description="Digital currency",
            domain="BITCOIN",
            confidence=0.95,
        )

        result = entity_to_dict(entity)

        assert result["id"] == "test-id"
        assert result["name"] == "Bitcoin"
        assert result["entity_type"] == "CRYPTOCURRENCY"
        assert result["description"] == "Digital currency"
        assert result["domain"] == "BITCOIN"

    def test_entities_to_dict_list(self):
        """Test converting list of entities to dictionaries."""
        entities = [
            Entity(
                id="1",
                name="Bitcoin",
                entity_type="CRYPTOCURRENCY",
                description="Digital currency",
                domain="BITCOIN",
                confidence=0.95,
            ),
            Entity(
                id="2",
                name="Ethereum",
                entity_type="CRYPTOCURRENCY",
                description="Smart contract platform",
                domain="DEFI",
                confidence=0.90,
            ),
        ]

        result = entities_to_dict_list(entities)

        assert len(result) == 2
        assert result[0]["name"] == "Bitcoin"
        assert result[1]["name"] == "Ethereum"

    def test_entities_to_dict_list_with_filter(self):
        """Test filtering entities by domain."""
        entities = [
            Entity(
                id="1",
                name="Bitcoin",
                entity_type="CRYPTOCURRENCY",
                description="Digital currency",
                domain="BITCOIN",
                confidence=0.95,
            ),
            Entity(
                id="2",
                name="Ethereum",
                entity_type="CRYPTOCURRENCY",
                description="Smart contract platform",
                domain="DEFI",
                confidence=0.90,
            ),
        ]

        result = entities_to_dict_list(entities, domain_filter="DEFI")

        assert len(result) == 1
        assert result[0]["name"] == "Ethereum"
        assert result[0]["domain"] == "DEFI"

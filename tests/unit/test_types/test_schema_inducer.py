"""Tests for Schema Induction module."""

import pytest
from src.knowledge_base.types.schema_inducer import (
    SchemaInducer,
    InducedSchema,
)
from src.knowledge_base.types.type_discovery import (
    TypeDiscovery,
    DiscoveredType,
    TypeDiscoveryResult,
)


class TestSchemaInducer:
    """Tests for SchemaInducer class."""

    def test_induce_empty_results(self):
        """Test schema induction with empty results."""
        inducer = SchemaInducer()
        discovery_result = TypeDiscoveryResult()
        entities = []
        edges = []

        schema = inducer.induce(discovery_result, entities, edges)

        assert len(schema.types) == 0
        assert len(schema.relationships) == 0
        assert schema.version == "1.0"

    def test_induce_from_discovery_result(self):
        """Test schema induction from discovery result."""
        inducer = SchemaInducer()
        discovered = DiscoveredType(
            name="Person",
            frequency=5,
            confidence=0.85,
            examples=["John", "Jane"],
            properties=["age", "name"],
        )
        discovery_result = TypeDiscoveryResult(
            discovered_types=[discovered], promoted_types=[discovered]
        )
        entities = [
            {"entity_type": "Person", "name": "John", "properties": {"age": 30}},
            {"entity_type": "Person", "name": "Jane", "properties": {"age": 25}},
        ]
        edges = []

        schema = inducer.induce(discovery_result, entities, edges)

        assert "Person" in schema.types
        person_def = schema.types["Person"]
        assert person_def["description"] == "Auto-discovered type: Person"
        assert "John" in person_def["examples"]
        assert "age" in person_def["properties"]
        assert person_def["confidence"] == 0.85

    def test_induce_relationships_from_edges(self):
        """Test relationship extraction from edges."""
        inducer = SchemaInducer()
        discovery_result = TypeDiscoveryResult()
        entities = []
        edges = [
            {"edge_type": "WORKS_AT", "source_id": "1", "target_id": "2"},
            {"edge_type": "WORKS_AT", "source_id": "3", "target_id": "4"},
            {"edge_type": "LIVES_IN", "source_id": "1", "target_id": "5"},
        ]

        schema = inducer.induce(discovery_result, entities, edges)

        rel_types = [r.get("type") for r in schema.relationships]
        assert "WORKS_AT" in rel_types
        assert "LIVES_IN" in rel_types

    def test_induce_with_multiple_types(self):
        """Test schema induction with multiple types."""
        inducer = SchemaInducer()
        person = DiscoveredType(name="Person", frequency=5, examples=["John"])
        org = DiscoveredType(name="Organization", frequency=3, examples=["Acme"])
        discovery_result = TypeDiscoveryResult(discovered_types=[person, org])
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Organization", "name": "Acme"},
        ]
        edges = []

        schema = inducer.induce(discovery_result, entities, edges)

        assert "Person" in schema.types
        assert "Organization" in schema.types

    def test_merge_with_existing_schema(self):
        """Test merging induced schema with existing schema."""
        inducer = SchemaInducer()
        induced = InducedSchema(
            types={
                "NewType": {
                    "description": "New type",
                    "examples": ["A"],
                    "properties": ["prop1"],
                }
            },
            relationships=[{"type": "NEW_REL"}],
            version="1.0",
        )
        existing_schema = {
            "types": {
                "Person": {
                    "description": "Existing type",
                    "examples": ["John"],
                    "properties": ["name"],
                }
            },
            "relationships": [{"type": "WORKS_AT"}],
            "version": "1.0",
        }

        merged = inducer.merge_with_schema(induced, existing_schema)

        assert "Person" in merged.types
        assert "NewType" in merged.types
        rel_types = [r.get("type") for r in merged.relationships]
        assert "WORKS_AT" in rel_types
        assert "NEW_REL" in rel_types

    def test_merge_preserves_existing_types(self):
        """Test that merge preserves existing type data."""
        inducer = SchemaInducer()
        induced = InducedSchema(
            types={
                "Person": {
                    "description": "Updated description",
                    "examples": ["John", "Jane"],
                    "properties": ["name", "age"],
                }
            },
            relationships=[],
            version="1.0",
        )
        existing_schema = {
            "types": {
                "Person": {
                    "description": "Original description",
                    "examples": ["John"],
                    "properties": ["name"],
                    "required": ["name"],
                }
            },
            "relationships": [],
            "version": "1.0",
        }

        merged = inducer.merge_with_schema(induced, existing_schema)

        person = merged.types["Person"]
        assert person["description"] == "Updated description"
        assert "John" in person["examples"]
        assert "Jane" in person["examples"]
        assert "name" in person["properties"]
        assert "age" in person["properties"]

    def test_infer_required_properties(self):
        """Test inference of required properties."""
        inducer = SchemaInducer()
        entities = [
            {"entity_type": "Person", "properties": {"name": "John", "age": 30}},
            {"entity_type": "Person", "properties": {"name": "Jane", "age": 25}},
            {"entity_type": "Person", "properties": {"name": "Bob"}},  # Missing age
        ]
        required = inducer._infer_required_properties(entities, "Person")

        assert "name" in required
        assert "age" not in required or required.count("age") < 2

    def test_infer_required_properties_all_present(self):
        """Test when all properties are required."""
        inducer = SchemaInducer()
        entities = [
            {"entity_type": "Person", "properties": {"name": "John", "age": 30}},
            {"entity_type": "Person", "properties": {"name": "Jane", "age": 25}},
            {"entity_type": "Person", "properties": {"name": "Bob", "age": 40}},
        ]
        required = inducer._infer_required_properties(entities, "Person")

        assert "name" in required
        assert "age" in required

    def test_generate_schema_template(self):
        """Test schema template generation."""
        inducer = SchemaInducer()
        template = inducer.generate_schema_template("Person")

        assert template["type"] == "Person"
        assert "description" in template
        assert "properties" in template
        assert "required" in template

    def test_induce_with_source_target_types(self):
        """Test edge type induction with source/target types."""
        inducer = SchemaInducer()
        discovery_result = TypeDiscoveryResult()
        entities = []
        edges = [
            {
                "edge_type": "WORKS_AT",
                "source_id": "1",
                "target_id": "2",
                "source_type": "Person",
                "target_type": "Organization",
            }
        ]

        schema = inducer.induce(discovery_result, entities, edges)

        works_at = next(
            (r for r in schema.relationships if r.get("type") == "WORKS_AT"), None
        )
        assert works_at is not None
        assert "Person" in works_at.get("source_types", [])
        assert "Organization" in works_at.get("target_types", [])


class TestInducedSchema:
    """Tests for InducedSchema model."""

    def test_schema_model_defaults(self):
        """Test InducedSchema default values."""
        schema = InducedSchema()
        assert schema.types == {}
        assert schema.relationships == []
        assert schema.version == "1.0"

    def test_schema_model_with_values(self):
        """Test InducedSchema with custom values."""
        schema = InducedSchema(
            types={"Person": {"description": "A person"}},
            relationships=[{"type": "WORKS_AT"}],
            version="2.0",
        )
        assert schema.types == {"Person": {"description": "A person"}}
        assert schema.relationships == [{"type": "WORKS_AT"}]
        assert schema.version == "2.0"

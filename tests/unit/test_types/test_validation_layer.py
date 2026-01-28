"""Tests for Validation Layer module."""

import pytest
from src.knowledge_base.types.validation_layer import (
    ValidationLayer,
    ValidationResult,
)


class TestValidationLayer:
    """Tests for ValidationLayer class."""

    def test_validate_empty_entities(self):
        """Test validation with empty entity list."""
        schema = {"types": {}, "relationships": []}
        layer = ValidationLayer(schema)
        result = layer.validate_entities([])
        assert result.is_valid is True
        assert result.validated_entities == 0

    def test_validate_empty_edges(self):
        """Test validation with empty edge list."""
        schema = {"types": {}, "relationships": []}
        layer = ValidationLayer(schema)
        result = layer.validate_edges([])
        assert result.is_valid is True
        assert result.validated_edges == 0

    def test_validate_valid_entities(self):
        """Test validation with valid entities."""
        schema = {"types": {"Person": {"required": ["name"]}}, "relationships": []}
        layer = ValidationLayer(schema)
        entities = [{"entity_type": "Person", "name": "John"}]
        result = layer.validate_entities(entities)
        assert result.is_valid is True
        assert result.validated_entities == 1

    def test_validate_invalid_entity_type(self):
        """Test validation fails for unknown entity type."""
        schema = {"types": {"Person": {}}, "relationships": []}
        layer = ValidationLayer(schema)
        entities = [{"entity_type": "UnknownType"}]
        result = layer.validate_entities(entities)
        assert result.is_valid is False
        assert any("Unknown entity type" in err for err in result.errors)

    def test_validate_missing_entity_type(self):
        """Test validation fails when entity_type is missing."""
        schema = {"types": {}, "relationships": []}
        layer = ValidationLayer(schema)
        entities = [
            {"name": "John"}  # No entity_type
        ]
        result = layer.validate_entities(entities)
        assert result.is_valid is False
        assert any("missing entity_type" in err.lower() for err in result.errors)

    def test_validate_missing_required_property(self):
        """Test validation fails for missing required properties."""
        schema = {
            "types": {"Person": {"required": ["name", "age"]}},
            "relationships": [],
        }
        layer = ValidationLayer(schema)
        entities = [
            {"entity_type": "Person", "name": "John"}  # Missing age
        ]
        result = layer.validate_entities(entities)
        assert result.is_valid is False
        assert any("Missing required property" in err for err in result.errors)

    def test_validate_valid_edges(self):
        """Test validation with valid edges."""
        schema = {"types": {}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        edges = [{"edge_type": "WORKS_AT", "source_id": "1", "target_id": "2"}]
        result = layer.validate_edges(edges)
        assert result.is_valid is True
        assert result.validated_edges == 1

    def test_validate_invalid_edge_type(self):
        """Test validation fails for unknown edge type."""
        schema = {"types": {}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        edges = [{"edge_type": "UNKNOWN_REL", "source_id": "1", "target_id": "2"}]
        result = layer.validate_edges(edges)
        assert result.is_valid is False
        assert any("Unknown edge type" in err for err in result.errors)

    def test_validate_missing_source_id(self):
        """Test validation fails for missing source_id."""
        schema = {"types": {}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        edges = [
            {"edge_type": "WORKS_AT", "target_id": "2"}  # Missing source_id
        ]
        result = layer.validate_edges(edges)
        assert result.is_valid is False
        assert any("missing source_id" in err.lower() for err in result.errors)

    def test_validate_missing_target_id(self):
        """Test validation fails for missing target_id."""
        schema = {"types": {}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        edges = [
            {"edge_type": "WORKS_AT", "source_id": "1"}  # Missing target_id
        ]
        result = layer.validate_edges(edges)
        assert result.is_valid is False
        assert any("missing target_id" in err.lower() for err in result.errors)

    def test_validate_all_entities_and_edges(self):
        """Test combined entity and edge validation."""
        schema = {
            "types": {"Person": {"required": []}},
            "relationships": [{"type": "WORKS_AT"}],
        }
        layer = ValidationLayer(schema)
        entities = [{"entity_type": "Person", "name": "John"}]
        edges = [{"edge_type": "WORKS_AT", "source_id": "1", "target_id": "2"}]
        result = layer.validate_all(entities, edges)
        assert result.is_valid is True
        assert result.validated_entities == 1
        assert result.validated_edges == 1

    def test_validate_all_fails_on_errors(self):
        """Test combined validation fails when either fails."""
        schema = {"types": {"Person": {}}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        entities = [
            {"entity_type": "Unknown"}  # Invalid type
        ]
        edges = [{"edge_type": "WORKS_AT", "source_id": "1", "target_id": "2"}]
        result = layer.validate_all(entities, edges)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(
            is_valid=True, validated_entities=5, validated_edges=3
        )
        assert result.is_valid is True
        assert result.validated_entities == 5
        assert result.validated_edges == 3
        assert result.errors == []

    def test_result_invalid(self):
        """Test invalid validation result with errors."""
        result = ValidationResult(
            is_valid=False, errors=["Error 1", "Error 2"], validated_entities=2
        )
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.validated_entities == 2

    def test_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(is_valid=True)
        assert result.errors == []
        assert result.warnings == []
        assert result.validated_entities == 0
        assert result.validated_edges == 0


class TestValidationLayerHelpers:
    """Tests for ValidationLayer helper methods."""

    def test_add_type(self):
        """Test adding a new type."""
        schema = {"types": {}, "relationships": []}
        layer = ValidationLayer(schema)
        layer.add_type("Person", {"description": "A person"})

        assert layer.is_type_valid("Person")
        assert "Person" in schema["types"]

    def test_add_edge_type(self):
        """Test adding a new edge type."""
        schema = {"types": {}, "relationships": []}
        layer = ValidationLayer(schema)
        layer.add_edge_type("WORKS_AT")

        assert layer.is_edge_type_valid("WORKS_AT")
        rel_types = [r.get("type") for r in schema["relationships"]]
        assert "WORKS_AT" in rel_types

    def test_is_type_valid(self):
        """Test type validation check."""
        schema = {"types": {"Person": {}, "Organization": {}}, "relationships": []}
        layer = ValidationLayer(schema)
        assert layer.is_type_valid("Person") is True
        assert layer.is_type_valid("Organization") is True
        assert layer.is_type_valid("Unknown") is False

    def test_is_edge_type_valid(self):
        """Test edge type validation check."""
        schema = {
            "types": {},
            "relationships": [{"type": "WORKS_AT"}, {"type": "LIVES_IN"}],
        }
        layer = ValidationLayer(schema)
        assert layer.is_edge_type_valid("WORKS_AT") is True
        assert layer.is_edge_type_valid("LIVES_IN") is True
        assert layer.is_edge_type_valid("UNKNOWN") is False

    def test_add_type_already_exists(self):
        """Test adding type that already exists."""
        schema = {"types": {"Person": {"description": "Old"}}, "relationships": []}
        layer = ValidationLayer(schema)
        layer.add_type("Person", {"description": "New"})

        assert "Person" in schema["types"]
        assert layer.is_type_valid("Person")

    def test_add_edge_type_already_exists(self):
        """Test adding edge type that already exists."""
        schema = {"types": {}, "relationships": [{"type": "WORKS_AT"}]}
        layer = ValidationLayer(schema)
        layer.add_edge_type("WORKS_AT")

        count = sum(1 for r in schema["relationships"] if r.get("type") == "WORKS_AT")
        assert count == 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_validate_entity_with_non_dict_properties(self):
        """Test validation when properties is not a dict."""
        schema = {"types": {"Person": {"required": ["name"]}}, "relationships": []}
        layer = ValidationLayer(schema)
        entities = [{"entity_type": "Person", "name": "John", "properties": "invalid"}]
        result = layer.validate_entities(entities)
        assert result.is_valid is True

    def test_validate_entity_with_missing_properties_field(self):
        """Test validation when properties field is missing."""
        schema = {"types": {"Person": {"required": []}}, "relationships": []}
        layer = ValidationLayer(schema)
        entities = [
            {"entity_type": "Person", "name": "John"}  # No properties
        ]
        result = layer.validate_entities(entities)
        assert result.is_valid is True

    def test_validate_default_edge_type(self):
        """Test validation with default edge type."""
        schema = {"types": {}, "relationships": [{"type": "RELATED_TO"}]}
        layer = ValidationLayer(schema)
        edges = [
            {"source_id": "1", "target_id": "2"}  # No edge_type, defaults to RELATED_TO
        ]
        result = layer.validate_edges(edges)
        assert result.is_valid is True

"""Tests for Type Discovery module."""

import pytest
from src.knowledge_base.types.type_discovery import (
    TypeDiscovery,
    DiscoveredType,
    TypeDiscoveryResult,
    TypeDiscoveryConfig,
)


class TestTypeDiscovery:
    """Tests for TypeDiscovery class."""

    def test_discover_empty_entities(self):
        """Test discovery with empty entity list."""
        discovery = TypeDiscovery()
        result = discovery.discover([])
        assert result.processing_time_ms == 0.0
        assert len(result.discovered_types) == 0
        assert len(result.promoted_types) == 0

    def test_discover_single_entity(self):
        """Test discovery with single entity below frequency threshold."""
        discovery = TypeDiscovery()
        entities = [{"entity_type": "Person", "name": "John"}]
        result = discovery.discover(entities)
        assert len(result.discovered_types) == 0

    def test_discover_multiple_entities_same_type(self):
        """Test discovery with multiple entities of same type."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Person", "name": "Jane"},
            {"entity_type": "Person", "name": "Bob"},
        ]
        result = discovery.discover(entities)
        assert len(result.discovered_types) == 1
        assert result.discovered_types[0].name == "Person"
        assert result.discovered_types[0].frequency == 3

    def test_discover_multiple_types(self):
        """Test discovery with multiple entity types."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Person", "name": "Jane"},
            {"entity_type": "Organization", "name": "Acme Corp"},
            {"entity_type": "Organization", "name": "Tech Inc"},
        ]
        result = discovery.discover(entities)
        type_names = [t.name for t in result.discovered_types]
        assert "Person" in type_names
        assert "Organization" in type_names
        assert len(result.discovered_types) == 2

    def test_discover_sorted_by_frequency(self):
        """Test that discovered types are sorted by frequency."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Person", "name": "Jane"},
            {"entity_type": "Person", "name": "Bob"},
            {"entity_type": "Organization", "name": "Acme"},
        ]
        result = discovery.discover(entities)
        assert result.discovered_types[0].name == "Person"
        assert result.discovered_types[1].name == "Organization"

    def test_examples_extraction(self):
        """Test example extraction from entities."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Person", "name": "Jane"},
            {"entity_type": "Person", "name": "Bob"},
        ]
        examples = discovery._extract_examples(entities, "Person")
        assert "John" in examples
        assert "Jane" in examples
        assert "Bob" in examples

    def test_examples_extraction_empty(self):
        """Test example extraction when no examples exist."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person"},  # No name
        ]
        examples = discovery._extract_examples(entities, "Person")
        assert len(examples) == 0

    def test_properties_extraction(self):
        """Test property extraction from entities."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "properties": {"age": 30, "city": "NYC"}},
            {"entity_type": "Person", "properties": {"age": 25, "city": "LA"}},
        ]
        properties = discovery._extract_properties(entities, "Person")
        assert "age" in properties
        assert "city" in properties

    def test_properties_extraction_empty(self):
        """Test property extraction when no properties exist."""
        discovery = TypeDiscovery()
        entities = [
            {"entity_type": "Person", "properties": {}},
        ]
        properties = discovery._extract_properties(entities, "Person")
        assert len(properties) == 0

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        discovery = TypeDiscovery()
        confidence = discovery._calculate_confidence(
            frequency=5, examples=["John", "Jane"], properties=["age", "city"]
        )
        assert 0.0 <= confidence <= 1.0

    def test_confidence_high_frequency(self):
        """Test confidence increases with frequency."""
        discovery = TypeDiscovery()
        low_conf = discovery._calculate_confidence(
            frequency=1, examples=["John"], properties=["age"]
        )
        high_conf = discovery._calculate_confidence(
            frequency=10,
            examples=["John", "Jane", "Bob", "Alice"],
            properties=["age", "city", "email"],
        )
        assert high_conf > low_conf

    def test_known_types_tracking(self):
        """Test known types are tracked correctly."""
        discovery = TypeDiscovery()
        discovery.set_known_types({"Person", "Organization"})
        entities = [
            {"entity_type": "Person", "name": "John"},
            {"entity_type": "Person", "name": "Jane"},
            {"entity_type": "UnknownType", "name": "Thing"},
        ]
        result = discovery.discover(entities)
        person_type = next(
            (t for t in result.discovered_types if t.name == "Person"), None
        )
        unknown_type = next(
            (t for t in result.discovered_types if t.name == "UnknownType"), None
        )
        assert person_type is not None
        assert unknown_type is not None
        assert person_type.is_promoted is False
        assert unknown_type.is_promoted is True

    def test_promotion_threshold(self):
        """Test type promotion based on confidence threshold."""
        config = TypeDiscoveryConfig(promotion_threshold=0.5, min_frequency=2)
        discovery = TypeDiscovery(config=config)
        entities = [
            {"entity_type": "NewType", "name": "Item1"},
            {"entity_type": "NewType", "name": "Item2"},
        ]
        result = discovery.discover(entities)
        new_type = next(
            (t for t in result.discovered_types if t.name == "NewType"), None
        )
        assert new_type is not None
        assert new_type.is_promoted is True
        assert new_type in result.promoted_types

    def test_max_new_types_limit(self):
        """Test max new types limit."""
        config = TypeDiscoveryConfig(max_new_types=2, min_frequency=2)
        discovery = TypeDiscovery(config=config)
        entities = [
            {"entity_type": "Type1", "name": "A"},
            {"entity_type": "Type1", "name": "B"},
            {"entity_type": "Type2", "name": "C"},
            {"entity_type": "Type2", "name": "D"},
            {"entity_type": "Type3", "name": "E"},
            {"entity_type": "Type3", "name": "F"},
        ]
        result = discovery.discover(entities)
        promoted_count = len(result.promoted_types)
        assert promoted_count <= 2

    def test_type_hierarchy_build(self):
        """Test type hierarchy building."""
        config = TypeDiscoveryConfig(enable_hierarchy=True)
        discovery = TypeDiscovery(config=config)
        entities = [
            {"entity_type": "Animal", "name": "Pet1"},
            {"entity_type": "Animal", "name": "Pet2"},
            {"entity_type": "Animals", "name": "Group1"},
            {"entity_type": "Animals", "name": "Group2"},
        ]
        result = discovery.discover(entities)
        assert len(result.type_hierarchy) > 0

    def test_discovered_type_model(self):
        """Test DiscoveredType pydantic model."""
        type_obj = DiscoveredType(
            name="TestType",
            frequency=5,
            confidence=0.8,
            examples=["A", "B"],
            properties=["prop1", "prop2"],
            parent_type="BaseType",
            is_promoted=True,
        )
        assert type_obj.name == "TestType"
        assert type_obj.frequency == 5
        assert type_obj.confidence == 0.8
        assert len(type_obj.examples) == 2
        assert len(type_obj.properties) == 2
        assert type_obj.parent_type == "BaseType"
        assert type_obj.is_promoted is True

    def test_type_discovery_result_model(self):
        """Test TypeDiscoveryResult pydantic model."""
        discovered = DiscoveredType(name="Test", frequency=3)
        result = TypeDiscoveryResult(
            discovered_types=[discovered],
            promoted_types=[discovered],
            type_hierarchy={"Base": ["Test"]},
            processing_time_ms=10.5,
        )
        assert len(result.discovered_types) == 1
        assert len(result.promoted_types) == 1
        assert result.type_hierarchy == {"Base": ["Test"]}
        assert result.processing_time_ms == 10.5


class TestConfidenceCalculation:
    """Tests for confidence calculation logic."""

    def test_confidence_bounds(self):
        """Test confidence is always between 0 and 1."""
        discovery = TypeDiscovery()
        for freq in [1, 5, 10, 20]:
            for ex_count in [0, 5, 15]:
                for prop_count in [0, 3, 10]:
                    conf = discovery._calculate_confidence(
                        frequency=freq,
                        examples=["A"] * ex_count,
                        properties=["p"] * prop_count,
                    )
                    assert 0.0 <= conf <= 1.0

    def test_confidence_no_examples_low(self):
        """Test confidence is lower with no examples."""
        discovery = TypeDiscovery()
        with_examples = discovery._calculate_confidence(
            frequency=5, examples=["A", "B", "C"], properties=["age"]
        )
        without_examples = discovery._calculate_confidence(
            frequency=5, examples=[], properties=["age"]
        )
        assert with_examples > without_examples

    def test_confidence_no_properties_low(self):
        """Test confidence is lower with no properties."""
        discovery = TypeDiscovery()
        with_props = discovery._calculate_confidence(
            frequency=5, examples=["A", "B"], properties=["age", "city", "email"]
        )
        without_props = discovery._calculate_confidence(
            frequency=5, examples=["A", "B"], properties=[]
        )
        assert with_props > without_props


class TestParentTypeFinding:
    """Tests for parent type finding."""

    def test_find_parent_plural_form(self):
        """Test finding parent with plural form."""
        discovery = TypeDiscovery()
        types = [
            DiscoveredType(name="Animal", frequency=5),
            DiscoveredType(name="Animals", frequency=3),
        ]
        parent = discovery._find_parent_type("Animals", types)
        assert parent is not None
        assert parent.name == "Animal"

    def test_find_parent_prefix(self):
        """Test finding parent with prefix."""
        discovery = TypeDiscovery()
        types = [
            DiscoveredType(name="Vehicle", frequency=5),
            DiscoveredType(name="Vehicle_Car", frequency=3),
        ]
        parent = discovery._find_parent_type("Vehicle_Car", types)
        assert parent is not None
        assert parent.name == "Vehicle"

    def test_find_parent_space(self):
        """Test finding parent with space prefix."""
        discovery = TypeDiscovery()
        types = [
            DiscoveredType(name="Event", frequency=5),
            DiscoveredType(name="Event Concert", frequency=3),
        ]
        parent = discovery._find_parent_type("Event Concert", types)
        assert parent is not None
        assert parent.name == "Event"

    def test_no_parent_found(self):
        """Test when no parent is found."""
        discovery = TypeDiscovery()
        types = [
            DiscoveredType(name="Apple", frequency=5),
            DiscoveredType(name="Banana", frequency=3),
        ]
        parent = discovery._find_parent_type("Banana", types)
        assert parent is None

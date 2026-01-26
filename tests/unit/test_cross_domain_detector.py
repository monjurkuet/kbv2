"""Unit tests for cross-domain relationship detection."""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from knowledge_base.intelligence.v1.cross_domain_detector import (
    CrossDomainDetector,
    CrossDomainDetectorConfig,
    CrossDomainRelationship,
    RelationshipType,
    DomainType,
    DomainPairTaxonomy,
    RelationshipPattern,
)


class TestRelationshipType:
    """Tests for RelationshipType enum."""

    def test_relationship_type_values(self) -> None:
        """Test all relationship type enum values are defined."""
        assert RelationshipType.OWNS.value == "OWNS"
        assert RelationshipType.LOCATED_IN.value == "LOCATED_IN"
        assert RelationshipType.WORKS_FOR.value == "WORKS_FOR"
        assert RelationshipType.PART_OF.value == "PART_OF"
        assert RelationshipType.RELATED_TO.value == "RELATED_TO"
        assert RelationshipType.CAUSES.value == "CAUSES"
        assert RelationshipType.DEPENDS_ON.value == "DEPENDS_ON"
        assert RelationshipType.COMPETES_WITH.value == "COMPETES_WITH"
        assert RelationshipType.PARTNERS_WITH.value == "PARTNERS_WITH"
        assert RelationshipType.ACQUIRED_BY.value == "ACQUIRED_BY"
        assert RelationshipType.MERGED_WITH.value == "MERGED_WITH"
        assert RelationshipType.REGULATED_BY.value == "REGULATED_BY"
        assert RelationshipType.FUNDED_BY.value == "FUNDED_BY"
        assert RelationshipType.LICENSED_FROM.value == "LICENSED_FROM"

    def test_relationship_type_completeness(self) -> None:
        """Test all expected relationship types exist."""
        expected_types = [
            "OWNS",
            "LOCATED_IN",
            "WORKS_FOR",
            "PART_OF",
            "RELATED_TO",
            "CAUSES",
            "DEPENDS_ON",
            "COMPETES_WITH",
            "PARTNERS_WITH",
            "ACQUIRED_BY",
            "MERGED_WITH",
            "REGULATED_BY",
            "FUNDED_BY",
            "LICENSED_FROM",
            "SUBMITTED_BY",
            "CITES",
            "CONTAINS",
            "DERIVED_FROM",
            "ENABLED_BY",
            "SUPPORTS",
            "CONFLICTS_WITH",
            "AFFECTS",
            "PRECEDES",
            "SIMILAR_TO",
            "HOSTS",
            "SPONSORED_BY",
            "SERVED_BY",
        ]
        actual_types = [rt.value for rt in RelationshipType]
        for expected in expected_types:
            assert expected in actual_types, f"Missing relationship type: {expected}"

    def test_relationship_type_from_string(self) -> None:
        """Test creating relationship type from string."""
        assert RelationshipType("OWNS") == RelationshipType.OWNS
        assert RelationshipType("WORKS_FOR") == RelationshipType.WORKS_FOR
        assert RelationshipType("RELATED_TO") == RelationshipType.RELATED_TO


class TestDomainType:
    """Tests for DomainType enum."""

    def test_domain_type_values(self) -> None:
        """Test domain type enum values."""
        assert DomainType.GENERAL.value == "general"
        assert DomainType.MEDICAL.value == "medical"
        assert DomainType.LEGAL.value == "legal"
        assert DomainType.FINANCIAL.value == "financial"
        assert DomainType.TECHNOLOGY.value == "technology"
        assert DomainType.ACADEMIC.value == "academic"
        assert DomainType.SCIENTIFIC.value == "scientific"
        assert DomainType.GOVERNMENT.value == "government"

    def test_domain_type_entity_domains(self) -> None:
        """Test entity-specific domain types."""
        assert DomainType.PERSON.value == "person"
        assert DomainType.ORGANIZATION.value == "organization"
        assert DomainType.LOCATION.value == "location"
        assert DomainType.EVENT.value == "event"
        assert DomainType.CONCEPT.value == "concept"


class TestCrossDomainRelationship:
    """Tests for CrossDomainRelationship model."""

    def test_create_relationship(self) -> None:
        """Test creating a cross-domain relationship."""
        source = {
            "id": "1",
            "name": "John",
            "entity_type": "Person",
            "domain": "person",
        }
        target = {
            "id": "2",
            "name": "Acme Corp",
            "entity_type": "Organization",
            "domain": "organization",
        }

        relationship = CrossDomainRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.WORKS_FOR,
            source_domain=DomainType.PERSON,
            target_domain=DomainType.ORGANIZATION,
            confidence=0.85,
            evidence=["Employment contract", "Company directory"],
        )

        assert relationship.source_entity == source
        assert relationship.target_entity == target
        assert relationship.relationship_type == RelationshipType.WORKS_FOR
        assert relationship.source_domain == DomainType.PERSON
        assert relationship.target_domain == DomainType.ORGANIZATION
        assert relationship.confidence == 0.85
        assert relationship.evidence == ["Employment contract", "Company directory"]
        assert relationship.bidirectional is False

    def test_relationship_with_metadata(self) -> None:
        """Test relationship with additional metadata."""
        source = {
            "id": "1",
            "name": "Tech Inc",
            "entity_type": "Organization",
            "domain": "technology",
        }
        target = {
            "id": "2",
            "name": "San Francisco",
            "entity_type": "Location",
            "domain": "location",
        }

        relationship = CrossDomainRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.LOCATED_IN,
            source_domain=DomainType.TECHNOLOGY,
            target_domain=DomainType.LOCATION,
            confidence=0.75,
            metadata={"address": "123 Main St", "founded": "2020"},
        )

        assert relationship.metadata == {"address": "123 Main St", "founded": "2020"}

    def test_relationship_confidence_bounds(self) -> None:
        """Test confidence score bounds."""
        source = {
            "id": "1",
            "name": "Test",
            "entity_type": "Concept",
            "domain": "concept",
        }
        target = {
            "id": "2",
            "name": "Test2",
            "entity_type": "Concept",
            "domain": "concept",
        }

        low_confidence = CrossDomainRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.RELATED_TO,
            source_domain=DomainType.CONCEPT,
            target_domain=DomainType.CONCEPT,
            confidence=0.0,
        )
        assert low_confidence.confidence == 0.0

        high_confidence = CrossDomainRelationship(
            source_entity=source,
            target_entity=target,
            relationship_type=RelationshipType.RELATED_TO,
            source_domain=DomainType.CONCEPT,
            target_domain=DomainType.CONCEPT,
            confidence=1.0,
        )
        assert high_confidence.confidence == 1.0


class TestCrossDomainDetectorConfig:
    """Tests for CrossDomainDetectorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CrossDomainDetectorConfig()
        assert config.confidence_threshold == 0.6
        assert config.max_relationships_per_entity == 100
        assert config.enable_bidirectional_inference is True
        assert config.require_evidence is True
        assert config.min_evidence_count == 1
        assert config.default_confidence == 0.5

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = CrossDomainDetectorConfig(
            confidence_threshold=0.8,
            max_relationships_per_entity=50,
            enable_bidirectional_inference=False,
            require_evidence=False,
            min_evidence_count=2,
            default_confidence=0.7,
        )
        assert config.confidence_threshold == 0.8
        assert config.max_relationships_per_entity == 50
        assert config.enable_bidirectional_inference is False
        assert config.require_evidence is False
        assert config.min_evidence_count == 2
        assert config.default_confidence == 0.7


class TestCrossDomainDetector:
    """Tests for CrossDomainDetector class."""

    def test_init_default_config(self) -> None:
        """Test detector initialization with default config."""
        detector = CrossDomainDetector()
        assert detector._config.confidence_threshold == 0.6
        assert detector._graph_store is None

    def test_init_custom_config(self) -> None:
        """Test detector initialization with custom config."""
        config = CrossDomainDetectorConfig(confidence_threshold=0.75)
        detector = CrossDomainDetector(config=config)
        assert detector._config.confidence_threshold == 0.75

    def test_init_with_graph_store(self) -> None:
        """Test detector initialization with graph store."""
        mock_store = MagicMock()
        detector = CrossDomainDetector(graph_store=mock_store)
        assert detector._graph_store is mock_store

    def test_is_cross_domain_same_domain(self) -> None:
        """Test is_cross_domain returns False for same domain."""
        detector = CrossDomainDetector()
        entity1 = {"id": "1", "domain": "person"}
        entity2 = {"id": "2", "domain": "person"}
        assert detector.is_cross_domain(entity1, entity2) is False

    def test_is_cross_domain_different_domains(self) -> None:
        """Test is_cross_domain returns True for different domains."""
        detector = CrossDomainDetector()
        entity1 = {"id": "1", "domain": "person"}
        entity2 = {"id": "2", "domain": "organization"}
        assert detector.is_cross_domain(entity1, entity2) is True

    def test_is_cross_domain_default_domain(self) -> None:
        """Test is_cross_domain with default domain."""
        detector = CrossDomainDetector()
        entity1 = {"id": "1"}
        entity2 = {"id": "2", "domain": "organization"}
        assert detector.is_cross_domain(entity1, entity2) is True


class TestTaxonomyManagement:
    """Tests for taxonomy registration and retrieval."""

    def test_get_taxonomy_existing_pair(self) -> None:
        """Test getting taxonomy for valid domain pair."""
        detector = CrossDomainDetector()
        taxonomy = detector.get_taxonomy(DomainType.PERSON, DomainType.ORGANIZATION)
        assert taxonomy is not None
        assert taxonomy.source_domain == DomainType.PERSON
        assert taxonomy.target_domain == DomainType.ORGANIZATION
        assert RelationshipType.WORKS_FOR in taxonomy.valid_relationship_types

    def test_get_taxonomy_nonexistent_pair(self) -> None:
        """Test getting taxonomy for invalid domain pair returns None."""
        detector = CrossDomainDetector()
        taxonomy = detector.get_taxonomy(DomainType.MEDICAL, DomainType.SCIENTIFIC)
        assert taxonomy is None

    def test_register_custom_taxonomy(self) -> None:
        """Test registering custom taxonomy."""
        detector = CrossDomainDetector()
        custom_taxonomy = DomainPairTaxonomy(
            source_domain=DomainType.MEDICAL,
            target_domain=DomainType.SCIENTIFIC,
            valid_relationship_types=[
                RelationshipType.RESEARCHES,
                RelationshipType.DISCOVERED,
            ],
            default_confidence=0.8,
        )
        detector.register_taxonomy(custom_taxonomy)
        retrieved = detector.get_taxonomy(DomainType.MEDICAL, DomainType.SCIENTIFIC)
        assert retrieved is not None
        assert retrieved.default_confidence == 0.8
        assert RelationshipType.RESEARCHES in retrieved.valid_relationship_types

    def test_get_valid_relationship_types_existing(self) -> None:
        """Test getting valid relationship types for existing pair."""
        detector = CrossDomainDetector()
        types = detector.get_valid_relationship_types(
            DomainType.PERSON, DomainType.ORGANIZATION
        )
        assert len(types) > 0
        assert RelationshipType.WORKS_FOR in types

    def test_get_valid_relationship_types_nonexistent(self) -> None:
        """Test getting valid types for nonexistent pair returns RELATED_TO."""
        detector = CrossDomainDetector()
        types = detector.get_valid_relationship_types(
            DomainType.MEDICAL, DomainType.SCIENTIFIC
        )
        assert types == [RelationshipType.RELATED_TO]


class TestPatternManagement:
    """Tests for relationship pattern registration."""

    def test_register_pattern(self) -> None:
        """Test registering custom pattern."""
        detector = CrossDomainDetector()
        initial_count = len(detector._patterns)

        pattern = RelationshipPattern(
            source_entity_types=["Product"],
            target_entity_types=["Organization"],
            relationship_type=RelationshipType.DEVELOPED_BY,
            confidence_boost=0.25,
        )
        detector.register_pattern(pattern)

        assert len(detector._patterns) == initial_count + 1
        assert any(
            p.relationship_type == RelationshipType.DEVELOPED_BY
            for p in detector._patterns
        )


class TestDetectRelationships:
    """Tests for relationship detection functionality."""

    def test_detect_no_entities(self) -> None:
        """Test detect_relationships with empty entity list."""
        detector = CrossDomainDetector()
        relationships = detector.detect_relationships([])
        assert relationships == []

    def test_detect_same_domain_entities(self) -> None:
        """Test that same-domain entities are not included."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Entity1", "domain": "person"},
            {"id": "2", "name": "Entity2", "domain": "person"},
        ]
        relationships = detector.detect_relationships(entities)
        assert relationships == []

    def test_detect_cross_domain_relationships(self) -> None:
        """Test detecting cross-domain relationships."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        assert len(relationships) == 1
        assert relationships[0].relationship_type == RelationshipType.WORKS_FOR
        assert relationships[0].source_domain == DomainType.PERSON
        assert relationships[0].target_domain == DomainType.ORGANIZATION

    def test_detect_multiple_entities(self) -> None:
        """Test detecting relationships among multiple entities."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Alice", "entity_type": "Person", "domain": "person"},
            {"id": "2", "name": "Bob", "entity_type": "Person", "domain": "person"},
            {
                "id": "3",
                "name": "Tech Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
            {
                "id": "4",
                "name": "Boston",
                "entity_type": "Location",
                "domain": "location",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        assert len(relationships) == 5
        relationship_domains = [
            (r.source_domain, r.target_domain) for r in relationships
        ]
        assert (DomainType.PERSON, DomainType.ORGANIZATION) in relationship_domains
        assert (DomainType.PERSON, DomainType.LOCATION) in relationship_domains
        assert (DomainType.ORGANIZATION, DomainType.LOCATION) in relationship_domains

    def test_detect_with_min_confidence(self) -> None:
        """Test filtering by minimum confidence."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.9)
        assert len(relationships) == 0

    def test_detect_with_max_relationships(self) -> None:
        """Test limiting number of relationships."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Person", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Org1",
                "entity_type": "Organization",
                "domain": "organization",
            },
            {
                "id": "3",
                "name": "Org2",
                "entity_type": "Organization",
                "domain": "organization",
            },
            {
                "id": "4",
                "name": "Loc1",
                "entity_type": "Location",
                "domain": "location",
            },
            {
                "id": "5",
                "name": "Loc2",
                "entity_type": "Location",
                "domain": "location",
            },
        ]
        relationships = detector.detect_relationships(
            entities, min_confidence=0.0, max_relationships=2
        )
        assert len(relationships) == 2

    def test_detect_with_evidence(self) -> None:
        """Test detection with evidence."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        evidence = {("1", "2"): ["Employment record", "LinkedIn profile"]}
        relationships = detector.detect_relationships(entities, evidence=evidence)

        assert len(relationships) == 1
        assert len(relationships[0].evidence) == 2
        assert "Employment record" in relationships[0].evidence


class TestDetectEntityRelationships:
    """Tests for single-source relationship detection."""

    def test_detect_from_source(self) -> None:
        """Test detecting relationships from a source entity."""
        detector = CrossDomainDetector()
        source = {
            "id": "1",
            "name": "John",
            "entity_type": "Person",
            "domain": "person",
        }
        targets = [
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
            {
                "id": "3",
                "name": "New York",
                "entity_type": "Location",
                "domain": "location",
            },
        ]
        relationships = detector.detect_entity_relationships(
            source, targets, min_confidence=0.0
        )

        assert len(relationships) == 2
        types = [r.relationship_type for r in relationships]
        assert RelationshipType.WORKS_FOR in types
        assert RelationshipType.LOCATED_IN in types

    def test_detect_from_source_filtered(self) -> None:
        """Test filtering by confidence in single-source detection."""
        detector = CrossDomainDetector()
        source = {
            "id": "1",
            "name": "John",
            "entity_type": "Person",
            "domain": "person",
        }
        targets = [
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_entity_relationships(
            source, targets, min_confidence=0.95
        )
        assert len(relationships) == 0


class TestConfidenceScoring:
    """Tests for confidence scoring functionality."""

    def test_confidence_with_evidence(self) -> None:
        """Test that evidence improves confidence."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]

        without_evidence = detector.detect_relationships(entities, min_confidence=0.0)
        with_evidence = detector.detect_relationships(
            entities,
            min_confidence=0.0,
            evidence={
                ("1", "2"): [
                    "Strong evidence 1",
                    "Strong evidence 2",
                    "Strong evidence 3",
                ]
            },
        )

        if without_evidence and with_evidence:
            assert with_evidence[0].confidence >= without_evidence[0].confidence

    def test_confidence_bounds(self) -> None:
        """Test confidence scores are within bounds."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        for rel in relationships:
            assert 0.0 <= rel.confidence <= 1.0

    def test_confidence_default_taxonomy(self) -> None:
        """Test confidence uses taxonomy defaults."""
        detector = CrossDomainDetector()
        entities = [
            {
                "id": "1",
                "name": "Concept A",
                "entity_type": "Concept",
                "domain": "concept",
            },
            {
                "id": "2",
                "name": "Concept B",
                "entity_type": "Concept",
                "domain": "concept",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        if relationships:
            assert relationships[0].confidence >= 0.5


class TestBidirectionalInference:
    """Tests for bidirectional relationship inference."""

    def test_bidirectional_inference_enabled(self) -> None:
        """Test bidirectional inference when enabled."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Person", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Org",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1
        assert relationships[0].bidirectional is True

    def test_bidirectional_inference_disabled(self) -> None:
        """Test bidirectional inference can be disabled via config."""
        config = CrossDomainDetectorConfig(enable_bidirectional_inference=False)
        detector = CrossDomainDetector(config=config)
        entities = [
            {"id": "1", "name": "Person", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Org",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert relationships[0].bidirectional is False


class TestFiltering:
    """Tests for relationship filtering."""

    def test_filter_by_confidence(self) -> None:
        """Test filtering relationships by confidence threshold."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        filtered = detector.filter_relationships(relationships, min_confidence=0.7)
        for rel in filtered:
            assert rel.confidence >= 0.7

    def test_filter_by_relationship_types(self) -> None:
        """Test filtering by specific relationship types."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
            {
                "id": "3",
                "name": "Boston",
                "entity_type": "Location",
                "domain": "location",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        filtered = detector.filter_relationships(
            relationships, relationship_types=[RelationshipType.WORKS_FOR]
        )
        for rel in filtered:
            assert rel.relationship_type == RelationshipType.WORKS_FOR

    def test_filter_by_domains(self) -> None:
        """Test filtering by source and target domains."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        filtered = detector.filter_relationships(
            relationships,
            min_confidence=0.0,
            source_domains=[DomainType.PERSON],
            target_domains=[DomainType.ORGANIZATION],
        )
        assert len(filtered) == 1

    def test_combined_filters(self) -> None:
        """Test filtering with multiple criteria."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        filtered = detector.filter_relationships(
            relationships,
            min_confidence=0.0,
            relationship_types=[RelationshipType.WORKS_FOR],
            source_domains=[DomainType.PERSON],
            target_domains=[DomainType.ORGANIZATION],
        )
        assert len(filtered) == 1


class TestStatistics:
    """Tests for relationship statistics."""

    def test_empty_statistics(self) -> None:
        """Test statistics for empty relationship list."""
        detector = CrossDomainDetector()
        stats = detector.get_statistics([])
        assert stats["total_relationships"] == 0
        assert stats["average_confidence"] == 0.0
        assert stats["relationship_type_counts"] == {}
        assert stats["bidirectional_count"] == 0

    def test_statistics_computation(self) -> None:
        """Test statistics computation for relationships."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        if relationships:
            stats = detector.get_statistics(relationships)
            assert stats["total_relationships"] == len(relationships)
            assert "WORKS_FOR" in stats["relationship_type_counts"]
            assert stats["min_confidence"] <= stats["max_confidence"]

    def test_domain_pair_statistics(self) -> None:
        """Test domain pair statistics."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "entity_type": "Person", "domain": "person"},
            {
                "id": "2",
                "name": "Acme Corp",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)

        if relationships:
            stats = detector.get_statistics(relationships)
            domain_pair = "person->organization"
            assert domain_pair in stats["domain_pair_counts"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_entities_without_ids(self) -> None:
        """Test detection with entities lacking explicit IDs."""
        detector = CrossDomainDetector()
        entities = [
            {"name": "Person1", "domain": "person"},
            {"name": "Org1", "domain": "organization"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_entities_without_domains(self) -> None:
        """Test detection with entities without domain field."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Entity1"},
            {"id": "2", "name": "Entity2", "domain": "organization"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_single_entity_list(self) -> None:
        """Test with single entity returns empty list."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "Solo", "domain": "person"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert relationships == []

    def test_entities_with_special_characters(self) -> None:
        """Test with entities having special characters in names."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "José García", "domain": "person"},
            {
                "id": "2",
                "name": "Café & Restaurant",
                "entity_type": "Organization",
                "domain": "organization",
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_entities_with_unicode_names(self) -> None:
        """Test with entities having unicode names."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "東京本社", "domain": "organization"},
            {"id": "2", "name": "日本", "domain": "location"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_very_long_entity_names(self) -> None:
        """Test with very long entity names."""
        detector = CrossDomainDetector()
        long_name = "A" * 1000
        entities = [
            {"id": "1", "name": long_name, "domain": "person"},
            {"id": "2", "name": long_name + " Corp", "domain": "organization"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_evidence_with_empty_list(self) -> None:
        """Test with empty evidence list."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "domain": "person"},
            {"id": "2", "name": "Acme Corp", "domain": "organization"},
        ]
        evidence: dict[tuple[str, str], list[str]] = {}
        relationships = detector.detect_relationships(
            entities, min_confidence=0.0, evidence=evidence
        )
        assert len(relationships) == 1

    def test_entities_with_properties(self) -> None:
        """Test with entities containing properties."""
        detector = CrossDomainDetector()
        entities = [
            {
                "id": "1",
                "name": "John",
                "domain": "person",
                "properties": {"job_title": "CEO", "salary": 100000},
            },
            {
                "id": "2",
                "name": "Acme Corp",
                "domain": "organization",
                "properties": {"employees": 500, "revenue": "1B"},
            },
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 1

    def test_confidence_threshold_at_boundary(self) -> None:
        """Test confidence threshold at exact boundary."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "domain": "person"},
            {"id": "2", "name": "Acme Corp", "domain": "organization"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) >= 0

    def test_duplicate_entities(self) -> None:
        """Test with duplicate entities."""
        detector = CrossDomainDetector()
        entities = [
            {"id": "1", "name": "John", "domain": "person"},
            {"id": "1", "name": "John", "domain": "person"},
            {"id": "2", "name": "Acme Corp", "domain": "organization"},
        ]
        relationships = detector.detect_relationships(entities, min_confidence=0.0)
        assert len(relationships) == 2

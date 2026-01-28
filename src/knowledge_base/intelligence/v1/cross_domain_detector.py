"""Cross-domain relationship detection module.

Detects relationships between entities in different domains and provides
a taxonomy of cross-domain relationship types with confidence scoring.
"""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RelationshipType(str, Enum):
    """Cross-domain relationship type taxonomy."""

    OWNS = "OWNS"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"
    CAUSES = "CAUSES"
    DEPENDS_ON = "DEPENDS_ON"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    ACQUIRED_BY = "ACQUIRED_BY"
    MERGED_WITH = "MERGED_WITH"
    REGULATED_BY = "REGULATED_BY"
    FUNDED_BY = "FUNDED_BY"
    LICENSED_FROM = "LICENSED_FROM"
    SUBMITTED_BY = "SUBMITTED_BY"
    CITES = "CITES"
    CONTAINS = "CONTAINS"
    DERIVED_FROM = "DERIVED_FROM"
    ENABLED_BY = "ENABLED_BY"
    SUPPORTS = "SUPPORTS"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    AFFECTS = "AFFECTS"
    PRECEDES = "PRECEDES"
    SIMILAR_TO = "SIMILAR_TO"
    HOSTS = "HOSTS"
    SPONSORED_BY = "SPONSORED_BY"
    SERVED_BY = "SERVED_BY"
    CREATED = "CREATED"
    DEVELOPED = "DEVELOPED"
    BORDERS = "BORDERS"
    INVESTED_IN = "INVESTED_IN"
    RESEARCHES = "RESEARCHES"
    DISCOVERED = "DISCOVERED"
    DEVELOPED_BY = "DEVELOPED_BY"
    USED_BY = "USED_BY"


class DomainType(str, Enum):
    """Domain types for cross-domain analysis."""

    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    ACADEMIC = "academic"
    SCIENTIFIC = "scientific"
    GOVERNMENT = "government"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"


class CrossDomainRelationship(BaseModel):
    """Represents a relationship between entities from different domains."""

    source_entity: dict[str, Any] = Field(
        ..., description="Source entity with id, name, type, domain"
    )
    target_entity: dict[str, Any] = Field(
        ..., description="Target entity with id, name, type, domain"
    )
    relationship_type: RelationshipType = Field(
        ..., description="Type of cross-domain relationship"
    )
    source_domain: DomainType = Field(..., description="Domain of the source entity")
    target_domain: DomainType = Field(..., description="Domain of the target entity")
    confidence: float = Field(
        ..., description="Confidence score for the relationship", ge=0.0, le=1.0
    )
    evidence: list[str] = Field(
        default_factory=list, description="Evidence supporting the relationship"
    )
    bidirectional: bool = Field(
        default=False, description="Whether the relationship is bidirectional"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DomainPairTaxonomy(BaseModel):
    """Defines valid relationship types for a pair of domains."""

    source_domain: DomainType
    target_domain: DomainType
    valid_relationship_types: list[RelationshipType] = Field(
        default_factory=list,
        description="Relationship types valid for this domain pair",
    )
    default_confidence: float = Field(
        default=0.5, description="Default confidence for unknown relationships"
    )
    requires_evidence: bool = Field(
        default=True, description="Whether evidence is required"
    )


class CrossDomainDetectorConfig(BaseSettings):
    """Configuration for cross-domain detection."""

    model_config = SettingsConfigDict()

    confidence_threshold: float = 0.6
    max_relationships_per_entity: int = 100
    enable_bidirectional_inference: bool = True
    require_evidence: bool = True
    min_evidence_count: int = 1
    default_confidence: float = 0.5


class RelationshipPattern(BaseModel):
    """Pattern for detecting relationships based on entity properties."""

    source_entity_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)
    source_domains: list[DomainType] = Field(default_factory=list)
    target_domains: list[DomainType] = Field(default_factory=list)
    relationship_type: RelationshipType
    confidence_boost: float = 0.0
    required_properties: list[str] = Field(default_factory=list)


class CrossDomainDetector:
    """Detects cross-domain relationships between entities.

    This class provides functionality to:
    - Identify relationships between entities from different domains
    - Score relationship confidence based on evidence
    - Maintain a taxonomy of valid domain-pair relationships
    - Filter relationships by confidence thresholds
    """

    def __init__(
        self,
        config: CrossDomainDetectorConfig | None = None,
        graph_store: Any | None = None,
    ) -> None:
        """Initialize cross-domain detector.

        Args:
            config: Cross-domain detection configuration.
            graph_store: Optional graph store for entity/relationship lookups.
        """
        self._config = config or CrossDomainDetectorConfig()
        self._graph_store = graph_store
        self._taxonomy: dict[tuple[DomainType, DomainType], DomainPairTaxonomy] = {}
        self._patterns: list[RelationshipPattern] = []
        self._initialize_taxonomy()
        self._initialize_patterns()

    def _initialize_taxonomy(self) -> None:
        """Initialize the domain pair relationship taxonomy."""
        taxonomy_entries = [
            DomainPairTaxonomy(
                source_domain=DomainType.PERSON,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.WORKS_FOR,
                    RelationshipType.OWNS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.7,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.PERSON,
                target_domain=DomainType.LOCATION,
                valid_relationship_types=[
                    RelationshipType.LOCATED_IN,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.7,
                requires_evidence=False,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.PERSON,
                target_domain=DomainType.CONCEPT,
                valid_relationship_types=[
                    RelationshipType.RELATED_TO,
                    RelationshipType.AFFECTS,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.ORGANIZATION,
                target_domain=DomainType.LOCATION,
                valid_relationship_types=[
                    RelationshipType.LOCATED_IN,
                    RelationshipType.HOSTS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.7,
                requires_evidence=False,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.ORGANIZATION,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.PARTNERS_WITH,
                    RelationshipType.COMPETES_WITH,
                    RelationshipType.ACQUIRED_BY,
                    RelationshipType.MERGED_WITH,
                    RelationshipType.FUNDED_BY,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.ORGANIZATION,
                target_domain=DomainType.CONCEPT,
                valid_relationship_types=[
                    RelationshipType.RELATED_TO,
                    RelationshipType.SUPPORTS,
                    RelationshipType.DEPENDS_ON,
                    RelationshipType.DEVELOPED,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.LOCATION,
                target_domain=DomainType.LOCATION,
                valid_relationship_types=[
                    RelationshipType.CONTAINS,
                    RelationshipType.PART_OF,
                    RelationshipType.BORDERS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.7,
                requires_evidence=False,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.CONCEPT,
                target_domain=DomainType.CONCEPT,
                valid_relationship_types=[
                    RelationshipType.DERIVED_FROM,
                    RelationshipType.CAUSES,
                    RelationshipType.SIMILAR_TO,
                    RelationshipType.CONFLICTS_WITH,
                    RelationshipType.PRECEDES,
                ],
                default_confidence=0.5,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.EVENT,
                target_domain=DomainType.PERSON,
                valid_relationship_types=[
                    RelationshipType.AFFECTS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.EVENT,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.AFFECTS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.TECHNOLOGY,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.RELATED_TO,
                    RelationshipType.DEPENDS_ON,
                    RelationshipType.LICENSED_FROM,
                ],
                default_confidence=0.7,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.MEDICAL,
                target_domain=DomainType.PERSON,
                valid_relationship_types=[
                    RelationshipType.AFFECTS,
                    RelationshipType.RELATED_TO,
                ],
                default_confidence=0.7,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.FINANCIAL,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.FUNDED_BY,
                    RelationshipType.REGULATED_BY,
                    RelationshipType.RELATED_TO,
                    RelationshipType.INVESTED_IN,
                ],
                default_confidence=0.7,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.ACADEMIC,
                target_domain=DomainType.PERSON,
                valid_relationship_types=[
                    RelationshipType.CITES,
                    RelationshipType.RELATED_TO,
                    RelationshipType.WORKS_FOR,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.SCIENTIFIC,
                target_domain=DomainType.CONCEPT,
                valid_relationship_types=[
                    RelationshipType.RELATED_TO,
                    RelationshipType.CAUSES,
                    RelationshipType.DERIVED_FROM,
                ],
                default_confidence=0.6,
                requires_evidence=True,
            ),
            DomainPairTaxonomy(
                source_domain=DomainType.GOVERNMENT,
                target_domain=DomainType.ORGANIZATION,
                valid_relationship_types=[
                    RelationshipType.REGULATED_BY,
                    RelationshipType.FUNDED_BY,
                    RelationshipType.LICENSED_FROM,
                ],
                default_confidence=0.7,
                requires_evidence=True,
            ),
        ]

        for entry in taxonomy_entries:
            key = (entry.source_domain, entry.target_domain)
            self._taxonomy[key] = entry

    def _initialize_patterns(self) -> None:
        """Initialize relationship detection patterns."""
        self._patterns = [
            RelationshipPattern(
                source_entity_types=["Person"],
                target_entity_types=["Organization"],
                source_domains=[DomainType.PERSON],
                target_domains=[DomainType.ORGANIZATION],
                relationship_type=RelationshipType.WORKS_FOR,
                confidence_boost=0.2,
                required_properties=["job_title", "role", "position"],
            ),
            RelationshipPattern(
                source_entity_types=["Person"],
                target_entity_types=["Location"],
                source_domains=[DomainType.PERSON],
                target_domains=[DomainType.LOCATION],
                relationship_type=RelationshipType.LOCATED_IN,
                confidence_boost=0.1,
                required_properties=["address", "city", "country"],
            ),
            RelationshipPattern(
                source_entity_types=["Organization"],
                target_entity_types=["Location"],
                source_domains=[DomainType.ORGANIZATION],
                target_domains=[DomainType.LOCATION],
                relationship_type=RelationshipType.LOCATED_IN,
                confidence_boost=0.1,
                required_properties=["headquarters", "address"],
            ),
            RelationshipPattern(
                source_entity_types=["Organization"],
                target_entity_types=["Organization"],
                source_domains=[DomainType.ORGANIZATION],
                target_domains=[DomainType.ORGANIZATION],
                relationship_type=RelationshipType.PARTNERS_WITH,
                confidence_boost=0.15,
                required_properties=["partner", "collaboration", "joint"],
            ),
            RelationshipPattern(
                source_entity_types=["Person"],
                target_entity_types=["Organization"],
                source_domains=[DomainType.PERSON],
                target_domains=[DomainType.ORGANIZATION],
                relationship_type=RelationshipType.OWNS,
                confidence_boost=0.2,
                required_properties=["owner", "founder", "shareholder"],
            ),
            RelationshipPattern(
                source_entity_types=["Concept"],
                target_entity_types=["Concept"],
                source_domains=[DomainType.CONCEPT],
                target_domains=[DomainType.CONCEPT],
                relationship_type=RelationshipType.CAUSES,
                confidence_boost=0.15,
                required_properties=["cause", "result", "effect"],
            ),
            RelationshipPattern(
                source_entity_types=["Person"],
                target_entity_types=["Concept"],
                source_domains=[DomainType.PERSON],
                target_domains=[DomainType.ACADEMIC, DomainType.SCIENTIFIC],
                relationship_type=RelationshipType.RESEARCHES,
                confidence_boost=0.2,
                required_properties=["research", "study", "paper"],
            ),
        ]

    def register_taxonomy(self, taxonomy: DomainPairTaxonomy) -> None:
        """Register a custom domain pair taxonomy.

        Args:
            taxonomy: Domain pair taxonomy to register.
        """
        key = (taxonomy.source_domain, taxonomy.target_domain)
        self._taxonomy[key] = taxonomy

    def register_pattern(self, pattern: RelationshipPattern) -> None:
        """Register a custom relationship detection pattern.

        Args:
            pattern: Pattern to register.
        """
        self._patterns.append(pattern)

    def get_taxonomy(
        self,
        source_domain: DomainType,
        target_domain: DomainType,
    ) -> DomainPairTaxonomy | None:
        """Get the taxonomy for a domain pair.

        Args:
            source_domain: Source domain.
            target_domain: Target domain.

        Returns:
            DomainPairTaxonomy if found, None otherwise.
        """
        key = (source_domain, target_domain)
        return self._taxonomy.get(key)

    def get_valid_relationship_types(
        self,
        source_domain: DomainType,
        target_domain: DomainType,
    ) -> list[RelationshipType]:
        """Get valid relationship types for a domain pair.

        Args:
            source_domain: Source domain.
            target_domain: Target domain.

        Returns:
            List of valid relationship types.
        """
        taxonomy = self.get_taxonomy(source_domain, target_domain)
        if taxonomy:
            return taxonomy.valid_relationship_types
        return [RelationshipType.RELATED_TO]

    def is_cross_domain(
        self,
        entity1: dict[str, Any],
        entity2: dict[str, Any],
    ) -> bool:
        """Check if two entities are from different domains.

        Args:
            entity1: First entity with domain field.
            entity2: Second entity with domain field.

        Returns:
            True if entities are from different domains.
        """
        domain1_str = entity1.get("domain", "general")
        domain2_str = entity2.get("domain", "general")

        try:
            domain1 = DomainType(domain1_str)
            domain2 = DomainType(domain2_str)
        except ValueError:
            domain1 = DomainType.GENERAL
            domain2 = DomainType.GENERAL

        return domain1 != domain2

    def _calculate_confidence(
        self,
        source_entity: dict[str, Any],
        target_entity: dict[str, Any],
        relationship_type: RelationshipType,
        evidence: list[str],
    ) -> float:
        """Calculate confidence score for a relationship.

        Args:
            source_entity: Source entity.
            target_entity: Target entity.
            relationship_type: Type of relationship.
            evidence: Evidence supporting the relationship.

        Returns:
            Confidence score between 0 and 1.
        """
        source_domain = DomainType(source_entity.get("domain", DomainType.GENERAL))
        target_domain = DomainType(target_entity.get("domain", DomainType.GENERAL))

        taxonomy = self.get_taxonomy(source_domain, target_domain)
        base_confidence = (
            taxonomy.default_confidence if taxonomy else self._config.default_confidence
        )

        if relationship_type in (
            relationship_type for relationship_type in RelationshipType
        ):
            pass

        evidence_boost = min(len(evidence) * 0.1, 0.3)

        properties_boost = 0.0
        source_props = source_entity.get("properties", {})
        target_props = target_entity.get("properties", {})

        for pattern in self._patterns:
            if pattern.relationship_type == relationship_type:
                if pattern.required_properties:
                    source_match = any(
                        prop in source_props for prop in pattern.required_properties
                    )
                    target_match = any(
                        prop in target_props for prop in pattern.required_properties
                    )
                    if source_match and target_match:
                        properties_boost = pattern.confidence_boost

        confidence = min(base_confidence + evidence_boost + properties_boost, 1.0)

        if (
            self._config.require_evidence
            and len(evidence) < self._config.min_evidence_count
        ):
            confidence *= 0.8

        return confidence

    def _infer_relationship_type(
        self,
        source_entity: dict[str, Any],
        target_entity: dict[str, Any],
    ) -> RelationshipType | None:
        """Infer the most likely relationship type between two entities.

        Args:
            source_entity: Source entity.
            target_entity: Target entity.

        Returns:
            Inferred relationship type or None.
        """
        source_type = source_entity.get("entity_type", "")
        target_type = target_entity.get("entity_type", "")
        source_domain = DomainType(source_entity.get("domain", DomainType.GENERAL))
        target_domain = DomainType(target_entity.get("domain", DomainType.GENERAL))

        for pattern in self._patterns:
            source_match = (
                not pattern.source_entity_types
                or source_type in pattern.source_entity_types
            )
            target_match = (
                not pattern.target_entity_types
                or target_type in pattern.target_entity_types
            )
            domain_source_match = (
                not pattern.source_domains or source_domain in pattern.source_domains
            )
            domain_target_match = (
                not pattern.target_domains or target_domain in pattern.target_domains
            )

            if (
                source_match
                and target_match
                and domain_source_match
                and domain_target_match
            ):
                return pattern.relationship_type

        if source_type == "Person" and target_type == "Organization":
            return RelationshipType.WORKS_FOR
        elif source_type == "Person" and target_type == "Location":
            return RelationshipType.LOCATED_IN
        elif source_type == "Organization" and target_type == "Location":
            return RelationshipType.LOCATED_IN
        elif source_type == "Organization" and target_type == "Organization":
            return RelationshipType.PARTNERS_WITH
        elif source_type == "Concept" and target_type == "Concept":
            return RelationshipType.RELATED_TO

        return RelationshipType.RELATED_TO

    def detect_relationships(
        self,
        entities: list[dict[str, Any]],
        evidence: dict[tuple[str, str], list[str]] | None = None,
        min_confidence: float | None = None,
        max_relationships: int | None = None,
    ) -> list[CrossDomainRelationship]:
        """Detect cross-domain relationships between entities.

        Args:
            entities: List of entities to analyze.
            evidence: Optional evidence mapping (source_id, target_id) -> evidence_list.
            min_confidence: Minimum confidence threshold for filtering.
            max_relationships: Maximum number of relationships to return.

        Returns:
            List of detected cross-domain relationships.
        """
        threshold = (
            self._config.confidence_threshold
            if min_confidence is None
            else min_confidence
        )
        max_rels = (
            self._config.max_relationships_per_entity
            if max_relationships is None
            else max_relationships
        )
        evidence_map = evidence or {}

        relationships: list[CrossDomainRelationship] = []

        for i, entity1 in enumerate(entities):
            entity1_id = entity1.get("id") or str(i)

            for j, entity2 in enumerate(entities):
                if i >= j:
                    continue

                entity2_id = entity2.get("id") or str(j)

                if not self.is_cross_domain(entity1, entity2):
                    continue

                source_domain = DomainType(entity1.get("domain", DomainType.GENERAL))
                target_domain = DomainType(entity2.get("domain", DomainType.GENERAL))

                valid_types = self.get_valid_relationship_types(
                    source_domain, target_domain
                )

                relationship_type = (
                    self._infer_relationship_type(entity1, entity2)
                    or RelationshipType.RELATED_TO
                )

                if relationship_type not in valid_types:
                    relationship_type = (
                        valid_types[0] if valid_types else RelationshipType.RELATED_TO
                    )

                relationship_evidence = evidence_map.get((entity1_id, entity2_id), [])

                confidence = self._calculate_confidence(
                    entity1, entity2, relationship_type, relationship_evidence
                )

                if confidence < threshold:
                    continue

                is_bidirectional = False
                reverse_types = self.get_valid_relationship_types(
                    target_domain, source_domain
                )
                if self._infer_relationship_type(entity2, entity1) in reverse_types:
                    if self._config.enable_bidirectional_inference:
                        is_bidirectional = True

                relationship = CrossDomainRelationship(
                    source_entity=entity1,
                    target_entity=entity2,
                    relationship_type=relationship_type,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    confidence=confidence,
                    evidence=relationship_evidence,
                    bidirectional=is_bidirectional,
                )

                relationships.append(relationship)

                if len(relationships) >= max_rels:
                    return sorted(
                        relationships, key=lambda r: r.confidence, reverse=True
                    )

        return sorted(relationships, key=lambda r: r.confidence, reverse=True)

    def detect_entity_relationships(
        self,
        source_entity: dict[str, Any],
        target_entities: list[dict[str, Any]],
        evidence_map: dict[tuple[str, str], list[str]] | None = None,
        min_confidence: float | None = None,
    ) -> list[CrossDomainRelationship]:
        """Detect relationships from a source entity to multiple target entities.

        Args:
            source_entity: Source entity to find relationships for.
            target_entities: List of potential target entities.
            evidence_map: Optional evidence mapping.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of detected cross-domain relationships.
        """
        threshold = (
            self._config.confidence_threshold
            if min_confidence is None
            else min_confidence
        )
        evidence = evidence_map or {}

        relationships: list[CrossDomainRelationship] = []

        for target in target_entities:
            if not self.is_cross_domain(source_entity, target):
                continue

            source_domain = DomainType(source_entity.get("domain", DomainType.GENERAL))
            target_domain = DomainType(target.get("domain", DomainType.GENERAL))

            relationship_type = (
                self._infer_relationship_type(source_entity, target)
                or RelationshipType.RELATED_TO
            )
            evidence_list = evidence.get(
                (source_entity.get("id", ""), target.get("id", "")), []
            )

            confidence = self._calculate_confidence(
                source_entity, target, relationship_type, evidence_list
            )

            if confidence < threshold:
                continue

            relationship = CrossDomainRelationship(
                source_entity=source_entity,
                target_entity=target,
                relationship_type=relationship_type,
                source_domain=source_domain,
                target_domain=target_domain,
                confidence=confidence,
                evidence=evidence_list,
            )

            relationships.append(relationship)

        return sorted(relationships, key=lambda r: r.confidence, reverse=True)

    def filter_relationships(
        self,
        relationships: list[CrossDomainRelationship],
        min_confidence: float | None = None,
        relationship_types: list[RelationshipType] | None = None,
        source_domains: list[DomainType] | None = None,
        target_domains: list[DomainType] | None = None,
    ) -> list[CrossDomainRelationship]:
        """Filter relationships by various criteria.

        Args:
            relationships: List of relationships to filter.
            min_confidence: Minimum confidence threshold.
            relationship_types: Allowed relationship types.
            source_domains: Allowed source domains.
            target_domains: Allowed target domains.

        Returns:
            Filtered list of relationships.
        """
        threshold = (
            self._config.confidence_threshold
            if min_confidence is None
            else min_confidence
        )

        filtered = [r for r in relationships if r.confidence >= threshold]

        if relationship_types:
            filtered = [
                r for r in filtered if r.relationship_type in relationship_types
            ]

        if source_domains:
            filtered = [r for r in filtered if r.source_domain in source_domains]

        if target_domains:
            filtered = [r for r in filtered if r.target_domain in target_domains]

        return filtered

    async def detect_cross_domain_relationships(
        self,
        entities: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
        document_domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """Detect cross-domain relationships for ingestion pipeline integration.

        Args:
            entities: List of entity dictionaries with id, name, type, domain.
            edges: Optional list of existing edge dictionaries.
            document_domain: Optional document domain context.

        Returns:
            List of cross-domain edge dictionaries ready for persistence.
        """
        if not entities or len(entities) < 2:
            return []

        relationships = self.detect_relationships(entities)

        cross_domain_edges = []
        for rel in relationships:
            edge_dict = {
                "id": str(uuid4()),
                "source_id": rel.source_entity.get("id"),
                "target_id": rel.target_entity.get("id"),
                "edge_type": rel.relationship_type.value,
                "properties": {
                    "confidence": rel.confidence,
                    "evidence": rel.evidence,
                    "bidirectional": rel.bidirectional,
                    "source_domain": rel.source_domain.value,
                    "target_domain": rel.target_domain.value,
                },
                "confidence": rel.confidence,
                "provenance": "cross_domain_detection",
                "domain": document_domain or "general",
            }
            cross_domain_edges.append(edge_dict)

        return cross_domain_edges

        return cross_domain_edges

    def get_statistics(
        self,
        relationships: list[CrossDomainRelationship],
    ) -> dict[str, Any]:
        """Get statistics about detected relationships.

        Args:
            relationships: List of relationships to analyze.

        Returns:
            Dictionary with relationship statistics.
        """
        if not relationships:
            return {
                "total_relationships": 0,
                "average_confidence": 0.0,
                "relationship_type_counts": {},
                "domain_pair_counts": {},
                "bidirectional_count": 0,
            }

        type_counts: dict[str, int] = {}
        domain_pair_counts: dict[str, int] = {}

        for rel in relationships:
            type_key = rel.relationship_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1

            domain_pair = f"{rel.source_domain.value}->{rel.target_domain.value}"
            domain_pair_counts[domain_pair] = domain_pair_counts.get(domain_pair, 0) + 1

        return {
            "total_relationships": len(relationships),
            "average_confidence": sum(r.confidence for r in relationships)
            / len(relationships),
            "relationship_type_counts": type_counts,
            "domain_pair_counts": domain_pair_counts,
            "bidirectional_count": sum(1 for r in relationships if r.bidirectional),
            "min_confidence": min(r.confidence for r in relationships),
            "max_confidence": max(r.confidence for r in relationships),
        }

"""Adaptive Type Discovery for KBV2.

This module provides functionality to discover new entity types from
extraction results and promote high-confidence types to the schema.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from collections import Counter
import time


class DiscoveredType(BaseModel):
    """A discovered entity type.

    Attributes:
        name: The name of the discovered type.
        frequency: Number of entities of this type found.
        confidence: Confidence score for the discovered type.
        examples: Example entity names of this type.
        properties: Common properties found for this type.
        parent_type: Parent type in the type hierarchy if applicable.
        is_promoted: Whether the type is ready for schema promotion.
    """

    name: str
    frequency: int = 0
    confidence: float = 0.0
    examples: List[str] = Field(default_factory=list)
    properties: List[str] = Field(default_factory=list)
    parent_type: Optional[str] = None
    is_promoted: bool = False


class TypeDiscoveryResult(BaseModel):
    """Result from type discovery.

    Attributes:
        discovered_types: List of all discovered types.
        promoted_types: List of types promoted to schema.
        type_hierarchy: Dictionary mapping parent types to child types.
        processing_time_ms: Time taken for discovery in milliseconds.
    """

    discovered_types: List[DiscoveredType] = Field(default_factory=list)
    promoted_types: List[DiscoveredType] = Field(default_factory=list)
    type_hierarchy: Dict[str, List[str]] = Field(default_factory=dict)
    processing_time_ms: float = 0.0


class TypeDiscoveryConfig(BaseModel):
    """Configuration for type discovery.

    Attributes:
        min_frequency: Minimum frequency to consider a type for discovery.
        promotion_threshold: Confidence threshold for schema promotion.
        max_new_types: Maximum number of new types to promote.
        enable_hierarchy: Whether to build type hierarchy.
    """

    min_frequency: int = 2
    promotion_threshold: float = 0.9
    max_new_types: int = 20
    enable_hierarchy: bool = True


class TypeHierarchy(BaseModel):
    """Type hierarchy for organization.

    Attributes:
        root_types: List of root types in the hierarchy.
        parent_child_map: Dictionary mapping parents to their children.
        type_depth: Dictionary mapping types to their depth in hierarchy.
    """

    root_types: List[str] = Field(default_factory=list)
    parent_child_map: Dict[str, List[str]] = Field(default_factory=dict)
    type_depth: Dict[str, int] = Field(default_factory=dict)


class TypeDiscovery:
    """Adaptive type discovery from entity extractions.

    This class analyzes entity extraction results to discover new types,
    calculate confidence scores, and build type hierarchies.
    """

    def __init__(self, config: Optional[TypeDiscoveryConfig] = None):
        """Initialize type discovery with optional configuration.

        Args:
            config: Optional configuration for discovery behavior.
        """
        self.config = config or TypeDiscoveryConfig()
        self.known_types: Set[str] = set()

    def discover(self, entities: List[Dict[str, Any]]) -> TypeDiscoveryResult:
        """Discover new entity types from extraction results.

        Args:
            entities: List of extracted entity dictionaries.

        Returns:
            TypeDiscoveryResult containing discovered and promoted types.
        """
        start = time.time()

        if not entities:
            return TypeDiscoveryResult(processing_time_ms=0.0)

        type_counts = Counter(e.get("entity_type", "Unknown") for e in entities)

        discovered: List[DiscoveredType] = []
        for type_name, count in type_counts.items():
            if count >= self.config.min_frequency:
                examples = self._extract_examples(entities, type_name)
                properties = self._extract_properties(entities, type_name)
                confidence = self._calculate_confidence(count, examples, properties)

                is_new = type_name not in self.known_types
                is_promoted = is_new and confidence >= self.config.promotion_threshold

                discovered_type = DiscoveredType(
                    name=type_name,
                    frequency=count,
                    confidence=confidence,
                    examples=examples[:10],
                    properties=properties[:10],
                    is_promoted=is_promoted,
                )
                discovered.append(discovered_type)

        discovered.sort(key=lambda x: -x.frequency)

        hierarchy: Dict[str, List[str]] = {}
        if self.config.enable_hierarchy:
            hierarchy = self._build_hierarchy(discovered)

        promoted = [t for t in discovered if t.is_promoted][: self.config.max_new_types]

        return TypeDiscoveryResult(
            discovered_types=discovered,
            promoted_types=promoted,
            type_hierarchy=hierarchy,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _extract_examples(
        self, entities: List[Dict[str, Any]], type_name: str
    ) -> List[str]:
        """Extract example entity names for a type.

        Args:
            entities: List of entity dictionaries.
            type_name: The type to extract examples for.

        Returns:
            List of example entity names.
        """
        return [
            e.get("name", "")
            for e in entities
            if e.get("entity_type") == type_name and e.get("name")
        ][:20]

    def _extract_properties(
        self, entities: List[Dict[str, Any]], type_name: str
    ) -> List[str]:
        """Extract common properties for a type.

        Args:
            entities: List of entity dictionaries.
            type_name: The type to extract properties for.

        Returns:
            List of unique property names.
        """
        all_props: List[str] = []
        for e in entities:
            if e.get("entity_type") == type_name:
                props = e.get("properties", {})
                if isinstance(props, dict):
                    all_props.extend(props.keys())
        return list(set(all_props))

    def _calculate_confidence(
        self, frequency: int, examples: List[str], properties: List[str]
    ) -> float:
        """Calculate confidence score for discovered type.

        Args:
            frequency: Number of entities of this type.
            examples: List of example entity names.
            properties: List of common properties.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        freq_score = min(frequency / 10, 1.0)

        if not examples:
            example_score = 0.3
        else:
            example_score = 0.7 + (min(len(examples) / 10, 1.0) * 0.3)

        prop_score = min(len(properties) / 5, 1.0) if properties else 0.2

        return freq_score * 0.4 + example_score * 0.4 + prop_score * 0.2

    def _build_hierarchy(self, types: List[DiscoveredType]) -> Dict[str, List[str]]:
        """Build type hierarchy based on naming patterns.

        Args:
            types: List of discovered types.

        Returns:
            Dictionary mapping parent types to child types.
        """
        hierarchy: Dict[str, List[str]] = {}

        for type_obj in types:
            parent = self._find_parent_type(type_obj.name, types)
            if parent:
                if parent.name not in hierarchy:
                    hierarchy[parent.name] = []
                if type_obj.name not in hierarchy[parent.name]:
                    hierarchy[parent.name].append(type_obj.name)
                type_obj.parent_type = parent.name

        return hierarchy

    def _find_parent_type(
        self, type_name: str, all_types: List[DiscoveredType]
    ) -> Optional[DiscoveredType]:
        """Find parent type for a given type.

        Args:
            type_name: The type name to find parent for.
            all_types: List of all discovered types.

        Returns:
            Parent DiscoveredType or None if not found.
        """
        type_lower = type_name.lower()

        for candidate in all_types:
            if candidate.name == type_name:
                continue

            candidate_lower = candidate.name.lower()

            if type_lower.startswith(f"{candidate_lower}s"):
                return candidate
            if type_lower.startswith(f"{candidate_lower}_"):
                return candidate
            if f"{candidate_lower} " in type_lower:
                return candidate

        return None

    def set_known_types(self, types: Set[str]) -> None:
        """Set known types from existing schema.

        Args:
            types: Set of known type names.
        """
        self.known_types = types

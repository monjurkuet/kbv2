"""Validation Layer for KBV2.

This module provides functionality to validate entities and edges
against a schema definition.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of schema validation.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of validation errors.
        warnings: List of validation warnings.
        validated_entities: Number of entities validated.
        validated_edges: Number of edges validated.
    """

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_entities: int = 0
    validated_edges: int = 0


class ValidationLayer:
    """Validate entities and edges against schema.

    This class provides validation functionality to ensure that
    entities and edges conform to a defined schema.
    """

    def __init__(self, schema: Dict[str, Any]):
        """Initialize validation layer with a schema.

        Args:
            schema: The schema dictionary to validate against.
        """
        self.schema = schema
        self.valid_types: set = set(schema.get("types", {}).keys())
        self.valid_edge_types: set = {
            rel.get("type") for rel in schema.get("relationships", [])
        }

    def validate_entities(self, entities: List[Dict[str, Any]]) -> ValidationResult:
        """Validate entities against schema.

        Args:
            entities: List of entity dictionaries to validate.

        Returns:
            ValidationResult containing validation outcome.
        """
        errors: List[str] = []
        validated = 0

        for entity in entities:
            entity_type = entity.get("entity_type")

            if entity_type is None:
                errors.append("Entity missing entity_type")
            elif entity_type not in self.valid_types:
                errors.append(f"Unknown entity type: {entity_type}")

            if entity_type and entity_type in self.valid_types:
                type_def = self.schema.get("types", {}).get(entity_type, {})
                required_props = type_def.get("required", [])

                props = entity.get("properties", {})
                if not isinstance(props, dict):
                    props = {}

                for prop in required_props:
                    if prop not in props:
                        errors.append(
                            f"Missing required property '{prop}' for {entity_type}"
                        )

            validated += 1

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, validated_entities=validated
        )

    def validate_edges(self, edges: List[Dict[str, Any]]) -> ValidationResult:
        """Validate edges against schema.

        Args:
            edges: List of edge dictionaries to validate.

        Returns:
            ValidationResult containing validation outcome.
        """
        errors: List[str] = []
        validated = 0

        for edge in edges:
            edge_type = edge.get("edge_type", "RELATED_TO")

            if edge_type not in self.valid_edge_types:
                errors.append(f"Unknown edge type: {edge_type}")

            if not edge.get("source_id"):
                errors.append("Edge missing source_id")
            if not edge.get("target_id"):
                errors.append("Edge missing target_id")

            validated += 1

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, validated_edges=validated
        )

    def validate_all(
        self, entities: List[Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate both entities and edges.

        Args:
            entities: List of entity dictionaries.
            edges: List of edge dictionaries.

        Returns:
            Combined ValidationResult for both entities and edges.
        """
        entity_result = self.validate_entities(entities)
        edge_result = self.validate_edges(edges)

        all_errors = entity_result.errors + edge_result.errors

        return ValidationResult(
            is_valid=entity_result.is_valid and edge_result.is_valid,
            errors=all_errors,
            validated_entities=entity_result.validated_entities,
            validated_edges=edge_result.validated_edges,
        )

    def add_type(self, type_name: str, type_def: Dict[str, Any]) -> None:
        """Add a new type to the valid types set.

        Args:
            type_name: Name of the type to add.
            type_def: Type definition dictionary.
        """
        self.valid_types.add(type_name)
        if "types" not in self.schema:
            self.schema["types"] = {}
        self.schema["types"][type_name] = type_def

    def add_edge_type(self, edge_type: str) -> None:
        """Add a new edge type to the valid edge types set.

        Args:
            edge_type: Name of the edge type to add.
        """
        self.valid_edge_types.add(edge_type)
        rel_exists = any(
            r.get("type") == edge_type for r in self.schema.get("relationships", [])
        )
        if not rel_exists:
            if "relationships" not in self.schema:
                self.schema["relationships"] = []
            self.schema["relationships"].append({"type": edge_type})

    def is_type_valid(self, type_name: str) -> bool:
        """Check if a type is valid.

        Args:
            type_name: The type name to check.

        Returns:
            True if the type is valid, False otherwise.
        """
        return type_name in self.valid_types

    def is_edge_type_valid(self, edge_type: str) -> bool:
        """Check if an edge type is valid.

        Args:
            edge_type: The edge type to check.

        Returns:
            True if the edge type is valid, False otherwise.
        """
        return edge_type in self.valid_edge_types

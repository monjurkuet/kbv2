"""Schema Induction for KBV2.

This module provides functionality to induce schemas from discovered types
and merge them with existing schemas.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class InducedSchema(BaseModel):
    """Schema induced from discovered types.

    Attributes:
        types: Dictionary mapping type names to type definitions.
        relationships: List of relationship type definitions.
        version: Schema version identifier.
    """

    types: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)
    version: str = "1.0"


class SchemaInducer:
    """Induce schema from discovered types and entities.

    This class analyzes discovered types and extraction results to create
    schema definitions that can be merged with existing schemas.
    """

    def induce(
        self,
        type_discovery_result,
        entities: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> InducedSchema:
        """Induce schema from entities and relationships.

        Args:
            type_discovery_result: Result from TypeDiscovery.discover().
            entities: List of extracted entity dictionaries.
            edges: List of extracted edge dictionaries.

        Returns:
            InducedSchema containing type and relationship definitions.
        """
        schema = InducedSchema()

        for type_obj in type_discovery_result.discovered_types:
            self_type = type_obj.name
            examples = type_obj.examples
            properties = type_obj.properties
            confidence = type_obj.confidence
            is_promoted = type_obj.is_promoted

            type_def: Dict[str, Any] = {
                "description": f"Auto-discovered type: {self_type}",
                "examples": examples,
                "properties": properties,
                "confidence": confidence,
                "is_promoted": is_promoted,
            }

            inferred_required = self._infer_required_properties(entities, self_type)
            if inferred_required:
                type_def["required"] = inferred_required

            schema.types[self_type] = type_def

        edge_types: set = set()
        for edge in edges:
            edge_type = edge.get("edge_type", "RELATED_TO")
            edge_types.add(edge_type)

            source_type = edge.get("source_type")
            target_type = edge.get("target_type")
            if source_type or target_type:
                existing = [
                    r for r in schema.relationships if r.get("type") == edge_type
                ]
                if not existing:
                    schema.relationships.append(
                        {
                            "type": edge_type,
                            "source_types": [source_type] if source_type else [],
                            "target_types": [target_type] if target_type else [],
                        }
                    )
                else:
                    if source_type and source_type not in existing[0].get(
                        "source_types", []
                    ):
                        existing[0]["source_types"].append(source_type)
                    if target_type and target_type not in existing[0].get(
                        "target_types", []
                    ):
                        existing[0]["target_types"].append(target_type)

        for edge_type in edge_types:
            if not any(r.get("type") == edge_type for r in schema.relationships):
                schema.relationships.append({"type": edge_type})

        return schema

    def _infer_required_properties(
        self, entities: List[Dict[str, Any]], type_name: str
    ) -> List[str]:
        """Infer required properties based on presence across entities.

        Args:
            entities: List of entity dictionaries.
            type_name: The type to analyze.

        Returns:
            List of property names inferred as required.
        """
        prop_counts: Dict[str, int] = {}
        prop_total = 0

        for entity in entities:
            if entity.get("entity_type") != type_name:
                continue

            props = entity.get("properties", {})
            if not isinstance(props, dict):
                continue

            prop_total += 1
            for prop in props:
                prop_counts[prop] = prop_counts.get(prop, 0) + 1

        if prop_total == 0:
            return []

        required: List[str] = []
        threshold = 0.8

        for prop, count in prop_counts.items():
            if prop_total > 0 and count / prop_total >= threshold:
                required.append(prop)

        return required

    def merge_with_schema(
        self, induced: InducedSchema, existing_schema: Dict[str, Any]
    ) -> InducedSchema:
        """Merge induced schema with existing schema.

        Args:
            induced: The induced schema to merge.
            existing_schema: The existing schema dictionary.

        Returns:
            Merged InducedSchema.
        """
        existing_types = existing_schema.get("types", {})
        existing_rels = existing_schema.get("relationships", [])

        merged_types: Dict[str, Dict[str, Any]] = dict(existing_types)

        for type_name, type_def in induced.types.items():
            if type_name not in merged_types:
                merged_types[type_name] = type_def
            else:
                existing_type = merged_types[type_name]
                existing_type["description"] = type_def.get(
                    "description", existing_type.get("description", "")
                )
                existing_examples = set(existing_type.get("examples", []))
                existing_examples.update(type_def.get("examples", []))
                existing_type["examples"] = list(existing_examples)

                existing_props = set(existing_type.get("properties", []))
                existing_props.update(type_def.get("properties", []))
                existing_type["properties"] = list(existing_props)

        existing_rel_types = {r.get("type") for r in existing_rels}
        merged_rels: List[Dict[str, str]] = list(existing_rels)

        for rel in induced.relationships:
            rel_type = rel.get("type")
            if rel_type and rel_type not in existing_rel_types:
                merged_rels.append(rel)

        version = existing_schema.get("version", induced.version)

        return InducedSchema(
            types=merged_types, relationships=merged_rels, version=version
        )

    def generate_schema_template(self, type_name: str) -> Dict[str, Any]:
        """Generate a template for a new type definition.

        Args:
            type_name: The name of the type.

        Returns:
            Template dictionary for the type.
        """
        return {
            "type": type_name,
            "description": f"Description for {type_name}",
            "properties": {},
            "required": [],
        }

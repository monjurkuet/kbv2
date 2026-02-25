"""Common utility functions for KBV2."""

from typing import Any, Sequence

from knowledge_base.storage.portable.kuzu_store import Entity


def entity_to_dict(entity: Entity) -> dict[str, Any]:
    """Convert an Entity to a dictionary representation.

    Args:
        entity: The entity to convert.

    Returns:
        Dictionary with entity fields.
    """
    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "domain": entity.domain,
    }


def entities_to_dict_list(
    entities: Sequence[Entity], domain_filter: str | None = None
) -> list[dict[str, Any]]:
    """Convert a list of entities to dictionary format.

    Args:
        entities: Sequence of entities to convert.
        domain_filter: Optional domain to filter by.

    Returns:
        List of entity dictionaries.
    """
    result = []
    for e in entities:
        if domain_filter is None or e.domain == domain_filter:
            result.append(entity_to_dict(e))
    return result

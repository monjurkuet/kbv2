"""KBV2 Types Package.

This package provides adaptive type discovery, schema induction,
and validation layer for the knowledge base.
"""

from src.knowledge_base.types.type_discovery import (
    TypeDiscovery,
    DiscoveredType,
    TypeDiscoveryResult,
    TypeDiscoveryConfig,
)

from src.knowledge_base.types.schema_inducer import (
    SchemaInducer,
    InducedSchema,
)

from src.knowledge_base.types.validation_layer import (
    ValidationLayer,
    ValidationResult,
)

__all__ = [
    "TypeDiscovery",
    "DiscoveredType",
    "TypeDiscoveryResult",
    "TypeDiscoveryConfig",
    "SchemaInducer",
    "InducedSchema",
    "ValidationLayer",
    "ValidationResult",
]

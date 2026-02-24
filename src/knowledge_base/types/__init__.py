"""KBV2 Types Package.

This package provides adaptive type discovery, schema induction,
and validation layer for the knowledge base.
"""

from .type_discovery import (
    TypeDiscovery,
    DiscoveredType,
    TypeDiscoveryResult,
    TypeDiscoveryConfig,
)

from .schema_inducer import (
    SchemaInducer,
    InducedSchema,
)

from .validation_layer import (
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

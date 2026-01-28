"""Domain detection module for KBV2."""

from knowledge_base.domain.domain_models import (
    Domain,
    DomainPrediction,
    DomainDetectionResult,
    DomainConfig,
)
from knowledge_base.domain.detection import DomainDetector
from knowledge_base.domain.ontology_snippets import DOMAIN_ONTOLOGIES

__all__ = [
    "Domain",
    "DomainPrediction",
    "DomainDetectionResult",
    "DomainConfig",
    "DomainDetector",
    "DOMAIN_ONTOLOGIES",
]

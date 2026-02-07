"""Self-improvement module for KBv2.

This module provides functionality for automated self-improvement including:
- Experience Bank: Store and retrieve high-quality extraction examples
- Prompt Evolution: Automated prompt optimization through mutation and selection
- Ontology Validator: Validate extractions against domain rules
- Domain Detection Feedback: Learn from domain classification accuracy
"""

from .experience_bank import (
    ExperienceBank,
    ExperienceBankConfig,
    ExperienceBankMiddleware,
    ExtractionExample,
    ExtractionExperienceRecord,
)
from .prompt_evolution import (
    PromptEvolutionEngine,
    PromptEvolutionConfig,
    PromptVariant,
    CryptoPromptTemplates,
)
from .ontology_validator import (
    OntologyValidator,
    ValidationReport,
    OntologyRule,
    OntologyViolation,
    CryptoOntologyRules,
)
from .domain_detection_feedback import (
    DomainDetectionSelfImprovement,
    DomainDetectionFeedbackRecord,
    DomainAccuracyStats,
    DomainImprovementSuggestion,
)

__all__ = [
    # Experience Bank
    "ExperienceBank",
    "ExperienceBankConfig",
    "ExperienceBankMiddleware",
    "ExtractionExample",
    "ExtractionExperienceRecord",
    # Prompt Evolution
    "PromptEvolutionEngine",
    "PromptEvolutionConfig",
    "PromptVariant",
    "CryptoPromptTemplates",
    # Ontology Validator
    "OntologyValidator",
    "ValidationReport",
    "OntologyRule",
    "OntologyViolation",
    "CryptoOntologyRules",
    # Domain Detection Feedback
    "DomainDetectionSelfImprovement",
    "DomainDetectionFeedbackRecord",
    "DomainAccuracyStats",
    "DomainImprovementSuggestion",
]

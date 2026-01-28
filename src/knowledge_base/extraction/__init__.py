"""Knowledge base extraction modules."""

from knowledge_base.extraction.template_registry import (
    ExtractionGoal,
    TemplateRegistry,
    get_default_goals,
)
from knowledge_base.extraction.guided_extractor import (
    GuidedExtractor,
    ExtractionPrompt,
    ExtractionPrompts,
)

__all__ = [
    "ExtractionGoal",
    "TemplateRegistry",
    "get_default_goals",
    "GuidedExtractor",
    "ExtractionPrompt",
    "ExtractionPrompts",
]

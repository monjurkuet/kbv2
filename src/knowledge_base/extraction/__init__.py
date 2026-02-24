"""Structured extraction for Knowledge Base System.

This module provides:
- Pydantic schemas for trading content extraction
- Entity extraction with validation
- Price target and trading setup extraction
- Pipeline for batch extraction
"""

from knowledge_base.extraction.schemas import (
    PriceTarget,
    TradingSetup,
    MarketAnalysis,
    EducationalConcept,
    VideoAnalysis,
    DocumentAnalysis,
)
from knowledge_base.extraction.pipeline import ExtractionPipeline

__all__ = [
    "PriceTarget",
    "TradingSetup",
    "MarketAnalysis",
    "EducationalConcept",
    "VideoAnalysis",
    "DocumentAnalysis",
    "ExtractionPipeline",
]

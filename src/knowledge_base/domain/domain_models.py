"""Domain detection models for KBV2."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Domain(str, Enum):
    """Supported domains."""

    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL = "FINANCIAL"
    MEDICAL = "MEDICAL"
    LEGAL = "LEGAL"
    SCIENTIFIC = "SCIENTIFIC"
    GENERAL = "GENERAL"


class DomainPrediction(BaseModel):
    """Single domain prediction with confidence and evidence."""

    domain: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: List[str] = Field(default_factory=list)
    keyword_matches: List[str] = Field(default_factory=list)
    reasoning: str = ""


class DomainDetectionResult(BaseModel):
    """Complete domain detection result."""

    primary_domain: str
    all_domains: List[DomainPrediction] = Field(default_factory=list)
    is_multi_domain: bool = False
    detection_method: str = "hybrid"
    processing_time_ms: float = 0.0
    confidence_threshold: float = 0.6


class DomainConfig(BaseModel):
    """Configuration for domain detection."""

    min_confidence: float = 0.6
    max_predictions: int = 3
    enable_keyword_screening: bool = True
    enable_llm_analysis: bool = True
    keyword_threshold: float = 0.3

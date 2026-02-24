"""Pydantic schemas for structured extraction from trading content.

These schemas define the structure for extracting:
- Price targets
- Trading setups
- Market analysis
- Educational concepts
- Entity relationships

All schemas are designed for use with Instructor/LLM-based extraction.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ==================== Enums ====================

class MarketBias(str, Enum):
    """Market sentiment/bias."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class SetupType(str, Enum):
    """Type of trading setup."""

    LONG = "long"
    SHORT = "short"
    RANGE = "range"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class Timeframe(str, Enum):
    """Trading timeframe."""

    SCALP = "scalp"  # Minutes
    INTRADAY = "intraday"  # Hours
    SWING = "swing"  # Days to weeks
    POSITION = "position"  # Weeks to months
    INVESTMENT = "investment"  # Months to years


class Confidence(str, Enum):
    """Confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


class EntityType(str, Enum):
    """Types of entities in trading content."""

    CRYPTOCURRENCY = "cryptocurrency"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    INDICATOR = "indicator"
    PATTERN = "pattern"
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    EXCHANGE = "exchange"
    TIMEFRAME = "timeframe"


# ==================== Price Target Schema ====================

class PriceTarget(BaseModel):
    """A price target extracted from trading content.

    This represents a specific price prediction or target mentioned
    in trading analysis or commentary.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str = Field(
        ...,
        description="Asset symbol (e.g., BTC, ETH, SOL)",
        min_length=1,
        max_length=20,
    )
    target_price: float = Field(
        ...,
        description="Target price value",
        gt=0,
    )
    timeframe: str = Field(
        ...,
        description="Expected timeframe for the target (e.g., '1-3 months', 'end of 2024')",
    )
    confidence: Optional[str] = Field(
        None,
        description="Confidence level if mentioned (high, medium, low)",
    )
    rationale: str = Field(
        ...,
        description="Reasoning or analysis supporting the target",
        min_length=10,
    )
    direction: str = Field(
        "up",
        description="Direction of target: 'up' for bullish, 'down' for bearish",
    )
    source_text: Optional[str] = Field(
        None,
        description="Original text mentioning the target",
    )
    timestamp: Optional[str] = Field(
        None,
        description="Timestamp in the video/content where mentioned",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        return v.upper().strip()

    @field_validator("target_price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price is positive."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return round(v, 2)


# ==================== Trading Setup Schema ====================

class TradingSetup(BaseModel):
    """A trading setup extracted from content.

    This represents a specific trade setup with entry, stop loss,
    and take profit levels.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str = Field(
        ...,
        description="Asset symbol",
        min_length=1,
    )
    setup_type: str = Field(
        ...,
        description="Type of setup: long, short, range, breakout, etc.",
    )
    entry_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions for entry",
    )
    entry_price: Optional[float] = Field(
        None,
        description="Suggested entry price or range",
        ge=0,
    )
    entry_range_low: Optional[float] = Field(
        None,
        description="Lower bound of entry range",
        ge=0,
    )
    entry_range_high: Optional[float] = Field(
        None,
        description="Upper bound of entry range",
        ge=0,
    )
    stop_loss: Optional[float] = Field(
        None,
        description="Stop loss price level",
        ge=0,
    )
    take_profit_levels: list[float] = Field(
        default_factory=list,
        description="Take profit price levels",
    )
    risk_reward_ratio: Optional[float] = Field(
        None,
        description="Risk/reward ratio",
        ge=0,
    )
    timeframe: Optional[str] = Field(
        None,
        description="Holding timeframe",
    )
    confidence: Optional[str] = Field(
        None,
        description="Confidence level",
    )
    invalidation_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that invalidate the setup",
    )
    source_text: Optional[str] = Field(
        None,
        description="Original text describing the setup",
    )

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        """Normalize symbol to uppercase."""
        return v.upper().strip()

    @field_validator("setup_type")
    @classmethod
    def validate_setup_type(cls, v: str) -> str:
        """Validate setup type."""
        valid_types = [e.value for e in SetupType]
        v_lower = v.lower()
        if v_lower not in valid_types:
            # Try to map common variations
            type_map = {
                "buy": "long",
                "sell": "short",
                "long position": "long",
                "short position": "short",
            }
            v_lower = type_map.get(v_lower, v_lower)
        return v_lower


# ==================== Market Analysis Schema ====================

class MarketAnalysis(BaseModel):
    """Market analysis extracted from content.

    This captures overall market sentiment, key observations,
    and analysis themes from trading content.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    market_bias: str = Field(
        ...,
        description="Overall market sentiment: bullish, bearish, neutral",
    )
    assets_analyzed: list[str] = Field(
        default_factory=list,
        description="Assets mentioned in the analysis",
    )
    key_observations: list[str] = Field(
        default_factory=list,
        description="Key market observations",
    )
    support_levels: list[float] = Field(
        default_factory=list,
        description="Key support levels mentioned",
    )
    resistance_levels: list[float] = Field(
        default_factory=list,
        description="Key resistance levels mentioned",
    )
    indicators_used: list[str] = Field(
        default_factory=list,
        description="Technical indicators referenced",
    )
    timeframes_analyzed: list[str] = Field(
        default_factory=list,
        description="Timeframes discussed",
    )
    key_themes: list[str] = Field(
        default_factory=list,
        description="Main themes or narratives",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Risk factors or concerns mentioned",
    )
    source_text: Optional[str] = Field(
        None,
        description="Relevant source text",
    )

    @field_validator("market_bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        """Validate market bias."""
        valid = [e.value for e in MarketBias]
        v_lower = v.lower()
        if v_lower not in valid:
            return "unclear"
        return v_lower

    @field_validator("assets_analyzed")
    @classmethod
    def normalize_assets(cls, v: list[str]) -> list[str]:
        """Normalize asset symbols."""
        return [a.upper().strip() for a in v if a.strip()]


# ==================== Educational Concept Schema ====================

class EducationalConcept(BaseModel):
    """An educational concept from trading content.

    This captures trading concepts, strategies, or educational
    content from training materials.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(
        ...,
        description="Name of the concept",
        min_length=2,
    )
    category: str = Field(
        ...,
        description="Category: strategy, indicator, pattern, risk_management, etc.",
    )
    description: str = Field(
        ...,
        description="Explanation of the concept",
        min_length=10,
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Key learning points",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Examples or case studies",
    )
    related_concepts: list[str] = Field(
        default_factory=list,
        description="Related concepts to explore",
    )
    practical_applications: list[str] = Field(
        default_factory=list,
        description="How to apply this concept",
    )
    common_mistakes: list[str] = Field(
        default_factory=list,
        description="Common mistakes to avoid",
    )
    source_text: Optional[str] = Field(
        None,
        description="Original source text",
    )

    @field_validator("category")
    @classmethod
    def normalize_category(cls, v: str) -> str:
        """Normalize category."""
        valid_categories = [
            "strategy",
            "indicator",
            "pattern",
            "risk_management",
            "market_structure",
            "psychology",
            "fundamental_analysis",
            "on_chain_analysis",
            "general",
        ]
        v_lower = v.lower().replace(" ", "_")
        if v_lower not in valid_categories:
            return "general"
        return v_lower


# ==================== Entity Schema ====================

class ExtractedEntity(BaseModel):
    """An entity extracted from content.

    Entities include assets, indicators, patterns, concepts,
    people, organizations, etc.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(
        ...,
        description="Entity name",
        min_length=1,
    )
    entity_type: str = Field(
        ...,
        description="Type of entity",
    )
    description: Optional[str] = Field(
        None,
        description="Brief description",
    )
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties",
    )
    confidence: float = Field(
        1.0,
        description="Extraction confidence",
        ge=0.0,
        le=1.0,
    )
    source_text: Optional[str] = Field(
        None,
        description="Text where entity was found",
    )
    mention_count: int = Field(
        1,
        description="Number of times mentioned",
    )

    @field_validator("entity_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate entity type."""
        valid = [e.value for e in EntityType]
        v_lower = v.lower()
        if v_lower not in valid:
            return "concept"
        return v_lower


# ==================== Relationship Schema ====================

class ExtractedRelationship(BaseModel):
    """A relationship between entities.

    This captures how entities relate to each other in the content.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity: str = Field(
        ...,
        description="Source entity name",
    )
    target_entity: str = Field(
        ...,
        description="Target entity name",
    )
    relation_type: str = Field(
        ...,
        description="Type of relationship",
    )
    description: Optional[str] = Field(
        None,
        description="Description of the relationship",
    )
    confidence: float = Field(
        1.0,
        description="Extraction confidence",
        ge=0.0,
        le=1.0,
    )
    source_text: Optional[str] = Field(
        None,
        description="Text indicating the relationship",
    )


# ==================== Document Analysis Schema ====================

class VideoAnalysis(BaseModel):
    """Complete analysis of a video transcript.

    This aggregates all extracted information from a video.
    """

    video_id: str = Field(
        ...,
        description="YouTube video ID or identifier",
    )
    title: Optional[str] = Field(
        None,
        description="Video title",
    )
    channel: Optional[str] = Field(
        None,
        description="Channel name",
    )
    published_at: Optional[datetime] = Field(
        None,
        description="Publication date",
    )
    price_targets: list[PriceTarget] = Field(
        default_factory=list,
        description="Extracted price targets",
    )
    trading_setups: list[TradingSetup] = Field(
        default_factory=list,
        description="Extracted trading setups",
    )
    market_analysis: Optional[MarketAnalysis] = Field(
        None,
        description="Market analysis",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="Entity relationships",
    )
    educational_concepts: list[EducationalConcept] = Field(
        default_factory=list,
        description="Educational content",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Main topics discussed",
    )
    summary: Optional[str] = Field(
        None,
        description="Brief summary of the content",
    )
    extraction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction",
    )


class DocumentAnalysis(BaseModel):
    """Complete analysis of a document.

    This aggregates all extracted information from a document.
    """

    document_id: str = Field(
        ...,
        description="Document identifier",
    )
    document_name: str = Field(
        ...,
        description="Document name/title",
    )
    document_type: str = Field(
        "unknown",
        description="Type of document (pdf, markdown, etc.)",
    )
    price_targets: list[PriceTarget] = Field(
        default_factory=list,
        description="Extracted price targets",
    )
    trading_setups: list[TradingSetup] = Field(
        default_factory=list,
        description="Extracted trading setups",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="Entity relationships",
    )
    educational_concepts: list[EducationalConcept] = Field(
        default_factory=list,
        description="Educational content",
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="Main topics covered",
    )
    summary: Optional[str] = Field(
        None,
        description="Document summary",
    )
    extraction_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the extraction",
    )

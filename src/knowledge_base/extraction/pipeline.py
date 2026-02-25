"""Extraction pipeline for structured data extraction from content.

This module provides a pipeline that:
1. Processes documents/chunks through the vision model
2. Extracts structured data using LLM with schema validation
3. Stores extracted entities and relationships
4. Links extractions to source documents
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel

from knowledge_base.extraction.schemas import (
    DocumentAnalysis,
    ExtractedEntity,
    ExtractedRelationship,
    MarketAnalysis,
    PriceTarget,
    TradingSetup,
    EducationalConcept,
    VideoAnalysis,
)
from knowledge_base.ingestion.vision_client import VisionModelClient

logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline."""

    # Model settings
    extraction_model: str = "qwen3-max"
    entity_model: str = "qwen3-max"

    # Batch settings
    chunk_batch_size: int = 5
    max_concurrent_extractions: int = 3

    # Output settings
    min_confidence: float = 0.5
    deduplicate_entities: bool = True


class ExtractionPipeline:
    """Pipeline for extracting structured data from content.

    This pipeline processes content through multiple extraction stages:
    1. Price target extraction
    2. Trading setup extraction
    3. Entity extraction
    4. Relationship extraction
    5. Educational concept extraction

    Example:
        >>> pipeline = ExtractionPipeline(vision_client)
        >>> analysis = await pipeline.extract_from_video(transcript_text, video_id)
        >>> print(analysis.price_targets)
    """

    def __init__(
        self,
        vision_client: VisionModelClient,
        config: Optional[ExtractionConfig] = None,
    ) -> None:
        """Initialize extraction pipeline.

        Args:
            vision_client: Vision model client for LLM calls.
            config: Extraction configuration.
        """
        self._client = vision_client
        self._config = config or ExtractionConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_extractions)

    # ==================== Main Extraction Methods ====================

    async def extract_from_video(
        self,
        transcript: str,
        video_id: str,
        title: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> VideoAnalysis:
        """Extract structured data from a video transcript.

        Args:
            transcript: Video transcript text.
            video_id: YouTube video ID.
            title: Video title.
            channel: Channel name.

        Returns:
            VideoAnalysis with all extracted data.
        """
        async with self._semaphore:
            # Run extractions in parallel
            tasks = [
                self._extract_price_targets(transcript),
                self._extract_trading_setups(transcript),
                self._extract_entities(transcript),
                self._extract_educational_concepts(transcript),
                self._extract_market_analysis(transcript),
                self._extract_summary(transcript),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Unpack results
            price_targets = results[0] if not isinstance(results[0], Exception) else []
            trading_setups = results[1] if not isinstance(results[1], Exception) else []
            entities = results[2] if not isinstance(results[2], Exception) else []
            concepts = results[3] if not isinstance(results[3], Exception) else []
            market_analysis = results[4] if not isinstance(results[4], Exception) else None
            summary = results[5] if not isinstance(results[5], Exception) else None

            # Extract relationships between entities
            relationships = await self._extract_relationships(
                transcript, entities if isinstance(entities, list) else []
            )

            # Deduplicate entities if configured
            if self._config.deduplicate_entities and isinstance(entities, list):
                entities = self._deduplicate_entities(entities)

            # Extract key concepts
            key_concepts = list(
                set(c.name for c in (concepts if isinstance(concepts, list) else []))
            )[:10]

            return VideoAnalysis(
                video_id=video_id,
                title=title,
                channel=channel,
                price_targets=price_targets if isinstance(price_targets, list) else [],
                trading_setups=trading_setups if isinstance(trading_setups, list) else [],
                market_analysis=market_analysis
                if isinstance(market_analysis, MarketAnalysis)
                else None,
                entities=entities if isinstance(entities, list) else [],
                relationships=relationships if isinstance(relationships, list) else [],
                educational_concepts=concepts if isinstance(concepts, list) else [],
                key_concepts=key_concepts,
                summary=summary if isinstance(summary, str) else None,
                extraction_metadata={
                    "extracted_at": datetime.utcnow().isoformat(),
                    "model_used": self._config.extraction_model,
                },
            )

    async def extract_from_document(
        self,
        content: str,
        document_id: str,
        document_name: str,
        document_type: str = "unknown",
    ) -> DocumentAnalysis:
        """Extract structured data from a document.

        Args:
            content: Document content.
            document_id: Document ID.
            document_name: Document name.
            document_type: Type of document.

        Returns:
            DocumentAnalysis with all extracted data.
        """
        async with self._semaphore:
            # Run extractions in parallel
            tasks = [
                self._extract_price_targets(content),
                self._extract_trading_setups(content),
                self._extract_entities(content),
                self._extract_educational_concepts(content),
                self._extract_summary(content),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Unpack results
            price_targets = results[0] if not isinstance(results[0], Exception) else []
            trading_setups = results[1] if not isinstance(results[1], Exception) else []
            entities = results[2] if not isinstance(results[2], Exception) else []
            concepts = results[3] if not isinstance(results[3], Exception) else []
            summary = results[4] if not isinstance(results[4], Exception) else None

            # Extract relationships
            relationships = await self._extract_relationships(
                content, entities if isinstance(entities, list) else []
            )

            # Deduplicate entities
            if self._config.deduplicate_entities and isinstance(entities, list):
                entities = self._deduplicate_entities(entities)

            return DocumentAnalysis(
                document_id=document_id,
                document_name=document_name,
                document_type=document_type,
                price_targets=price_targets if isinstance(price_targets, list) else [],
                trading_setups=trading_setups if isinstance(trading_setups, list) else [],
                entities=entities if isinstance(entities, list) else [],
                relationships=relationships if isinstance(relationships, list) else [],
                educational_concepts=concepts if isinstance(concepts, list) else [],
                key_concepts=list(
                    set(c.name for c in (concepts if isinstance(concepts, list) else []))
                )[:10],
                summary=summary if isinstance(summary, str) else None,
                extraction_metadata={
                    "extracted_at": datetime.utcnow().isoformat(),
                    "model_used": self._config.extraction_model,
                },
            )

    # ==================== Individual Extraction Methods ====================

    async def _extract_price_targets(self, text: str) -> list[PriceTarget]:
        """Extract price targets from text.

        Args:
            text: Text to extract from.

        Returns:
            List of price targets.
        """
        schema = {
            "type": "object",
            "properties": {
                "price_targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "target_price": {"type": "number"},
                            "timeframe": {"type": "string"},
                            "confidence": {"type": "string"},
                            "rationale": {"type": "string"},
                            "direction": {"type": "string"},
                            "source_text": {"type": "string"},
                        },
                        "required": ["symbol", "target_price", "timeframe", "rationale"],
                    },
                },
            },
            "required": ["price_targets"],
        }

        try:
            result = await self._client.extract_structured(
                text,
                schema,
                model=self._config.extraction_model,
            )

            targets = []
            for item in result.data.get("price_targets", []):
                try:
                    targets.append(PriceTarget(**item))
                except Exception as e:
                    logger.warning(f"Invalid price target: {e}")

            return targets
        except Exception as e:
            logger.error(f"Price target extraction failed: {e}")
            return []

    async def _extract_trading_setups(self, text: str) -> list[TradingSetup]:
        """Extract trading setups from text.

        Args:
            text: Text to extract from.

        Returns:
            List of trading setups.
        """
        schema = {
            "type": "object",
            "properties": {
                "setups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "setup_type": {"type": "string"},
                            "entry_conditions": {"type": "array", "items": {"type": "string"}},
                            "entry_price": {"type": "number"},
                            "stop_loss": {"type": "number"},
                            "take_profit_levels": {"type": "array", "items": {"type": "number"}},
                            "risk_reward_ratio": {"type": "number"},
                            "timeframe": {"type": "string"},
                            "confidence": {"type": "string"},
                            "invalidation_conditions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_text": {"type": "string"},
                        },
                        "required": ["symbol", "setup_type"],
                    },
                },
            },
            "required": ["setups"],
        }

        try:
            result = await self._client.extract_structured(
                text,
                schema,
                model=self._config.extraction_model,
            )

            setups = []
            for item in result.data.get("setups", []):
                try:
                    setups.append(TradingSetup(**item))
                except Exception as e:
                    logger.warning(f"Invalid trading setup: {e}")

            return setups
        except Exception as e:
            logger.error(f"Trading setup extraction failed: {e}")
            return []

    async def _extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: Text to extract from.

        Returns:
            List of entities.
        """
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence": {"type": "number"},
                            "source_text": {"type": "string"},
                        },
                        "required": ["name", "entity_type"],
                    },
                },
            },
            "required": ["entities"],
        }

        try:
            result = await self._client.extract_structured(
                text,
                schema,
                model=self._config.entity_model,
            )

            entities = []
            for item in result.data.get("entities", []):
                try:
                    entities.append(ExtractedEntity(**item))
                except Exception as e:
                    logger.warning(f"Invalid entity: {e}")

            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def _extract_relationships(
        self,
        text: str,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships between entities.

        Args:
            text: Text to extract from.
            entities: List of extracted entities.

        Returns:
            List of relationships.
        """
        if len(entities) < 2:
            return []

        entity_names = [e.name for e in entities[:20]]  # Limit to avoid huge prompts

        schema = {
            "type": "object",
            "properties": {
                "relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_entity": {"type": "string"},
                            "target_entity": {"type": "string"},
                            "relation_type": {"type": "string"},
                            "description": {"type": "string"},
                            "confidence": {"type": "number"},
                            "source_text": {"type": "string"},
                        },
                        "required": ["source_entity", "target_entity", "relation_type"],
                    },
                },
            },
            "required": ["relationships"],
        }

        # Add context about entities
        context = f"Known entities: {', '.join(entity_names)}\n\nText to analyze:\n{text[:5000]}"

        try:
            result = await self._client.extract_structured(
                context,
                schema,
                model=self._config.entity_model,
            )

            relationships = []
            for item in result.data.get("relationships", []):
                try:
                    # Validate entity names exist
                    if (
                        item.get("source_entity") in entity_names
                        and item.get("target_entity") in entity_names
                    ):
                        relationships.append(ExtractedRelationship(**item))
                except Exception as e:
                    logger.warning(f"Invalid relationship: {e}")

            return relationships
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    async def _extract_educational_concepts(self, text: str) -> list[EducationalConcept]:
        """Extract educational concepts from text.

        Args:
            text: Text to extract from.

        Returns:
            List of educational concepts.
        """
        schema = {
            "type": "object",
            "properties": {
                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "category": {"type": "string"},
                            "description": {"type": "string"},
                            "key_points": {"type": "array", "items": {"type": "string"}},
                            "examples": {"type": "array", "items": {"type": "string"}},
                            "related_concepts": {"type": "array", "items": {"type": "string"}},
                            "practical_applications": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "common_mistakes": {"type": "array", "items": {"type": "string"}},
                            "source_text": {"type": "string"},
                        },
                        "required": ["name", "category", "description"],
                    },
                },
            },
            "required": ["concepts"],
        }

        try:
            result = await self._client.extract_structured(
                text,
                schema,
                model=self._config.extraction_model,
            )

            concepts = []
            for item in result.data.get("concepts", []):
                try:
                    concepts.append(EducationalConcept(**item))
                except Exception as e:
                    logger.warning(f"Invalid educational concept: {e}")

            return concepts
        except Exception as e:
            logger.error(f"Educational concept extraction failed: {e}")
            return []

    async def _extract_market_analysis(self, text: str) -> Optional[MarketAnalysis]:
        """Extract market analysis from text.

        Args:
            text: Text to extract from.

        Returns:
            Market analysis if found.
        """
        schema = {
            "type": "object",
            "properties": {
                "market_bias": {"type": "string"},
                "assets_analyzed": {"type": "array", "items": {"type": "string"}},
                "key_observations": {"type": "array", "items": {"type": "string"}},
                "support_levels": {"type": "array", "items": {"type": "number"}},
                "resistance_levels": {"type": "array", "items": {"type": "number"}},
                "indicators_used": {"type": "array", "items": {"type": "string"}},
                "timeframes_analyzed": {"type": "array", "items": {"type": "string"}},
                "key_themes": {"type": "array", "items": {"type": "string"}},
                "risk_factors": {"type": "array", "items": {"type": "string"}},
                "source_text": {"type": "string"},
            },
            "required": ["market_bias"],
        }

        try:
            result = await self._client.extract_structured(
                text,
                schema,
                model=self._config.extraction_model,
            )

            return MarketAnalysis(**result.data)
        except Exception as e:
            logger.error(f"Market analysis extraction failed: {e}")
            return None

    async def _extract_summary(self, text: str) -> Optional[str]:
        """Extract a summary from text.

        Args:
            text: Text to summarize.

        Returns:
            Summary text.
        """
        schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the main points (2-3 sentences)",
                },
            },
            "required": ["summary"],
        }

        try:
            result = await self._client.extract_structured(
                text[:3000],  # Limit input length
                schema,
                model=self._config.extraction_model,
            )

            return result.data.get("summary")
        except Exception as e:
            logger.error(f"Summary extraction failed: {e}")
            return None

    # ==================== Utility Methods ====================

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Deduplicate entities by name and type.

        Args:
            entities: List of entities.

        Returns:
            Deduplicated list.
        """
        seen = {}
        for entity in entities:
            key = (entity.name.lower(), entity.entity_type.lower())
            if key not in seen:
                seen[key] = entity
            else:
                # Merge properties and increment mention count
                seen[key].mention_count += 1
                seen[key].properties.update(entity.properties)
                if entity.source_text and not seen[key].source_text:
                    seen[key].source_text = entity.source_text

        return list(seen.values())

    async def extract_batch(
        self,
        items: list[dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> list[Union[VideoAnalysis, DocumentAnalysis]]:
        """Extract from multiple items in batch.

        Args:
            items: List of items with 'content', 'id', 'type', etc.
            progress_callback: Optional callback for progress.

        Returns:
            List of analysis results.
        """
        results = []
        total = len(items)

        for i, item in enumerate(items):
            try:
                if item.get("type") == "video":
                    result = await self.extract_from_video(
                        transcript=item["content"],
                        video_id=item["id"],
                        title=item.get("title"),
                        channel=item.get("channel"),
                    )
                else:
                    result = await self.extract_from_document(
                        content=item["content"],
                        document_id=item["id"],
                        document_name=item.get("name", item["id"]),
                        document_type=item.get("document_type", "unknown"),
                    )

                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total, item["id"])

            except Exception as e:
                logger.error(f"Failed to extract from {item['id']}: {e}")

        return results

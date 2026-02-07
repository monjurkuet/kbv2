"""Integration of Experience Bank with MultiAgentExtractor.

This module provides the integration layer to connect the Experience Bank
with the multi-agent extraction pipeline for few-shot prompting.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from knowledge_base.intelligence.v1.self_improvement import (
    ExperienceBank,
    ExperienceBankMiddleware,
)
from knowledge_base.persistence.v1.schema import Chunk
import logging

logger = logging.getLogger(__name__)


class ExperienceEnhancedPerception:
    """Enhances perception agent with experience-based few-shot prompting."""

    def __init__(
        self,
        experience_bank: Optional[ExperienceBank] = None,
        enable_few_shot: bool = True,
        max_examples: int = 3,
    ):
        self.experience_bank = experience_bank
        self.enable_few_shot = enable_few_shot
        self.max_examples = max_examples
        self.middleware = None

        if experience_bank:
            self.middleware = ExperienceBankMiddleware(experience_bank)

    async def enrich_prompt(
        self, base_prompt: str, chunk: Chunk, domain: str = "GENERAL"
    ) -> str:
        """Enrich extraction prompt with few-shot examples.

        Args:
            base_prompt: Original perception prompt
            chunk: Document chunk being processed
            domain: Domain for filtering examples

        Returns:
            Enriched prompt with few-shot examples
        """
        if not self.enable_few_shot or not self.middleware:
            return base_prompt

        try:
            enriched = await self.middleware.enrich_prompt_with_examples(
                base_prompt=base_prompt,
                text=chunk.text,
                domain=domain,
                k=self.max_examples,
            )
            return enriched
        except Exception as e:
            # Log but don't fail extraction
            logger.warning(f"Failed to enrich prompt with examples: {e}", exc_info=True)
            return base_prompt

    async def store_successful_extraction(
        self,
        chunk: Chunk,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        quality_score: float,
        domain: str = "GENERAL",
        extraction_method: str = "multi_agent",
    ) -> Optional[UUID]:
        """Store successful extraction in experience bank.

        Args:
            chunk: Source chunk
            entities: Extracted entities
            relationships: Extracted relationships
            quality_score: Quality score from evaluation
            domain: Domain label
            extraction_method: Method used

        Returns:
            Experience ID if stored
        """
        if not self.middleware or quality_score < 0.85:
            return None

        try:
            return await self.middleware.store_successful_extraction(
                text=chunk.text,
                entities=entities,
                relationships=relationships,
                quality_score=quality_score,
                domain=domain,
                chunk_id=chunk.id,
                extraction_method=extraction_method,
            )
        except Exception as e:
            logger.warning(f"Failed to store extraction experience: {e}", exc_info=True)
            return None


class ExperienceBankTracker:
    """Tracks extraction statistics for experience bank monitoring."""

    def __init__(self, experience_bank: Optional[ExperienceBank] = None):
        self.experience_bank = experience_bank
        self.stats = {
            "stored_count": 0,
            "retrieved_count": 0,
            "quality_scores": [],
            "domains": set(),
        }

    def record_storage(self, quality_score: float, domain: str):
        """Record a successful storage."""
        self.stats["stored_count"] += 1
        self.stats["quality_scores"].append(quality_score)
        self.stats["domains"].add(domain)

    def record_retrieval(self, count: int):
        """Record example retrievals."""
        self.stats["retrieved_count"] += count

    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        avg_quality = 0.0
        if self.stats["quality_scores"]:
            avg_quality = sum(self.stats["quality_scores"]) / len(
                self.stats["quality_scores"]
            )

        return {
            "stored_count": self.stats["stored_count"],
            "retrieved_count": self.stats["retrieved_count"],
            "average_quality": avg_quality,
            "domains": list(self.stats["domains"]),
        }

    async def get_experience_bank_stats(self) -> Optional[Dict[str, Any]]:
        """Get underlying experience bank statistics."""
        if not self.experience_bank:
            return None

        try:
            return await self.experience_bank.get_experience_statistics()
        except Exception as e:
            logger.warning(f"Failed to get experience bank stats: {e}", exc_info=True)
            return None


def create_crypto_domain_mapping() -> Dict[str, str]:
    """Create mapping for crypto domain detection.

    Maps from detected domain to experience bank domain label.
    """
    return {
        "BITCOIN": "CRYPTO_BITCOIN",
        "DIGITAL_ASSETS": "CRYPTO_ASSETS",
        "STABLECOINS": "CRYPTO_STABLECOINS",
        "BLOCKCHAIN_INFRA": "CRYPTO_INFRA",
        "DEFI": "CRYPTO_DEFI",
        "CRYPTO_MARKETS": "CRYPTO_MARKETS",
        "INSTITUTIONAL_CRYPTO": "CRYPTO_INSTITUTIONAL",
        "CRYPTO_REGULATION": "CRYPTO_REGULATION",
        "CRYPTO_AI": "CRYPTO_AI",
        "TOKENIZATION": "CRYPTO_TOKENIZATION",
        "GENERAL": "GENERAL",
    }


async def get_domain_examples(
    experience_bank: ExperienceBank,
    domain: str,
    min_quality: float = 0.90,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Get examples for a specific crypto domain.

    Args:
        experience_bank: Experience bank instance
        domain: Domain label (e.g., "CRYPTO_DEFI")
        min_quality: Minimum quality threshold
        limit: Maximum examples to return

    Returns:
        List of extraction examples
    """
    examples = await experience_bank.get_experiences_by_domain(
        domain=domain, min_quality=min_quality, limit=limit
    )

    return [
        {
            "text": ex.text_snippet,
            "entities": ex.entities,
            "relationships": ex.relationships,
            "quality": ex.quality_score,
            "patterns": ex.extraction_patterns,
        }
        for ex in examples
    ]

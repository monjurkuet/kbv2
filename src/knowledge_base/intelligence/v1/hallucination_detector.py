"""LLM-as-Judge hallucination detection service for KBV2.

This module provides entity quality verification using LLM-based judgment
to detect fabricated attributes vs. supported attributes.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from knowledge_base.common.resilient_gateway import (
    ResilientGatewayClient,
    ResilientGatewayConfig,
)
from knowledge_base.config.constants import (
    DEFAULT_LLM_MODEL,
    HALLUCINATION_THRESHOLD,
    LLM_GATEWAY_URL,
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
)


class VerificationStatus(str, Enum):
    """Status of attribute verification."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    CONFLICTING = "conflicting"


class RiskLevel(str, Enum):
    """Risk level for hallucinated entities."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttributeVerification:
    """Verification result for a single attribute."""

    attribute_name: str
    claimed_value: str
    status: VerificationStatus
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: str = ""
    explanation: str = ""


@dataclass
class EntityVerification:
    """Complete verification result for an entity."""

    entity_name: str
    entity_type: str
    risk_level: RiskLevel
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    attributes: list[AttributeVerification] = field(default_factory=list)
    total_attributes: int = 0
    supported_count: int = 0
    unsupported_count: int = 0
    inconclusive_count: int = 0
    is_hallucinated: bool = False
    hallucination_reasons: list[str] = field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        """Ratio of supported attributes."""
        if self.total_attributes == 0:
            return 0.0
        return self.supported_count / self.total_attributes


class HallucinationDetectorConfig(BaseModel):
    """Configuration for hallucination detector."""

    model_config = {"arbitrary_types_allowed": True}

    gateway_client: ResilientGatewayClient | None = Field(
        default=None, description="Resilient gateway client instance"
    )
    url: str = Field(default=LLM_GATEWAY_URL, description="LLM gateway URL")
    model: str = Field(default=DEFAULT_LLM_MODEL, description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for verification")
    max_tokens: int = Field(default=1024, description="Max tokens for verification")
    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE, description="Batch size for batch verification"
    )
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold")
    hallucination_threshold: float = Field(
        default=HALLUCINATION_THRESHOLD,
        description="Unsupported ratio threshold for hallucination",
    )


class HallucinationDetector:
    """LLM-as-Judge service for detecting hallucinations in entity data."""

    def __init__(self, config: HallucinationDetectorConfig | None = None) -> None:
        """Initialize hallucination detector.

        Args:
            config: Detector configuration.
        """
        self._config = config or HallucinationDetectorConfig()
        self._gateway_client = self._config.gateway_client

    async def _get_gateway_client(self) -> ResilientGatewayClient:
        """Get or create resilient gateway client.

        Returns:
            Resilient gateway client instance.
        """
        if self._gateway_client is None:
            gateway_config = ResilientGatewayConfig(
                url=self._config.url,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            self._gateway_client = ResilientGatewayClient(gateway_config)
        return self._gateway_client

    async def verify_attribute(
        self,
        entity_name: str,
        entity_type: str,
        attribute_name: str,
        claimed_value: str,
        context: str,
    ) -> AttributeVerification:
        """Verify a single attribute using LLM judgment.

        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity.
            attribute_name: Name of the attribute to verify.
            claimed_value: Claimed value of the attribute.
            context: Supporting context or evidence.

        Returns:
            Verification result for the attribute.
        """
        client = await self._get_gateway_client()

        system_prompt = """You are a factual verification expert. Your task is to determine
whether an attribute claim is SUPPORTED or UNSUPPORTED based on the provided context.

Evaluation criteria:
- SUPPORTED: The context clearly confirms the attribute claim
- UNSUPPORTED: The context contradicts or does not mention the claim
- INCONCLUSIVE: Context is insufficient or ambiguous
- CONFLICTING: Context provides conflicting information

Provide your assessment with confidence score (0.0-1.0) and brief reasoning."""

        user_prompt = f"""Verify the following attribute claim:

Entity: {entity_name} ({entity_type})
Attribute: {attribute_name}
Claimed Value: {claimed_value}

Context:
{context}

Respond with JSON only:
{{
    "status": "supported|unsupported|inconclusive|conflicting",
    "confidence": 0.0-1.0,
    "evidence": "brief evidence from context",
    "explanation": "brief explanation of reasoning"
}}"""

        try:
            response_text = await client.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            response = self._parse_json_response(response_text)

            status = VerificationStatus(response.get("status", "inconclusive"))
            confidence = float(response.get("confidence", 0.5))
            evidence = response.get("evidence", "")
            explanation = response.get("explanation", "")

            return AttributeVerification(
                attribute_name=attribute_name,
                claimed_value=claimed_value,
                status=status,
                confidence=confidence,
                evidence=evidence,
                explanation=explanation,
            )
        except Exception as e:
            return AttributeVerification(
                attribute_name=attribute_name,
                claimed_value=claimed_value,
                status=VerificationStatus.INCONCLUSIVE,
                confidence=0.0,
                evidence="",
                explanation=f"Verification error: {str(e)}",
            )

    @staticmethod
    def _parse_json_response(response_text: str) -> dict[str, Any]:
        """Parse JSON response from the LLM gateway."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {}

    def _calculate_risk_level(
        self, entity_verification: EntityVerification
    ) -> RiskLevel:
        """Calculate risk level based on verification results.

        Args:
            entity_verification: Entity verification result.

        Returns:
            Risk level.
        """
        unsupported_ratio = entity_verification.unsupported_count / max(
            entity_verification.total_attributes, 1
        )
        confidence = entity_verification.overall_confidence

        if unsupported_ratio >= 0.5 or confidence < 0.3:
            return RiskLevel.CRITICAL
        elif unsupported_ratio >= 0.3 or confidence < 0.5:
            return RiskLevel.HIGH
        elif unsupported_ratio >= 0.15 or confidence < 0.7:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _create_entity_verification(
        self,
        entity_name: str,
        entity_type: str,
        attributes: list[AttributeVerification],
    ) -> EntityVerification:
        """Create entity verification from attribute verifications.

        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity.
            attributes: List of attribute verifications.

        Returns:
            Complete entity verification.
        """
        total = len(attributes)
        supported = sum(
            1 for a in attributes if a.status == VerificationStatus.SUPPORTED
        )
        unsupported = sum(
            1 for a in attributes if a.status == VerificationStatus.UNSUPPORTED
        )
        inconclusive = sum(
            1 for a in attributes if a.status == VerificationStatus.INCONCLUSIVE
        )

        confidences = [a.confidence for a in attributes if a.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        unsupported_ratio = unsupported / max(total, 1)
        is_hallucinated = unsupported_ratio >= self._config.hallucination_threshold

        hallucination_reasons = []
        for attr in attributes:
            if attr.status == VerificationStatus.UNSUPPORTED:
                hallucination_reasons.append(
                    f"Attribute '{attr.attribute_name}': {attr.explanation}"
                )
            elif attr.status == VerificationStatus.CONFLICTING:
                hallucination_reasons.append(
                    f"Attribute '{attr.attribute_name}' has conflicting evidence"
                )

        risk_level = RiskLevel.LOW
        if is_hallucinated:
            if unsupported_ratio >= 0.7:
                risk_level = RiskLevel.CRITICAL
            elif unsupported_ratio >= 0.5:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.MEDIUM
        elif avg_confidence < 0.5:
            risk_level = RiskLevel.MEDIUM

        return EntityVerification(
            entity_name=entity_name,
            entity_type=entity_type,
            overall_confidence=avg_confidence,
            risk_level=risk_level,
            attributes=attributes,
            total_attributes=total,
            supported_count=supported,
            unsupported_count=unsupported,
            inconclusive_count=inconclusive,
            is_hallucinated=is_hallucinated,
            hallucination_reasons=hallucination_reasons,
        )

    async def verify_entity(
        self,
        entity_name: str,
        entity_type: str,
        attributes: dict[str, str],
        context: str,
    ) -> EntityVerification:
        """Verify all attributes of an entity.

        Args:
            entity_name: Name of the entity.
            entity_type: Type of the entity.
            attributes: Dictionary of attribute names to claimed values.
            context: Supporting context or evidence.

        Returns:
            Complete entity verification.
        """
        attribute_verifications = []

        for attr_name, attr_value in attributes.items():
            verification = await self.verify_attribute(
                entity_name=entity_name,
                entity_type=entity_type,
                attribute_name=attr_name,
                claimed_value=attr_value,
                context=context,
            )
            attribute_verifications.append(verification)

        return self._create_entity_verification(
            entity_name=entity_name,
            entity_type=entity_type,
            attributes=attribute_verifications,
        )

    async def verify_entity_batch(
        self,
        entities: list[dict[str, Any]],
        context_key: str = "context",
    ) -> list[EntityVerification]:
        """Verify multiple entities in batch for efficiency.

        Args:
            entities: List of entity dictionaries with 'name', 'type', 'attributes', and context.
            context_key: Key for accessing context in entity dict.

        Returns:
            List of entity verifications.
        """
        results = []

        for i in range(0, len(entities), self._config.batch_size):
            batch = entities[i : i + self._config.batch_size]
            batch_results = await self._verify_batch(batch, context_key)
            results.extend(batch_results)

        return results

    async def _verify_batch(
        self,
        entities: list[dict[str, Any]],
        context_key: str,
    ) -> list[EntityVerification]:
        """Verify a batch of entities.

        Args:
            entities: List of entity dictionaries.
            context_key: Key for accessing context.

        Returns:
            List of entity verifications.
        """
        tasks = []
        for entity in entities:
            tasks.append(
                self.verify_entity(
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    attributes=entity.get("attributes", {}),
                    context=entity.get(context_key, ""),
                )
            )
        return await asyncio.gather(*tasks)

    async def verify_claim(
        self,
        claim: str,
        evidence: str,
    ) -> dict[str, Any]:
        """Verify a standalone claim against evidence.

        Args:
            claim: The claim to verify.
            evidence: Evidence to verify against.

        Returns:
            Dictionary with verification results.
        """
        client = await self._get_llm_client()

        system_prompt = """You are a factual verification expert. Evaluate whether the claim
is supported by the evidence. Respond with:
- supported: Evidence clearly confirms the claim
- unsupported: Evidence contradicts or does not support the claim
- inconclusive: Evidence is insufficient

Provide confidence score and brief reasoning."""

        user_prompt = f"""Claim: {claim}

Evidence: {evidence}

Respond with JSON:
{{
    "status": "supported|unsupported|inconclusive",
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}"""

        try:
            response = await client.complete_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            return {
                "claim": claim,
                "status": response.get("status", "inconclusive"),
                "confidence": float(response.get("confidence", 0.5)),
                "explanation": response.get("explanation", ""),
            }
        except Exception as e:
            return {
                "claim": claim,
                "status": "inconclusive",
                "confidence": 0.0,
                "explanation": f"Verification error: {str(e)}",
            }

    async def compare_entities(
        self,
        primary_entity: dict[str, Any],
        secondary_entity: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        """Compare two entity descriptions for consistency.

        Args:
            primary_entity: Primary entity with name, type, attributes.
            secondary_entity: Secondary entity to compare.
            context: Supporting context.

        Returns:
            Comparison results.
        """
        client = await self._get_llm_client()

        primary_str = json.dumps(primary_entity, indent=2)
        secondary_str = json.dumps(secondary_entity, indent=2)

        system_prompt = """You are an entity consistency checker. Compare two entity
descriptions and identify:
1. Contradictions
2. Complementary information
3. Potential hallucinations

Focus on factual consistency."""

        user_prompt = f"""Primary Entity:
{primary_str}

Secondary Entity:
{secondary_str}

Context:
{context}

Respond with JSON:
{{
    "consistency_score": 0.0-1.0,
    "contradictions": ["list of contradictions"],
    "complementary": ["list of complementary info"],
    "hallucination_flags": ["potential hallucinations in either entity"],
    "summary": "brief summary of comparison"
}}"""

        try:
            response = await client.complete_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )

            return {
                "primary_entity": primary_entity.get("name", "unknown"),
                "secondary_entity": secondary_entity.get("name", "unknown"),
                "consistency_score": float(response.get("consistency_score", 0.5)),
                "contradictions": response.get("contradictions", []),
                "complementary": response.get("complementary", []),
                "hallucination_flags": response.get("hallucination_flags", []),
                "summary": response.get("summary", ""),
            }
        except Exception as e:
            return {
                "primary_entity": primary_entity.get("name", "unknown"),
                "secondary_entity": secondary_entity.get("name", "unknown"),
                "consistency_score": 0.0,
                "contradictions": [],
                "complementary": [],
                "hallucination_flags": [f"Comparison error: {str(e)}"],
                "summary": "Failed to complete comparison",
            }

    def get_verification_summary(
        self,
        verifications: list[EntityVerification],
    ) -> dict[str, Any]:
        """Generate summary of multiple verifications.

        Args:
            verifications: List of entity verifications.

        Returns:
            Summary statistics.
        """
        if not verifications:
            return {
                "total_entities": 0,
                "hallucinated_count": 0,
                "risk_distribution": {},
                "average_confidence": 0.0,
            }

        hallucinated = sum(1 for v in verifications if v.is_hallucinated)
        avg_confidence = sum(v.overall_confidence for v in verifications) / len(
            verifications
        )

        risk_distribution = {}
        for v in verifications:
            risk_level = v.risk_level.value
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

        return {
            "total_entities": len(verifications),
            "hallucinated_count": hallucinated,
            "hallucination_rate": hallucinated / len(verifications),
            "risk_distribution": risk_distribution,
            "average_confidence": avg_confidence,
        }

    async def close(self) -> None:
        """Close the LLM client."""
        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None


import asyncio

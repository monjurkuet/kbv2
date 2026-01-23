"""Verbatim-grounded entity resolution agent."""

import asyncio
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.common.gateway import GatewayClient
from knowledge_base.persistence.v1.schema import (
    EntityResolution,
    Entity,
)
from knowledge_base.persistence.v1.vector_store import VectorStore


class ResolutionConfig(BaseSettings):
    """Resolution configuration."""

    model_config = SettingsConfigDict()

    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.85
    max_candidates: int = 10


class EntityCandidate(BaseModel):
    """Entity resolution candidate."""

    entity_id: UUID = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    entity_type: str | None = Field(None, description="Entity type")
    description: str | None = Field(None, description="Entity description")
    similarity: float = Field(..., description="Vector similarity")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )


class ResolutionAgent:
    """Agent for verbatim-grounded entity resolution."""

    def __init__(
        self,
        gateway: GatewayClient,
        vector_store: VectorStore,
        config: ResolutionConfig | None = None,
    ) -> None:
        """Initialize resolution agent.

        Args:
            gateway: LLM gateway client.
            vector_store: Vector store for similarity search.
            config: Resolution configuration.
        """
        self._gateway = gateway
        self._vector_store = vector_store
        self._config = config or ResolutionConfig()

    async def resolve_entity(
        self,
        entity: Entity,
        candidate_entities: list[Entity],
        source_text: str,
    ) -> EntityResolution:
        """Resolve entity against candidates using hybrid matching.

        Args:
            entity: Entity to resolve.
            candidate_entities: List of candidate entities.
            source_text: Source text containing verbatim context.

        Returns:
            Entity resolution with grounding quote and confidence.
        """
        if not candidate_entities:
            return EntityResolution(
                entity_id=entity.id,
                merged_entity_ids=[],
                grounding_quote="No candidates found",
                confidence_score=1.0,
                human_review_required=False,
            )

        candidates = [
            EntityCandidate(
                entity_id=c.id,
                name=c.name,
                entity_type=c.entity_type,
                description=c.description,
                similarity=0.0,
                properties=c.properties or {},
            )
            for c in candidate_entities
        ]

        resolution = await self._llm_resolve(entity, candidates, source_text)

        if resolution.confidence_score < self._config.confidence_threshold:
            resolution.human_review_required = True

        return resolution

    async def _llm_resolve(
        self,
        entity: Entity,
        candidates: list[EntityCandidate],
        source_text: str,
    ) -> EntityResolution:
        """Use LLM to reason about entity resolution.

        Args:
            entity: Entity to resolve.
            candidates: Candidate entities.
            source_text: Source text.

        Returns:
            Entity resolution.
        """
        system_prompt = self._get_resolution_system_prompt()

        user_prompt = self._get_resolution_user_prompt(entity, candidates, source_text)

        response = await self._gateway.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2,
        )

        return self._parse_resolution_response(response, entity.id, source_text)

    def _get_resolution_system_prompt(self) -> str:
        """Get resolution system prompt."""
        return """You are an expert entity resolution system. Your task is to determine if two or more entity references refer to the same real-world entity.

Rules:
1. Consider semantic similarity of names
2. Consider entity type consistency
3. Consider contextual information
4. ALWAYS provide a verbatim quote from the source text that supports your decision
5. Assign a confidence score (0.0-1.0) based on evidence strength
6. MERGING IS PROHIBITED WITHOUT A VERBATIM CITATION - you must provide a direct quote from the source text to justify merging entities

Output in JSON format:
{
  "decision": "keep|merge",
  "target_entity_id": "<uuid>",
  "merged_entity_ids": ["<uuid>", ...],
  "grounding_quote": "verbatim excerpt from source text",
  "confidence_score": 0.85,
  "reasoning": "explanation of decision"
}

Be conservative. If uncertain, prefer keeping entities separate."""

    def _get_resolution_user_prompt(
        self,
        entity: Entity,
        candidates: list[EntityCandidate],
        source_text: str,
    ) -> str:
        """Get resolution user prompt."""
        candidate_descriptions = "\n\n".join(
            [
                f"Candidate {i + 1}:\n"
                f"  ID: {c.entity_id}\n"
                f"  Name: {c.name}\n"
                f"  Type: {c.entity_type or 'N/A'}\n"
                f"  Description: {c.description or 'N/A'}\n"
                f"  Properties: {c.properties}"
                for i, c in enumerate(candidates)
            ]
        )

        prompt = f"""Determine if the following entity should be merged with any existing entities.

Target Entity:
  ID: {entity.id}
  Name: {entity.name}
  Type: {entity.entity_type or "N/A"}
  Description: {entity.description or "N/A"}
  Properties: {entity.properties or {}}

{candidate_descriptions}

Source Text:
{source_text}

Provide your analysis in JSON format."""

        return prompt

    def _parse_resolution_response(
        self,
        response: str,
        entity_id: UUID,
        source_text: str,
    ) -> EntityResolution:
        """Parse LLM resolution response.

        Args:
            response: LLM JSON response.
            entity_id: Target entity ID.
            source_text: Source text for quote validation.

        Returns:
            Parsed entity resolution.
        """
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return EntityResolution(
                entity_id=entity_id,
                merged_entity_ids=[],
                grounding_quote="Failed to parse response",
                confidence_score=0.0,
                human_review_required=True,
            )

        decision = data.get("decision", "keep")

        if decision == "keep":
            return EntityResolution(
                entity_id=entity_id,
                merged_entity_ids=[],
                grounding_quote=data.get("grounding_quote", "Kept as separate entity"),
                confidence_score=data.get("confidence_score", 1.0),
                human_review_required=False,
            )

        target_id = data.get("target_entity_id")
        merged_ids = data.get("merged_entity_ids", [])
        grounding_quote = data.get("grounding_quote", "")
        confidence = data.get("confidence_score", 0.0)

        if grounding_quote and grounding_quote not in source_text:
            grounding_quote = f"[Quote not found in source] {grounding_quote}"

        try:
            target_uuid = UUID(target_id) if target_id else entity_id
        except (ValueError, TypeError):
            target_uuid = entity_id

        merged_uuids = []
        for mid in merged_ids:
            try:
                merged_uuids.append(UUID(mid))
            except (ValueError, TypeError):
                pass

        return EntityResolution(
            entity_id=target_uuid,
            merged_entity_ids=merged_uuids,
            grounding_quote=grounding_quote,
            confidence_score=confidence,
            human_review_required=confidence < self._config.confidence_threshold,
        )

    async def batch_resolve_entities(
        self,
        entities: list[tuple[Entity, list[Entity], str]],
    ) -> list[EntityResolution]:
        """Resolve multiple entities in batch.

        Args:
            entities: List of (entity, candidates, source_text) tuples.

        Returns:
            List of entity resolutions.
        """
        tasks = [
            self.resolve_entity(entity, candidates, source_text)
            for entity, candidates, source_text in entities
        ]

        return await asyncio.gather(*tasks)

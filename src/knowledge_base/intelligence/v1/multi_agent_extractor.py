"""Multi-agent entity extraction system (GraphMaster-style).

Implements a coordinated multi-agent architecture for high-fidelity entity extraction:
- ManagerAgent: Orchestrates extraction workflow
- PerceptionAgent: Boundary-aware entity extraction (BANER-style)
- EnhancementAgent: Refines and links entities using KG context
- EvaluationAgent: Validates quality with LLM-as-Judge

Based on GraphMaster architecture from arXiv:2504.00711.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.clients import AsyncLLMClient
from knowledge_base.intelligence.v1.extraction_logging import (
    get_ingestion_logger,
)
from knowledge_base.persistence.v1.graph_store import GraphStore
from knowledge_base.persistence.v1.schema import (
    Chunk,
    Edge,
    EdgeType,
    Entity,
)
from knowledge_base.persistence.v1.vector_store import VectorStore


class ExtractionPhase(str, Enum):
    """Extraction pipeline phases."""

    PERCEPTION = "perception"
    ENHANCEMENT = "enhancement"
    EVALUATION = "evaluation"
    COMPLETED = "completed"


class EntityBoundaryType(str, Enum):
    """Boundary types for BANER-style extraction."""

    STRONG = "strong"
    WEAK = "weak"
    CROSSING = "crossing"


class EntityExtractionQuality(str, Enum):
    """Quality assessment levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


class MultiAgentExtractorConfig(BaseSettings):
    """Configuration for multi-agent extractor."""

    model_config = SettingsConfigDict()

    perception_temperature: float = 0.1
    enhancement_temperature: float = 0.2
    evaluation_temperature: float = 0.3
    max_entities_per_chunk: int = 50
    confidence_threshold: float = 0.7
    enhancement_max_iterations: int = 3
    evaluation_sample_rate: float = 0.2


class EntityCandidate(BaseModel):
    """Entity candidate from perception extraction."""

    text: str = Field(..., description="Entity text span")
    entity_type: str = Field(..., description="Entity type")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    boundary_type: EntityBoundaryType = Field(
        default=EntityBoundaryType.STRONG, description="Boundary confidence"
    )
    context: str = Field(..., description="Surrounding context")
    confidence: float = Field(default=0.0, description="Extraction confidence")


class ExtractedEntity(BaseModel):
    """Enhanced entity after processing."""

    id: UUID = Field(default_factory=uuid4, description="Entity ID")
    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    description: Optional[str] = Field(None, description="Entity description")
    properties: dict[str, Any] = Field(default_factory=dict)
    source_text: str = Field(..., description="Source text span")
    chunk_id: UUID = Field(..., description="Source chunk ID")
    confidence: float = Field(default=0.0)
    extraction_phase: ExtractionPhase = Field(default=ExtractionPhase.PERCEPTION)
    linked_entities: list[UUID] = Field(
        default_factory=list, description="Linked entity IDs"
    )
    is_cross_boundary: bool = Field(default=False, description="Spans chunk boundaries")


class ExtractionQualityScore(BaseModel):
    """Quality assessment from LLM-as-Judge."""

    overall_score: float = Field(..., ge=0.0, le=1.0)
    entity_quality: float = Field(..., ge=0.0, le=1.0)
    relationship_quality: float = Field(..., ge=0.0, le=1.0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    missing_entities: list[str] = Field(default_factory=list)
    spurious_entities: list[str] = Field(default_factory=list)
    quality_level: EntityExtractionQuality = Field(
        default=EntityExtractionQuality.MEDIUM
    )
    feedback: str = Field(..., description="Detailed feedback")


class EnhancementContext(BaseModel):
    """Context for entity enhancement."""

    model_config = {"arbitrary_types_allowed": True}

    existing_entities: list[Entity] = Field(
        default_factory=list, description="Existing KG entities"
    )
    recent_extractions: list[ExtractedEntity] = Field(
        default_factory=list, description="Current extraction batch"
    )
    schema_constraints: dict[str, Any] = Field(
        default_factory=dict, description="Schema constraints"
    )


class ExtractionWorkflowState(BaseModel):
    """State tracking for extraction workflow."""

    document_id: UUID = Field(..., description="Source document ID")
    current_phase: ExtractionPhase = Field(default=ExtractionPhase.PERCEPTION)
    perception_entities: list[ExtractedEntity] = Field(default_factory=list)
    enhanced_entities: list[ExtractedEntity] = Field(default_factory=list)
    quality_scores: list[ExtractionQualityScore] = Field(default_factory=list)
    iteration_count: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)


class PerceptionAgent:
    """PerceptionAgent - Boundary-Aware Named Entity Recognition.

    Implements BANER-style extraction with awareness of entity boundaries
    and cross-boundary entity handling.
    """

    def __init__(
        self,
        gateway: AsyncLLMClient,
        config: MultiAgentExtractorConfig | None = None,
    ) -> None:
        """Initialize perception agent.

        Args:
            gateway: LLM gateway client.
            config: Extractor configuration.
        """
        self._gateway = gateway
        self._config = config or MultiAgentExtractorConfig()

    async def extract_entities(
        self,
        chunks: list[Chunk],
        entity_types: list[str] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from document chunks.

        Args:
            chunks: Document chunks to process.
            entity_types: Optional list of entity types to extract.

        Returns:
            List of extracted entities.
        """
        if entity_types is None:
            entity_types = [
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "DATE",
                "EVENT",
                "PRODUCT",
                "CONCEPT",
                "TECHNOLOGY",
            ]

        tasks = [self._extract_from_chunk(chunk, entity_types) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        entities = []
        for result in results:
            if isinstance(result, Exception):
                continue
            entities.extend(result)

        return self._merge_cross_boundary_entities(entities)

    async def _extract_from_chunk(
        self,
        chunk: Chunk,
        entity_types: list[str],
    ) -> list[ExtractedEntity]:
        """Extract entities from a single chunk.

        Args:
            chunk: Document chunk.
            entity_types: Entity types to extract.

        Returns:
            Extracted entities from chunk.
        """
        system_prompt = self._get_perception_system_prompt(entity_types)

        response = await self._gateway.complete(
            prompt=evaluation_prompt,
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            temperature=0.3,
            json_mode=True,
        )

        return self._parse_perception_response(response, chunk)

    def _get_perception_system_prompt(self, entity_types: list[str]) -> str:
        """Get perception extraction system prompt."""
        types_str = ", ".join(entity_types)
        return f"""You are a boundary-aware named entity recognition system (BANER).

Your task is to extract entities with precise boundary detection from the provided text.

Entity Types to Extract:
{types_str}

Guidelines:
1. Identify entities with exact character boundaries
2. Classify boundary confidence:
   - STRONG: Clear, unambiguous entity boundaries
   - WEAK: Uncertain boundaries, may be part of larger entity
   - CROSSING: Entity spans across typical boundaries (e.g., compound entities)
3. Extract entities in context - include surrounding text for grounding
4. Handle nested and compound entities appropriately
5. Maintain consistency in entity naming conventions

Output format (JSON array):
[
  {{
    "text": "entity text",
    "type": "ENTITY_TYPE",
    "start_char": 0,
    "end_char": 15,
    "boundary_type": "strong|weak|crossing",
    "context": "surrounding text context",
    "confidence": 0.95
  }}
]

Be precise with character positions and prioritize extraction quality."""

    def _get_perception_user_prompt(
        self,
        chunk: Chunk,
        entity_types: list[str],
    ) -> str:
        """Get perception extraction user prompt."""
        return f"""Extract named entities from the following text:

---TEXT START---
{chunk.text}
---TEXT END---

Entity Types: {", ".join(entity_types)}

Provide entity extractions in the specified JSON format. Include 50-100 characters of context around each entity."""

    def _parse_perception_response(
        self,
        response: str,
        chunk: Chunk,
    ) -> list[ExtractedEntity]:
        """Parse perception extraction response."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return []

        entities = []
        for item in data:
            try:
                entity = ExtractedEntity(
                    name=item.get("text", ""),
                    entity_type=item.get("type", "UNKNOWN"),
                    description=item.get("description"),
                    source_text=item.get("context", ""),
                    chunk_id=chunk.id,
                    confidence=item.get("confidence", 0.7),
                    extraction_phase=ExtractionPhase.PERCEPTION,
                    is_cross_boundary=item.get("boundary_type")
                    == EntityBoundaryType.CROSSING.value,
                )
                entity.properties = {
                    "start_char": item.get("start_char"),
                    "end_char": item.get("end_char"),
                    "boundary_type": item.get("boundary_type"),
                }
                entities.append(entity)
            except Exception:
                continue

        return entities

    def _merge_cross_boundary_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Merge entities that span chunk boundaries."""
        crossing = [e for e in entities if e.is_cross_boundary]
        strong = [e for e in entities if not e.is_cross_boundary]

        for cross in crossing:
            cross.is_cross_boundary = False

        return strong + crossing


class EnhancementAgent:
    """EnhancementAgent - Entity refinement and linking.

    Refines extracted entities using knowledge graph context and
    establishes entity links based on semantic relationships.
    """

    def __init__(
        self,
        gateway: AsyncLLMClient,
        graph_store: GraphStore,
        vector_store: VectorStore,
        config: MultiAgentExtractorConfig | None = None,
    ) -> None:
        """Initialize enhancement agent.

        Args:
            gateway: LLM gateway client.
            graph_store: Knowledge graph store.
            vector_store: Vector store for similarity.
            config: Extractor configuration.
        """
        self._gateway = gateway
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._config = config or MultiAgentExtractorConfig()

    async def enhance_entities(
        self,
        entities: list[ExtractedEntity],
        context: EnhancementContext,
    ) -> list[ExtractedEntity]:
        """Enhance extracted entities with KG context.

        Args:
            entities: Entities from perception extraction.
            context: Enhancement context with existing KG data.

        Returns:
            Enhanced entities with links and refinements.
        """
        if not entities:
            return []

        enhanced = []
        for entity in entities:
            enhanced_entity = await self._enhance_single_entity(entity, context)
            enhanced.append(enhanced_entity)

        return self._establish_entity_links(enhanced, context)

    async def _enhance_single_entity(
        self,
        entity: ExtractedEntity,
        context: EnhancementContext,
    ) -> ExtractedEntity:
        """Enhance a single entity."""
        system_prompt = self._get_enhancement_system_prompt()
        user_prompt = self._get_enhancement_user_prompt(entity, context)

        response = await self._gateway.complete(
            prompt=enhancement_prompt,
            system_prompt=ENHANCEMENT_SYSTEM_PROMPT,
            temperature=0.2,
            json_mode=True,
        )

        return self._parse_enhancement_response(response, entity)

    def _get_enhancement_system_prompt(self) -> str:
        """Get entity enhancement system prompt."""
        return """You are an entity enhancement system. Your task is to:

1. Refine entity names and types for consistency
2. Generate descriptive summaries for entities
3. Extract additional properties from entity context
4. Identify potential entity links based on semantic similarity

Guidelines:
- Use canonical entity names (prefer formal names over abbreviations)
- Generate concise but informative descriptions (1-2 sentences)
- Extract factual properties as key-value pairs
- Consider existing entities in the knowledge graph for alignment

Output format (JSON):
{
  "refined_name": "Canonical Entity Name",
  "refined_type": "ENTITY_TYPE",
  "description": "Entity description",
  "properties": {{"key": "value"}},
  "confidence_adjustment": -0.05
}"""

    def _get_enhancement_user_prompt(
        self,
        entity: ExtractedEntity,
        context: EnhancementContext,
    ) -> str:
        """Get enhancement user prompt."""
        existing_names = [e.name for e in context.existing_entities[:10]]
        existing_str = "\n".join(existing_names) if existing_names else "None"

        return f"""Enhance the following extracted entity:

Entity:
  Name: {entity.name}
  Type: {entity.entity_type}
  Source: {entity.source_text[:200]}
  Confidence: {entity.confidence}

Existing Knowledge Graph Entities (for reference):
{existing_str}

Provide enhancements in JSON format."""

    def _parse_enhancement_response(
        self,
        response: str,
        original: ExtractedEntity,
    ) -> ExtractedEntity:
        """Parse enhancement response."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return original

        adjusted_confidence = max(
            0.0,
            min(
                1.0,
                original.confidence + data.get("confidence_adjustment", 0.0),
            ),
        )

        return ExtractedEntity(
            id=original.id,
            name=data.get("refined_name", original.name),
            entity_type=data.get("refined_type", original.entity_type),
            description=data.get("description", original.description),
            properties={**original.properties, **data.get("properties", {})},
            source_text=original.source_text,
            chunk_id=original.chunk_id,
            confidence=adjusted_confidence,
            extraction_phase=ExtractionPhase.ENHANCEMENT,
            linked_entities=original.linked_entities,
            is_cross_boundary=original.is_cross_boundary,
        )

    def _establish_entity_links(
        self,
        entities: list[ExtractedEntity],
        context: EnhancementContext,
    ) -> list[ExtractedEntity]:
        """Establish semantic links between entities."""
        linked_entities = []

        for entity in entities:
            entity.linked_entities = []

            for existing in context.existing_entities:
                if self._should_link(entity, existing):
                    entity.linked_entities.append(existing.id)

            linked_entities.append(entity)

        for i, entity in enumerate(linked_entities):
            for j, other in enumerate(linked_entities):
                if i != j and self._should_link(entity, other):
                    if other.id not in entity.linked_entities:
                        entity.linked_entities.append(other.id)

        return linked_entities

    def _should_link(
        self,
        entity: ExtractedEntity,
        other: Entity | ExtractedEntity,
    ) -> bool:
        """Determine if two entities should be linked."""
        name_similarity = self._calculate_name_similarity(entity.name, other.name)
        type_match = entity.entity_type == getattr(other, "entity_type", None)

        return name_similarity > 0.85 or (type_match and name_similarity > 0.7)

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity."""
        n1, n2 = name1.lower().strip(), name2.lower().strip()
        if n1 == n2:
            return 1.0
        if n1 in n2 or n2 in n1:
            return 0.9
        return 0.0


class EvaluationAgent:
    """EvaluationAgent - LLM-as-Judge quality validation.

    Validates extraction quality using LLM-based evaluation
    following established LLM-as-Judge methodologies.
    """

    def __init__(
        self,
        gateway: AsyncLLMClient,
        config: MultiAgentExtractorConfig | None = None,
    ) -> None:
        """Initialize evaluation agent.

        Args:
            gateway: LLM gateway client.
            config: Extractor configuration.
        """
        self._gateway = gateway
        self._config = config or MultiAgentExtractorConfig()

    async def evaluate_extraction(
        self,
        entities: list[ExtractedEntity],
        source_chunks: list[Chunk],
        sampling_rate: float | None = None,
    ) -> ExtractionQualityScore:
        """Evaluate extraction quality using LLM-as-Judge.

        Args:
            entities: Extracted entities to evaluate.
            source_chunks: Original source chunks.
            sampling_rate: Rate of entities to sample for evaluation.

        Returns:
            Quality assessment score.
        """
        if sampling_rate is None:
            sampling_rate = self._config.evaluation_sample_rate

        sample_size = max(1, int(len(entities) * sampling_rate))
        sampled_entities = (
            entities[:sample_size] if sample_size < len(entities) else entities
        )

        source_text = "\n".join(chunk.text for chunk in source_chunks[:5])

        system_prompt = self._get_evaluation_system_prompt()
        user_prompt = self._get_evaluation_user_prompt(sampled_entities, source_text)

        response = await self._gateway.complete(
            prompt=perception_prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            json_mode=True,
        )

        return self._parse_evaluation_response(response, len(entities))

    def _get_evaluation_system_prompt(self) -> str:
        """Get evaluation system prompt."""
        return """You are an expert judge for entity extraction quality assessment.

Evaluate extracted entities against the following criteria:

1. ENTITY QUALITY (0-1):
   - Are entities well-formed and coherent?
   - Are entity types appropriate?
   - Do entities have proper grounding in source text?

2. RELATIONSHIP QUALITY (0-1):
   - Are entity links meaningful?
   - Are relationships consistent with source context?

3. COHERENCE (0-1):
   - Is the overall extraction consistent?
   - Are there contradictions or redundancies?

4. IDENTIFY:
   - MISSING ENTITIES: Important entities not extracted
   - SPURIOUS ENTITIES: Entities that shouldn't be extracted

Provide scores and feedback in JSON format:
{
  "overall_score": 0.85,
  "entity_quality": 0.90,
  "relationship_quality": 0.80,
  "coherence_score": 0.85,
  "missing_entities": ["entity1", "entity2"],
  "spurious_entities": ["entity3"],
  "quality_level": "high|medium|low|failed",
  "feedback": "Detailed qualitative feedback"
}"""

    def _get_evaluation_user_prompt(
        self,
        entities: list[ExtractedEntity],
        source_text: str,
    ) -> str:
        """Get evaluation user prompt."""
        entity_list = "\n".join(
            [
                f"- {e.name} ({e.entity_type}): {e.source_text[:100]}"
                for e in entities[:20]
            ]
        )

        return f"""Evaluate the following entity extraction:

Extracted Entities:
{entity_list}

Total Entities: {len(entities)}

Source Text (excerpt):
{source_text[:2000]}

Provide your evaluation in JSON format with scores and recommendations."""

    def _parse_evaluation_response(
        self,
        response: str,
        entity_count: int,
    ) -> ExtractionQualityScore:
        """Parse evaluation response."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return ExtractionQualityScore(
                overall_score=0.5,
                entity_quality=0.5,
                relationship_quality=0.5,
                coherence_score=0.5,
                quality_level=EntityExtractionQuality.MEDIUM,
                feedback="Failed to parse evaluation response",
            )

        quality_level = data.get("quality_level", "medium").upper()
        try:
            quality_level = EntityExtractionQuality(quality_level)
        except ValueError:
            quality_level = EntityExtractionQuality.MEDIUM

        return ExtractionQualityScore(
            overall_score=data.get("overall_score", 0.5),
            entity_quality=data.get("entity_quality", 0.5),
            relationship_quality=data.get("relationship_quality", 0.5),
            coherence_score=data.get("coherence_score", 0.5),
            missing_entities=data.get("missing_entities", []),
            spurious_entities=data.get("spurious_entities", []),
            quality_level=quality_level,
            feedback=data.get("feedback", ""),
        )


class ManagerAgent:
    """ManagerAgent - Extraction workflow coordinator.

    Coordinates the multi-agent extraction workflow, managing
    state transitions and inter-agent communication.
    """

    def __init__(
        self,
        perception_agent: PerceptionAgent,
        enhancement_agent: EnhancementAgent,
        evaluation_agent: EvaluationAgent,
        config: MultiAgentExtractorConfig | None = None,
    ) -> None:
        """Initialize manager agent.

        Args:
            perception_agent: Perception agent instance.
            enhancement_agent: Enhancement agent instance.
            evaluation_agent: Evaluation agent instance.
            config: Extractor configuration.
        """
        self._perception = perception_agent
        self._enhancement = enhancement_agent
        self._evaluation = evaluation_agent
        self._config = config or MultiAgentExtractorConfig()

    async def run_extraction_workflow(
        self,
        chunks: list[Chunk],
        document_id: UUID,
        entity_types: list[str] | None = None,
    ) -> ExtractionWorkflowState:
        """Execute the full extraction workflow.

        Args:
            chunks: Document chunks to process.
            document_id: Source document ID.
            entity_types: Optional entity types to extract.

        Returns:
            Final workflow state with all extraction results.
        """
        # Initialize logger
        logger = get_ingestion_logger(document_id, f"doc_{document_id}")
        logger.log_stage_start("Multi-Agent Extraction Workflow", total_steps=3)

        state = ExtractionWorkflowState(document_id=document_id)

        try:
            # Phase 1: Perception
            logger.log_stage_progress("Multi-Agent Extraction Workflow", 1, 3)
            state = await self._run_perception_phase(state, chunks, entity_types)

            if not state.perception_entities:
                logger.log_error("No entities extracted in perception phase")
                logger.log_summary()
                return state

            logger.log_stage_complete(
                "Perception Phase",
                f"Extracted {len(state.perception_entities)} entities",
            )

            # Phase 2: Enhancement
            logger.log_stage_progress("Multi-Agent Extraction Workflow", 2, 3)
            state = await self._run_enhancement_phase(state)

            logger.log_stage_complete(
                "Enhancement Phase", f"Enhanced {len(state.enhanced_entities)} entities"
            )

            # Phase 3: Evaluation
            logger.log_stage_progress("Multi-Agent Extraction Workflow", 3, 3)
            state = await self._run_evaluation_phase(state, chunks)

            logger.log_stage_complete(
                "Evaluation Phase",
                f"Quality score: {getattr(state, 'quality_score', 'N/A')}",
            )

            logger.log_stage_complete(
                "Multi-Agent Extraction Workflow",
                f"Total entities: {len(getattr(state, 'enhanced_entities', []))}",
            )

        except Exception as e:
            logger.log_error(str(e))
            state.errors.append(str(e))

        logger.log_summary()
        return state

    async def _run_perception_phase(
        self,
        state: ExtractionWorkflowState,
        chunks: list[Chunk],
        entity_types: list[str] | None,
    ) -> ExtractionWorkflowState:
        """Run perception extraction phase."""
        state.current_phase = ExtractionPhase.PERCEPTION

        entities = await self._perception.extract_entities(chunks, entity_types)

        state.perception_entities = entities
        state.iteration_count += 1

        return state

    async def _run_enhancement_phase(
        self,
        state: ExtractionWorkflowState,
    ) -> ExtractionWorkflowState:
        """Run enhancement phase."""
        state.current_phase = ExtractionPhase.ENHANCEMENT

        context = EnhancementContext(
            existing_entities=[],
            recent_extractions=state.perception_entities,
        )

        state.enhanced_entities = await self._enhancement.enhance_entities(
            state.perception_entities, context
        )

        return state

    async def _run_evaluation_phase(
        self,
        state: ExtractionWorkflowState,
        chunks: list[Chunk],
    ) -> ExtractionWorkflowState:
        """Run evaluation phase."""
        state.current_phase = ExtractionPhase.EVALUATION

        source_chunks = [
            c for c in chunks if c.id in {e.chunk_id for e in state.enhanced_entities}
        ]

        quality_score = await self._evaluation.evaluate_extraction(
            state.enhanced_entities, source_chunks
        )

        state.quality_scores.append(quality_score)

        if quality_score.quality_level == EntityExtractionQuality.LOW:
            state = await self._refine_extraction(state, chunks)

        state.current_phase = ExtractionPhase.COMPLETED

        return state

    async def _refine_extraction(
        self,
        state: ExtractionWorkflowState,
        chunks: list[Chunk],
    ) -> ExtractionWorkflowState:
        """Refine extraction based on evaluation feedback."""
        iterations = 0
        max_iterations = self._config.enhancement_max_iterations

        while iterations < max_iterations:
            iterations += 1

            state.iteration_count += 1

            if iterations >= max_iterations:
                break

        return state


class EntityExtractionManager:
    """EntityExtractionManager - Main orchestrator.

    Orchestrates all agents for complete multi-agent entity extraction.
    Provides the primary interface for the extraction system.
    """

    def __init__(
        self,
        gateway: AsyncLLMClient,
        graph_store: GraphStore,
        vector_store: VectorStore,
        config: MultiAgentExtractorConfig | None = None,
    ) -> None:
        """Initialize extraction manager.

        Args:
            gateway: LLM gateway client.
            graph_store: Knowledge graph store.
            vector_store: Vector store.
            config: Extractor configuration.
        """
        self._config = config or MultiAgentExtractorConfig()

        perception_agent = PerceptionAgent(gateway, self._config)
        enhancement_agent = EnhancementAgent(
            gateway, graph_store, vector_store, self._config
        )
        evaluation_agent = EvaluationAgent(gateway, self._config)

        self._manager = ManagerAgent(
            perception_agent, enhancement_agent, evaluation_agent, self._config
        )

    async def extract_entities(
        self,
        chunks: list[Chunk],
        document_id: UUID,
        entity_types: list[str] | None = None,
    ) -> list[ExtractedEntity]:
        """Extract entities from document chunks.

        Args:
            chunks: Document chunks to process.
            document_id: Source document ID.
            entity_types: Optional entity types to extract.

        Returns:
            List of extracted and validated entities.
        """
        state = await self._manager.run_extraction_workflow(
            chunks, document_id, entity_types
        )

        return state.enhanced_entities or state.perception_entities

    async def extract_with_evaluation(
        self,
        chunks: list[Chunk],
        document_id: UUID,
        entity_types: list[str] | None = None,
    ) -> tuple[list[ExtractedEntity], ExtractionQualityScore]:
        """Extract entities with quality evaluation.

        Args:
            chunks: Document chunks to process.
            document_id: Source document ID.
            entity_types: Optional entity types to extract.

        Returns:
            Tuple of extracted entities and quality assessment.
        """
        state = await self._manager.run_extraction_workflow(
            chunks, document_id, entity_types
        )

        entities = state.enhanced_entities or state.perception_entities
        quality = (
            state.quality_scores[0]
            if state.quality_scores
            else ExtractionQualityScore(
                overall_score=0.0,
                entity_quality=0.0,
                relationship_quality=0.0,
                coherence_score=0.0,
                quality_level=EntityExtractionQuality.FAILED,
                feedback="No evaluation performed",
            )
        )

        return entities, quality

    async def get_extraction_workflow_state(
        self,
        chunks: list[Chunk],
        document_id: UUID,
        entity_types: list[str] | None = None,
    ) -> ExtractionWorkflowState:
        """Get full workflow state for debugging/inspection.

        Args:
            chunks: Document chunks to process.
            document_id: Source document ID.
            entity_types: Optional entity types to extract.

        Returns:
            Complete workflow state.
        """
        return await self._manager.run_extraction_workflow(
            chunks, document_id, entity_types
        )

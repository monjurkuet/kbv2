"""Unit tests for multi-agent entity extraction system."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from knowledge_base.intelligence.v1.multi_agent_extractor import (
    EnhancementAgent,
    EnhancementContext,
    EntityCandidate,
    EntityExtractionManager,
    EntityExtractionQuality,
    EntityExtractionQuality,
    EvaluationAgent,
    ExtractedEntity,
    ExtractionPhase,
    ExtractionQualityScore,
    ExtractionWorkflowState,
    ManagerAgent,
    MultiAgentExtractorConfig,
    PerceptionAgent,
    PerceptionAgent,
)


class TestMultiAgentExtractorConfig:
    """Tests for MultiAgentExtractorConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = MultiAgentExtractorConfig()

        assert config.perception_temperature == 0.1
        assert config.enhancement_temperature == 0.2
        assert config.evaluation_temperature == 0.3
        assert config.max_entities_per_chunk == 50
        assert config.confidence_threshold == 0.7
        assert config.enhancement_max_iterations == 3
        assert config.evaluation_sample_rate == 0.2

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = MultiAgentExtractorConfig(
            perception_temperature=0.05,
            enhancement_temperature=0.15,
            evaluation_temperature=0.25,
            confidence_threshold=0.8,
        )

        assert config.perception_temperature == 0.05
        assert config.enhancement_temperature == 0.15
        assert config.evaluation_temperature == 0.25
        assert config.confidence_threshold == 0.8


class TestEntityCandidate:
    """Tests for EntityCandidate model."""

    def test_entity_candidate_creation(self):
        """Test creating an entity candidate."""
        candidate = EntityCandidate(
            text="John Smith",
            entity_type="PERSON",
            start_char=0,
            end_char=10,
            context="John Smith works at Acme Corp",
            confidence=0.95,
        )

        assert candidate.text == "John Smith"
        assert candidate.entity_type == "PERSON"
        assert candidate.start_char == 0
        assert candidate.end_char == 10
        assert candidate.confidence == 0.95

    def test_entity_candidate_with_boundary_type(self):
        """Test entity candidate with boundary type."""
        from knowledge_base.intelligence.v1.multi_agent_extractor import (
            EntityBoundaryType,
        )

        candidate = EntityCandidate(
            text="New York",
            entity_type="LOCATION",
            start_char=5,
            end_char=13,
            context="Located in New York",
            confidence=0.88,
            boundary_type=EntityBoundaryType.STRONG,
        )

        assert candidate.boundary_type == EntityBoundaryType.STRONG


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_extracted_entity_creation(self):
        """Test creating an extracted entity."""
        chunk_id = uuid4()
        entity = ExtractedEntity(
            name="Acme Corporation",
            entity_type="ORGANIZATION",
            source_text="Acme Corporation is a leading tech company",
            chunk_id=chunk_id,
            confidence=0.92,
        )

        assert entity.name == "Acme Corporation"
        assert entity.entity_type == "ORGANIZATION"
        assert entity.chunk_id == chunk_id
        assert entity.confidence == 0.92
        assert entity.extraction_phase == ExtractionPhase.PERCEPTION
        assert entity.linked_entities == []
        assert entity.is_cross_boundary is False

    def test_extracted_entity_with_properties(self):
        """Test extracted entity with properties."""
        entity = ExtractedEntity(
            name="2024-01-15",
            entity_type="DATE",
            source_text="Event scheduled for 2024-01-15",
            chunk_id=uuid4(),
            properties={"format": "ISO8601", "precision": "day"},
        )

        assert entity.properties["format"] == "ISO8601"
        assert entity.properties["precision"] == "day"

    def test_extracted_entity_default_id(self):
        """Test that entity ID is auto-generated."""
        entity1 = ExtractedEntity(
            name="Test Entity",
            entity_type="CONCEPT",
            source_text="Test source",
            chunk_id=uuid4(),
        )
        entity2 = ExtractedEntity(
            name="Test Entity",
            entity_type="CONCEPT",
            source_text="Test source",
            chunk_id=uuid4(),
        )

        assert entity1.id != entity2.id

    def test_extracted_entity_phase_transition(self):
        """Test entity phase transitions."""
        entity = ExtractedEntity(
            name="Test",
            entity_type="PERSON",
            source_text="Test",
            chunk_id=uuid4(),
        )

        entity.extraction_phase = ExtractionPhase.ENHANCEMENT
        assert entity.extraction_phase == ExtractionPhase.ENHANCEMENT

        entity.extraction_phase = ExtractionPhase.EVALUATION
        assert entity.extraction_phase == ExtractionPhase.EVALUATION


class TestExtractionQualityScore:
    """Tests for ExtractionQualityScore model."""

    def test_quality_score_creation(self):
        """Test creating a quality score."""
        score = ExtractionQualityScore(
            overall_score=0.85,
            entity_quality=0.90,
            relationship_quality=0.80,
            coherence_score=0.85,
            quality_level=EntityExtractionQuality.HIGH,
            feedback="Excellent extraction quality",
        )

        assert score.overall_score == 0.85
        assert score.entity_quality == 0.90
        assert score.quality_level == EntityExtractionQuality.HIGH

    def test_quality_score_bounds(self):
        """Test quality score bounds validation."""
        with pytest.raises(ValidationError):
            ExtractionQualityScore(
                overall_score=1.5,
                entity_quality=0.9,
                relationship_quality=0.8,
                coherence_score=0.85,
                quality_level=EntityExtractionQuality.MEDIUM,
            )

        with pytest.raises(ValidationError):
            ExtractionQualityScore(
                overall_score=-0.1,
                entity_quality=0.9,
                relationship_quality=0.8,
                coherence_score=0.85,
                quality_level=EntityExtractionQuality.MEDIUM,
            )

    def test_quality_level_enum(self):
        """Test quality level enum values."""
        assert EntityExtractionQuality.HIGH.value == "high"
        assert EntityExtractionQuality.MEDIUM.value == "medium"
        assert EntityExtractionQuality.LOW.value == "low"
        assert EntityExtractionQuality.FAILED.value == "failed"


class TestExtractionWorkflowState:
    """Tests for ExtractionWorkflowState model."""

    def test_workflow_state_creation(self):
        """Test creating workflow state."""
        document_id = uuid4()
        state = ExtractionWorkflowState(document_id=document_id)

        assert state.document_id == document_id
        assert state.current_phase == ExtractionPhase.PERCEPTION
        assert state.perception_entities == []
        assert state.enhanced_entities == []
        assert state.quality_scores == []
        assert state.iteration_count == 0
        assert state.errors == []

    def test_workflow_state_transitions(self):
        """Test workflow phase transitions."""
        state = ExtractionWorkflowState(document_id=uuid4())

        state.current_phase = ExtractionPhase.ENHANCEMENT
        assert state.current_phase == ExtractionPhase.ENHANCEMENT

        state.current_phase = ExtractionPhase.EVALUATION
        assert state.current_phase == ExtractionPhase.EVALUATION

        state.current_phase = ExtractionPhase.COMPLETED
        assert state.current_phase == ExtractionPhase.COMPLETED


class TestEnhancementContext:
    """Tests for EnhancementContext model."""

    def test_enhancement_context_creation(self):
        """Test creating enhancement context."""
        context = EnhancementContext()

        assert context.existing_entities == []
        assert context.recent_extractions == []
        assert context.schema_constraints == {}

    def test_enhancement_context_with_data(self):
        """Test enhancement context with data."""
        from knowledge_base.persistence.v1.schema import Entity

        existing_entity = Entity(
            id=uuid4(),
            name="Existing",
            entity_type="PERSON",
        )
        existing_entities = [existing_entity]
        recent_extractions = [
            ExtractedEntity(
                name="New Entity",
                entity_type="PERSON",
                source_text="New source",
                chunk_id=uuid4(),
            )
        ]

        context = EnhancementContext(
            existing_entities=existing_entities,
            recent_extractions=recent_extractions,
            schema_constraints={"types": ["PERSON", "ORGANIZATION"]},
        )

        assert len(context.existing_entities) == 1
        assert len(context.recent_extractions) == 1
        assert context.schema_constraints["types"] == ["PERSON", "ORGANIZATION"]


class TestPerceptionAgent:
    """Tests for PerceptionAgent."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='[{"text": "Test Entity", "type": "PERSON", "start_char": 0, "end_char": 11, "boundary_type": "strong", "context": "Test Entity was here", "confidence": 0.9}]'
        )
        return gateway

    @pytest.fixture
    def agent(self, mock_gateway):
        """Create perception agent."""
        config = MultiAgentExtractorConfig()
        return PerceptionAgent(mock_gateway, config)

    @pytest.mark.asyncio
    async def test_extract_entities_basic(self, agent, mock_gateway):
        """Test basic entity extraction."""
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "John Smith works at Acme Corp"

        entities = await agent.extract_entities([chunk])

        assert len(entities) == 1
        assert entities[0].name == "Test Entity"
        assert entities[0].entity_type == "PERSON"
        mock_gateway.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_entities_with_types(self, agent, mock_gateway):
        """Test entity extraction with specific types."""
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "John Smith works at Acme Corp"

        entities = await agent.extract_entities([chunk], entity_types=["PERSON"])

        assert len(entities) == 1
        mock_gateway.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_entities_handles_invalid_json(self, agent, mock_gateway):
        """Test handling of invalid JSON response."""
        mock_gateway.generate_text = AsyncMock(return_value="Invalid JSON")
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "Test text"

        entities = await agent.extract_entities([chunk])

        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_entities_handles_empty_chunks(self, agent):
        """Test extraction with empty chunks."""
        entities = await agent.extract_entities([])
        assert entities == []


class TestEnhancementAgent:
    """Tests for EnhancementAgent."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='{"refined_name": "John Smith", "refined_type": "PERSON", "description": "A person named John Smith", "properties": {"age": "30"}, "confidence_adjustment": 0.05}'
        )
        return gateway

    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_gateway, mock_graph_store, mock_vector_store):
        """Create enhancement agent."""
        config = MultiAgentExtractorConfig()
        return EnhancementAgent(
            mock_gateway, mock_graph_store, mock_vector_store, config
        )

    @pytest.mark.asyncio
    async def test_enhance_entities_basic(self, agent, mock_gateway):
        """Test basic entity enhancement."""
        entity = ExtractedEntity(
            name="John Smith",
            entity_type="PERSON",
            source_text="John Smith works here",
            chunk_id=uuid4(),
            confidence=0.85,
        )
        context = EnhancementContext()

        enhanced = await agent.enhance_entities([entity], context)

        assert len(enhanced) == 1
        assert enhanced[0].name == "John Smith"
        assert enhanced[0].entity_type == "PERSON"
        mock_gateway.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_entities_empty_list(self, agent):
        """Test enhancement with empty entity list."""
        enhanced = await agent.enhance_entities([], EnhancementContext())
        assert enhanced == []

    @pytest.mark.asyncio
    async def test_enhance_entities_with_invalid_json(self, agent, mock_gateway):
        """Test enhancement handling invalid JSON."""
        mock_gateway.generate_text = AsyncMock(return_value="Invalid JSON")
        entity = ExtractedEntity(
            name="Test",
            entity_type="PERSON",
            source_text="Test",
            chunk_id=uuid4(),
        )

        enhanced = await agent.enhance_entities([entity], EnhancementContext())

        assert len(enhanced) == 1
        assert enhanced[0].name == "Test"

    def test_should_link_similar_names(self, agent):
        """Test entity linking with similar names."""
        entity = ExtractedEntity(
            name="John Smith",
            entity_type="PERSON",
            source_text="Test",
            chunk_id=uuid4(),
        )
        existing = MagicMock()
        existing.id = uuid4()
        existing.name = "John Smith"
        existing.entity_type = "PERSON"

        assert agent._should_link(entity, existing) is True

    def test_should_link_different_names(self, agent):
        """Test entity not linking with different names."""
        entity = ExtractedEntity(
            name="John Smith",
            entity_type="PERSON",
            source_text="Test",
            chunk_id=uuid4(),
        )
        existing = MagicMock()
        existing.id = uuid4()
        existing.name = "Jane Doe"
        existing.entity_type = "PERSON"

        assert agent._should_link(entity, existing) is False

    def test_calculate_name_similarity_exact(self, agent):
        """Test name similarity for exact matches."""
        similarity = agent._calculate_name_similarity("John Smith", "John Smith")
        assert similarity == 1.0

    def test_calculate_name_similarity_containment(self, agent):
        """Test name similarity for containment."""
        similarity = agent._calculate_name_similarity("John", "John Smith")
        assert similarity == 0.9

    def test_calculate_name_similarity_different(self, agent):
        """Test name similarity for different names."""
        similarity = agent._calculate_name_similarity("John", "Jane")
        assert similarity == 0.0


class TestEvaluationAgent:
    """Tests for EvaluationAgent."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='{"overall_score": 0.85, "entity_quality": 0.90, "relationship_quality": 0.80, "coherence_score": 0.85, "missing_entities": [], "spurious_entities": [], "quality_level": "medium", "feedback": "Good extraction"}'
        )
        return gateway

    @pytest.fixture
    def agent(self, mock_gateway):
        """Create evaluation agent."""
        config = MultiAgentExtractorConfig()
        return EvaluationAgent(mock_gateway, config)

    @pytest.mark.asyncio
    async def test_evaluate_extraction_basic(self, agent, mock_gateway):
        """Test basic extraction evaluation."""
        entity = ExtractedEntity(
            name="Test Entity",
            entity_type="PERSON",
            source_text="Test source",
            chunk_id=uuid4(),
        )
        chunk = MagicMock()
        chunk.text = "Test source text"

        score = await agent.evaluate_extraction([entity], [chunk])

        assert score.overall_score == 0.85
        assert score.entity_quality == 0.90
        assert score.quality_level == EntityExtractionQuality.MEDIUM
        mock_gateway.generate_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_extraction_with_invalid_json(self, agent, mock_gateway):
        """Test evaluation handling invalid JSON."""
        mock_gateway.generate_text = AsyncMock(return_value="Invalid JSON")
        entity = ExtractedEntity(
            name="Test",
            entity_type="PERSON",
            source_text="Test",
            chunk_id=uuid4(),
        )
        chunk = MagicMock()
        chunk.text = "Test source"

        score = await agent.evaluate_extraction([entity], [chunk])

        assert score.overall_score == 0.5
        assert score.quality_level == EntityExtractionQuality.MEDIUM

    def test_parse_evaluation_response_invalid_level(self, agent):
        """Test parsing response with invalid quality level."""
        response = '{"overall_score": 0.85, "entity_quality": 0.9, "relationship_quality": 0.8, "coherence_score": 0.85, "quality_level": "invalid", "feedback": "Test"}'
        score = agent._parse_evaluation_response(response, 1)

        assert score.quality_level == EntityExtractionQuality.MEDIUM


class TestManagerAgent:
    """Tests for ManagerAgent."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='[{"text": "Test Entity", "type": "PERSON", "start_char": 0, "end_char": 11, "boundary_type": "strong", "context": "Test source", "confidence": 0.9}]'
        )
        return gateway

    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_gateway, mock_graph_store, mock_vector_store):
        """Create manager agent."""
        config = MultiAgentExtractorConfig()
        perception_agent = PerceptionAgent(mock_gateway, config)
        enhancement_agent = EnhancementAgent(
            mock_gateway, mock_graph_store, mock_vector_store, config
        )
        evaluation_agent = EvaluationAgent(mock_gateway, config)
        return ManagerAgent(
            perception_agent, enhancement_agent, evaluation_agent, config
        )

    @pytest.mark.asyncio
    async def test_run_extraction_workflow(
        self, mock_gateway, mock_graph_store, mock_vector_store
    ):
        """Test running extraction workflow."""
        mock_gateway.generate_text = AsyncMock(
            side_effect=[
                '[{"text": "Test Entity", "type": "PERSON", "start_char": 0, "end_char": 11, "boundary_type": "strong", "context": "Test source", "confidence": 0.9}]',
                '{"refined_name": "Test Entity", "refined_type": "PERSON", "description": "Enhanced entity", "properties": {}, "confidence_adjustment": 0.0}',
                '{"overall_score": 0.85, "entity_quality": 0.90, "relationship_quality": 0.80, "coherence_score": 0.85, "missing_entities": [], "spurious_entities": [], "quality_level": "high", "feedback": "Good extraction"}',
            ]
        )
        config = MultiAgentExtractorConfig()
        perception_agent = PerceptionAgent(mock_gateway, config)
        enhancement_agent = EnhancementAgent(
            mock_gateway, mock_graph_store, mock_vector_store, config
        )
        evaluation_agent = EvaluationAgent(mock_gateway, config)
        manager = ManagerAgent(
            perception_agent, enhancement_agent, evaluation_agent, config
        )

        document_id = uuid4()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "Test document content"

        state = await manager.run_extraction_workflow([chunk], document_id)

        assert state.document_id == document_id
        assert len(state.perception_entities) == 1
        assert len(state.enhanced_entities) == 1
        assert len(state.quality_scores) == 1

    @pytest.mark.asyncio
    async def test_run_workflow_empty_chunks(
        self, mock_gateway, mock_graph_store, mock_vector_store
    ):
        """Test workflow with empty chunks."""
        config = MultiAgentExtractorConfig()
        perception_agent = PerceptionAgent(mock_gateway, config)
        enhancement_agent = EnhancementAgent(
            mock_gateway, mock_graph_store, mock_vector_store, config
        )
        evaluation_agent = EvaluationAgent(mock_gateway, config)
        manager = ManagerAgent(
            perception_agent, enhancement_agent, evaluation_agent, config
        )

        state = await manager.run_extraction_workflow([], uuid4())

        assert state.perception_entities == []
        assert state.errors == []

    @pytest.mark.asyncio
    async def test_run_workflow_no_entities(self, manager, mock_gateway):
        """Test workflow when no entities extracted."""
        mock_gateway.generate_text = AsyncMock(return_value="[]")
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "Test"

        state = await manager.run_extraction_workflow([chunk], uuid4())

        assert state.current_phase == ExtractionPhase.PERCEPTION
        assert state.perception_entities == []


class TestEntityExtractionManager:
    """Tests for EntityExtractionManager."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='[{"text": "Test", "type": "PERSON", "start_char": 0, "end_char": 4, "boundary_type": "strong", "context": "Test context", "confidence": 0.9}]'
        )
        return gateway

    @pytest.fixture
    def mock_graph_store(self):
        """Create mock graph store."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        return MagicMock()

    @pytest.fixture
    def manager(self, mock_gateway, mock_graph_store, mock_vector_store):
        """Create extraction manager."""
        return EntityExtractionManager(
            mock_gateway, mock_graph_store, mock_vector_store
        )

    @pytest.mark.asyncio
    async def test_extract_entities(self, manager, mock_gateway):
        """Test entity extraction."""
        document_id = uuid4()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "John Smith works here"

        entities = await manager.extract_entities([chunk], document_id)

        assert len(entities) == 1
        assert entities[0].name == "Test"
        mock_gateway.generate_text.assert_called()

    @pytest.mark.asyncio
    async def test_extract_with_evaluation(
        self, mock_gateway, mock_graph_store, mock_vector_store
    ):
        """Test extraction with quality evaluation."""
        mock_gateway.generate_text = AsyncMock(
            side_effect=[
                '[{"text": "John Smith", "type": "PERSON", "start_char": 0, "end_char": 10, "boundary_type": "strong", "context": "John Smith works here", "confidence": 0.9}]',
                '{"refined_name": "John Smith", "refined_type": "PERSON", "description": "A person", "properties": {}, "confidence_adjustment": 0.0}',
                '{"overall_score": 0.85, "entity_quality": 0.90, "relationship_quality": 0.80, "coherence_score": 0.85, "missing_entities": [], "spurious_entities": [], "quality_level": "high", "feedback": "Good extraction"}',
            ]
        )
        manager = EntityExtractionManager(
            mock_gateway, mock_graph_store, mock_vector_store
        )
        document_id = uuid4()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "John Smith works here"

        entities, quality = await manager.extract_with_evaluation([chunk], document_id)

        assert len(entities) == 1
        assert quality.overall_score > 0
        assert isinstance(quality, ExtractionQualityScore)

    @pytest.mark.asyncio
    async def test_get_extraction_workflow_state(self, manager, mock_gateway):
        """Test getting workflow state."""
        document_id = uuid4()
        chunk = MagicMock()
        chunk.id = uuid4()
        chunk.text = "Test"

        state = await manager.get_extraction_workflow_state([chunk], document_id)

        assert isinstance(state, ExtractionWorkflowState)
        assert state.document_id == document_id


class TestExtractionPhase:
    """Tests for ExtractionPhase enum."""

    def test_phase_order(self):
        """Test extraction phase values."""
        assert ExtractionPhase.PERCEPTION.value == "perception"
        assert ExtractionPhase.ENHANCEMENT.value == "enhancement"
        assert ExtractionPhase.EVALUATION.value == "evaluation"
        assert ExtractionPhase.COMPLETED.value == "completed"


class TestEntityBoundaryType:
    """Tests for EntityBoundaryType enum."""

    def test_boundary_types(self):
        """Test boundary type values."""
        from knowledge_base.intelligence.v1.multi_agent_extractor import (
            EntityBoundaryType,
        )

        assert EntityBoundaryType.STRONG.value == "strong"
        assert EntityBoundaryType.WEAK.value == "weak"
        assert EntityBoundaryType.CROSSING.value == "crossing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

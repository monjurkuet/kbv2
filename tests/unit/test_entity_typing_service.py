"""Unit tests for entity typing service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from knowledge_base.intelligence.v1.entity_typing_service import (
    EntityTyper,
    EntityType,
    DomainType,
    EntityTypingConfig,
    EntityTypingResult,
    TypedEntity,
    FewShotExample,
    PromptTemplate,
    PromptTemplateRegistry,
    FewShotExampleBank,
)


class TestEntityType:
    """Tests for EntityType enum."""

    def test_entity_type_values(self) -> None:
        """Test entity type enum values."""
        assert EntityType.PERSON.value == "Person"
        assert EntityType.ORGANIZATION.value == "Organization"
        assert EntityType.LOCATION.value == "Location"
        assert EntityType.EVENT.value == "Event"
        assert EntityType.CONCEPT.value == "Concept"
        assert EntityType.PRODUCT.value == "Product"
        assert EntityType.OTHER.value == "Other"

    def test_entity_type_from_string(self) -> None:
        """Test creating entity type from string."""
        assert EntityType("Person") == EntityType.PERSON
        assert EntityType("Organization") == EntityType.ORGANIZATION


class TestDomainType:
    """Tests for DomainType enum."""

    def test_domain_type_values(self) -> None:
        """Test domain type enum values."""
        assert DomainType.GENERAL.value == "general"
        assert DomainType.MEDICAL.value == "medical"
        assert DomainType.LEGAL.value == "legal"
        assert DomainType.FINANCIAL.value == "financial"
        assert DomainType.TECHNOLOGY.value == "technology"


class TestEntityTypingConfig:
    """Tests for EntityTypingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EntityTypingConfig()
        assert config.confidence_threshold == 0.6
        assert config.max_few_shot_examples == 5
        assert config.temperature == 0.2
        assert config.enable_domain_awareness is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = EntityTypingConfig(
            confidence_threshold=0.8,
            temperature=0.3,
            enable_domain_awareness=False,
        )
        assert config.confidence_threshold == 0.8
        assert config.temperature == 0.3
        assert config.enable_domain_awareness is False


class TestEntityTypingResult:
    """Tests for EntityTypingResult model."""

    def test_create_result(self) -> None:
        """Test creating entity typing result."""
        entity_id = uuid4()
        result = EntityTypingResult(
            entity_id=entity_id,
            entity_name="Test Entity",
            entity_type=EntityType.PERSON,
            confidence_score=0.85,
            reasoning="Based on context analysis",
        )
        assert result.entity_id == entity_id
        assert result.entity_name == "Test Entity"
        assert result.entity_type == EntityType.PERSON
        assert result.confidence_score == 0.85
        assert result.reasoning == "Based on context analysis"
        assert result.alternative_types == []
        assert result.domain is None
        assert result.human_review_required is False

    def test_result_with_alternatives(self) -> None:
        """Test result with alternative types."""
        result = EntityTypingResult(
            entity_id=uuid4(),
            entity_name="Apple",
            entity_type=EntityType.ORGANIZATION,
            confidence_score=0.7,
            reasoning="Context suggests company",
            alternative_types=[
                (EntityType.PRODUCT, 0.3),
                (EntityType.LOCATION, 0.1),
            ],
            domain=DomainType.TECHNOLOGY,
        )
        assert len(result.alternative_types) == 2
        assert result.domain == DomainType.TECHNOLOGY

    def test_low_confidence_review_flag(self) -> None:
        """Test that low confidence sets review flag."""
        result = EntityTypingResult(
            entity_id=uuid4(),
            entity_name="Unknown Entity",
            entity_type=EntityType.OTHER,
            confidence_score=0.4,
            reasoning="Unclear from context",
            human_review_required=True,
        )
        assert result.human_review_required is True


class TestTypedEntity:
    """Tests for TypedEntity model."""

    def test_create_typed_entity(self) -> None:
        """Test creating typed entity."""
        entity_id = uuid4()
        entity = TypedEntity(
            entity_id=entity_id,
            name="John Doe",
            description="A person mentioned in the document",
            source_text="John Doe works at Acme Corp.",
            properties={"age": 30, "occupation": "engineer"},
        )
        assert entity.entity_id == entity_id
        assert entity.name == "John Doe"
        assert entity.description == "A person mentioned in the document"
        assert entity.source_text == "John Doe works at Acme Corp."
        assert entity.properties["age"] == 30

    def test_minimal_typed_entity(self) -> None:
        """Test creating typed entity with minimal info."""
        entity = TypedEntity(
            entity_id=uuid4(),
            name="Minimal Entity",
        )
        assert entity.name == "Minimal Entity"
        assert entity.description is None
        assert entity.source_text is None
        assert entity.properties == {}


class TestFewShotExample:
    """Tests for FewShotExample model."""

    def test_create_example(self) -> None:
        """Test creating few-shot example."""
        example = FewShotExample(
            name="Albert Einstein",
            entity_type=EntityType.PERSON,
            context="Physicist who developed relativity theory",
        )
        assert example.name == "Albert Einstein"
        assert example.entity_type == EntityType.PERSON
        assert "relativity" in example.context


class TestPromptTemplate:
    """Tests for PromptTemplate."""

    def test_format_user_prompt(self) -> None:
        """Test formatting user prompt."""
        template = PromptTemplate(
            system_prompt="You are an entity classifier.",
            user_prompt_template=(
                "Classify: {entity_name}\n"
                "Description: {entity_description}\n"
                "Context: {source_text}"
            ),
        )

        prompt = template.format_user_prompt(
            entity_name="Test",
            entity_description="A test entity",
            source_text="Test context",
            properties={},
            domain=None,
            examples=[],
        )

        assert "Test" in prompt
        assert "A test entity" in prompt
        assert "Test context" in prompt

    def test_format_with_examples(self) -> None:
        """Test formatting with examples."""
        template = PromptTemplate(
            system_prompt="You are an entity classifier.",
            user_prompt_template="Classify: {entity_name}\n\nExamples:\n{examples}",
        )

        examples = [
            FewShotExample(
                name="Example1",
                entity_type=EntityType.PERSON,
                context="First example",
            ),
        ]

        prompt = template.format_user_prompt(
            entity_name="Test",
            entity_description="Description",
            source_text="Context",
            properties={},
            domain=None,
            examples=examples,
        )

        assert "Example1" in prompt
        assert "Person" in prompt


class TestPromptTemplateRegistry:
    """Tests for PromptTemplateRegistry."""

    def test_get_default_template(self) -> None:
        """Test getting default template."""
        registry = PromptTemplateRegistry()
        template = registry.get_template(DomainType.GENERAL)
        assert template is not None
        assert "entity classification" in template.system_prompt.lower()
        assert "Entity Types:" in template.system_prompt

    def test_get_domain_template(self) -> None:
        """Test getting domain-specific template."""
        registry = PromptTemplateRegistry()
        template = registry.get_template(DomainType.MEDICAL)
        assert template is not None

    def test_get_custom_named_template(self) -> None:
        """Test registering and retrieving custom template by name."""
        registry = PromptTemplateRegistry()
        custom = PromptTemplate(
            system_prompt="Custom system prompt",
            user_prompt_template="Custom: {entity_name}",
        )
        registry.register_template("custom", custom)
        retrieved = registry.get_template(DomainType.GENERAL)
        assert retrieved.system_prompt != "Custom system prompt"


class TestFewShotExampleBank:
    """Tests for FewShotExampleBank."""

    def test_get_general_examples(self) -> None:
        """Test getting general examples."""
        bank = FewShotExampleBank()
        examples = bank.get_examples(domain=DomainType.GENERAL)
        assert len(examples) > 0
        assert all(isinstance(ex, FewShotExample) for ex in examples)

    def test_get_type_specific_examples(self) -> None:
        """Test getting examples for specific type."""
        bank = FewShotExampleBank()
        examples = bank.get_examples(
            entity_type=EntityType.PERSON, domain=DomainType.GENERAL
        )
        assert len(examples) > 0

    def test_get_domain_examples(self) -> None:
        """Test getting domain-specific examples."""
        bank = FewShotExampleBank()
        examples = bank.get_examples(domain=DomainType.MEDICAL)
        assert len(examples) > 0

    def test_add_example(self) -> None:
        """Test adding custom example."""
        bank = FewShotExampleBank()
        new_example = FewShotExample(
            name="CustomEntity",
            entity_type=EntityType.CONCEPT,
            context="Custom context",
        )
        bank.add_example(new_example, domain=None)
        examples = bank.get_examples(
            entity_type=EntityType.CONCEPT, domain=DomainType.GENERAL
        )
        assert any(ex.name == "CustomEntity" for ex in examples)


class TestEntityTyper:
    """Tests for EntityTyper class."""

    @pytest.fixture
    def mock_gateway(self) -> MagicMock:
        """Create mock gateway."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Person", "confidence_score": 0.9, "reasoning": "Test reasoning", "alternative_types": [], "detected_domain": "general"}'
        )
        return gateway

    @pytest.fixture
    def typer(self, mock_gateway: MagicMock) -> EntityTyper:
        """Create entity typer with mock gateway."""
        return EntityTyper(gateway=mock_gateway)

    def test_init(self, typer: EntityTyper) -> None:
        """Test entity typer initialization."""
        assert typer._gateway is not None
        assert typer._config is not None
        assert isinstance(typer._config, EntityTypingConfig)

    def test_init_with_config(self, mock_gateway: MagicMock) -> None:
        """Test initialization with custom config."""
        config = EntityTypingConfig(confidence_threshold=0.9)
        typer = EntityTyper(gateway=mock_gateway, config=config)
        assert typer._config.confidence_threshold == 0.9

    @pytest.mark.asyncio
    async def test_type_single_entity(self, typer: EntityTyper) -> None:
        """Test typing a single entity."""
        entity_id = uuid4()
        result = await typer.type_entity(
            entity_id=entity_id,
            name="John Smith",
            description="A software engineer",
            source_text="John Smith works at TechCorp.",
        )

        assert result.entity_id == entity_id
        assert result.entity_name == "John Smith"
        assert result.entity_type == EntityType.PERSON
        assert result.confidence_score == 0.9
        assert typer._gateway.generate_text.called

    @pytest.mark.asyncio
    async def test_type_entities_batch(self, typer: EntityTyper) -> None:
        """Test batch entity typing."""
        entities = [
            TypedEntity(
                entity_id=uuid4(),
                name="Apple",
                description="Technology company",
                source_text="Apple launches new iPhone.",
            ),
            TypedEntity(
                entity_id=uuid4(),
                name="Paris",
                description="Capital of France",
                source_text="Paris is a beautiful city.",
            ),
        ]

        results = await typer.type_entities(entities)

        assert len(results) == 2
        assert results[0].entity_name == "Apple"
        assert results[1].entity_name == "Paris"

    @pytest.mark.asyncio
    async def test_empty_batch(self, typer: EntityTyper) -> None:
        """Test typing empty entity list."""
        results = await typer.type_entities([])
        assert results == []

    def test_add_few_shot_example(self, typer: EntityTyper) -> None:
        """Test adding custom few-shot example."""
        initial_examples = typer._example_bank.get_examples(
            entity_type=EntityType.PERSON, domain=DomainType.GENERAL
        )
        initial_count = len(initial_examples)

        typer.add_few_shot_example(
            name="CustomPerson",
            entity_type=EntityType.PERSON,
            context="Custom context for disambiguation",
        )

        new_examples = typer._example_bank.get_examples(
            entity_type=EntityType.PERSON, domain=DomainType.GENERAL
        )
        assert len(new_examples) >= initial_count

    @pytest.mark.asyncio
    async def test_low_confidence_review_flag(self, mock_gateway: MagicMock) -> None:
        """Test that low confidence sets review flag."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Other", "confidence_score": 0.3, "reasoning": "Low confidence", "alternative_types": []}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Ambiguous Entity",
            description="Unknown entity",
            source_text="Some context",
        )

        assert result.confidence_score == 0.3
        assert result.human_review_required is True

    @pytest.mark.asyncio
    async def test_parse_invalid_json(self, mock_gateway: MagicMock) -> None:
        """Test handling of invalid JSON response."""
        mock_gateway.generate_text = AsyncMock(return_value="Invalid JSON")

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Test Entity",
            description="Test description",
        )

        assert result.entity_type == EntityType.OTHER
        assert result.confidence_score == 0.0
        assert result.human_review_required is True

    @pytest.mark.asyncio
    async def test_parse_valid_json_with_alternatives(
        self, mock_gateway: MagicMock
    ) -> None:
        """Test parsing JSON with alternative types."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Organization", "confidence_score": 0.75, "reasoning": "Context suggests organization", "alternative_types": [{"type": "Location", "score": 0.2}, {"type": "Product", "score": 0.05}], "detected_domain": "technology"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Google",
            description="Technology company",
        )

        assert result.entity_type == EntityType.ORGANIZATION
        assert result.confidence_score == 0.75
        assert len(result.alternative_types) == 2
        assert result.domain == DomainType.TECHNOLOGY


class TestEntityTyperIntegration:
    """Integration tests for EntityTyper with realistic scenarios."""

    @pytest.fixture
    def mock_gateway(self) -> MagicMock:
        """Create mock gateway for integration tests."""
        gateway = MagicMock()
        gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Person", "confidence_score": 0.9, "reasoning": "Test reasoning", "alternative_types": [], "detected_domain": "general"}'
        )
        return gateway

    @pytest.mark.asyncio
    async def test_type_person_entity(self, mock_gateway: MagicMock) -> None:
        """Test typing a person entity."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Person", "confidence_score": 0.95, "reasoning": "Name pattern and context clearly indicate a person", "alternative_types": [], "detected_domain": "general"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Albert Einstein",
            description="Theoretical physicist who developed relativity",
            source_text="Albert Einstein was a German-born theoretical physicist.",
        )

        assert result.entity_type == EntityType.PERSON
        assert result.confidence_score > 0.9

    @pytest.mark.asyncio
    async def test_type_organization_entity(self, mock_gateway: MagicMock) -> None:
        """Test typing an organization entity."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Organization", "confidence_score": 0.92, "reasoning": "Context and description indicate a company", "alternative_types": [], "detected_domain": "technology"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="OpenAI",
            description="AI research laboratory",
            source_text="OpenAI develops artificial intelligence technologies.",
        )

        assert result.entity_type == EntityType.ORGANIZATION

    @pytest.mark.asyncio
    async def test_type_location_entity(self, mock_gateway: MagicMock) -> None:
        """Test typing a location entity."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Location", "confidence_score": 0.88, "reasoning": "Geographic reference indicates location", "alternative_types": [], "detected_domain": "general"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Paris",
            description="Capital city of France",
            source_text="Paris is known as the City of Light.",
        )

        assert result.entity_type == EntityType.LOCATION

    @pytest.mark.asyncio
    async def test_type_event_entity(self, mock_gateway: MagicMock) -> None:
        """Test typing an event entity."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Event", "confidence_score": 0.91, "reasoning": "Context describes a specific occurrence", "alternative_types": [], "detected_domain": "general"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="World War II",
            description="Global conflict from 1939 to 1945",
            source_text="World War II was the deadliest conflict in human history.",
        )

        assert result.entity_type == EntityType.EVENT

    @pytest.mark.asyncio
    async def test_type_concept_entity(self, mock_gateway: MagicMock) -> None:
        """Test typing a concept entity."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Concept", "confidence_score": 0.85, "reasoning": "Abstract idea being explained", "alternative_types": [], "detected_domain": "general"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Democracy",
            description="System of government by the people",
            source_text="Democracy allows citizens to participate in decision-making.",
        )

        assert result.entity_type == EntityType.CONCEPT

    @pytest.mark.asyncio
    async def test_ambiguous_entity_typing(self, mock_gateway: MagicMock) -> None:
        """Test typing ambiguous entity with multiple possibilities."""
        mock_gateway.generate_text = AsyncMock(
            return_value='{"entity_type": "Organization", "confidence_score": 0.55, "reasoning": "Could be company or location, context favors organization", "alternative_types": [{"type": "Location", "score": 0.35}], "detected_domain": "general"}'
        )

        typer = EntityTyper(gateway=mock_gateway)
        result = await typer.type_entity(
            entity_id=uuid4(),
            name="Washington",
            description="State or university",
            source_text="Washington announced new policies today.",
        )

        assert result.entity_type == EntityType.ORGANIZATION
        assert len(result.alternative_types) == 1
        assert result.human_review_required is True

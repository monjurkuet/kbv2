"""Unit tests for guided extraction."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_base.extraction.guided_extractor import (
    GuidedExtractor,
    ExtractionPrompt,
    ExtractionPrompts,
)
from knowledge_base.extraction.template_registry import (
    ExtractionGoal,
    TemplateRegistry,
)


class TestExtractionPrompt:
    """Tests for ExtractionPrompt model."""

    def test_prompt_creation(self):
        """Test creating an extraction prompt."""
        prompt = ExtractionPrompt(
            goal_name="test_goal",
            system_prompt="You are an extractor.",
            user_prompt="Extract from this text.",
            target_entities=["Person", "Organization"],
            target_relationships=["related_to"],
            priority=1,
        )
        assert prompt.goal_name == "test_goal"
        assert prompt.priority == 1
        assert len(prompt.target_entities) == 2

    def test_prompt_json_serialization(self):
        """Test prompt can be serialized to JSON."""
        prompt = ExtractionPrompt(
            goal_name="test",
            system_prompt="System",
            user_prompt="User",
            target_entities=["E"],
            target_relationships=["r"],
            priority=2,
        )
        json_data = prompt.model_dump()
        assert json_data["goal_name"] == "test"
        assert json_data["priority"] == 2


class TestExtractionPrompts:
    """Tests for ExtractionPrompts collection."""

    def test_prompts_creation(self):
        """Test creating extraction prompts collection."""
        prompts = ExtractionPrompts(
            domain="TECHNOLOGY",
            prompts=[
                ExtractionPrompt(
                    goal_name="software",
                    system_prompt="S",
                    user_prompt="U",
                    target_entities=["Software"],
                    target_relationships=["uses"],
                    priority=1,
                )
            ],
        )
        assert prompts.domain == "TECHNOLOGY"
        assert len(prompts.prompts) == 1

    def test_prompts_user_goals_flag(self):
        """Test user_goals_used flag."""
        auto_prompts = ExtractionPrompts(
            domain="GENERAL", prompts=[], user_goals_used=False
        )
        assert auto_prompts.user_goals_used is False

        user_prompts = ExtractionPrompts(
            domain="TECHNOLOGY",
            prompts=[],
            user_goals_used=True,
            original_user_goals=["software_systems"],
        )
        assert user_prompts.user_goals_used is True
        assert user_prompts.original_user_goals == ["software_systems"]


class TestGuidedExtractor:
    """Tests for GuidedExtractor class."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway client."""
        gateway = AsyncMock()
        return gateway

    @pytest.fixture
    def extractor(self, mock_gateway):
        """Create a guided extractor with mock gateway."""
        registry = TemplateRegistry()
        return GuidedExtractor(llm_client=mock_gateway, template_registry=registry)

    def test_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor.llm is not None
        assert extractor.templates is not None

    def test_initialization_with_custom_registry(self, mock_gateway):
        """Test extractor with custom template registry."""
        custom_registry = TemplateRegistry()
        extractor = GuidedExtractor(
            llm_client=mock_gateway, template_registry=custom_registry
        )
        assert extractor.templates is custom_registry

    @pytest.mark.asyncio
    async def test_generate_extraction_prompts_auto_mode(self, extractor):
        """Test auto mode generates prompts without user goals."""
        text = "The Python programming language is used for web development."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain="TECHNOLOGY"
        )

        assert prompts.domain == "TECHNOLOGY"
        assert len(prompts.prompts) > 0
        assert prompts.user_goals_used is False

    @pytest.mark.asyncio
    async def test_generate_extraction_prompts_user_mode(self, extractor):
        """Test user-guided mode generates prompts from user goals."""
        text = "The Python programming language is used for web development."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["software_systems"],
            domain="TECHNOLOGY",
        )

        assert prompts.user_goals_used is True
        assert "software_systems" in prompts.original_user_goals

    @pytest.mark.asyncio
    async def test_auto_mode_detects_domain_from_content(self, extractor, mock_gateway):
        """Test auto mode detects domain from content."""
        tech_text = "The Python API integrates with the Kubernetes database server."

        prompts = await extractor.generate_extraction_prompts(
            document_text=tech_text, user_goals=None, domain=None
        )

        assert prompts.domain in ["TECHNOLOGY", "GENERAL"]

    @pytest.mark.asyncio
    async def test_user_mode_interprets_matching_goals(self, extractor):
        """Test user mode matches goals correctly."""
        text = "Python is a programming language."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=["software_systems"], domain="TECHNOLOGY"
        )

        assert len(prompts.prompts) > 0
        goal_names = [p.goal_name for p in prompts.prompts]
        assert any("software" in name for name in goal_names)

    @pytest.mark.asyncio
    async def test_user_mode_fallback_to_defaults_on_no_match(self, extractor):
        """Test fallback to defaults when no goals match."""
        text = "Some random text."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["nonexistent_goal_name"],
            domain="TECHNOLOGY",
        )

        assert len(prompts.prompts) > 0

    @pytest.mark.asyncio
    async def test_auto_mode_returns_prioritized_prompts(self, extractor):
        """Test auto mode returns prompts sorted by priority."""
        text = "Python is used with PostgreSQL."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain="TECHNOLOGY"
        )

        priorities = [p.priority for p in prompts.prompts]
        assert priorities == sorted(priorities)

    def test_build_system_prompt_includes_domain(self, extractor):
        """Test system prompt includes domain information."""
        goal = ExtractionGoal(
            name="test_goal",
            description="Test description",
            target_entities=["Entity"],
            target_relationships=["rel"],
            priority=1,
            examples=["Example"],
        )

        prompt = extractor._build_system_prompt(goal, "TECHNOLOGY")

        assert "TECHNOLOGY" in prompt
        assert "test_goal" in prompt
        assert "Test description" in prompt
        assert "Entity" in prompt

    def test_build_system_prompt_includes_examples(self, extractor):
        """Test system prompt includes example entities."""
        goal = ExtractionGoal(
            name="test_goal",
            description="Test",
            target_entities=["Person"],
            target_relationships=["knows"],
            priority=1,
            examples=["John", "Jane"],
        )

        prompt = extractor._build_system_prompt(goal, "GENERAL")

        assert "John" in prompt
        assert "Jane" in prompt

    def test_build_user_prompt_includes_document_text(self, extractor):
        """Test user prompt includes document text."""
        goal = ExtractionGoal(
            name="test",
            description="Test",
            target_entities=["E"],
            target_relationships=["r"],
            priority=1,
        )

        text = "This is the document text to extract from."
        prompt = extractor._build_user_prompt(text, goal, None)

        assert text in prompt

    def test_build_user_prompt_includes_previous_entities(self, extractor):
        """Test user prompt includes previous extraction context."""
        from knowledge_base.ingestion.v1.gleaning_service import (
            ExtractionResult,
            ExtractedEntity,
        )

        goal = ExtractionGoal(
            name="test",
            description="Test",
            target_entities=["E"],
            target_relationships=["r"],
            priority=1,
        )

        previous = ExtractionResult(
            entities=[ExtractedEntity(name="Previous Entity", entity_type="PREV")]
        )

        text = "New text."
        prompt = extractor._build_user_prompt(text, goal, previous)

        assert "Previous Entity" in prompt

    def test_build_extraction_schema(self, extractor):
        """Test building extraction JSON schema."""
        goals = [
            ExtractionGoal(
                name="goal1",
                description="G1",
                target_entities=["Person", "Organization"],
                target_relationships=["related_to"],
                priority=1,
            ),
            ExtractionGoal(
                name="goal2",
                description="G2",
                target_entities=["Location"],
                target_relationships=["located_in"],
                priority=2,
            ),
        ]

        schema = extractor._build_extraction_schema(goals)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "entities" in schema["properties"]
        assert "edges" in schema["properties"]

    @pytest.mark.asyncio
    async def test_detect_domain_technology(self, extractor):
        """Test technology domain detection."""
        text = "The Python API uses the Kubernetes database server."

        domain = await extractor._detect_domain(text)

        assert domain == "TECHNOLOGY"

    @pytest.mark.asyncio
    async def test_detect_domain_financial(self, extractor):
        """Test financial domain detection."""
        text = "The company's revenue grew by 25% this quarter."

        domain = await extractor._detect_domain(text)

        assert domain == "FINANCIAL"

    @pytest.mark.asyncio
    async def test_detect_domain_medical(self, extractor):
        """Test medical domain detection."""
        text = "The patient was diagnosed with diabetes and prescribed treatment."

        domain = await extractor._detect_domain(text)

        assert domain == "MEDICAL"

    @pytest.mark.asyncio
    async def test_detect_domain_general_for_unknown(self, extractor):
        """Test general domain for unknown content."""
        text = "This is a random sentence with no domain indicators."

        domain = await extractor._detect_domain(text)

        assert domain == "GENERAL"

    @pytest.mark.asyncio
    async def test_extract_with_prompts_executes_all_prompts(
        self, extractor, mock_gateway
    ):
        """Test extraction executes all prompts."""
        mock_gateway.generate_text.return_value = """{
            "entities": [{"name": "Test", "type": "TEST", "confidence": 0.9}],
            "edges": [],
            "information_density": 0.5
        }"""

        prompts = ExtractionPrompts(
            domain="TECHNOLOGY",
            prompts=[
                ExtractionPrompt(
                    goal_name="test_goal",
                    system_prompt="System",
                    user_prompt="Extract",
                    target_entities=["TEST"],
                    target_relationships=["rel"],
                    priority=1,
                )
            ],
        )

        results = await extractor.extract_with_prompts(prompts)

        assert len(results) == 1
        assert mock_gateway.generate_text.call_count == 1

    @pytest.mark.asyncio
    async def test_parse_extraction_result(self, extractor):
        """Test parsing extraction result from LLM response."""
        response = """{
            "entities": [
                {"name": "Python", "type": "Software", "description": "Language", "confidence": 0.95}
            ],
            "edges": [
                {"source": "Python", "target": "Web", "type": "used_for", "confidence": 0.9}
            ],
            "information_density": 0.7
        }"""

        result = extractor._parse_extraction_result(response, "test_goal")

        assert len(result.entities) == 1
        assert result.entities[0].name == "Python"
        assert result.information_density == 0.7

    @pytest.mark.asyncio
    async def test_parse_extraction_result_handles_invalid_json(self, extractor):
        """Test parsing handles invalid JSON gracefully."""
        response = "Invalid JSON { broken"

        result = extractor._parse_extraction_result(response, "test_goal")

        assert len(result.entities) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_parse_extraction_result_handles_markdown_code_blocks(
        self, extractor
    ):
        """Test parsing handles markdown code blocks."""
        response = """```json
{
    "entities": [{"name": "Test", "type": "TEST", "confidence": 0.9}],
    "edges": [],
    "information_density": 0.5
}
```"""

        result = extractor._parse_extraction_result(response, "test_goal")

        assert len(result.entities) == 1


class TestGuidedExtractorAutoMode:
    """Tests specifically for automated (user_goals=None) mode."""

    @pytest.fixture
    def mock_gateway(self):
        return AsyncMock()

    @pytest.fixture
    def extractor(self, mock_gateway):
        return GuidedExtractor(llm_client=mock_gateway)

    @pytest.mark.asyncio
    async def test_auto_mode_uses_domain_defaults(self, extractor):
        """Test auto mode uses domain-specific default goals."""
        text = "Python uses PostgreSQL database."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain="TECHNOLOGY"
        )

        assert prompts.user_goals_used is False
        assert prompts.domain == "TECHNOLOGY"
        assert len(prompts.prompts) > 0

    @pytest.mark.asyncio
    async def test_auto_mode_falls_back_to_general_for_unknown_domain(self, extractor):
        """Test auto mode falls back to GENERAL for unknown domain."""
        text = "Random text without domain indicators."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain="UNKNOWN"
        )

        assert prompts.domain == "UNKNOWN"
        assert len(prompts.prompts) > 0

    @pytest.mark.asyncio
    async def test_auto_mode_does_not_require_domain_parameter(self, extractor):
        """Test auto mode can detect domain from content."""
        text = "The API connects to the cloud server using Python."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain=None
        )

        assert prompts.domain is not None


class TestGuidedExtractorUserMode:
    """Tests specifically for user-guided (user_goals provided) mode."""

    @pytest.fixture
    def mock_gateway(self):
        return AsyncMock()

    @pytest.fixture
    def extractor(self, mock_gateway):
        return GuidedExtractor(llm_client=mock_gateway)

    @pytest.mark.asyncio
    async def test_user_mode_uses_user_goals(self, extractor):
        """Test user mode uses the provided user goals."""
        text = "Python software is used."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["software_systems"],
            domain="TECHNOLOGY",
        )

        assert prompts.user_goals_used is True
        assert "software_systems" in prompts.original_user_goals

    @pytest.mark.asyncio
    async def test_user_mode_matches_exact_goal_name(self, extractor):
        """Test user mode matches exact goal names."""
        text = "Technology content."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["architecture"],
            domain="TECHNOLOGY",
        )

        goal_names = [p.goal_name for p in prompts.prompts]
        assert "architecture" in goal_names or any(
            "architecture" in name for name in goal_names
        )

    @pytest.mark.asyncio
    async def test_user_mode_creates_general_prompt_for_unmatched(self, extractor):
        """Test user mode creates general extraction for unmatched goals."""
        text = "Some content."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["completely_unknown_goal"],
            domain="TECHNOLOGY",
        )

        assert len(prompts.prompts) > 0

    @pytest.mark.asyncio
    async def test_user_mode_multiple_goals(self, extractor):
        """Test user mode handles multiple goals."""
        text = "Python and financial data."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text,
            user_goals=["software_systems", "financial_metrics"],
            domain=None,
        )

        assert len(prompts.prompts) >= 1


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing extraction."""

    @pytest.fixture
    def mock_gateway(self):
        return AsyncMock()

    def test_extractor_works_without_template_registry(self, mock_gateway):
        """Test extractor can be created without explicit template registry."""
        extractor = GuidedExtractor(llm_client=mock_gateway)
        assert extractor.templates is not None

    def test_prompts_model_backward_compatible(self):
        """Test ExtractionPrompts model is backward compatible."""
        prompts = ExtractionPrompts(domain="GENERAL", prompts=[])
        assert prompts.domain == "GENERAL"
        assert prompts.prompts == []
        assert prompts.user_goals_used is False

    @pytest.mark.asyncio
    async def test_extract_without_user_goals_uses_auto_mode(self, mock_gateway):
        """Test extraction without user goals uses automated mode."""
        extractor = GuidedExtractor(llm_client=mock_gateway)
        text = "Python API."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=None, domain="TECHNOLOGY"
        )

        assert prompts.user_goals_used is False

    @pytest.mark.asyncio
    async def test_extract_with_empty_user_goals_uses_auto_mode(self, mock_gateway):
        """Test extraction with empty user goals list uses auto mode."""
        extractor = GuidedExtractor(llm_client=mock_gateway)
        text = "Python API."

        prompts = await extractor.generate_extraction_prompts(
            document_text=text, user_goals=[], domain="TECHNOLOGY"
        )

        assert prompts.user_goals_used is False

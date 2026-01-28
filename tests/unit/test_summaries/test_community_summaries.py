"""Unit tests for community_summaries module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.knowledge_base.summaries.community_summaries import (
    HierarchyLevel,
    CommunitySummary,
    CommunityNamingResult,
    MultiLevelSummary,
    CommunitySummarizer,
)


class TestHierarchyLevel:
    """Tests for HierarchyLevel enum."""

    def test_hierarchy_level_values(self):
        """Test that hierarchy levels have correct string values."""
        assert HierarchyLevel.MACRO.value == "macro"
        assert HierarchyLevel.MESO.value == "meso"
        assert HierarchyLevel.MICRO.value == "micro"
        assert HierarchyLevel.NANO.value == "nano"

    def test_hierarchy_level_is_string_enum(self):
        """Test that HierarchyLevel is a string enum."""
        assert isinstance(HierarchyLevel.MACRO, str)
        assert HierarchyLevel.MACRO == "macro"


class TestCommunitySummary:
    """Tests for CommunitySummary model."""

    def test_community_summary_creation(self):
        """Test basic CommunitySummary creation."""
        summary = CommunitySummary(
            community_id="test-id",
            name="Test Community",
            level=HierarchyLevel.MACRO,
            summary="A test community summary",
            key_entities=["entity1", "entity2"],
            key_relationships=["rel1", "rel2"],
            entity_count=10,
            coherence_score=0.85,
        )

        assert summary.community_id == "test-id"
        assert summary.name == "Test Community"
        assert summary.level == HierarchyLevel.MACRO
        assert summary.summary == "A test community summary"
        assert summary.key_entities == ["entity1", "entity2"]
        assert summary.key_relationships == ["rel1", "rel2"]
        assert summary.entity_count == 10
        assert summary.coherence_score == 0.85
        assert isinstance(summary.generated_at, datetime)

    def test_community_summary_defaults(self):
        """Test CommunitySummary default values."""
        summary = CommunitySummary(
            community_id="test-id",
            name="Test",
            level=HierarchyLevel.MICRO,
            summary="Summary",
        )

        assert summary.key_entities == []
        assert summary.key_relationships == []
        assert summary.parent_community_id is None
        assert summary.child_community_ids == []
        assert summary.entity_count == 0
        assert summary.coherence_score == 0.0

    def test_community_summary_parent_child_relationship(self):
        """Test parent and child community IDs."""
        parent = CommunitySummary(
            community_id="parent-1",
            name="Parent",
            level=HierarchyLevel.MACRO,
            summary="Parent summary",
        )
        child = CommunitySummary(
            community_id="child-1",
            name="Child",
            level=HierarchyLevel.MESO,
            summary="Child summary",
            parent_community_id="parent-1",
            child_community_ids=["grandchild-1"],
        )

        assert parent.community_id == "parent-1"
        assert child.parent_community_id == "parent-1"
        assert child.child_community_ids == ["grandchild-1"]


class TestCommunityNamingResult:
    """Tests for CommunityNamingResult model."""

    def test_community_naming_result_creation(self):
        """Test CommunityNamingResult creation."""
        result = CommunityNamingResult(
            suggested_name="Machine Learning Systems",
            description="Communities related to ML",
            key_themes=["neural networks", "deep learning"],
            confidence=0.92,
        )

        assert result.suggested_name == "Machine Learning Systems"
        assert result.description == "Communities related to ML"
        assert result.key_themes == ["neural networks", "deep learning"]
        assert result.confidence == 0.92


class TestMultiLevelSummary:
    """Tests for MultiLevelSummary model."""

    def test_multi_level_summary_creation(self):
        """Test MultiLevelSummary creation with all levels."""
        macro = CommunitySummary(
            community_id="macro-1",
            name="Macro 1",
            level=HierarchyLevel.MACRO,
            summary="Macro summary",
        )
        meso = CommunitySummary(
            community_id="meso-1",
            name="Meso 1",
            level=HierarchyLevel.MESO,
            summary="Meso summary",
        )
        micro = CommunitySummary(
            community_id="micro-1",
            name="Micro 1",
            level=HierarchyLevel.MICRO,
            summary="Micro summary",
        )

        summary = MultiLevelSummary(
            document_id="doc-123",
            macro_communities=[macro],
            meso_communities=[meso],
            micro_communities=[micro],
            hierarchy_tree={"type": "hierarchy"},
        )

        assert summary.document_id == "doc-123"
        assert len(summary.macro_communities) == 1
        assert len(summary.meso_communities) == 1
        assert len(summary.micro_communities) == 1
        assert summary.hierarchy_tree["type"] == "hierarchy"

    def test_multi_level_summary_defaults(self):
        """Test MultiLevelSummary default values."""
        summary = MultiLevelSummary(document_id="doc-1")

        assert summary.macro_communities == []
        assert summary.meso_communities == []
        assert summary.micro_communities == []
        assert summary.hierarchy_tree == {}


class TestCommunitySummarizer:
    """Tests for CommunitySummarizer class."""

    def test_summarizer_initialization_without_llm(self):
        """Test summarizer initialization without LLM client."""
        summarizer = CommunitySummarizer()
        assert summarizer.llm is None

    def test_summarizer_initialization_with_llm(self):
        """Test summarizer initialization with LLM client."""
        mock_llm = MagicMock()
        summarizer = CommunitySummarizer(llm_client=mock_llm)
        assert summarizer.llm is mock_llm

    def test_group_entities_by_community(self):
        """Test grouping entities by community ID."""
        summarizer = CommunitySummarizer()
        entities = [
            {"id": "e1", "name": "Entity1", "community_id": "comm-1"},
            {"id": "e2", "name": "Entity2", "community_id": "comm-1"},
            {"id": "e3", "name": "Entity3", "community_id": "comm-2"},
            {"id": "e4", "name": "Entity4", "community_id": None},
        ]

        grouped = summarizer._group_entities_by_community(entities)

        assert "comm-1" in grouped
        assert "comm-2" in grouped
        assert "unassigned" in grouped
        assert len(grouped["comm-1"]) == 2
        assert len(grouped["comm-2"]) == 1
        assert len(grouped["unassigned"]) == 1

    def test_group_entities_empty_list(self):
        """Test grouping empty entity list."""
        summarizer = CommunitySummarizer()
        grouped = summarizer._group_entities_by_community([])
        assert grouped == {}

    def test_group_edges_by_community(self):
        """Test grouping edges by community ID."""
        summarizer = CommunitySummarizer()
        edges = [
            {"id": "edge1", "source": "e1", "target": "e2", "community_id": "comm-1"},
            {"id": "edge2", "source": "e2", "target": "e3", "community_id": "comm-1"},
            {"id": "edge3", "source": "e3", "target": "e4", "community_id": "comm-2"},
        ]

        grouped = summarizer._group_edges_by_community(edges)

        assert len(grouped["comm-1"]) == 2
        assert len(grouped["comm-2"]) == 1

    def test_summarize_entities(self):
        """Test entity summarization."""
        summarizer = CommunitySummarizer()
        entities = [
            {"name": "Entity1", "entity_type": "Person"},
            {"name": "Entity2", "entity_type": "Person"},
            {"name": "Entity3", "entity_type": "Organization"},
        ]

        summary = summarizer._summarize_entities(entities)

        assert "3 entities" in summary
        assert "Person" in summary
        assert "Organization" in summary
        assert "Entity1" in summary

    def test_summarize_empty_entities(self):
        """Test summarizing empty entity list."""
        summarizer = CommunitySummarizer()
        summary = summarizer._summarize_entities([])
        assert summary == "No entities in this community."

    def test_summarize_single_entity(self):
        """Test summarizing single entity."""
        summarizer = CommunitySummarizer()
        entities = [{"name": "Single", "entity_type": "Concept"}]

        summary = summarizer._summarize_entities(entities)

        assert "1 entities" in summary
        assert "Concept" in summary
        assert "Single" in summary

    def test_create_macro_communities(self):
        """Test macro community creation."""
        summarizer = CommunitySummarizer()
        community_entities = {
            "comm-1": [
                {"name": "e1", "entity_type": "Person"},
                {"name": "e2", "entity_type": "Person"},
            ],
            "comm-2": [
                {"name": "e3", "entity_type": "Organization"},
                {"name": "e4", "entity_type": "Organization"},
                {"name": "e5", "entity_type": "Organization"},
            ],
        }
        community_edges = {}
        communities = []

        macro_summaries = summarizer._create_macro_communities(
            community_entities, community_edges, communities
        )

        assert len(macro_summaries) == 2
        for summary in macro_summaries:
            assert summary.level == HierarchyLevel.MACRO
            assert "Cluster" in summary.name

    def test_create_meso_communities(self):
        """Test meso community creation."""
        summarizer = CommunitySummarizer()
        community_entities = {
            "comm-1": [
                {"name": "e1", "entity_type": "Person"},
                {"name": "e2", "entity_type": "Person"},
            ],
            "comm-2": [
                {"name": "e3", "entity_type": "Concept"},
            ],
        }
        community_edges = {}
        communities = []

        meso_summaries = summarizer._create_meso_communities(
            community_entities, community_edges, communities
        )

        assert len(meso_summaries) == 1
        assert meso_summaries[0].level == HierarchyLevel.MESO

    def test_create_meso_communities_min_entities(self):
        """Test meso community creation requires minimum entities."""
        summarizer = CommunitySummarizer()
        community_entities = {
            "comm-1": [
                {"name": "e1", "entity_type": "Person"},
            ]
        }
        community_edges = {}
        communities = []

        meso_summaries = summarizer._create_meso_communities(
            community_entities, community_edges, communities
        )

        assert len(meso_summaries) == 0

    def test_create_micro_communities(self):
        """Test micro community creation."""
        summarizer = CommunitySummarizer()
        community_entities = {
            "comm-1": [
                {
                    "id": "id-1",
                    "name": "e1",
                    "entity_type": "Person",
                    "description": "A person",
                },
                {
                    "id": "id-2",
                    "name": "e2",
                    "entity_type": "Person",
                    "description": "Another person",
                },
            ]
        }
        community_edges = {
            "comm-1": [{"source": "e1", "target": "e2", "relationship_type": "knows"}]
        }
        communities = []

        micro_summaries = summarizer._create_micro_communities(
            community_entities, community_edges, communities
        )

        assert len(micro_summaries) == 2
        for summary in micro_summaries:
            assert summary.level == HierarchyLevel.MICRO
            assert summary.entity_count == 1
            assert summary.coherence_score == 1.0

    def test_create_micro_communities_limiting(self):
        """Test micro community creation limits to 100 total."""
        summarizer = CommunitySummarizer()
        community_entities = {}
        for i in range(5):
            community_entities[f"comm-{i}"] = [
                {"id": f"id-{j}", "name": f"e{j}", "entity_type": "Person"}
                for j in range(30)
            ]
        community_edges = {}
        communities = []

        micro_summaries = summarizer._create_micro_communities(
            community_entities, community_edges, communities
        )

        assert len(micro_summaries) == 100

    def test_build_hierarchy_tree(self):
        """Test hierarchy tree building."""
        summarizer = CommunitySummarizer()

        macro = [
            CommunitySummary(
                community_id="macro-1",
                name="Macro 1",
                level=HierarchyLevel.MACRO,
                summary="Macro summary",
            )
        ]
        meso = [
            CommunitySummary(
                community_id="meso-1",
                name="Meso 1",
                level=HierarchyLevel.MESO,
                summary="Meso summary",
            )
        ]
        micro = [
            CommunitySummary(
                community_id="micro-1",
                name="Micro 1",
                level=HierarchyLevel.MICRO,
                summary="Micro summary",
            )
        ]

        tree = summarizer._build_hierarchy_tree(macro, meso, micro)

        assert tree["type"] == "hierarchy"
        assert tree["levels"]["macro"] == 1
        assert tree["levels"]["meso"] == 1
        assert tree["levels"]["micro"] == 1
        assert "nodes" in tree
        assert "macro-1" in tree["nodes"]
        assert "meso-1" in tree["nodes"]
        assert "micro-1" in tree["nodes"]

    def test_build_hierarchy_tree_empty(self):
        """Test hierarchy tree building with empty lists."""
        summarizer = CommunitySummarizer()

        tree = summarizer._build_hierarchy_tree([], [], [])

        assert tree["type"] == "hierarchy"
        assert tree["levels"]["macro"] == 0
        assert tree["levels"]["meso"] == 0
        assert tree["levels"]["micro"] == 0
        assert tree["nodes"] == {}

    def test_calculate_parent_score(self):
        """Test parent-child score calculation."""
        summarizer = CommunitySummarizer()

        parent = CommunitySummary(
            community_id="parent",
            name="Parent",
            level=HierarchyLevel.MACRO,
            summary="Parent summary",
            key_entities=["apple", "banana"],
            entity_count=10,
        )
        child = CommunitySummary(
            community_id="child",
            name="Child",
            level=HierarchyLevel.MESO,
            summary="Child summary",
            key_entities=["apple", "cherry"],
            entity_count=5,
        )

        score = summarizer._calculate_parent_score(child, parent)

        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_get_summary_by_level(self):
        """Test getting summaries by hierarchy level."""
        summarizer = CommunitySummarizer()

        macro = CommunitySummary(
            community_id="macro-1",
            name="Macro",
            level=HierarchyLevel.MACRO,
            summary="Macro",
        )
        meso = CommunitySummary(
            community_id="meso-1",
            name="Meso",
            level=HierarchyLevel.MESO,
            summary="Meso",
        )

        summary = MultiLevelSummary(
            document_id="doc-1", macro_communities=[macro], meso_communities=[meso]
        )

        macro_result = summarizer.get_summary_by_level(summary, HierarchyLevel.MACRO)
        meso_result = summarizer.get_summary_by_level(summary, HierarchyLevel.MESO)
        micro_result = summarizer.get_summary_by_level(summary, HierarchyLevel.MICRO)

        assert len(macro_result) == 1
        assert len(meso_result) == 1
        assert len(micro_result) == 0

    def test_get_community_path(self):
        """Test getting community path to root."""
        summarizer = CommunitySummarizer()

        macro = CommunitySummary(
            community_id="macro-1",
            name="Macro",
            level=HierarchyLevel.MACRO,
            summary="Macro",
            parent_community_id=None,
        )
        meso = CommunitySummary(
            community_id="meso-1",
            name="Meso",
            level=HierarchyLevel.MESO,
            summary="Meso",
            parent_community_id="macro-1",
        )
        micro = CommunitySummary(
            community_id="micro-1",
            name="Micro",
            level=HierarchyLevel.MICRO,
            summary="Micro",
            parent_community_id="meso-1",
        )

        summary = MultiLevelSummary(
            document_id="doc-1",
            macro_communities=[macro],
            meso_communities=[meso],
            micro_communities=[micro],
        )

        path = summarizer.get_community_path(summary, "micro-1")

        assert len(path) == 3
        assert path[0].community_id == "micro-1"
        assert path[1].community_id == "meso-1"
        assert path[2].community_id == "macro-1"

    @pytest.mark.asyncio
    async def test_generate_summaries(self):
        """Test full summary generation."""
        summarizer = CommunitySummarizer()
        entities = [
            {
                "id": "e1",
                "name": "Entity1",
                "entity_type": "Person",
                "community_id": "comm-1",
            },
            {
                "id": "e2",
                "name": "Entity2",
                "entity_type": "Person",
                "community_id": "comm-1",
            },
            {
                "id": "e3",
                "name": "Entity3",
                "entity_type": "Organization",
                "community_id": "comm-2",
            },
        ]
        edges = []
        communities = []

        result = await summarizer.generate_summaries(
            communities, entities, edges, "doc-123"
        )

        assert isinstance(result, MultiLevelSummary)
        assert result.document_id == "doc-123"
        assert len(result.macro_communities) >= 0
        assert len(result.meso_communities) >= 0
        assert len(result.micro_communities) >= 0
        assert "nodes" in result.hierarchy_tree

    @pytest.mark.asyncio
    async def test_generate_summaries_with_llm_naming(self):
        """Test summary generation with LLM-based naming."""
        mock_llm = MagicMock()
        mock_llm.generate_text = AsyncMock(
            return_value='{"name": "AI Systems", "confidence": 0.9}'
        )

        summarizer = CommunitySummarizer(llm_client=mock_llm)
        entities = [
            {
                "id": "e1",
                "name": "ChatGPT",
                "entity_type": "AI Model",
                "community_id": "comm-1",
            },
            {
                "id": "e2",
                "name": "GPT-4",
                "entity_type": "AI Model",
                "community_id": "comm-1",
            },
        ]
        edges = []
        communities = []

        result = await summarizer.generate_summaries(
            communities, entities, edges, "doc-123"
        )

        for macro in result.macro_communities:
            if macro.entity_count > 0:
                assert macro.coherence_score > 0.0

    @pytest.mark.asyncio
    async def test_generate_summaries_empty_input(self):
        """Test summary generation with empty inputs."""
        summarizer = CommunitySummarizer()

        result = await summarizer.generate_summaries([], [], [], "doc-empty")

        assert result.document_id == "doc-empty"
        assert result.macro_communities == []
        assert result.meso_communities == []
        assert result.micro_communities == []
        assert result.hierarchy_tree["levels"]["macro"] == 0

    @pytest.mark.asyncio
    async def test_generate_community_names_no_llm(self):
        """Test name generation without LLM client."""
        summarizer = CommunitySummarizer()

        communities = [
            CommunitySummary(
                community_id="test",
                name="Original Name",
                level=HierarchyLevel.MACRO,
                summary="Test summary",
            )
        ]

        await summarizer._generate_community_names(communities, HierarchyLevel.MACRO)

        assert communities[0].name == "Original Name"

    @pytest.mark.asyncio
    async def test_generate_community_names_with_llm(self):
        """Test name generation with LLM client."""
        mock_llm = MagicMock()
        mock_llm.generate_text = AsyncMock(
            return_value='{"name": "New Name", "confidence": 0.85}'
        )

        summarizer = CommunitySummarizer(llm_client=mock_llm)

        communities = [
            CommunitySummary(
                community_id="test",
                name="Original Name",
                level=HierarchyLevel.MACRO,
                summary="Test summary",
                key_entities=["e1", "e2"],
            )
        ]

        await summarizer._generate_community_names(communities, HierarchyLevel.MACRO)

        assert communities[0].name == "New Name"
        assert communities[0].coherence_score == 0.85

    def test_create_naming_prompt(self):
        """Test naming prompt creation."""
        summarizer = CommunitySummarizer()

        community = CommunitySummary(
            community_id="test",
            name="Test",
            level=HierarchyLevel.MACRO,
            summary="Contains 10 entities of types: Person, Organization",
            key_entities=["Entity1", "Entity2"],
        )

        prompt = summarizer._create_naming_prompt(community, HierarchyLevel.MACRO)

        assert "broad thematic category" in prompt
        assert "Test summary" in prompt
        assert "Entity1, Entity2" in prompt

    def test_hierarchy_level_comparison(self):
        """Test hierarchy level string comparison."""
        assert HierarchyLevel.MACRO == "macro"
        assert HierarchyLevel.MESO < HierarchyLevel.MACRO
        assert HierarchyLevel.MICRO < HierarchyLevel.MESO

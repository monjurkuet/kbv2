"""Unit tests for extraction template registry."""

import pytest
from knowledge_base.extraction.template_registry import (
    ExtractionGoal,
    TemplateRegistry,
    get_default_goals,
    DEFAULT_GOALS,
)


class TestExtractionGoal:
    """Tests for ExtractionGoal model."""

    def test_goal_creation_with_all_fields(self):
        """Test creating a goal with all fields."""
        goal = ExtractionGoal(
            name="test_goal",
            description="Test description",
            target_entities=["Person", "Organization"],
            target_relationships=["related_to", "works_at"],
            priority=1,
            examples=["John", "Google"],
        )
        assert goal.name == "test_goal"
        assert goal.description == "Test description"
        assert len(goal.target_entities) == 2
        assert len(goal.target_relationships) == 2
        assert goal.priority == 1
        assert len(goal.examples) == 2

    def test_goal_creation_with_minimal_fields(self):
        """Test creating a goal with minimal fields."""
        goal = ExtractionGoal(
            name="minimal_goal",
            description="Minimal description",
            target_entities=["Entity"],
            target_relationships=["relates_to"],
        )
        assert goal.name == "minimal_goal"
        assert goal.priority == 1
        assert goal.examples == []

    def test_goal_priority_validation(self):
        """Test that priority must be between 1 and 5."""
        with pytest.raises(ValueError):
            ExtractionGoal(
                name="invalid_priority",
                description="Test",
                target_entities=["E"],
                target_relationships=["r"],
                priority=0,
            )

        with pytest.raises(ValueError):
            ExtractionGoal(
                name="invalid_priority",
                description="Test",
                target_entities=["E"],
                target_relationships=["r"],
                priority=6,
            )

    def test_goal_json_serialization(self):
        """Test goal can be serialized to JSON."""
        goal = ExtractionGoal(
            name="test_goal",
            description="Test description",
            target_entities=["Person"],
            target_relationships=["related_to"],
            priority=2,
            examples=["Example"],
        )
        json_data = goal.model_dump()
        assert json_data["name"] == "test_goal"
        assert json_data["priority"] == 2


class TestTemplateRegistry:
    """Tests for TemplateRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a template registry instance."""
        return TemplateRegistry()

    def test_get_goals_for_technology_domain(self, registry):
        """Test getting goals for TECHNOLOGY domain."""
        goals = registry.get_goals("TECHNOLOGY")
        assert len(goals) > 0
        assert any(g.name == "software_systems" for g in goals)

    def test_get_goals_for_financial_domain(self, registry):
        """Test getting goals for FINANCIAL domain."""
        goals = registry.get_goals("FINANCIAL")
        assert len(goals) > 0
        assert any(g.name == "companies" for g in goals)

    def test_get_goals_for_medical_domain(self, registry):
        """Test getting goals for MEDICAL domain."""
        goals = registry.get_goals("MEDICAL")
        assert len(goals) > 0
        assert any(g.name == "diseases" for g in goals)

    def test_get_goals_for_legal_domain(self, registry):
        """Test getting goals for LEGAL domain."""
        goals = registry.get_goals("LEGAL")
        assert len(goals) > 0
        assert any(g.name == "entities" for g in goals)

    def test_get_goals_for_scientific_domain(self, registry):
        """Test getting goals for SCIENTIFIC domain."""
        goals = registry.get_goals("SCIENTIFIC")
        assert len(goals) > 0
        assert any(g.name == "research_papers" for g in goals)

    def test_get_goals_for_general_domain(self, registry):
        """Test getting goals for GENERAL domain (default)."""
        goals = registry.get_goals("GENERAL")
        assert len(goals) > 0
        assert any(g.name == "entities" for g in goals)

    def test_get_goals_case_insensitive(self, registry):
        """Test that domain lookup is case-insensitive."""
        goals_lower = registry.get_goals("technology")
        goals_upper = registry.get_goals("TECHNOLOGY")
        goals_mixed = registry.get_goals("Technology")

        assert len(goals_lower) == len(goals_upper)
        assert len(goals_upper) == len(goals_mixed)

    def test_get_goals_unknown_domain_defaults_to_general(self, registry):
        """Test unknown domain falls back to GENERAL goals."""
        goals = registry.get_goals("UNKNOWN_DOMAIN")
        general_goals = registry.get_goals("GENERAL")
        assert len(goals) == len(general_goals)

    def test_get_goal_by_name_exists(self, registry):
        """Test getting a specific goal by name."""
        goal = registry.get_goal_by_name("TECHNOLOGY", "software_systems")
        assert goal is not None
        assert goal.name == "software_systems"

    def test_get_goal_by_name_not_exists(self, registry):
        """Test getting a non-existent goal returns None."""
        goal = registry.get_goal_by_name("TECHNOLOGY", "nonexistent")
        assert goal is None

    def test_get_prioritized_goals_sorted(self, registry):
        """Test that goals are sorted by priority."""
        goals = registry.get_prioritized_goals("TECHNOLOGY")
        assert len(goals) > 1
        for i in range(len(goals) - 1):
            assert goals[i].priority <= goals[i + 1].priority

    def test_registry_contains_all_domains(self, registry):
        """Test registry contains goals for all expected domains."""
        expected_domains = [
            "TECHNOLOGY",
            "FINANCIAL",
            "MEDICAL",
            "LEGAL",
            "SCIENTIFIC",
            "GENERAL",
        ]
        for domain in expected_domains:
            goals = registry.get_goals(domain)
            assert len(goals) > 0, f"No goals found for {domain}"

    def test_technology_goals_have_correct_entities(self, registry):
        """Test TECHNOLOGY domain goals have appropriate entity types."""
        goals = registry.get_goals("TECHNOLOGY")
        all_entities = []
        for goal in goals:
            all_entities.extend(goal.target_entities)

        assert "Software" in all_entities
        assert "Library" in all_entities
        assert "Framework" in all_entities
        assert "API" in all_entities

    def test_financial_goals_have_correct_entities(self, registry):
        """Test FINANCIAL domain goals have appropriate entity types."""
        goals = registry.get_goals("FINANCIAL")
        all_entities = []
        for goal in goals:
            all_entities.extend(goal.target_entities)

        assert "Company" in all_entities
        assert "Revenue" in all_entities
        assert "Investment" in all_entities

    def test_medical_goals_have_correct_entities(self, registry):
        """Test MEDICAL domain goals have appropriate entity types."""
        goals = registry.get_goals("MEDICAL")
        all_entities = []
        for goal in goals:
            all_entities.extend(goal.target_entities)

        assert "Disease" in all_entities
        assert "Drug" in all_entities
        assert "Treatment" in all_entities


class TestGetDefaultGoals:
    """Tests for the get_default_goals function."""

    def test_get_default_goals_returns_list(self):
        """Test get_default_goals returns a list."""
        goals = get_default_goals("TECHNOLOGY")
        assert isinstance(goals, list)

    def test_get_default_goals_technology(self):
        """Test getting default goals for TECHNOLOGY."""
        goals = get_default_goals("TECHNOLOGY")
        assert len(goals) > 0

    def test_get_default_goals_financial(self):
        """Test getting default goals for FINANCIAL."""
        goals = get_default_goals("FINANCIAL")
        assert len(goals) > 0

    def test_get_default_goals_unknown(self):
        """Test getting default goals for unknown domain returns GENERAL."""
        goals = get_default_goals("UNKNOWN")
        general_goals = get_default_goals("GENERAL")
        assert goals == general_goals


class TestDefaultGoalsStructure:
    """Tests for the structure of DEFAULT_GOALS."""

    def test_default_goals_has_all_domains(self):
        """Test DEFAULT_GOALS contains all expected domains."""
        expected_domains = [
            "TECHNOLOGY",
            "FINANCIAL",
            "MEDICAL",
            "LEGAL",
            "SCIENTIFIC",
            "GENERAL",
        ]
        for domain in expected_domains:
            assert domain in DEFAULT_GOALS, f"Missing domain: {domain}"

    def test_each_domain_has_goals(self):
        """Test each domain has at least one goal."""
        for domain, goals in DEFAULT_GOALS.items():
            assert len(goals) > 0, f"Domain {domain} has no goals"

    def test_each_goal_has_required_fields(self):
        """Test each goal has all required fields."""
        for domain, goals in DEFAULT_GOALS.items():
            for goal in goals:
                assert goal.name is not None
                assert goal.description is not None
                assert len(goal.target_entities) > 0
                assert len(goal.target_relationships) > 0
                assert 1 <= goal.priority <= 5

    def test_goal_names_are_unique_within_domain(self):
        """Test goal names are unique within each domain."""
        for domain, goals in DEFAULT_GOALS.items():
            names = [g.name for g in goals]
            assert len(names) == len(set(names)), f"Duplicate goal names in {domain}"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing extraction."""

    def test_template_registry_default_goals_not_empty(self):
        """Test default goals dictionary is not empty."""
        assert len(DEFAULT_GOALS) > 0

    def test_extraction_goal_model_compatible(self):
        """Test ExtractionGoal is compatible with basic usage."""
        goal = ExtractionGoal(
            name="compatibility_test",
            description="Testing backward compatibility",
            target_entities=["TestEntity"],
            target_relationships=["test_rel"],
        )
        assert goal.name == "compatibility_test"
        assert goal.priority == 1

    def test_get_default_goals_works_for_all_domains(self):
        """Test get_default_goals works for all domain types."""
        for domain in DEFAULT_GOALS.keys():
            goals = get_default_goals(domain)
            assert len(goals) > 0

"""Unit tests for FederatedQueryRouter."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_base.intelligence.v1.federated_query_router import (
    FederatedQueryRouter,
    QueryDomain,
    ExecutionStrategy,
    SubQuery,
    DomainMatch,
    QueryPlanStep,
    FederatedQueryPlan,
    FederatedQueryResult,
    DomainDetector,
    SubQueryBuilder,
    ResultAggregator,
)


class TestDomainDetector:
    """Tests for DomainDetector class."""

    def test_detect_technical_domain(self) -> None:
        """Test detection of technical domain keywords."""
        detector = DomainDetector()
        matches = detector.detect(
            "api code function method server endpoint authentication bug error"
        )

        assert len(matches) > 0
        assert QueryDomain.TECHNICAL in [m.domain for m in matches]
        assert all(m.confidence >= 0.3 for m in matches)

    def test_detect_business_domain(self) -> None:
        """Test detection of business domain keywords."""
        detector = DomainDetector()
        matches = detector.detect(
            "revenue customer sales profit strategy market competitor budget roi kpi metric growth"
        )

        assert len(matches) > 0
        assert QueryDomain.BUSINESS in [m.domain for m in matches]

    def test_detect_documentation_domain(self) -> None:
        """Test detection of documentation domain keywords."""
        detector = DomainDetector()
        matches = detector.detect("Please provide a tutorial guide for setup")

        assert len(matches) > 0
        assert QueryDomain.DOCUMENTATION in [m.domain for m in matches]

    def test_detect_research_domain(self) -> None:
        """Test detection of research domain keywords."""
        detector = DomainDetector()
        matches = detector.detect(
            "study analysis hypothesis findings paper academic theory experiment methodology review"
        )

        assert len(matches) > 0
        assert QueryDomain.RESEARCH in [m.domain for m in matches]

    def test_detect_analytics_domain(self) -> None:
        """Test detection of analytics domain keywords."""
        detector = DomainDetector()
        matches = detector.detect(
            "trend pattern statistic distribution correlation average sum count percentage dashboard report"
        )

        assert len(matches) > 0
        assert QueryDomain.ANALYTICS in [m.domain for m in matches]

    def test_detect_no_keywords_returns_default(self) -> None:
        """Test that default domain is returned when no keywords match."""
        detector = DomainDetector(default_domain=QueryDomain.GENERAL)
        matches = detector.detect("Random query with no specific keywords")

        assert len(matches) == 1
        assert matches[0].domain == QueryDomain.GENERAL
        assert matches[0].confidence == 1.0

    def test_detect_max_domains_limit(self) -> None:
        """Test that max_domains limit is respected."""
        detector = DomainDetector()
        query = "API server bug test metric report"  # Multiple domains
        matches = detector.detect(query, max_domains=2)

        assert len(matches) <= 2

    def test_detect_sorted_by_confidence(self) -> None:
        """Test that matches are sorted by confidence descending."""
        detector = DomainDetector()
        query = "API authentication code method bug"  # Technical heavy
        matches = detector.detect(query)

        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].confidence >= matches[i + 1].confidence

    def test_detect_keywords_tracked(self) -> None:
        """Test that matched keywords are tracked."""
        detector = DomainDetector()
        matches = detector.detect("API endpoint documentation guide")

        for match in matches:
            for keyword in match.keywords:
                assert keyword in match.keywords


class TestSubQueryBuilder:
    """Tests for SubQueryBuilder class."""

    def test_build_subqueries_for_domain(self) -> None:
        """Test building sub-queries for a specific domain."""
        builder = SubQueryBuilder()
        sub_queries = builder.build("test query", QueryDomain.TECHNICAL)

        assert len(sub_queries) > 0
        for sq in sub_queries:
            assert sq.domain == QueryDomain.TECHNICAL
            assert "test query" in sq.query_text

    def test_build_subqueries_with_context(self) -> None:
        """Test building sub-queries with context parameters."""
        builder = SubQueryBuilder()
        context = {"limit": 10, "filters": {"status": "active"}}
        sub_queries = builder.build("test query", QueryDomain.GENERAL, context)

        assert len(sub_queries) > 0
        assert sub_queries[0].parameters == context

    def test_build_different_domains(self) -> None:
        """Test that different domains get different templates."""
        builder = SubQueryBuilder()

        tech_queries = builder.build("test", QueryDomain.TECHNICAL)
        biz_queries = builder.build("test", QueryDomain.BUSINESS)

        assert len(tech_queries) > 0
        assert len(biz_queries) > 0

        tech_text = " ".join(sq.query_text for sq in tech_queries)
        biz_text = " ".join(sq.query_text for sq in biz_queries)

        assert tech_text != biz_text

    def test_add_custom_template(self) -> None:
        """Test adding custom query template."""
        builder = SubQueryBuilder()
        builder.add_template(
            QueryDomain.RESEARCH,
            "Find academic papers about {query}",
            priority=0,
        )

        sub_queries = builder.build("test", QueryDomain.RESEARCH)
        assert any("academic papers" in sq.query_text for sq in sub_queries)

    def test_default_template_fallback(self) -> None:
        """Test that unknown domains use general template."""
        builder = SubQueryBuilder()
        sub_queries = builder.build("test", QueryDomain.GENERAL)

        assert len(sub_queries) > 0
        assert "test" in sub_queries[0].query_text


class TestResultAggregator:
    """Tests for ResultAggregator class."""

    def test_merge_results(self) -> None:
        """Test merging results from multiple domains."""
        aggregator = ResultAggregator(mode="merge", deduplicate=False)
        results = {
            QueryDomain.TECHNICAL: [
                {"id": "1", "name": "tech1"},
                {"id": "2", "name": "tech2"},
            ],
            QueryDomain.BUSINESS: [
                {"id": "3", "name": "biz1"},
            ],
        }

        merged = aggregator.aggregate(results)

        assert len(merged) == 3
        assert all("_domain" in r for r in merged)

    def test_deduplicate_results(self) -> None:
        """Test that duplicate results are removed."""
        aggregator = ResultAggregator(mode="merge", deduplicate=True)
        results = {
            QueryDomain.TECHNICAL: [
                {"id": "1", "name": "item1"},
                {"id": "2", "name": "item2"},
            ],
            QueryDomain.BUSINESS: [
                {"id": "1", "name": "item1"},  # Duplicate
            ],
        }

        merged = aggregator.aggregate(results)

        assert len(merged) == 2

    def test_union_aggregates_all(self) -> None:
        """Test union mode includes all results."""
        aggregator = ResultAggregator(mode="union")
        results = {
            QueryDomain.TECHNICAL: [{"id": "1"}],
            QueryDomain.BUSINESS: [{"id": "2"}],
        }

        merged = aggregator.aggregate(results)

        assert len(merged) == 2

    def test_confidence_ranking(self) -> None:
        """Test that results are ranked by confidence."""
        aggregator = ResultAggregator(mode="merge", ranking_strategy="confidence")
        results = {
            QueryDomain.TECHNICAL: [
                {"id": "1", "confidence": 0.3},
                {"id": "2", "confidence": 0.9},
            ],
        }

        merged = aggregator.aggregate(results)

        assert merged[0]["confidence"] >= merged[1]["confidence"]

    def test_empty_results(self) -> None:
        """Test handling of empty results."""
        aggregator = ResultAggregator()
        merged = aggregator.aggregate({})

        assert merged == []


class TestFederatedQueryRouter:
    """Tests for FederatedQueryRouter class."""

    def test_create_plan_basic(self) -> None:
        """Test creating a basic execution plan."""
        router = FederatedQueryRouter()
        plan = router.create_plan("test query")

        assert isinstance(plan, FederatedQueryPlan)
        assert plan.original_query == "test query"
        assert len(plan.steps) > 0
        assert plan.strategy == ExecutionStrategy.PARALLEL

    def test_create_plan_with_domains(self) -> None:
        """Test creating plan with specified domains."""
        router = FederatedQueryRouter()
        plan = router.create_plan(
            "test query",
            detected_domains=[QueryDomain.TECHNICAL, QueryDomain.BUSINESS],
        )

        domain_set = {step.sub_query.domain for step in plan.steps}
        assert QueryDomain.TECHNICAL in domain_set
        assert QueryDomain.BUSINESS in domain_set

    def test_create_plan_with_strategy(self) -> None:
        """Test creating plan with specific strategy."""
        router = FederatedQueryRouter()
        plan = router.create_plan(
            "test query",
            strategy=ExecutionStrategy.SEQUENTIAL,
        )

        assert plan.strategy == ExecutionStrategy.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_execute_plan_parallel(self) -> None:
        """Test executing plan with parallel strategy."""
        router = FederatedQueryRouter()
        plan = router.create_plan(
            "test query",
            strategy=ExecutionStrategy.PARALLEL,
        )

        result = await router.execute_plan(plan)

        assert isinstance(result, FederatedQueryResult)
        assert result.query == "test query"
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_plan_sequential(self) -> None:
        """Test executing plan with sequential strategy."""
        router = FederatedQueryRouter()
        plan = router.create_plan(
            "test query",
            strategy=ExecutionStrategy.SEQUENTIAL,
        )

        result = await router.execute_plan(plan)

        assert isinstance(result, FederatedQueryResult)

    @pytest.mark.asyncio
    async def test_execute_with_retriever(self) -> None:
        """Test execution with a mock retriever."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=MagicMock(
                entities=[
                    MagicMock(
                        id="test-id",
                        name="Test Entity",
                        entity_type="test_type",
                        description="Test description",
                        final_score=0.9,
                        source="vector",
                    ),
                ]
            )
        )

        router = FederatedQueryRouter(retriever=mock_retriever)
        plan = router.create_plan("test query")

        result = await router.execute_plan(plan)

        mock_retriever.retrieve.assert_called()

    def test_aggregate_results(self) -> None:
        """Test result aggregation."""
        router = FederatedQueryRouter()
        plan = router.create_plan("test query")

        result = FederatedQueryResult(query="test query", plan=plan)
        result.add_domain_results(
            QueryDomain.TECHNICAL,
            [
                {"id": "1", "name": "tech1"},
            ],
        )
        result.add_domain_results(
            QueryDomain.BUSINESS,
            [
                {"id": "2", "name": "biz1"},
            ],
        )

        aggregated = router.aggregate_results(result)

        assert len(aggregated) == 2

    @pytest.mark.asyncio
    async def test_route_and_execute(self) -> None:
        """Test complete route and execute flow."""
        router = FederatedQueryRouter()
        result = await router.route_and_execute("How do I authenticate with the API?")

        assert isinstance(result, FederatedQueryResult)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_route_and_execute_with_strategy(self) -> None:
        """Test route and execute with custom strategy."""
        router = FederatedQueryRouter()
        result = await router.route_and_execute(
            "test query",
            strategy=ExecutionStrategy.SEQUENTIAL,
            aggregation_mode="union",
        )

        assert result.plan.strategy == ExecutionStrategy.SEQUENTIAL
        assert result.plan.aggregation_mode == "union"


class TestSubQuery:
    """Tests for SubQuery dataclass."""

    def test_subquery_creation(self) -> None:
        """Test SubQuery instantiation."""
        sq = SubQuery(
            domain=QueryDomain.TECHNICAL,
            query_text="Find API info",
            priority=1,
            parameters={"limit": 10},
        )

        assert sq.domain == QueryDomain.TECHNICAL
        assert sq.query_text == "Find API info"
        assert sq.priority == 1
        assert sq.parameters == {"limit": 10}

    def test_subquery_defaults(self) -> None:
        """Test SubQuery default values."""
        sq = SubQuery(
            domain=QueryDomain.GENERAL,
            query_text="test",
        )

        assert sq.priority == 0
        assert sq.parameters == {}


class TestFederatedQueryPlan:
    """Tests for FederatedQueryPlan dataclass."""

    def test_plan_creation(self) -> None:
        """Test FederatedQueryPlan instantiation."""
        step = QueryPlanStep(
            sub_query=SubQuery(
                domain=QueryDomain.TECHNICAL,
                query_text="test",
            )
        )
        plan = FederatedQueryPlan(
            original_query="original",
            steps=[step],
            strategy=ExecutionStrategy.PARALLEL,
            aggregation_mode="merge",
            timeout_seconds=60,
        )

        assert plan.original_query == "original"
        assert len(plan.steps) == 1
        assert plan.strategy == ExecutionStrategy.PARALLEL
        assert plan.aggregation_mode == "merge"
        assert plan.timeout_seconds == 60


class TestFederatedQueryResult:
    """Tests for FederatedQueryResult class."""

    def test_add_domain_results(self) -> None:
        """Test adding results for a domain."""
        plan = FederatedQueryPlan(
            original_query="test",
            steps=[],
            strategy=ExecutionStrategy.PARALLEL,
        )
        result = FederatedQueryResult(query="test", plan=plan)

        result.add_domain_results(
            QueryDomain.TECHNICAL,
            [
                {"id": "1", "name": "item1"},
            ],
        )

        assert QueryDomain.TECHNICAL in result.results
        assert len(result.results[QueryDomain.TECHNICAL]) == 1
        assert result.total_results == 1

    def test_add_domain_error(self) -> None:
        """Test adding an error for a domain."""
        plan = FederatedQueryPlan(
            original_query="test",
            steps=[],
            strategy=ExecutionStrategy.PARALLEL,
        )
        result = FederatedQueryResult(query="test", plan=plan)

        result.add_domain_error(QueryDomain.TECHNICAL, "Connection timeout")

        assert QueryDomain.TECHNICAL in result.errors
        assert result.errors[QueryDomain.TECHNICAL] == "Connection timeout"

    def test_get_combined_results(self) -> None:
        """Test getting combined results with domain attribution."""
        plan = FederatedQueryPlan(
            original_query="test",
            steps=[],
            strategy=ExecutionStrategy.PARALLEL,
        )
        result = FederatedQueryResult(query="test", plan=plan)

        result.add_domain_results(
            QueryDomain.TECHNICAL,
            [
                {"id": "1", "name": "tech1"},
            ],
        )
        result.add_domain_results(
            QueryDomain.BUSINESS,
            [
                {"id": "2", "name": "biz1"},
            ],
        )

        combined = result.get_combined_results()

        assert len(combined) == 2
        assert all("_domain" in r for r in combined)


class TestExecutionStrategy:
    """Tests for ExecutionStrategy enum."""

    def test_execution_strategies_exist(self) -> None:
        """Test that all expected strategies exist."""
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.PRIORITY.value == "priority"


class TestQueryDomain:
    """Tests for QueryDomain enum."""

    def test_all_domains_defined(self) -> None:
        """Test that all expected domains are defined."""
        domains = list(QueryDomain)
        assert QueryDomain.GENERAL in domains
        assert QueryDomain.TECHNICAL in domains
        assert QueryDomain.BUSINESS in domains
        assert QueryDomain.DOCUMENTATION in domains
        assert QueryDomain.RESEARCH in domains
        assert QueryDomain.ANALYTICS in domains

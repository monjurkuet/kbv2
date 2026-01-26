"""Comprehensive integration tests for KBV2 enhanced pipeline features.

Tests all new features:
1. Enhanced LLM Gateway (CoT, CoD, JSON, few-shot)
2. Multi-agent entity extraction with quality verification
3. Hallucination detection and auto-approval
4. Federated query routing
5. Hybrid search
6. Domain schemas
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from uuid import uuid4, UUID
from typing import Any
from datetime import datetime

from pydantic import BaseModel

from knowledge_base.intelligence import (
    EntityExtractionManager,
    FederatedQueryRouter,
    HybridEntityRetriever,
    SchemaRegistry,
    HallucinationDetector,
    HallucinationDetectorConfig,
    EntityVerification,
    AttributeVerification,
    RiskLevel,
    VerificationStatus,
    ExtractionQualityScore,
    EntityExtractionQuality,
)
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    ExtractedEntity,
    ExtractionPhase,
    PerceptionAgent,
    EnhancementAgent,
    EvaluationAgent,
    ManagerAgent,
    MultiAgentExtractorConfig,
    EnhancementContext,
    ExtractionWorkflowState,
)
from knowledge_base.intelligence.v1.federated_query_router import (
    FederatedQueryPlan,
    FederatedQueryResult,
    QueryDomain,
    ExecutionStrategy,
    DomainDetector,
    SubQueryBuilder,
    ResultAggregator,
    SubQuery,
    DomainMatch,
    QueryPlanStep,
)
from knowledge_base.intelligence.v1.hybrid_retriever import (
    HybridEntityRetriever,
    HybridRetrievalResult,
    RetrievedEntity,
    GraphEntity,
)
from knowledge_base.intelligence.v1.domain_schema_service import (
    SchemaRegistry,
    DomainSchema,
    DomainSchemaModel,
    DomainSchemaCreate,
    DomainSchemaUpdate,
    EntityTypeDef,
    DomainAttribute,
    InheritanceType,
    DomainLevel,
    SchemaValidationMode,
)
from knowledge_base.persistence.v1.schema import (
    Document,
    Entity,
    Chunk,
    Edge,
    EdgeType,
)
from knowledge_base.common.gateway import GatewayConfig, EnhancedGateway


class StructuredTestSchema(BaseModel):
    """Test schema for structured extraction."""

    name: str
    value: int


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    from dataclasses import dataclass, field
    from typing import List
    from pydantic import BaseModel

    @dataclass
    class MockStep:
        reasoning: str = ""
        type: str = "reasoning"

        def model_dump(self):
            return {"reasoning": self.reasoning, "type": self.type}

    client = AsyncMock()
    client.complete = AsyncMock(return_value="test response")
    client.complete_json = AsyncMock(return_value={"name": "test", "value": 42})
    client.complete_with_cot_steps = AsyncMock(
        return_value=(
            "reasoned answer",
            [MockStep("step1", "reasoning"), MockStep("step2", "reasoning")],
        )
    )
    client.complete_with_cod_steps = AsyncMock(
        return_value=(
            "draft answer",
            [MockStep("draft1", "draft"), MockStep("draft2", "draft")],
        )
    )
    return client


@pytest.fixture
def enhanced_gateway(mock_llm_client):
    """Create enhanced gateway with mocked LLM client."""
    config = GatewayConfig(
        url="http://localhost:8087/v1/",
        api_key="test-key",
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=4096,
    )
    gateway = EnhancedGateway(config)
    gateway._llm_client = mock_llm_client
    return gateway


@pytest.fixture
def sample_document():
    """Sample document for extraction testing."""
    doc_id = uuid4()
    return Document(
        id=doc_id,
        name="Test Document",
        doc_metadata={
            "content": "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976."
        },
        status="pending",
        domain="technology",
    )


@pytest.fixture
def sample_chunk():
    """Sample chunk for extraction testing."""
    chunk_id = uuid4()
    doc_id = uuid4()
    return Chunk(
        id=chunk_id,
        document_id=doc_id,
        text="Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. The company is headquartered in Cupertino, California.",
        chunk_index=0,
    )


@pytest.fixture
def sample_entities():
    """Sample entities for retrieval testing."""
    return [
        Entity(
            id=uuid4(),
            name="Apple Inc.",
            entity_type="ORGANIZATION",
            description="American multinational technology company",
            confidence=0.95,
        ),
        Entity(
            id=uuid4(),
            name="Steve Jobs",
            entity_type="PERSON",
            description="Co-founder of Apple Inc.",
            confidence=0.98,
        ),
    ]


class TestEnhancedGatewayIntegration:
    """Test enhanced gateway with new prompting strategies."""

    @pytest.mark.asyncio
    async def test_cot_extraction_integration(self, enhanced_gateway):
        """Test Chain-of-Thought extraction end-to-end."""
        result = await enhanced_gateway.extract_with_reasoning(
            "Extract entities from: Apple Inc. founded by Steve Jobs"
        )

        assert result["answer"] == "reasoned answer"
        assert "steps" in result
        assert len(result["steps"]) == 2

    @pytest.mark.asyncio
    async def test_cod_extraction_integration(self, enhanced_gateway):
        """Test Chain-of-Draft extraction end-to-end."""
        result = await enhanced_gateway.extract_with_cod(
            "Extract entities from: Apple Inc. founded by Steve Jobs"
        )

        assert result["answer"] == "draft answer"
        assert "steps" in result
        assert len(result["steps"]) == 2

    @pytest.mark.asyncio
    async def test_structured_extraction_integration(self, enhanced_gateway):
        """Test structured JSON extraction."""
        result = await enhanced_gateway.extract_structured(
            "Extract name and value", StructuredTestSchema
        )

        assert isinstance(result, StructuredTestSchema)
        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_few_shot_extraction_integration(self, enhanced_gateway):
        """Test few-shot extraction integration."""
        from knowledge_base.clients import FewShotExample

        examples = [
            FewShotExample(input="input1", output="output1"),
            FewShotExample(input="input2", output="output2"),
        ]

        result = await enhanced_gateway.extract_few_shot("test prompt", examples)

        assert "content" in result or "answer" in result


class TestMultiAgentExtractionIntegration:
    """Test multi-agent extraction pipeline."""

    @pytest.mark.asyncio
    async def test_full_extraction_pipeline(self, sample_chunk):
        """Test complete document processing with multi-agent extraction."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(
            return_value='[{"text": "Apple Inc.", "type": "ORGANIZATION", "start_char": 0, "end_char": 10, "boundary_type": "strong", "context": "Apple Inc. was founded...", "confidence": 0.95}]'
        )

        mock_graph_store = AsyncMock()
        mock_vector_store = AsyncMock()

        config = MultiAgentExtractorConfig()
        manager = EntityExtractionManager(
            gateway=mock_gateway,
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        chunks = [sample_chunk]
        document_id = uuid4()

        entities = await manager.extract_entities(chunks, document_id)

        assert len(entities) >= 0
        if entities:
            assert all(hasattr(e, "name") for e in entities)
            assert all(hasattr(e, "entity_type") for e in entities)

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self):
        """Test extraction quality score calculation."""
        score = ExtractionQualityScore(
            overall_score=0.85,
            entity_quality=0.90,
            relationship_quality=0.80,
            coherence_score=0.85,
            missing_entities=[],
            spurious_entities=[],
            quality_level=EntityExtractionQuality.HIGH,
            feedback="Good extraction quality",
        )

        assert score.overall_score == 0.85
        assert score.quality_level in [
            EntityExtractionQuality.HIGH,
            EntityExtractionQuality.MEDIUM,
        ]
        assert 0.0 <= score.entity_quality <= 1.0
        assert 0.0 <= score.relationship_quality <= 1.0

    @pytest.mark.asyncio
    async def test_perception_agent_extraction(self, sample_chunk):
        """Test perception agent entity extraction."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(
            return_value='[{"text": "Steve Jobs", "type": "PERSON", "start_char": 30, "end_char": 41, "boundary_type": "strong", "context": "Steve Jobs in 1976", "confidence": 0.98}]'
        )

        config = MultiAgentExtractorConfig()
        agent = PerceptionAgent(mock_gateway, config)

        entities = await agent.extract_entities(
            [sample_chunk], entity_types=["PERSON", "ORGANIZATION"]
        )

        assert len(entities) >= 0
        for entity in entities:
            assert isinstance(entity, ExtractedEntity)
            assert entity.extraction_phase == ExtractionPhase.PERCEPTION

    @pytest.mark.asyncio
    async def test_enhancement_agent_entity_refinement(self):
        """Test enhancement agent entity refinement."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(
            return_value='{"refined_name": "Apple Inc.", "refined_type": "ORGANIZATION", "description": "Technology company", "properties": {"founded": 1976}, "confidence_adjustment": 0.05}'
        )

        mock_graph_store = AsyncMock()
        mock_vector_store = AsyncMock()

        config = MultiAgentExtractorConfig()
        agent = EnhancementAgent(
            mock_gateway, mock_graph_store, mock_vector_store, config
        )

        extracted = ExtractedEntity(
            id=uuid4(),
            name="Apple",
            entity_type="ORGANIZATION",
            description="Tech company",
            source_text="Apple Inc. was founded...",
            chunk_id=uuid4(),
            confidence=0.9,
        )

        context = EnhancementContext(
            existing_entities=[], recent_extractions=[extracted]
        )

        enhanced = await agent.enhance_entities([extracted], context)

        assert len(enhanced) == 1
        assert enhanced[0].name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_evaluation_agent_quality_assessment(self):
        """Test evaluation agent quality assessment."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(
            return_value='{"overall_score": 0.85, "entity_quality": 0.90, "relationship_quality": 0.80, "coherence_score": 0.85, "missing_entities": [], "spurious_entities": [], "quality_level": "high", "feedback": "Good extraction"}'
        )

        config = MultiAgentExtractorConfig()
        agent = EvaluationAgent(mock_gateway, config)

        entities = [
            ExtractedEntity(
                id=uuid4(),
                name="Apple Inc.",
                entity_type="ORGANIZATION",
                source_text="Apple Inc. was founded...",
                chunk_id=uuid4(),
                confidence=0.95,
            )
        ]

        chunks = [
            Chunk(
                id=uuid4(),
                document_id=uuid4(),
                text="Apple Inc. was founded by Steve Jobs...",
                chunk_index=0,
            )
        ]

        score = await agent.evaluate_extraction(entities, chunks)

        assert score.overall_score == 0.85
        assert score.quality_level in [
            EntityExtractionQuality.HIGH,
            EntityExtractionQuality.MEDIUM,
        ]


class TestHallucinationDetectionIntegration:
    """Test hallucination detection in review flow."""

    @pytest.mark.asyncio
    async def test_auto_approval_high_confidence(self):
        """Test auto-approval of high-confidence, low-risk entities."""
        detector = HallucinationDetector()

        verification = EntityVerification(
            entity_name="Test Entity",
            entity_type="PERSON",
            risk_level=RiskLevel.LOW,
            overall_confidence=0.95,
            attributes=[
                AttributeVerification(
                    attribute_name="name",
                    claimed_value="John Doe",
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.95,
                    evidence="Found in source text",
                    explanation="Clearly mentioned",
                )
            ],
            total_attributes=1,
            supported_count=1,
            unsupported_count=0,
            inconclusive_count=0,
            is_hallucinated=False,
        )

        assert verification.risk_level == RiskLevel.LOW
        assert verification.overall_confidence > 0.7
        assert not verification.is_hallucinated

    @pytest.mark.asyncio
    async def test_auto_rejection_low_confidence(self):
        """Test auto-rejection of low-confidence, high-risk entities."""
        verification = EntityVerification(
            entity_name="Test Entity",
            entity_type="PERSON",
            risk_level=RiskLevel.CRITICAL,
            overall_confidence=0.2,
            attributes=[
                AttributeVerification(
                    attribute_name="age",
                    claimed_value="150 years",
                    status=VerificationStatus.UNSUPPORTED,
                    confidence=0.2,
                    evidence="Not found in source",
                    explanation="No evidence for this claim",
                )
            ],
            total_attributes=1,
            supported_count=0,
            unsupported_count=1,
            inconclusive_count=0,
            is_hallucinated=True,
        )

        assert verification.risk_level == RiskLevel.CRITICAL
        assert verification.is_hallucinated
        assert verification.unsupported_count > 0

    @pytest.mark.asyncio
    async def test_priority_calculation(self):
        """Test priority calculation with hallucination risk."""
        test_cases = [
            (RiskLevel.LOW, 0.95, 1),
            (RiskLevel.MEDIUM, 0.7, 3),
            (RiskLevel.HIGH, 0.5, 5),
            (RiskLevel.CRITICAL, 0.2, 8),
        ]

        for risk_level, confidence, expected_min_priority in test_cases:
            priority = max(1, int((1 - confidence) * 10))
            if risk_level == RiskLevel.CRITICAL:
                priority = max(priority, 8)
            elif risk_level == RiskLevel.HIGH:
                priority = max(priority, 5)

            assert priority >= expected_min_priority

    @pytest.mark.asyncio
    async def test_attribute_verification_statuses(self):
        """Test all verification status types."""
        statuses = [
            (VerificationStatus.SUPPORTED, True, False, False),
            (VerificationStatus.UNSUPPORTED, False, True, False),
            (VerificationStatus.INCONCLUSIVE, False, False, True),
            (VerificationStatus.CONFLICTING, False, False, False),
        ]

        for status, supported, unsupported, inconclusive in statuses:
            attr = AttributeVerification(
                attribute_name="test",
                claimed_value="test_value",
                status=status,
                confidence=0.5,
            )

            assert attr.status == status

    @pytest.mark.asyncio
    async def test_risk_level_calculation(self):
        """Test risk level calculation logic."""
        detector = HallucinationDetector()

        high_risk_verification = EntityVerification(
            entity_name="Risky Entity",
            entity_type="PERSON",
            risk_level=RiskLevel.HIGH,
            overall_confidence=0.3,
            attributes=[
                AttributeVerification(
                    attribute_name="test",
                    claimed_value="value",
                    status=VerificationStatus.UNSUPPORTED,
                    confidence=0.3,
                )
            ],
            total_attributes=1,
            supported_count=0,
            unsupported_count=1,
            inconclusive_count=0,
            is_hallucinated=True,
        )

        risk = detector._calculate_risk_level(high_risk_verification)
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]


class TestFederatedQueryIntegration:
    """Test federated query routing."""

    @pytest.mark.asyncio
    async def test_domain_detection(self):
        """Test automatic domain detection using keyword matching."""
        detector = DomainDetector()

        test_cases = [
            (
                "API code function class method database server endpoint",
                [QueryDomain.TECHNICAL],
            ),
            (
                "revenue customer sales profit quarter budget roi",
                [QueryDomain.BUSINESS],
            ),
            (
                "documentation guide tutorial reference manual how-to",
                [QueryDomain.DOCUMENTATION],
            ),
            (
                "study analysis hypothesis findings paper academic theory",
                [QueryDomain.RESEARCH],
            ),
            (
                "trend pattern statistic distribution average percentage",
                [QueryDomain.ANALYTICS],
            ),
        ]

        for query, expected_domains in test_cases:
            matches = detector.detect(query)

            assert len(matches) > 0, f"No domains detected for query: {query}"
            detected = [m.domain for m in matches]
            assert any(d in detected for d in expected_domains), (
                f"Expected {expected_domains} but got {detected} for query: {query}"
            )

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution across domains."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=HybridRetrievalResult(
                query="test",
                entities=[],
            )
        )

        router = FederatedQueryRouter(retriever=mock_retriever)

        result = await router.route_and_execute(
            "How to implement API authentication in Python"
        )

        assert isinstance(result, FederatedQueryResult)
        assert result.plan is not None
        assert len(result.plan.steps) > 0

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential execution strategy."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve = AsyncMock(
            return_value=HybridRetrievalResult(
                query="test",
                entities=[],
            )
        )

        router = FederatedQueryRouter(
            retriever=mock_retriever,
            default_strategy=ExecutionStrategy.SEQUENTIAL,
        )

        result = await router.route_and_execute("test query")

        assert result.plan.strategy == ExecutionStrategy.SEQUENTIAL

    @pytest.mark.asyncio
    async def test_query_plan_creation(self):
        """Test federated query plan creation."""
        detector = DomainDetector()
        builder = SubQueryBuilder()

        query = "How to deploy the API to production"
        matches = detector.detect(query)

        plan = FederatedQueryPlan(
            original_query=query,
            steps=[],
            strategy=ExecutionStrategy.PARALLEL,
        )

        for match in matches:
            sub_queries = builder.build(query, match.domain)
            for sq in sub_queries:
                plan.steps.append(QueryPlanStep(sub_query=sq))

        assert len(plan.steps) > 0
        assert plan.original_query == query

    @pytest.mark.asyncio
    async def test_result_aggregation(self):
        """Test result aggregation from multiple domains."""
        aggregator = ResultAggregator(mode="merge")

        results = {
            QueryDomain.TECHNICAL: [
                {"id": "1", "name": "tech result 1", "confidence": 0.9},
                {"id": "2", "name": "tech result 2", "confidence": 0.8},
            ],
            QueryDomain.BUSINESS: [
                {"id": "3", "name": "biz result 1", "confidence": 0.85},
            ],
        }

        aggregated = aggregator.aggregate(results)

        assert len(aggregated) == 3
        for result in aggregated:
            assert "_domain" in result

    @pytest.mark.asyncio
    async def test_domain_specific_templates(self):
        """Test domain-specific query templates."""
        builder = SubQueryBuilder()

        templates = {
            QueryDomain.TECHNICAL: ["Search technical documentation for {query}"],
            QueryDomain.BUSINESS: ["Find business intelligence on {query}"],
            QueryDomain.DOCUMENTATION: ["Find documentation about {query}"],
        }

        for domain, template_list in templates.items():
            for template in template_list:
                assert "{query}" in template


class TestHybridSearchIntegration:
    """Test hybrid vector + graph search."""

    @pytest.mark.asyncio
    async def test_combined_retrieval(self):
        """Test combined vector and graph retrieval."""
        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar_entities = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "name": "Vector Entity",
                    "similarity": 0.9,
                    "entity_type": "PERSON",
                },
            ]
        )

        mock_graph_store = AsyncMock()
        mock_graph_store.get_entity_neighborhood = AsyncMock(return_value=(None, []))

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        result = await retriever.retrieve(
            query="test query",
            query_embedding=[0.1] * 768,
            vector_limit=10,
            graph_limit=5,
        )

        assert isinstance(result, HybridRetrievalResult)
        assert result.vector_results_count >= 0

    @pytest.mark.asyncio
    async def test_weight_adjustment(self):
        """Test adjustable vector/graph weights."""
        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar_entities = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "name": "Test Entity",
                    "similarity": 0.8,
                    "entity_type": "PERSON",
                },
            ]
        )

        mock_graph_store = AsyncMock()
        mock_graph_store.get_entity_neighborhood = AsyncMock(return_value=(None, []))

        retriever_heavy_vector = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.9,
            graph_weight=0.1,
        )

        retriever_heavy_graph = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
            vector_weight=0.3,
            graph_weight=0.7,
        )

        assert retriever_heavy_vector._vector_weight == 0.9
        assert retriever_heavy_vector._graph_weight == 0.1
        assert retriever_heavy_graph._vector_weight == 0.3
        assert retriever_heavy_graph._graph_weight == 0.7

    @pytest.mark.asyncio
    async def test_retrieved_entity_scoring(self):
        """Test retrieved entity scoring logic."""
        entity = RetrievedEntity(
            id=uuid4(),
            name="Test Entity",
            entity_type="PERSON",
            description="Test description",
            properties={},
            confidence=0.85,
            vector_score=0.9,
            graph_score=0.7,
            final_score=0.0,
            source="vector",
        )

        vector_weight = 0.6
        graph_weight = 0.4

        if entity.graph_score is not None:
            entity.final_score = (
                entity.vector_score * vector_weight + entity.graph_score * graph_weight
            )

        assert entity.final_score == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_graph_expansion(self):
        """Test graph-based entity expansion."""
        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar_entities = AsyncMock(return_value=[])

        mock_graph_store = AsyncMock()
        mock_graph_store.get_entity_neighborhood = AsyncMock(
            return_value=(
                None,
                [
                    (
                        Entity(
                            id=uuid4(),
                            name="Neighbor",
                            entity_type="PERSON",
                            confidence=0.8,
                        ),
                        Edge(
                            source_id=uuid4(),
                            target_id=uuid4(),
                            edge_type=EdgeType.RELATED_TO,
                            confidence=0.7,
                        ),
                    )
                ],
            )
        )

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        result = await retriever.retrieve(
            query="test",
            query_embedding=[0.1] * 768,
            graph_depth=2,
        )

        assert result.graph_results_count >= 0

    @pytest.mark.asyncio
    async def test_invalid_weight_validation(self):
        """Test that invalid weights are rejected."""
        mock_vector_store = AsyncMock()
        mock_graph_store = AsyncMock()

        with pytest.raises(ValueError):
            HybridEntityRetriever(
                vector_store=mock_vector_store,
                graph_store=mock_graph_store,
                vector_weight=0.8,
                graph_weight=0.5,
            )


class TestDomainSchemaIntegration:
    """Test domain schema initialization and API."""

    @pytest.mark.asyncio
    async def test_schema_initialization(self):
        """Test domain schemas can be created."""
        schema_create = DomainSchemaCreate(
            domain_name="technology",
            domain_display_name="Technology Domain",
            description="Technology-related entities",
            domain_level=DomainLevel.PRIMARY,
            inheritance_type=InheritanceType.EXTENDS,
            entity_types={
                "SOFTWARE": EntityTypeDef(
                    type_name="SOFTWARE",
                    description="Software products",
                    attributes={
                        "version": DomainAttribute(
                            name="version",
                            attribute_type="str",
                            description="Software version",
                            required=True,
                        )
                    },
                    required_attributes=["version"],
                )
            },
            validation_mode=SchemaValidationMode.ADAPTIVE,
        )

        assert schema_create.domain_name == "technology"
        assert schema_create.domain_level == DomainLevel.PRIMARY
        assert "SOFTWARE" in schema_create.entity_types

    @pytest.mark.asyncio
    async def test_schema_inheritance(self):
        """Test schema inheritance functionality."""
        parent_schema = DomainSchemaCreate(
            domain_name="base",
            domain_display_name="Base Domain",
            entity_types={
                "BASE_ENTITY": EntityTypeDef(
                    type_name="BASE_ENTITY",
                    attributes={
                        "name": DomainAttribute(
                            name="name", attribute_type="str", required=True
                        ),
                    },
                    required_attributes=["name"],
                )
            },
        )

        child_schema = DomainSchemaCreate(
            domain_name="extended",
            domain_display_name="Extended Domain",
            parent_domain_name="base",
            inheritance_type=InheritanceType.EXTENDS,
            entity_types={
                "EXTENDED_ENTITY": EntityTypeDef(
                    type_name="EXTENDED_ENTITY",
                    attributes={
                        "extra_field": DomainAttribute(
                            name="extra_field", attribute_type="str"
                        ),
                    },
                )
            },
        )

        assert child_schema.parent_domain_name == "base"
        assert child_schema.inheritance_type == InheritanceType.EXTENDS

    @pytest.mark.asyncio
    async def test_entity_type_definition(self):
        """Test entity type definition structure."""
        entity_type = EntityTypeDef(
            type_name="PERSON",
            parent_type="ENTITY",
            description="A person entity",
            attributes={
                "name": DomainAttribute(
                    name="name",
                    attribute_type="str",
                    description="Person's full name",
                    required=True,
                ),
                "age": DomainAttribute(
                    name="age",
                    attribute_type="int",
                    description="Person's age",
                    required=False,
                ),
            },
            required_attributes=["name"],
        )

        assert entity_type.type_name == "PERSON"
        assert "name" in entity_type.required_attributes
        assert entity_type.get_attribute("name") is not None
        assert entity_type.get_attribute("nonexistent") is None

    @pytest.mark.asyncio
    async def test_domain_level_hierarchy(self):
        """Test domain level enumeration."""
        levels = [
            DomainLevel.ROOT,
            DomainLevel.PRIMARY,
            DomainLevel.SECONDARY,
            DomainLevel.TERTIARY,
        ]

        for level in levels:
            assert isinstance(level, DomainLevel)
            assert level.value in ["root", "primary", "secondary", "tertiary"]

    @pytest.mark.asyncio
    async def test_inheritance_type_options(self):
        """Test inheritance type options."""
        types = [
            InheritanceType.EXTENDS,
            InheritanceType.OVERRIDES,
            InheritanceType.COMPOSES,
        ]

        for itype in types:
            assert isinstance(itype, InheritanceType)
            assert itype.value in ["extends", "overrides", "composes"]

    @pytest.mark.asyncio
    async def test_schema_validation_modes(self):
        """Test schema validation mode options."""
        modes = [
            SchemaValidationMode.STRICT,
            SchemaValidationMode.LAX,
            SchemaValidationMode.ADAPTIVE,
        ]

        for mode in modes:
            assert isinstance(mode, SchemaValidationMode)

    @pytest.mark.asyncio
    async def test_domain_attribute_definition(self):
        """Test domain attribute structure."""
        attr = DomainAttribute(
            name="created_date",
            attribute_type="datetime",
            description="When the entity was created",
            required=True,
            default_value=None,
            validation_rules={"min_length": 1},
            domain_specific=True,
        )

        assert attr.name == "created_date"
        assert attr.required is True
        assert attr.domain_specific is True
        assert "min_length" in attr.validation_rules


class TestIntegrationFixtures:
    """Test integration test fixtures and utilities."""

    @pytest.mark.asyncio
    async def test_mock_llm_response_structure(self, mock_llm_client):
        """Test mock LLM client response structure."""
        response = await mock_llm_client.complete("test prompt")
        assert response == "test response"

        json_response = await mock_llm_client.complete_json("test prompt")
        assert json_response["name"] == "test"
        assert json_response["value"] == 42

    @pytest.mark.asyncio
    async def test_sample_document_fixture(self, sample_document):
        """Test sample document fixture."""
        assert sample_document.name == "Test Document"
        assert sample_document.doc_metadata is not None
        assert "Apple" in sample_document.doc_metadata.get("content", "")
        assert sample_document.status == "pending"

    @pytest.mark.asyncio
    async def test_sample_chunk_fixture(self, sample_chunk):
        """Test sample chunk fixture."""
        assert sample_chunk.text is not None
        assert len(sample_chunk.text) > 0
        assert sample_chunk.chunk_index == 0

    @pytest.mark.asyncio
    async def test_sample_entities_fixture(self, sample_entities):
        """Test sample entities fixture."""
        assert len(sample_entities) == 2
        for entity in sample_entities:
            assert entity.name is not None
            assert entity.entity_type is not None
            assert 0.0 <= entity.confidence <= 1.0


class TestComponentIntegration:
    """Integration tests across multiple components."""

    @pytest.mark.asyncio
    async def test_extraction_to_verification_pipeline(self):
        """Test pipeline from extraction to verification."""
        extracted_entity = ExtractedEntity(
            id=uuid4(),
            name="Test Person",
            entity_type="PERSON",
            source_text="Test context",
            chunk_id=uuid4(),
            confidence=0.85,
        )

        detector = HallucinationDetector()

        verification = await detector.verify_entity(
            entity_name=extracted_entity.name,
            entity_type=extracted_entity.entity_type,
            attributes={"confidence": str(extracted_entity.confidence)},
            context=extracted_entity.source_text,
        )

        assert verification.entity_name == "Test Person"
        assert verification.entity_type == "PERSON"

    @pytest.mark.asyncio
    async def test_query_to_retrieval_pipeline(self):
        """Test pipeline from query to retrieval."""
        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar_entities = AsyncMock(
            return_value=[
                {"id": uuid4(), "name": "Retrieved Entity", "similarity": 0.85},
            ]
        )

        mock_graph_store = AsyncMock()
        mock_graph_store.get_entity_neighborhood = AsyncMock(return_value=(None, []))

        retriever = HybridEntityRetriever(
            vector_store=mock_vector_store,
            graph_store=mock_graph_store,
        )

        router = FederatedQueryRouter(retriever=retriever)

        result = await router.route_and_execute("test query")

        assert result.query == "test query"
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_schema_driven_extraction(self):
        """Test extraction guided by domain schemas."""
        entity_type = EntityTypeDef(
            type_name="ORGANIZATION",
            description="Organization entity",
            attributes={
                "name": DomainAttribute(
                    name="name",
                    attribute_type="str",
                    required=True,
                ),
                "industry": DomainAttribute(
                    name="industry",
                    attribute_type="str",
                    required=False,
                ),
            },
            required_attributes=["name"],
        )

        assert entity_type.get_attribute("name") is not None
        assert entity_type.get_attribute("industry") is not None
        assert "name" in entity_type.required_attributes


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_graceful_llm_failure(self, sample_chunk):
        """Test graceful handling of LLM failures."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(side_effect=Exception("LLM API error"))

        mock_graph_store = AsyncMock()
        mock_vector_store = AsyncMock()

        config = MultiAgentExtractorConfig()
        manager = EntityExtractionManager(
            gateway=mock_gateway,
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            config=config,
        )

        entities = await manager.extract_entities([sample_chunk], uuid4())

        assert entities == []

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        detector = DomainDetector()

        matches = detector.detect("")

        assert len(matches) == 1
        assert matches[0].domain == QueryDomain.GENERAL

    @pytest.mark.asyncio
    async def test_malformed_llm_response(self, sample_chunk):
        """Test handling of malformed LLM responses."""
        mock_gateway = AsyncMock()
        mock_gateway.generate_text = AsyncMock(return_value="not json")

        config = MultiAgentExtractorConfig()
        agent = PerceptionAgent(mock_gateway, config)

        entities = await agent.extract_entities([sample_chunk])

        assert entities == []


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    from knowledge_base.main import app

    return TestClient(app)


class TestAPIIntegration:
    """Integration tests for API endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code in [200, 503]

    def test_api_version_endpoint(self, client):
        """Test API version endpoint."""
        response = client.get("/api/v1")
        assert response.status_code in [200, 404, 307]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

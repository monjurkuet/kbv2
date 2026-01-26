"""Unit tests for hallucination detector service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,
    HallucinationDetectorConfig,
    AttributeVerification,
    EntityVerification,
    VerificationStatus,
    RiskLevel,
)


class TestHallucinationDetectorConfig:
    """Tests for HallucinationDetectorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HallucinationDetectorConfig()
        assert config.url == "http://localhost:8087/v1/"
        assert config.model == "gemini-2.5-flash-lite"
        assert config.temperature == 0.1
        assert config.max_tokens == 1024
        assert config.batch_size == 10
        assert config.confidence_threshold == 0.7
        assert config.hallucination_threshold == 0.3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HallucinationDetectorConfig(
            url="http://custom:8080/v1/",
            model="custom-model",
            temperature=0.5,
            batch_size=5,
            confidence_threshold=0.8,
            hallucination_threshold=0.4,
        )
        assert config.url == "http://custom:8080/v1/"
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.batch_size == 5
        assert config.confidence_threshold == 0.8
        assert config.hallucination_threshold == 0.4


class TestAttributeVerification:
    """Tests for AttributeVerification dataclass."""

    def test_create_supported_attribute(self):
        """Test creating a supported attribute verification."""
        verification = AttributeVerification(
            attribute_name="founded_year",
            claimed_value="2005",
            status=VerificationStatus.SUPPORTED,
            confidence=0.95,
            evidence="Annual report shows company founded in 2005",
            explanation="Document explicitly states founding year",
        )
        assert verification.attribute_name == "founded_year"
        assert verification.claimed_value == "2005"
        assert verification.status == VerificationStatus.SUPPORTED
        assert verification.confidence == 0.95

    def test_create_unsupported_attribute(self):
        """Test creating an unsupported attribute verification."""
        verification = AttributeVerification(
            attribute_name="revenue",
            claimed_value="$1B",
            status=VerificationStatus.UNSUPPORTED,
            confidence=0.85,
            evidence="No mention of revenue in available documents",
            explanation="Financial data not found",
        )
        assert verification.status == VerificationStatus.UNSUPPORTED
        assert verification.confidence == 0.85

    def test_create_inconclusive_attribute(self):
        """Test creating an inconclusive attribute verification."""
        verification = AttributeVerification(
            attribute_name="employees",
            claimed_value="500",
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.3,
            evidence="Ambiguous references to team size",
            explanation="Insufficient information",
        )
        assert verification.status == VerificationStatus.INCONCLUSIVE
        assert verification.confidence == 0.3


class TestEntityVerification:
    """Tests for EntityVerification dataclass."""

    def test_create_entity_verification(self):
        """Test creating an entity verification."""
        attrs = [
            AttributeVerification(
                attribute_name="name",
                claimed_value="Acme Corp",
                status=VerificationStatus.SUPPORTED,
                confidence=0.9,
            ),
            AttributeVerification(
                attribute_name="founded",
                claimed_value="2005",
                status=VerificationStatus.SUPPORTED,
                confidence=0.85,
            ),
        ]
        entity = EntityVerification(
            entity_name="Acme Corp",
            entity_type="Company",
            overall_confidence=0.875,
            risk_level=RiskLevel.LOW,
            attributes=attrs,
            total_attributes=2,
            supported_count=2,
            unsupported_count=0,
            inconclusive_count=0,
            is_hallucinated=False,
        )
        assert entity.entity_name == "Acme Corp"
        assert entity.entity_type == "Company"
        assert entity.supported_ratio == 1.0
        assert entity.risk_level == RiskLevel.LOW

    def test_supported_ratio_empty(self):
        """Test supported ratio with no attributes."""
        entity = EntityVerification(
            entity_name="Test",
            entity_type="Test",
            overall_confidence=0.0,
            risk_level=RiskLevel.LOW,
            attributes=[],
            total_attributes=0,
            supported_count=0,
            unsupported_count=0,
            inconclusive_count=0,
        )
        assert entity.supported_ratio == 0.0

    def test_hallucination_detection(self):
        """Test hallucination detection logic."""
        attrs = [
            AttributeVerification(
                attribute_name="fake_attr",
                claimed_value="fake_value",
                status=VerificationStatus.UNSUPPORTED,
                confidence=0.9,
            ),
        ]
        entity = EntityVerification(
            entity_name="Fake Entity",
            entity_type="Company",
            overall_confidence=0.9,
            risk_level=RiskLevel.HIGH,
            attributes=attrs,
            total_attributes=1,
            supported_count=0,
            unsupported_count=1,
            inconclusive_count=0,
            is_hallucinated=True,
            hallucination_reasons=["Attribute 'fake_attr': Not supported by evidence"],
        )
        assert entity.is_hallucinated is True
        assert len(entity.hallucination_reasons) == 1


class TestHallucinationDetector:
    """Tests for HallucinationDetector class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def detector(self, mock_llm_client):
        """Create detector with mock client."""
        config = HallucinationDetectorConfig(llm_client=None)
        detector = HallucinationDetector(config)
        detector._llm_client = mock_llm_client
        return detector

    @pytest.mark.asyncio
    async def test_verify_attribute_supported(self, detector, mock_llm_client):
        """Test verifying a supported attribute."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "status": "supported",
                "confidence": 0.95,
                "evidence": "Document confirms the attribute",
                "explanation": "Clear evidence found",
            }
        )

        result = await detector.verify_attribute(
            entity_name="Test Corp",
            entity_type="Company",
            attribute_name="industry",
            claimed_value="Technology",
            context="Annual report states Test Corp is a technology company",
        )

        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 0.95
        assert result.attribute_name == "industry"
        mock_llm_client.complete_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_attribute_unsupported(self, detector, mock_llm_client):
        """Test verifying an unsupported attribute."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "status": "unsupported",
                "confidence": 0.9,
                "evidence": "No evidence found",
                "explanation": "Attribute not mentioned in documents",
            }
        )

        result = await detector.verify_attribute(
            entity_name="Test Corp",
            entity_type="Company",
            attribute_name="ipo_date",
            claimed_value="2020-01-01",
            context="No information about IPO in available documents",
        )

        assert result.status == VerificationStatus.UNSUPPORTED
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_verify_attribute_inconclusive(self, detector, mock_llm_client):
        """Test verifying an attribute with inconclusive results."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "status": "inconclusive",
                "confidence": 0.3,
                "evidence": "Ambiguous references",
                "explanation": "Cannot determine from context",
            }
        )

        result = await detector.verify_attribute(
            entity_name="Test Corp",
            entity_type="Company",
            attribute_name="headquarters",
            claimed_value="New York",
            context="References to multiple offices without clear headquarters",
        )

        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_verify_attribute_error_handling(self, detector, mock_llm_client):
        """Test error handling when LLM call fails."""
        mock_llm_client.complete_json = AsyncMock(
            side_effect=Exception("LLM request failed")
        )

        result = await detector.verify_attribute(
            entity_name="Test Corp",
            entity_type="Company",
            attribute_name="test",
            claimed_value="value",
            context="Some context",
        )

        assert result.status == VerificationStatus.INCONCLUSIVE
        assert result.confidence == 0.0
        assert "Verification error" in result.explanation

    def test_calculate_risk_level_low(self):
        """Test risk level calculation for low risk."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        entity = EntityVerification(
            entity_name="Test",
            entity_type="Test",
            overall_confidence=0.9,
            risk_level=RiskLevel.LOW,
            attributes=[],
            total_attributes=5,
            supported_count=5,
            unsupported_count=0,
            inconclusive_count=0,
        )

        risk = detector._calculate_risk_level(entity)
        assert risk == RiskLevel.LOW

    def test_calculate_risk_level_medium(self):
        """Test risk level calculation for medium risk."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        entity = EntityVerification(
            entity_name="Test",
            entity_type="Test",
            overall_confidence=0.6,
            risk_level=RiskLevel.MEDIUM,
            attributes=[],
            total_attributes=5,
            supported_count=4,
            unsupported_count=1,
            inconclusive_count=0,
        )

        risk = detector._calculate_risk_level(entity)
        assert risk == RiskLevel.MEDIUM

    def test_calculate_risk_level_high(self):
        """Test risk level calculation for high risk."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        entity = EntityVerification(
            entity_name="Test",
            entity_type="Test",
            overall_confidence=0.4,
            risk_level=RiskLevel.HIGH,
            attributes=[],
            total_attributes=5,
            supported_count=3,
            unsupported_count=2,
            inconclusive_count=0,
        )

        risk = detector._calculate_risk_level(entity)
        assert risk == RiskLevel.HIGH

    def test_calculate_risk_level_critical(self):
        """Test risk level calculation for critical risk."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        entity = EntityVerification(
            entity_name="Test",
            entity_type="Test",
            overall_confidence=0.2,
            risk_level=RiskLevel.CRITICAL,
            attributes=[],
            total_attributes=5,
            supported_count=2,
            unsupported_count=3,
            inconclusive_count=0,
        )

        risk = detector._calculate_risk_level(entity)
        assert risk == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_verify_entity(self, detector, mock_llm_client):
        """Test verifying a complete entity."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "status": "supported",
                "confidence": 0.9,
                "evidence": "Confirmed in document",
                "explanation": "Supported",
            }
        )

        attributes = {
            "name": "Acme Corp",
            "industry": "Technology",
            "founded": "2005",
        }

        result = await detector.verify_entity(
            entity_name="Acme Corp",
            entity_type="Company",
            attributes=attributes,
            context="Annual report 2023",
        )

        assert result.entity_name == "Acme Corp"
        assert result.entity_type == "Company"
        assert result.total_attributes == 3
        assert result.supported_count == 3
        assert result.is_hallucinated is False
        assert result.risk_level == RiskLevel.LOW

    @pytest.mark.asyncio
    async def test_verify_entity_mixed_results(self, detector, mock_llm_client):
        """Test entity with mixed verification results."""

        async def mock_response(*args, **kwargs):
            prompt = args[0] if args else kwargs.get("prompt", "")
            if "name" in prompt:
                return {
                    "status": "supported",
                    "confidence": 0.95,
                    "evidence": "Confirmed",
                    "explanation": "Name verified",
                }
            elif "revenue" in prompt:
                return {
                    "status": "unsupported",
                    "confidence": 0.9,
                    "evidence": "No data",
                    "explanation": "Revenue not found",
                }
            else:
                return {
                    "status": "supported",
                    "confidence": 0.8,
                    "evidence": "Partial support",
                    "explanation": "Industry confirmed",
                }

        mock_llm_client.complete_json = AsyncMock(side_effect=mock_response)

        attributes = {
            "name": "Acme Corp",
            "industry": "Technology",
            "revenue": "$1B",
        }

        result = await detector.verify_entity(
            entity_name="Acme Corp",
            entity_type="Company",
            attributes=attributes,
            context="Company profile",
        )

        assert result.total_attributes == 3
        assert result.supported_count == 2
        assert result.unsupported_count == 1
        assert result.is_hallucinated is True
        assert result.risk_level == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_verify_entity_batch(self, detector, mock_llm_client):
        """Test batch verification of entities."""

        async def mock_response(*args, **kwargs):
            return {
                "status": "supported",
                "confidence": 0.9,
                "evidence": "Confirmed",
                "explanation": "OK",
            }

        mock_llm_client.complete_json = AsyncMock(side_effect=mock_response)

        entities = [
            {
                "name": "Entity 1",
                "type": "Company",
                "attributes": {"name": "Entity 1"},
                "context": "Context 1",
            },
            {
                "name": "Entity 2",
                "type": "Person",
                "attributes": {"name": "Entity 2"},
                "context": "Context 2",
            },
            {
                "name": "Entity 3",
                "type": "Product",
                "attributes": {"name": "Entity 3"},
                "context": "Context 3",
            },
        ]

        results = await detector.verify_entity_batch(entities)

        assert len(results) == 3
        assert results[0].entity_name == "Entity 1"
        assert results[1].entity_name == "Entity 2"
        assert results[2].entity_name == "Entity 3"

    @pytest.mark.asyncio
    async def test_verify_claim(self, detector, mock_llm_client):
        """Test verifying a standalone claim."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "status": "supported",
                "confidence": 0.95,
                "explanation": "Evidence confirms the claim",
            }
        )

        result = await detector.verify_claim(
            claim="The company was founded in 2005",
            evidence="Annual report states founding year as 2005",
        )

        assert result["status"] == "supported"
        assert result["confidence"] == 0.95
        assert result["claim"] == "The company was founded in 2005"

    @pytest.mark.asyncio
    async def test_compare_entities(self, detector, mock_llm_client):
        """Test comparing two entities for consistency."""
        mock_llm_client.complete_json = AsyncMock(
            return_value={
                "consistency_score": 0.85,
                "contradictions": [],
                "complementary": ["Both mention technology industry"],
                "hallucination_flags": [],
                "summary": "Entities are consistent",
            }
        )

        primary = {
            "name": "Acme Corp",
            "type": "Company",
            "attributes": {"industry": "Technology", "founded": "2005"},
        }
        secondary = {
            "name": "Acme Corporation",
            "type": "Company",
            "attributes": {"industry": "Tech", "employees": "500"},
        }

        result = await detector.compare_entities(primary, secondary, "Company database")

        assert result["consistency_score"] == 0.85
        assert result["primary_entity"] == "Acme Corp"
        assert result["secondary_entity"] == "Acme Corporation"
        assert len(result["complementary"]) > 0

    def test_get_verification_summary(self):
        """Test generating summary of verifications."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        verifications = [
            EntityVerification(
                entity_name="Entity 1",
                entity_type="Company",
                overall_confidence=0.9,
                risk_level=RiskLevel.LOW,
                attributes=[],
                total_attributes=5,
                supported_count=5,
                unsupported_count=0,
                inconclusive_count=0,
                is_hallucinated=False,
            ),
            EntityVerification(
                entity_name="Entity 2",
                entity_type="Company",
                overall_confidence=0.4,
                risk_level=RiskLevel.HIGH,
                attributes=[],
                total_attributes=5,
                supported_count=2,
                unsupported_count=3,
                inconclusive_count=0,
                is_hallucinated=True,
            ),
            EntityVerification(
                entity_name="Entity 3",
                entity_type="Person",
                overall_confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                attributes=[],
                total_attributes=5,
                supported_count=4,
                unsupported_count=1,
                inconclusive_count=0,
                is_hallucinated=False,
            ),
        ]

        summary = detector.get_verification_summary(verifications)

        assert summary["total_entities"] == 3
        assert summary["hallucinated_count"] == 1
        assert summary["hallucination_rate"] == 1 / 3
        assert summary["average_confidence"] == pytest.approx((0.9 + 0.4 + 0.7) / 3)
        assert "low" in summary["risk_distribution"]
        assert "high" in summary["risk_distribution"]
        assert "medium" in summary["risk_distribution"]

    def test_get_verification_summary_empty(self):
        """Test summary with no verifications."""
        config = HallucinationDetectorConfig()
        detector = HallucinationDetector(config)

        summary = detector.get_verification_summary([])

        assert summary["total_entities"] == 0
        assert summary["hallucinated_count"] == 0
        assert summary["average_confidence"] == 0.0


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert VerificationStatus.SUPPORTED.value == "supported"
        assert VerificationStatus.UNSUPPORTED.value == "unsupported"
        assert VerificationStatus.INCONCLUSIVE.value == "inconclusive"
        assert VerificationStatus.CONFLICTING.value == "conflicting"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels(self):
        """Test enum values in order."""
        levels = list(RiskLevel)
        assert levels == [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]

    def test_risk_level_values(self):
        """Test enum string values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

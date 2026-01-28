"""Tests for domain detection module."""

import pytest
import pytest_asyncio

from knowledge_base.domain.domain_models import (
    Domain,
    DomainPrediction,
    DomainDetectionResult,
    DomainConfig,
)
from knowledge_base.domain.detection import DomainDetector
from knowledge_base.domain.ontology_snippets import DOMAIN_ONTOLOGIES


class TestDomainModels:
    """Test Pydantic domain models."""

    def test_domain_enum(self):
        """Test Domain enum values."""
        assert Domain.TECHNOLOGY.value == "TECHNOLOGY"
        assert Domain.FINANCIAL.value == "FINANCIAL"
        assert Domain.MEDICAL.value == "MEDICAL"
        assert Domain.LEGAL.value == "LEGAL"
        assert Domain.SCIENTIFIC.value == "SCIENTIFIC"
        assert Domain.GENERAL.value == "GENERAL"

    def test_domain_prediction(self):
        """Test DomainPrediction model."""
        pred = DomainPrediction(
            domain="TECHNOLOGY",
            confidence=0.85,
            supporting_evidence=["software: 5", "api: 3"],
            keyword_matches=["software", "api"],
            reasoning="Strong keyword matches",
        )
        assert pred.domain == "TECHNOLOGY"
        assert pred.confidence == 0.85
        assert len(pred.supporting_evidence) == 2

    def test_domain_prediction_confidence_bounds(self):
        """Test confidence is bounded between 0 and 1."""
        pred = DomainPrediction(domain="TECHNOLOGY", confidence=0.5)
        assert pred.confidence == 0.5

        with pytest.raises(ValueError):
            DomainPrediction(domain="TECHNOLOGY", confidence=1.5)

        with pytest.raises(ValueError):
            DomainPrediction(domain="TECHNOLOGY", confidence=-0.1)

    def test_domain_detection_result(self):
        """Test DomainDetectionResult model."""
        predictions = [
            DomainPrediction(domain="TECHNOLOGY", confidence=0.9),
            DomainPrediction(domain="SCIENTIFIC", confidence=0.3),
        ]
        result = DomainDetectionResult(
            primary_domain="TECHNOLOGY",
            all_domains=predictions,
            is_multi_domain=False,
            detection_method="hybrid",
            processing_time_ms=45.5,
            confidence_threshold=0.6,
        )
        assert result.primary_domain == "TECHNOLOGY"
        assert len(result.all_domains) == 2
        assert result.is_multi_domain is False
        assert result.processing_time_ms == 45.5

    def test_domain_config(self):
        """Test DomainConfig model."""
        config = DomainConfig(
            min_confidence=0.7,
            max_predictions=5,
            enable_keyword_screening=True,
            enable_llm_analysis=True,
            keyword_threshold=0.4,
        )
        assert config.min_confidence == 0.7
        assert config.max_predictions == 5
        assert config.enable_keyword_screening is True


class TestKeywordScreening:
    """Test keyword-based domain screening."""

    def test_technology_detection(self):
        """Test detection of technology-related content."""
        text = "The software uses a REST API to communicate with the cloud server. Python and JavaScript are the main programming languages."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None
        assert tech_pred.confidence > 0.5
        assert (
            "python" in tech_pred.keyword_matches
            or "javascript" in tech_pred.keyword_matches
        )

    def test_financial_detection(self):
        """Test detection of financial content."""
        text = "The company's quarterly earnings show revenue growth of 15%. Stock prices increased after the investment announcement."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        fin_pred = next((p for p in predictions if p.domain == "FINANCIAL"), None)
        assert fin_pred is not None
        assert fin_pred.confidence > 0.3

    def test_medical_detection(self):
        """Test detection of medical content."""
        text = "The patient was diagnosed with the disease. Clinical trials show the treatment is effective."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        med_pred = next((p for p in predictions if p.domain == "MEDICAL"), None)
        assert med_pred is not None
        assert med_pred.confidence > 0.3

    def test_empty_text(self):
        """Test handling of empty text."""
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync("")
        assert predictions == []

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync("   \n\t  ")
        assert predictions == []

    def test_no_keyword_matches(self):
        """Test text with no matching keywords."""
        text = "This is a generic text about various topics that don't match specific domains."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)
        assert predictions == []

    def test_keyword_normalization(self):
        """Test that keyword matching is case-insensitive."""
        text = "SOFTWARE and API and DATABASE are important"
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None
        assert tech_pred.confidence > 0


class TestDomainDetection:
    """Test full domain detection pipeline."""

    @pytest_asyncio.fixture
    async def detector(self):
        """Create a detector instance."""
        return DomainDetector()

    @pytest.mark.asyncio
    async def test_single_domain_detection(self, detector):
        """Test detection of a single primary domain."""
        text = "The new software framework provides an API for database integration."
        result = await detector.detect_domain(text)

        assert result.primary_domain == "TECHNOLOGY"
        assert result.is_multi_domain is False
        assert len(result.all_domains) >= 1
        assert result.detection_method in ["keyword", "hybrid"]

    @pytest.mark.asyncio
    async def test_multi_domain_detection(self, detector):
        """Test detection when content spans multiple domains."""
        text = "The research study analyzed financial data using machine learning algorithms."
        result = await detector.detect_domain(text)

        assert result.primary_domain in ["TECHNOLOGY", "FINANCIAL", "SCIENTIFIC"]
        assert len(result.all_domains) >= 1

    @pytest.mark.asyncio
    async def test_empty_document(self, detector):
        """Test handling of empty documents."""
        result = await detector.detect_domain("")

        assert result.primary_domain == "GENERAL"
        assert result.detection_method == "fallback"
        assert result.processing_time_ms == 0.0

    @pytest.mark.asyncio
    async def test_whitespace_document(self, detector):
        """Test handling of whitespace-only documents."""
        result = await detector.detect_domain("   \n\t  ")

        assert result.primary_domain == "GENERAL"
        assert result.detection_method == "fallback"

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self, detector):
        """Test that processing time is recorded."""
        text = "This is a test document with some technology keywords like software and code."
        result = await detector.detect_domain(text)

        assert result.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_confidence_threshold_applied(self, detector):
        """Test that confidence threshold is included in result."""
        text = "Software development is important."
        result = await detector.detect_domain(text)

        assert result.confidence_threshold == detector.config.min_confidence


class TestConfidenceCalibration:
    """Test confidence calibration between methods."""

    def test_keyword_confidence_normalization(self):
        """Test that keyword scores are properly normalized."""
        text = "software software software api api database server cloud"
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None
        assert tech_pred.confidence <= 1.0

    def test_multiple_domain_ranking(self):
        """Test that domains are ranked by confidence."""
        text = "The research study used software algorithms to analyze financial data."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        if len(predictions) > 1:
            for i in range(len(predictions) - 1):
                assert predictions[i].confidence >= predictions[i + 1].confidence


class TestDomainConfig:
    """Test DomainConfig customization."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DomainConfig()
        assert config.min_confidence == 0.6
        assert config.max_predictions == 3
        assert config.enable_keyword_screening is True
        assert config.enable_llm_analysis is True
        assert config.keyword_threshold == 0.3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DomainConfig(
            min_confidence=0.8,
            max_predictions=5,
            enable_keyword_screening=False,
            enable_llm_analysis=True,
        )
        assert config.min_confidence == 0.8
        assert config.max_predictions == 5
        assert config.enable_keyword_screening is False

    def test_config_with_detector(self):
        """Test detector with custom configuration."""
        config = DomainConfig(min_confidence=0.7, max_predictions=2)
        detector = DomainDetector(config=config)

        assert detector.config.min_confidence == 0.7
        assert detector.config.max_predictions == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test handling of very long text."""
        long_text = "software " * 1000 + "financial " * 500
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(long_text)

        assert len(predictions) >= 1

    def test_special_characters_in_text(self):
        """Test text with special characters."""
        text = "The @API #framework works with $database and [software]."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None

    def test_unicode_characters(self):
        """Test text with unicode characters."""
        text = "The company 上市 in China. 股价 increased after the earnings report."
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        fin_pred = next((p for p in predictions if p.domain == "FINANCIAL"), None)
        assert fin_pred is not None

    def test_mixed_case_keywords(self):
        """Test case-insensitive keyword matching."""
        text = "SOFTWARE API DATABASE Server Cloud Python"
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None
        assert tech_pred.confidence > 0

    def test_contribution_capping(self):
        """Test that individual keyword contributions are capped."""
        text = "software " * 20
        detector = DomainDetector()
        predictions = detector._keyword_screening_sync(text)

        tech_pred = next((p for p in predictions if p.domain == "TECHNOLOGY"), None)
        assert tech_pred is not None
        assert tech_pred.confidence <= 1.0

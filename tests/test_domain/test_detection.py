"""Tests for domain detection."""

import pytest

from knowledge_base.domain.detection import DomainDetector
from knowledge_base.domain.domain_models import DomainConfig


class TestDomainDetector:
    """Tests for DomainDetector class."""

    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = DomainDetector()
        assert detector.ontology is not None
        assert detector.config is not None

    def test_detector_with_custom_config(self):
        """Test detector with custom config."""
        config = DomainConfig(min_confidence=0.5)
        detector = DomainDetector(config=config)
        assert detector.config.min_confidence == 0.5

    @pytest.mark.asyncio
    async def test_detect_domain_with_crypto_content(self):
        """Test domain detection with Bitcoin content."""
        detector = DomainDetector()

        text = """
        Bitcoin ETF approval by the SEC marks a significant milestone 
        for institutional cryptocurrency adoption. The spot Bitcoin 
        ETF allows investors to gain exposure to Bitcoin without 
        directly holding the underlying asset.
        """

        result = await detector.detect_domain(text, top_k=3)

        # Should detect a crypto-related domain
        assert result.primary_domain in [
            "BITCOIN",
            "INSTITUTION_CRYPTO",
            "CRYPTO_MARKETS",
            "CRYPTO_TRADING",
        ]
        assert result.all_domains is not None
        assert len(result.all_domains) > 0
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_detect_domain_with_empty_text(self):
        """Test domain detection with empty text."""
        detector = DomainDetector()

        result = await detector.detect_domain("")

        assert result.primary_domain == "GENERAL"
        assert result.detection_method == "fallback"

    @pytest.mark.asyncio
    async def test_detect_domain_with_medical_content(self):
        """Test domain detection with medical content."""
        detector = DomainDetector()

        text = """
        The patient presented with symptoms of acute myocarditis.
        ECG shows ST elevation in leads V1-V4. Cardiac MRI confirmed
        the diagnosis with late gadolinium enhancement.
        """

        result = await detector.detect_domain(text, top_k=3)

        assert result.primary_domain in ["MEDICAL", "GENERAL"]

    @pytest.mark.asyncio
    async def test_detect_domain_keyword_only(self):
        """Test domain detection with keyword screening only."""
        config = DomainConfig(enable_keyword_screening=True, enable_llm_analysis=False)
        detector = DomainDetector(config=config)

        text = "Ethereum DeFi protocols Uniswap Aave lending yield farming"

        result = await detector.detect_domain(text, top_k=3)

        assert result.primary_domain == "DEFI"
        assert result.detection_method == "keyword"

"""Tests for EnhancedGateway wrapper."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from knowledge_base.common.gateway import GatewayConfig, EnhancedGateway, GatewayClient
from knowledge_base.clients import FewShotExample, PromptingStrategy


class TestSchema(BaseModel):
    """Test schema for structured extraction."""

    name: str
    value: int


@pytest.fixture
def enhanced_gateway():
    """Create an enhanced gateway with mocked LLM client."""
    config = GatewayConfig(
        url="http://localhost:8087/v1/",
        api_key="test-key",
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        max_tokens=4096,
    )
    gateway = EnhancedGateway(config)
    return gateway


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    return client


class TestEnhancedGateway:
    """Test EnhancedGateway class."""

    @pytest.mark.asyncio
    async def test_extract_with_reasoning(self, enhanced_gateway, mock_llm_client):
        """Test Chain-of-Thought extraction."""
        enhanced_gateway._llm_client = mock_llm_client

        mock_llm_client.complete_with_cot_steps = AsyncMock(return_value=("answer", []))

        result = await enhanced_gateway.extract_with_reasoning("test prompt")

        assert result["answer"] == "answer"
        assert "steps" in result
        mock_llm_client.complete_with_cot_steps.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_extract_with_cod(self, enhanced_gateway, mock_llm_client):
        """Test Chain-of-Draft extraction."""
        enhanced_gateway._llm_client = mock_llm_client

        mock_llm_client.complete_with_cod_steps = AsyncMock(
            return_value=("draft answer", [])
        )

        result = await enhanced_gateway.extract_with_cod("test prompt")

        assert result["answer"] == "draft answer"
        assert "steps" in result
        mock_llm_client.complete_with_cod_steps.assert_called_once_with("test prompt")

    @pytest.mark.asyncio
    async def test_extract_structured(self, enhanced_gateway, mock_llm_client):
        """Test structured JSON extraction."""
        enhanced_gateway._llm_client = mock_llm_client

        mock_llm_client.complete_json = AsyncMock(
            return_value={"name": "test", "value": 42}
        )

        result = await enhanced_gateway.extract_structured("extract info", TestSchema)

        assert isinstance(result, TestSchema)
        assert result.name == "test"
        assert result.value == 42
        mock_llm_client.complete_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_few_shot(self, enhanced_gateway, mock_llm_client):
        """Test few-shot extraction."""
        enhanced_gateway._llm_client = mock_llm_client

        mock_llm_client.complete = AsyncMock(return_value="extracted content")

        examples = [
            FewShotExample(input="input1", output="output1"),
            FewShotExample(input="input2", output="output2"),
        ]

        result = await enhanced_gateway.extract_few_shot("test prompt", examples)

        assert result["content"] == "extracted content"
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, enhanced_gateway):
        """Test that existing GatewayClient methods still work."""
        with patch.object(
            enhanced_gateway, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "test-id",
                "created": 1234567890,
                "model": "test-model",
                "choices": [{"message": {"content": "test response"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            response = await enhanced_gateway.generate_text("test prompt")

            assert response == "test response"

    @pytest.mark.asyncio
    async def test_lazy_llm_client_creation(self, enhanced_gateway):
        """Test that LLM client is created lazily."""
        assert enhanced_gateway._llm_client is None

        with patch(
            "knowledge_base.common.gateway.create_llm_client", new_callable=AsyncMock
        ) as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client

            client = await enhanced_gateway._get_llm_client()

            assert client is mock_client
            mock_create.assert_called_once()

"""Test ModelRotationManager functionality."""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock


def test_model_rotation_manager_creation():
    """Test ModelRotationManager can be created."""
    from knowledge_base.clients.rotation_manager import (
        ModelRotationManager,
        ModelRotationConfig,
        GatewayConfig,
    )

    config = GatewayConfig(url="http://localhost:8087/v1/")
    rotation_config = ModelRotationConfig(
        models_endpoint="http://localhost:8087/v1/models"
    )
    manager = ModelRotationManager(config=config, rotation_config=rotation_config)

    assert manager is not None
    assert manager._rotation_config.rotation_delay >= 5.0, (
        "Retry delay must be at least 5 seconds"
    )


def test_default_models_ar_ordered_correctly():
    """Test that default fallback models are ordered for optimal rotation."""
    from knowledge_base.clients.rotation_manager import ModelRotationConfig

    config = ModelRotationConfig()

    expected_models = [
        "kimi-k2-thinking",
        "qwen3-max",
        "glm-4.7",
        "deepseek-v3.2-reasoner",
        "gemini-2.5-flash-lite",
    ]
    assert config.fallback_models == expected_models
    assert config.rotation_delay == 5.0, "Rotation delay must be exactly 5 seconds"


@pytest.mark.asyncio
async def test_fetch_available_models():
    """Test fetching models from endpoint."""
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    mock_response = Mock()
    mock_response.json.return_value = {
        "data": [
            {"id": "model1", "created": 123456, "owned_by": "provider1"},
            {"id": "model2", "created": 234567, "owned_by": "provider2"},
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        manager = ModelRotationManager()

        models = await manager._fetch_available_models()

        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models


@pytest.mark.asyncio
async def test_model_fallback_on_fetch_failure():
    """Test that fallback models are used when fetch fails."""
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection failed")):
        manager = ModelRotationManager()
        manager._rotation_config.fallback_models = [
            "test-model-1",
            "test-model-2",
        ]

        models = await manager._fetch_available_models()

        assert models == ["test-model-1", "test-model-2"]


@pytest.mark.asyncio
async def test_is_rate_limit_error():
    """Test rate limit error detection."""
    import httpx
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    manager = ModelRotationManager()

    mock_response = Mock()
    mock_response.status_code = 429

    rate_limit_error = httpx.HTTPStatusError(
        "Too many requests",
        request=Mock(),
        response=mock_response,
    )

    assert manager._is_rate_limit_error(rate_limit_error) is True

    other_error = httpx.HTTPStatusError(
        "Bad request",
        request=Mock(),
        response=Mock(status_code=400),
    )

    assert manager._is_rate_limit_error(other_error) is False

    # Test message-based detection
    api_error = Exception("API rate limit exceeded, try again later")
    assert manager._is_rate_limit_error(api_error) is True


@pytest.mark.asyncio
async def test_get_next_model_rotation():
    """Test model rotation cycles correctly."""
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    manager = ModelRotationManager()
    manager._models = ["model1", "model2", "model3"]
    manager._current_model_index = 0

    assert manager._get_next_model() == "model1"
    assert manager._get_next_model() == "model2"
    assert manager._get_next_model() == "model3"
    assert manager._get_next_model() == "model1"  # Should cycle back


def test_extract_content_from_response():
    """Test content extraction from chat completion response."""
    from knowledge_base.clients.rotation_manager import ModelRotationManager
    from knowledge_base.common.gateway import ChatCompletionResponse

    manager = ModelRotationManager()

    mock_response = ChatCompletionResponse(
        id="test-123",
        created=123456,
        model="test-model",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "Generated text response",
                }
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )

    content = manager._extract_content(mock_response)
    assert content == "Generated text response"


def test_call_llm_interface_signature():
    """Test that call_llm has the expected interface for orchestrator."""
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    manager = ModelRotationManager()

    import inspect

    sig = inspect.signature(manager.call_llm)
    params = list(sig.parameters.keys())

    expected_params = [
        "messages",
        "model",
        "temperature",
        "max_tokens",
        "response_format",
    ]

    for param in expected_params:
        assert param in params, f"call_llm missing expected parameter: {param}"


def test_create_rotation_manager_factory():
    """Test factory function for creating rotation manager."""
    from knowledge_base.clients.rotation_manager import create_rotation_manager

    manager = create_rotation_manager(
        gateway_url="http://test:8087/v1/",
        models_endpoint="http://test:8087/v1/models",
        fallback_models=["test-model"],
    )

    assert manager is not None
    assert manager._config.url == "http://test:8087/v1/"


if __name__ == "__main__":
    print("Running ModelRotationManager tests...")

    test_model_rotation_manager_creation()
    print("✓ ModelRotationManager creation test passed")

    test_default_models_ar_ordered_correctly()
    print("✓ Default models configuration test passed")

    asyncio.run(test_fetch_available_models())
    print("✓ Fetch available models test passed")

    asyncio.run(test_model_fallback_on_fetch_failure())
    print("✓ Model fallback test passed")

    asyncio.run(test_is_rate_limit_error())
    print("✓ Rate limit error detection test passed")

    asyncio.run(test_get_next_model_rotation())
    print("✓ Model rotation test passed")

    test_extract_content_from_response()
    print("✓ Content extraction test passed")

    test_call_llm_interface_signature()
    print("✓ Interface signature test passed")

    test_create_rotation_manager_factory()
    print("✓ Factory function test passed")

    print("\nAll tests passed! ✓")

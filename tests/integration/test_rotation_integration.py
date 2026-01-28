"""Integration test to verify model rotation logic with actual HTTP errors."""

import asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock
import pytest

from knowledge_base.common.resilient_gateway.gateway import (
    ResilientGatewayClient,
    ResilientGatewayConfig,
)
from knowledge_base.common.gateway import ChatMessage


@pytest.mark.asyncio
async def test_resilient_gateway_rotates_on_429():
    """Test that ResilientGatewayClient rotates models when encountering 429 errors."""

    # Configure with fast settings for testing
    config = ResilientGatewayConfig(
        url="http://localhost:8087/v1/",
        model="test-model-1",
        temperature=0.7,
        max_tokens=1000,
        retry_max_attempts=1,  # Only 1 retry per model
        retry_base_delay=0.01,  # Very fast
        retry_jitter=False,
        retry_on_status_codes=[429, 500],
        model_switching_enabled=True,
        fallback_models=["test-model-2", "test-model-3"],
    )

    client = ResilientGatewayClient(config)
    messages = [ChatMessage(role="user", content="Test prompt")]

    # Track which models were called
    called_models = []
    attempt_count = 0

    async def mock_chat_completion(*args, **kwargs):
        nonlocal attempt_count
        model = kwargs.get("model", "unknown")
        called_models.append(model)
        attempt_count += 1

        # First model returns 429, second returns 429, third succeeds
        if len(called_models) <= 2:
            raise httpx.HTTPStatusError(
                f"Rate limit on {model}", request=Mock(), response=Mock(status_code=429)
            )
        else:
            # Success on third model
            from knowledge_base.common.gateway import ChatCompletionResponse

            return ChatCompletionResponse(
                id="test-123",
                created=123456,
                model=model,
                choices=[{"message": {"role": "assistant", "content": "Success!"}}],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

    # Patch the base client's chat_completion
    with patch.object(
        client._base_client, "chat_completion", side_effect=mock_chat_completion
    ):
        try:
            response = await client.chat_completion(messages)

            # Verify rotation happened
            assert len(called_models) >= 2, (
                f"Expected at least 2 model calls, got {len(called_models)}"
            )
            assert "test-model-1" in called_models, (
                "First model should have been called"
            )
            assert "test-model-2" in called_models, (
                "Second model (fallback) should have been called"
            )
            assert response.model == "test-model-3" or len(called_models) >= 3, (
                f"Should have reached third model or completed on second"
            )

            print(f"✅ Model rotation working! Models tried: {called_models}")

        finally:
            await client.close()


@pytest.mark.asyncio
async def test_resilient_gateway_retries_429_with_exponential_backoff():
    """Test that 429 errors trigger retries with exponential backoff before rotation."""

    config = ResilientGatewayConfig(
        url="http://localhost:8087/v1/",
        model="test-model",
        retry_max_attempts=3,  # 3 retries before giving up
        retry_base_delay=0.01,  # 10ms for fast testing
        retry_jitter=False,
        retry_on_status_codes=[429],
        model_switching_enabled=False,  # Disable model switching to test retry only
    )

    client = ResilientGatewayClient(config)
    messages = [ChatMessage(role="user", content="Test prompt")]

    call_times = []

    async def mock_chat_completion(*args, **kwargs):
        import time

        call_times.append(time.time())

        # Fail first 2 attempts, succeed on 3rd
        if len(call_times) <= 2:
            raise httpx.HTTPStatusError(
                "Rate limited", request=Mock(), response=Mock(status_code=429)
            )

        from knowledge_base.common.gateway import ChatCompletionResponse

        return ChatCompletionResponse(
            id="test-123",
            created=123456,
            model=kwargs.get("model", "test-model"),
            choices=[
                {"message": {"role": "assistant", "content": "Success after retries!"}}
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    with patch.object(
        client._base_client, "chat_completion", side_effect=mock_chat_completion
    ):
        try:
            response = await client.chat_completion(messages)

            # Should have attempted 3 times
            assert len(call_times) == 3, f"Expected 3 attempts, got {len(call_times)}"

            # Verify exponential backoff timing (roughly): 10ms, 20ms delays
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]

            assert 0.005 <= delay1 <= 0.02, (
                f"First delay should be ~10ms, got {delay1}s"
            )
            assert 0.01 <= delay2 <= 0.04, (
                f"Second delay should be ~20ms, got {delay2}s"
            )

            print(
                f"✅ Exponential backoff working! Attempts: {len(call_times)}, delays: {delay1:.4f}s, {delay2:.4f}s"
            )

        finally:
            await client.close()


@pytest.mark.asyncio
async def test_model_rotation_manager_rotates_on_error():
    """Test ModelRotationManager with actual error propagation."""
    from knowledge_base.clients.rotation_manager import (
        ModelRotationManager,
        ModelRotationConfig,
    )

    config = ModelRotationConfig(
        fallback_models=["model-a", "model-b", "model-c"],
        rotation_delay=0.01,  # Fast for testing
        max_rotation_attempts=3,
    )

    manager = ModelRotationManager(rotation_config=config)
    messages = [{"role": "user", "content": "Test"}]

    attempted_models = []

    async def mock_chat_completion(*args, **kwargs):
        model = kwargs.get("model")
        attempted_models.append(model)

        # First two models fail with 429, third succeeds
        index = len(attempted_models)
        if index <= 2:
            raise httpx.HTTPStatusError(
                f"Rate limited on attempt {index}",
                request=Mock(),
                response=Mock(status_code=429),
            )

        from knowledge_base.common.gateway import ChatCompletionResponse

        return ChatCompletionResponse(
            id="test-123",
            created=123456,
            model=model,
            choices=[{"message": {"role": "assistant", "content": "Success!"}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    # Patch gateway client
    with patch.object(
        manager._gateway_client, "chat_completion", side_effect=mock_chat_completion
    ):
        try:
            result = await manager.call_llm(messages)

            # Verify rotation worked
            assert len(attempted_models) >= 2, (
                f"Expected at least 2 models tried, got {len(attempted_models)}"
            )
            assert result["success"] is True, f"Expected success, got: {result}"
            assert result["attempts"] >= 2, (
                f"Expected multiple attempts, got {result['attempts']}"
            )
            assert len(result.get("models_tried", [])) >= 2, (
                "Should have tried multiple models"
            )

            print(
                f"✅ ModelRotationManager rotation working! Models: {attempted_models}, result: {result}"
            )

        finally:
            await manager.close()


if __name__ == "__main__":
    print("Running model rotation integration tests...\n")

    asyncio.run(test_resilient_gateway_rotates_on_429())
    print()

    asyncio.run(test_resilient_gateway_retries_429_with_exponential_backoff())
    print()

    asyncio.run(test_model_rotation_manager_rotates_on_error())
    print("\nAll integration tests passed! ✓")

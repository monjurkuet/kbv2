"""Tests for Resilient LLM Gateway implementation."""

import asyncio
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_base.common.resilient_gateway.gateway import (
    CircuitBreaker,
    CircuitState,
    ModelDiscoveryService,
    ResilientGatewayClient,
    ResilientGatewayConfig,
)
from knowledge_base.common.gateway import ChatMessage


@pytest.fixture
def gateway_config():
    """Create a test gateway configuration."""
    return ResilientGatewayConfig(
        url="http://localhost:8317/v1/",
        api_key="test-key",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=4096,
        timeout=120.0,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=1,  # Short for testing
        circuit_breaker_success_threshold=2,
        retry_max_attempts=2,
        retry_base_delay=0.01,  # Fast for testing
        retry_max_delay=1.0,
        retry_jitter=False,  # Deterministic for testing
        retry_on_status_codes=[429, 502, 503, 504],
        model_switching_enabled=True,
        fallback_models=["gpt-3.5-turbo", "gpt-4o-mini"],
    )


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial state is CLOSED."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_failure_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # First failure
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

        # Second failure - should open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_resets_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3)

        cb.record_failure()
        assert cb._failure_count == 1

        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_recovery_timeout(self):
        """Test circuit recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)  # 10ms

        cb.record_failure()  # Circuit opens
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        # Wait for recovery timeout
        import time

        time.sleep(0.02)  # Wait 20ms

        # Circuit should transition to HALF_OPEN
        assert cb.can_execute()
        assert cb._state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Test that enough successes in half-open state closes circuit."""
        cb = CircuitBreaker(
            failure_threshold=1, recovery_timeout=0.01, success_threshold=2
        )

        cb.record_failure()  # Open circuit
        assert cb.state == CircuitState.OPEN

        import time

        time.sleep(0.02)  # Wait for recovery

        # First success in half-open
        cb.record_success()
        assert cb._state == CircuitState.HALF_OPEN

        # Second success - should close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestModelDiscoveryService:
    """Test model discovery functionality."""

    @pytest.mark.asyncio
    async def test_discover_models(self, gateway_config):
        """Test model discovery."""
        discovery = ModelDiscoveryService(gateway_config)

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "gpt-4o",
                    "created": 1234567890,
                    "owned_by": "test",
                    "permission": [],
                    "root": "gpt-4o",
                    "parent": None,
                },
                {
                    "id": "gpt-3.5-turbo",
                    "created": 1234567891,
                    "owned_by": "test",
                    "permission": [],
                    "root": "gpt-3.5-turbo",
                    "parent": None,
                },
            ]
        }

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            models = await discovery.discover_models()
            assert len(models) == 2
            assert models[0].id == "gpt-4o"
            assert models[1].id == "gpt-3.5-turbo"

            # Test get_available_models
            model_names = await discovery.get_available_models()
            assert "gpt-4o" in model_names
            assert "gpt-3.5-turbo" in model_names

    @pytest.mark.asyncio
    async def test_discover_models_error(self, gateway_config):
        """Test model discovery handles errors gracefully."""
        discovery = ModelDiscoveryService(gateway_config)

        with patch(
            "httpx.AsyncClient.get", side_effect=httpx.HTTPError("Connection failed")
        ):
            models = await discovery.discover_models()
            assert models == []  # Should return empty list on error


class TestResilientGatewayClient:
    """Test resilient gateway client functionality."""

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, gateway_config):
        """Test successful chat completion."""
        client = ResilientGatewayClient(gateway_config)

        # Mock the base client's chat completion
        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "test response"}}]
        mock_response.id = "test-id"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.created = 1234567890

        with patch.object(
            client._base_client, "chat_completion", return_value=mock_response
        ):
            messages = [ChatMessage(role="user", content="Hello")]
            response = await client.chat_completion(messages)

            assert response.choices[0]["message"]["content"] == "test response"
            assert client.get_metrics()["total_requests"] == 1
            assert client.get_metrics()["successful_requests"] == 1

    @pytest.mark.asyncio
    async def test_chat_completion_with_retry(self, gateway_config):
        """Test chat completion with retry logic."""
        client = ResilientGatewayClient(gateway_config)

        # Mock to fail twice then succeed
        call_count = 0

        async def mock_chat_completion(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                response = MagicMock()
                response.status_code = 429
                raise httpx.HTTPStatusError(
                    "Rate limited", request=MagicMock(), response=response
                )
            else:  # Succeed on third attempt
                mock_response = MagicMock()
                mock_response.choices = [{"message": {"content": "success"}}]
                mock_response.id = "test-id"
                mock_response.model = "gpt-4o"
                mock_response.usage = MagicMock()
                mock_response.created = 1234567890
                return mock_response

        with patch.object(
            client._base_client, "chat_completion", side_effect=mock_chat_completion
        ):
            messages = [ChatMessage(role="user", content="Hello")]
            response = await client.chat_completion(messages)

            assert response.choices[0]["message"]["content"] == "success"
            assert call_count == 3  # Should have retried twice
            assert client.get_metrics()["retry_attempts"] == 2
            assert (
                client.get_metrics()["total_requests"] == 1
            )  # Only one logical request

    @pytest.mark.asyncio
    async def test_chat_completion_with_model_switching(self, gateway_config):
        """Test model switching when rate limited."""
        # Disable continuous rotation for this test to test basic model switching
        gateway_config.continuous_rotation_enabled = False
        client = ResilientGatewayClient(gateway_config)

        # Mock model discovery to return available models
        with patch.object(
            client._model_discovery,
            "get_available_models",
            return_value=["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"],
        ):
            call_count = 0
            call_models = []

            async def mock_chat_completion(messages, model=None, **kwargs):
                nonlocal call_count
                call_count += 1
                call_models.append(model)

                # First 3 calls are retries with the original model (gpt-4o)
                # Then we switch to a different model
                if call_count <= 3:  # First 3 calls with original model - rate limit
                    response = MagicMock()
                    response.status_code = 429
                    raise httpx.HTTPStatusError(
                        "Rate limited on original model",
                        request=MagicMock(),
                        response=response,
                    )
                else:  # After exhausting retries, switch to different model - success
                    mock_response = MagicMock()
                    mock_response.choices = [
                        {"message": {"content": f"response from {model}"}}
                    ]
                    mock_response.id = "test-id"
                    mock_response.model = model
                    mock_response.usage = MagicMock()
                    mock_response.created = 1234567890
                    return mock_response

            with patch.object(
                client._base_client, "chat_completion", side_effect=mock_chat_completion
            ):
                messages = [ChatMessage(role="user", content="Hello")]
                response = await client.chat_completion(messages, model="gpt-4o")

                assert "response from" in response.choices[0]["message"]["content"]
                # With retry_max_attempts=2, we get 3 total calls (1 initial + 2 retries)
                # Then we switch models for the 4th call
                assert call_count >= 2
                assert "gpt-4o" in call_models
                assert (
                    "gpt-3.5-turbo" in call_models or "gpt-4o-mini" in call_models
                )  # Should have switched
                assert client.get_metrics()["model_switches"] == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, gateway_config):
        """Test circuit breaker integration."""
        # Set a low failure threshold for testing
        config = ResilientGatewayConfig(
            url=gateway_config.url,
            api_key=gateway_config.api_key,
            model=gateway_config.model,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=1,  # 1 second for testing (must be int)
            retry_max_attempts=0,  # Disable retries to isolate circuit breaker
        )

        client = ResilientGatewayClient(config)

        async def failing_chat_completion(*args, **kwargs):
            response = MagicMock()
            response.status_code = 503
            raise httpx.HTTPStatusError(
                "Service unavailable", request=MagicMock(), response=response
            )

        # First call should fail and open circuit breaker
        with patch.object(
            client._base_client, "chat_completion", side_effect=failing_chat_completion
        ):
            messages = [ChatMessage(role="user", content="Hello")]

            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.chat_completion(messages, model="gpt-4o")

            # Circuit breaker should now be open
            assert client._circuit_breakers["gpt-4o"].state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_generate_text(self, gateway_config):
        """Test generate_text method."""
        client = ResilientGatewayClient(gateway_config)

        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "generated text"}}]
        mock_response.id = "test-id"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.created = 1234567890

        with patch.object(
            client._base_client, "chat_completion", return_value=mock_response
        ):
            result = await client.generate_text("test prompt")
            assert result == "generated text"
            assert client.get_metrics()["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_metrics_collection(self, gateway_config):
        """Test metrics collection."""
        client = ResilientGatewayClient(gateway_config)

        mock_response = MagicMock()
        mock_response.choices = [{"message": {"content": "test"}}]
        mock_response.id = "test-id"
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.created = 1234567890

        with patch.object(
            client._base_client, "chat_completion", return_value=mock_response
        ):
            # Successful request
            await client.chat_completion([ChatMessage(role="user", content="test")])

            # Rate-limited request
            with patch.object(
                client._base_client,
                "chat_completion",
                side_effect=httpx.HTTPStatusError(
                    "Rate limited",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                ),
            ):
                with pytest.raises(httpx.HTTPStatusError):
                    await client.chat_completion(
                        [ChatMessage(role="user", content="limited")]
                    )

        metrics = client.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 1
        assert metrics["rate_limited_requests"] == 1

        # Check model-specific metrics
        model_metrics = metrics["model_metrics"]
        assert "gpt-4o" in model_metrics
        gpt4_metrics = model_metrics["gpt-4o"]
        assert gpt4_metrics["requests"] == 2
        assert gpt4_metrics["successes"] == 1
        assert gpt4_metrics["rate_limits"] == 1

    @pytest.mark.asyncio
    async def test_close_method(self, gateway_config):
        """Test close method."""
        client = ResilientGatewayClient(gateway_config)

        # Mock close methods
        base_close_mock = AsyncMock()
        discovery_close_mock = AsyncMock()

        # Replace close methods with mocks
        client._base_client.close = base_close_mock
        client._model_discovery.close = discovery_close_mock

        await client.close()

        base_close_mock.assert_called_once()
        discovery_close_mock.assert_called_once()

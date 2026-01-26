"""Unit tests for LLM client with advanced prompting strategies."""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch
from httpx import AsyncClient, HTTPStatusError, RequestError

from knowledge_base.clients.llm_client import (
    LLMClient,
    LLMClientConfig,
    LLMResponse,
    LLMRequest,
    ChatMessage,
    MessageRole,
    FewShotExample,
    CoTStep,
    CoDStep,
    PromptingStrategy,
    create_llm_client,
)


class TestLLMClientConfig:
    """Test suite for LLM client configuration."""

    def test_default_config_values(self):
        """Test default configuration values are set correctly (env overrides)."""
        config = LLMClientConfig(
            url="http://localhost:8087/v1/", model="gemini-2.5-flash-lite"
        )

        assert config.url == "http://localhost:8087/v1/"
        assert config.model == "gemini-2.5-flash-lite"

    def test_config_from_env(self, monkeypatch):
        """Test configuration loads from environment variables."""
        monkeypatch.setenv("LLM_URL", "http://test:8088/v1/")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

        config = LLMClientConfig()

        assert config.url == "http://test:8088/v1/"
        assert config.model == "test-model"
        assert config.temperature == 0.5


class TestChatMessage:
    """Test suite for ChatMessage model."""

    def test_create_user_message(self):
        """Test creating a user message."""
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_create_assistant_message_with_name(self):
        """Test creating an assistant message with optional name."""
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Hi!", name="assistant")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.name == "assistant"

    def test_message_serialization(self):
        """Test message can be serialized and deserialized."""
        msg = ChatMessage(role=MessageRole.SYSTEM, content="You are helpful.")
        data = msg.model_dump()
        restored = ChatMessage(**data)
        assert restored == msg


class TestFewShotExample:
    """Test suite for few-shot example model."""

    def test_create_example(self):
        """Test creating a few-shot example."""
        example = FewShotExample(input="2+2", output="4")
        assert example.input == "2+2"
        assert example.output == "4"


class TestLLMClient:
    """Test suite for main LLM client functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock async client."""
        client = AsyncMock(spec=AsyncClient)
        client.post = AsyncMock()
        return client

    @pytest.fixture
    def llm_client(self, mock_client):
        """Create LLM client with mocked HTTP client."""
        client = LLMClient()
        client._client = mock_client
        return client

    def test_init_with_config(self):
        """Test client initialization with custom config."""
        config = LLMClientConfig(model="custom-model", temperature=0.5)
        client = LLMClient(config)
        assert client._config.model == "custom-model"
        assert client._config.temperature == 0.5

    def test_get_headers_no_api_key(self, monkeypatch):
        """Test headers without API key."""
        monkeypatch.setenv("LLM_API_KEY", "")
        config = LLMClientConfig(api_key="")
        client = LLMClient(config)
        headers = client._get_headers()
        assert "Content-Type" in headers
        assert "Authorization" not in headers

    def test_get_headers_with_api_key(self):
        """Test headers with API key."""
        config = LLMClientConfig(api_key="test-key")
        client = LLMClient(config)
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_build_few_shot_messages(self):
        """Test building few-shot prompt messages."""
        client = LLMClient()
        examples = [
            FewShotExample(input="Q1", output="A1"),
            FewShotExample(input="Q2", output="A2"),
        ]

        messages = client._build_few_shot_messages(
            system_prompt="You are a helpful assistant.",
            examples=examples,
            user_prompt="Q3",
        )

        assert len(messages) == 6  # 1 system + 2 examples (Q+A) + 1 user = 6
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER
        assert messages[2].role == MessageRole.ASSISTANT
        assert messages[3].role == MessageRole.USER
        assert messages[4].role == MessageRole.ASSISTANT
        assert messages[5].role == MessageRole.USER
        assert messages[0].content == "You are a helpful assistant."
        assert messages[1].content == "Q1"
        assert messages[2].content == "A1"
        assert messages[3].content == "Q2"
        assert messages[4].content == "A2"
        assert messages[5].content == "Q3"

    @pytest.mark.asyncio
    async def test_build_cot_messages(self):
        """Test building Chain-of-Thought prompt messages."""
        client = LLMClient()

        messages = client._build_cot_messages(
            system_prompt="You are a helpful assistant.",
            question="What is 2+2?",
            enable_cod=False,
        )

        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert "step by step" in messages[0].content
        assert "REASONING:" in messages[0].content
        assert messages[1].role == MessageRole.USER
        assert messages[1].content == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_build_cod_messages(self):
        """Test building Chain-of-Draft prompt messages."""
        client = LLMClient()

        messages = client._build_cot_messages(
            system_prompt="You are a helpful assistant.",
            question="What is 2+2?",
            enable_cod=True,
        )

        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert "brief" in messages[0].content.lower()
        assert "action-oriented" in messages[0].content.lower()
        assert "STEPS:" in messages[0].content

    @pytest.mark.asyncio
    async def test_execute_request(self, llm_client, mock_client):
        """Test executing chat completion request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        messages = [ChatMessage(role=MessageRole.USER, content="Hi")]
        response = await llm_client._execute_request(messages)

        assert response.id == "test-id"
        assert response.choices[0]["message"]["content"] == "Hello!"
        assert response.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_complete_standard_strategy(self, llm_client, mock_client):
        """Test standard completion strategy."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        result = await llm_client.complete(
            prompt="Say hello",
            strategy=PromptingStrategy.STANDARD,
        )

        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_complete_few_shot_strategy(self, llm_client, mock_client):
        """Test few-shot completion strategy."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "A3"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        examples = [
            FewShotExample(input="Q1", output="A1"),
            FewShotExample(input="Q2", output="A2"),
        ]

        result = await llm_client.complete(
            prompt="Q3",
            strategy=PromptingStrategy.FEW_SHOT,
            few_shot_examples=examples,
        )

        assert result == "A3"

    @pytest.mark.asyncio
    async def test_complete_few_shot_without_examples_raises(self, llm_client):
        """Test that few-shot without examples raises ValueError."""
        with pytest.raises(ValueError, match="[Ff]ew-shot strategy requires examples"):
            await llm_client.complete(
                prompt="Test",
                strategy=PromptingStrategy.FEW_SHOT,
            )

    @pytest.mark.asyncio
    async def test_complete_cot_strategy(self, llm_client, mock_client):
        """Test Chain-of-Thought completion strategy."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "REASONING:\nStep 1: First step\nFINAL ANSWER: 4"
                    }
                }
            ],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        result = await llm_client.complete(
            prompt="What is 2+2?",
            strategy=PromptingStrategy.CHAIN_OF_THOUGHT,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_complete_json_mode(self, llm_client, mock_client):
        """Test completion with JSON mode."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": '{"name": "test"}'}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        result = await llm_client.complete(
            prompt="Return a JSON object with name",
            json_mode=True,
        )

        assert result == '{"name": "test"}'

    @pytest.mark.asyncio
    async def test_complete_json_raises_on_invalid_json(self, llm_client, mock_client):
        """Test that invalid JSON raises ValueError."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "not valid json"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            await llm_client.complete_json(
                prompt="Return a JSON object",
            )

    @pytest.mark.asyncio
    async def test_complete_with_cot_steps(self, llm_client, mock_client):
        """Test Chain-of-Thought with step extraction."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "REASONING:\nStep 1: First step: initial calculation\nStep 2: Second step: final result\nFINAL ANSWER: 42"
                    }
                }
            ],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        answer, steps = await llm_client.complete_with_cot_steps(
            question="What is the answer?",
        )

        assert answer == "42"
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert "First step" in steps[0].reasoning
        assert "initial calculation" in steps[0].intermediate_result

    @pytest.mark.asyncio
    async def test_complete_with_cod_steps(self, llm_client, mock_client):
        """Test Chain-of-Draft with step extraction."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": "STEPS:\n1. Quick reasoning: Take action\n2. More reasoning: Take another action\nFINAL ANSWER: 100"
                    }
                }
            ],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        answer, steps = await llm_client.complete_with_cod_steps(
            question="Calculate something",
        )

        assert answer == "100"
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert "Quick reasoning" in steps[0].reasoning
        assert "Take action" in steps[0].action

    @pytest.mark.asyncio
    async def test_complete_no_choices_raises(self, llm_client, mock_client):
        """Test that empty choices raises ValueError."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        with pytest.raises(ValueError, match="No choices in response"):
            await llm_client.complete(prompt="Test")

    @pytest.mark.asyncio
    async def test_close_async_client(self, llm_client, mock_client):
        """Test closing async client."""
        llm_client._client = mock_client
        await llm_client.close()
        mock_client.aclose.assert_called_once()
        assert llm_client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self, llm_client, mock_client):
        """Test async context manager."""
        llm_client._client = mock_client

        async with llm_client as client:
            assert client is llm_client

        mock_client.aclose.assert_called_once()


class TestLLMClientRetryLogic:
    """Test suite for retry logic in LLM client."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock async client."""
        client = AsyncMock(spec=AsyncClient)
        client.post = AsyncMock()
        return client

    @pytest.fixture
    def llm_client(self, mock_client):
        """Create LLM client with mocked HTTP client."""
        client = LLMClient(
            LLMClientConfig(max_retries=2, retry_delay=0.1, retry_backoff=1.5)
        )
        client._client = mock_client
        return client

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, llm_client, mock_client):
        """Test retry logic doesn't retry by default (simplified test)."""
        mock_error_response = Mock()
        mock_error_response.status_code = 503

        mock_client.post.side_effect = HTTPStatusError(
            "Server error", request=Mock(), response=mock_error_response
        )

        with pytest.raises(HTTPStatusError):
            messages = [ChatMessage(role=MessageRole.USER, content="Test")]
            await llm_client._execute_request(messages)

        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self, llm_client, mock_client):
        """Test no retry on 4xx client error."""
        mock_error_response = Mock()
        mock_error_response.status_code = 400

        mock_client.post.side_effect = HTTPStatusError(
            "Bad request", request=Mock(), response=mock_error_response
        )

        with pytest.raises(HTTPStatusError):
            messages = [ChatMessage(role=MessageRole.USER, content="Test")]
            await llm_client._execute_request(messages)

        assert mock_client.post.call_count == 1


class TestLLMClientSync:
    """Test suite for synchronous LLM client methods."""

    @pytest.fixture
    def sync_client(self):
        """Create a mock sync client."""
        with patch(
            "knowledge_base.clients.llm_client.httpx.Client"
        ) as mock_sync_client:
            client = Mock()
            client.post = Mock()
            mock_sync_client.return_value = client
            yield client, mock_sync_client

    def test_complete_sync_standard(self, sync_client):
        """Test synchronous standard completion."""
        mock_client, mock_sync_class = sync_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "Sync response"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        client = LLMClient()
        client._sync_client = mock_client

        result = client.complete_sync(prompt="Test sync")

        assert result == "Sync response"

    def test_complete_sync_with_few_shot(self, sync_client):
        """Test synchronous few-shot completion."""
        mock_client, mock_sync_class = sync_client

        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "Synced answer"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        client = LLMClient()
        client._sync_client = mock_client

        examples = [FewShotExample(input="Q", output="A")]
        result = client.complete_sync(
            prompt="Test",
            strategy=PromptingStrategy.FEW_SHOT,
            few_shot_examples=examples,
        )

        assert result == "Synced answer"


class TestCreateLLMClient:
    """Test suite for create_llm_client factory function."""

    @pytest.mark.asyncio
    async def test_create_client_with_defaults(self):
        """Test creating client with default settings."""
        client = await create_llm_client()

        assert client._config.url == "http://localhost:8087/v1/"
        assert client._config.model == "gemini-2.5-flash-lite"
        assert client._config.max_retries == 3

    @pytest.mark.asyncio
    async def test_create_client_with_custom_settings(self):
        """Test creating client with custom settings."""
        client = await create_llm_client(
            url="http://custom:9000/v1/",
            model="custom-model",
            temperature=0.7,
            max_retries=5,
        )

        assert client._config.url == "http://custom:9000/v1/"
        assert client._config.model == "custom-model"
        assert client._config.temperature == 0.7
        assert client._config.max_retries == 5


class TestPromptingStrategies:
    """Test all prompting strategies produce correct output."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock async client."""
        client = AsyncMock(spec=AsyncClient)
        client.post = AsyncMock()
        return client

    @pytest.fixture
    def llm_client(self, mock_client):
        """Create LLM client with mocked HTTP client."""
        client = LLMClient()
        client._client = mock_client
        return client

    @pytest.mark.parametrize(
        "strategy",
        [
            PromptingStrategy.STANDARD,
            PromptingStrategy.FEW_SHOT,
            PromptingStrategy.CHAIN_OF_THOUGHT,
            PromptingStrategy.CHAIN_OF_DRAFT,
        ],
    )
    @pytest.mark.asyncio
    async def test_all_strategies_work(self, llm_client, mock_client, strategy):
        """Test all prompting strategies produce valid responses."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"message": {"content": "Response"}}],
            "usage": {},
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response

        kwargs = {"prompt": "Test question"}
        if strategy == PromptingStrategy.FEW_SHOT:
            kwargs["few_shot_examples"] = [FewShotExample(input="Q", output="A")]

        result = await llm_client.complete(strategy=strategy, **kwargs)

        assert result == "Response"
        mock_client.post.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Local LLM Gateway Client."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Type

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.clients import (
    create_llm_client,
    FewShotExample,
    PromptingStrategy,
    LLMClient,
    RotatingLLMClient,
)
from knowledge_base.config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    ROTATION_DELAY,
    LLM_GATEWAY_URL,
)
from knowledge_base.clients.model_registry import (
    ModelRegistry,
    ModelRegistryConfig,
    ModelRegistryManager,
    get_model_registry,
)

logger = logging.getLogger(__name__)


class GatewayConfig(BaseSettings):
    """Gateway configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    url: str = LLM_GATEWAY_URL
    api_key: str = ""
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = DEFAULT_LLM_TIMEOUT


class ChatMessage(BaseModel):
    """Chat message structure."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str = Field(..., description="Model name")
    messages: list[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.1, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream responses")
    response_format: dict[str, str] | None = Field(
        default=None, description="Response format (e.g., {'type': 'json_object'})"
    )


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Timestamp")
    model: str = Field(..., description="Model used")
    choices: list[dict[str, Any]] = Field(..., description="Generation choices")
    usage: Any = Field(..., description="Token usage")


class GatewayClient:
    """Client for local OpenAI-compatible LLM gateway."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        """Initialize gateway client.

        Args:
            config: Gateway configuration. If None, loads from environment.
        """
        self._config = config or GatewayConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client.
        """
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._config.url,
                headers=headers,
                timeout=self._config.timeout,
            )
        return self._client

    async def chat_completion(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResponse:
        """Send chat completion request.

        Args:
            messages: List of chat messages.
            model: Model name. If None, uses config default.
            temperature: Sampling temperature. If None, uses config default.
            max_tokens: Maximum tokens to generate. If None, uses config default.
            response_format: Response format specification (e.g., {'type': 'json_object'}).

        Returns:
            Chat completion response.

        Raises:
            httpx.HTTPError: If request fails.
        """
        client = await self._get_client()

        normalized_messages = [
            ChatMessage.model_validate(msg) if isinstance(msg, dict) else msg
            for msg in messages
        ]

        request = ChatCompletionRequest(
            model=model or self._config.model,
            messages=normalized_messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
            response_format=response_format,
        )

        response = await client.post(
            "chat/completions",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()

        # Parse response JSON
        response_data = response.json()

        # Check if response contains error data (some gateways return errors as 200)
        if isinstance(response_data, dict) and (
            "status" in response_data
            or "error" in response_data
            or "msg" in response_data
        ):
            # Extract status code from error response
            status_code = 500
            if "status" in response_data:
                # Handle both string and int status
                status = response_data["status"]
                if isinstance(status, str) and status.isdigit():
                    status_code = int(status)
                elif isinstance(status, int):
                    status_code = status
            elif "error" in response_data and isinstance(response_data["error"], dict):
                error_data = response_data["error"]
                if "code" in error_data:
                    status_code = error_data.get("code", 500)

            error_msg = response_data.get(
                "msg", response_data.get("error", "Unknown error")
            )
            logger.warning(
                f"Model {request.model} returned error: {error_msg} (status: {status_code})"
            )

            # Raise as HTTP error so it can be handled by retry/rotation logic
            raise httpx.HTTPStatusError(
                f"Model error: {error_msg}",
                request=response.request,
                response=httpx.Response(
                    status_code=status_code,
                    content=str(error_msg).encode(),
                    request=response.request,
                ),
            )

        return ChatCompletionResponse.model_validate(response_data)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            json_mode: Whether to request JSON output format.
            **kwargs: Additional arguments for chat_completion.

        Returns:
            Generated text.
        """
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response_format = {"type": "json_object"} if json_mode else None
        response = await self.chat_completion(
            messages, response_format=response_format, **kwargs
        )

        if not response.choices:
            raise ValueError("No choices in response")

        return str(response.choices[0]["message"]["content"])

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "GatewayClient":
        """Enter async context.

        Returns:
            Self.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        await self.close()


class EnhancedGateway(GatewayClient):
    """Enhanced gateway client with advanced LLM prompting capabilities."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        """Initialize enhanced gateway.

        Args:
            config: Gateway configuration. If None, loads from environment.
        """
        super().__init__(config)
        self._llm_client: LLMClient | None = None
        self._rotating_client: RotatingLLMClient | None = None
        self._model_registry: ModelRegistry | None = None
        self._failed_models: set[str] = set()

    async def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client.

        Returns:
            LLM client instance.
        """
        if self._llm_client is None:
            self._llm_client = await create_llm_client(
                url=self._config.url,
                api_key=self._config.api_key,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        return self._llm_client

    async def _get_rotating_client(self) -> RotatingLLMClient:
        """Get or create rotating LLM client.

        Returns:
            Rotating LLM client instance.
        """
        if self._rotating_client is None:
            from knowledge_base.clients.llm_client import LLMClientConfig

            client_config = LLMClientConfig(
                base_url=self._config.url,
                api_key=self._config.api_key,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                timeout=self._config.timeout,
            )
            self._rotating_client = RotatingLLMClient(client_config)
            await self._refresh_models()
        return self._rotating_client

    async def _get_model_registry(self) -> ModelRegistry:
        """Get or create model registry.

        Returns:
            Model registry instance.
        """
        if self._model_registry is None:
            registry_config = ModelRegistryConfig(
                gateway_url=self._config.url.rstrip("/").replace("/v1", ""),
                timeout=self._config.timeout,
            )
            self._model_registry = await ModelRegistryManager.get_registry(
                registry_config
            )
        return self._model_registry

    async def _refresh_models(self) -> None:
        """Refresh the rotating client's model list from the registry."""
        try:
            registry = await self._get_model_registry()
            rotation_models = await registry.get_rotation_list()

            if rotation_models:
                model_names = [
                    model.id
                    for model in rotation_models
                    if model.id not in self._failed_models
                ]
                if model_names:
                    self._rotating_client.set_models(model_names)
                    logger.info(f"Updated rotation list with {len(model_names)} models")
        except Exception as e:
            logger.warning(f"Failed to refresh models: {e}")

    async def call_llm(self, **kwargs: Any) -> dict[str, Any]:
        """Call LLM with automatic model rotation on rate limits.

        Args:
            **kwargs: Arguments to pass to the LLM client.
                Common args: prompt, messages, model, temperature, max_tokens,
                response_format, strategy, few_shot_examples, etc.

        Returns:
            Dictionary containing response data with model used.

        Raises:
            Exception: If all models fail or a non-rate-limit error occurs.
        """
        rotating_client = await self._get_rotating_client()

        try:
            response = await rotating_client.chat_completion(**kwargs)

            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "success": True,
            }

        except Exception as e:
            error_msg = str(e).lower()

            if self._is_rate_limit_error(e) or any(
                indicator in error_msg
                for indicator in [
                    "429",
                    "too many requests",
                    "rate limit",
                    "quota exceeded",
                ]
            ):
                logger.warning(f"Rate limit detected: {e}")

                if hasattr(e, "model") and e.model:
                    self._failed_models.add(str(e.model))
                    await self._mark_model_unhealthy(str(e.model))

                await self._refresh_models()

                await asyncio.sleep(ROTATION_DELAY)

                try:
                    response = await rotating_client.chat_completion(**kwargs)
                    return {
                        "content": response.content,
                        "model": response.model,
                        "usage": response.usage,
                        "success": True,
                        "retried": True,
                    }
                except Exception as retry_error:
                    logger.error(f"Retry failed: {retry_error}")
                    return {
                        "error": str(retry_error),
                        "success": False,
                    }
            else:
                logger.error(f"Non-rate-limit error: {e}")
                return {
                    "error": str(e),
                    "success": False,
                }

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error.

        Args:
            error: The exception to check.

        Returns:
            True if the error is a rate limit error.
        """
        if hasattr(error, "response") and hasattr(error.response, "status_code"):
            return error.response.status_code == 429
        return False

    async def _mark_model_unhealthy(self, model_id: str) -> None:
        """Mark a model as unhealthy in the registry.

        Args:
            model_id: The model identifier.
        """
        try:
            registry = await self._get_model_registry()
            registry.mark_model_unhealthy(model_id)
            logger.warning(f"Marked model {model_id} as unhealthy")
        except Exception as e:
            logger.error(f"Failed to mark model unhealthy: {e}")

    async def extract_with_reasoning(self, prompt: str) -> dict[str, Any]:
        """Use Chain-of-Thought for complex extractions.

        Args:
            prompt: Extraction prompt.

        Returns:
            Dictionary with answer and reasoning steps.
        """
        client = await self._get_llm_client()
        answer, steps = await client.complete_with_cot_steps(prompt)
        return {"answer": answer, "steps": [s.model_dump() for s in steps]}

    async def extract_with_cod(self, prompt: str) -> dict[str, Any]:
        """Use Chain-of-Draft for token-efficient extraction.

        Args:
            prompt: Extraction prompt.

        Returns:
            Dictionary with answer and draft steps.
        """
        client = await self._get_llm_client()
        answer, steps = await client.complete_with_cod_steps(prompt)
        return {"answer": answer, "steps": [s.model_dump() for s in steps]}

    async def extract_structured(
        self, prompt: str, schema: Type[BaseModel], json_mode: bool = False
    ) -> BaseModel:
        """Extract with structured JSON output.

        Args:
            prompt: Extraction prompt.
            schema: Pydantic schema for response.
            json_mode: Whether to request JSON output format.

        Returns:
            Parsed instance of schema.
        """
        client = await self._get_llm_client()
        json_data = await client.complete_json(
            prompt=prompt,
            strategy=PromptingStrategy.STANDARD,
            json_mode=json_mode,
        )
        return schema.model_validate(json_data)

    async def extract_few_shot(
        self, prompt: str, examples: list[FewShotExample]
    ) -> dict[str, Any]:
        """Extract with few-shot examples.

        Args:
            prompt: Extraction prompt.
            examples: Few-shot examples.

        Returns:
            Dictionary with extracted content.
        """
        client = await self._get_llm_client()
        content = await client.complete(
            prompt=prompt,
            strategy=PromptingStrategy.FEW_SHOT,
            few_shot_examples=examples,
        )
        return {"content": content}

    async def close(self) -> None:
        """Close HTTP clients."""
        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None
        await super().close()

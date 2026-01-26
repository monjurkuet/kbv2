"""Local LLM Gateway Client."""

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
)


class GatewayConfig(BaseSettings):
    """Gateway configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    url: str = "http://localhost:8087/v1/"
    api_key: str = ""
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 120.0


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

        return ChatCompletionResponse.model_validate(response.json())

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
        self, prompt: str, schema: Type[BaseModel]
    ) -> BaseModel:
        """Extract with structured JSON output.

        Args:
            prompt: Extraction prompt.
            schema: Pydantic schema for response.

        Returns:
            Parsed instance of schema.
        """
        client = await self._get_llm_client()
        json_data = await client.complete_json(
            prompt=prompt,
            strategy=PromptingStrategy.STANDARD,
            json_mode=True,
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

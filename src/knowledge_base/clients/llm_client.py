"""LLM Client with advanced prompting strategies for KBV2.

This module provides an OpenAI-compatible client for the local LLM gateway
with support for few-shot prompting, Chain-of-Thought (CoT), Chain-of-Draft (CoD),
and structured JSON output.
"""

import json
from typing import Any, Callable
from enum import Enum

import warnings

warnings.warn(
    "llm_client is deprecated since v0.6.0. "
    "Use ResilientGatewayClient from knowledge_base.common.resilient_gateway.gateway instead, "
    "or UnifiedLLMClient from knowledge_base.clients.unified_llm_client.",
    DeprecationWarning,
    stacklevel=2,
)

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.config.constants import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    LLM_GATEWAY_URL,
)


class PromptingStrategy(str, Enum):
    """Available prompting strategies."""

    STANDARD = "standard"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CHAIN_OF_DRAFT = "chain_of_draft"


class LLMClientConfig(BaseSettings):
    """LLM client configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    url: str = LLM_GATEWAY_URL
    api_key: str = ""
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = DEFAULT_LLM_TIMEOUT
    max_retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY
    retry_backoff: float = 2.0


class MessageRole(str, Enum):
    """Message role types."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Chat message structure."""

    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(None, description="Optional message name")


class FewShotExample(BaseModel):
    """Example for few-shot prompting."""

    input: str = Field(..., description="Example input")
    output: str = Field(..., description="Expected output")


class CoTStep(BaseModel):
    """Step in Chain-of-Thought reasoning."""

    step_number: int = Field(..., description="Step number", ge=1)
    reasoning: str = Field(..., description="Reasoning content")
    intermediate_result: str | None = Field(
        None, description="Intermediate result if any"
    )


class CoDStep(BaseModel):
    """Step in Chain-of-Draft reasoning (token-efficient)."""

    step_number: int = Field(..., description="Step number", ge=1)
    reasoning: str = Field(..., description="Brief reasoning")
    action: str = Field(..., description="Action taken")


class LLMRequest(BaseModel):
    """LLM completion request."""

    model: str = Field(..., description="Model name")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, description="Max tokens to generate")
    stream: bool = Field(default=False, description="Enable streaming")
    response_format: dict[str, str] | None = Field(
        default=None, description="Response format specification"
    )


class LLMResponse(BaseModel):
    """LLM completion response."""

    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[dict[str, Any]] = Field(..., description="Response choices")
    usage: dict[str, int] | None = Field(None, description="Token usage")


class LLMClient:
    """OpenAI-compatible LLM client with advanced prompting capabilities."""

    def __init__(self, config: LLMClientConfig | None = None) -> None:
        """Initialize LLM client.

        Args:
            config: Client configuration. Loads from environment if None.
        """
        self._config = config or LLMClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for requests.

        Returns:
            Dictionary of HTTP headers.
        """
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client.

        Returns:
            Async HTTP client.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.url,
                headers=self._get_headers(),
                timeout=self._config.timeout,
            )
        return self._client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client.

        Returns:
            Sync HTTP client.
        """
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self._config.url,
                headers=self._get_headers(),
                timeout=self._config.timeout,
            )
        return self._sync_client

    async def _request_with_retry(
        self,
        request_func: Callable,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Execute request with retry logic.

        Args:
            request_func: Async request function to execute.
            *args: Positional arguments for request function.
            **kwargs: Keyword arguments for request function.

        Returns:
            LLM response.

        Raises:
            httpx.HTTPError: If all retries are exhausted.
        """
        last_exception: Exception | None = None
        delay = self._config.retry_delay

        for attempt in range(self._config.max_retries + 1):
            try:
                return await request_func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    last_exception = e
                else:
                    raise
            except httpx.RequestError as e:
                last_exception = e

            if attempt < self._config.max_retries:
                await httpx.AsyncClient().aclose()
                await self._sleep(delay)
                delay *= self._config.retry_backoff

        raise last_exception or Exception("Request failed after all retries")

    async def _sleep(self, seconds: float) -> None:
        """Async sleep.

        Args:
            seconds: Seconds to sleep.
        """
        import asyncio

        await asyncio.sleep(seconds)

    def _build_few_shot_messages(
        self,
        system_prompt: str,
        examples: list[FewShotExample],
        user_prompt: str,
    ) -> list[ChatMessage]:
        """Build messages for few-shot prompting.

        Args:
            system_prompt: System instruction.
            examples: Few-shot examples.
            user_prompt: Current user query.

        Returns:
            List of chat messages.
        """
        messages: list[ChatMessage] = []

        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))

        for example in examples:
            messages.append(ChatMessage(role=MessageRole.USER, content=example.input))
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=example.output)
            )

        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
        return messages

    def _build_cot_messages(
        self,
        system_prompt: str,
        question: str,
        enable_cod: bool = False,
    ) -> list[ChatMessage]:
        """Build messages for Chain-of-Thought or Chain-of-Draft prompting.

        Args:
            system_prompt: System instruction.
            question: User question.
            enable_cod: Use Chain-of-Draft (token-efficient) instead of CoT.

        Returns:
            List of chat messages.
        """
        if enable_cod:
            cot_prompt = (
                f"{system_prompt}\n\n"
                "Think through the problem step by step, but keep each step brief and action-oriented.\n"
                "For each step, provide:\n"
                "1. Brief reasoning (1-2 sentences)\n"
                "2. The action/calculation taken\n"
                "Then provide your final answer.\n\n"
                "Structure your response as:\n"
                "STEPS:\n"
                "1. [reasoning]: [action]\n"
                "2. [reasoning]: [action]\n"
                "...\n"
                "FINAL ANSWER: [answer]"
            )
        else:
            cot_prompt = (
                f"{system_prompt}\n\n"
                "Think through the problem step by step.\n"
                "For each step:\n"
                "1. Explain your reasoning in detail\n"
                "2. Note any intermediate conclusions\n"
                "Then provide your final answer.\n\n"
                "Structure your response as:\n"
                "REASONING:\n"
                "Step 1: [detailed reasoning]\n"
                "Step 2: [detailed reasoning]\n"
                "...\n"
                "FINAL ANSWER: [answer]"
            )

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=cot_prompt),
            ChatMessage(role=MessageRole.USER, content=question),
        ]

    async def _execute_request(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> LLMResponse:
        """Execute chat completion request.

        Args:
            messages: List of chat messages.
            model: Model name override.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            response_format: Response format specification.

        Returns:
            LLM response.
        """
        client = await self._get_async_client()

        request_data = {
            "model": model or self._config.model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature or self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": False,
        }

        if response_format:
            request_data["response_format"] = response_format

        request_data = {k: v for k, v in request_data.items() if v is not None}

        response = await client.post("chat/completions", json=request_data)
        response.raise_for_status()

        return LLMResponse(**response.json())

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        strategy: PromptingStrategy = PromptingStrategy.STANDARD,
        few_shot_examples: list[FewShotExample] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        cod_steps: list[CoDStep] | None = None,
    ) -> str:
        """Generate completion using specified strategy.

        Args:
            prompt: User prompt.
            system_prompt: Optional system instruction.
            strategy: Prompting strategy to use.
            few_shot_examples: Examples for few-shot prompting.
            temperature: Temperature override.
            max_tokens: Max tokens override.
            json_mode: Request JSON output.
            cod_steps: Previous CoD steps for continuity.

        Returns:
            Generated text content.
        """
        messages: list[ChatMessage] = []

        if strategy == PromptingStrategy.FEW_SHOT:
            if not few_shot_examples:
                raise ValueError("Few-shot strategy requires examples")
            messages = self._build_few_shot_messages(
                system_prompt or "You are a helpful assistant.",
                few_shot_examples,
                prompt,
            )
        elif strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            messages = self._build_cot_messages(
                system_prompt
                or "You are a helpful assistant that reasons step by step.",
                prompt,
                enable_cod=False,
            )
        elif strategy == PromptingStrategy.CHAIN_OF_DRAFT:
            messages = self._build_cot_messages(
                system_prompt
                or "You are a helpful assistant that provides concise reasoning.",
                prompt,
                enable_cod=True,
            )
        else:
            if system_prompt:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
                )
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response_format = {"type": "json_object"} if json_mode else None

        response = await self._execute_request(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        if not response.choices:
            raise ValueError("No choices in response")

        return str(response.choices[0]["message"]["content"])

    async def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        strategy: PromptingStrategy = PromptingStrategy.STANDARD,
        few_shot_examples: list[FewShotExample] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Generate structured JSON output.

        Args:
            prompt: User prompt (should describe desired JSON structure).
            system_prompt: Optional system instruction.
            strategy: Prompting strategy to use.
            few_shot_examples: Examples for few-shot prompting.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            Parsed JSON response.
        """
        json_prompt = f"{prompt}\n\nRespond with valid JSON only."

        content = await self.complete(
            prompt=json_prompt,
            system_prompt=system_prompt
            or "You are a JSON generator. Always output valid JSON.",
            strategy=strategy,
            few_shot_examples=few_shot_examples,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response: {e}\nResponse: {content}"
            ) from e

    async def complete_with_cot_steps(
        self,
        question: str,
        system_prompt: str | None = None,
    ) -> tuple[str, list[CoTStep]]:
        """Complete with Chain-of-Thought reasoning, extracting steps.

        Args:
            question: User question.
            system_prompt: Optional system instruction.

        Returns:
            Tuple of (final answer, list of reasoning steps).
        """
        content = await self.complete(
            prompt=question,
            system_prompt=system_prompt,
            strategy=PromptingStrategy.CHAIN_OF_THOUGHT,
        )

        steps: list[CoTStep] = []
        step_number = 1

        if "REASONING:" in content and "FINAL ANSWER:" in content:
            reasoning_part = (
                content.split("FINAL ANSWER:")[0].replace("REASONING:", "").strip()
            )
            answer = content.split("FINAL ANSWER:")[1].strip()

            for line in reasoning_part.split("\n"):
                line = line.strip()
                if line.lower().startswith(
                    f"step {step_number}:"
                ) or line.lower().startswith(f"{step_number}."):
                    step_content = (
                        line.split(":", 1)[-1].strip() if ":" in line else line
                    )
                    if ": " in step_content:
                        parts = step_content.split(": ", 1)
                        steps.append(
                            CoTStep(
                                step_number=step_number,
                                reasoning=parts[0],
                                intermediate_result=parts[1]
                                if len(parts) > 1
                                else None,
                            )
                        )
                    else:
                        steps.append(
                            CoTStep(step_number=step_number, reasoning=step_content)
                        )
                    step_number += 1
                elif line.lower().startswith("step ") and ":" in line:
                    match = line.split(":", 1)
                    if len(match) == 2:
                        step_label = match[0].strip()
                        step_content = match[1].strip()
                        if " " in step_label:
                            try:
                                num = int(step_label.split(" ")[1])
                                steps.append(
                                    CoTStep(
                                        step_number=num,
                                        reasoning=step_content.split(":")[0].strip()
                                        if ":" in step_content
                                        else step_content,
                                        intermediate_result=step_content.split(":", 1)[
                                            1
                                        ].strip()
                                        if ":" in step_content
                                        else None,
                                    )
                                )
                            except (ValueError, IndexError):
                                pass
        else:
            answer = content

        return answer, steps

    async def complete_with_cod_steps(
        self,
        question: str,
        system_prompt: str | None = None,
    ) -> tuple[str, list[CoDStep]]:
        """Complete with Chain-of-Draft reasoning, extracting steps.

        Args:
            question: User question.
            system_prompt: Optional system instruction.

        Returns:
            Tuple of (final answer, list of draft steps).
        """
        content = await self.complete(
            prompt=question,
            system_prompt=system_prompt,
            strategy=PromptingStrategy.CHAIN_OF_DRAFT,
        )

        steps: list[CoDStep] = []
        step_number = 1

        if "STEPS:" in content and "FINAL ANSWER:" in content:
            steps_part = content.split("FINAL ANSWER:")[0].replace("STEPS:", "").strip()
            answer = content.split("FINAL ANSWER:")[1].strip()

            for line in steps_part.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    line = line.lstrip("0123456789.- ").strip()
                    if ": " in line:
                        parts = line.split(": ", 1)
                        steps.append(
                            CoDStep(
                                step_number=step_number,
                                reasoning=parts[0],
                                action=parts[1] if len(parts) > 1 else "",
                            )
                        )
                        step_number += 1
        else:
            answer = content

        return answer, steps

    def complete_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        strategy: PromptingStrategy = PromptingStrategy.STANDARD,
        few_shot_examples: list[FewShotExample] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Synchronous version of complete.

        Args:
            Same as async complete().

        Returns:
            Generated text content.
        """
        import asyncio

        messages: list[ChatMessage] = []

        if strategy == PromptingStrategy.FEW_SHOT:
            if not few_shot_examples:
                raise ValueError("Few-shot strategy requires examples")
            messages = self._build_few_shot_messages(
                system_prompt or "You are a helpful assistant.",
                few_shot_examples,
                prompt,
            )
        elif strategy == PromptingStrategy.CHAIN_OF_THOUGHT:
            messages = self._build_cot_messages(
                system_prompt
                or "You are a helpful assistant that reasons step by step.",
                prompt,
                enable_cod=False,
            )
        elif strategy == PromptingStrategy.CHAIN_OF_DRAFT:
            messages = self._build_cot_messages(
                system_prompt
                or "You are a helpful assistant that provides concise reasoning.",
                prompt,
                enable_cod=True,
            )
        else:
            if system_prompt:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
                )
            messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response_format = {"type": "json_object"} if json_mode else None

        client = self._get_sync_client()

        request_data = {
            "model": self._config.model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature or self._config.temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": False,
        }

        if response_format:
            request_data["response_format"] = response_format

        request_data = {k: v for k, v in request_data.items() if v is not None}

        response = client.post("chat/completions", json=request_data)
        response.raise_for_status()

        llm_response = LLMResponse(**response.json())

        if not llm_response.choices:
            raise ValueError("No choices in response")

        return str(llm_response.choices[0]["message"]["content"])

    async def close(self) -> None:
        """Close HTTP clients."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def __aenter__(self) -> "LLMClient":
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
        """Exit async context."""
        await self.close()


async def create_llm_client(
    url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> LLMClient:
    """Factory function to create LLM client with custom settings.

    Args:
        url: Gateway URL override.
        api_key: API key override.
        model: Model name override.
        temperature: Default temperature.
        max_tokens: Default max tokens.
        max_retries: Max retry attempts.

    Returns:
        Configured LLM client.
    """
    config = LLMClientConfig(
        url=url or LLM_GATEWAY_URL,
        api_key=api_key or "",
        model=model or DEFAULT_LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    return LLMClient(config)

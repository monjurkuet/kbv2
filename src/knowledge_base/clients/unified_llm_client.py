"""Unified LLM client with middleware support."""

import asyncio
import logging
import httpx
from typing import List, Optional
from knowledge_base.config.constants import (
    DEFAULT_LLM_MODEL,
    LLM_GATEWAY_URL,
    DEFAULT_LLM_TIMEOUT,
    MAX_RETRIES,
)
from knowledge_base.clients.models import ChatMessage, LLMResponse, ModelConfig
from knowledge_base.clients.middleware import (
    RetryMiddleware,
    RotationMiddleware,
    CircuitBreakerMiddleware,
)

logger = logging.getLogger(__name__)


class UnifiedLLMClient:
    """Single interface for all LLM operations with middleware support."""

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_url: str = LLM_GATEWAY_URL,
        timeout: float = DEFAULT_LLM_TIMEOUT,
        enable_retry: bool = True,
        enable_rotation: bool = False,
        enable_circuit_breaker: bool = True,
    ):
        self.model = model
        self.api_url = api_url
        self.timeout = timeout

        # Initialize middleware
        self._retry_middleware = (
            RetryMiddleware(max_retries=MAX_RETRIES) if enable_retry else None
        )
        self._rotation_middleware = (
            RotationMiddleware(api_url=api_url) if enable_rotation else None
        )
        self._circuit_breaker = (
            CircuitBreakerMiddleware() if enable_circuit_breaker else None
        )

    async def chat_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """Execute chat completion with middleware chain."""

        async def _do_chat_completion() -> LLMResponse:
            payload = {
                "model": self.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    finish_reason=data["choices"][0].get("finish_reason"),
                    usage=data.get("usage"),
                )

        # Apply middleware chain
        result = _do_chat_completion

        if self._circuit_breaker:
            result = lambda: self._circuit_breaker.execute(_do_chat_completion)

        if self._rotation_middleware:
            result = lambda: self._rotation_middleware.execute(result)

        if self._retry_middleware:
            result = lambda: self._retry_middleware.execute(result)

        return await result()

"""Async LLM client using OpenAI SDK with random model rotation.

Features:
- AsyncOpenAI client (built-in retry logic)
- Random model selection on EVERY call (no default model)
- Model rotation on ANY error type
- Continuous retry until success or max attempts
- Automatic model discovery from API

Environment Variables:
- LLM_API_BASE: API base URL (default: http://localhost:8087/v1)
- LLM_API_KEY: API key (default: sk-dummy)
- LLM_MAX_ROTATION_ATTEMPTS: Max models to try (default: 10)
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_MAX_ROTATION_ATTEMPTS = 10


class AsyncLLMClient:
    """Async LLM client with random model rotation on every call.

    This client automatically:
    1. Fetches available models from the API
    2. Randomly selects a model on EVERY call (no default)
    3. Rotates to a different model on ANY error
    4. Retries until success or max attempts exhausted
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_rotation_attempts: Optional[int] = None,
    ) -> None:
        """Initialize the async LLM client.

        Args:
            base_url: API base URL. Defaults to LLM_API_BASE env var.
            api_key: API key. Defaults to LLM_API_KEY env var.
            max_rotation_attempts: Max models to try before giving up.
        """
        self._client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_API_BASE", "http://localhost:8087/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "sk-dummy"),
        )
        self._max_rotation_attempts = max_rotation_attempts or int(
            os.getenv("LLM_MAX_ROTATION_ATTEMPTS", str(DEFAULT_MAX_ROTATION_ATTEMPTS))
        )
        self._available_models: List[str] = []
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Fetch available models from API if not already done."""
        if not self._initialized:
            await self._fetch_models()
            self._initialized = True

    async def _fetch_models(self) -> None:
        """Fetch available models from the API."""
        try:
            response = await self._client.models.list()
            all_models = [model.id for model in response.data]
            # Filter out gemini-3 models (cooldown/rate limit issues)
            self._available_models = [
                m
                for m in all_models
                if "gemini-3" not in m.lower() and "gemini-4" not in m.lower()
            ]
            filtered_count = len(all_models) - len(self._available_models)
            if filtered_count > 0:
                logger.info(
                    f"Filtered out {filtered_count} gemini-3/4 models due to rate limits"
                )
            logger.info(f"Discovered {len(self._available_models)} usable models")
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            self._available_models = []

    def _get_random_model(self) -> str:
        """Get a random model from available models.

        Returns:
            Random model name from the available list.
        """
        if not self._available_models:
            raise RuntimeError(
                "No models available. Call _ensure_initialized() first or "
                "handle models not being fetched."
            )
        return random.choice(self._available_models)

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a completion with random model rotation on errors.

        Every call randomly selects a model. If an error occurs, retries
        with a different random model until success or max attempts.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            json_mode: Whether to request JSON output format.
            **kwargs: Additional arguments for OpenAI API.

        Returns:
            Dict with:
            - content: The response text
            - reasoning: Thinking content (if reasoning model)
            - model: Model used
            - usage: Token usage
        """
        await self._ensure_initialized()

        messages: List[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None

        for attempt in range(self._max_rotation_attempts):
            model = self._get_random_model()  # Random selection EVERY call

            try:
                create_params: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }

                if json_mode:
                    create_params["response_format"] = {"type": "json_object"}

                # OpenAI's built-in retry handles transient errors per model
                response = await self._client.chat.completions.create(**create_params)

                msg = response.choices[0].message

                return {
                    "content": msg.content or "",
                    "reasoning": getattr(msg, "reasoning_content", None),
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                }

            except Exception as e:
                logger.warning(f"Model {model} failed (attempt {attempt + 1}): {e}")
                last_error = e
                # Continue loop to try different random model

        # All attempts exhausted
        if last_error:
            raise last_error
        raise RuntimeError(f"All {self._max_rotation_attempts} models failed")

    async def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a JSON completion.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            **kwargs: Additional arguments for OpenAI API.

        Returns:
            Dict with content, reasoning, model, usage, and parsed json.
        """
        result = await self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True,
            **kwargs,
        )

        # Parse JSON from content
        import json

        try:
            result["json"] = json.loads(result["content"])
        except json.JSONDecodeError:
            result["json"] = None

        return result

    async def list_models(self) -> List[str]:
        """List available models.

        Returns:
            List of available model IDs.
        """
        await self._ensure_initialized()
        return self._available_models.copy()

    async def close(self) -> None:
        """Close the async client."""
        await self._client.close()

    async def __aenter__(self) -> "AsyncLLMClient":
        """Enter async context.

        Returns:
            Self.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()


# Singleton instance
_llm_client: AsyncLLMClient | None = None


def get_llm_client() -> AsyncLLMClient:
    """Get or create singleton async LLM client.

    Returns:
        AsyncLLMClient instance.
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncLLMClient()
    return _llm_client


async def test_llm_client() -> None:
    """Test the LLM client functionality."""
    try:
        client = AsyncLLMClient()

        # Test model listing
        models = await client.list_models()
        print(f"✅ Available models: {len(models)}")

        # Test completion with random model
        result = await client.complete(
            prompt="Say 'Hello from random model'",
            temperature=0.5,
        )
        print(f"✅ Response from model {result['model']}: {result['content'][:50]}...")

        await client.close()
        print("✅ LLM client test passed!")

    except Exception as e:
        print(f"❌ LLM client test failed: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_llm_client())

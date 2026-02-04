"""Rotating LLM Client with model fallback for rate limit handling.

This module provides automatic model rotation when encountering rate limits (429 errors).
It cycles through available models with 5-second delays between retries.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import logging

import warnings

warnings.warn(
    "rotating_llm_client is deprecated since v0.6.0. "
    "Use ResilientGatewayClient from knowledge_base.common.resilient_gateway.gateway instead.",
    DeprecationWarning,
    stacklevel=2,
)

import httpx
from pydantic import BaseModel, Field

from knowledge_base.clients.llm_client import (
    LLMClient,
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMClientConfig,
)


logger = logging.getLogger(__name__)


class ModelRotationConfig(BaseModel):
    """Configuration for model rotation."""

    models: List[str] = Field(
        default_factory=lambda: [
            "kimi-k2-thinking",  # Primary: Kimi (excellent Chinese/English)
            "qwen3-max",  # Secondary: Qwen (strong reasoning)
            "glm-4.7",  # Tertiary: GLM (good overall)
            "deepseek-v3.2-reasoner",  # Quaternary: DeepSeek (reasoning)
            "gemini-2.5-flash-lite",  # Fallback: Gemini (reliable)
        ],
        description="List of models to rotate through on rate limits",
    )
    retry_delay: float = Field(
        default=5.0,
        description="Delay in seconds between retries (minimum 5 seconds as requested)",
    )
    max_retries: int = Field(
        default=10, description="Maximum number of retries across all models"
    )
    rate_limit_status_codes: List[int] = Field(
        default_factory=lambda: [429, 503, 529],
        description="HTTP status codes that trigger model rotation",
    )
    rate_limit_messages: List[str] = Field(
        default_factory=lambda: [
            "too many requests",
            "rate limit",
            "quota exceeded",
            "try again later",
        ],
        description="Error message substrings that indicate rate limiting",
    )


class RotatingLLMClient(LLMClient):
    """LLM Client that automatically rotates models on rate limits.

    When a rate limit error (429) is encountered, this client will:
    1. Wait 5 seconds (or configured delay)
    2. Switch to the next model in the rotation
    3. Retry the request
    4. Continue until successful or max retries exhausted
    """

    def __init__(self, config: Optional[LLMClientConfig] = None) -> None:
        """Initialize rotating LLM client.

        Args:
            config: Base LLM client configuration. If None, uses defaults.
        """
        super().__init__(config)
        self.rotation_config = ModelRotationConfig()
        self._current_model_index = 0
        self._retry_count = 0

    def _get_next_model(self) -> str:
        """Get the next model in the rotation.

        Returns:
            Model name from the rotation list
        """
        model = self.rotation_config.models[self._current_model_index]
        self._current_model_index = (self._current_model_index + 1) % len(
            self.rotation_config.models
        )
        return model

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if an error is a rate limit error.

        Args:
            error: The exception to check

        Returns:
            True if this is a rate limit error, False otherwise
        """
        # Check for HTTP status codes
        if isinstance(error, httpx.HTTPStatusError):
            if (
                error.response.status_code
                in self.rotation_config.rate_limit_status_codes
            ):
                return True

        # Check error message content
        error_str = str(error).lower()
        for indicator in self.rotation_config.rate_limit_messages:
            if indicator in error_str:
                return True

        return False

    async def chat_completion(
        self,
        messages: List[ChatMessage] | List[Dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Dict[str, str] | None = None,
        strategy: str | None = None,
        examples: List[Any] | None = None,
    ) -> LLMResponse:
        """Chat completion with automatic model rotation on rate limits.

        Args:
            messages: List of chat messages
            model: Model name. If None, uses rotation. If provided, skips rotation.
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Response format specification
            strategy: Prompting strategy
            examples: Few-shot examples

        Returns:
            LLM response

        Raises:
            Exception: If all retries exhausted or non-rate-limit error occurs
        """
        # Use provided model if specified, otherwise use rotation
        if model:
            # Direct model bypasses rotation (useful for specific model calls)
            return await super().chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                strategy=strategy,
                examples=examples,
            )

        # Start with first model in rotation
        current_model = self._get_next_model()

        for attempt in range(self.rotation_config.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Using model '{current_model}'")

                # Call parent class method
                response = await super().chat_completion(
                    messages=messages,
                    model=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    strategy=strategy,
                    examples=examples,
                )

                logger.info(f"✅ Successfully completed with model '{current_model}'")
                self._retry_count = 0  # Reset on success
                return response

            except Exception as e:
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    self._retry_count += 1
                    old_model = current_model
                    current_model = self._get_next_model()

                    # Wait 5 seconds (or configured delay) before retry
                    wait_time = max(self.rotation_config.retry_delay, 5.0)
                    logger.warning(
                        f"Rate limit hit on model '{old_model}': {str(e)[:100]}..."
                    )
                    logger.warning(
                        f"Retrying in {wait_time}s with model '{current_model}'... "
                        f"(attempt {attempt + 1}/{self.rotation_config.max_retries})"
                    )

                    await asyncio.sleep(wait_time)
                    continue

                # Non-rate-limit error - raise immediately
                logger.error(
                    f"❌ Non-recoverable error with model '{current_model}': {e}"
                )
                raise

        # All retries exhausted
        raise Exception(
            f"All {self.rotation_config.max_retries} retries exhausted. "
            f"Tried {min(self.rotation_config.max_retries, len(self.rotation_config.models))} models. "
            f"Last error: Rate limit on '{current_model}'"
        )

    def set_models(self, models: List[str]) -> None:
        """Set the list of models to rotate through.

        Args:
            models: List of model names in rotation order
        """
        self.rotation_config.models = models
        self._current_model_index = 0
        logger.info(f"Updated model rotation: {', '.join(models)}")

    def configure_from_registry(self) -> None:
        """Configure models from the ModelRegistry.

        Fetches available models from the LLM Gateway and sets up
        a recommended rotation based on provider priority.
        """
        try:
            # Import here to avoid circular dependency
            from knowledge_base.clients.model_registry import ModelRegistryManager

            registry = ModelRegistryManager.get_registry()
            if registry:
                # Get recommended models for each provider
                providers = ["kimi", "qwen", "glm", "deepseek", "gemini"]
                models = []

                for provider in providers:
                    model = registry.get_recommended_model(provider)
                    if model:
                        models.append(model)

                if models:
                    self.set_models(models)
                    logger.info(
                        f"✅ Configured {len(models)} models from registry: {', '.join(models)}"
                    )
                else:
                    logger.warning(
                        "⚠️ No models available from registry, using defaults"
                    )
            else:
                logger.warning("⚠️ Model registry not available, using default rotation")

        except Exception as e:
            logger.warning(f"⚠️ Could not configure from registry: {e}, using defaults")


def create_rotating_llm_client(
    config: Optional[LLMClientConfig] = None,
) -> RotatingLLMClient:
    """Create a rotating LLM client with default configuration.

    Args:
        config: Optional LLM client configuration

    Returns:
        RotatingLLMClient instance
    """
    client = RotatingLLMClient(config)

    # Try to configure from registry (will use defaults if registry unavailable)
    client.configure_from_registry()

    return client


# Default model configurations for different use cases
RECOMMENDED_ROTATIONS = {
    "primary": [
        "kimi-k2-thinking",  # Best overall for reasoning
        "qwen3-max",  # Strong fallback
        "glm-4.7",  # Good general performance
        "deepseek-v3.2-reasoner",  # Reasoning specialized
        "gemini-2.5-flash-lite",  # Fast and reliable
    ],
    "fast": [
        "gemini-2.5-flash-lite",  # Very fast
        "kimi-k2-0905",  # Fast and capable
        "qwen3-32b",  # Good balance
        "glm-4.6",  # Reliable
    ],
    "reasoning": [
        "kimi-k2-thinking",  # Excellent reasoning
        "deepseek-v3.2-reasoner",  # Reasoning specialized
        "qwen3-235b-a22b-thinking-2507",  # Long context reasoning
        "gemini-3-pro-preview",  # Advanced reasoning
    ],
    "multilingual": [
        "kimi-k2-thinking",  # Excellent Chinese/English
        "qwen3-max",  # Strong multilingual
        "deepseek-v3",  # Good multilingual
        "glm-4.7",  # Chinese specialized
    ],
}

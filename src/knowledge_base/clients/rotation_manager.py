"""Model Rotation Manager that integrates with GatewayClient infrastructure.

This module provides automatic model rotation for rate limit handling,
wrapping the existing GatewayClient to provide a drop-in replacement
for direct gateway calls in the orchestrator.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field

from knowledge_base.common.gateway import (
    ChatCompletionResponse,
    ChatMessage,
    GatewayClient,
    GatewayConfig,
)

logger = logging.getLogger(__name__)


class ModelRotationConfig(BaseModel):
    """Configuration for model rotation manager."""

    models_endpoint: str = Field(
        default="http://localhost:8087/v1/models",
        description="Endpoint to fetch available models",
    )
    rotation_delay: float = Field(
        default=5.0,
        description="Minimum delay in seconds between retries (5 second minimum)",
    )
    rate_limit_status_codes: List[int] = Field(
        default_factory=lambda: [429, 503, 529],
        description="HTTP status codes that trigger model rotation",
    )
    max_rotation_attempts: int = Field(
        default=10,
        description="Maximum number of models to try before giving up",
    )
    fallback_models: List[str] = Field(
        default_factory=lambda: [
            "kimi-k2-thinking",
            "qwen3-max",
            "glm-4.7",
            "deepseek-v3.2-reasoner",
            "gemini-2.5-flash-lite",
        ],
        description="Default models to use if endpoint fetch fails",
    )


class ModelInfo(BaseModel):
    """Model information from models endpoint."""

    id: str
    created: int
    owned_by: str


class ModelRotationManager:
    """Manages model rotation by wrapping GatewayClient.

    This class provides automatic model switching when encountering rate limits,
    with a simple call_llm interface for the orchestrator.
    """

    def __init__(
        self,
        gateway_client: Optional[GatewayClient] = None,
        config: Optional[GatewayConfig] = None,
        rotation_config: Optional[ModelRotationConfig] = None,
    ) -> None:
        """Initialize model rotation manager.

        Args:
            gateway_client: Existing GatewayClient instance. If None, creates new one.
            config: Gateway configuration. If None, loads from environment.
            rotation_config: Model rotation configuration. If None, uses defaults.
        """
        self._gateway_client = gateway_client or GatewayClient(config)
        self._config = config or GatewayConfig()
        self._rotation_config = rotation_config or ModelRotationConfig()
        self._models: List[str] = []
        self._current_model_index = 0
        self._failed_models: set[str] = set()
        self._discovery_client: Optional[httpx.AsyncClient] = None

    async def _get_discovery_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for model discovery."""
        if self._discovery_client is None:
            self._discovery_client = httpx.AsyncClient(
                timeout=10.0,
            )
        return self._discovery_client

    async def _fetch_available_models(self) -> List[str]:
        """Fetch available models from the gateway endpoint.

        Returns:
            List of available model IDs.
        """
        try:
            client = await self._get_discovery_client()

            logger.info(f"Fetching models from {self._rotation_config.models_endpoint}")
            response = await client.get(self._rotation_config.models_endpoint)
            response.raise_for_status()

            data = response.json()
            models_data = data.get("data", [])

            models = [ModelInfo(**model).id for model in models_data]
            logger.info(f"Successfully fetched {len(models)} models")

            return models

        except Exception as e:
            logger.warning(f"Failed to fetch models from endpoint: {e}")
            logger.warning(
                f"Using fallback models: {self._rotation_config.fallback_models}"
            )
            return self._rotation_config.fallback_models.copy()

    async def _initialize_models(self) -> None:
        """Initialize the models list."""
        if not self._models:
            self._models = await self._fetch_available_models()
            self._current_model_index = 0
            logger.info(
                f"Initialized model rotation with: {', '.join(self._models[:5])}{'...' if len(self._models) > 5 else ''}"
            )

    def _get_next_model(self) -> str:
        """Get the next model in rotation.

        Returns:
            Next available model, cycling through the list.
        """
        if not self._models:
            raise ValueError("No models available for rotation")

        model = self._models[self._current_model_index]
        self._current_model_index = (self._current_model_index + 1) % len(self._models)
        return model

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error.

        Args:
            error: The exception to check.

        Returns:
            True if this is a rate limit error.
        """
        if isinstance(error, httpx.HTTPStatusError):
            return (
                error.response.status_code
                in self._rotation_config.rate_limit_status_codes
            )

        error_str = str(error).lower()
        rate_limit_indicators = [
            "too many requests",
            "rate limit",
            "quota exceeded",
            "try again later",
        ]

        return any(indicator in error_str for indicator in rate_limit_indicators)

    async def call_llm(
        self,
        messages: List[ChatMessage] | List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call LLM with automatic model rotation on rate limits.

        This is the main interface for the orchestrator. It wraps the GatewayClient
        and automatically rotates models when encountering rate limits.

        Args:
            messages: List of chat messages.
            model: Specific model to use. If None, uses rotation.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            response_format: Response format specification.
            **kwargs: Additional arguments passed to GatewayClient.

        Returns:
            Dictionary with response data including:
            - content: Generated text
            - model: Model used
            - usage: Token usage info
            - success: Whether the call succeeded
            - attempts: Number of models tried

        Raises:
            Exception: If all models fail or a non-rate-limit error occurs.
        """
        # Initialize models on first call
        await self._initialize_models()

        # If specific model requested, use it directly
        if model:
            try:
                response = await self._gateway_client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )
                return {
                    "content": self._extract_content(response),
                    "model": response.model,
                    "usage": response.usage,
                    "success": True,
                    "attempts": 1,
                }
            except Exception as e:
                logger.error(f"Error with specified model '{model}': {e}")
                return {
                    "error": str(e),
                    "success": False,
                    "model": model,
                    "attempts": 1,
                }

        # Use rotation strategy
        last_error = None
        models_tried = []

        for attempt in range(
            min(self._rotation_config.max_rotation_attempts, len(self._models))
        ):
            current_model = self._get_next_model()
            models_tried.append(current_model)

            logger.info(f"Attempt {attempt + 1}: Using model '{current_model}'")

            try:
                response = await self._gateway_client.chat_completion(
                    messages=messages,
                    model=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )

                logger.info(f"✅ Successfully completed with model '{current_model}'")
                self._failed_models.discard(current_model)

                return {
                    "content": self._extract_content(response),
                    "model": response.model,
                    "usage": response.usage,
                    "success": True,
                    "attempts": attempt + 1,
                    "models_tried": models_tried,
                }

            except Exception as e:
                last_error = e

                if self._is_rate_limit_error(e):
                    logger.warning(
                        f"Rate limit on '{current_model}': {str(e)[:100]}..."
                    )
                    self._failed_models.add(current_model)

                    # Calculate wait time (minimum 5 seconds)
                    wait_time = max(self._rotation_config.rotation_delay, 5.0)
                    logger.warning(
                        f"Waiting {wait_time}s before trying next model... "
                        f"({attempt + 1}/{self._rotation_config.max_rotation_attempts})"
                    )

                    await asyncio.sleep(wait_time)
                    continue

                else:
                    logger.error(f"Non-rate-limit error with '{current_model}': {e}")
                    return {
                        "error": str(e),
                        "success": False,
                        " model": current_model,
                        "attempts": attempt + 1,
                        "models_tried": models_tried,
                    }

        # All models failed
        error_msg = f"All {len(models_tried)} models failed. Last error: {last_error}"
        logger.error(error_msg)

        return {
            "error": error_msg,
            "success": False,
            "attempts": len(models_tried),
            "models_tried": models_tried,
        }

    def _extract_content(self, response: ChatCompletionResponse) -> str:
        """Extract text content from response.

        Args:
            response: Chat completion response.

        Returns:
            Extracted text content.
        """
        if not response.choices:
            raise ValueError("No choices in response")

        return str(response.choices[0]["message"]["content"])

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate text from prompt with model rotation support.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            model: Specific model to use. If None, uses rotation.
            json_mode: Whether to request JSON output.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with response data.
        """
        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response_format = {"type": "json_object"} if json_mode else None

        return await self.call_llm(
            messages=messages,
            model=model,
            response_format=response_format,
            **kwargs,
        )

    def get_available_models(self) -> List[str]:
        """Get current list of available models.

        Returns:
            List of model IDs.
        """
        # If models haven't been initialized yet, return fallback models
        if not self._models:
            return self._rotation_config.fallback_models.copy()
        return self._models.copy()

    async def refresh_models(self) -> None:
        """Refresh the models list from the endpoint."""
        logger.info("Refreshing available models...")
        self._models = await self._fetch_available_models()
        self._current_model_index = 0
        logger.info(
            f"Refreshed models: {', '.join(self._models[:5])}{'...' if len(self._models) > 5 else ''}"
        )

    async def close(self) -> None:
        """Close all clients."""
        await self._gateway_client.close()
        if self._discovery_client:
            await self._discovery_client.aclose()
            self._discovery_client = None

    async def __aenter__(self) -> "ModelRotationManager":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()


def create_rotation_manager(
    gateway_url: str = "http://localhost:8087/v1/",
    models_endpoint: str = "http://localhost:8087/v1/models",
    fallback_models: Optional[List[str]] = None,
) -> ModelRotationManager:
    """Create a ModelRotationManager with custom configuration.

    Args:
        gateway_url: LLM Gateway base URL.
        models_endpoint: Models endpoint URL.
        fallback_models: Fallback models if fetch fails.

    Returns:
        Configured ModelRotationManager instance.
    """
    config = GatewayConfig(url=gateway_url)
    rotation_config = ModelRotationConfig(
        models_endpoint=models_endpoint,
        fallback_models=fallback_models or ModelRotationConfig().fallback_models,
    )

    return ModelRotationManager(config=config, rotation_config=rotation_config)

    def get_current_rotation(self) -> List[str]:
        """Get the current model rotation list.

        Returns:
            Copy of the current models list for rotation.
        """
        if not self._models:
            return self._rotation_config.fallback_models.copy()
        return self._models.copy()

"""Model Registry for dynamically fetching and managing LLM Gateway models.

This module provides a centralized registry for fetching, categorizing,
and managing available models from the LLM Gateway with support for
health checking, caching, and provider-based model selection.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import httpx
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ModelData(BaseModel):
    """Model data from LLM Gateway API."""

    id: str = Field(..., description="Model identifier")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    owned_by: str = Field(..., description="Model provider/owner")


class ModelInfo:
    """Enhanced model information with health tracking."""

    def __init__(self, model_data: ModelData) -> None:
        self.id = model_data.id
        self.provider = self._extract_provider(model_data.id, model_data.owned_by)
        self.raw_data = model_data
        self.is_healthy = True
        self.last_checked = 0.0
        self.consecutive_failures = 0

    @staticmethod
    def _extract_provider(model_id: str, owned_by: str) -> str:
        """Extract provider name from model ID or owned_by field."""
        model_lower = model_id.lower()
        owned_lower = owned_by.lower()

        provider_mapping = {
            "kimi": ["kimi", "moonshot"],
            "qwen": ["qwen", "alibaba", "qwen-max", "qwen-plus"],
            "glm": ["glm", "zhipu", "chatglm"],
            "deepseek": ["deepseek"],
            "gemini": ["gemini", "google"],
            "gpt": ["gpt", "openai"],
            "claude": ["claude", "anthropic"],
            "llama": ["llama", "meta"],
            "mistral": ["mistral"],
        }

        for provider, keywords in provider_mapping.items():
            if any(
                keyword in model_lower or keyword in owned_lower for keyword in keywords
            ):
                return provider

        return owned_by.split("/")[0] if "/" in owned_by else owned_by


class ModelRegistryConfig(BaseModel):
    """Model Registry configuration."""

    gateway_url: str = Field(
        default="http://localhost:8087", description="LLM Gateway base URL"
    )
    cache_ttl: float = Field(default=300.0, description="Cache TTL in seconds")
    health_check_interval: float = Field(
        default=60.0, description="Health check interval in seconds"
    )
    max_consecutive_failures: int = Field(
        default=3, description="Max failures before marking unhealthy"
    )
    timeout: float = Field(default=10.0, description="HTTP request timeout")


class ModelRegistry:
    """Dynamic model registry for LLM Gateway."""

    def __init__(self, config: Optional[ModelRegistryConfig] = None) -> None:
        self.config = config or ModelRegistryConfig()
        self.models: Dict[str, List[ModelInfo]] = {}
        self._last_fetch: float = 0
        self._fetch_lock: asyncio.Lock = asyncio.Lock()
        self._client: Optional[httpx.AsyncClient] = None
        self._provider_priority: Dict[str, int] = {
            "kimi": 1,
            "qwen": 2,
            "glm": 3,
            "deepseek": 4,
            "gemini": 5,
            "gpt": 6,
            "claude": 7,
            "llama": 8,
            "mistral": 9,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def _close_client(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_models(
        self, force_refresh: bool = False
    ) -> Dict[str, List[ModelInfo]]:
        """Fetch models from LLM Gateway API.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            Dictionary of provider -> model list
        """
        async with self._fetch_lock:
            now = time.time()
            if not force_refresh and now - self._last_fetch < self.config.cache_ttl:
                return self.models

            try:
                client = await self._get_client()
                url = f"{self.config.gateway_url.rstrip('/')}/v1/models"

                response = await client.get(url)
                response.raise_for_status()

                data = response.json()
                model_list = [ModelData(**model) for model in data.get("data", [])]

                self.models = self._categorize_models(model_list)
                self._last_fetch = now

                logger.info(f"Fetched {len(model_list)} models from gateway")
                return self.models

            except Exception as e:
                logger.error(f"Failed to fetch models: {e}")
                if self.models:
                    logger.warning("Using cached models due to fetch failure")
                    return self.models
                raise

    def _categorize_models(
        self, model_list: List[ModelData]
    ) -> Dict[str, List[ModelInfo]]:
        """Categorize models by provider.

        Args:
            model_list: List of model data from API

        Returns:
            Dictionary mapping providers to model lists
        """
        categorized: Dict[str, List[ModelInfo]] = {}

        for model_data in model_list:
            model_info = ModelInfo(model_data)
            provider = model_info.provider

            if provider not in categorized:
                categorized[provider] = []

            categorized[provider].append(model_info)

        # Sort models within each provider
        for provider in categorized:
            categorized[provider].sort(key=lambda m: m.id)

        return categorized

    async def get_provider_models(self, provider: str) -> List[ModelInfo]:
        """Get all models for a specific provider.

        Args:
            provider: Provider name

        Returns:
            List of model info for the provider
        """
        if not self.models:
            await self.fetch_models()

        return self.models.get(provider.lower(), [])

    async def get_recommended_model(self, provider: str) -> Optional[ModelInfo]:
        """Get the best available model for a provider.

        Args:
            provider: Provider name

        Returns:
            Best model for the provider, or None if unavailable
        """
        models = await self.get_provider_models(provider)

        if not models:
            return None

        # Return first healthy model, or fall back to first model
        for model in models:
            if model.is_healthy:
                return model

        return models[0]

    async def get_fallback_model(
        self, exclude_provider: Optional[str] = None
    ) -> Optional[ModelInfo]:
        """Get a model from any available provider.

        Args:
            exclude_provider: Optional provider to exclude

        Returns:
            Available model from any provider
        """
        if not self.models:
            await self.fetch_models()

        # Sort providers by priority
        sorted_providers = sorted(
            self.models.keys(), key=lambda p: self._provider_priority.get(p, 999)
        )

        for provider in sorted_providers:
            if exclude_provider and provider.lower() == exclude_provider.lower():
                continue

            models = self.models[provider]
            for model in models:
                if model.is_healthy:
                    return model

            # If no healthy models, return first model
            if models:
                return models[0]

        return None

    def should_use_provider(self, provider: str) -> bool:
        """Check if a provider has healthy models.

        Args:
            provider: Provider name

        Returns:
            True if provider has at least one healthy model
        """
        provider = provider.lower()
        if provider not in self.models:
            return False

        return any(model.is_healthy for model in self.models[provider])

    def mark_model_healthy(self, model_id: str, provider: Optional[str] = None) -> None:
        """Mark a model as healthy.

        Args:
            model_id: Model identifier
            provider: Optional provider (will be inferred if None)
        """
        provider = provider or self._extract_provider_from_id(model_id)

        if provider in self.models:
            for model in self.models[provider]:
                if model.id == model_id:
                    model.is_healthy = True
                    model.consecutive_failures = 0
                    model.last_checked = time.time()
                    break

    def mark_model_unhealthy(
        self, model_id: str, provider: Optional[str] = None
    ) -> None:
        """Mark a model as unhealthy.

        Args:
            model_id: Model identifier
            provider: Optional provider (will be inferred if None)
        """
        provider = provider or self._extract_provider_from_id(model_id)

        if provider in self.models:
            for model in self.models[provider]:
                if model.id == model_id:
                    model.consecutive_failures += 1
                    model.last_checked = time.time()

                    if (
                        model.consecutive_failures
                        >= self.config.max_consecutive_failures
                    ):
                        model.is_healthy = False
                        logger.warning(f"Model {model_id} marked as unhealthy")
                    break

    def _extract_provider_from_id(self, model_id: str) -> str:
        """Extract provider from model ID.

        Args:
            model_id: Model identifier

        Returns:
            Provider name
        """
        model_lower = model_id.lower()

        for provider in self.models:
            if provider in model_lower:
                return provider

        return model_id.split("-")[0] if "-" in model_id else "unknown"

    async def get_all_providers(self) -> List[str]:
        """Get list of all available providers.

        Returns:
            List of provider names
        """
        if not self.models:
            await self.fetch_models()

        return list(self.models.keys())

    async def get_rotation_list(self) -> List[ModelInfo]:
        """Get a rotation list of healthy models across all providers.

        Returns:
            List of model info for rotation
        """
        if not self.models:
            await self.fetch_models()

        rotation_list: List[ModelInfo] = []

        # Gather all healthy models
        for provider in self.models:
            for model in self.models[provider]:
                if model.is_healthy:
                    rotation_list.append(model)

        # If no healthy models, include all models
        if not rotation_list:
            for provider in self.models:
                rotation_list.extend(self.models[provider])

        return rotation_list

    async def close(self) -> None:
        """Close the registry and cleanup resources."""
        await self._close_client()


class ModelRegistryManager:
    """Singleton manager for the model registry."""

    _instance: Optional[ModelRegistry] = None
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_registry(
        cls, config: Optional[ModelRegistryConfig] = None
    ) -> ModelRegistry:
        """Get or create the model registry instance.

        Args:
            config: Optional registry configuration

        Returns:
            ModelRegistry instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = ModelRegistry(config)
                try:
                    await cls._instance.fetch_models()
                except Exception as e:
                    logger.error(f"Failed to initialize model registry: {e}")
                    # Don't raise - allow lazy initialization

            return cls._instance

    @classmethod
    async def reset_registry(cls) -> None:
        """Reset the registry instance (useful for testing)."""
        async with cls._lock:
            if cls._instance:
                await cls._instance.close()
                cls._instance = None


async def get_model_registry() -> ModelRegistry:
    """Convenience function to get the model registry.

    Returns:
        ModelRegistry instance
    """
    return await ModelRegistryManager.get_registry()


async def fetch_and_categorize_models(
    gateway_url: str = "http://localhost:8087",
) -> Dict[str, List[ModelInfo]]:
    """Fetch and categorize models from LLM Gateway.

    Args:
        gateway_url: LLM Gateway base URL

    Returns:
        Dictionary of provider -> model list
    """
    registry = ModelRegistry(ModelRegistryConfig(gateway_url=gateway_url))
    return await registry.fetch_models()

"""Rotation middleware for LLM model rotation."""

import asyncio
import logging
from typing import Callable, TypeVar, List
from knowledge_base.config.constants import (
    DEFAULT_LLM_MODEL,
    LLM_GATEWAY_URL,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RotationMiddleware:
    """Middleware that rotates between multiple LLM models."""

    def __init__(
        self,
        models: List[str] | None = None,
        api_url: str = LLM_GATEWAY_URL,
    ):
        self.models = models or [DEFAULT_LLM_MODEL]
        self.api_url = api_url
        self.current_index = 0

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with model rotation."""
        # For now, just use the default model
        # Rotation logic can be enhanced later
        kwargs.setdefault("model", self.models[0])
        kwargs.setdefault("api_url", self.api_url)
        return await func(*args, **kwargs)

    def rotate_model(self) -> str:
        """Rotate to the next available model."""
        self.current_index = (self.current_index + 1) % len(self.models)
        return self.models[self.current_index]

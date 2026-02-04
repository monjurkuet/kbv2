"""Retry middleware for LLM calls."""

import asyncio
import logging
from typing import Callable, TypeVar, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RetryMiddleware:
    """Middleware that adds retry logic to LLM calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.base_delay, min=2, max=10),
            retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
            reraise=True,
        )
        async def _retry_wrapper():
            return await func(*args, **kwargs)

        return await _retry_wrapper()

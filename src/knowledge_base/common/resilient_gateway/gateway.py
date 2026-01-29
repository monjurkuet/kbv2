"""Resilient LLM Gateway Client with Circuit Breaker, Retry, and Model Discovery."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.common.gateway import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    GatewayClient,
    GatewayConfig,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation."""

    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _success_count: int = field(default=0, init=False)
    _opened_time: Optional[datetime] = field(default=None, init=False)

    def can_execute(self) -> bool:
        """Check if request can be executed based on circuit state."""
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self._last_failure_time
                and (datetime.now() - self._last_failure_time).total_seconds()
                >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0  # Reset success count for half-open state
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record successful operation."""
        self._failure_count = 0

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self._last_failure_time
                and (datetime.now() - self._last_failure_time).total_seconds()
                >= self.recovery_timeout
            ):
                # Transition to HALF_OPEN to handle this success
                self._state = CircuitState.HALF_OPEN
                self._success_count = 1  # Count this success in the half-open state
                # If we have enough successes already (this is the first one), close the circuit
                # This would only happen if success_threshold is 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(
                        "Circuit breaker transitioning to CLOSED state after successful requests"
                    )
            else:
                # Recovery timeout hasn't passed yet, keep as OPEN
                pass
        elif self._state == CircuitState.HALF_OPEN:
            # If we're in half-open state, increment success count
            self._success_count += 1
            # If we have enough successes, close the circuit
            if self._success_count >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._success_count = 0
                logger.info(
                    "Circuit breaker transitioning to CLOSED state after successful requests"
                )

    def record_failure(self) -> None:
        """Record failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_time = datetime.now()
            logger.warning(
                f"Circuit breaker transitioning to OPEN state after {self.failure_threshold} failures"
            )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state


class ModelInfo(BaseModel):
    """Model information from discovery service."""

    id: str
    created: int
    owned_by: str
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelDiscoveryService:
    """Service to discover available models from LLM gateway."""

    def __init__(self, gateway_config: GatewayConfig):
        self._config = gateway_config
        self._client: Optional[httpx.AsyncClient] = None
        self._available_models: List[ModelInfo] = []
        self._last_discovery_time: Optional[datetime] = None
        self._discovery_interval = 300  # 5 minutes

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for model discovery."""
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

    async def discover_models(self) -> List[ModelInfo]:
        """Discover available models from the LLM gateway."""
        client = await self._get_client()

        # Check if we need to refresh the model list
        if (
            self._last_discovery_time
            and (datetime.now() - self._last_discovery_time).total_seconds()
            < self._discovery_interval
        ):
            logger.debug("Using cached model list")
            return self._available_models

        try:
            response = await client.get("models")
            response.raise_for_status()

            data = response.json()
            models_data = data.get("data", [])

            self._available_models = [ModelInfo(**model) for model in models_data]
            self._last_discovery_time = datetime.now()

            logger.info(f"Discovered {len(self._available_models)} models")
            return self._available_models
        except Exception as e:
            logger.error(f"Failed to discover models: {e}")
            # Return cached models if available, otherwise empty list
            return self._available_models if self._available_models else []

    async def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        models = await self.discover_models()
        return [model.id for model in models]

    async def close(self) -> None:
        """Close discovery client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class ResilientGatewayConfig(GatewayConfig):
    """Extended configuration for resilient gateway."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Override the default model to use gemini-2.5-flash-lite
    model: str = "gemini-2.5-flash-lite"

    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_success_threshold: int = 3

    # Retry settings
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_jitter: bool = True
    retry_on_status_codes: List[int] = [429, 502, 503, 504]

    # Model switching settings
    model_switching_enabled: bool = True
    fallback_models: List[str] = []

    # Continuous rotation settings
    continuous_rotation_enabled: bool = True  # If True, loop through models until success
    rotation_delay: float = 5.0  # Delay in seconds between full rotation cycles

    # Metrics settings
    enable_metrics: bool = True


class GatewayMetrics:
    """Metrics collection for gateway performance."""

    def __init__(self):
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.rate_limited_requests: int = 0
        self.retry_attempts: int = 0
        self.model_switches: int = 0
        self.start_time = time.time()

        # Model-specific metrics
        self.model_metrics: Dict[str, Dict[str, int]] = {}

    def record_request(
        self, model: str, success: bool, status_code: Optional[int] = None
    ):
        """Record a request outcome."""
        self.total_requests += 1

        # Update model-specific metrics
        if model not in self.model_metrics:
            self.model_metrics[model] = {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "rate_limits": 0,
            }

        self.model_metrics[model]["requests"] += 1

        if success:
            self.successful_requests += 1
            self.model_metrics[model]["successes"] += 1
        else:
            self.failed_requests += 1
            self.model_metrics[model]["failures"] += 1

        if status_code == 429:
            self.rate_limited_requests += 1
            self.model_metrics[model]["rate_limits"] += 1

    def record_retry(self):
        """Record a retry attempt."""
        self.retry_attempts += 1

    def record_model_switch(self):
        """Record a model switch."""
        self.model_switches += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        total_time = time.time() - self.start_time
        success_rate = (self.successful_requests / max(self.total_requests, 1)) * 100

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "retry_attempts": self.retry_attempts,
            "model_switches": self.model_switches,
            "success_rate": round(success_rate, 2),
            "uptime_seconds": round(total_time, 2),
            "model_metrics": self.model_metrics,
        }


async def exponential_backoff(
    attempt: int, base_delay: float, max_delay: float, jitter: bool = True
):
    """Implement exponential backoff with optional jitter."""
    delay = min(base_delay * (2**attempt), max_delay)

    if jitter:
        import random

        delay = delay * (0.5 + random.random() * 0.5)  # Add 50-100% jitter

    logger.debug(f"Waiting {delay:.2f} seconds before retry {attempt + 1}")
    await asyncio.sleep(delay)


class ResilientGatewayClient:
    """Resilient LLM Gateway Client with advanced features."""

    def __init__(self, config: ResilientGatewayConfig | None = None) -> None:
        """Initialize resilient gateway client.

        Args:
            config: Resilient gateway configuration. If None, loads from environment.
        """
        self._config = config or ResilientGatewayConfig()
        self._base_client = GatewayClient(self._config)
        self._model_discovery = ModelDiscoveryService(self._config)

        # Initialize circuit breaker for each model
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._initialize_circuit_breakers()

        # Initialize metrics
        self._metrics = GatewayMetrics() if self._config.enable_metrics else None

        # Track active models
        self._active_models: Set[str] = set()
        self._current_model_index = 0

        # List of available models for random selection
        self._available_models_list: List[str] = []
        self._random_model_selection = True  # Enable random model selection by default

    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for configured models."""
        all_models = [self._config.model] + self._config.fallback_models
        for model in all_models:
            self._circuit_breakers[model] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_failure_threshold,
                recovery_timeout=self._config.circuit_breaker_recovery_timeout,
                success_threshold=self._config.circuit_breaker_success_threshold,
            )

    async def _initialize_available_models(self):
        """Initialize list of available models for random selection."""
        if not self._available_models_list:
            models = await self._get_available_models()
            if models:
                self._available_models_list = models
                logger.info(f"Initialized {len(models)} models for random selection: {models}")

    def _select_random_model(self) -> str:
        """Select a random model from available models."""
        if not self._available_models_list:
            # Fallback to configured model if list not initialized
            return self._config.model

        # Filter out models with open circuit breakers
        available = [
            model for model in self._available_models_list
            if self._circuit_breakers.get(model, CircuitBreaker()).can_execute()
        ]

        if not available:
            # If all circuits are open, return default model (circuit breaker will handle it)
            logger.warning("All model circuits are open, falling back to default model")
            return self._config.model

        # Select random model
        selected = random.choice(available)
        logger.debug(f"Randomly selected model: {selected} from {len(available)} available models")
        return selected

    def get_metrics(self) -> Dict[str, Any]:
        """Get current gateway metrics."""
        if self._metrics:
            return self._metrics.get_metrics()
        return {}

    async def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        if self._config.model_switching_enabled:
            return await self._model_discovery.get_available_models()
        else:
            return [self._config.model] + self._config.fallback_models

    async def _get_circuit_breaker(self, model: str) -> CircuitBreaker:
        """Get or create circuit breaker for model."""
        if model not in self._circuit_breakers:
            self._circuit_breakers[model] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_failure_threshold,
                recovery_timeout=self._config.circuit_breaker_recovery_timeout,
                success_threshold=self._config.circuit_breaker_success_threshold,
            )
        return self._circuit_breakers[model]

    async def _should_retry(self, status_code: int) -> bool:
        """Determine if request should be retried."""
        return status_code in self._config.retry_on_status_codes

    async def _try_model(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionResponse | None, int]:
        """Try to execute request with a specific model."""
        model = request.model
        circuit_breaker = await self._get_circuit_breaker(model)

        if not circuit_breaker.can_execute():
            logger.debug(f"Circuit breaker prevents execution for model {model}")
            return None, 503  # Service unavailable due to circuit breaker

        try:
            response = await self._base_client.chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=request.response_format,
            )
            circuit_breaker.record_success()
            return response, 200
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.warning(f"Model {model} failed with status {status_code}: {e}")
            circuit_breaker.record_failure()
            return None, status_code
        except Exception as e:
            logger.error(f"Model {model} failed with error: {e}")
            circuit_breaker.record_failure()
            return None, 500

    async def _execute_with_retry(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionResponse | None, int]:
        """Execute request with retry logic."""
        status_code = 500
        last_error = None

        # If continuous rotation is enabled, don't retry same model (rotate instead)
        if getattr(self._config, 'continuous_rotation_enabled', True):
            # Try once without retry
            response, status_code = await self._try_model(request)
            if response is not None:
                return response, 200
            return None, status_code

        # Original retry logic for single-pass mode
        for attempt in range(self._config.retry_max_attempts + 1):
            if attempt > 0:
                if self._metrics:
                    self._metrics.record_retry()

                await exponential_backoff(
                    attempt - 1,
                    self._config.retry_base_delay,
                    self._config.retry_max_delay,
                    self._config.retry_jitter,
                )

            response, status_code = await self._try_model(request)

            if response is not None:
                return response, 200

            # Check if we should continue retrying
            if not await self._should_retry(status_code):
                break

            logger.info(
                f"Attempt {attempt + 1} failed with status {status_code}, retrying..."
            )
            last_error = status_code

        return None, status_code

    async def _get_next_available_model(
        self, current_model: str, original_request: ChatCompletionRequest
    ) -> str:
        """Get next available model for fallback."""
        available_models = await self._get_available_models()

        # Find next model that's not the current one and has a closed circuit
        for i in range(len(available_models)):
            model = available_models[i]
            if model == current_model:
                continue

            circuit_breaker = await self._get_circuit_breaker(model)
            if circuit_breaker.can_execute():
                logger.info(f"Switching from {current_model} to {model}")
                if self._metrics:
                    self._metrics.record_model_switch()
                return model

        # If no other model is available, return the original model (for circuit breaker to handle)
        return current_model

    async def chat_completion(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResponse:
        """Send chat completion request with resilience features.

        Args:
            messages: List of chat messages.
            model: Model name. If None, uses config default or random selection.
            temperature: Sampling temperature. If None, uses config default.
            max_tokens: Maximum tokens to generate. If None, uses config default.
            response_format: Response format specification.

        Returns:
            Chat completion response.

        Raises:
            httpx.HTTPError: If all retry attempts and model switches fail.
        """
        # Initialize available models list if needed
        if self._random_model_selection and not self._available_models_list:
            await self._initialize_available_models()

        # Normalize messages
        normalized_messages = [
            ChatMessage.model_validate(msg) if isinstance(msg, dict) else msg
            for msg in messages
        ]

        # Select model - use random selection if enabled and no specific model requested
        selected_model = model
        if not selected_model and self._random_model_selection:
            selected_model = self._select_random_model()
        elif not selected_model:
            selected_model = self._config.model

        # Prepare request
        request = ChatCompletionRequest(
            model=selected_model,
            messages=normalized_messages,
            temperature=temperature or self._config.temperature,
            max_tokens=max_tokens or self._config.max_tokens,
            response_format=response_format,
        )

        # Track original model for error reporting
        original_model = request.model
        final_response = None
        final_status_code = 500

        # First try with original model
        response, status_code = await self._execute_with_retry(request)
        final_response = response
        final_status_code = status_code

        # If we got an error and model switching is enabled, handle model rotation
        if final_response is None and self._config.model_switching_enabled:
            available_models = await self._get_available_models()
            # Skip the original model since we already tried it
            other_models = [m for m in available_models if m != original_model]

            # Check if continuous rotation is enabled (default: True)
            if getattr(self._config, 'continuous_rotation_enabled', True):
                # Continuous rotation mode: keep looping through models until success
                cycle_count = 0

                # Keep looping through models until we get a successful response
                while final_response is None:
                    cycle_count += 1
                    logger.info(f"Model rotation cycle {cycle_count} (status: {status_code}), trying {len(other_models)} models")

                    # Try each available model (skipping the original)
                    for model in other_models:
                        # Skip if this model just failed (unless it's been a while)
                        circuit_breaker = await self._get_circuit_breaker(model)
                        if not circuit_breaker.can_execute():
                            logger.debug(f"Skipping {model} - circuit breaker open")
                            continue

                        # Record model switch
                        if self._metrics:
                            self._metrics.record_model_switch()

                        # Create request for this model
                        model_request = ChatCompletionRequest(
                            model=model,
                            messages=request.messages,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            response_format=request.response_format,
                        )

                        # Try the model (without retry when continuous rotation is enabled)
                        response, status_code = await self._execute_with_retry(model_request)

                        if response is not None:
                            logger.info(f"Success with model {model} after {cycle_count} cycles")
                            final_response = response
                            final_status_code = 200
                            break

                        # Small delay between model attempts to avoid hammering
                        await asyncio.sleep(0.1)

                    # If all models failed, wait before next cycle (regardless of error type)
                    if final_response is None:
                        logger.info(f"All models failed in cycle {cycle_count} (last status: {status_code}), waiting {self._config.rotation_delay}s")
                        await asyncio.sleep(self._config.rotation_delay)
            else:
                # Single-pass mode: try each model once (original behavior)
                logger.info(f"Rate limited on {original_model}, trying {len(other_models)} other models")

                for other_model in other_models:
                    # Check circuit breaker before trying
                    circuit_breaker = await self._get_circuit_breaker(other_model)
                    if not circuit_breaker.can_execute():
                        logger.debug(f"Skipping {other_model} - circuit breaker open")
                        continue

                    # Create a new request with the different model
                    switched_request = ChatCompletionRequest(
                        model=other_model,
                        messages=request.messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        response_format=request.response_format,
                    )

                    # Try with new model (with retry logic)
                    response, status_code = await self._execute_with_retry(switched_request)

                    if response is not None:
                        logger.info(f"Successfully completed request using model {other_model}")
                        final_response = response
                        final_status_code = 200
                        break

                    # If response was received (any success), break
                    if response is not None:
                        final_response = response
                        final_status_code = 200
                        break

                # If all models failed, try fallback models as a last resort
                if final_response is None:
                    for fallback_model in self._config.fallback_models:
                        if fallback_model == original_model:
                            continue

                        # Check circuit breaker for fallback model
                        fb_circuit_breaker = await self._get_circuit_breaker(fallback_model)
                        if not fb_circuit_breaker.can_execute():
                            logger.debug(f"Skipping fallback {fallback_model} - circuit breaker open")
                            continue

                        # Record model switch
                        if self._metrics:
                            self._metrics.record_model_switch()

                        fb_request = ChatCompletionRequest(
                            model=fallback_model,
                            messages=request.messages,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            response_format=request.response_format,
                        )
                        response, status_code = await self._execute_with_retry(fb_request)

                        if response is not None:
                            logger.info(f"Successfully completed request using fallback model {fallback_model}")
                            final_response = response
                            final_status_code = 200
                            break

        # Record the final result for metrics
        if self._metrics:
            # Use the original model for metric tracking to represent the logical request
            self._metrics.record_request(
                original_model,
                success=(final_response is not None),
                status_code=final_status_code,
            )

        # If we still failed, raise the last error
        if final_response is None:
            error_msg = f"LLM request failed after all retries and model switches. Last status: {final_status_code}, Original model: {original_model}"
            logger.error(error_msg)
            if final_status_code == 429:
                raise httpx.HTTPStatusError(
                    f"Rate limited: {error_msg}",
                    request=httpx.Request("POST", "chat/completions"),
                    response=httpx.Response(
                        status_code=429, content=error_msg.encode()
                    ),
                )
            else:
                raise httpx.HTTPStatusError(
                    error_msg,
                    request=httpx.Request("POST", "chat/completions"),
                    response=httpx.Response(
                        status_code=final_status_code, content=error_msg.encode()
                    ),
                )

        return final_response

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate text from prompt with resilience features.

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
        """Close all clients."""
        await self._base_client.close()
        await self._model_discovery.close()

    async def __aenter__(self) -> "ResilientGatewayClient":
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context."""
        await self.close()

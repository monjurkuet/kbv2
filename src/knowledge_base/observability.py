"""SRE-Lite observability with Logfire."""

import time
from contextlib import asynccontextmanager
from typing import Any, Callable
from functools import wraps

import logfire
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogfireConfig(BaseSettings):
    """Logfire configuration."""

    model_config = SettingsConfigDict(env_prefix="LOGFIRE_")

    project: str = "knowledge-base"
    send_to_logfire: bool = False
    sampling_rate: float = 1.0


class Observability:
    """SRE-Lite observability manager."""

    _instance: "Observability | None" = None
    _initialized: bool = False

    def __init__(self, config: LogfireConfig | None = None) -> None:
        """Initialize observability.

        Args:
            config: Logfire configuration.
        """
        if Observability._initialized:
            return

        self._config = config or LogfireConfig()
        self._initialize_logfire()
        Observability._initialized = True
        Observability._instance = self

    def _initialize_logfire(self) -> None:
        """Initialize Logfire client."""
        if self._config.send_to_logfire:
            logfire.configure(
                send_to_logfire=True,
            )
        else:
            logfire.configure(
                send_to_logfire=False,
            )

    @classmethod
    def get_instance(cls) -> "Observability":
        """Get singleton instance.

        Returns:
            Observability instance.

        Raises:
            RuntimeError: If not initialized.
        """
        if cls._instance is None:
            raise RuntimeError("Observability not initialized")
        return cls._instance

    def trace_operation(
        self,
        operation_name: str,
        **tags: Any,
    ) -> Callable[..., Any]:
        """Decorator to trace operations.

        Args:
            operation_name: Name of the operation.
            **tags: Additional tags to include.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                with logfire.span(
                    operation_name,
                    **tags,
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        span.set_attribute("success", True)
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )

                        logfire.info(
                            "operation_completed",
                            operation=operation_name,
                            duration_ms=(time.time() - start_time) * 1000,
                        )

                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )

                        logfire.error(
                            "operation_failed",
                            operation=operation_name,
                            error=str(e),
                            duration_ms=(time.time() - start_time) * 1000,
                        )

                        raise

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = time.time()

                with logfire.span(
                    operation_name,
                    **tags,
                ) as span:
                    try:
                        result = func(*args, **kwargs)

                        span.set_attribute("success", True)
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )

                        logfire.info(
                            "operation_completed",
                            operation=operation_name,
                            duration_ms=(time.time() - start_time) * 1000,
                        )

                        return result
                    except Exception as e:
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.set_attribute(
                            "duration_ms", (time.time() - start_time) * 1000
                        )

                        logfire.error(
                            "operation_failed",
                            operation=operation_name,
                            error=str(e),
                            duration_ms=(time.time() - start_time) * 1000,
                        )

                        raise

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @asynccontextmanager
    async def trace_context(
        self,
        operation_name: str,
        **tags: Any,
    ):
        """Context manager for tracing operations.

        Args:
            operation_name: Name of the operation.
            **tags: Additional tags.

        Yields:
            Span context.
        """
        start_time = time.time()

        with logfire.span(
            operation_name,
            **tags,
        ) as span:
            try:
                yield span

                span.set_attribute("success", True)
                span.set_attribute("duration_ms", (time.time() - start_time) * 1000)

                logfire.info(
                    "operation_completed",
                    operation=operation_name,
                    duration_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                span.set_attribute("duration_ms", (time.time() - start_time) * 1000)

                logfire.error(
                    "operation_failed",
                    operation=operation_name,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                )

                raise

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        **tags: Any,
    ) -> None:
        """Log a metric.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            unit: Unit of measurement.
            **tags: Additional tags.
        """
        logfire.info(
            f"{metric_name}: {value} {unit}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **tags,
        )

    def log_event(
        self,
        event_name: str,
        level: str = "info",
        **attributes: Any,
    ) -> None:
        """Log an event.

        Args:
            event_name: Name of the event.
            level: Log level (debug, info, warning, error).
            **attributes: Event attributes.
        """
        log_method = getattr(logfire, level, logfire.info)
        log_method(
            event_name,
            **attributes,
        )

    def track_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        **tags: Any,
    ) -> None:
        """Track token usage.

        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            model: Model name.
            **tags: Additional tags.
        """
        total_tokens = prompt_tokens + completion_tokens

        logfire.info(
            f"LLM tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            **tags,
        )


def get_observability() -> Observability:
    """Get observability instance.

    Returns:
        Observability instance.

    Raises:
        RuntimeError: If not initialized.
    """
    return Observability.get_instance()


def trace_operation(operation_name: str, **tags: Any) -> Callable[..., Any]:
    """Decorator to trace operations.

    Args:
        operation_name: Name of the operation.
        **tags: Additional tags.

    Returns:
        Decorator function.
    """
    return get_observability().trace_operation(operation_name, **tags)

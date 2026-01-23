"""Compatibility layer to maintain backward compatibility with existing GatewayClient."""

from typing import Any

from knowledge_base.common.gateway import ChatMessage, GatewayClient, GatewayConfig
from knowledge_base.common.resilient_gateway.gateway import (
    ResilientGatewayClient,
    ResilientGatewayConfig,
)


class GatewayClientWrapper(GatewayClient):
    """Wrapper around ResilientGatewayClient to maintain backward compatibility."""

    def __init__(self, config: GatewayConfig | None = None) -> None:
        """Initialize wrapper with resilient client.

        Args:
            config: Gateway configuration. If None, loads from environment.
        """
        # Convert GatewayConfig to ResilientGatewayConfig
        if config is None:
            resilient_config = ResilientGatewayConfig()
        else:
            # Create resilient config from regular config
            resilient_config = ResilientGatewayConfig(
                url=config.url,
                api_key=config.api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                # Keep default resilient settings
            )

        # Initialize the resilient client
        self._resilient_client = ResilientGatewayClient(resilient_config)

        # Maintain compatibility by using the same config structure
        self._config = config or GatewayConfig()
        self._client = None  # For backward compatibility with base class properties

    async def chat_completion(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ):
        """Forward to resilient client."""
        return await self._resilient_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> str:
        """Forward to resilient client."""
        return await self._resilient_client.generate_text(
            prompt=prompt, system_prompt=system_prompt, json_mode=json_mode, **kwargs
        )

    async def close(self) -> None:
        """Close resilient client."""
        await self._resilient_client.close()

    def get_metrics(self):
        """Get metrics from resilient client."""
        return self._resilient_client.get_metrics()


# Override the original GatewayClient to maintain compatibility
# This allows existing imports to work seamlessly
GatewayClient = GatewayClientWrapper

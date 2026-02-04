"""Client interfaces for KBV2 LLM interactions."""

import warnings

warnings.warn(
    "knowledge_base.clients is deprecated since v0.6.0. "
    "Use knowledge_base.common.resilient_gateway for LLM operations.",
    DeprecationWarning,
    stacklevel=2,
)

from knowledge_base.clients.llm_client import (
    LLMClient,
    LLMClientConfig,
    ChatMessage,
    LLMRequest,
    LLMResponse,
    PromptingStrategy,
    FewShotExample,
    create_llm_client,
)

from knowledge_base.clients.rotating_llm_client import (
    RotatingLLMClient,
    ModelRotationConfig,
    create_rotating_llm_client,
    RECOMMENDED_ROTATIONS,
)

__all__ = [
    # Original LLM Client
    "LLMClient",
    "LLMClientConfig",
    "ChatMessage",
    "LLMRequest",
    "LLMResponse",
    "PromptingStrategy",
    "FewShotExample",
    "create_llm_client",
    # Rotating LLM Client
    "RotatingLLMClient",
    "ModelRotationConfig",
    "create_rotating_llm_client",
    "RECOMMENDED_ROTATIONS",
]

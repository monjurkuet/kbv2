"""Client interfaces for KBV2 LLM interactions."""

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

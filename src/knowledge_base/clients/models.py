"""Shared data models for LLM clients."""

from typing import Optional
from pydantic import BaseModel


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""

    model_name: str
    api_url: str
    timeout: float = 120.0
    max_retries: int = 3

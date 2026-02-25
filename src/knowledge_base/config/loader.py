"""Configuration loader for KBV2.

Loads configuration from YAML file with environment variable overrides for secrets.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM configuration."""

    gateway_url: str = "http://localhost:8087/v1"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 4096


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    api_base: str = "http://localhost:11434"
    model: str = "bge-m3"
    dimension: int = 1024


class StorageConfig(BaseModel):
    """Storage configuration."""

    data_dir: str = "data"


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_chunk_size: int = 1536


class DomainConfig(BaseModel):
    """Domain detection configuration."""

    default_domain: str = "GENERAL"
    confidence_threshold: float = 0.7


class RAGConfig(BaseModel):
    """RAG pipeline configuration."""

    default_mode: str = "HYBRID"
    top_k: int = 10


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8088
    websocket_port: int = 8765


class CORSConfig(BaseModel):
    """CORS configuration."""

    allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = False
    allow_methods: list[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class Config(BaseModel):
    """Root configuration model."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)


_config: Optional[Config] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Config object with loaded or default values.
    """
    global _config

    if _config is not None:
        return _config

    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _config = Config(**data)
    else:
        _config = Config()

    return _config


def get_config() -> Config:
    """Get the current configuration, loading if necessary.

    Returns:
        Config object.
    """
    if _config is None:
        return load_config()
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """Force reload configuration from file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Newly loaded Config object.
    """
    global _config
    _config = None
    return load_config(config_path)

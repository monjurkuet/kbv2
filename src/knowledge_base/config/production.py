"""Production deployment configuration for KBv2 Crypto Knowledgebase.

This module provides configuration for production deployment with monitoring,
metrics collection, and health checks.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for KBv2."""

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://agentzero@localhost/knowledge_base"
        )
    )
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_timeout: int = 30

    # LLM Gateway
    llm_api_base: str = field(
        default_factory=lambda: os.getenv("LLM_API_BASE", "http://localhost:8087/v1")
    )
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "default"))
    llm_timeout: int = 120
    llm_max_retries: int = 3

    # Embedding (Ollama)
    embedding_api_base: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_BASE", "http://localhost:11434"
        )
    )
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = 1024

    # Self-Improvement Features
    enable_experience_bank: bool = True
    enable_prompt_evolution: bool = True
    enable_ontology_validation: bool = True

    # Experience Bank Settings
    experience_bank_min_quality: float = 0.85
    experience_bank_max_size: int = 10000
    experience_bank_retrieval_k: int = 3

    # Prompt Evolution Settings
    prompt_evolution_enabled_domains: List[str] = field(
        default_factory=lambda: [
            "BITCOIN",
            "DEFI",
            "INSTITUTIONAL_CRYPTO",
            "STABLECOINS",
            "CRYPTO_REGULATION",
        ]
    )
    prompt_evolution_evaluation_interval: int = 86400  # 24 hours

    # Monitoring & Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    metrics_endpoint: str = "/metrics"

    # Health Check
    health_check_port: int = 8080
    health_check_endpoint: str = "/health"

    # Performance
    max_concurrent_documents: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Data Pipeline Integration
    data_pipeline_webhook_url: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_WEBHOOK_URL")
    )
    data_pipeline_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DATA_PIPELINE_API_KEY")
    )
    enable_realtime_data: bool = field(
        default_factory=lambda: os.getenv("ENABLE_REALTIME_DATA", "false").lower()
        == "true"
    )

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "database_url": self.database_url.split("@")[-1]
            if "@" in self.database_url
            else "configured",
            "llm_api_base": self.llm_api_base,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "enable_experience_bank": self.enable_experience_bank,
            "enable_prompt_evolution": self.enable_prompt_evolution,
            "enable_ontology_validation": self.enable_ontology_validation,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "health_check_port": self.health_check_port,
            "max_concurrent_documents": self.max_concurrent_documents,
        }


# Global config instance
_config: Optional[ProductionConfig] = None


def get_config() -> ProductionConfig:
    """Get or create production config singleton."""
    global _config
    if _config is None:
        _config = ProductionConfig()
        logger.info("Production configuration initialized")
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None


# Environment-based configuration presets
PRODUCTION_PRESETS = {
    "development": {
        "enable_metrics": False,
        "max_concurrent_documents": 2,
        "experience_bank_min_quality": 0.80,
    },
    "staging": {
        "enable_metrics": True,
        "max_concurrent_documents": 3,
        "experience_bank_min_quality": 0.85,
    },
    "production": {
        "enable_metrics": True,
        "max_concurrent_documents": 5,
        "experience_bank_min_quality": 0.90,
    },
}


def apply_preset(env: str) -> ProductionConfig:
    """Apply environment preset to config.

    Args:
        env: Environment name (development, staging, production)

    Returns:
        Config with preset applied
    """
    config = get_config()
    preset = PRODUCTION_PRESETS.get(env, PRODUCTION_PRESETS["production"])

    for key, value in preset.items():
        setattr(config, key, value)

    logger.info(f"Applied {env} preset to configuration")
    return config

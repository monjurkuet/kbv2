"""Centralized constants for KBV2."""

from typing import Final

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================
LLM_GATEWAY_URL: Final[str] = "http://localhost:8087/v1"
LLM_GATEWAY_PORT: Final[int] = 8087
WEBSOCKET_PORT: Final[int] = 8765
DATABASE_PORT: Final[int] = 5432
EMBEDDING_URL: Final[str] = "http://localhost:11434"

# ============================================================================
# TIMEOUTS
# ============================================================================
DEFAULT_LLM_TIMEOUT: Final[float] = 120.0
DEFAULT_HTTP_TIMEOUT: Final[float] = 60.0
ROTATION_DELAY: Final[float] = 5.0
INGESTION_TIMEOUT: Final[float] = 3600.0

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_CHUNK_OVERLAP: Final[int] = 50
SEMANTIC_CHUNK_SIZE: Final[int] = 1536
OVERLAP_RATIO: Final[float] = 0.25

# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================
MIN_EXTRACTION_QUALITY_SCORE: Final[float] = 0.5
ENTITY_SIMILARITY_THRESHOLD: Final[float] = 0.85
DOMAIN_CONFIDENCE_THRESHOLD: Final[float] = 0.6
HALLUCINATION_THRESHOLD: Final[float] = 0.3

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================
VECTOR_SEARCH_WEIGHT: Final[float] = 0.6
GRAPH_SEARCH_WEIGHT: Final[float] = 0.4
DEFAULT_TOP_K_RESULTS: Final[int] = 10

# ============================================================================
# ENTITY CONFIGURATION
# ============================================================================
MIN_ENTITY_CONFIDENCE: Final[float] = 0.5
MAX_LONG_TAIL_ENTITIES: Final[int] = 5
LONG_TAIL_THRESHOLD: Final[int] = 20

# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================
DEFAULT_CLUSTER_RESOLUTION: Final[float] = 1.0

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
DEFAULT_EMBEDDING_MODEL: Final[str] = "nomic-embed-text"
EMBEDDING_DIMENSION: Final[int] = 768

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
DEFAULT_LLM_MODEL: Final[str] = "gemini-2.5-flash-lite"
MAX_RETRIES: Final[int] = 3
RETRY_DELAY: Final[float] = 1.0

# ============================================================================
# BATCH CONFIGURATION
# ============================================================================
DEFAULT_BATCH_SIZE: Final[int] = 10
MAX_CONCURRENT_REQUESTS: Final[int] = 5

# ============================================================================
# REVIEW QUEUE CONFIGURATION
# ============================================================================
DEFAULT_REVIEW_PRIORITY: Final[int] = 5

# ============================================================================
# COMMUNITY CONFIGURATION
# ============================================================================
MIN_COMMUNITY_SIZE: Final[int] = 3
MAX_COMMUNITY_LEVELS: Final[int] = 4

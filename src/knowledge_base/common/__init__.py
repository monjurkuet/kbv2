"""Common utilities for KBV2."""

from .exceptions import (
    KBV2BaseException,
    # Ingestion
    IngestionError,
    DocumentParseError,
    ChunkingError,
    EmbeddingError,
    # Extraction
    ExtractionError,
    EntityExtractionError,
    RelationshipExtractionError,
    HallucinationDetectionError,
    # Resolution
    ResolutionError,
    EntityResolutionError,
    DuplicateEntityError,
    # Validation
    ValidationError,
    SchemaValidationError,
    TypeValidationError,
    DomainValidationError,
    # LLM Client
    LLMClientError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMConnectionError,
    LLMResponseError,
    CircuitBreakerOpenError,
    # Database
    DatabaseError,
    SessionError,
    QueryError,
    MigrationError,
    # Configuration
    ConfigurationError,
    MissingConfigurationError,
    InvalidConfigurationError,
    # Clustering
    ClusteringError,
    ClusteringResolutionError,
    InsufficientDataError,
    # Review Queue
    ReviewQueueError,
    ReviewItemNotFoundError,
    ReviewQueueFullError,
)

from .dependencies import (
    get_config,
    set_config,
    get_sqlite_store,
    get_chroma_store,
    get_kuzu_store,
    get_hybrid_search,
    initialize_storage,
    close_storage,
)

from .offset_service import (
    TextPositionSelector,
    TextQuoteSelector,
    TextSpan,
    OffsetCalculationService,
    HighlightService,
)

__all__ = [
    # Exceptions
    "KBV2BaseException",
    "IngestionError",
    "DocumentParseError",
    "ChunkingError",
    "EmbeddingError",
    "ExtractionError",
    "EntityExtractionError",
    "RelationshipExtractionError",
    "HallucinationDetectionError",
    "ResolutionError",
    "EntityResolutionError",
    "DuplicateEntityError",
    "ValidationError",
    "SchemaValidationError",
    "TypeValidationError",
    "DomainValidationError",
    "LLMClientError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMResponseError",
    "CircuitBreakerOpenError",
    "DatabaseError",
    "SessionError",
    "QueryError",
    "MigrationError",
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    "ClusteringError",
    "ClusteringResolutionError",
    "InsufficientDataError",
    "ReviewQueueError",
    "ReviewItemNotFoundError",
    "ReviewQueueFullError",
    # Dependencies
    "get_config",
    "set_config",
    "get_sqlite_store",
    "get_chroma_store",
    "get_kuzu_store",
    "get_hybrid_search",
    "initialize_storage",
    "close_storage",
    # Offset Service
    "TextPositionSelector",
    "TextQuoteSelector",
    "TextSpan",
    "OffsetCalculationService",
    "HighlightService",
]

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

__all__ = [
    "KBV2BaseException",
    # Ingestion
    "IngestionError",
    "DocumentParseError",
    "ChunkingError",
    "EmbeddingError",
    # Extraction
    "ExtractionError",
    "EntityExtractionError",
    "RelationshipExtractionError",
    "HallucinationDetectionError",
    # Resolution
    "ResolutionError",
    "EntityResolutionError",
    "DuplicateEntityError",
    # Validation
    "ValidationError",
    "SchemaValidationError",
    "TypeValidationError",
    "DomainValidationError",
    # LLM Client
    "LLMClientError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMConnectionError",
    "LLMResponseError",
    "CircuitBreakerOpenError",
    # Database
    "DatabaseError",
    "SessionError",
    "QueryError",
    "MigrationError",
    # Configuration
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    # Clustering
    "ClusteringError",
    "ClusteringResolutionError",
    "InsufficientDataError",
    # Review Queue
    "ReviewQueueError",
    "ReviewItemNotFoundError",
    "ReviewQueueFullError",
]

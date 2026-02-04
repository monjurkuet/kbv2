"""Custom exception hierarchy for KBV2."""

from typing import Optional, Dict, Any


class KBV2BaseException(Exception):
    """Base exception for all KBV2 errors.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        context: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message


# ============================================================================
# INGESTION EXCEPTIONS
# ============================================================================


class IngestionError(KBV2BaseException):
    """Base exception for document ingestion errors."""

    pass


class DocumentParseError(IngestionError):
    """Raised when document parsing fails."""

    pass


class ChunkingError(IngestionError):
    """Raised when document chunking fails."""

    pass


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""

    pass


# ============================================================================
# EXTRACTION EXCEPTIONS
# ============================================================================


class ExtractionError(KBV2BaseException):
    """Base exception for entity/relationship extraction errors."""

    pass


class EntityExtractionError(ExtractionError):
    """Raised when entity extraction fails."""

    pass


class RelationshipExtractionError(ExtractionError):
    """Raised when relationship extraction fails."""

    pass


class HallucinationDetectionError(ExtractionError):
    """Raised when hallucination detection fails."""

    pass


# ============================================================================
# RESOLUTION EXCEPTIONS
# ============================================================================


class ResolutionError(KBV2BaseException):
    """Base exception for entity resolution errors."""

    pass


class EntityResolutionError(ResolutionError):
    """Raised when entity resolution fails."""

    pass


class DuplicateEntityError(ResolutionError):
    """Raised when duplicate entities are detected."""

    pass


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================


class ValidationError(KBV2BaseException):
    """Base exception for validation errors."""

    pass


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""

    pass


class TypeValidationError(ValidationError):
    """Raised when type validation fails."""

    pass


class DomainValidationError(ValidationError):
    """Raised when domain validation fails."""

    pass


# ============================================================================
# LLM CLIENT EXCEPTIONS
# ============================================================================


class LLMClientError(KBV2BaseException):
    """Base exception for LLM client errors."""

    pass


class LLMTimeoutError(LLMClientError):
    """Raised when LLM request times out."""

    pass


class LLMRateLimitError(LLMClientError):
    """Raised when LLM rate limit is exceeded."""

    pass


class LLMConnectionError(LLMClientError):
    """Raised when LLM connection fails."""

    pass


class LLMResponseError(LLMClientError):
    """Raised when LLM response is invalid."""

    pass


class CircuitBreakerOpenError(LLMClientError):
    """Raised when circuit breaker is open."""

    pass


# ============================================================================
# DATABASE EXCEPTIONS
# ============================================================================


class DatabaseError(KBV2BaseException):
    """Base exception for database errors."""

    pass


class SessionError(DatabaseError):
    """Raised when database session error occurs."""

    pass


class QueryError(DatabaseError):
    """Raised when database query fails."""

    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""

    pass


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================


class ConfigurationError(KBV2BaseException):
    """Base exception for configuration errors."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""

    pass


# ============================================================================
# CLUSTERING EXCEPTIONS
# ============================================================================


class ClusteringError(KBV2BaseException):
    """Base exception for clustering errors."""

    pass


class ClusteringResolutionError(ClusteringError):
    """Raised when clustering resolution is invalid."""

    pass


class InsufficientDataError(ClusteringError):
    """Raised when there's insufficient data for clustering."""

    pass


# ============================================================================
# REVIEW QUEUE EXCEPTIONS
# ============================================================================


class ReviewQueueError(KBV2BaseException):
    """Base exception for review queue errors."""

    pass


class ReviewItemNotFoundError(ReviewQueueError):
    """Raised when review item is not found."""

    pass


class ReviewQueueFullError(ReviewQueueError):
    """Raised when review queue is full."""

    pass

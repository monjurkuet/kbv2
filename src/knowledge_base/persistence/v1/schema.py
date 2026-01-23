"""Database schema models."""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY, FLOAT as PG_FLOAT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import UserDefinedType

from datetime import datetime
from enum import Enum
from knowledge_base.common.temporal_utils import TemporalType


class Vector(UserDefinedType):
    """Vector type for pgvector."""

    cache_ok = True

    def __init__(self, dimensions=None):
        """Initialize vector type.

        Args:
            dimensions: Vector dimensions.
        """
        self.dimensions = dimensions

    def get_col_spec(self, **kw):
        """Get column specification."""
        if self.dimensions:
            return f"vector({self.dimensions})"
        return "vector"

    def bind_processor(self, dialect):
        """Get bind processor."""

        def process(value):
            if value is None:
                return None
            return f"[{','.join(str(x) for x in value)}]"

        return process

    def result_processor(self, dialect, coltype):
        """Get result processor."""

        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip("[]")
                if value:
                    return [float(x) for x in value.split(",")]
                return []
            return list(value)

        return process


Base = declarative_base()


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PARTITIONED = "partitioned"
    EMBEDDED = "embedded"
    EXTRACTED = "extracted"
    COMPLETED = "completed"
    FAILED = "failed"


class EdgeType(str, Enum):
    """Relationship edge types based on research from 2026 (DOREMI) on document-level relation extraction with long-tail optimization."""

    # Default/fallback relation types - critical for handling long-tail distributions
    UNKNOWN = "unknown"
    NOTA = "none_of_the_above"  # Following NOTA (none-of-the-above) research from recent papers
    HYPOTHETICAL = (
        "hypothetical"  # For uncertain relations (addressing long-tail sparse data)
    )

    # Core semantic relations for document-level extraction
    RELATED_TO = "related_to"  # General relation when specific type is unclear
    MENTIONS = "mentions"  # Document-specific relation
    REFERENCES = "references"  # Document-specific relation
    DISCUSSES = "discusses"  # Document-specific relation

    # Hierarchical relations (common in knowledge graphs)
    PART_OF = "part_of"
    SUBCLASS_OF = "subclass_of"  # Taxonomic relationship
    INSTANCE_OF = "instance_of"  # An entity is an instance of a class
    CONTAINS = "contains"

    # Causal relations (important for knowledge graph completion)
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    INFLUENCES = "influences"
    INFLUENCED_BY = "influenced_by"

    # Temporal relations (for temporal knowledge graphs)
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    CO_OCCURS_WITH = "co_occurs_with"

    # Spatial relations
    LOCATED_IN = "located_in"
    LOCATED_NEAR = "located_near"

    # Social and organizational relations (common in real-world data)
    WORKS_FOR = "works_for"
    WORKS_WITH = "works_with"
    REPORTS_TO = "reports_to"
    KNOWS = "knows"
    COLLEAGUE_OF = "colleague_of"

    # Ownership and possession
    OWNS = "owns"
    MANAGES = "manages"
    OPERATES = "operates"

    # Activity relations (for document-level extraction)
    PARTICIPATES_IN = "participates_in"
    PERFORMS = "performs"
    TARGETS = "targets"
    AFFECTS = "affects"

    # Validity relations (for handling knowledge evolution)
    INVALIDATED_BY = "invalidated_by"
    REPLACED_BY = "replaced_by"
    SUCCEEDED_BY = "succeeded_by"
    PREDECESSOR_OF = "predecessor_of"


class Document(Base):
    """Document model."""

    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(500), nullable=False, index=True)
    source_uri = Column(String(2000), nullable=True)
    mime_type = Column(String(100), nullable=True)
    status = Column(String(50), nullable=False, default=DocumentStatus.PENDING)
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    domain = Column(String(100), nullable=True, index=True)

    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """Document chunk model."""

    __tablename__ = "chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id"),
        nullable=False,
        index=True,
    )
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    chunk_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    embedding = Column(Vector(768), nullable=True)

    document = relationship("Document", back_populates="chunks")
    entities = relationship(
        "Entity", secondary="chunk_entities", back_populates="chunks"
    )

    __table_args__ = (Index("idx_chunk_document_index", "document_id", "chunk_index"),)


class Entity(Base):
    """Entity/Node model following knowledge graph best practices."""

    __tablename__ = "entities"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(500), nullable=False, index=True)
    entity_type = Column(
        String(100), nullable=True, index=True
    )  # Follows standard schema patterns
    description = Column(Text, nullable=True)
    properties = Column(JSON, nullable=True)  # Flexible property storage
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    embedding = Column(Vector(768), nullable=True)
    uri = Column(
        String(500), nullable=True, unique=True
    )  # Unique identifier following RDF patterns
    source_text = Column(Text, nullable=True)  # For provenance tracking
    domain = Column(String(100), nullable=True, index=True)

    chunks = relationship(
        "Chunk", secondary="chunk_entities", back_populates="entities"
    )
    source_edges = relationship(
        "Edge",
        foreign_keys="Edge.source_id",
        back_populates="source_entity",
    )
    target_edges = relationship(
        "Edge",
        foreign_keys="Edge.target_id",
        back_populates="target_entity",
    )
    community_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("communities.id"),
        nullable=True,
        index=True,
    )
    community = relationship("Community", back_populates="entities")


class Edge(Base):
    """Relationship edge model following knowledge graph best practices."""

    __tablename__ = "edges"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
    )
    target_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
    )
    edge_type = Column(
        String(100), nullable=False, index=True
    )  # Standard relationship types
    properties = Column(
        JSON, nullable=True
    )  # Flexible property storage for relationship metadata
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Knowledge graph best practices additions
    temporal_validity_start = Column(
        DateTime, nullable=True
    )  # For temporal knowledge graphs
    temporal_validity_end = Column(
        DateTime, nullable=True
    )  # For temporal knowledge graphs
    provenance = Column(String(200), nullable=True)  # Source of the relationship
    source_text = Column(Text, nullable=True)  # Text that established this relationship
    is_directed = Column(
        Boolean, nullable=False, default=True
    )  # Explicit direction flag
    domain = Column(String(100), nullable=True, index=True)

    source_entity = relationship(
        "Entity",
        foreign_keys=[source_id],
        back_populates="source_edges",
    )
    target_entity = relationship(
        "Entity",
        foreign_keys=[target_id],
        back_populates="target_edges",
    )

    __table_args__ = (
        Index("idx_edge_source_target", "source_id", "target_id"),
        Index("idx_edge_type_confidence", "edge_type", "confidence"),
    )


class Community(Base):
    """Community cluster model."""

    __tablename__ = "communities"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(500), nullable=False)
    level = Column(Integer, nullable=False, default=0, index=True)
    resolution = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    entity_count = Column(Integer, nullable=False, default=0)
    parent_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("communities.id"),
        nullable=True,
    )
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    entities = relationship("Entity", back_populates="community")
    parent = relationship("Community", remote_side=[id])


class ChunkEntity(Base):
    """Chunk-Entity junction table."""

    __tablename__ = "chunk_entities"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("chunks.id"),
        nullable=False,
        index=True,
    )
    entity_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("entities.id"),
        nullable=False,
        index=True,
    )
    grounding_quote = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_chunk_entity_chunk", "chunk_id", "entity_id"),
        Index("idx_chunk_entity_entity", "entity_id", "chunk_id"),
    )


class EntityResolution(BaseModel):
    """Entity resolution result."""

    entity_id: UUID = Field(..., description="Kept entity ID")
    merged_entity_ids: list[UUID] = Field(
        default_factory=list, description="Merged entity IDs"
    )
    grounding_quote: str = Field(
        ..., description="Verbatim quote supporting resolution"
    )
    confidence_score: float = Field(
        ..., description="Confidence in resolution", ge=0.0, le=1.0
    )
    human_review_required: bool = Field(
        default=False, description="Flag for manual review"
    )


class ReviewStatus(str, Enum):
    """Status of review items."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ReviewQueue(Base):
    """Review queue table for tracking items requiring human review."""

    __tablename__ = "review_queue"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    item_type = Column(
        String(50), nullable=False
    )  # 'entity_resolution' or 'edge_validation'
    entity_id = Column(PG_UUID(as_uuid=True), ForeignKey("entities.id"), nullable=True)
    edge_id = Column(PG_UUID(as_uuid=True), ForeignKey("edges.id"), nullable=True)
    document_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True
    )
    merged_entity_ids = Column(JSON)  # IDs of entities to be merged
    confidence_score = Column(Float)  # Confidence score of the original decision
    grounding_quote = Column(Text)  # Quote that supports the resolution
    source_text = Column(Text)  # Original source text
    status = Column(
        String(20), default=ReviewStatus.PENDING
    )  # pending, approved, rejected
    priority = Column(Integer, default=5)  # 1-10 scale for review priority
    reviewer_notes = Column(Text)  # Notes from the reviewer
    reviewed_by = Column(String(100), nullable=True)  # Reviewer identifier
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    entity = relationship("Entity")
    edge = relationship("Edge")
    document = relationship("Document")


class TemporalEntityClaim(BaseModel):
    """Temporal entity claim."""

    entity_id: UUID = Field(..., description="Related entity ID")
    chunk_id: UUID = Field(..., description="Source chunk ID")
    claim_text: str = Field(..., description="Claim text")
    temporal_type: TemporalType = Field(..., description="Temporal classification")
    iso8601_date: str | None = Field(None, description="Normalized ISO-8601 date")
    invalidated_by_id: UUID | None = Field(None, description="ID of invalidating claim")

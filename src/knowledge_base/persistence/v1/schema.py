"""Database schema models."""

from datetime import datetime
from enum import Enum
from typing import Optional
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
from pgvector.sqlalchemy import VECTOR

# Remove Vector class and use VECTOR directly

from knowledge_base.common.temporal_utils import TemporalType


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
    AFFECTS_INVERSE = "affected_by"  # Inverse relation for bidirectional graphs

    # Attribute and characteristic relations (new for 2026)
    HAS_ATTRIBUTE = "has_attribute"
    CHARACTERIZED_BY = "characterized_by"
    EXEMPLIFIES = "exemplifies"
    ILLUSTRATES = "illustrates"
    DEMONSTRATES = "demonstrates"

    # Temporal sequence relations (expanding temporal modeling)
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONCURRENT_WITH = "concurrent_with"
    OVERLAPS_WITH = "overlaps_with"

    # Causal relation expansions
    PREVENTS = "prevents"
    ENABLES = "enables"
    CONTRIBUTES_TO = "contributes_to"
    MITIGATES = "mitigates"
    EXACERBATES = "exacerbates"

    # Spatial relation expansions (more precise spatial modeling)
    ADJACENT_TO = "adjacent_to"
    NEAR = "near"
    WITHIN = "within"
    SURROUNDS = "surrounds"
    CONNECTS_TO = "connects_to"

    # Quantitative relations (for numeric reasoning)
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL_TO = "equal_to"
    APPROXIMATELY = "approximately"
    PROPORTIONAL_TO = "proportional_to"

    # Membership and categorization
    MEMBER_OF = "member_of"
    AFFILIATED_WITH = "affiliated_with"
    ASSOCIATED_WITH = "associated_with"

    # Process and transformation
    TRANSFORMS_INTO = "transforms_into"
    DERIVES_FROM = "derives_from"
    EVOLVES_INTO = "evolves_into"
    PRODUCES = "produces"
    CONSUMES = "consumes"

    # Future planning and prediction
    PLANNED_FOR = "planned_for"
    EXPECTED_TO = "expected_to"
    INTENDS_TO = "intends_to"

    # Negations and opposites (for handling contradictory information)
    NOT_RELATED_TO = "not_related_to"
    CONTRADICTS = "contradicts"
    OPPOSES = "opposes"

    # Uncertainty and qualification
    MAY_RELATE_TO = "may_relate_to"

    # Complex relations for knowledge graphs
    SAME_AS = "same_as"
    SIMILAR_TO = "similar_to"
    EQUIVALENT_TO = "equivalent_to"
    ANALOGOUS_TO = "analogous_to"
    RELATED_VIA = "related_via"


class ReviewStatus(str, Enum):
    """Review queue status."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


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
    community_reports_generated = Column(Integer, nullable=False, default=0)
    total_communities = Column(Integer, nullable=False, default=0)

    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    review_items = relationship("ReviewQueue", back_populates="document")


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
    # Using dimension 1024 (perfect for bge-m3 exact match)
    # Works with bge-m3 (native 1024), smaller models use padding
    # OpenAI models would need truncation (1536->1024, 3072->1024)
    embedding = Column(VECTOR(1024), nullable=True)

    document = relationship("Document", back_populates="chunks")
    entities = relationship(
        "Entity", secondary="chunk_entities", back_populates="chunks"
    )

    __table_args__ = (
        Index("idx_chunk_document_index", "document_id", "chunk_index"),
        Index(
            "idx_chunk_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


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
    # Using dimension 1024 (perfect for bge-m3 exact match)
    # Works with bge-m3 (native 1024), smaller models use padding
    # OpenAI models would need truncation (1536->1024, 3072->1024)
    embedding = Column(VECTOR(1024), nullable=True)
    uri = Column(
        String(500), nullable=True, unique=True
    )  # Unique identifier following RDF patterns
    source_text = Column(Text, nullable=True)  # For provenance tracking
    domain = Column(String(100), nullable=True, index=True)
    community_reports_generated = Column(Integer, nullable=False, default=0)
    total_communities = Column(Integer, nullable=False, default=0)

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
    review_items = relationship("ReviewQueue", back_populates="entity")

    __table_args__ = (
        Index(
            "idx_entity_embedding_ivfflat",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


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
    community_reports_generated = Column(Integer, nullable=False, default=0)
    total_communities = Column(Integer, nullable=False, default=0)

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
    review_items = relationship("ReviewQueue", back_populates="edge")

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
    parent = relationship(
        "Community",
        remote_side=[id],
        back_populates="children",
    )
    children = relationship("Community", back_populates="parent")


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

    entity_id: UUID
    confidence_score: float
    merged_entity_ids: list[UUID] = Field(default_factory=list)
    human_review_required: bool = False
    grounding_quote: Optional[str] = None


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
    entity = relationship("Entity", back_populates="review_items")
    edge = relationship("Edge", back_populates="review_items")
    document = relationship("Document", back_populates="review_items")


class TemporalEntityClaim(BaseModel):
    """Temporal entity claim."""

    entity_id: UUID = Field(..., description="Related entity ID")
    chunk_id: UUID = Field(..., description="Source chunk ID")
    claim_text: str = Field(..., description="Claim text")
    temporal_type: TemporalType = Field(..., description="Temporal classification")
    iso8601_date: str | None = Field(None, description="Normalized ISO-8601 date")
    invalidated_by_id: UUID | None = Field(None, description="ID of invalidating claim")

"""Experience Bank for KBv2.

Stores high-quality extraction examples and retrieves them for few-shot prompting.
This enables self-improvement by reusing successful extraction patterns.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as SQLUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

Base = declarative_base()


class ExtractionExperienceRecord(Base):
    """Database record for extraction experiences."""

    __tablename__ = "extraction_experiences"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    text_snippet = Column(Text, nullable=False)
    text_embedding_id = Column(String, nullable=True, index=True)

    # Extraction results
    entities = Column(JSONB, default=list)
    relationships = Column(JSONB, default=list)
    extraction_patterns = Column(JSONB, default=dict)

    # Metadata
    domain = Column(String, nullable=False, index=True)
    entity_types = Column(JSONB, default=list)
    quality_score = Column(Float, nullable=False, index=True)
    extraction_method = Column(String, nullable=True)

    # Source tracking
    document_id = Column(SQLUUID(as_uuid=True), nullable=True)
    chunk_id = Column(SQLUUID(as_uuid=True), nullable=True)

    # Usage tracking
    retrieval_count = Column(Integer, default=0)
    last_retrieved_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_experiences_domain_quality", "domain", "quality_score"),
        Index("ix_experiences_entity_types", "entity_types", postgresql_using="gin"),
    )


class ExtractionExample(BaseModel):
    """An extraction example for few-shot prompting."""

    id: UUID = Field(default_factory=uuid4)
    text_snippet: str = Field(..., description="Source text")
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    extraction_patterns: Dict[str, Any] = Field(default_factory=dict)
    domain: str = Field(default="GENERAL")
    quality_score: float = Field(default=0.0)
    entity_types: List[str] = Field(default_factory=list)

    def to_few_shot_format(self) -> str:
        """Convert to few-shot prompt format."""
        example_text = f"Text: {self.text_snippet[:500]}\n\n"
        example_text += "Extracted Entities:\n"

        for entity in self.entities[:10]:
            name = entity.get("name", "Unknown")
            entity_type = entity.get("entity_type", "Unknown")
            desc = entity.get("description", "")
            example_text += f"- {name} ({entity_type}): {desc[:100]}\n"

        if self.relationships:
            example_text += "\nRelationships:\n"
            for rel in self.relationships[:5]:
                source = rel.get("source", "Unknown")
                target = rel.get("target", "Unknown")
                rel_type = rel.get("relationship_type", "RELATED_TO")
                example_text += f"- {source} --[{rel_type}]--> {target}\n"

        return example_text


class ExperienceBankConfig(BaseModel):
    """Configuration for Experience Bank."""

    min_quality_threshold: float = 0.85
    max_storage_size: int = 10000
    similarity_top_k: int = 3
    max_text_length: int = 2000
    enable_pattern_extraction: bool = True


class ExperienceBank:
    """Bank of high-quality extraction experiences."""

    def __init__(
        self, session, vector_store=None, config: Optional[ExperienceBankConfig] = None
    ):
        self.session = session
        self.vector_store = vector_store
        self.config = config or ExperienceBankConfig()

    async def store_experience(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        quality_score: float,
        domain: str = "GENERAL",
        document_id: Optional[UUID] = None,
        chunk_id: Optional[UUID] = None,
        extraction_method: Optional[str] = None,
    ) -> Optional[UUID]:
        """Store an extraction experience if quality is high enough."""
        if quality_score < self.config.min_quality_threshold:
            return None

        patterns = {}
        if self.config.enable_pattern_extraction:
            patterns = self._extract_patterns(entities, relationships)

        entity_types = list(
            set(
                e.get("entity_type", "Unknown")
                for e in entities
                if e.get("entity_type")
            )
        )

        experience = ExtractionExperienceRecord(
            id=uuid4(),
            text_snippet=text[: self.config.max_text_length],
            entities=entities,
            relationships=relationships,
            extraction_patterns=patterns,
            domain=domain,
            entity_types=entity_types,
            quality_score=quality_score,
            document_id=document_id,
            chunk_id=chunk_id,
            extraction_method=extraction_method,
            retrieval_count=0,
            created_at=datetime.utcnow(),
        )

        self.session.add(experience)
        await self.session.commit()

        return experience.id

    async def retrieve_similar_examples(
        self,
        text: str,
        domain: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        k: int = 3,
        min_quality: float = 0.85,
    ) -> List[ExtractionExample]:
        """Retrieve similar extraction examples."""
        query = select(ExtractionExperienceRecord).where(
            ExtractionExperienceRecord.quality_score >= min_quality
        )

        if domain:
            query = query.where(ExtractionExperienceRecord.domain == domain)

        if entity_types:
            query = query.where(
                ExtractionExperienceRecord.entity_types.overlap(entity_types)
            )

        query = query.order_by(ExtractionExperienceRecord.quality_score.desc()).limit(k)

        result = await self.session.execute(query)
        records = result.scalars().all()

        return [self._record_to_example(r) for r in records]

    async def get_experiences_by_domain(
        self, domain: str, min_quality: float = 0.85, limit: int = 100
    ) -> List[ExtractionExample]:
        """Get all experiences for a domain."""
        query = (
            select(ExtractionExperienceRecord)
            .where(
                ExtractionExperienceRecord.domain == domain,
                ExtractionExperienceRecord.quality_score >= min_quality,
            )
            .order_by(ExtractionExperienceRecord.quality_score.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        records = result.scalars().all()

        return [self._record_to_example(r) for r in records]

    async def get_experience_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences."""
        query = select(ExtractionExperienceRecord)
        result = await self.session.execute(query)
        records = result.scalars().all()

        total_count = len(records)

        domain_stats = {}
        for record in records:
            domain = record.domain
            if domain not in domain_stats:
                domain_stats[domain] = {"count": 0, "total_quality": 0.0}
            domain_stats[domain]["count"] += 1
            domain_stats[domain]["total_quality"] += record.quality_score

        for domain in domain_stats:
            count = domain_stats[domain]["count"]
            total = domain_stats[domain]["total_quality"]
            domain_stats[domain]["avg_quality"] = total / count if count > 0 else 0.0
            del domain_stats[domain]["total_quality"]

        return {
            "total_experiences": total_count,
            "domain_distribution": domain_stats,
            "config": self.config.dict(),
        }

    def _extract_patterns(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract patterns from entities and relationships."""
        patterns = {
            "entity_type_distribution": {},
            "relationship_type_distribution": {},
            "entity_count": len(entities),
            "relationship_count": len(relationships),
        }

        for entity in entities:
            entity_type = entity.get("entity_type", "Unknown")
            patterns["entity_type_distribution"][entity_type] = (
                patterns["entity_type_distribution"].get(entity_type, 0) + 1
            )

        for rel in relationships:
            rel_type = rel.get("relationship_type", "RELATED_TO")
            patterns["relationship_type_distribution"][rel_type] = (
                patterns["relationship_type_distribution"].get(rel_type, 0) + 1
            )

        return patterns

    def _record_to_example(
        self, record: ExtractionExperienceRecord
    ) -> ExtractionExample:
        """Convert database record to example."""
        return ExtractionExample(
            id=record.id,
            text_snippet=record.text_snippet,
            entities=record.entities or [],
            relationships=record.relationships or [],
            extraction_patterns=record.extraction_patterns or {},
            domain=record.domain,
            quality_score=record.quality_score,
            entity_types=record.entity_types or [],
        )


class ExperienceBankMiddleware:
    """Middleware to integrate Experience Bank with extraction pipeline."""

    def __init__(self, experience_bank: ExperienceBank):
        self.experience_bank = experience_bank

    async def enrich_prompt_with_examples(
        self, base_prompt: str, text: str, domain: str = "GENERAL", k: int = 3
    ) -> str:
        """Enrich a prompt with few-shot examples."""
        examples = await self.experience_bank.retrieve_similar_examples(
            text=text, domain=domain, k=k
        )

        if not examples:
            return base_prompt

        few_shot_section = "\n\n## Examples of High-Quality Extractions\n\n"

        for i, example in enumerate(examples, 1):
            few_shot_section += (
                f"### Example {i} (Quality: {example.quality_score:.2f})\n"
            )
            few_shot_section += example.to_few_shot_format()
            few_shot_section += "\n---\n\n"

        return base_prompt + few_shot_section

    async def store_successful_extraction(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        quality_score: float,
        domain: str = "GENERAL",
        **kwargs,
    ) -> Optional[UUID]:
        """Store a successful extraction for future reuse."""
        return await self.experience_bank.store_experience(
            text=text,
            entities=entities,
            relationships=relationships,
            quality_score=quality_score,
            domain=domain,
            **kwargs,
        )

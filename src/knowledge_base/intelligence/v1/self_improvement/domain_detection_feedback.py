"""Domain Detection Self-Improvement System.

Tracks domain detection accuracy, collects feedback, and improves classification over time.
Uses a feedback loop where:
1. Detection is made with confidence score
2. Quality of downstream extraction is measured
3. Low quality = possible misclassification
4. System learns which indicators work best per domain
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    select,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as SQLUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

Base = declarative_base()


class DomainDetectionFeedbackRecord(Base):
    """Database record for domain detection feedback."""

    __tablename__ = "domain_detection_feedback"

    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Detection details
    document_id = Column(SQLUUID(as_uuid=True), nullable=False, index=True)
    detected_domain = Column(String, nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    detection_method = Column(String, nullable=True)  # 'llm', 'keyword', 'fallback'

    # Content indicators found
    crypto_indicators = Column(JSONB, default=list)
    domain_scores = Column(JSONB, default=dict)

    # User feedback (if provided)
    user_correction = Column(String, nullable=True)
    feedback_source = Column(
        String, nullable=True
    )  # 'user', 'quality_correlation', 'manual'

    # Quality correlation (measured after extraction)
    extraction_quality = Column(Float, nullable=True)
    entity_count = Column(Integer, nullable=True)
    was_accurate = Column(String, nullable=True)  # 'yes', 'no', 'unknown'

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_feedback_domain_accuracy", "detected_domain", "was_accurate"),
        Index("ix_feedback_confidence", "confidence"),
    )


@dataclass
class DomainAccuracyStats:
    """Statistics for domain detection accuracy."""

    domain: str
    total_classifications: int
    confirmed_accurate: int
    confirmed_inaccurate: int
    unknown_accuracy: int
    average_confidence: float
    average_extraction_quality: float
    accuracy_rate: (
        float  # confirmed_accurate / (confirmed_accurate + confirmed_inaccurate)
    )


@dataclass
class DomainImprovementSuggestion:
    """Suggestion for improving domain detection."""

    domain: str
    issue: str
    suggestion: str
    confidence: float
    examples: List[str] = field(default_factory=list)


class DomainDetectionSelfImprovement:
    """Self-improvement system for domain detection.

    Learns from:
    - User corrections
    - Extraction quality correlation
    - Confidence vs accuracy relationship
    """

    # Quality threshold below which we suspect wrong domain
    QUALITY_SUSPICION_THRESHOLD = 0.6

    # Minimum samples needed for stats
    MIN_SAMPLES = 5

    def __init__(self, session: AsyncSession):
        self.session = session
        self._cache: Dict[str, Any] = {}

    async def record_detection(
        self,
        document_id: UUID,
        detected_domain: str,
        confidence: float,
        detection_method: str,
        crypto_indicators: List[str],
        domain_scores: Dict[str, float],
        extraction_quality: Optional[float] = None,
        entity_count: Optional[int] = None,
        user_correction: Optional[str] = None,
    ) -> UUID:
        """Record a domain detection for learning.

        Args:
            document_id: Document ID
            detected_domain: Domain that was detected
            confidence: Detection confidence (0.0-1.0)
            detection_method: How detection was made ('llm', 'keyword', 'fallback')
            crypto_indicators: Crypto keywords found in content
            domain_scores: Scores for each domain
            extraction_quality: Quality of downstream extraction (if available)
            entity_count: Number of entities extracted
            user_correction: User-provided correct domain (if any)

        Returns:
            Feedback record ID
        """
        # Determine accuracy based on available signals
        was_accurate = "unknown"

        if user_correction:
            # User provided explicit feedback
            was_accurate = "yes" if user_correction == detected_domain else "no"
            feedback_source = "user"
        elif extraction_quality is not None:
            # Infer accuracy from extraction quality
            if extraction_quality >= 0.8:
                was_accurate = "yes"
            elif extraction_quality <= 0.4:
                was_accurate = "no"
            feedback_source = "quality_correlation"
        else:
            feedback_source = "auto"

        record = DomainDetectionFeedbackRecord(
            id=uuid4(),
            document_id=document_id,
            detected_domain=detected_domain,
            confidence=confidence,
            detection_method=detection_method,
            crypto_indicators=crypto_indicators,
            domain_scores=domain_scores,
            user_correction=user_correction,
            feedback_source=feedback_source,
            extraction_quality=extraction_quality,
            entity_count=entity_count,
            was_accurate=was_accurate,
        )

        self.session.add(record)
        await self.session.commit()

        logger.info(
            f"Recorded domain detection feedback: {detected_domain} "
            f"(confidence: {confidence:.2f}, accuracy: {was_accurate})"
        )

        return record.id

    async def get_domain_accuracy_stats(
        self,
        domain: Optional[str] = None,
        days: int = 30,
    ) -> List[DomainAccuracyStats]:
        """Get accuracy statistics for domains.

        Args:
            domain: Specific domain or None for all
            days: Lookback period in days

        Returns:
            List of accuracy statistics
        """
        since = datetime.utcnow() - timedelta(days=days)

        query = (
            select(
                DomainDetectionFeedbackRecord.detected_domain,
                func.count().label("total"),
                func.sum(
                    func.case(
                        (DomainDetectionFeedbackRecord.was_accurate == "yes", 1),
                        else_=0,
                    )
                ).label("accurate"),
                func.sum(
                    func.case(
                        (DomainDetectionFeedbackRecord.was_accurate == "no", 1), else_=0
                    )
                ).label("inaccurate"),
                func.avg(DomainDetectionFeedbackRecord.confidence).label(
                    "avg_confidence"
                ),
                func.avg(DomainDetectionFeedbackRecord.extraction_quality).label(
                    "avg_quality"
                ),
            )
            .where(DomainDetectionFeedbackRecord.created_at >= since)
            .group_by(DomainDetectionFeedbackRecord.detected_domain)
        )

        if domain:
            query = query.where(DomainDetectionFeedbackRecord.detected_domain == domain)

        result = await self.session.execute(query)

        stats = []
        for row in result:
            accurate = row.accurate or 0
            inaccurate = row.inaccurate or 0
            total_with_feedback = accurate + inaccurate

            accuracy_rate = (
                accurate / total_with_feedback if total_with_feedback > 0 else 0.0
            )

            stats.append(
                DomainAccuracyStats(
                    domain=row.detected_domain,
                    total_classifications=row.total,
                    confirmed_accurate=accurate,
                    confirmed_inaccurate=inaccurate,
                    unknown_accuracy=row.total - total_with_feedback,
                    average_confidence=row.avg_confidence or 0.0,
                    average_extraction_quality=row.avg_quality or 0.0,
                    accuracy_rate=accuracy_rate,
                )
            )

        return stats

    async def get_improvement_suggestions(
        self,
        min_samples: int = 10,
    ) -> List[DomainImprovementSuggestion]:
        """Generate improvement suggestions based on feedback.

        Args:
            min_samples: Minimum samples needed for suggestion

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Get accuracy stats for all domains
        stats_list = await self.get_domain_accuracy_stats(days=90)

        for stats in stats_list:
            if stats.total_classifications < min_samples:
                continue

            # Issue: Low accuracy rate
            if stats.accuracy_rate < 0.7:
                suggestions.append(
                    DomainImprovementSuggestion(
                        domain=stats.domain,
                        issue=f"Low accuracy rate: {stats.accuracy_rate:.1%}",
                        suggestion="Review domain keywords and LLM prompt. Consider splitting into sub-domains.",
                        confidence=1 - stats.accuracy_rate,
                        examples=[],
                    )
                )

            # Issue: Low average extraction quality
            if stats.average_extraction_quality < 0.6:
                suggestions.append(
                    DomainImprovementSuggestion(
                        domain=stats.domain,
                        issue=f"Low extraction quality: {stats.average_extraction_quality:.2f}",
                        suggestion="Domain definition may be too broad. Downstream extraction struggles.",
                        confidence=0.6 - stats.average_extraction_quality,
                        examples=[],
                    )
                )

            # Issue: High confidence but low accuracy
            if stats.average_confidence > 0.8 and stats.accuracy_rate < 0.6:
                suggestions.append(
                    DomainImprovementSuggestion(
                        domain=stats.domain,
                        issue="Overconfident but inaccurate",
                        suggestion="Adjust confidence threshold or improve detection criteria.",
                        confidence=stats.average_confidence - stats.accuracy_rate,
                        examples=[],
                    )
                )

        # Sort by confidence (most important first)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)

        return suggestions

    async def get_suspicious_classifications(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent classifications that may be wrong.

        Args:
            limit: Maximum number to return

        Returns:
            List of suspicious classifications
        """
        # Find high-confidence detections with low extraction quality
        query = (
            select(DomainDetectionFeedbackRecord)
            .where(
                DomainDetectionFeedbackRecord.confidence >= 0.7,
                DomainDetectionFeedbackRecord.extraction_quality <= 0.5,
            )
            .order_by(DomainDetectionFeedbackRecord.created_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        records = result.scalars().all()

        suspicious = []
        for record in records:
            suspicious.append(
                {
                    "id": str(record.id),
                    "document_id": str(record.document_id),
                    "detected_domain": record.detected_domain,
                    "confidence": record.confidence,
                    "extraction_quality": record.extraction_quality,
                    "created_at": record.created_at.isoformat(),
                }
            )

        return suspicious

    async def should_suspect_domain(
        self,
        detected_domain: str,
        confidence: float,
        extraction_quality: Optional[float] = None,
    ) -> bool:
        """Check if a domain detection should be suspected as wrong.

        Args:
            detected_domain: The detected domain
            confidence: Detection confidence
            extraction_quality: Quality of extraction (if available)

        Returns:
            True if domain should be re-evaluated
        """
        # High confidence with very low extraction quality is suspicious
        if confidence >= 0.8 and extraction_quality is not None:
            if extraction_quality <= 0.3:
                return True

        # Check historical accuracy for this domain
        stats_list = await self.get_domain_accuracy_stats(
            domain=detected_domain, days=60
        )
        if stats_list:
            stats = stats_list[0]
            if stats.total_classifications >= self.MIN_SAMPLES:
                # If domain historically has low accuracy, be suspicious
                if stats.accuracy_rate < 0.5:
                    return True

        return False

    async def get_alternative_domains(
        self,
        content_sample: str,
        current_domain: str,
        top_k: int = 3,
    ) -> List[tuple]:
        """Get alternative domain suggestions based on similar past detections.

        Args:
            content_sample: Sample of content to compare
            current_domain: Currently detected domain
            top_k: Number of alternatives to suggest

        Returns:
            List of (domain, confidence) tuples
        """
        # Get past misclassifications that were corrected
        query = (
            select(
                DomainDetectionFeedbackRecord.user_correction,
                func.count().label("count"),
            )
            .where(
                DomainDetectionFeedbackRecord.detected_domain == current_domain,
                DomainDetectionFeedbackRecord.was_accurate == "no",
                DomainDetectionFeedbackRecord.user_correction.isnot(None),
            )
            .group_by(DomainDetectionFeedbackRecord.user_correction)
            .order_by(func.count().desc())
            .limit(top_k)
        )

        result = await self.session.execute(query)

        alternatives = []
        for row in result:
            if row.user_correction != current_domain:
                confidence = min(row.count / 5, 0.9)  # Cap at 0.9
                alternatives.append((row.user_correction, confidence))

        return alternatives

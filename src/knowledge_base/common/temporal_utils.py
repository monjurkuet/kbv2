"""Temporal Information Extraction (TIE) utilities."""

from datetime import datetime, timezone
from enum import Enum

import dateparser
from pydantic import BaseModel, Field


class TemporalType(str, Enum):
    """Temporal claim classification."""

    ATEMPORAL = "atemporal"
    STATIC = "static"
    DYNAMIC = "dynamic"


class TemporalClaim(BaseModel):
    """Temporal information claim."""

    text: str = Field(..., description="Claim text")
    temporal_type: TemporalType = Field(..., description="Temporal classification")
    start_date: datetime | None = Field(None, description="Valid from date")
    end_date: datetime | None = Field(None, description="Valid until date")
    iso8601_date: str | None = Field(None, description="Normalized ISO-8601 date")
    confidence: float = Field(default=1.0, description="Confidence score")


class TemporalNormalizer:
    """Normalizes temporal expressions to ISO-8601 format."""

    def __init__(self, reference_date: datetime | None = None) -> None:
        """Initialize temporal normalizer.

        Args:
            reference_date: Reference date for relative expressions.
                If None, uses current UTC time.
        """
        self._reference_date = reference_date or datetime.now(timezone.utc)

    def normalize_relative_date(
        self,
        text: str,
        reference_date: datetime | None = None,
    ) -> datetime | None:
        """Normalize relative date expression to absolute datetime.

        Args:
            text: Text containing relative date expression.
            reference_date: Reference date. If None, uses instance default.

        Returns:
            Normalized datetime or None if no date found.
        """
        ref_date = reference_date or self._reference_date

        parsed = dateparser.parse(
            text,
            settings={
                "RELATIVE_BASE": ref_date,
                "TIMEZONE": "UTC",
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )

        return parsed

    def normalize_iso8601(self, date_obj: datetime) -> str:
        """Convert datetime to ISO-8601 string.

        Args:
            date_obj: Datetime object.

        Returns:
            ISO-8601 string.
        """
        if date_obj.tzinfo is None:
            date_obj = date_obj.replace(tzinfo=timezone.utc)
        return date_obj.isoformat()

    def classify_claim_temporal_type(
        self,
        claim_text: str,
        context: str | None = None,
    ) -> TemporalType:
        """Classify claim temporal type.

        Args:
            claim_text: The claim text to classify.
            context: Optional surrounding context.

        Returns:
            Temporal type classification.
        """
        claim_lower = claim_text.lower()

        atemporal_patterns = [
            "is a",
            "are a",
            "is defined as",
            "refers to",
            "means",
            "definition",
        ]

        static_patterns = [
            "was born",
            "died",
            "founded",
            "established",
            "created",
            "invented",
        ]

        dynamic_patterns = [
            "currently",
            "now",
            "recently",
            "as of",
            "plans to",
            "will",
            "going to",
            "expects to",
        ]

        # Check if claim contains date-related patterns
        date_patterns = [
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\b\d{4}\b",  # Just a year
        ]

        import re

        has_date = any(re.search(pattern, claim_lower) for pattern in date_patterns)

        # If there's a date, it's either STATIC or DYNAMIC depending on context
        if has_date:
            # Check for dynamic indicators
            for pattern in dynamic_patterns:
                if pattern in claim_lower:
                    return TemporalType.DYNAMIC

            # Default to STATIC for dated events
            return TemporalType.STATIC

        for pattern in atemporal_patterns:
            if pattern in claim_lower:
                return TemporalType.ATEMPORAL

        for pattern in dynamic_patterns:
            if pattern in claim_lower:
                return TemporalType.DYNAMIC

        for pattern in static_patterns:
            if pattern in claim_lower:
                return TemporalType.STATIC

        if any(
            word in claim_lower
            for word in [
                "yesterday",
                "today",
                "tomorrow",
                "last week",
                "next month",
                "ago",
            ]
        ):
            return TemporalType.DYNAMIC

        return TemporalType.ATEMPORAL

    def extract_temporal_info(
        self,
        text: str,
        reference_date: datetime | None = None,
    ) -> TemporalClaim:
        """Extract and normalize temporal information from text.

        Args:
            text: Text containing temporal information.
            reference_date: Reference date for relative expressions.

        Returns:
            Temporal claim with normalized dates.
        """
        temporal_type = self.classify_claim_temporal_type(text)

        normalized_date = None
        iso_date = None

        if temporal_type in (TemporalType.STATIC, TemporalType.DYNAMIC):
            # Try to extract date from the text
            # First try to parse the full text
            normalized_date = self.normalize_relative_date(text, reference_date)

            # If that fails, try to extract a date pattern from the text
            if not normalized_date:
                import re

                # Look for date patterns at the beginning of the text
                # Matches patterns like "August 2021", "May 2023", "Aug 1, 2023"
                date_pattern = r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
                match = re.search(date_pattern, text, re.IGNORECASE)
                if match:
                    date_text = match.group(0)
                    normalized_date = self.normalize_relative_date(
                        date_text, reference_date
                    )

            # Try another pattern for dates like "Aug 1, 2023"
            if not normalized_date:
                date_pattern = r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}"
                match = re.search(date_pattern, text, re.IGNORECASE)
                if match:
                    date_text = match.group(0)
                    normalized_date = self.normalize_relative_date(
                        date_text, reference_date
                    )

            # Try pattern for "September 15, 2024"
            if not normalized_date:
                date_pattern = r"^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"
                match = re.search(date_pattern, text, re.IGNORECASE)
                if match:
                    date_text = match.group(0)
                    normalized_date = self.normalize_relative_date(
                        date_text, reference_date
                    )

            if normalized_date:
                iso_date = self.normalize_iso8601(normalized_date)

        return TemporalClaim(
            text=text,
            temporal_type=temporal_type,
            start_date=normalized_date,
            end_date=None,
            iso8601_date=iso_date,
        )

    def check_invalidated(
        self,
        old_claim: TemporalClaim,
        new_claim: TemporalClaim,
    ) -> bool:
        """Check if new claim invalidates old claim.

        Args:
            old_claim: Existing temporal claim.
            new_claim: New temporal claim.

        Returns:
            True if new claim invalidates old claim.
        """
        if old_claim.temporal_type != TemporalType.DYNAMIC:
            return False

        if new_claim.temporal_type != TemporalType.DYNAMIC:
            return False

        if not old_claim.start_date or not new_claim.start_date:
            return False

        if new_claim.start_date > old_claim.start_date:
            return True

        return False

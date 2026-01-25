"""
Service for calculating absolute text offsets for entity grounding.

This module provides the critical "Offset Service" logic from the high-level guide,
converting grounding_quote + chunk_text → absolute integer offsets for frontend
text highlighting. Implements W3C Web Annotation Data Model compliance.

Core Algorithm:
1. Reconstruct text stream from chunks (maintaining global offset)
2. Find grounding_quote within chunk.text using exact string matching
3. Calculate absolute start/end positions for highlighting
4. Generate W3C-compliant TextPositionSelector and TextQuoteSelector
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID

from knowledge_base.persistence.v1.schema import Chunk, ChunkEntity, Entity


logger = logging.getLogger(__name__)


class TextPositionSelector:
    """W3C TextPositionSelector for exact character offsets."""

    def __init__(self, start: int, end: int):
        """
        Initialize TextPositionSelector.

        Args:
            start: Absolute start character offset (0-indexed)
            end: Absolute end character offset (exclusive)
        """
        self.type = "TextPositionSelector"
        self.start = start
        self.end = end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"type": self.type, "start": self.start, "end": self.end}


class TextQuoteSelector:
    """W3C TextQuoteSelector for fuzzy matching with context."""

    def __init__(
        self, exact: str, prefix: Optional[str] = None, suffix: Optional[str] = None
    ):
        """
        Initialize TextQuoteSelector.

        Args:
            exact: Exact text to match
            prefix: Text before the exact match (for context)
            suffix: Text after the exact match (for context)
        """
        self.type = "TextQuoteSelector"
        self.exact = exact
        self.prefix = prefix
        self.suffix = suffix

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"type": self.type, "exact": self.exact}
        if self.prefix:
            result["prefix"] = self.prefix
        if self.suffix:
            result["suffix"] = self.suffix
        return result


class TextSpan:
    """
    Text span with entity grounding and W3C annotation selectors.

    Represents a specific occurrence of an entity in a document with
    absolute character offsets and W3C-compliant selectors for reliable
    text highlighting in the frontend.
    """

    def __init__(
        self,
        start_offset: int,
        end_offset: int,
        text: str,
        entity_id: Optional[UUID] = None,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        confidence: float = 1.0,
        grounding_quote: Optional[str] = None,
        selectors: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize TextSpan.

        Args:
            start_offset: Absolute start position in document (0-indexed)
            end_offset: Absolute end position in document (exclusive)
            text: Extracted text snippet
            entity_id: Linked entity UUID
            entity_name: Entity name for display
            entity_type: Entity type (Person, Organization, etc.)
            confidence: Extraction confidence score
            grounding_quote: Original verbatim quote from document
            selectors: W3C annotation selectors
        """
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.text = text
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.confidence = confidence
        self.grounding_quote = grounding_quote
        self.selectors = selectors or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "text": self.text,
            "entity_id": str(self.entity_id) if self.entity_id else None,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "grounding_quote": self.grounding_quote,
            "selectors": self.selectors,
        }


class OffsetCalculationService:
    """
    Service for calculating absolute text offsets from grounding quotes.

    Implements the core algorithm from the high-level logic guide:
    1. Reconstruct text stream from chunks in order
    2. Find grounding_quote within chunk.text
    3. Calculate absolute offsets relative to document start
    4. Generate W3C-compliant selectors
    """

    @staticmethod
    def calculate_absolute_offsets(
        chunks: List[Chunk], chunk_entities: List[Tuple[ChunkEntity, Entity]]
    ) -> List[TextSpan]:
        """
        Calculate absolute offsets for entity grounding.

        This is the critical function that converts database-stored grounding_quote
        strings into integer offsets needed for frontend text highlighting.

        Args:
            chunks: Ordered list of document chunks (sorted by chunk_index)
            chunk_entities: List of (chunk_entity, entity) tuples with grounding data

        Returns:
            List of TextSpan objects with absolute offsets, sorted by position
        """
        # Sort chunks by index to ensure correct order
        sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)

        text_spans = []
        global_offset = 0  # Running offset from document start

        # Create mapping of chunk ID to chunk for efficient lookup
        chunk_by_id = {chunk.id: chunk for chunk in sorted_chunks}

        # Process each chunk_entity
        for chunk_entity, entity in chunk_entities:
            chunk = chunk_by_id.get(chunk_entity.chunk_id)

            if not chunk or not chunk.text:
                logger.warning(
                    f"Chunk not found or empty for chunk_entity {chunk_entity.id}"
                )
                continue

            grounding_quote = chunk_entity.grounding_quote

            if not grounding_quote:
                logger.warning(f"No grounding_quote for chunk_entity {chunk_entity.id}")
                continue

            # Find the quote within the chunk text using exact string matching
            local_start = chunk.text.find(grounding_quote)

            if local_start == -1:
                # Quote not found - possible data drift, encoding issues, or OCR errors
                logger.warning(
                    f"Grounding quote not found in chunk text. "
                    f"Chunk ID: {chunk.id}, "
                    f"Quote: '{grounding_quote[:50]}...', "
                    f"Chunk text length: {len(chunk.text)}"
                )

                # Try fuzzy matching as fallback
                fuzzy_start = _fuzzy_find(grounding_quote, chunk.text)
                if fuzzy_start is not None:
                    logger.info(f"Fuzzy match found at position {fuzzy_start}")
                    local_start = fuzzy_start
                else:
                    # Skip this span if no match found
                    continue

            # Calculate absolute offsets
            absolute_start = global_offset + local_start
            absolute_end = absolute_start + len(grounding_quote)

            # Extract context for TextQuoteSelector
            prefix = _extract_context(chunk.text, local_start, 50, before=True)
            suffix = _extract_context(
                chunk.text, local_start + len(grounding_quote), 50, before=False
            )

            # Create selectors
            position_selector = TextPositionSelector(absolute_start, absolute_end)
            quote_selector = TextQuoteSelector(
                exact=grounding_quote, prefix=prefix, suffix=suffix
            )

            # Create TextSpan
            text_span = TextSpan(
                start_offset=absolute_start,
                end_offset=absolute_end,
                text=chunk.text[local_start : local_start + len(grounding_quote)],
                entity_id=entity.id if entity else None,
                entity_name=entity.name if entity else None,
                entity_type=entity.entity_type if entity else None,
                confidence=chunk_entity.confidence_score or 0.0,
                grounding_quote=grounding_quote,
                selectors=[position_selector.to_dict(), quote_selector.to_dict()],
            )

            text_spans.append(text_span)

        # Sort spans by start_offset for sequential highlighting
        text_spans.sort(key=lambda span: span.start_offset)

        return text_spans

    @staticmethod
    def calculate_offsets_for_document(
        db_session: Any,
        document_id: UUID,
        confidence_threshold: float = 0.0,
        entity_types: Optional[List[str]] = None,
    ) -> List[TextSpan]:
        """
        Calculate offsets for all entities in a specific document.

        High-level method that fetches required data and calculates offsets.

        Args:
            db_session: Database session
            document_id: Document UUID
            confidence_threshold: Minimum confidence to include
            entity_types: Optional filter for entity types

        Returns:
            List of TextSpan objects for the document
        """
        # This would be implemented with actual ORM queries
        # For now, it's a placeholder showing the API contract
        pass


def _extract_context(
    text: str, position: int, length: int, before: bool = True
) -> Optional[str]:
    """
    Extract context text around a position.

    Args:
        text: Full text to extract from
        position: Character position
        length: Number of characters to extract
        before: If True, extract characters before position, else after

    Returns:
        Context string or None if out of bounds
    """
    if before:
        start = max(0, position - length)
        end = position
    else:
        start = position
        end = min(len(text), position + length)

    if start >= end or start >= len(text):
        return None

    return text[start:end]


def _fuzzy_find(query: str, text: str, max_distance: int = 5) -> Optional[int]:
    """
    Fuzzy string matching for grounding quotes.

    Finds approximate matches when exact matching fails due to minor differences
    (whitespace, punctuation, encoding issues).

    Args:
        query: String to find (grounding_quote)
        text: Text to search in (chunk.text)
        max_distance: Maximum Levenshtein distance

    Returns:
        Starting position of best match or None
    """
    # This is a simple fuzzy matcher - production would use difflib or fuzzywuzzy
    try:
        import difflib

        # Try to find close matches
        matcher = difflib.SequenceMatcher(None, query, text)
        matches = matcher.get_matching_blocks()

        if matches and matches[0].size > len(query) * 0.8:  # 80% match threshold
            return matches[0].a

    except Exception as e:
        logger.warning(f"Fuzzy matching failed for query '{query[:30]}...': {e}")

    return None


class HighlightService:
    """
    Service for generating text highlights from calculated spans.

    Handles merging overlapping spans and creating non-overlapping
    highlight segments for the frontend text viewer.
    """

    @staticmethod
    def merge_overlapping_spans(spans: List[TextSpan]) -> List[TextSpan]:
        """
        Merge overlapping text spans to prevent duplicate highlighting.

        Args:
            spans: List of TextSpan objects

        Returns:
            List of merged spans without overlaps
        """
        if not spans:
            return []

        # Sort by start_offset, then by length (longest first)
        sorted_spans = sorted(
            spans, key=lambda s: (s.start_offset, -(s.end_offset - s.start_offset))
        )

        merged = []
        current = sorted_spans[0]

        for span in sorted_spans[1:]:
            # Check for overlap
            if span.start_offset <= current.end_offset:
                # Overlapping - extend current if this span extends further
                if span.end_offset > current.end_offset:
                    current.end_offset = span.end_offset
                    current.text += span.text[current.end_offset - span.start_offset :]
            else:
                # No overlap - add current and start new
                merged.append(current)
                current = span

        merged.append(current)
        return merged

    @staticmethod
    def create_highlight_regions(
        full_text: str, spans: List[TextSpan]
    ) -> List[Dict[str, Any]]:
        """
        Create non-overlapping highlight regions from spans.

        Args:
            full_text: Complete document text
            spans: TextSpan objects with entity references

        Returns:
            List of highlight regions for frontend rendering
        """
        regions = []
        last_end = 0

        for span in spans:
            # Add non-highlighted text before this span
            if span.start_offset > last_end:
                regions.append(
                    {
                        "text": full_text[last_end : span.start_offset],
                        "highlighted": False,
                        "entities": [],
                    }
                )

            # Add highlighted span
            regions.append(
                {
                    "text": span.text,
                    "highlighted": True,
                    "entities": [
                        {
                            "id": str(span.entity_id),
                            "name": span.entity_name,
                            "type": span.entity_type,
                            "confidence": span.confidence,
                        }
                    ],
                    "start_offset": span.start_offset,
                    "end_offset": span.end_offset,
                    "selectors": span.selectors,
                }
            )

            last_end = span.end_offset

        # Add remaining text after last span
        if last_end < len(full_text):
            regions.append(
                {"text": full_text[last_end:], "highlighted": False, "entities": []}
            )

        return regions

"""Tests for SemanticChunker."""

import pytest

from src.knowledge_base.partitioning.semantic_chunker import (
    SemanticChunker,
    Chunk,
    ChunkWithLinks,
)


class TestSemanticChunkerInitialization:
    """Tests for SemanticChunker initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        chunker = SemanticChunker()
        assert chunker.chunk_size == 1536
        assert chunker.overlap_ratio == 0.25
        assert chunker.min_chunk_size == 256

    def test_custom_initialization(self):
        """Test custom initialization values."""
        chunker = SemanticChunker(
            chunk_size=2048, overlap_ratio=0.3, min_chunk_size=512
        )
        assert chunker.chunk_size == 2048
        assert chunker.overlap_ratio == 0.3
        assert chunker.min_chunk_size == 512

    def test_invalid_chunk_size_zero(self):
        """Test that zero chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """Test that negative chunk size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            SemanticChunker(chunk_size=-100)

    def test_invalid_overlap_ratio_negative(self):
        """Test that negative overlap ratio raises ValueError."""
        with pytest.raises(ValueError, match="overlap_ratio must be in range"):
            SemanticChunker(overlap_ratio=-0.1)

    def test_invalid_overlap_ratio_one(self):
        """Test that overlap ratio of 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="overlap_ratio must be in range"):
            SemanticChunker(overlap_ratio=1.0)

    def test_invalid_min_chunk_size_zero(self):
        """Test that zero min_chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            SemanticChunker(min_chunk_size=0)

    def test_invalid_min_chunk_size_larger_than_chunk_size(self):
        """Test that min_chunk_size > chunk_size raises ValueError."""
        with pytest.raises(
            ValueError, match="min_chunk_size cannot be larger than chunk_size"
        ):
            SemanticChunker(chunk_size=512, min_chunk_size=1024)

    def test_invalid_max_chunk_size_smaller_than_chunk_size(self):
        """Test that max_chunk_size < chunk_size raises ValueError."""
        with pytest.raises(
            ValueError, match="max_chunk_size cannot be smaller than chunk_size"
        ):
            SemanticChunker(chunk_size=2048, max_chunk_size=1024)


class TestMaxChunkSize:
    """Tests for max chunk size property."""

    def test_default_max_chunk_size(self):
        """Test default max chunk size is 2048."""
        chunker = SemanticChunker()
        assert chunker.max_chunk_size == 2048

    def test_custom_max_chunk_size(self):
        """Test custom max chunk size."""
        chunker = SemanticChunker(max_chunk_size=3000)
        assert chunker.max_chunk_size == 3000


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_empty_text_token_count(self):
        """Test token count for empty text."""
        chunker = SemanticChunker()
        assert chunker._count_tokens("") == 0

    def test_short_text_token_count(self):
        """Test token count for short text."""
        chunker = SemanticChunker()
        text = "Hello world"
        tokens = chunker._count_tokens(text)
        assert tokens >= 2
        assert tokens == len(text) // 4 + 1


class TestSemanticBoundaries:
    """Tests for semantic boundary detection."""

    def test_empty_text_boundaries(self):
        """Test boundaries for empty text."""
        chunker = SemanticChunker()
        boundaries = chunker._find_semantic_boundaries("")
        assert boundaries == []

    def test_paragraph_boundaries(self):
        """Test boundaries detect paragraph breaks."""
        chunker = SemanticChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        boundaries = chunker._find_semantic_boundaries(text)
        assert len(boundaries) > 0
        assert 0 in boundaries

    def test_heading_boundaries(self):
        """Test boundaries detect markdown headings."""
        chunker = SemanticChunker()
        text = "# Title\n\nSome content here.\n\n## Subtitle\n\nMore content."
        boundaries = chunker._find_semantic_boundaries(text)
        assert len(boundaries) > 0


class TestSplitIntoSentences:
    """Tests for sentence splitting."""

    def test_empty_text_sentences(self):
        """Test splitting empty text."""
        chunker = SemanticChunker()
        sentences = chunker._split_into_sentences("")
        assert sentences == []

    def test_single_sentence(self):
        """Test splitting single sentence."""
        chunker = SemanticChunker()
        text = "This is a single sentence."
        sentences = chunker._split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_multiple_sentences(self):
        """Test splitting multiple sentences."""
        chunker = SemanticChunker()
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._split_into_sentences(text)
        assert len(sentences) == 3


class TestMergeSmallChunks:
    """Tests for small chunk merging."""

    def test_no_merging_needed(self):
        """Test when no merging is needed."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.0, min_chunk_size=50)
        chunks = [
            Chunk(id="1", text="Chunk 1 text", token_count=60, chunk_index=0),
            Chunk(id="2", text="Chunk 2 text", token_count=70, chunk_index=1),
        ]
        merged = chunker._merge_small_chunks(chunks, 50)
        assert len(merged) == 2

    def test_merge_small_chunks(self):
        """Test merging small chunks."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.0, min_chunk_size=50)
        chunks = [
            Chunk(id="1", text="Small", token_count=10, chunk_index=0),
            Chunk(id="2", text="Also small", token_count=15, chunk_index=1),
            Chunk(
                id="3",
                text="This is a much larger chunk with more content",
                token_count=80,
                chunk_index=2,
            ),
        ]
        merged = chunker._merge_small_chunks(chunks, 50)
        assert len(merged) <= 3

    def test_empty_chunks_list(self):
        """Test merging with empty list."""
        chunker = SemanticChunker()
        merged = chunker._merge_small_chunks([], 50)
        assert merged == []

    def test_single_chunk(self):
        """Test merging with single chunk."""
        chunker = SemanticChunker()
        chunks = [Chunk(id="1", text="Single chunk", token_count=100, chunk_index=0)]
        merged = chunker._merge_small_chunks(chunks, 50)
        assert len(merged) == 1


class TestChunkMethod:
    """Tests for the chunk method."""

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        chunker = SemanticChunker()
        with pytest.raises(ValueError, match="document_text cannot be empty"):
            chunker.chunk("")

    def test_single_chunk_document(self):
        """Test chunking a document that fits in one chunk."""
        chunker = SemanticChunker(
            chunk_size=1000, overlap_ratio=0.0, min_chunk_size=100
        )
        text = "This is a relatively short document that should fit in a single chunk."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_multiple_chunks(self):
        """Test chunking a document that requires multiple chunks."""
        chunker = SemanticChunker(chunk_size=50, overlap_ratio=0.0, min_chunk_size=20)
        text = "Sentence one. " * 50
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_token_count(self):
        """Test that chunk token counts are reasonable."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.0, min_chunk_size=20)
        text = "Word " * 200
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert chunk.token_count > 0
            assert chunk.chunk_index >= 0

    def test_chunk_ids_are_unique(self):
        """Test that chunk IDs are unique."""
        chunker = SemanticChunker(chunk_size=50, overlap_ratio=0.0, min_chunk_size=10)
        text = "Sentence one. " * 50
        chunks = chunker.chunk(text)
        ids = [chunk.id for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_are_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = SemanticChunker(chunk_size=50, overlap_ratio=0.0, min_chunk_size=10)
        text = "Sentence one. " * 50
        chunks = chunker.chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestChunkWithOverlapMethod:
    """Tests for the chunk_with_overlap method."""

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        chunker = SemanticChunker()
        with pytest.raises(ValueError, match="document_text cannot be empty"):
            chunker.chunk_with_overlap("")

    def test_single_chunk_no_overlap(self):
        """Test single chunk has no next/previous links."""
        chunker = SemanticChunker(
            chunk_size=1000, overlap_ratio=0.25, min_chunk_size=100
        )
        text = "Short document."
        chunks = chunker.chunk_with_overlap(text)
        assert len(chunks) == 1
        assert chunks[0].previous_chunk_id is None
        assert chunks[0].next_chunk_id is None

    def test_overlap_links_present(self):
        """Test that overlap links are correctly set."""
        chunker = SemanticChunker(chunk_size=50, overlap_ratio=0.25, min_chunk_size=10)
        text = "First sentence. Second sentence. " * 50
        chunks = chunker.chunk_with_overlap(text)
        assert len(chunks) > 1
        for i, chunk_link in enumerate(chunks):
            if i > 0:
                assert chunk_link.previous_chunk_id is not None
            else:
                assert chunk_link.previous_chunk_id is None
            if i < len(chunks) - 1:
                assert chunk_link.next_chunk_id is not None
            else:
                assert chunk_link.next_chunk_id is None

    def test_overlap_ratio_calculation(self):
        """Test that overlap ratio is correctly applied."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.25, min_chunk_size=10)
        assert chunker._overlap_tokens == 25

    def test_chunk_with_links_contains_valid_chunk(self):
        """Test that ChunkWithLinks contains a valid Chunk."""
        chunker = SemanticChunker(
            chunk_size=1000, overlap_ratio=0.25, min_chunk_size=100
        )
        text = "Test document for chunking."
        chunks = chunker.chunk_with_overlap(text)
        for chunk_link in chunks:
            assert isinstance(chunk_link.chunk, Chunk)
            assert chunk_link.chunk.text is not None
            assert chunk_link.chunk.token_count > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        chunker = SemanticChunker()
        text = "   \n\n   \n\n   "
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_very_long_sentence(self):
        """Test handling of very long sentence."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.1, min_chunk_size=20)
        text = "Word " * 500
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_special_characters(self):
        """Test handling of special characters."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.0, min_chunk_size=20)
        text = "Hello! @#$%^&*() World. 123 456 789."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_multiline_text(self):
        """Test handling of multiline text."""
        chunker = SemanticChunker(chunk_size=100, overlap_ratio=0.0, min_chunk_size=20)
        text = "Line one.\nLine two.\nLine three.\n" * 20
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestChunkIdGeneration:
    """Tests for chunk ID generation."""

    def test_id_format(self):
        """Test that generated IDs have correct format."""
        chunker = SemanticChunker()
        chunks = []
        chunk_id = chunker._generate_chunk_id(chunks, 0)
        assert len(chunk_id) == 16
        assert chunk_id.isalnum()

    def test_different_indices_different_ids(self):
        """Test that different indices produce different IDs."""
        chunker = SemanticChunker()
        chunks = []
        id1 = chunker._generate_chunk_id(chunks, 0)
        id2 = chunker._generate_chunk_id(chunks, 1)
        assert id1 != id2

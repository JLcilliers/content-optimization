"""
Tests for document chunking functionality.

These tests ensure that documents are properly chunked
for comparison in the diffing algorithm.
"""

import pytest

from seo_optimizer.diffing.chunker import (
    ContentChunk,
    DocumentChunker,
    extract_chunks_from_text,
    normalize_text,
)


class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_lowercase(self) -> None:
        """Text should be lowercased."""
        assert normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        """Multiple whitespace should collapse to single space."""
        assert normalize_text("Hello   World") == "hello world"
        assert normalize_text("Hello\n\nWorld") == "hello world"
        assert normalize_text("Hello\t\tWorld") == "hello world"

    def test_strip(self) -> None:
        """Leading/trailing whitespace should be stripped."""
        assert normalize_text("  Hello World  ") == "hello world"

    def test_smart_quotes(self) -> None:
        """Smart quotes should be normalized."""
        assert normalize_text("\u201cHello\u201d") == '"hello"'
        assert normalize_text("\u2018Hello\u2019") == "'hello'"

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert normalize_text("") == ""

    def test_whitespace_only(self) -> None:
        """Whitespace only should return empty string."""
        assert normalize_text("   ") == ""


class TestContentChunk:
    """Tests for ContentChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """ContentChunk should be created with required fields."""
        chunk = ContentChunk(
            chunk_id="c0",
            text="Hello world",
            normalized_text="hello world",
            source_node_ids=["p0"],
            start_char=0,
            end_char=11,
        )
        assert chunk.chunk_id == "c0"
        assert chunk.text == "Hello world"
        assert chunk.char_count == 11
        assert chunk.word_count == 2

    def test_chunk_hashable(self) -> None:
        """ContentChunk should be hashable by normalized text."""
        chunk1 = ContentChunk(
            chunk_id="c0",
            text="Hello World",
            normalized_text="hello world",
            source_node_ids=["p0"],
            start_char=0,
            end_char=11,
        )
        chunk2 = ContentChunk(
            chunk_id="c1",
            text="hello world",
            normalized_text="hello world",
            source_node_ids=["p1"],
            start_char=0,
            end_char=11,
        )
        # Same normalized text = same hash
        assert hash(chunk1) == hash(chunk2)


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    def test_default_initialization(self) -> None:
        """DocumentChunker should initialize with default settings."""
        chunker = DocumentChunker()
        assert chunker.default_chunk_size == 400
        assert chunker.chunk_overlap == 50
        assert chunker.min_chunk_size == 50

    def test_custom_initialization(self) -> None:
        """DocumentChunker should accept custom settings."""
        chunker = DocumentChunker(
            default_chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=25,
        )
        assert chunker.default_chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.min_chunk_size == 25


class TestChunkFixedSize:
    """Tests for fixed-size chunking."""

    def test_short_text_single_chunk(self) -> None:
        """Short text should be a single chunk."""
        chunker = DocumentChunker(default_chunk_size=100)
        chunks = chunker.chunk_fixed_size("Hello world", doc_id="test")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_empty_text_no_chunks(self) -> None:
        """Empty text should return no chunks."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_fixed_size("", doc_id="test")
        assert len(chunks) == 0

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple chunks."""
        chunker = DocumentChunker(default_chunk_size=50, chunk_overlap=10)
        long_text = "This is a longer piece of text that should be split into multiple chunks for processing. " * 3
        chunks = chunker.chunk_fixed_size(long_text, doc_id="test")
        assert len(chunks) > 1

    def test_chunks_have_overlap(self) -> None:
        """Adjacent chunks should have overlapping content."""
        chunker = DocumentChunker(default_chunk_size=50, chunk_overlap=20)
        long_text = "This is a test sentence. " * 10
        chunks = chunker.chunk_fixed_size(long_text, doc_id="test")
        if len(chunks) >= 2:
            # Check that second chunk starts before first chunk ends
            assert chunks[1].start_char < chunks[0].end_char


class TestChunkBySentences:
    """Tests for sentence-level chunking."""

    def test_multiple_sentences(self) -> None:
        """Multiple sentences should create multiple chunks."""
        chunker = DocumentChunker()
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_by_sentences(text, doc_id="test")
        assert len(chunks) == 3

    def test_single_sentence(self) -> None:
        """Single sentence should be one chunk."""
        chunker = DocumentChunker()
        text = "Just one sentence here"
        chunks = chunker.chunk_by_sentences(text, doc_id="test")
        assert len(chunks) == 1

    def test_empty_text_no_chunks(self) -> None:
        """Empty text should return no chunks."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_by_sentences("", doc_id="test")
        assert len(chunks) == 0


class TestChunkFromText:
    """Tests for chunk_from_text convenience method."""

    def test_paragraph_based_chunking(self) -> None:
        """Text with paragraphs should chunk by paragraph."""
        chunker = DocumentChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk_from_text(text, doc_id="test")
        assert len(chunks) == 3

    def test_single_paragraph_short(self) -> None:
        """Short single paragraph should be one chunk."""
        chunker = DocumentChunker()
        text = "Just a short paragraph."
        chunks = chunker.chunk_from_text(text, doc_id="test")
        assert len(chunks) == 1

    def test_empty_text(self) -> None:
        """Empty text should return no chunks."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_from_text("", doc_id="test")
        assert len(chunks) == 0


class TestExtractChunksFromText:
    """Tests for extract_chunks_from_text convenience function."""

    def test_returns_chunks(self) -> None:
        """Should return a list of ContentChunks."""
        chunks = extract_chunks_from_text("Hello world", doc_id="test")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, ContentChunk) for c in chunks)

    def test_empty_text(self) -> None:
        """Empty text should return empty list."""
        chunks = extract_chunks_from_text("", doc_id="test")
        assert chunks == []

    def test_preserves_position(self) -> None:
        """Chunks should have correct position information."""
        text = "First paragraph.\n\nSecond paragraph."
        chunks = extract_chunks_from_text(text, doc_id="test")
        # First chunk should start at 0
        assert chunks[0].start_char == 0


class TestChunkMetadata:
    """Tests for chunk metadata tracking."""

    def test_chunk_has_source_node_ids(self) -> None:
        """Chunks should track source node IDs."""
        chunks = extract_chunks_from_text("Test content", doc_id="doc1")
        assert len(chunks) > 0
        assert all(len(c.source_node_ids) > 0 for c in chunks)

    def test_chunk_has_position_info(self) -> None:
        """Chunks should have start and end character positions."""
        text = "Hello world"
        chunks = extract_chunks_from_text(text, doc_id="test")
        assert len(chunks) > 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

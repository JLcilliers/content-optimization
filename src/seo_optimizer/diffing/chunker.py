"""
Document Chunker - Splits documents into comparable chunks.

Implements hierarchical chunking per Content_Ingestion research:
- Split by H2/H3 boundaries (semantic structure)
- Fallback to fixed-size chunks with overlap
- Track position metadata for precise highlighting

Reference: Content_Ingestion_and_Normalization_Research.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seo_optimizer.ingestion.models import ContentNode, DocumentAST


@dataclass
class ContentChunk:
    """
    A chunk of content for comparison.

    Chunks are the atomic units of comparison in the diffing algorithm.
    Each chunk tracks its origin and position for highlight calculation.
    """

    # Unique identifier for this chunk
    chunk_id: str

    # The text content of the chunk
    text: str

    # Normalized text for comparison (lowercase, whitespace normalized)
    normalized_text: str

    # Source node ID(s) this chunk came from
    source_node_ids: list[str]

    # Character position in the full document
    start_char: int
    end_char: int

    # Heading level if this chunk starts with a heading (0 = not a heading)
    heading_level: int = 0

    # Additional metadata
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())

    def __hash__(self) -> int:
        """Make chunk hashable by its normalized text."""
        return hash(self.normalized_text)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Operations:
    - Lowercase
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    - Remove soft hyphens and zero-width characters

    Args:
        text: Raw text to normalize

    Returns:
        Normalized text suitable for comparison
    """
    if not text:
        return ""

    # Remove soft hyphens and zero-width characters
    text = text.replace("\u00ad", "")  # Soft hyphen
    text = text.replace("\u200b", "")  # Zero-width space
    text = text.replace("\ufeff", "")  # BOM

    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # Smart double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # Smart single quotes

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip
    return text.strip()


class DocumentChunker:
    """
    Splits documents into comparable chunks.

    Implements multiple chunking strategies:
    1. Heading-based: Split at H2/H3 boundaries (preferred for structure)
    2. Fixed-size: 256-512 tokens with overlap (fallback)
    3. Semantic: Split when sentence similarity drops (advanced)
    """

    def __init__(
        self,
        default_chunk_size: int = 400,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            default_chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between adjacent chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged)
        """
        self.default_chunk_size = default_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: DocumentAST) -> list[ContentChunk]:
        """
        Chunk a document using the best available strategy.

        Attempts heading-based chunking first, falls back to fixed-size
        if no headings are present.

        Args:
            doc: The document AST to chunk

        Returns:
            List of ContentChunks for comparison
        """
        # Try heading-based chunking first
        chunks = self.chunk_by_headings(doc)

        # If no meaningful chunks from headings, use fixed-size
        if len(chunks) <= 1 and doc.full_text:
            chunks = self.chunk_fixed_size(
                doc.full_text,
                doc_id=doc.doc_id,
            )

        return chunks

    def chunk_by_headings(self, doc: DocumentAST) -> list[ContentChunk]:
        """
        Split document at H2/H3 boundaries (natural chunk points).

        Per Content_Ingestion research: "H2/H3 tags act as natural
        Chunking Boundaries"

        Args:
            doc: Document AST to chunk

        Returns:
            List of chunks aligned with heading structure
        """
        from seo_optimizer.ingestion.models import NodeType

        chunks: list[ContentChunk] = []
        current_chunk_nodes: list[ContentNode] = []
        current_chunk_text: list[str] = []
        current_start_char = 0
        chunk_idx = 0

        for node in doc.nodes:
            # Check if this is a heading that should start a new chunk
            is_heading = node.node_type == NodeType.HEADING
            heading_level = node.metadata.get("heading_level", 0) if is_heading else 0

            # H2 and H3 start new chunks (per research)
            if is_heading and heading_level in (2, 3) and current_chunk_text:
                # Save current chunk
                chunk = self._create_chunk(
                    chunk_idx=chunk_idx,
                    nodes=current_chunk_nodes,
                    texts=current_chunk_text,
                    start_char=current_start_char,
                    doc_id=doc.doc_id,
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Start new chunk
                current_start_char = node.position.start_char
                current_chunk_nodes = []
                current_chunk_text = []

            # Add node to current chunk
            current_chunk_nodes.append(node)
            current_chunk_text.append(node.text_content)

        # Don't forget the last chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                chunk_idx=chunk_idx,
                nodes=current_chunk_nodes,
                texts=current_chunk_text,
                start_char=current_start_char,
                doc_id=doc.doc_id,
            )
            chunks.append(chunk)

        return chunks

    def chunk_fixed_size(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
        doc_id: str = "doc",
    ) -> list[ContentChunk]:
        """
        Fixed-size chunking with overlap for continuity.

        Per AI_Content_Optimization research: "Fixed-size chunks:
        256-512 tokens with 50-token overlap"

        Args:
            text: Full text to chunk
            chunk_size: Target chunk size (default: self.default_chunk_size)
            overlap: Overlap between chunks (default: self.chunk_overlap)
            doc_id: Document ID for chunk naming

        Returns:
            List of fixed-size chunks with overlap
        """
        if not text:
            return []

        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.chunk_overlap

        chunks: list[ContentChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))

            # Try to end at a word boundary
            if end < len(text):
                # Look for last space in the chunk
                last_space = text.rfind(" ", start, end)
                if last_space > start + self.min_chunk_size:
                    end = last_space

            # Extract chunk text
            chunk_text = text[start:end]

            # Create chunk
            chunk = ContentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                text=chunk_text,
                normalized_text=normalize_text(chunk_text),
                source_node_ids=[f"{doc_id}_fixed_{chunk_idx}"],
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
            chunk_idx += 1

            # Move to next chunk with overlap
            start = end - overlap
            if start >= len(text) - self.min_chunk_size:
                break

        return chunks

    def chunk_by_sentences(
        self,
        text: str,
        doc_id: str = "doc",
    ) -> list[ContentChunk]:
        """
        Split text into sentence-level chunks.

        Useful for fine-grained comparison and semantic analysis.

        Args:
            text: Text to split into sentences
            doc_id: Document ID for chunk naming

        Returns:
            List of sentence-level chunks
        """
        if not text:
            return []

        # Simple sentence splitting (handles . ! ? followed by space and capital)
        # More sophisticated splitting would use spaCy
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        chunks: list[ContentChunk] = []
        char_offset = 0

        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            # Find actual position in original text
            start = text.find(sentence, char_offset)
            if start == -1:
                start = char_offset
            end = start + len(sentence)

            chunk = ContentChunk(
                chunk_id=f"{doc_id}_sent_{idx}",
                text=sentence,
                normalized_text=normalize_text(sentence),
                source_node_ids=[f"{doc_id}_sent_{idx}"],
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
            char_offset = end

        return chunks

    def chunk_from_text(
        self,
        text: str,
        doc_id: str = "doc",
    ) -> list[ContentChunk]:
        """
        Create chunks from plain text.

        Convenience method for creating chunks from raw text input.
        Uses paragraph-based splitting if possible, otherwise fixed-size.

        Args:
            text: Plain text to chunk
            doc_id: Document ID for naming

        Returns:
            List of content chunks
        """
        if not text:
            return []

        # Split by double newline (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)

        if len(paragraphs) > 1:
            # Use paragraph-based chunking
            chunks: list[ContentChunk] = []
            char_offset = 0

            for idx, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue

                start = text.find(para, char_offset)
                if start == -1:
                    start = char_offset
                end = start + len(para)

                chunk = ContentChunk(
                    chunk_id=f"{doc_id}_para_{idx}",
                    text=para,
                    normalized_text=normalize_text(para),
                    source_node_ids=[f"{doc_id}_para_{idx}"],
                    start_char=start,
                    end_char=end,
                )
                chunks.append(chunk)
                char_offset = end

            return chunks

        # Single paragraph - use sentence or fixed-size chunking
        if len(text) > self.default_chunk_size * 2:
            return self.chunk_fixed_size(text, doc_id=doc_id)

        # Small text - return as single chunk
        return [
            ContentChunk(
                chunk_id=f"{doc_id}_full",
                text=text,
                normalized_text=normalize_text(text),
                source_node_ids=[doc_id],
                start_char=0,
                end_char=len(text),
            )
        ]

    def _create_chunk(
        self,
        chunk_idx: int,
        nodes: list[ContentNode],
        texts: list[str],
        start_char: int,
        doc_id: str,
    ) -> ContentChunk:
        """
        Create a ContentChunk from a list of nodes.

        Args:
            chunk_idx: Index of this chunk
            nodes: Content nodes in this chunk
            texts: Text content from each node
            start_char: Starting character position
            doc_id: Document ID

        Returns:
            A ContentChunk containing the combined content
        """
        from seo_optimizer.ingestion.models import NodeType

        combined_text = "\n".join(texts)

        # Check if first node is a heading
        heading_level = 0
        if nodes and nodes[0].node_type == NodeType.HEADING:
            heading_level = nodes[0].metadata.get("heading_level", 0)

        return ContentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_idx}",
            text=combined_text,
            normalized_text=normalize_text(combined_text),
            source_node_ids=[n.node_id for n in nodes],
            start_char=start_char,
            end_char=start_char + len(combined_text),
            heading_level=heading_level,
        )


def extract_chunks_from_text(
    text: str,
    doc_id: str = "doc",
) -> list[ContentChunk]:
    """
    Convenience function to extract chunks from plain text.

    Args:
        text: Plain text to chunk
        doc_id: Document identifier

    Returns:
        List of content chunks
    """
    chunker = DocumentChunker()
    return chunker.chunk_from_text(text, doc_id=doc_id)

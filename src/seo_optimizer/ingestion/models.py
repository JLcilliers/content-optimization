"""
Document AST Models - Internal Representation for DOCX Content

These models provide a structured representation of document content
that enables precise diffing and formatting preservation.

Key design decisions:
- Immutable OriginalSnapshot for diffing baseline
- Position tracking for highlight boundary calculation
- Run-level formatting preservation for DOCX reconstruction
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
from typing import Any


class NodeType(str, Enum):
    """Types of content nodes in the document AST."""

    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    IMAGE = "image"
    HYPERLINK = "hyperlink"
    TEXT_RUN = "text_run"


@dataclass
class PositionInfo:
    """
    Position information for a content node.

    Used for:
    - Mapping original positions for diffing
    - Calculating highlight boundaries
    - Tracking document structure
    """

    # Unique position identifier (e.g., "p0", "h1", "t0_r1_c2")
    position_id: str

    # Character offsets within the full document text
    start_char: int
    end_char: int

    # Optional: line numbers for debugging
    start_line: int | None = None
    end_line: int | None = None

    # Parent position ID for hierarchy tracking
    parent_id: str | None = None


@dataclass
class FormattingInfo:
    """
    Formatting information for a text run or paragraph.

    Preserves all formatting needed for DOCX reconstruction.
    """

    # Text formatting
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False

    # Font properties
    font_name: str | None = None
    font_size: float | None = None  # Points
    font_color: str | None = None  # Hex color

    # Paragraph formatting (when applicable)
    alignment: str | None = None  # left, center, right, justify
    indent_left: float | None = None  # Inches
    indent_right: float | None = None
    indent_first_line: float | None = None
    space_before: float | None = None  # Points
    space_after: float | None = None
    line_spacing: float | None = None

    # Style reference
    style_name: str | None = None

    # Heading level (1-6 for headings, None otherwise)
    heading_level: int | None = None


@dataclass
class TextRun:
    """
    A segment of text with uniform formatting.

    DOCX documents store text in "runs" - segments where formatting
    is consistent. This structure preserves run boundaries for
    accurate highlighting.
    """

    text: str
    formatting: FormattingInfo
    position: PositionInfo

    # Optional hyperlink URL if this run is a link
    hyperlink_url: str | None = None


@dataclass
class ContentNode:
    """
    A node in the document AST representing a structural element.

    Can represent: paragraphs, headings, list items, table cells, etc.
    """

    # Unique identifier for this node
    node_id: str = ""

    # Type of content
    node_type: NodeType = NodeType.PARAGRAPH

    # For simple nodes: the text content
    # For complex nodes: concatenation of child text
    text_content: str = ""

    # Position in original document (optional for flexibility)
    position: PositionInfo | None = None

    # Text runs preserving formatting boundaries
    runs: list[TextRun] = field(default_factory=list)

    # Child nodes (for containers like tables, lists)
    children: list["ContentNode"] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Generate position if not provided."""
        if self.position is None:
            self.position = PositionInfo(
                position_id=self.node_id or "auto",
                start_char=0,
                end_char=len(self.text_content) if self.text_content else 0,
            )


@dataclass
class DocumentMetadata:
    """Metadata about the source document."""

    # File information
    source_path: str | None = None
    file_size: int | None = None

    # Document properties
    title: str | None = None
    author: str | None = None
    created: datetime | None = None
    modified: datetime | None = None

    # Processing information
    parsed_at: datetime = field(default_factory=datetime.now)
    parser_version: str = "1.0.0"


@dataclass
class DocumentAST:
    """
    Complete Abstract Syntax Tree representation of a document.

    This is the primary internal representation used throughout
    the optimization pipeline.
    """

    # Root nodes (top-level content elements)
    nodes: list[ContentNode] = field(default_factory=list)

    # Document metadata (accepts dict for backward compatibility)
    metadata: DocumentMetadata | dict = field(default_factory=dict)

    # Unique document identifier
    doc_id: str = ""

    # Full text content (for diffing)
    full_text: str = ""

    # Total character count
    char_count: int = 0

    def __post_init__(self) -> None:
        """Convert dict metadata to DocumentMetadata if needed."""
        if isinstance(self.metadata, dict):
            self.metadata = DocumentMetadata(**self.metadata) if self.metadata else DocumentMetadata()
        if not self.doc_id:
            import uuid
            self.doc_id = str(uuid.uuid4())[:8]

    def get_node_by_id(self, node_id: str) -> ContentNode | None:
        """Find a node by its ID (recursive search)."""

        def search(nodes: list[ContentNode]) -> ContentNode | None:
            for node in nodes:
                if node.node_id == node_id:
                    return node
                result = search(node.children)
                if result:
                    return result
            return None

        return search(self.nodes)

    def get_node_by_position(self, position_id: str) -> ContentNode | None:
        """Find a node by its position ID."""

        def search(nodes: list[ContentNode]) -> ContentNode | None:
            for node in nodes:
                if node.position.position_id == position_id:
                    return node
                result = search(node.children)
                if result:
                    return result
            return None

        return search(self.nodes)


@dataclass(frozen=True)
class OriginalSnapshot:
    """
    Immutable snapshot of original document content for diffing.

    This is a frozen dataclass to prevent any accidental modifications
    that could corrupt the diffing baseline.

    Key properties:
    - Immutable: cannot be changed after creation
    - Hashable: can verify integrity
    - Position-mapped: enables precise change detection
    """

    # Document identifier
    doc_id: str

    # Hash of full content for integrity verification
    content_hash: str

    # Position ID to original text mapping
    # Stored as tuple of tuples for immutability
    text_by_position: tuple[tuple[str, str], ...]

    # Structure fingerprint (detects structural changes)
    structure_fingerprint: str

    # Timestamp when snapshot was created
    created_at: str  # ISO format string (frozen dataclass needs hashable)

    @classmethod
    def from_document_ast(cls, ast: DocumentAST) -> "OriginalSnapshot":
        """Create an immutable snapshot from a DocumentAST."""
        # Build position -> text mapping
        text_mapping: list[tuple[str, str]] = []

        def extract_text(nodes: list[ContentNode]) -> None:
            for node in nodes:
                text_mapping.append((node.position.position_id, node.text_content))
                extract_text(node.children)

        extract_text(ast.nodes)

        # Calculate content hash
        content_hash = sha256(ast.full_text.encode()).hexdigest()

        # Calculate structure fingerprint (node types and positions)
        structure_parts: list[str] = []

        def build_fingerprint(nodes: list[ContentNode], depth: int = 0) -> None:
            for node in nodes:
                structure_parts.append(f"{depth}:{node.node_type.value}")
                build_fingerprint(node.children, depth + 1)

        build_fingerprint(ast.nodes)
        structure_fingerprint = sha256("|".join(structure_parts).encode()).hexdigest()[:16]

        return cls(
            doc_id=ast.doc_id,
            content_hash=content_hash,
            text_by_position=tuple(text_mapping),
            structure_fingerprint=structure_fingerprint,
            created_at=datetime.now().isoformat(),
        )

    def get_text_for_position(self, position_id: str) -> str | None:
        """Get original text for a given position ID."""
        for pos_id, text in self.text_by_position:
            if pos_id == position_id:
                return text
        return None

    def verify_integrity(self, full_text: str) -> bool:
        """Verify that content hasn't changed since snapshot."""
        return sha256(full_text.encode()).hexdigest() == self.content_hash

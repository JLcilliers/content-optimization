# Topic A: Content Ingestion & Normalization
## Technical Specification for SEO + AI Content Optimization Tool

**Document Version:** 1.0
**Date:** 2026-01-16
**Author:** Technical Research & Analysis

---

## Executive Summary

Content ingestion and normalization forms the foundational layer of any SEO and AI content optimization tool. This document presents a comprehensive analysis of approaches for parsing diverse document formats (DOCX, HTML, Markdown, Google Docs) into a unified, semantic representation suitable for AI-powered analysis and optimization.

The research identifies **trafilatura** and **selectolax** as optimal choices for HTML parsing (30x faster than BeautifulSoup), **python-docx** with supplementary XML parsing for comprehensive DOCX handling, and **mistune** for markdown processing. For documents exceeding token limits, we recommend a hybrid chunking strategy combining heading-aware splitting with semantic boundary detection using embedding-based similarity thresholds.

The proposed internal representation uses a JSON-based document AST with Pydantic validation, supporting hierarchical structure preservation while remaining serializable and extensible. This approach balances processing speed (critical for user-facing tools), accuracy (preserving document semantics), and implementation complexity.

Key performance targets include: sub-second processing for documents under 50KB, >95% structure preservation accuracy, and graceful handling of edge cases including nested tables, tracked changes, and embedded media. The architecture supports both batch processing and real-time ingestion pipelines.

---

## 1. Background Research

### Current State of Document Parsing in Python (2025)

The Python ecosystem offers mature libraries for document parsing, with recent performance improvements driven by:

1. **Native code integration**: Libraries like selectolax (Cython wrapper around C parsers) deliver 5-30x speedups over pure Python implementations
2. **Modern Python features**: Type hints, async/await, and Pydantic models enable robust, maintainable parsing pipelines
3. **AI/RAG focus**: Growing demand for document chunking and semantic preservation in LLM applications has spawned specialized tools
4. **Standards compliance**: CommonMark, HTML5, and OOXML (Office Open XML) standards provide reliable parsing targets

### Evolution of Document Processing

Traditional document parsing focused on text extraction with minimal structure preservation. Modern RAG (Retrieval-Augmented Generation) applications require:

- **Semantic structure retention**: Headings, lists, and tables as first-class elements
- **Metadata preservation**: Styles, formatting hints, source location tracking
- **Chunk boundary awareness**: Splitting documents while maintaining semantic coherence
- **Cross-format normalization**: Unified representation across input formats

---

## 2. Library Comparison Matrix

### 2.1 DOCX Parsing Libraries

| Feature | python-docx | docx2python | mammoth | Unstructured |
|---------|-------------|-------------|---------|--------------|
| **Maintenance Status** | Active (2025) | Active (2025) | Active (2025) | Active (2025) |
| **Parsing Approach** | DOM-based | Comprehensive extraction | HTML conversion | Multi-format unified |
| **Structure Preservation** | Excellent | Good | Excellent (as HTML) | Excellent |
| **Table Support** | Full access to cells | Nested list format | HTML tables | Semantic elements |
| **Nested Tables** | Supported | Supported (nxm normalization) | Converts to HTML | Supported |
| **Image Extraction** | Via XML access | Built-in | Limited | Built-in |
| **Alt Text Support** | Via PR #227 (descr attr) | Limited | Limited | Good |
| **Performance** | Moderate | Fast | Fast | Moderate |
| **Headers/Footers** | Supported | Separate attributes | Not primary focus | Supported |
| **Track Changes** | Limited (requires XML) | No | No | Limited |
| **Style Preservation** | Full | Minimal | Semantic only | Semantic |
| **Write Capabilities** | Yes | No | No | No |
| **Best For** | Full control, editing | Structured data extraction | Web ingestion | RAG pipelines |
| **Installation Size** | Small (~500KB) | Small | Small | Large (~50MB+) |

**Decision Matrix:**

```
Use python-docx when:
- Need read/write capabilities
- Require run-level style control
- Processing clean corporate documents
- Need deterministic, rule-based extraction

Use docx2python when:
- Extracting tables with complex nesting
- Need separate header/footer handling
- Processing speed is critical
- Only reading, not writing

Use mammoth when:
- Converting DOCX to web-ready HTML
- Feeding parsed content to HTML processors
- Style-to-semantic conversion needed
- Integration with HTML-based workflows

Use Unstructured when:
- Processing multiple formats uniformly
- Building RAG/LLM ingestion pipelines
- Need automatic element detection
- Semantic chunking is required
```

### 2.2 HTML Parsing Libraries

| Feature | selectolax | lxml | trafilatura | BeautifulSoup 4 |
|---------|------------|------|-------------|-----------------|
| **Performance (100k ops)** | 3.4s | 6.4s (xpath), 10s (css) | ~5s | 95.4s |
| **Speed vs BS4** | 30x faster | 14x faster | 19x faster | Baseline |
| **Parser Backend** | Modest/lexbor (C) | libxml2 (C) | lxml + custom rules | Multiple (lxml, html.parser, html5lib) |
| **HTML5 Compliance** | Yes | Partial | Yes | Yes (with html5lib) |
| **Malformed HTML** | Excellent | Excellent | Excellent | Excellent |
| **CSS Selectors** | Yes | Yes (cssselect) | Limited | Yes |
| **XPath Support** | Limited | Full | Via lxml | Limited |
| **Main Content Extraction** | Manual | Manual | **Built-in** | Manual |
| **Boilerplate Removal** | Manual | Manual | **Automatic** | Manual |
| **API Ease of Use** | Moderate | Moderate | Simple | **Easiest** |
| **Memory Footprint** | Low | Low | Moderate | High |
| **Benchmarks (F1 score)** | N/A | N/A | **0.937** | N/A |
| **Best For** | Speed-critical scraping | XML processing | Content extraction | Learning, prototyping |

**Decision Matrix:**

```
Use selectolax when:
- Processing large volumes of HTML
- Speed is the primary concern
- Drop-in BS4 replacement needed
- Memory efficiency matters

Use lxml when:
- Need full XPath support
- Processing XML documents
- Require low-level control
- Speed important but not critical

Use trafilatura when:
- Extracting main article content
- Removing navigation/boilerplate
- Processing web pages for NLP/SEO
- Need high precision/recall

Use BeautifulSoup when:
- Learning/prototyping
- Speed not critical
- Need gentle learning curve
- One-off scripts
```

**Recommendation for SEO Tool:** **trafilatura** (primary) + **selectolax** (fallback for custom parsing)

- trafilatura's 93.7% F1 score for content extraction is unmatched
- Automatic boilerplate removal critical for SEO content
- Can fallback to selectolax for edge cases needing custom logic

### 2.3 Markdown Parsing Libraries

| Feature | mistune | markdown-it-py | marko |
|---------|---------|----------------|-------|
| **Maintenance Status** | Active (3.0+) | Active (2025) | Active (2025) |
| **Performance** | **Fastest** (3.62s) | Moderate (9.03s) | Slow (3x Python-Markdown) |
| **CommonMark Compliance** | Compatible (not strict) | **Strict** | **Strict** |
| **GitHub-Flavored Markdown** | Via plugins | Via extensions | **Built-in** |
| **Tables** | Yes | Yes | Yes |
| **Strikethrough** | Yes | Yes | Yes |
| **Task Lists** | Yes | Yes | Yes |
| **Extensibility** | Plugins | Configurable rules | High extensibility |
| **AST Access** | Yes | Yes | **Excellent** |
| **Rendering to HTML** | Built-in | Built-in | Built-in |
| **Rendering to Custom** | Moderate | Good | **Excellent** |
| **Best For** | Speed, general use | CommonMark strict | GFM, extensibility |

**Decision Matrix:**

```
Use mistune when:
- Processing speed is critical
- CommonMark strict compliance not required
- Need simple plugin architecture
- General Markdown parsing

Use markdown-it-py when:
- Need CommonMark compliance
- JavaScript markdown-it parity needed
- Configurable syntax rules important
- Moderate performance acceptable

Use marko when:
- GitHub-Flavored Markdown required
- Need extensive customization
- AST manipulation is primary use case
- Speed not critical
```

**Recommendation:** **mistune** (primary) with fallback to **marko** for GFM-specific features

---

## 3. Internal Representation Schema

### 3.1 Design Principles

The normalized content structure must:

1. **Be format-agnostic**: Same representation for content from DOCX, HTML, or Markdown
2. **Preserve semantics**: Heading hierarchy, list nesting, table structure
3. **Support metadata**: Styles, source locations, formatting hints
4. **Enable chunking**: Allow efficient semantic splitting
5. **Be serializable**: JSON-compatible for storage/transmission
6. **Validate strictly**: Pydantic models ensure data integrity

### 3.2 Document AST Specification

```python
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class NodeType(str, Enum):
    """Enumeration of all supported node types."""
    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TEXT = "text"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    HORIZONTAL_RULE = "horizontal_rule"
    LINK = "link"
    STRONG = "strong"
    EMPHASIS = "emphasis"


class SourceLocation(BaseModel):
    """Tracks original location in source document."""
    format: str  # "docx", "html", "markdown", "gdocs"
    path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    xpath: Optional[str] = None  # For HTML
    paragraph_index: Optional[int] = None  # For DOCX


class Metadata(BaseModel):
    """Extensible metadata for any node."""
    source: Optional[SourceLocation] = None
    styles: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    # For tracking changes / comments
    revision_id: Optional[str] = None
    comment_id: Optional[str] = None
    author: Optional[str] = None
    timestamp: Optional[str] = None


class DocumentNode(BaseModel):
    """Base class for all document nodes."""
    type: NodeType
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    children: list["DocumentNode"] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator('children')
    @classmethod
    def validate_children(cls, v, info):
        """Validate child node types are appropriate for parent."""
        # Validation logic based on node type
        return v


class TextNode(DocumentNode):
    """Leaf node containing actual text content."""
    type: Literal[NodeType.TEXT] = NodeType.TEXT
    content: str
    formatting: dict[str, bool] = Field(default_factory=dict)  # bold, italic, underline, etc.


class HeadingNode(DocumentNode):
    """Heading node (h1-h6)."""
    type: Literal[NodeType.HEADING] = NodeType.HEADING
    level: int = Field(ge=1, le=6)

    @field_validator('children')
    @classmethod
    def heading_children_are_inline(cls, v):
        """Headings can only contain inline elements."""
        allowed = {NodeType.TEXT, NodeType.LINK, NodeType.STRONG, NodeType.EMPHASIS}
        if any(child.type not in allowed for child in v):
            raise ValueError("Headings can only contain inline elements")
        return v


class ParagraphNode(DocumentNode):
    """Paragraph node."""
    type: Literal[NodeType.PARAGRAPH] = NodeType.PARAGRAPH
    alignment: Optional[str] = None  # "left", "center", "right", "justify"


class ListNode(DocumentNode):
    """Ordered or unordered list."""
    type: Literal[NodeType.LIST] = NodeType.LIST
    ordered: bool = False
    start: int = 1  # For ordered lists

    @field_validator('children')
    @classmethod
    def list_children_are_items(cls, v):
        """Lists can only contain list items."""
        if any(child.type != NodeType.LIST_ITEM for child in v):
            raise ValueError("Lists can only contain list items")
        return v


class ListItemNode(DocumentNode):
    """List item (can contain nested lists)."""
    type: Literal[NodeType.LIST_ITEM] = NodeType.LIST_ITEM
    checked: Optional[bool] = None  # For task lists


class TableNode(DocumentNode):
    """Table structure."""
    type: Literal[NodeType.TABLE] = NodeType.TABLE
    rows: int
    columns: int
    has_header: bool = False

    @field_validator('children')
    @classmethod
    def table_children_are_rows(cls, v):
        """Tables can only contain rows."""
        if any(child.type != NodeType.TABLE_ROW for child in v):
            raise ValueError("Tables can only contain table rows")
        return v


class TableRowNode(DocumentNode):
    """Table row."""
    type: Literal[NodeType.TABLE_ROW] = NodeType.TABLE_ROW
    is_header: bool = False

    @field_validator('children')
    @classmethod
    def row_children_are_cells(cls, v):
        """Rows can only contain cells."""
        if any(child.type != NodeType.TABLE_CELL for child in v):
            raise ValueError("Table rows can only contain cells")
        return v


class TableCellNode(DocumentNode):
    """Table cell (can contain nested tables)."""
    type: Literal[NodeType.TABLE_CELL] = NodeType.TABLE_CELL
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


class ImageNode(DocumentNode):
    """Image reference."""
    type: Literal[NodeType.IMAGE] = NodeType.IMAGE
    src: str  # URL or base64-encoded data
    alt_text: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    caption: Optional[str] = None  # Extracted from surrounding context
    children: list[DocumentNode] = Field(default_factory=list)  # Images don't have children


class CodeBlockNode(DocumentNode):
    """Code block with syntax highlighting info."""
    type: Literal[NodeType.CODE_BLOCK] = NodeType.CODE_BLOCK
    language: Optional[str] = None
    code: str


class LinkNode(DocumentNode):
    """Hyperlink (inline element)."""
    type: Literal[NodeType.LINK] = NodeType.LINK
    url: str
    title: Optional[str] = None


class Document(BaseModel):
    """Root document container."""
    version: str = "1.0"
    root: DocumentNode
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """Deserialize from JSON."""
        return cls.model_validate_json(json_str)
```

### 3.3 Example Document Representation

```json
{
  "version": "1.0",
  "metadata": {
    "title": "SEO Best Practices Guide",
    "author": "Content Team",
    "created": "2026-01-15T10:30:00Z",
    "source_format": "docx"
  },
  "root": {
    "type": "document",
    "id": "doc-root-001",
    "children": [
      {
        "type": "heading",
        "id": "h1-001",
        "level": 1,
        "children": [
          {
            "type": "text",
            "id": "txt-001",
            "content": "SEO Best Practices Guide",
            "formatting": {"bold": true}
          }
        ],
        "metadata": {
          "source": {
            "format": "docx",
            "paragraph_index": 0
          },
          "styles": {"style_name": "Heading 1"}
        }
      },
      {
        "type": "paragraph",
        "id": "p-001",
        "children": [
          {
            "type": "text",
            "id": "txt-002",
            "content": "This guide covers essential SEO techniques for "
          },
          {
            "type": "strong",
            "id": "strong-001",
            "children": [
              {
                "type": "text",
                "id": "txt-003",
                "content": "content optimization"
              }
            ]
          },
          {
            "type": "text",
            "id": "txt-004",
            "content": " and keyword research."
          }
        ]
      },
      {
        "type": "table",
        "id": "tbl-001",
        "rows": 3,
        "columns": 2,
        "has_header": true,
        "children": [
          {
            "type": "table_row",
            "id": "tr-001",
            "is_header": true,
            "children": [
              {
                "type": "table_cell",
                "id": "tc-001",
                "is_header": true,
                "children": [
                  {
                    "type": "text",
                    "id": "txt-005",
                    "content": "Technique"
                  }
                ]
              },
              {
                "type": "table_cell",
                "id": "tc-002",
                "is_header": true,
                "children": [
                  {
                    "type": "text",
                    "id": "txt-006",
                    "content": "Impact"
                  }
                ]
              }
            ]
          }
        ],
        "metadata": {
          "source": {
            "format": "docx",
            "paragraph_index": 5
          }
        }
      }
    ]
  }
}
```

### 3.4 Schema Advantages

1. **Type Safety**: Pydantic validates all nodes at runtime
2. **Extensibility**: `metadata.attributes` allows custom fields without schema changes
3. **Queryability**: Tree structure enables XPath-like queries
4. **Reversibility**: Sufficient metadata to reconstruct source format
5. **Diff-Friendly**: UUIDs enable change tracking across versions
6. **Chunking-Ready**: Heading nodes naturally define semantic boundaries

---

## 4. Semantic Chunking Algorithm

### 4.1 Problem Statement

Modern LLMs have context windows ranging from 8k to 200k tokens. However:

- **Lost-in-the-middle**: Large contexts reduce retrieval accuracy
- **Cost**: Longer contexts increase API costs
- **Latency**: More tokens = slower processing
- **Precision**: Smaller, focused chunks improve RAG relevance

**Target**: Split documents into 500-1500 token chunks while preserving semantic coherence.

### 4.2 Chunking Strategy Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Strategy 1: Heading-Aware Splitting                    │
│ - Split at heading boundaries (h1 > h2 > h3...)        │
│ - Preserve heading hierarchy in chunk metadata         │
│ - Best for well-structured documents                   │
└─────────────────────────────────────────────────────────┘
                        ↓ (if chunk > max_tokens)
┌─────────────────────────────────────────────────────────┐
│ Strategy 2: Semantic Boundary Detection                │
│ - Embed sentences with sentence-transformers           │
│ - Calculate cosine similarity between adjacent chunks  │
│ - Split where similarity < threshold                   │
│ - Best for narrative content                           │
└─────────────────────────────────────────────────────────┘
                        ↓ (if still too large)
┌─────────────────────────────────────────────────────────┐
│ Strategy 3: Recursive Character Splitting              │
│ - Split on: ["\n\n", "\n", ". ", " "]                 │
│ - Add overlap for context preservation                │
│ - Fallback for any content                            │
└─────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Specification

```python
from typing import Protocol
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np


class TokenCounter(Protocol):
    """Protocol for token counting."""

    def count(self, text: str) -> int:
        """Count tokens in text."""
        ...


class TiktokenCounter:
    """Fast token counting using tiktoken."""

    def __init__(self, model: str = "gpt-4"):
        self.encoding = tiktoken.encoding_for_model(model)

    def count(self, text: str) -> int:
        """Count tokens with exact model encoding."""
        return len(self.encoding.encode(text))


class ApproximateCounter:
    """Fast approximate counter for large texts."""

    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Approximate token count (fast for >10MB texts)."""
        return int(len(text) / self.chars_per_token)


class DocumentChunk(BaseModel):
    """A chunk of a document."""
    id: str
    content: str
    token_count: int
    heading_path: list[str]  # ["Chapter 1", "Section 1.2", "Subsection 1.2.3"]
    node_ids: list[str]  # IDs of nodes included in chunk
    metadata: dict[str, Any] = Field(default_factory=dict)
    overlap_prev: int = 0  # Tokens overlapping with previous chunk
    overlap_next: int = 0  # Tokens overlapping with next chunk


class SemanticChunker:
    """Hybrid chunking strategy for document AST."""

    def __init__(
        self,
        target_tokens: int = 1000,
        max_tokens: int = 1500,
        min_tokens: int = 500,
        overlap_tokens: int = 100,
        similarity_threshold: float = 0.6,
        token_counter: Optional[TokenCounter] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.similarity_threshold = similarity_threshold
        self.token_counter = token_counter or TiktokenCounter()
        self.embedder = SentenceTransformer(embedding_model)

    def chunk_document(self, doc: Document) -> list[DocumentChunk]:
        """
        Chunk document using hybrid strategy.

        Algorithm:
        1. Extract heading hierarchy
        2. Split at heading boundaries if possible
        3. For large sections, apply semantic splitting
        4. Add overlap between chunks for context
        """
        chunks: list[DocumentChunk] = []

        # Step 1: Extract sections based on headings
        sections = self._extract_sections(doc.root)

        for section in sections:
            section_text = self._node_to_text(section.node)
            section_tokens = self.token_counter.count(section_text)

            if section_tokens <= self.max_tokens:
                # Section fits in one chunk
                chunks.append(DocumentChunk(
                    id=f"chunk-{len(chunks)}",
                    content=section_text,
                    token_count=section_tokens,
                    heading_path=section.heading_path,
                    node_ids=self._collect_node_ids(section.node)
                ))
            else:
                # Section too large, apply semantic splitting
                sub_chunks = self._semantic_split(
                    section.node,
                    section.heading_path
                )
                chunks.extend(sub_chunks)

        # Step 2: Add overlap between chunks
        chunks = self._add_overlap(chunks)

        return chunks

    def _extract_sections(self, node: DocumentNode) -> list["Section"]:
        """
        Extract document sections based on heading hierarchy.

        Returns sections with heading path context.
        """
        sections: list[Section] = []
        current_section: Optional[Section] = None
        heading_stack: list[tuple[int, str]] = []

        def walk(n: DocumentNode):
            nonlocal current_section

            if n.type == NodeType.HEADING:
                # Start new section
                heading_text = self._node_to_text(n)
                level = n.level

                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

                # Save previous section
                if current_section:
                    sections.append(current_section)

                # Start new section
                current_section = Section(
                    node=DocumentNode(type=NodeType.DOCUMENT, children=[]),
                    heading_path=[h[1] for h in heading_stack]
                )

            if current_section:
                current_section.node.children.append(n)

            # Recurse
            for child in n.children:
                walk(child)

        walk(node)

        # Add final section
        if current_section:
            sections.append(current_section)

        return sections

    def _semantic_split(
        self,
        node: DocumentNode,
        heading_path: list[str]
    ) -> list[DocumentChunk]:
        """
        Split large sections using semantic similarity.

        Algorithm:
        1. Extract sentences/paragraphs
        2. Embed each unit
        3. Calculate similarity between adjacent units
        4. Split where similarity drops below threshold
        5. Group units into chunks within token limits
        """
        # Extract text units (paragraphs)
        units = self._extract_text_units(node)

        if len(units) <= 1:
            # Fall back to recursive splitting
            return self._recursive_split(node, heading_path)

        # Embed units
        embeddings = self.embedder.encode([u.text for u in units])

        # Calculate similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
            similarities.append(sim)

        # Find split points (low similarity)
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_indices.append(i + 1)
        split_indices.append(len(units))

        # Create chunks from splits
        chunks: list[DocumentChunk] = []
        for i in range(len(split_indices) - 1):
            start = split_indices[i]
            end = split_indices[i + 1]

            chunk_units = units[start:end]
            chunk_text = "\n\n".join([u.text for u in chunk_units])
            chunk_tokens = self.token_counter.count(chunk_text)

            # If still too large, recursively split
            if chunk_tokens > self.max_tokens:
                # Combine units until hitting max_tokens
                current_chunk: list[TextUnit] = []
                current_tokens = 0

                for unit in chunk_units:
                    unit_tokens = self.token_counter.count(unit.text)

                    if current_tokens + unit_tokens > self.max_tokens and current_chunk:
                        # Save current chunk
                        chunks.append(DocumentChunk(
                            id=f"chunk-{len(chunks)}",
                            content="\n\n".join([u.text for u in current_chunk]),
                            token_count=current_tokens,
                            heading_path=heading_path,
                            node_ids=[u.node_id for u in current_chunk]
                        ))
                        current_chunk = []
                        current_tokens = 0

                    current_chunk.append(unit)
                    current_tokens += unit_tokens

                # Save final chunk
                if current_chunk:
                    chunks.append(DocumentChunk(
                        id=f"chunk-{len(chunks)}",
                        content="\n\n".join([u.text for u in current_chunk]),
                        token_count=current_tokens,
                        heading_path=heading_path,
                        node_ids=[u.node_id for u in current_chunk]
                    ))
            else:
                chunks.append(DocumentChunk(
                    id=f"chunk-{len(chunks)}",
                    content=chunk_text,
                    token_count=chunk_tokens,
                    heading_path=heading_path,
                    node_ids=[u.node_id for u in chunk_units]
                ))

        return chunks

    def _recursive_split(
        self,
        node: DocumentNode,
        heading_path: list[str]
    ) -> list[DocumentChunk]:
        """
        Fallback: recursive character splitting.

        Splits on: paragraph breaks > line breaks > sentences > words
        """
        text = self._node_to_text(node)
        separators = ["\n\n", "\n", ". ", " "]

        chunks: list[DocumentChunk] = []
        current_chunk = ""
        current_tokens = 0

        def split_text(txt: str, seps: list[str]) -> list[str]:
            if not seps:
                return [txt]

            sep = seps[0]
            parts = txt.split(sep)

            if len(parts) == 1:
                # Separator not found, try next
                return split_text(txt, seps[1:])

            return parts

        parts = split_text(text, separators)

        for part in parts:
            part_tokens = self.token_counter.count(part)

            if current_tokens + part_tokens > self.max_tokens and current_chunk:
                # Save chunk
                chunks.append(DocumentChunk(
                    id=f"chunk-{len(chunks)}",
                    content=current_chunk,
                    token_count=current_tokens,
                    heading_path=heading_path,
                    node_ids=self._collect_node_ids(node)
                ))
                current_chunk = ""
                current_tokens = 0

            current_chunk += part + " "
            current_tokens += part_tokens

        # Save final chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                id=f"chunk-{len(chunks)}",
                content=current_chunk,
                token_count=current_tokens,
                heading_path=heading_path,
                node_ids=self._collect_node_ids(node)
            ))

        return chunks

    def _add_overlap(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """
        Add overlapping context between consecutive chunks.

        Takes last N tokens from previous chunk and prepends to next.
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks: list[DocumentChunk] = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk: no previous overlap
                overlapped_chunks.append(chunk)
                continue

            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            prev_words = prev_chunk.content.split()

            # Estimate overlap (roughly)
            overlap_words = []
            overlap_tokens = 0
            for word in reversed(prev_words):
                word_tokens = self.token_counter.count(word)
                if overlap_tokens + word_tokens > self.overlap_tokens:
                    break
                overlap_words.insert(0, word)
                overlap_tokens += word_tokens

            overlap_text = " ".join(overlap_words)

            # Prepend overlap to current chunk
            new_content = overlap_text + "\n\n" + chunk.content
            new_token_count = self.token_counter.count(new_content)

            overlapped_chunks.append(DocumentChunk(
                id=chunk.id,
                content=new_content,
                token_count=new_token_count,
                heading_path=chunk.heading_path,
                node_ids=chunk.node_ids,
                metadata=chunk.metadata,
                overlap_prev=overlap_tokens
            ))

        return overlapped_chunks

    def _node_to_text(self, node: DocumentNode) -> str:
        """Extract plain text from node tree."""
        if node.type == NodeType.TEXT:
            return node.content

        return " ".join([self._node_to_text(child) for child in node.children])

    def _collect_node_ids(self, node: DocumentNode) -> list[str]:
        """Collect all node IDs in subtree."""
        ids = [node.id]
        for child in node.children:
            ids.extend(self._collect_node_ids(child))
        return ids

    def _extract_text_units(self, node: DocumentNode) -> list["TextUnit"]:
        """Extract paragraphs as text units for semantic chunking."""
        units: list[TextUnit] = []

        def walk(n: DocumentNode):
            if n.type == NodeType.PARAGRAPH:
                text = self._node_to_text(n)
                if text.strip():
                    units.append(TextUnit(text=text, node_id=n.id))
            else:
                for child in n.children:
                    walk(child)

        walk(node)
        return units


class Section(BaseModel):
    """Document section with heading context."""
    node: DocumentNode
    heading_path: list[str]


class TextUnit(BaseModel):
    """Text unit for semantic chunking."""
    text: str
    node_id: str
```

### 4.4 Chunking Algorithm Performance

**Benchmarks** (approximate, based on research):

| Strategy | Speed | Accuracy | Best For |
|----------|-------|----------|----------|
| Heading-aware | Fast (O(n)) | High (if well-structured) | Technical docs, reports |
| Semantic (embedding) | Moderate (O(n*d)) | Very High | Narrative content, articles |
| Recursive character | Fast (O(n)) | Moderate | Fallback, unstructured text |

**Parameter Tuning**:

```python
# Conservative (high quality)
chunker = SemanticChunker(
    target_tokens=800,
    max_tokens=1200,
    min_tokens=400,
    overlap_tokens=150,
    similarity_threshold=0.7
)

# Aggressive (speed/cost)
chunker = SemanticChunker(
    target_tokens=1500,
    max_tokens=2000,
    min_tokens=800,
    overlap_tokens=50,
    similarity_threshold=0.5
)
```

---

## 5. Edge Case Analysis

### 5.1 Nested Tables

**Challenge**: Tables within table cells create complex hierarchies.

**Solutions**:

1. **Flatten on parse** (docx2python approach):
   ```python
   # Represent as nxm matrix, filling merged cells
   table = [
       ["A1", "B1", "C1"],
       ["A2", "B2 (nested table)", "C2"],
       ["A3", "B3", "C3"]
   ]
   ```

2. **Preserve hierarchy** (python-docx approach):
   ```python
   TableCellNode(
       children=[
           ParagraphNode(children=[TextNode(content="Cell content")]),
           TableNode(children=[...])  # Nested table
       ]
   )
   ```

**Recommendation**: Preserve hierarchy in AST, provide flatten utility for downstream consumers.

```python
def flatten_table(table_node: TableNode) -> list[list[str]]:
    """
    Flatten table to 2D array, expanding nested tables.

    Nested tables converted to markdown-style representation.
    """
    rows: list[list[str]] = []

    for row in table_node.children:
        cells: list[str] = []
        for cell in row.children:
            cell_content = []
            for child in cell.children:
                if child.type == NodeType.TABLE:
                    # Represent nested table as markdown
                    nested = flatten_table(child)
                    md = "\n".join([" | ".join(row) for row in nested])
                    cell_content.append(f"[Nested Table]\n{md}")
                else:
                    cell_content.append(node_to_text(child))
            cells.append(" ".join(cell_content))
        rows.append(cells)

    return rows
```

**Edge Cases**:
- **Deeply nested tables** (3+ levels): Warn user, flatten beyond depth threshold
- **Merged cells with nested tables**: Store rowspan/colspan metadata
- **Empty cells in nested tables**: Preserve structure with empty strings

### 5.2 Embedded Images

**Challenges**:
1. **Image data extraction**: Binary blobs in DOCX, URLs in HTML
2. **Alt text availability**: May be missing or generic
3. **Caption detection**: Not always semantically linked
4. **Inline vs. block**: Position context matters

**Solutions**:

```python
class ImageExtractor:
    """Extract images with metadata from various formats."""

    def extract_from_docx(self, doc_path: str) -> list[ImageNode]:
        """Extract images from DOCX using python-docx + XML."""
        from docx import Document
        from docx.oxml import parse_xml

        document = Document(doc_path)
        images: list[ImageNode] = []

        # Iterate through inline shapes
        for rel in document.part.rels.values():
            if "image" in rel.target_ref:
                image_part = rel.target_part
                image_bytes = image_part.blob

                # Get alt text from XML
                # Look for <wp:docPr descr="alt text here"/>
                alt_text = None
                for shape in document.inline_shapes:
                    try:
                        # Access XML element
                        docPr = shape._inline.docPr
                        alt_text = docPr.get('descr')
                        if alt_text:
                            break
                    except AttributeError:
                        continue

                # Encode as base64 for storage
                import base64
                encoded = base64.b64encode(image_bytes).decode('utf-8')

                images.append(ImageNode(
                    src=f"data:image/{image_part.content_type.split('/')[1]};base64,{encoded}",
                    alt_text=alt_text or "",
                    width=shape.width if hasattr(shape, 'width') else None,
                    height=shape.height if hasattr(shape, 'height') else None
                ))

        return images

    def extract_from_html(self, html: str, base_url: str = "") -> list[ImageNode]:
        """Extract images from HTML with alt text."""
        from selectolax.parser import HTMLParser

        tree = HTMLParser(html)
        images: list[ImageNode] = []

        for img in tree.css('img'):
            src = img.attributes.get('src', '')

            # Resolve relative URLs
            if src.startswith('/'):
                src = base_url.rstrip('/') + src
            elif not src.startswith(('http://', 'https://', 'data:')):
                src = base_url.rstrip('/') + '/' + src

            # Extract caption from figure element
            caption = None
            parent = img.parent
            if parent and parent.tag == 'figure':
                figcaption = parent.css_first('figcaption')
                if figcaption:
                    caption = figcaption.text()

            images.append(ImageNode(
                src=src,
                alt_text=img.attributes.get('alt', ''),
                title=img.attributes.get('title'),
                width=int(w) if (w := img.attributes.get('width')) else None,
                height=int(h) if (h := img.attributes.get('height')) else None,
                caption=caption
            ))

        return images

    def detect_caption(self, image_node: ImageNode, context_nodes: list[DocumentNode]) -> Optional[str]:
        """
        Heuristically detect image caption from surrounding nodes.

        Rules:
        1. Next paragraph starting with "Figure X:"
        2. Previous paragraph in italics/smaller font
        3. Following text in <figcaption> tag
        """
        # Implementation depends on format-specific rules
        pass
```

**Edge Cases**:
- **Missing alt text**: Generate placeholder, flag for user review
- **Embedded SVG**: Extract as text (for SEO), preserve original
- **Image maps**: Extract all linked regions with alt text
- **Background images**: CSS backgrounds not extracted (note in docs)

### 5.3 Track Changes / Revision Marks

**Challenge**: python-docx has limited support for track changes.

**Solutions**:

1. **Direct XML parsing**:
```python
from docx import Document
from lxml import etree


def extract_revisions(docx_path: str) -> list[dict]:
    """Extract tracked changes from DOCX XML."""
    doc = Document(docx_path)
    revisions = []

    # Access underlying XML
    for paragraph in doc.paragraphs:
        for child in paragraph._element:
            # Look for <w:ins> (insertions) and <w:del> (deletions)
            if child.tag.endswith('ins'):
                revisions.append({
                    'type': 'insertion',
                    'author': child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}author'),
                    'date': child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}date'),
                    'text': ''.join(child.itertext())
                })
            elif child.tag.endswith('del'):
                revisions.append({
                    'type': 'deletion',
                    'author': child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}author'),
                    'date': child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}date'),
                    'text': ''.join(child.itertext())
                })

    return revisions
```

2. **Third-party libraries**:
   - **Python-Redlines**: Compare two DOCX files, generate redline document
   - **docxreviews2txt**: Extract review changes to plain text with HTML tags

**Recommendation**:
- **For SEO tool**: Accept all changes before processing (most stable)
- **For version tracking**: Store revision metadata separately
- **User option**: "Process with/without track changes"

**Edge Cases**:
- **Conflicting changes**: Multiple reviewers, overlapping edits
- **Comments on revisions**: Nested metadata
- **Moved text**: `<w:moveFrom>` and `<w:moveTo>` tags

### 5.4 Comments and Annotations

**Challenge**: Comments not part of main document flow.

**Solution**:

```python
def extract_comments(docx_path: str) -> dict[str, str]:
    """
    Extract comments from DOCX.

    Returns mapping of comment ID to comment text.
    """
    from docx import Document
    import zipfile
    from lxml import etree

    comments = {}

    # DOCX is a ZIP file
    with zipfile.ZipFile(docx_path, 'r') as docx:
        # Comments stored in word/comments.xml
        if 'word/comments.xml' in docx.namelist():
            comments_xml = docx.read('word/comments.xml')
            tree = etree.fromstring(comments_xml)

            # Namespace
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            for comment in tree.xpath('//w:comment', namespaces=ns):
                comment_id = comment.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id')
                author = comment.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}author')
                date = comment.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}date')
                text = ''.join(comment.itertext())

                comments[comment_id] = {
                    'author': author,
                    'date': date,
                    'text': text
                }

    return comments


# Link comments to text ranges
def link_comments_to_text(doc: Document, comments: dict) -> dict:
    """Find comment anchors in text and link to comment content."""
    comment_ranges = {}

    for paragraph in doc.paragraphs:
        for child in paragraph._element:
            if child.tag.endswith('commentRangeStart'):
                comment_id = child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id')
                comment_ranges[comment_id] = {'start': paragraph, 'text': ''}
            elif child.tag.endswith('commentRangeEnd'):
                comment_id = child.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id')
                if comment_id in comment_ranges:
                    comment_ranges[comment_id]['end'] = paragraph

    return comment_ranges
```

**Recommendation for SEO tool**:
- Store comments as separate metadata layer
- Don't include in token count or SEO analysis
- Optionally display in UI for context

### 5.5 Multi-Column Layouts

**Challenge**: DOCX sections can have multiple columns, affecting reading order.

**Detection**:
```python
def detect_columns(doc: Document) -> dict:
    """Detect multi-column sections."""
    from docx.oxml.ns import qn

    column_info = {}

    for section in doc.sections:
        # Access section properties
        sect_pr = section._sectPr
        cols = sect_pr.find(qn('w:cols'))

        if cols is not None:
            num_cols = cols.get(qn('w:num'), '1')
            column_info[section] = {
                'columns': int(num_cols),
                'space_between': cols.get(qn('w:space'))
            }

    return column_info
```

**Solution**: python-docx reads in document order, which may not match visual order in multi-column layouts.

**Recommendation**:
- Warn users about multi-column sections
- Consider manual reflow or convert to single-column for processing

### 5.6 Headers and Footers

**Solution**:

```python
def extract_headers_footers(doc: Document) -> dict:
    """Extract headers and footers from all sections."""
    hf_content = {
        'headers': [],
        'footers': []
    }

    for section in doc.sections:
        # Headers
        hf_content['headers'].append({
            'first_page': ''.join([p.text for p in section.first_page_header.paragraphs]),
            'even_pages': ''.join([p.text for p in section.even_page_header.paragraphs]),
            'odd_pages': ''.join([p.text for p in section.header.paragraphs])
        })

        # Footers
        hf_content['footers'].append({
            'first_page': ''.join([p.text for p in section.first_page_footer.paragraphs]),
            'even_pages': ''.join([p.text for p in section.even_page_footer.paragraphs]),
            'odd_pages': ''.join([p.text for p in section.footer.paragraphs])
        })

    return hf_content
```

**Recommendation**:
- Store headers/footers separately (not in main content)
- Extract page numbers, document metadata
- Don't include in SEO analysis (usually boilerplate)

---

## 6. Google Docs Integration

### 6.1 Approach Comparison

| Approach | Pros | Cons | Complexity |
|----------|------|------|------------|
| **Google Docs API** | Direct access, structured data | Requires OAuth, API quotas | High |
| **Export to DOCX** | Reuse existing DOCX parser | Extra conversion step | Medium |
| **Published HTML** | Simple HTTP fetch | Requires public link, limited metadata | Low |

### 6.2 Recommended Approach: Export to DOCX

**Rationale**:
1. Reuse robust DOCX parsing infrastructure
2. Avoid OAuth complexity for end users
3. Google Docs export preserves most formatting
4. Supports offline/batch processing

**Implementation**:

```python
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from io import BytesIO


class GoogleDocsImporter:
    """Import Google Docs via export to DOCX."""

    def __init__(self, credentials: Credentials):
        self.service = build('drive', 'v3', credentials=credentials)
        self.docs_service = build('docs', 'v1', credentials=credentials)

    def export_to_docx(self, doc_id: str) -> bytes:
        """
        Export Google Doc to DOCX format.

        Args:
            doc_id: Google Docs document ID (from URL)

        Returns:
            DOCX file as bytes
        """
        # Export using Drive API
        request = self.service.files().export_media(
            fileId=doc_id,
            mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

        docx_bytes = BytesIO()
        downloader = MediaIoBaseDownload(docx_bytes, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        docx_bytes.seek(0)
        return docx_bytes.read()

    def import_document(self, doc_id: str) -> Document:
        """
        Import Google Doc into normalized Document AST.

        Workflow:
        1. Export to DOCX
        2. Parse with python-docx
        3. Convert to Document AST
        """
        # Export
        docx_bytes = self.export_to_docx(doc_id)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp.write(docx_bytes)
            tmp_path = tmp.name

        try:
            # Parse as DOCX
            from .docx_parser import DOCXParser
            parser = DOCXParser()
            doc_ast = parser.parse(tmp_path)

            # Add Google Docs metadata
            doc_ast.metadata['source_type'] = 'google_docs'
            doc_ast.metadata['doc_id'] = doc_id

            return doc_ast
        finally:
            # Cleanup temp file
            import os
            os.unlink(tmp_path)
```

### 6.3 Direct API Approach (Alternative)

For advanced use cases requiring real-time collaboration features:

```python
def parse_via_api(doc_id: str, credentials: Credentials) -> Document:
    """Parse Google Doc using Docs API directly."""
    service = build('docs', 'v1', credentials=credentials)

    # Fetch document
    gdoc = service.documents().get(documentId=doc_id).execute()

    # Convert to Document AST
    root = DocumentNode(type=NodeType.DOCUMENT, children=[])

    for element in gdoc.get('body', {}).get('content', []):
        if 'paragraph' in element:
            para = element['paragraph']
            para_node = parse_paragraph(para)
            root.children.append(para_node)
        elif 'table' in element:
            table = element['table']
            table_node = parse_table(table)
            root.children.append(table_node)

    return Document(root=root, metadata={'source': 'google_docs', 'doc_id': doc_id})


def parse_paragraph(para: dict) -> ParagraphNode:
    """Convert Google Docs paragraph to ParagraphNode."""
    children = []

    for elem in para.get('elements', []):
        if 'textRun' in elem:
            text_run = elem['textRun']
            content = text_run.get('content', '')

            # Extract formatting
            style = text_run.get('textStyle', {})
            formatting = {
                'bold': style.get('bold', False),
                'italic': style.get('italic', False),
                'underline': style.get('underline', False)
            }

            children.append(TextNode(content=content, formatting=formatting))

    return ParagraphNode(children=children)
```

### 6.4 Handling Google Docs-Specific Features

**Suggestions/Comments**:
- Google Docs API provides `comments` and `suggestedChanges`
- Export to DOCX loses these features
- If needed, fetch via API separately:

```python
def get_comments(doc_id: str, credentials: Credentials) -> list[dict]:
    """Fetch comments from Google Doc."""
    service = build('drive', 'v3', credentials=credentials)

    comments = []
    page_token = None

    while True:
        response = service.comments().list(
            fileId=doc_id,
            fields='comments(content,author,createdTime,quotedFileContent),nextPageToken',
            pageToken=page_token
        ).execute()

        comments.extend(response.get('comments', []))
        page_token = response.get('nextPageToken')

        if not page_token:
            break

    return comments
```

**Recommendation**:
- **Default**: Export to DOCX (simplest, covers 90% of use cases)
- **Advanced**: Direct API access for real-time features
- **Hybrid**: Export for content, API for comments/suggestions

---

## 7. Implementation Specifications

### 7.1 Recommended Library Choices

| Format | Primary Library | Rationale | Fallback |
|--------|----------------|-----------|----------|
| **DOCX** | python-docx | Most mature, full control, active maintenance | docx2python (speed), Unstructured (unified) |
| **HTML** | trafilatura | Best content extraction (93.7% F1), automatic boilerplate removal | selectolax (custom parsing) |
| **Markdown** | mistune | Fastest, good enough CommonMark support | marko (GFM support) |
| **Google Docs** | Export to DOCX | Reuses existing infrastructure, simpler auth | Docs API (real-time features) |

### 7.2 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐     │
│  │  DOCX   │  │  HTML   │  │ Markdown │  │ Google Docs  │     │
│  └────┬────┘  └────┬────┘  └─────┬────┘  └──────┬───────┘     │
└───────┼────────────┼─────────────┼───────────────┼─────────────┘
        │            │             │               │
        ▼            ▼             ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FORMAT-SPECIFIC PARSERS                      │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ python- │  │trafila- │  │ mistune  │  │ Export +     │     │
│  │  docx   │  │  tura   │  │          │  │ python-docx  │     │
│  └────┬────┘  └────┬────┘  └─────┬────┘  └──────┬───────┘     │
└───────┼────────────┼─────────────┼───────────────┼─────────────┘
        │            │             │               │
        └────────────┴─────────────┴───────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  NORMALIZATION LAYER                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────┐         │
│  │  Convert to Document AST (Pydantic models)        │         │
│  │  - Unified node types across formats              │         │
│  │  - Metadata preservation                          │         │
│  │  - Source location tracking                       │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER                             │
│                                                                 │
│  ┌───────────────────────────────────────────────────┐         │
│  │  Pydantic validation                              │         │
│  │  - Schema compliance                              │         │
│  │  - Node relationship validation                   │         │
│  │  - Metadata completeness checks                   │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ENHANCEMENT LAYER                             │
│                                                                 │
│  ┌────────────────────┐  ┌────────────────────────┐            │
│  │ Image extraction   │  │ Comment extraction     │            │
│  │ & alt text         │  │ & linking              │            │
│  └────────────────────┘  └────────────────────────┘            │
│  ┌────────────────────┐  ┌────────────────────────┐            │
│  │ Table normalization│  │ Heading hierarchy      │            │
│  │ & flattening       │  │ validation             │            │
│  └────────────────────┘  └────────────────────────┘            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CHUNKING LAYER                              │
│                                                                 │
│  ┌───────────────────────────────────────────────────┐         │
│  │  Semantic Chunker                                 │         │
│  │  1. Heading-aware splitting                       │         │
│  │  2. Semantic boundary detection (embeddings)      │         │
│  │  3. Recursive character fallback                  │         │
│  │  4. Overlap addition                              │         │
│  └───────────────────────────────────────────────────┘         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐        │
│  │ JSON export │  │ Vector store │  │ SEO analysis   │        │
│  │             │  │ ingestion    │  │ pipeline       │        │
│  └─────────────┘  └──────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Error Handling Strategies

```python
from typing import Union
from enum import Enum


class ParseErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = "warning"  # Continue processing, log issue
    ERROR = "error"      # Skip problematic element, continue
    CRITICAL = "critical"  # Abort processing


class ParseError(Exception):
    """Base exception for parsing errors."""

    def __init__(
        self,
        message: str,
        severity: ParseErrorSeverity = ParseErrorSeverity.ERROR,
        context: Optional[dict] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}


class ParserResult(BaseModel):
    """Result of parsing operation."""
    document: Optional[Document] = None
    errors: list[ParseError] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if parsing succeeded."""
        return self.document is not None and not any(
            e.severity == ParseErrorSeverity.CRITICAL for e in self.errors
        )


class RobustParser:
    """Parser with comprehensive error handling."""

    def parse(self, file_path: str) -> ParserResult:
        """Parse file with error recovery."""
        result = ParserResult(
            metadata={'source_path': file_path}
        )

        try:
            # Detect format
            format_type = self._detect_format(file_path)

            # Select parser
            parser = self._get_parser(format_type)

            # Parse
            result.document = parser.parse(file_path)

        except FileNotFoundError:
            result.errors.append(ParseError(
                f"File not found: {file_path}",
                severity=ParseErrorSeverity.CRITICAL,
                context={'path': file_path}
            ))
        except PermissionError:
            result.errors.append(ParseError(
                f"Permission denied: {file_path}",
                severity=ParseErrorSeverity.CRITICAL,
                context={'path': file_path}
            ))
        except ParseError as e:
            result.errors.append(e)
            if e.severity != ParseErrorSeverity.CRITICAL:
                # Attempt partial recovery
                result.warnings.append(f"Partial parse after error: {e}")
        except Exception as e:
            result.errors.append(ParseError(
                f"Unexpected error: {str(e)}",
                severity=ParseErrorSeverity.CRITICAL,
                context={'exception_type': type(e).__name__}
            ))

        return result

    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension and magic bytes."""
        import mimetypes

        # Check extension
        mime_type, _ = mimetypes.guess_type(file_path)

        # Check magic bytes for confirmation
        with open(file_path, 'rb') as f:
            header = f.read(8)

        # DOCX: ZIP signature (PK..)
        if header[:2] == b'PK':
            return 'docx'
        # HTML: <html or <!DOCTYPE
        elif header[:5].lower() in (b'<html', b'<!doc'):
            return 'html'
        # Markdown: plain text (default)
        else:
            return 'markdown'
```

**Error Recovery Strategies**:

1. **Malformed tables**: Skip problematic rows, preserve remaining structure
2. **Missing images**: Create placeholder ImageNode with warning
3. **Invalid XML in DOCX**: Fall back to text extraction only
4. **Encoding issues**: Try multiple encodings (UTF-8, Latin-1, CP1252)
5. **Corrupted files**: Extract salvageable portions, report corruption

### 7.4 Performance Considerations

**Optimization Strategies**:

1. **Lazy loading**: Don't parse entire document upfront
   ```python
   class LazyDocument:
       """Document with lazy node loading."""

       def __init__(self, source_path: str):
           self.source_path = source_path
           self._cache: dict[str, DocumentNode] = {}

       def get_node(self, node_id: str) -> DocumentNode:
           """Load node on demand."""
           if node_id not in self._cache:
               self._cache[node_id] = self._load_node(node_id)
           return self._cache[node_id]
   ```

2. **Streaming processing**: For very large documents
   ```python
   def stream_paragraphs(docx_path: str) -> Iterator[ParagraphNode]:
       """Yield paragraphs one at a time without loading entire doc."""
       doc = Document(docx_path)
       for para in doc.paragraphs:
           yield parse_paragraph(para)
   ```

3. **Parallel processing**: Multiple documents concurrently
   ```python
   from concurrent.futures import ProcessPoolExecutor

   def batch_parse(file_paths: list[str]) -> list[Document]:
       """Parse multiple files in parallel."""
       with ProcessPoolExecutor() as executor:
           results = executor.map(parse_document, file_paths)
       return list(results)
   ```

4. **Caching**: Memoize expensive operations
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def embed_sentence(sentence: str) -> np.ndarray:
       """Cache sentence embeddings."""
       return embedder.encode(sentence)
   ```

**Performance Targets**:

| Document Size | Target Time | Memory Limit |
|---------------|-------------|--------------|
| < 50 KB | < 1 second | < 50 MB |
| 50-500 KB | < 5 seconds | < 200 MB |
| 500 KB - 5 MB | < 30 seconds | < 500 MB |
| > 5 MB | Streaming mode | < 1 GB |

### 7.5 Testing Strategy

```python
import pytest
from pathlib import Path


class TestDocumentParsing:
    """Comprehensive test suite for document parsing."""

    @pytest.fixture
    def test_files(self) -> Path:
        """Path to test fixtures."""
        return Path(__file__).parent / "fixtures"

    def test_docx_basic_structure(self, test_files):
        """Test parsing DOCX with headings, paragraphs, lists."""
        parser = DOCXParser()
        doc = parser.parse(test_files / "basic_structure.docx")

        assert doc.root.type == NodeType.DOCUMENT
        assert len(doc.root.children) > 0

        # Check heading hierarchy
        headings = [n for n in doc.root.children if n.type == NodeType.HEADING]
        assert len(headings) >= 3
        assert headings[0].level == 1

    def test_docx_nested_tables(self, test_files):
        """Test handling of nested tables."""
        parser = DOCXParser()
        doc = parser.parse(test_files / "nested_tables.docx")

        tables = [n for n in doc.root.children if n.type == NodeType.TABLE]
        assert len(tables) > 0

        # Check for nested table in cell
        first_table = tables[0]
        cell_with_table = None
        for row in first_table.children:
            for cell in row.children:
                if any(c.type == NodeType.TABLE for c in cell.children):
                    cell_with_table = cell
                    break

        assert cell_with_table is not None

    def test_html_content_extraction(self, test_files):
        """Test HTML main content extraction with trafilatura."""
        parser = HTMLParser()
        doc = parser.parse(test_files / "article_with_nav.html")

        # Should extract main content, exclude navigation
        text = extract_text(doc)
        assert "Navigation" not in text  # Nav boilerplate removed
        assert "Article Content" in text  # Main content preserved

    def test_chunking_heading_aware(self, test_files):
        """Test heading-aware chunking."""
        parser = DOCXParser()
        doc = parser.parse(test_files / "long_document.docx")

        chunker = SemanticChunker(target_tokens=500, max_tokens=1000)
        chunks = chunker.chunk_document(doc)

        # Verify chunks respect heading boundaries
        for chunk in chunks:
            assert chunk.token_count <= 1000
            assert len(chunk.heading_path) > 0  # Each chunk has heading context

    def test_image_alt_text_extraction(self, test_files):
        """Test image extraction with alt text."""
        parser = DOCXParser()
        doc = parser.parse(test_files / "document_with_images.docx")

        images = [n for n in doc.root.children if n.type == NodeType.IMAGE]
        assert len(images) > 0
        assert images[0].alt_text is not None

    @pytest.mark.parametrize("format_type,file_name", [
        ("docx", "test.docx"),
        ("html", "test.html"),
        ("markdown", "test.md")
    ])
    def test_cross_format_consistency(self, test_files, format_type, file_name):
        """Test same content parsed from different formats produces similar AST."""
        parser = get_parser(format_type)
        doc = parser.parse(test_files / file_name)

        # All should have same basic structure
        assert doc.root.type == NodeType.DOCUMENT
        headings = [n for n in doc.root.children if n.type == NodeType.HEADING]
        assert len(headings) >= 2  # All test files have at least 2 headings

    def test_error_recovery_malformed_docx(self, test_files):
        """Test parser handles corrupted DOCX gracefully."""
        parser = RobustParser()
        result = parser.parse(test_files / "corrupted.docx")

        assert not result.success
        assert any(e.severity == ParseErrorSeverity.CRITICAL for e in result.errors)

    def test_performance_large_document(self, test_files, benchmark):
        """Benchmark parsing performance on large documents."""
        parser = DOCXParser()

        # Benchmark should complete in < 5 seconds for 500KB file
        doc = benchmark(parser.parse, test_files / "large_document.docx")
        assert doc is not None
```

**Test Coverage Requirements**:
- **Unit tests**: 90%+ coverage of parsing logic
- **Integration tests**: End-to-end pipeline for each format
- **Edge case tests**: All scenarios in Section 5
- **Performance tests**: Verify targets in Section 7.4
- **Regression tests**: Lock down fixed bugs

---

## 8. Success Metrics

### 8.1 Structure Preservation Accuracy

**Metric**: Percentage of document structure elements correctly preserved.

**Measurement**:
```python
def calculate_structure_accuracy(original_doc: Document, parsed_doc: Document) -> float:
    """
    Compare parsed document to ground truth.

    Checks:
    - Heading count and levels
    - Paragraph count
    - List structure (ordered/unordered, nesting)
    - Table dimensions
    - Image count
    """
    score = 0.0
    total = 0.0

    # Heading accuracy
    orig_headings = extract_headings(original_doc)
    parsed_headings = extract_headings(parsed_doc)
    score += len(set(orig_headings) & set(parsed_headings))
    total += len(orig_headings)

    # Paragraph accuracy
    orig_para_count = count_nodes(original_doc, NodeType.PARAGRAPH)
    parsed_para_count = count_nodes(parsed_doc, NodeType.PARAGRAPH)
    score += min(orig_para_count, parsed_para_count)
    total += orig_para_count

    # Table accuracy
    orig_tables = extract_tables(original_doc)
    parsed_tables = extract_tables(parsed_doc)
    for orig, parsed in zip(orig_tables, parsed_tables):
        if orig.rows == parsed.rows and orig.columns == parsed.columns:
            score += 1
    total += len(orig_tables)

    return score / total if total > 0 else 0.0
```

**Target**: >95% accuracy for well-formed documents

### 8.2 Processing Speed Benchmarks

**Benchmarks**:

```python
import time
from statistics import mean, stdev


def benchmark_parser(parser: Parser, test_files: list[str], iterations: int = 10) -> dict:
    """Benchmark parser performance."""
    results = {
        'file_sizes': [],
        'parse_times': [],
        'tokens_per_second': []
    }

    for file_path in test_files:
        file_size = Path(file_path).stat().st_size
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            doc = parser.parse(file_path)
            end = time.perf_counter()
            times.append(end - start)

        avg_time = mean(times)
        std_time = stdev(times)

        # Estimate tokens
        text = extract_text(doc)
        token_count = len(text.split())  # Rough estimate

        results['file_sizes'].append(file_size)
        results['parse_times'].append(avg_time)
        results['tokens_per_second'].append(token_count / avg_time)

        print(f"{file_path}: {avg_time:.3f}s ± {std_time:.3f}s ({token_count/avg_time:.0f} tokens/s)")

    return results
```

**Targets**:

| Operation | Target |
|-----------|--------|
| DOCX parsing (50KB) | < 500ms |
| HTML content extraction | < 200ms |
| Markdown parsing | < 100ms |
| Chunking (10,000 tokens) | < 2s |
| End-to-end (DOCX -> chunks) | < 5s for 500KB |

### 8.3 Memory Usage Constraints

**Monitoring**:

```python
import tracemalloc
import psutil


def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage of function."""
    tracemalloc.start()
    process = psutil.Process()

    # Baseline
    baseline = process.memory_info().rss / 1024 / 1024  # MB

    # Execute
    result = func(*args, **kwargs)

    # Peak
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024

    print(f"Peak memory: {peak_mb:.2f} MB (baseline: {baseline:.2f} MB)")

    return result, peak_mb
```

**Targets**:

| Document Size | Max Memory |
|---------------|------------|
| < 1 MB | < 100 MB |
| 1-10 MB | < 500 MB |
| > 10 MB | Streaming mode (< 1 GB) |

### 8.4 Test Coverage Requirements

**Coverage Targets**:

```bash
# Install coverage tools
pip install pytest-cov coverage

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Targets:
# - Overall: >90%
# - Critical paths (parsers): >95%
# - Error handling: >85%
# - Edge cases: 100% of documented scenarios
```

**Required Test Categories**:

1. **Unit Tests**: Each parser function
2. **Integration Tests**: End-to-end pipelines
3. **Performance Tests**: Speed benchmarks
4. **Edge Case Tests**: All scenarios from Section 5
5. **Regression Tests**: Previously fixed bugs
6. **Property-Based Tests**: Using Hypothesis for fuzzing

```python
from hypothesis import given, strategies as st


@given(st.text(min_size=100, max_size=10000))
def test_chunker_never_exceeds_max_tokens(text):
    """Property: chunker never produces chunks > max_tokens."""
    chunker = SemanticChunker(max_tokens=1000)

    # Create simple document
    doc = Document(root=DocumentNode(
        type=NodeType.DOCUMENT,
        children=[ParagraphNode(children=[TextNode(content=text)])]
    ))

    chunks = chunker.chunk_document(doc)

    # Property: all chunks within limit
    for chunk in chunks:
        assert chunk.token_count <= 1000
```

---

## 9. Sources and References

### Research Sources

**DOCX Parsing:**
- [Technical Comparison — Python Libraries for Document Parsing](https://medium.com/@hchenna/technical-comparison-python-libraries-for-document-parsing-318d2c89c44e)
- [NLP-docx2python vs python-docx Tests](https://www.kaggle.com/code/toddgardiner/nlp-docx2python-vs-python-docx-tests)
- [docx2python PyPI](https://pypi.org/project/docx2python/)
- [mammoth PyPI](https://pypi.org/project/mammoth/)
- [python-docx Documentation](https://python-docx.readthedocs.io/)

**HTML Parsing:**
- [Efficient Web Scraping: lxml, BeautifulSoup, and Selectolax](https://medium.com/@yahyamrafe202/in-depth-comparison-of-web-scraping-parsers-lxml-beautifulsoup-and-selectolax-4f268ddea8df)
- [BeautifulSoup vs lxml Performance Comparison](https://dev.to/dmitriiweb/beautifulsoup-vs-lxml-a-practical-performance-comparison-1l0a)
- [Extracting text from HTML: a very fast approach](https://rushter.com/blog/python-fast-html-parser/)
- [Trafilatura Documentation](https://trafilatura.readthedocs.io/)

**Semantic Chunking:**
- [Chunking Strategies for LLM Applications | Pinecone](https://www.pinecone.io/learn/chunking-strategies/)
- [Document Chunking for RAG: 9 Strategies Tested](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)
- [LLM Chunking | Redis](https://redis.io/blog/llm-chunking/)
- [Breaking up is hard to do: Chunking in RAG applications](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)

**Token Counting:**
- [Counting tokens at scale using tiktoken](https://www.dsdev.in/counting-tokens-at-scale-using-tiktoken)
- [Integrate Tiktoken in Python Applications](https://www.cloudproinc.com.au/index.php/2025/09/03/integrate-tiktoken-in-python-applications/)
- [semchunk GitHub](https://github.com/isaacus-dev/semchunk)

**Document Normalization:**
- [Universal AST Schema Framework](https://www.emergentmind.com/topics/universal-abstract-syntax-tree-ast-schema)
- [MLCPD: Multi-Language Code Parsing Dataset](https://arxiv.org/html/2510.16357)

**Google Docs:**
- [Python quickstart | Google Docs API](https://developers.google.com/workspace/docs/api/quickstart/python)
- [How to Get Document Texts with Google Docs API](https://endgrate.com/blog/how-to-get-document-texts-with-the-google-docs-api-in-python)

**Track Changes:**
- [revisions/track changes · Issue #340](https://github.com/python-openxml/python-docx/issues/340)
- [Python-Redlines GitHub](https://github.com/JSv4/Python-Redlines)
- [docxreviews2txt GitHub](https://github.com/alanlivio/docxreviews2txt)

**Pydantic & Validation:**
- [Pydantic Models Documentation](https://docs.pydantic.dev/latest/concepts/models/)
- [Pydantic: Simplifying Data Validation in Python](https://realpython.com/python-pydantic/)

**Markdown Parsing:**
- [Mistune GitHub](https://github.com/lepture/mistune)
- [Marko Documentation](https://marko-py.readthedocs.io/)
- [markdown-it-py PyPI](https://pypi.org/project/markdown-it-py/)

**Nested Tables & Edge Cases:**
- [Working with Tables — python-docx](https://python-docx.readthedocs.io/en/latest/user/tables.html)
- [Create Table in Word DOCX in Python](https://blog.aspose.com/words/create-table-in-word-using-python/)

**Image Extraction:**
- [Automating Image Extraction from DOCX](https://dev.to/allen_yang_f905170c5a197b/automating-image-extraction-from-docx-files-with-python-533f)
- [Add support for image title and alt text](https://github.com/python-openxml/python-docx/pull/227/files)

**Unstructured Library:**
- [Unstructured PyPI](https://pypi.org/project/unstructured/)
- [Build an unstructured data pipeline for RAG](https://docs.databricks.com/aws/en/generative-ai/tutorials/ai-cookbook/quality-data-pipeline-rag)
- [Parsing Documents: An Introduction to Unstructured](https://www.tetranyde.com/blog/unstructured/)

---

## Appendix A: Code Examples Repository Structure

```
content-ingestion/
├── src/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── base.py              # Base parser protocol
│   │   ├── docx_parser.py       # DOCX parsing with python-docx
│   │   ├── html_parser.py       # HTML with trafilatura
│   │   ├── markdown_parser.py   # Markdown with mistune
│   │   └── gdocs_parser.py      # Google Docs integration
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── nodes.py             # Pydantic node models
│   │   └── document.py          # Document AST
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── semantic.py          # Semantic chunker
│   │   └── strategies.py        # Chunking strategies
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── token_counter.py     # Token counting
│   │   └── image_extractor.py   # Image handling
│   └── pipeline.py              # End-to-end pipeline
├── tests/
│   ├── fixtures/                # Test documents
│   ├── test_parsers.py
│   ├── test_chunking.py
│   └── test_edge_cases.py
├── pyproject.toml               # Modern Python packaging
└── README.md
```

---

## Appendix B: Installation and Setup

```toml
# pyproject.toml
[project]
name = "content-ingestion"
version = "1.0.0"
description = "Document ingestion and normalization for SEO + AI tools"
requires-python = ">=3.12"
dependencies = [
    "python-docx>=1.1.0",
    "trafilatura>=1.12.0",
    "selectolax>=0.3.21",
    "mistune>=3.0.0",
    "pydantic>=2.9.0",
    "tiktoken>=0.8.0",
    "sentence-transformers>=3.0.0",
    "numpy>=2.0.0",
    "google-api-python-client>=2.150.0",
    "google-auth>=2.35.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=5.0.0",
    "pytest-benchmark>=4.0.0",
    "hypothesis>=6.112.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
```

```bash
# Install using uv (fastest)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run tests
pytest

# Check types
mypy src/

# Format code
ruff format src/
ruff check src/ --fix
```

---

**END OF DOCUMENT**

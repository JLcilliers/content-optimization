# Section 1: DOCX Parsing & Structure Preservation

**Technical Specification**
**Version:** 1.0
**Date:** 2026-01-16
**Module:** `src/ingestion/`

---

## Executive Summary

The DOCX parsing module serves as the foundation for the SEO content optimization tool's critical diffing and highlighting capabilities. This module must extract content from Microsoft Word .docx files while preserving exact structural hierarchy, formatting metadata, and position information necessary for precise change detection.

The core challenge is maintaining a **perfect bidirectional mapping** between the original document structure and an internal representation (DocumentAST) that enables character-level diffing while supporting lossless reconstruction of the output DOCX. Any deviation in structure preservation directly impacts the tool's ability to accurately highlight new content—the primary user-facing value proposition.

This specification defines the data models, parsing algorithms, and position tracking mechanisms required to achieve zero-loss document ingestion. The design prioritizes **immutability of the OriginalSnapshot** (enabling reliable diffing), **comprehensive formatting capture** (enabling faithful reconstruction), and **robust edge case handling** (ensuring production reliability across diverse document structures).

---

## 1. python-docx Capabilities Analysis

### 1.1 Document Structure Access

python-docx (v1.1.2+) provides a hierarchical object model for accessing DOCX content:

**Document-Level Access:**
```python
from docx import Document

doc = Document('input.docx')
# Access top-level elements in order
for element in doc.element.body:
    # element is an lxml element representing paragraph, table, etc.
    pass
```

**Key Capabilities:**
- Iterate through document body in sequential order
- Access paragraphs via `doc.paragraphs`
- Access tables via `doc.tables`
- Access sections via `doc.sections`
- Access core properties (metadata) via `doc.core_properties`

**Limitations:**
- No direct access to headers/footers without section iteration
- Footnotes and endnotes require deeper XML parsing
- Document-level comments require `doc.element` XML traversal

### 1.2 Paragraph and Run Objects

**Paragraph Structure:**
```python
for paragraph in doc.paragraphs:
    # Paragraph properties
    paragraph.text              # Combined text (loses formatting boundaries)
    paragraph.style             # Style object (e.g., 'Heading 1', 'Normal')
    paragraph.alignment         # WD_ALIGN_PARAGRAPH enum
    paragraph.paragraph_format  # ParagraphFormat object

    # Run iteration (preserves formatting boundaries)
    for run in paragraph.runs:
        run.text               # Text with consistent formatting
        run.bold               # Boolean or None
        run.italic             # Boolean or None
        run.font               # Font object
```

**Run-Level Granularity:**
- A "run" is the atomic unit of text with consistent formatting
- Paragraph text is split across multiple runs when formatting changes
- Example: "Hello **world**" = 2 runs: ["Hello ", {"text": "world", "bold": True}]

**Critical Insight:**
```python
# BAD: Loses formatting boundaries
text = paragraph.text  # "Hello world"

# GOOD: Preserves formatting structure
runs = [{"text": run.text, "bold": run.bold} for run in paragraph.runs]
# [{"text": "Hello ", "bold": False}, {"text": "world", "bold": True}]
```

### 1.3 Style and Formatting Access

**Paragraph-Level Formatting:**
```python
pf = paragraph.paragraph_format
pf.left_indent          # Length object (e.g., Pt(36))
pf.right_indent
pf.first_line_indent    # Hanging indent or first line indent
pf.line_spacing         # Float or None
pf.space_before         # Length object
pf.space_after          # Length object
pf.keep_together        # Boolean
pf.keep_with_next       # Boolean
pf.page_break_before    # Boolean
```

**Run-Level Formatting:**
```python
font = run.font
font.name               # Font family name (e.g., 'Calibri')
font.size               # Length object (e.g., Pt(11))
font.bold               # Boolean or None
font.italic             # Boolean or None
font.underline          # Boolean or WD_UNDERLINE enum
font.strike             # Boolean
font.color.rgb          # RGBColor object
font.highlight_color    # WD_COLOR_INDEX enum (e.g., YELLOW)
font.subscript          # Boolean
font.superscript        # Boolean
```

**Style Objects:**
```python
style = paragraph.style
style.name              # String (e.g., 'Heading 1')
style.style_id          # Internal ID (e.g., 'Heading1')
style.type              # WD_STYLE_TYPE enum (PARAGRAPH, CHARACTER, etc.)
style.base_style        # Parent style (inheritance chain)
```

### 1.4 Lists (Bulleted and Numbered)

**List Detection:**
```python
# Lists are identified via paragraph numbering format
paragraph.style.name    # May be 'List Bullet' or 'List Number'

# More reliable: check numbering format
from docx.enum.text import WD_LIST_NUMBER_STYLE

if paragraph._element.pPr is not None:
    numPr = paragraph._element.pPr.numPr
    if numPr is not None:
        ilvl = numPr.ilvl  # List level (0 = top level, 1 = nested, etc.)
        numId = numPr.numId  # Numbering definition ID
```

**List Limitations:**
- python-docx does not provide high-level list abstractions
- List detection requires XML element inspection
- Nesting level must be extracted from `ilvl` (indentation level)
- List type (bullet vs. number) requires numbering.xml parsing

**Workaround Strategy:**
```python
def get_list_info(paragraph):
    """Extract list metadata from paragraph XML"""
    pPr = paragraph._element.pPr
    if pPr is None:
        return None

    numPr = pPr.numPr
    if numPr is None:
        return None

    return {
        'level': int(numPr.ilvl.val) if numPr.ilvl is not None else 0,
        'num_id': int(numPr.numId.val) if numPr.numId is not None else None,
        'is_list': True
    }
```

### 1.5 Tables

**Table Access:**
```python
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            # Each cell contains paragraphs
            for paragraph in cell.paragraphs:
                # Process paragraph content
                pass
```

**Table Properties:**
```python
table.style             # Table style name
table.alignment         # WD_TABLE_ALIGNMENT enum
table.autofit           # Boolean
len(table.rows)         # Row count
len(table.columns)      # Column count

# Cell-level
cell.text               # Combined text (loses formatting)
cell.width              # Column width
cell.vertical_alignment # WD_CELL_VERTICAL_ALIGNMENT
```

**Merged Cells:**
```python
# Merged cells share the same cell object
# Detect via cell identity comparison
if table.cell(0, 0) is table.cell(0, 1):
    # Cells are merged
    pass
```

### 1.6 Headings

**Heading Detection:**
```python
# Method 1: Via style name
if paragraph.style.name.startswith('Heading'):
    level = int(paragraph.style.name.split()[-1])  # e.g., "Heading 2" -> 2

# Method 2: Via outline level
outline_level = paragraph.paragraph_format.outline_level
if outline_level is not None:
    level = outline_level + 1  # 0-indexed, so add 1
```

### 1.7 Hyperlinks

**Limitation:** python-docx does not expose hyperlinks directly.

**Workaround via XML:**
```python
from docx.oxml import parse_xml
from docx.oxml.ns import qn

def extract_hyperlinks(paragraph):
    """Extract hyperlinks from paragraph XML"""
    hyperlinks = []

    for hyperlink in paragraph._element.findall('.//w:hyperlink', namespaces={
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    }):
        # Extract relationship ID
        r_id = hyperlink.get(qn('r:id'))
        if r_id:
            # Resolve URL from document relationships
            url = paragraph.part.rels[r_id].target_ref

            # Extract hyperlink text
            text = ''.join(node.text for node in hyperlink.findall('.//w:t', namespaces={
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            }))

            hyperlinks.append({'text': text, 'url': url})

    return hyperlinks
```

### 1.8 Images

**Image Access:**
```python
from docx.oxml import parse_xml

def extract_images(paragraph):
    """Extract image metadata from paragraph"""
    images = []

    # Images are stored in run elements
    for run in paragraph.runs:
        # Check for drawing elements
        drawings = run._element.findall('.//w:drawing', namespaces={
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        })

        for drawing in drawings:
            # Extract relationship ID
            blip = drawing.find('.//a:blip', namespaces={
                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'
            })

            if blip is not None:
                r_id = blip.get(qn('r:embed'))
                if r_id:
                    image_part = run.part.rels[r_id].target_part
                    images.append({
                        'r_id': r_id,
                        'filename': image_part.partname.split('/')[-1],
                        'content_type': image_part.content_type
                    })

    return images
```

**Alt Text Extraction:**
```python
# Alt text stored in docPr element
doc_pr = drawing.find('.//wp:docPr', namespaces={
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
})

if doc_pr is not None:
    alt_text = doc_pr.get('descr', '')  # Description attribute
```

### 1.9 Limitations Summary

| Feature | python-docx Support | Workaround Required |
|---------|---------------------|---------------------|
| Paragraphs | Native | No |
| Runs (formatting) | Native | No |
| Tables | Native | No |
| Headings | Via styles | No |
| Lists | No | XML parsing |
| Hyperlinks | No | XML parsing |
| Images | Partial | XML parsing for metadata |
| Comments | No | XML parsing |
| Tracked Changes | No | XML parsing |
| Footnotes/Endnotes | No | XML parsing |
| Headers/Footers | Via sections | Iteration required |
| Text boxes | No | XML parsing |

---

## 2. Internal Data Model Specification

### 2.1 DocumentAST Class

**Purpose:** In-memory representation of document structure enabling manipulation and diffing.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

@dataclass
class DocumentAST:
    """
    Abstract Syntax Tree representation of a DOCX document.

    Attributes:
        id: Unique document identifier (UUID4)
        nodes: Ordered list of top-level content nodes
        metadata: Document metadata (title, author, etc.)
        created_at: Timestamp of AST creation
        source_hash: SHA-256 hash of source DOCX bytes
    """
    id: str
    nodes: List['ContentNode']
    metadata: 'DocumentMetadata'
    created_at: datetime
    source_hash: str

    def find_node(self, node_id: str) -> Optional['ContentNode']:
        """Recursively find node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
            if node.children:
                found = self._find_in_children(node, node_id)
                if found:
                    return found
        return None

    def _find_in_children(self, parent: 'ContentNode', node_id: str) -> Optional['ContentNode']:
        for child in parent.children:
            if child.id == node_id:
                return child
            if child.children:
                found = self._find_in_children(child, node_id)
                if found:
                    return found
        return None

    def get_all_nodes(self) -> List['ContentNode']:
        """Flatten tree to list (depth-first traversal)"""
        result = []
        for node in self.nodes:
            result.append(node)
            result.extend(self._flatten_children(node))
        return result

    def _flatten_children(self, parent: 'ContentNode') -> List['ContentNode']:
        result = []
        for child in parent.children:
            result.append(child)
            if child.children:
                result.extend(self._flatten_children(child))
        return result

@dataclass
class DocumentMetadata:
    """Document-level metadata from core properties"""
    title: Optional[str] = None
    subject: Optional[str] = None
    author: Optional[str] = None
    keywords: Optional[str] = None
    comments: Optional[str] = None
    last_modified_by: Optional[str] = None
    revision: Optional[int] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
```

### 2.2 ContentNode Types

```python
class NodeType(Enum):
    """Content node types matching DOCX elements"""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    IMAGE = "image"
    HYPERLINK = "hyperlink"

@dataclass
class ContentNode:
    """
    Single element in document structure.

    Attributes:
        id: Unique position identifier (e.g., "p0", "h1", "t0r2c1")
        type: Node type from NodeType enum
        content: Text content (plain text, no formatting)
        runs: List of formatted text runs (preserves formatting boundaries)
        formatting: Node-level formatting (paragraph format, style, etc.)
        position: Position information for diffing
        children: Child nodes (e.g., table contains rows, row contains cells)
        metadata: Type-specific metadata (e.g., heading level, list nesting)
    """
    id: str
    type: NodeType
    content: str
    runs: List['TextRun'] = field(default_factory=list)
    formatting: 'FormattingInfo' = field(default_factory=lambda: FormattingInfo())
    position: 'PositionInfo' = field(default_factory=lambda: PositionInfo())
    children: List['ContentNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_plain_text(self) -> str:
        """Get all text including children (recursive)"""
        text = self.content
        for child in self.children:
            text += child.get_plain_text()
        return text

    def get_character_count(self) -> int:
        """Total characters including children"""
        return len(self.get_plain_text())

@dataclass
class TextRun:
    """
    Formatted text segment (atomic unit with consistent formatting).

    Corresponds to python-docx Run object.
    """
    text: str
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None  # Points
    font_color: Optional[str] = None  # RGB hex (e.g., "FF0000")
    highlight_color: Optional[str] = None  # Color name (e.g., "YELLOW")
    hyperlink_url: Optional[str] = None
```

### 2.3 PositionInfo

```python
@dataclass
class PositionInfo:
    """
    Position tracking for precise diffing.

    Attributes:
        document_index: Index in flattened document (0-based)
        parent_id: ID of parent node (None for top-level)
        sibling_index: Index among siblings (0-based)
        depth: Nesting depth (0 = top-level)
        character_offset: Character offset from document start
        original_xml_path: XPath to original XML element (debugging)
    """
    document_index: int = 0
    parent_id: Optional[str] = None
    sibling_index: int = 0
    depth: int = 0
    character_offset: int = 0
    original_xml_path: Optional[str] = None
```

### 2.4 FormattingInfo

```python
@dataclass
class ParagraphFormat:
    """Paragraph-level formatting"""
    style_name: Optional[str] = None  # e.g., "Heading 1", "Normal"
    style_id: Optional[str] = None
    alignment: Optional[str] = None  # LEFT, CENTER, RIGHT, JUSTIFY
    left_indent: Optional[float] = None  # Points
    right_indent: Optional[float] = None
    first_line_indent: Optional[float] = None
    space_before: Optional[float] = None  # Points
    space_after: Optional[float] = None
    line_spacing: Optional[float] = None
    keep_together: bool = False
    keep_with_next: bool = False
    page_break_before: bool = False

@dataclass
class TableFormat:
    """Table-level formatting"""
    style_name: Optional[str] = None
    alignment: Optional[str] = None
    autofit: bool = False
    row_count: int = 0
    column_count: int = 0

@dataclass
class CellFormat:
    """Table cell formatting"""
    width: Optional[float] = None  # Points
    vertical_alignment: Optional[str] = None  # TOP, CENTER, BOTTOM
    background_color: Optional[str] = None  # RGB hex

@dataclass
class FormattingInfo:
    """
    Comprehensive formatting metadata for content nodes.

    Type-specific formatting is stored in corresponding attributes.
    """
    paragraph: Optional[ParagraphFormat] = None
    table: Optional[TableFormat] = None
    cell: Optional[CellFormat] = None
```

### 2.5 Node Metadata Examples

```python
# Heading metadata
{
    'level': 2,  # H2
    'outline_level': 1  # 0-indexed
}

# List metadata
{
    'is_list': True,
    'list_level': 0,  # Top-level list item
    'num_id': 1,  # Numbering definition ID
    'is_ordered': False  # Bullet list
}

# Image metadata
{
    'filename': 'image1.png',
    'content_type': 'image/png',
    'relationship_id': 'rId5',
    'alt_text': 'Company logo',
    'width': 300,  # Pixels
    'height': 200
}

# Hyperlink metadata
{
    'url': 'https://example.com',
    'relationship_id': 'rId7'
}
```

---

## 3. Parsing Algorithm

### 3.1 High-Level Parsing Flow

```python
def parse_docx(docx_path: str) -> tuple[DocumentAST, OriginalSnapshot]:
    """
    Parse DOCX file into DocumentAST and create immutable snapshot.

    Args:
        docx_path: Path to source DOCX file

    Returns:
        Tuple of (DocumentAST, OriginalSnapshot)

    Raises:
        ValueError: If DOCX is corrupted or unsupported
        FileNotFoundError: If docx_path does not exist
    """
    # Step 1: Load DOCX
    doc = Document(docx_path)

    # Step 2: Calculate source hash
    source_hash = calculate_file_hash(docx_path)

    # Step 3: Extract metadata
    metadata = extract_metadata(doc)

    # Step 4: Parse body elements in order
    nodes = []
    character_offset = 0
    document_index = 0

    for element in doc.element.body:
        node, chars_added = parse_element(
            element=element,
            document_index=document_index,
            character_offset=character_offset,
            depth=0,
            parent_id=None
        )

        if node:
            nodes.append(node)
            character_offset += chars_added
            document_index += 1

    # Step 5: Create DocumentAST
    ast = DocumentAST(
        id=generate_uuid(),
        nodes=nodes,
        metadata=metadata,
        created_at=datetime.now(),
        source_hash=source_hash
    )

    # Step 6: Create OriginalSnapshot
    snapshot = create_snapshot(ast)

    return ast, snapshot
```

### 3.2 Element Parsing Dispatcher

```python
def parse_element(
    element: Any,
    document_index: int,
    character_offset: int,
    depth: int,
    parent_id: Optional[str]
) -> tuple[Optional[ContentNode], int]:
    """
    Parse single element from DOCX body.

    Returns:
        Tuple of (ContentNode or None, character_count)
    """
    tag = element.tag.split('}')[-1]  # Remove namespace

    # Dispatch to appropriate handler
    if tag == 'p':
        return parse_paragraph(element, document_index, character_offset, depth, parent_id)
    elif tag == 'tbl':
        return parse_table(element, document_index, character_offset, depth, parent_id)
    else:
        # Unknown element type, log warning
        logger.warning(f"Skipping unknown element type: {tag}")
        return None, 0
```

### 3.3 Paragraph Parsing

```python
def parse_paragraph(
    element: Any,
    document_index: int,
    character_offset: int,
    depth: int,
    parent_id: Optional[str]
) -> tuple[ContentNode, int]:
    """
    Parse paragraph element into ContentNode.

    Handles:
    - Headings (via style detection)
    - Lists (via numbering format)
    - Regular paragraphs
    - Images within paragraphs
    """
    # Wrap in python-docx Paragraph object
    from docx.text.paragraph import Paragraph
    paragraph = Paragraph(element, None)

    # Determine node type
    node_type, node_metadata = determine_paragraph_type(paragraph)

    # Generate position ID
    position_id = generate_position_id(node_type, document_index)

    # Extract runs (preserves formatting boundaries)
    runs = []
    content = ""

    for run in paragraph.runs:
        text_run = TextRun(
            text=run.text,
            bold=run.bold,
            italic=run.italic,
            underline=run.underline,
            font_name=run.font.name,
            font_size=run.font.size.pt if run.font.size else None,
            font_color=rgb_to_hex(run.font.color.rgb) if run.font.color.rgb else None,
            highlight_color=get_highlight_color_name(run.font.highlight_color)
        )
        runs.append(text_run)
        content += run.text

    # Extract paragraph formatting
    formatting = FormattingInfo(
        paragraph=extract_paragraph_format(paragraph)
    )

    # Create position info
    position = PositionInfo(
        document_index=document_index,
        parent_id=parent_id,
        sibling_index=0,  # Set by parent
        depth=depth,
        character_offset=character_offset,
        original_xml_path=get_xml_path(element)
    )

    # Create node
    node = ContentNode(
        id=position_id,
        type=node_type,
        content=content,
        runs=runs,
        formatting=formatting,
        position=position,
        children=[],
        metadata=node_metadata
    )

    return node, len(content)
```

### 3.4 Heading Detection

```python
def determine_paragraph_type(paragraph) -> tuple[NodeType, Dict[str, Any]]:
    """
    Determine if paragraph is heading, list item, or regular paragraph.

    Returns:
        Tuple of (NodeType, metadata_dict)
    """
    # Check if heading
    if paragraph.style.name.startswith('Heading'):
        level = int(paragraph.style.name.split()[-1])
        return NodeType.HEADING, {'level': level}

    # Check if list item
    list_info = get_list_info(paragraph)
    if list_info and list_info['is_list']:
        return NodeType.LIST_ITEM, {
            'is_list': True,
            'list_level': list_info['level'],
            'num_id': list_info['num_id'],
            'is_ordered': is_numbered_list(list_info['num_id'])
        }

    # Regular paragraph
    return NodeType.PARAGRAPH, {}
```

### 3.5 Table Parsing

```python
def parse_table(
    element: Any,
    document_index: int,
    character_offset: int,
    depth: int,
    parent_id: Optional[str]
) -> tuple[ContentNode, int]:
    """
    Parse table element into ContentNode with nested structure.

    Structure:
        TABLE
        └── ROW
            └── CELL
                └── PARAGRAPH(S)
    """
    from docx.table import Table
    table = Table(element, None)

    position_id = generate_position_id(NodeType.TABLE, document_index)

    # Parse rows
    row_nodes = []
    total_chars = 0
    row_index = 0

    for row in table.rows:
        cell_nodes = []
        cell_index = 0

        for cell in row.cells:
            # Parse cell paragraphs
            para_nodes = []
            cell_char_offset = character_offset + total_chars

            for para in cell.paragraphs:
                para_node, chars = parse_paragraph(
                    element=para._element,
                    document_index=document_index,
                    character_offset=cell_char_offset,
                    depth=depth + 2,
                    parent_id=f"{position_id}_r{row_index}_c{cell_index}"
                )
                para_nodes.append(para_node)
                total_chars += chars
                cell_char_offset += chars

            # Create cell node
            cell_node = ContentNode(
                id=f"{position_id}_r{row_index}_c{cell_index}",
                type=NodeType.TABLE_CELL,
                content=''.join(p.content for p in para_nodes),
                formatting=FormattingInfo(
                    cell=extract_cell_format(cell)
                ),
                position=PositionInfo(
                    document_index=document_index,
                    parent_id=f"{position_id}_r{row_index}",
                    sibling_index=cell_index,
                    depth=depth + 2,
                    character_offset=character_offset + total_chars
                ),
                children=para_nodes,
                metadata={}
            )
            cell_nodes.append(cell_node)
            cell_index += 1

        # Create row node
        row_node = ContentNode(
            id=f"{position_id}_r{row_index}",
            type=NodeType.TABLE_ROW,
            content='',
            formatting=FormattingInfo(),
            position=PositionInfo(
                document_index=document_index,
                parent_id=position_id,
                sibling_index=row_index,
                depth=depth + 1,
                character_offset=character_offset
            ),
            children=cell_nodes,
            metadata={}
        )
        row_nodes.append(row_node)
        row_index += 1

    # Create table node
    table_node = ContentNode(
        id=position_id,
        type=NodeType.TABLE,
        content='',
        formatting=FormattingInfo(
            table=extract_table_format(table)
        ),
        position=PositionInfo(
            document_index=document_index,
            parent_id=parent_id,
            sibling_index=0,
            depth=depth,
            character_offset=character_offset
        ),
        children=row_nodes,
        metadata={
            'row_count': len(table.rows),
            'column_count': len(table.columns)
        }
    )

    return table_node, total_chars
```

### 3.6 Position ID Generation Scheme

```python
def generate_position_id(node_type: NodeType, document_index: int) -> str:
    """
    Generate stable position identifier for node.

    Format:
        h{index}  - Heading (e.g., h0, h1, h2)
        p{index}  - Paragraph (e.g., p0, p1, p2)
        l{index}  - List item (e.g., l0, l1, l2)
        t{index}  - Table (e.g., t0, t1, t2)
        img{index} - Image (e.g., img0, img1)

    Nested elements append to parent ID:
        t0_r0_c0  - Table 0, Row 0, Cell 0
        t0_r0_c0_p0 - Paragraph 0 within that cell
    """
    prefix_map = {
        NodeType.HEADING: 'h',
        NodeType.PARAGRAPH: 'p',
        NodeType.LIST_ITEM: 'l',
        NodeType.TABLE: 't',
        NodeType.TABLE_ROW: 'r',
        NodeType.TABLE_CELL: 'c',
        NodeType.IMAGE: 'img'
    }

    prefix = prefix_map.get(node_type, 'n')
    return f"{prefix}{document_index}"
```

### 3.7 List Handling

```python
def parse_list_structure(nodes: List[ContentNode]) -> List[ContentNode]:
    """
    Post-process flat list items into nested LIST nodes.

    Converts:
        [LIST_ITEM(level=0), LIST_ITEM(level=1), LIST_ITEM(level=0)]

    Into:
        [LIST([LIST_ITEM, LIST([LIST_ITEM])]), LIST([LIST_ITEM])]
    """
    structured = []
    list_stack = []

    for node in nodes:
        if node.type != NodeType.LIST_ITEM:
            # Non-list item: close any open lists
            while list_stack:
                structured.append(list_stack.pop(0))
            structured.append(node)
            continue

        level = node.metadata.get('list_level', 0)

        # Close lists deeper than current level
        while len(list_stack) > level + 1:
            parent = list_stack.pop()
            if list_stack:
                list_stack[-1].children.append(parent)
            else:
                structured.append(parent)

        # Create new list container if needed
        if len(list_stack) == level:
            list_node = ContentNode(
                id=f"list_{node.id}",
                type=NodeType.LIST,
                content='',
                formatting=FormattingInfo(),
                position=node.position,
                children=[],
                metadata={'list_level': level}
            )
            list_stack.append(list_node)

        # Add item to current list
        list_stack[level].children.append(node)

    # Close remaining lists
    while list_stack:
        structured.append(list_stack.pop(0))

    return structured
```

---

## 4. OriginalSnapshot Design

### 4.1 Purpose and Immutability

**Purpose:**
The OriginalSnapshot serves as an immutable, canonical record of the document's content at ingestion time. This snapshot enables precise diffing by providing:

1. Position-based text lookup (position_id -> original_text)
2. Document integrity verification (hash checking)
3. Change detection baseline (immutable reference point)

**Immutability Guarantee:**
- Created once during parsing
- Never modified after creation
- Stored separately from mutable DocumentAST
- Enables reliable diffing even if AST is modified

### 4.2 Snapshot Structure

```python
from dataclasses import dataclass
from typing import Dict
import hashlib
import json

@dataclass(frozen=True)  # Immutable
class OriginalSnapshot:
    """
    Immutable snapshot of original document content.

    Attributes:
        document_id: UUID of source DocumentAST
        document_hash: SHA-256 hash of source DOCX bytes
        structure_fingerprint: Hash of document structure (node types + positions)
        text_by_position: Mapping of position_id -> original text
        metadata_by_position: Mapping of position_id -> node metadata
        created_at: Snapshot creation timestamp
    """
    document_id: str
    document_hash: str
    structure_fingerprint: str
    text_by_position: Dict[str, str]
    metadata_by_position: Dict[str, Dict[str, Any]]
    created_at: datetime

    def verify_integrity(self, docx_path: str) -> bool:
        """Verify snapshot matches source DOCX"""
        current_hash = calculate_file_hash(docx_path)
        return current_hash == self.document_hash

    def get_text(self, position_id: str) -> Optional[str]:
        """Get original text for position ID"""
        return self.text_by_position.get(position_id)

    def get_metadata(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get original metadata for position ID"""
        return self.metadata_by_position.get(position_id)
```

### 4.3 Snapshot Creation Algorithm

```python
def create_snapshot(ast: DocumentAST) -> OriginalSnapshot:
    """
    Create immutable snapshot from DocumentAST.

    Args:
        ast: Parsed DocumentAST

    Returns:
        Immutable OriginalSnapshot
    """
    text_by_position = {}
    metadata_by_position = {}

    # Flatten AST to position mappings
    for node in ast.get_all_nodes():
        text_by_position[node.id] = node.content
        metadata_by_position[node.id] = {
            'type': node.type.value,
            'depth': node.position.depth,
            'character_offset': node.position.character_offset,
            **node.metadata
        }

    # Generate structure fingerprint
    structure_fingerprint = generate_structure_fingerprint(ast)

    return OriginalSnapshot(
        document_id=ast.id,
        document_hash=ast.source_hash,
        structure_fingerprint=structure_fingerprint,
        text_by_position=text_by_position,
        metadata_by_position=metadata_by_position,
        created_at=datetime.now()
    )
```

### 4.4 Structure Fingerprint Generation

```python
def generate_structure_fingerprint(ast: DocumentAST) -> str:
    """
    Generate hash of document structure (not content).

    Used to detect structural changes (node addition/removal/reordering)
    independently from content changes.

    Fingerprint includes:
    - Node types in order
    - Node IDs in order
    - Nesting structure
    """
    structure_repr = []

    for node in ast.get_all_nodes():
        structure_repr.append({
            'id': node.id,
            'type': node.type.value,
            'depth': node.position.depth,
            'parent_id': node.position.parent_id
        })

    # Create stable JSON representation
    json_str = json.dumps(structure_repr, sort_keys=True)

    # Hash
    return hashlib.sha256(json_str.encode()).hexdigest()
```

### 4.5 Hash Generation for Integrity

```python
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file contents"""
    sha256 = hashlib.sha256()

    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()
```

---

## 5. Edge Case Handling Matrix

| Edge Case | Detection Method | Handling Strategy | Implementation Notes |
|-----------|------------------|-------------------|---------------------|
| **Tracked Changes** | Check for `w:ins`, `w:del` elements in XML | Strip tracked changes, use final text | Use `accept_all_revisions()` preprocessing |
| **Comments** | Check for `w:commentReference` elements | Extract comment text to metadata | Store comments in node metadata, don't include in content |
| **Images** | Detect `w:drawing` elements in runs | Create IMAGE node with alt text | Extract alt text from `wp:docPr[@descr]` |
| **Image Alt Text** | Parse `wp:docPr` element | Store in node metadata | Preserve for accessibility |
| **Hyperlinks** | Detect `w:hyperlink` elements | Extract URL + text, create HYPERLINK node | Resolve relationship ID to URL |
| **Nested Tables** | Table within table cell | Recursive parsing | Parse cell content as full element sequence |
| **Merged Cells** | Cell object identity comparison | Single cell node spanning multiple positions | Track merge span in metadata |
| **Empty Paragraphs** | `paragraph.text == ""` | Preserve as node (maintains spacing) | Include empty nodes for structure fidelity |
| **Text Boxes** | Detect `w:txbxContent` elements | Extract as separate paragraph sequence | Treat as floating content |
| **Footnotes/Endnotes** | Check `w:footnoteReference`, `w:endnoteReference` | Extract to metadata | Link to footnote content via ID |
| **Headers/Footers** | Iterate `doc.sections[].header/footer` | Parse separately, mark with section ID | Store in separate AST sections |
| **Page Breaks** | `paragraph_format.page_break_before` | Record in formatting metadata | Preserve for reconstruction |
| **Section Breaks** | Multiple `doc.sections` | Create SECTION_BREAK nodes | Maintain section properties |
| **Non-Standard Fonts** | `run.font.name` not in system fonts | Record font name, flag for validation | Warn user if font unavailable |
| **Embedded Objects** | Detect `o:OLEObject` elements | Extract object metadata | Store object type + data reference |
| **Smart Quotes** | UTF-8 characters U+2018-U+201F | Preserve as-is | No normalization (maintains author intent) |
| **Non-Breaking Spaces** | UTF-8 character U+00A0 | Preserve as-is | Required for formatting preservation |
| **Tab Characters** | `\t` in run text | Preserve as-is | Maintain alignment |
| **Line Breaks (Soft)** | `w:br` elements | Convert to `\n` in run text | Distinguish from paragraph breaks |
| **Symbols/Special Chars** | `w:sym` elements | Extract Unicode equivalent | Map symbol font + char code to Unicode |
| **Equation Objects** | `m:oMath` elements | Extract MathML if possible | Store equation representation |

### 5.1 Tracked Changes Preprocessing

```python
def accept_all_revisions(doc: Document) -> Document:
    """
    Accept all tracked changes in document before parsing.

    This ensures we parse the "final" version of the document
    without revision markup complicating the structure.
    """
    from docx.oxml import parse_xml

    # Remove all deletion elements
    for del_element in doc.element.body.findall('.//w:del', namespaces={
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    }):
        del_element.getparent().remove(del_element)

    # Unwrap all insertion elements (keep content, remove markup)
    for ins_element in doc.element.body.findall('.//w:ins', namespaces={
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    }):
        parent = ins_element.getparent()
        index = parent.index(ins_element)

        # Move children out of insertion wrapper
        for child in ins_element:
            parent.insert(index, child)
            index += 1

        # Remove insertion wrapper
        parent.remove(ins_element)

    return doc
```

### 5.2 Comment Extraction

```python
def extract_comments(paragraph) -> List[Dict[str, Any]]:
    """
    Extract comments from paragraph.

    Returns:
        List of comment dictionaries with author, text, date
    """
    comments = []

    # Find comment reference elements
    for comment_ref in paragraph._element.findall('.//w:commentReference', namespaces={
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    }):
        comment_id = comment_ref.get(qn('w:id'))

        # Resolve comment from document comments part
        comments_part = paragraph.part.package.part_related_by(
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments'
        )

        if comments_part:
            comment_element = comments_part.element.find(
                f'.//w:comment[@w:id="{comment_id}"]',
                namespaces={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            )

            if comment_element is not None:
                comments.append({
                    'id': comment_id,
                    'author': comment_element.get(qn('w:author')),
                    'date': comment_element.get(qn('w:date')),
                    'text': ''.join(comment_element.itertext())
                })

    return comments
```

---

## 6. Code Examples

### 6.1 Basic Parsing Example

```python
from src.ingestion.docx_parser import parse_docx
from pathlib import Path

# Parse DOCX file
docx_path = Path("input/webpage_content.docx")
ast, snapshot = parse_docx(str(docx_path))

# Access document structure
print(f"Document ID: {ast.id}")
print(f"Total nodes: {len(ast.get_all_nodes())}")
print(f"Source hash: {ast.source_hash}")

# Iterate top-level nodes
for node in ast.nodes:
    print(f"{node.type.value}: {node.content[:50]}...")

# Verify integrity
assert snapshot.verify_integrity(str(docx_path))
```

### 6.2 Position Mapping Example

```python
# Find node by position ID
node = ast.find_node("p5")
if node:
    print(f"Node p5 content: {node.content}")
    print(f"Character offset: {node.position.character_offset}")
    print(f"Depth: {node.position.depth}")

# Get original text from snapshot
original_text = snapshot.get_text("p5")
assert original_text == node.content  # Should match

# Get all paragraph texts
paragraphs = [
    node for node in ast.get_all_nodes()
    if node.type == NodeType.PARAGRAPH
]

for para in paragraphs:
    print(f"{para.id}: {para.content}")
```

### 6.3 Formatting Extraction Example

```python
# Extract heading hierarchy
headings = [
    node for node in ast.get_all_nodes()
    if node.type == NodeType.HEADING
]

for heading in headings:
    level = heading.metadata['level']
    style = heading.formatting.paragraph.style_name
    print(f"{'  ' * (level - 1)}H{level}: {heading.content} [{style}]")

# Output:
# H1: Introduction [Heading 1]
#   H2: Background [Heading 2]
#     H3: Historical Context [Heading 3]
#   H2: Methodology [Heading 2]

# Extract formatted runs from paragraph
node = ast.find_node("p10")
for run in node.runs:
    formatting_flags = []
    if run.bold:
        formatting_flags.append("BOLD")
    if run.italic:
        formatting_flags.append("ITALIC")

    flags_str = f"[{', '.join(formatting_flags)}]" if formatting_flags else ""
    print(f"  '{run.text}' {flags_str}")

# Output:
#   'This is ' []
#   'important' [BOLD]
#   ' text with ' []
#   'emphasis' [ITALIC]
```

### 6.4 Table Structure Example

```python
# Find all tables
tables = [
    node for node in ast.get_all_nodes()
    if node.type == NodeType.TABLE
]

for table in tables:
    print(f"Table {table.id}:")
    print(f"  Rows: {table.metadata['row_count']}")
    print(f"  Columns: {table.metadata['column_count']}")

    # Iterate rows
    for row in table.children:
        row_texts = []
        for cell in row.children:
            # Get cell content (may have multiple paragraphs)
            cell_text = ' '.join(p.content for p in cell.children)
            row_texts.append(cell_text)

        print(f"  | {' | '.join(row_texts)} |")

# Output:
# Table t0:
#   Rows: 3
#   Columns: 2
#   | Header 1 | Header 2 |
#   | Value 1 | Value 2 |
#   | Value 3 | Value 4 |
```

### 6.5 List Structure Example

```python
# Find list items
list_items = [
    node for node in ast.get_all_nodes()
    if node.type == NodeType.LIST_ITEM
]

# Group by list level
from collections import defaultdict
by_level = defaultdict(list)

for item in list_items:
    level = item.metadata.get('list_level', 0)
    by_level[level].append(item)

# Print hierarchically
def print_list_item(item, indent=0):
    prefix = "•" if not item.metadata.get('is_ordered') else f"{indent + 1}."
    print(f"{'  ' * indent}{prefix} {item.content}")

for level in sorted(by_level.keys()):
    for item in by_level[level]:
        print_list_item(item, indent=level)

# Output:
# • First item
#   • Nested item
#   • Another nested item
# • Second item
```

---

## 7. Testing Strategy

### 7.1 Test Fixture Requirements

**Minimal Test Fixtures:**

1. **simple.docx** - Basic document
   - 1 heading (H1)
   - 3 paragraphs
   - No formatting
   - ~100 words

2. **formatted.docx** - Formatting variations
   - Bold, italic, underline text
   - Multiple font sizes and colors
   - Highlighted text
   - Mixed formatting within paragraphs

3. **structured.docx** - Complex structure
   - Heading hierarchy (H1-H3)
   - Bulleted lists (nested)
   - Numbered lists (nested)
   - Table (3x3)
   - ~500 words

4. **edge_cases.docx** - Edge case coverage
   - Tracked changes (insertions/deletions)
   - Comments
   - Images with alt text
   - Hyperlinks
   - Footnotes
   - Empty paragraphs
   - Page breaks

5. **real_world.docx** - Production-like content
   - Extracted webpage content
   - Mixed formatting
   - Tables with merged cells
   - Nested lists
   - ~2000 words

### 7.2 Key Test Scenarios

```python
import pytest
from src.ingestion.docx_parser import parse_docx
from src.ingestion.models import NodeType

class TestDocxParsing:

    def test_basic_parsing(self, simple_docx_path):
        """Test parsing of simple document"""
        ast, snapshot = parse_docx(simple_docx_path)

        # Verify AST structure
        assert ast.id is not None
        assert len(ast.nodes) > 0

        # Verify snapshot creation
        assert snapshot.document_id == ast.id
        assert snapshot.verify_integrity(simple_docx_path)

    def test_heading_detection(self, structured_docx_path):
        """Test heading hierarchy extraction"""
        ast, _ = parse_docx(structured_docx_path)

        headings = [n for n in ast.get_all_nodes() if n.type == NodeType.HEADING]

        # Verify heading levels
        assert any(h.metadata['level'] == 1 for h in headings)
        assert any(h.metadata['level'] == 2 for h in headings)
        assert any(h.metadata['level'] == 3 for h in headings)

    def test_formatting_preservation(self, formatted_docx_path):
        """Test that formatting is preserved across runs"""
        ast, _ = parse_docx(formatted_docx_path)

        # Find paragraph with mixed formatting
        para = next(n for n in ast.get_all_nodes()
                   if n.type == NodeType.PARAGRAPH and len(n.runs) > 1)

        # Verify runs have different formatting
        assert any(r.bold for r in para.runs)
        assert any(r.italic for r in para.runs)

    def test_table_structure(self, structured_docx_path):
        """Test table parsing with correct hierarchy"""
        ast, _ = parse_docx(structured_docx_path)

        tables = [n for n in ast.get_all_nodes() if n.type == NodeType.TABLE]
        assert len(tables) > 0

        table = tables[0]

        # Verify table structure
        assert table.metadata['row_count'] > 0
        assert table.metadata['column_count'] > 0

        # Verify row children
        assert all(child.type == NodeType.TABLE_ROW for child in table.children)

        # Verify cell children
        row = table.children[0]
        assert all(child.type == NodeType.TABLE_CELL for child in row.children)

    def test_list_nesting(self, structured_docx_path):
        """Test nested list handling"""
        ast, _ = parse_docx(structured_docx_path)

        list_items = [n for n in ast.get_all_nodes() if n.type == NodeType.LIST_ITEM]

        # Verify multiple nesting levels exist
        levels = {item.metadata['list_level'] for item in list_items}
        assert len(levels) > 1
        assert 0 in levels  # Top-level items

    def test_position_stability(self, simple_docx_path):
        """Test that position IDs are stable across parses"""
        ast1, _ = parse_docx(simple_docx_path)
        ast2, _ = parse_docx(simple_docx_path)

        nodes1 = ast1.get_all_nodes()
        nodes2 = ast2.get_all_nodes()

        # Same document should produce same position IDs
        assert [n.id for n in nodes1] == [n.id for n in nodes2]

    def test_character_offset_accuracy(self, simple_docx_path):
        """Test character offset calculation"""
        ast, _ = parse_docx(simple_docx_path)

        nodes = ast.get_all_nodes()

        # Verify offsets are monotonically increasing
        offsets = [n.position.character_offset for n in nodes]
        assert offsets == sorted(offsets)

        # Verify offset matches cumulative text length
        cumulative_length = 0
        for node in nodes:
            assert node.position.character_offset == cumulative_length
            cumulative_length += len(node.content)

    def test_snapshot_immutability(self, simple_docx_path):
        """Test that snapshot is immutable"""
        _, snapshot = parse_docx(simple_docx_path)

        # Attempt to modify should raise error
        with pytest.raises(AttributeError):
            snapshot.document_hash = "modified"

    def test_tracked_changes_handling(self, edge_cases_docx_path):
        """Test that tracked changes are resolved"""
        ast, _ = parse_docx(edge_cases_docx_path)

        # Document should contain final text (not tracked change markup)
        # Verify by checking that content is readable
        all_text = ''.join(n.content for n in ast.get_all_nodes())
        assert len(all_text) > 0

    def test_image_extraction(self, edge_cases_docx_path):
        """Test image metadata extraction"""
        ast, _ = parse_docx(edge_cases_docx_path)

        # Find nodes with image metadata
        nodes_with_images = [
            n for n in ast.get_all_nodes()
            if 'filename' in n.metadata
        ]

        if nodes_with_images:
            img_node = nodes_with_images[0]
            assert 'filename' in img_node.metadata
            assert 'alt_text' in img_node.metadata
```

### 7.3 Validation Criteria

**Correctness Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Structure fidelity | 100% | All document elements captured |
| Position ID stability | 100% | Same document = same IDs |
| Formatting preservation | 100% | All run-level formatting captured |
| Character offset accuracy | ±0 chars | Exact offset calculation |
| Snapshot immutability | 100% | No modification possible |

**Performance Benchmarks:**

| Document Size | Max Parse Time | Max Memory |
|---------------|----------------|------------|
| Small (<500 words) | 50ms | 5MB |
| Medium (500-2000 words) | 200ms | 20MB |
| Large (2000-10000 words) | 1s | 100MB |

**Edge Case Coverage:**

- Tracked changes: Handle gracefully (accept revisions)
- Comments: Extract to metadata
- Images: Extract alt text and filename
- Hyperlinks: Extract URL and text
- Empty paragraphs: Preserve for structure
- Nested tables: Parse recursively
- Merged cells: Track in metadata
- Non-standard fonts: Record and flag

**Error Handling:**

```python
def test_corrupted_docx(self, corrupted_docx_path):
    """Test graceful handling of corrupted DOCX"""
    with pytest.raises(ValueError, match="corrupted"):
        parse_docx(corrupted_docx_path)

def test_missing_file(self):
    """Test handling of nonexistent file"""
    with pytest.raises(FileNotFoundError):
        parse_docx("nonexistent.docx")

def test_non_docx_file(self, text_file_path):
    """Test handling of non-DOCX file"""
    with pytest.raises(ValueError, match="not a valid DOCX"):
        parse_docx(text_file_path)
```

---

## 8. Implementation Checklist

### Phase 1: Core Parsing (Week 1)

- [ ] Implement `DocumentAST` and `ContentNode` data models
- [ ] Implement `parse_docx()` main entry point
- [ ] Implement paragraph parsing with run extraction
- [ ] Implement heading detection
- [ ] Implement position ID generation
- [ ] Implement character offset calculation
- [ ] Write tests for basic parsing (simple.docx)

### Phase 2: Structure Handling (Week 1-2)

- [ ] Implement table parsing with recursive cell handling
- [ ] Implement list detection and nesting
- [ ] Implement `OriginalSnapshot` creation
- [ ] Implement structure fingerprint generation
- [ ] Write tests for structured documents (structured.docx)

### Phase 3: Formatting (Week 2)

- [ ] Implement `FormattingInfo` extraction (paragraph format)
- [ ] Implement run-level formatting capture
- [ ] Implement style inheritance handling
- [ ] Write tests for formatting preservation (formatted.docx)

### Phase 4: Edge Cases (Week 3)

- [ ] Implement tracked changes preprocessing
- [ ] Implement comment extraction
- [ ] Implement image metadata extraction
- [ ] Implement hyperlink extraction
- [ ] Write tests for edge cases (edge_cases.docx)

### Phase 5: Integration & Testing (Week 3-4)

- [ ] Integration testing with real-world documents
- [ ] Performance benchmarking
- [ ] Error handling hardening
- [ ] Documentation completion
- [ ] Code review and refactoring

---

## 9. Dependencies

**Required Packages:**

```toml
[dependencies]
python-docx = "^1.1.2"
lxml = "^5.1.0"  # Required by python-docx
typing-extensions = "^4.9.0"
pydantic = "^2.5.0"  # For data validation
```

**Python Version:**
- Minimum: Python 3.12
- Recommended: Python 3.12+

---

## 10. Open Questions & Decisions

### 10.1 Resolved Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| List representation | Post-process flat items into nested LIST nodes | Matches document structure better for diffing |
| Position ID scheme | Type prefix + index (e.g., p0, h1, t0) | Human-readable and stable |
| Snapshot immutability | Use `@dataclass(frozen=True)` | Enforces immutability at runtime |
| Tracked changes | Accept all revisions during preprocessing | Simplifies parsing, users expect final content |

### 10.2 Open Questions

1. **Footnote positioning**: Should footnotes be inline (in paragraph metadata) or separate nodes?
   - **Impact:** Affects diffing if footnotes are modified
   - **Recommendation:** Inline metadata, with position reference

2. **Header/footer handling**: Should headers/footers be part of main AST or separate?
   - **Impact:** Affects structure fingerprint
   - **Recommendation:** Separate section with section_id reference

3. **Performance optimization**: Should parsing be streaming or batch?
   - **Impact:** Memory usage for large documents
   - **Recommendation:** Batch for MVP, streaming for v2 if needed

4. **XML caching**: Should we cache parsed XML elements?
   - **Impact:** Memory vs. speed tradeoff
   - **Recommendation:** No caching for MVP (simplicity), profile later

---

## 11. Rationale Summary

### Key Design Decisions

1. **Immutable Snapshot Design**
   - **Why:** Diffing requires stable baseline; mutations would corrupt change detection
   - **Tradeoff:** Memory overhead (duplicate storage) vs. correctness guarantee
   - **Decision:** Correctness is critical, memory is acceptable

2. **Run-Level Formatting Preservation**
   - **Why:** Output must exactly match input formatting; paragraph.text loses boundaries
   - **Tradeoff:** Complexity (iterate runs) vs. fidelity
   - **Decision:** Fidelity is non-negotiable for user trust

3. **Position ID Stability**
   - **Why:** Diffing relies on position matching; UUIDs would break across parses
   - **Tradeoff:** Deterministic IDs (type+index) vs. globally unique IDs
   - **Decision:** Determinism enables reliable diffing of same document

4. **Recursive Table Parsing**
   - **Why:** Tables contain paragraphs which may contain images/hyperlinks/etc.
   - **Tradeoff:** Complexity (recursive parsing) vs. structure accuracy
   - **Decision:** Accuracy required; complexity is manageable

5. **Tracked Changes Preprocessing**
   - **Why:** Parsing revision markup complicates structure; users expect final content
   - **Tradeoff:** Loss of revision history vs. parsing simplicity
   - **Decision:** History not needed for SEO optimization use case

---

**Document Version:** 1.0
**Last Updated:** 2026-01-16
**Author:** Technical Specification
**Status:** Ready for Implementation

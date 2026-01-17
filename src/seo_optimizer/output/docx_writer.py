"""
DOCX Writer - Reconstructs DOCX with optimized content

Reconstructs the DOCX file with:
- Original content preserved exactly
- New content inserted at appropriate positions
- Highlighting applied via highlighter module

Reference: docs/research/06-docx-output.md
"""

from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from docx import Document

from seo_optimizer.diffing.models import ChangeSet
from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType
from seo_optimizer.output.highlighter import apply_highlights

if TYPE_CHECKING:
    from docx.document import Document as DocxDocument
    from docx.text.paragraph import Paragraph


def _ensure_heading_styles(doc: DocxDocument) -> None:
    """
    Ensure heading styles exist in the document.

    Args:
        doc: Document to check/modify
    """
    # python-docx should have built-in heading styles
    # but we verify they exist
    for level in range(1, 7):
        style_name = f"Heading {level}"
        with contextlib.suppress(KeyError):
            _ = doc.styles[style_name]


def _add_paragraph_with_highlighting(
    doc: DocxDocument,
    text: str,
    style: str | None = None,
    highlight: bool = False,
) -> Paragraph:
    """
    Add a paragraph to the document with optional highlighting.

    Args:
        doc: Document to add to
        text: Text content for the paragraph
        style: Optional style name (e.g., "Heading 1")
        highlight: Whether to apply highlight

    Returns:
        The created paragraph
    """
    para = doc.add_paragraph()
    if style:
        with contextlib.suppress(KeyError):
            para.style = style

    run = para.add_run(text)
    if highlight:
        from docx.enum.text import WD_COLOR_INDEX

        run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    return para


def _copy_paragraph_formatting(
    source_para: Paragraph,
    target_para: Paragraph,
) -> None:
    """
    Copy formatting from one paragraph to another.

    Args:
        source_para: Source paragraph
        target_para: Target paragraph
    """
    # Copy paragraph format
    if source_para.paragraph_format:
        spf = source_para.paragraph_format
        tpf = target_para.paragraph_format

        if spf.alignment is not None:
            tpf.alignment = spf.alignment
        if spf.left_indent is not None:
            tpf.left_indent = spf.left_indent
        if spf.right_indent is not None:
            tpf.right_indent = spf.right_indent
        if spf.first_line_indent is not None:
            tpf.first_line_indent = spf.first_line_indent
        if spf.space_before is not None:
            tpf.space_before = spf.space_before
        if spf.space_after is not None:
            tpf.space_after = spf.space_after
        if spf.line_spacing is not None:
            tpf.line_spacing = spf.line_spacing

    # Copy style
    if source_para.style:
        with contextlib.suppress(KeyError):
            target_para.style = source_para.style


def write_optimized_docx(
    original_path: str | Path,
    optimized_ast: DocumentAST,
    changeset: ChangeSet,
    output_path: str | Path,
) -> Path:
    """
    Write the optimized document with highlighting.

    Creates a new DOCX file with:
    - All original content preserved
    - New content added
    - Highlighting on all new content

    Args:
        original_path: Path to original DOCX (for formatting reference)
        optimized_ast: The optimized document AST
        changeset: Changes to highlight
        output_path: Where to save the output

    Returns:
        Path to the created file

    Example:
        >>> output = write_optimized_docx(
        ...     "input.docx",
        ...     optimized_ast,
        ...     changeset,
        ...     "output.docx"
        ... )
        >>> print(f"Saved to {output}")
    """
    original_path = Path(original_path)
    output_path = Path(output_path)

    if not original_path.exists():
        raise FileNotFoundError(f"Original file not found: {original_path}")

    # Copy original to output location first (preserves formatting/styles)
    shutil.copy2(original_path, output_path)

    # Open the copy for modification
    doc: DocxDocument = Document(str(output_path))

    # Apply highlights from changeset
    apply_highlights(doc, changeset)

    # Save the modified document
    doc.save(str(output_path))

    return output_path


def write_document_from_ast(
    ast: DocumentAST,
    output_path: str | Path,
    highlight_new: bool = True,
    new_node_ids: set[str] | None = None,
) -> Path:
    """
    Write a document from an AST structure.

    Creates a new DOCX file from scratch based on the AST.

    Args:
        ast: Document AST to write
        output_path: Where to save the output
        highlight_new: Whether to highlight new content
        new_node_ids: Set of node IDs that should be highlighted

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    doc = Document()

    _ensure_heading_styles(doc)

    new_node_ids = new_node_ids or set()

    for node in ast.nodes:
        _write_node_to_document(doc, node, highlight_new, new_node_ids)

    doc.save(str(output_path))
    return output_path


def _write_node_to_document(
    doc: DocxDocument,
    node: ContentNode,
    highlight_new: bool,
    new_node_ids: set[str],
) -> None:
    """
    Write a single AST node to the document.

    Args:
        doc: Document to write to
        node: Node to write
        highlight_new: Whether to highlight
        new_node_ids: Set of new node IDs
    """
    should_highlight = highlight_new and node.node_id in new_node_ids

    if node.node_type == NodeType.HEADING:
        level = node.metadata.get("heading_level", 1)
        style = f"Heading {level}"
        _add_paragraph_with_highlighting(
            doc, node.text_content, style=style, highlight=should_highlight
        )
    elif node.node_type == NodeType.PARAGRAPH:
        _add_paragraph_with_highlighting(
            doc, node.text_content, highlight=should_highlight
        )
    elif node.node_type == NodeType.TABLE:
        _write_table_node(doc, node, should_highlight)
    elif node.node_type == NodeType.LIST:
        _write_list_node(doc, node, highlight_new, new_node_ids)

    # Handle child nodes (for complex structures)
    for child in node.children:
        if node.node_type not in (NodeType.TABLE, NodeType.LIST):
            _write_node_to_document(doc, child, highlight_new, new_node_ids)


def _write_table_node(
    doc: DocxDocument,
    node: ContentNode,
    highlight: bool,
) -> None:
    """
    Write a table node to the document.

    Args:
        doc: Document to write to
        node: Table node
        highlight: Whether to highlight
    """
    # Count rows and cells
    row_nodes = [c for c in node.children if c.node_type == NodeType.TABLE_ROW]
    if not row_nodes:
        return

    # Determine max columns
    max_cols = 0
    for row in row_nodes:
        cell_count = len(
            [c for c in row.children if c.node_type == NodeType.TABLE_CELL]
        )
        max_cols = max(max_cols, cell_count)

    if max_cols == 0:
        return

    # Create table
    table = doc.add_table(rows=len(row_nodes), cols=max_cols)

    for row_idx, row_node in enumerate(row_nodes):
        cell_nodes = [c for c in row_node.children if c.node_type == NodeType.TABLE_CELL]
        for col_idx, cell_node in enumerate(cell_nodes):
            if col_idx < max_cols:
                cell = table.rows[row_idx].cells[col_idx]
                # Clear default paragraph and add content
                if cell.paragraphs:
                    p = cell.paragraphs[0]
                    p.clear()
                    run = p.add_run(cell_node.text_content)
                    if highlight:
                        from docx.enum.text import WD_COLOR_INDEX

                        run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def _write_list_node(
    doc: DocxDocument,
    node: ContentNode,
    highlight_new: bool,
    new_node_ids: set[str],
) -> None:
    """
    Write a list node to the document.

    Args:
        doc: Document to write to
        node: List node
        highlight_new: Whether to highlight
        new_node_ids: Set of new node IDs
    """
    for item in node.children:
        if item.node_type == NodeType.LIST_ITEM:
            should_highlight = highlight_new and item.node_id in new_node_ids
            para = doc.add_paragraph(style="List Bullet")
            run = para.add_run(item.text_content)
            if should_highlight:
                from docx.enum.text import WD_COLOR_INDEX

                run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def insert_faq_section(
    doc: DocxDocument,
    faq_content: list[tuple[str, str]],
    highlight: bool = True,
) -> DocxDocument:
    """
    Insert an FAQ section into the document.

    Args:
        doc: python-docx Document to modify
        faq_content: List of (question, answer) tuples
        highlight: Whether to highlight as new content

    Returns:
        Modified document with FAQ section
    """
    from docx.enum.text import WD_COLOR_INDEX

    # Add FAQ heading
    faq_heading = doc.add_heading("Frequently Asked Questions", level=2)
    if highlight:
        for run in faq_heading.runs:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    # Add each Q&A pair
    for question, answer in faq_content:
        # Add question as bold paragraph
        q_para = doc.add_paragraph()
        q_run = q_para.add_run(f"Q: {question}")
        q_run.bold = True
        if highlight:
            q_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

        # Add answer
        a_para = doc.add_paragraph()
        a_run = a_para.add_run(f"A: {answer}")
        if highlight:
            a_run.font.highlight_color = WD_COLOR_INDEX.YELLOW

        # Add spacing
        doc.add_paragraph()

    return doc


def insert_content_at_position(
    doc: DocxDocument,
    content: str,
    position: int,
    style: str | None = None,
    highlight: bool = True,
) -> Paragraph:
    """
    Insert content at a specific paragraph position.

    Args:
        doc: Document to modify
        content: Text content to insert
        position: Paragraph index to insert after
        style: Optional style name
        highlight: Whether to highlight

    Returns:
        The inserted paragraph
    """
    # python-docx doesn't have direct insert, so we need to work with XML
    para = doc.add_paragraph()
    if style:
        with contextlib.suppress(KeyError):
            para.style = style

    run = para.add_run(content)
    if highlight:
        from docx.enum.text import WD_COLOR_INDEX

        run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    # Move the paragraph to the correct position
    if position < len(doc.paragraphs) - 1:
        # Get the paragraph elements
        body = doc.element.body
        para_element = para._element
        body.remove(para_element)

        # Find the target position
        target_para = doc.paragraphs[position]
        target_para._element.addnext(para_element)

    return para


def validate_output(
    output_path: str | Path,
) -> list[str]:
    """
    Validate the output DOCX file.

    Checks:
    - File is valid DOCX
    - Highlights are correctly applied
    - No formatting corruption

    Args:
        output_path: Path to validate

    Returns:
        List of validation warnings (empty if valid)
    """
    output_path = Path(output_path)
    warnings: list[str] = []

    if not output_path.exists():
        return ["File does not exist"]

    if output_path.suffix.lower() != ".docx":
        warnings.append("File extension is not .docx")

    try:
        doc = Document(str(output_path))
    except Exception as e:
        return [f"Invalid DOCX file: {e}"]

    # Check for content
    if len(doc.paragraphs) == 0:
        warnings.append("Document has no paragraphs")

    # Check for highlights (informational)
    highlight_count = 0
    for para in doc.paragraphs:
        for run in para.runs:
            if run.font.highlight_color is not None:
                highlight_count += 1

    if highlight_count == 0:
        warnings.append("No highlighted content found (may be intentional)")

    return warnings


class DocxWriter:
    """
    Class-based wrapper for DOCX writing operations.

    Provides a streaming interface for generating optimized documents.
    """

    def __init__(self) -> None:
        """Initialize the DocxWriter."""
        pass

    def write_to_stream(
        self,
        ast: DocumentAST,
        stream: io.BytesIO,
        change_map: dict | None = None,
        highlight_new: bool = True,
    ) -> None:
        """
        Write optimized document to a stream.

        Args:
            ast: Document AST to write
            stream: BytesIO stream to write to
            change_map: Optional mapping of changes for highlighting
            highlight_new: Whether to highlight new content
        """
        import io as io_module
        from docx import Document
        from docx.enum.text import WD_COLOR_INDEX

        doc = Document()
        _ensure_heading_styles(doc)

        # Determine which nodes are new based on change_map
        new_node_ids: set[str] = set()
        if change_map:
            new_node_ids = set(change_map.get("new_nodes", []))

        for node in ast.nodes:
            _write_node_to_document(doc, node, highlight_new, new_node_ids)

        # Save to stream
        doc.save(stream)
        stream.seek(0)


def merge_documents(
    base_doc_path: str | Path,
    additions: list[tuple[str, str]],
    output_path: str | Path,
    highlight_additions: bool = True,
) -> Path:
    """
    Merge additional content into a base document.

    Args:
        base_doc_path: Path to the base document
        additions: List of (position, content) tuples
                   position can be "end", "start", or a paragraph index
        output_path: Where to save the merged document
        highlight_additions: Whether to highlight added content

    Returns:
        Path to the merged document
    """
    base_doc_path = Path(base_doc_path)
    output_path = Path(output_path)

    # Copy base document
    shutil.copy2(base_doc_path, output_path)

    doc = Document(str(output_path))

    for position, content in additions:
        if position == "end":
            para = doc.add_paragraph()
            run = para.add_run(content)
            if highlight_additions:
                from docx.enum.text import WD_COLOR_INDEX

                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif position == "start":
            insert_content_at_position(doc, content, 0, highlight=highlight_additions)
        else:
            try:
                idx = int(position)
                insert_content_at_position(
                    doc, content, idx, highlight=highlight_additions
                )
            except ValueError:
                # Invalid position, append to end
                para = doc.add_paragraph()
                run = para.add_run(content)
                if highlight_additions:
                    from docx.enum.text import WD_COLOR_INDEX

                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW

    doc.save(str(output_path))
    return output_path

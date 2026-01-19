"""
Output Module - DOCX Reconstruction with Highlighting

This module handles:
- Reconstructing DOCX with preserved formatting
- Multi-color highlighting (green=new, yellow=modified, strikethrough=deleted)
- Run-level precision for highlight boundaries
- Proper heading styles (H1, H2, H3)
- Document header with optimization summary

Reference: docs/research/06-docx-output.md
"""

from seo_optimizer.output.docx_writer import (
    DocxWriter,
    OptimizedDocumentWriter,
    PreservingDocxWriter,
    add_color_legend,
    add_faq_section_enhanced,
    add_optimization_header,
    apply_heading_style,
    insert_content_at_position,
    insert_faq_section,
    merge_documents,
    setup_document_styles,
    validate_output,
    write_document_from_ast,
    write_optimized_docx,
)
from seo_optimizer.output.highlighter import (
    HIGHLIGHT_COLOR_INDEX,
    HIGHLIGHT_COLORS,
    add_paragraph_with_changes,
    add_text_with_changes,
    apply_change_formatting,
    apply_highlights,
    create_highlighted_run,
    highlight_new_paragraph,
    highlight_region,
    highlight_text_in_paragraph,
    split_run_for_highlight,
)

__all__ = [
    # Writer classes
    "DocxWriter",
    "OptimizedDocumentWriter",
    "PreservingDocxWriter",
    # Writer functions
    "write_optimized_docx",
    "write_document_from_ast",
    "insert_faq_section",
    "add_faq_section_enhanced",
    "insert_content_at_position",
    "validate_output",
    "merge_documents",
    # Style functions
    "setup_document_styles",
    "apply_heading_style",
    "add_optimization_header",
    "add_color_legend",
    # Highlighter functions
    "apply_highlights",
    "apply_change_formatting",
    "add_text_with_changes",
    "add_paragraph_with_changes",
    "highlight_region",
    "highlight_text_in_paragraph",
    "highlight_new_paragraph",
    "create_highlighted_run",
    "split_run_for_highlight",
    # Constants
    "HIGHLIGHT_COLOR_INDEX",
    "HIGHLIGHT_COLORS",
]

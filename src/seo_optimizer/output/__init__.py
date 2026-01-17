"""
Output Module - DOCX Reconstruction with Highlighting

This module handles:
- Reconstructing DOCX with preserved formatting
- Applying yellow highlighting to new content only
- Run-level precision for highlight boundaries

Reference: docs/research/06-docx-output.md
"""

from seo_optimizer.output.docx_writer import (
    insert_content_at_position,
    insert_faq_section,
    merge_documents,
    validate_output,
    write_document_from_ast,
    write_optimized_docx,
)
from seo_optimizer.output.highlighter import (
    HIGHLIGHT_COLOR_INDEX,
    apply_highlights,
    create_highlighted_run,
    highlight_new_paragraph,
    highlight_region,
    highlight_text_in_paragraph,
    split_run_for_highlight,
)

__all__ = [
    # Writer functions
    "write_optimized_docx",
    "write_document_from_ast",
    "insert_faq_section",
    "insert_content_at_position",
    "validate_output",
    "merge_documents",
    # Highlighter functions
    "apply_highlights",
    "highlight_region",
    "highlight_text_in_paragraph",
    "highlight_new_paragraph",
    "create_highlighted_run",
    "split_run_for_highlight",
    "HIGHLIGHT_COLOR_INDEX",
]

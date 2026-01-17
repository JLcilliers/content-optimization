"""
Ingestion Module - DOCX Parsing and Structure Extraction

This module handles:
- Parsing DOCX files while preserving exact structure
- Creating Document AST representation
- Generating OriginalSnapshot for diffing
- Position mapping for precise change tracking
"""

from seo_optimizer.ingestion.docx_parser import (
    create_snapshot,
    parse_docx,
    parse_docx_with_snapshot,
)
from seo_optimizer.ingestion.models import (
    ContentNode,
    DocumentAST,
    DocumentMetadata,
    FormattingInfo,
    NodeType,
    OriginalSnapshot,
    PositionInfo,
    TextRun,
)

__all__ = [
    # Parser functions
    "parse_docx",
    "parse_docx_with_snapshot",
    "create_snapshot",
    # Models
    "ContentNode",
    "DocumentAST",
    "DocumentMetadata",
    "FormattingInfo",
    "NodeType",
    "OriginalSnapshot",
    "PositionInfo",
    "TextRun",
]

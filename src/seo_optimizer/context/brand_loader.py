"""
Brand Document Loader

Parses brand documents (style guides, about pages, etc.)
to extract business context information.

Reference: docs/research/03-business-context.md section 2
"""

from pathlib import Path

from seo_optimizer.context.business_context import BusinessContext
from seo_optimizer.ingestion.models import DocumentAST


def load_brand_documents(
    file_paths: list[str | Path],
) -> list[DocumentAST]:
    """
    Load and parse brand documents.

    Args:
        file_paths: Paths to brand document files (DOCX)

    Returns:
        List of parsed DocumentAST objects

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    raise NotImplementedError(
        "Brand document loading not yet implemented. "
        "See docs/research/03-business-context.md."
    )


def extract_context_from_brand_docs(
    brand_docs: list[DocumentAST],
) -> BusinessContext:
    """
    Extract business context from brand documents.

    Analyzes brand documents to identify:
    - Business name and type
    - Products/services offered
    - Target audience
    - Tone and style preferences
    - Terminology rules

    Args:
        brand_docs: List of parsed brand documents

    Returns:
        BusinessContext with high confidence (0.8-1.0)

    Note:
        Brand document context has highest priority and confidence.
    """
    raise NotImplementedError(
        "Brand context extraction not yet implemented. "
        "See docs/research/03-business-context.md section 2."
    )

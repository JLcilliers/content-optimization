"""
Business Context Model and Builder

Builds a context model from:
1. Brand documents (if provided)
2. Content inference (from source document)
3. Defaults (when nothing is available)

The context informs FAQ generation and content enhancement
to ensure generated content aligns with the business.

Reference: docs/research/03-business-context.md
"""

from dataclasses import dataclass, field
from typing import Any

from seo_optimizer.ingestion.models import DocumentAST


@dataclass
class ToneProfile:
    """Profile describing the desired content tone."""

    formality: str = "professional"  # casual, professional, formal
    voice: str = "third_person"  # first_person, second_person, third_person
    style: str = "informative"  # informative, persuasive, conversational


@dataclass
class BusinessContext:
    """
    Complete business context for informing content generation.

    Used by FAQ generator and content enhancer to ensure
    generated content aligns with the business.
    """

    # Business identification
    business_name: str | None = None
    industry: str | None = None
    business_type: str | None = None  # B2B, B2C, SaaS, etc.

    # Products and services
    products_services: list[str] = field(default_factory=list)
    value_propositions: list[str] = field(default_factory=list)

    # Audience
    target_audience: str | None = None

    # Voice and tone
    tone: ToneProfile = field(default_factory=ToneProfile)

    # Terminology preferences
    preferred_terms: dict[str, str] = field(default_factory=dict)
    banned_terms: list[str] = field(default_factory=list)

    # Confidence in this context
    confidence: float = 0.5

    # Source of context (brand_docs, inferred, default)
    source: str = "default"

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


def build_context(
    source_doc: DocumentAST,
    brand_docs: list[DocumentAST] | None = None,
) -> BusinessContext:
    """
    Build business context from available sources.

    Priority:
    1. Brand documents (if provided) - highest confidence
    2. Content inference (from source) - medium confidence
    3. Defaults - lowest confidence

    Args:
        source_doc: The source document being optimized
        brand_docs: Optional list of brand context documents

    Returns:
        BusinessContext with appropriate confidence level

    Example:
        >>> context = build_context(source_ast)
        >>> print(f"Inferred business: {context.business_name}")
        >>> print(f"Confidence: {context.confidence}")
    """
    raise NotImplementedError(
        "Context building not yet implemented. "
        "See docs/research/03-business-context.md."
    )


def infer_context_from_content(doc: DocumentAST) -> BusinessContext:
    """
    Infer business context from document content alone.

    Uses NLP to extract:
    - Business entities
    - Industry indicators
    - Product/service mentions
    - Tone characteristics

    Args:
        doc: Document to analyze

    Returns:
        BusinessContext with confidence based on inference quality

    Note:
        Confidence is typically 0.4-0.7 for inferred context.
    """
    raise NotImplementedError(
        "Context inference not yet implemented. "
        "See docs/research/03-business-context.md section 3."
    )

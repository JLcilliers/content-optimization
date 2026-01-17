"""
Validators - Core validation functions for guardrails

Provides validation functions for:
- Keyword density and over-optimization
- Content grounding
- Highlight accuracy

Reference: docs/research/07-guardrails.md
"""

from dataclasses import dataclass, field

from seo_optimizer.analysis.keyword_mapper import KeywordAnalysis
from seo_optimizer.diffing.models import ChangeSet
from seo_optimizer.ingestion.models import DocumentAST, OriginalSnapshot


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


def validate_keyword_density(
    analysis: KeywordAnalysis,
    max_density: float = 0.025,
    reject_threshold: float = 0.04,
) -> ValidationResult:
    """
    Validate keyword density is within safe limits.

    Args:
        analysis: Keyword analysis results
        max_density: Warning threshold (default 2.5%)
        reject_threshold: Rejection threshold (default 4%)

    Returns:
        ValidationResult with warnings/errors

    Reference:
        docs/research/07-guardrails.md section 2
    """
    raise NotImplementedError(
        "Keyword density validation not yet implemented. "
        "See docs/research/07-guardrails.md section 2."
    )


def validate_grounding(
    generated_content: str,
    source_doc: DocumentAST,
    min_similarity: float = 0.7,
) -> ValidationResult:
    """
    Validate that generated content is grounded in source.

    Checks that all claims in generated content can be traced
    back to the source document.

    Args:
        generated_content: Content to validate
        source_doc: Source document for grounding
        min_similarity: Minimum similarity for grounding

    Returns:
        ValidationResult indicating grounding status

    Reference:
        docs/research/07-guardrails.md section 3
    """
    raise NotImplementedError(
        "Grounding validation not yet implemented. "
        "See docs/research/07-guardrails.md section 3."
    )


def validate_highlight_accuracy(
    changeset: ChangeSet,
    original: OriginalSnapshot,
    optimized: DocumentAST,
) -> ValidationResult:
    """
    Validate that highlights are accurate (no false positives).

    CRITICAL: This is a safety check to prevent highlighting
              content that existed in the original.

    Args:
        changeset: Changes to validate
        original: Original document snapshot
        optimized: Optimized document

    Returns:
        ValidationResult with any detected issues

    Reference:
        docs/research/07-guardrails.md section 5
    """
    raise NotImplementedError(
        "Highlight accuracy validation not yet implemented. "
        "See docs/research/07-guardrails.md section 5."
    )

"""
Gap Detector - Identifies optimization opportunities

Detects gaps in content that present optimization opportunities:
- Missing keywords in key positions
- Missing FAQ section
- Thin content sections
- Semantic gaps (missing related topics)

Reference: docs/research/02-keyword-optimization.md section 4
"""

from dataclasses import dataclass, field
from enum import Enum

from seo_optimizer.analysis.keyword_mapper import KeywordAnalysis
from seo_optimizer.context.business_context import BusinessContext
from seo_optimizer.ingestion.models import DocumentAST


class GapType(str, Enum):
    """Types of optimization gaps."""

    MISSING_KEYWORD_TITLE = "missing_keyword_title"
    MISSING_KEYWORD_H1 = "missing_keyword_h1"
    MISSING_KEYWORD_HEADING = "missing_keyword_heading"
    MISSING_FAQ = "missing_faq"
    THIN_CONTENT = "thin_content"
    SEMANTIC_GAP = "semantic_gap"


@dataclass
class OptimizationGap:
    """A single optimization opportunity."""

    gap_type: GapType
    description: str
    location: str | None = None  # Node ID if applicable
    priority: int = 1  # 1 = highest priority
    recommendation: str = ""


@dataclass
class OptimizationPlan:
    """Complete plan for optimizing a document."""

    doc_id: str
    gaps: list[OptimizationGap] = field(default_factory=list)
    has_faq: bool = False
    needs_faq: bool = False
    total_gaps: int = 0
    priority_score: float = 0.0

    def __post_init__(self) -> None:
        self.total_gaps = len(self.gaps)
        if self.gaps:
            self.priority_score = sum(g.priority for g in self.gaps) / len(self.gaps)


def detect_gaps(
    doc: DocumentAST,
    keyword_analysis: KeywordAnalysis,
    context: BusinessContext | None = None,
) -> OptimizationPlan:
    """
    Detect all optimization gaps in a document.

    Analyzes the document for:
    - Missing keywords in key positions
    - Missing FAQ section
    - Thin content needing expansion
    - Semantic gaps

    Args:
        doc: Document AST to analyze
        keyword_analysis: Keyword mapping results
        context: Optional business context

    Returns:
        OptimizationPlan with prioritized gaps

    Example:
        >>> plan = detect_gaps(doc_ast, keyword_analysis)
        >>> for gap in plan.gaps:
        ...     print(f"{gap.gap_type}: {gap.description}")
    """
    raise NotImplementedError(
        "Gap detection not yet implemented. "
        "See docs/research/02-keyword-optimization.md section 4."
    )


def detect_missing_faq(doc: DocumentAST) -> bool:
    """
    Detect if the document is missing an FAQ section.

    Looks for:
    - "FAQ" or "Frequently Asked Questions" headings
    - Q&A patterns in content
    - "Question" + "Answer" structure

    Args:
        doc: Document AST to check

    Returns:
        True if FAQ section is missing

    Reference:
        docs/research/04-faq-generation.md section 2
    """
    raise NotImplementedError(
        "FAQ detection not yet implemented. "
        "See docs/research/04-faq-generation.md section 2."
    )

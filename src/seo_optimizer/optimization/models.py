"""
Optimization Models - Configuration and Result Data Structures

Defines all data structures for the content optimization engine:
- Configuration options (mode, limits, feature toggles)
- Change tracking and results
- FAQ and meta tag structures

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OptimizationMode(str, Enum):
    """Optimization intensity levels."""

    CONSERVATIVE = "conservative"  # Minimal changes, preserve voice
    BALANCED = "balanced"  # Moderate optimization
    AGGRESSIVE = "aggressive"  # Maximum optimization


class ContentType(str, Enum):
    """Content intent classification for optimization strategy."""

    INFORMATIONAL = "informational"  # Know intent
    COMMERCIAL = "commercial"  # Investigate intent
    TRANSACTIONAL = "transactional"  # Do intent
    LOCAL = "local"  # Go intent
    # Additional content types for broader use
    ARTICLE = "article"
    PRODUCT = "product"
    SERVICE = "service"
    LANDING_PAGE = "landing_page"
    BLOG_POST = "blog_post"


class ChangeType(str, Enum):
    """Types of optimization changes."""

    HEADING = "heading"
    KEYWORD = "keyword"
    ENTITY = "entity"
    READABILITY = "readability"
    FAQ = "faq"
    META = "meta"
    REDUNDANCY = "redundancy"
    STRUCTURE = "structure"


@dataclass
class OptimizationConfig:
    """Configuration for optimization behavior."""

    # Mode settings
    mode: OptimizationMode = OptimizationMode.BALANCED
    content_type: ContentType = ContentType.INFORMATIONAL

    # Keyword configuration
    primary_keyword: str | None = None
    secondary_keywords: list[str] = field(default_factory=list)
    semantic_entities: list[str] = field(default_factory=list)

    # Brand/Meta settings
    brand_name: str | None = None

    # Density thresholds (percentage)
    max_keyword_density: float = 2.5
    min_keyword_density: float = 1.0
    keyword_stuffing_threshold: float = 5.0  # >5% = penalty
    max_entity_density: float = 5.0  # >5% = entity stuffing

    # Readability targets
    target_reading_grade: float = 8.0  # Flesch-Kincaid grade level
    max_sentence_length: int = 25  # Words
    min_sentence_length: int = 5  # Words

    # Feature toggles
    generate_faq: bool = True
    optimize_headings: bool = True
    inject_keywords: bool = True
    inject_entities: bool = True
    improve_readability: bool = True
    generate_meta_tags: bool = True
    resolve_redundancy: bool = True
    filter_ai_vocabulary: bool = True

    # FAQ settings
    faq_count: int = 5
    max_faq_items: int = 5  # Maximum FAQ entries to generate
    faq_answer_min_words: int = 40
    faq_answer_max_words: int = 60
    enhance_existing_faq: bool = True  # Enhance existing FAQ sections

    # Meta tag settings
    title_max_chars: int = 60
    title_min_chars: int = 50
    description_max_chars: int = 158
    description_min_chars: int = 120

    # Safety limits
    max_changes_per_section: int = 5
    max_total_changes: int = 50
    preserve_quotes: bool = True  # Don't modify quoted text
    preserve_statistics: bool = True  # Don't modify numbers/stats
    preserve_proper_nouns: bool = True  # Don't modify names

    # Conservative mode settings
    conservative_similarity_threshold: float = 0.80  # Don't change if uncertain

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_keyword_density <= self.min_keyword_density:
            raise ValueError("max_keyword_density must be greater than min_keyword_density")
        if self.max_sentence_length <= self.min_sentence_length:
            raise ValueError("max_sentence_length must be greater than min_sentence_length")


@dataclass
class OptimizationChange:
    """Record of a single optimization change."""

    change_type: ChangeType
    location: str  # Where the change was made (e.g., "Heading 2: Introduction")
    original: str  # Original content
    optimized: str  # New content
    reason: str  # Why this change was made
    impact_score: float = 0.0  # Expected GEO score improvement (0-10)

    # Metadata
    section_id: str | None = None
    position: int | None = None  # Character position in document

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"[{self.change_type.value}] {self.location}: {self.reason}"

    @property
    def is_significant(self) -> bool:
        """Check if this is a significant change (impact > 1)."""
        return self.impact_score > 1.0


@dataclass
class FAQEntry:
    """Single FAQ question-answer pair."""

    question: str  # Natural language question
    answer: str  # 40-60 word direct answer (BLUF)
    extended_answer: str | None = None  # Additional context
    html_id: str = ""  # kebab-case ID for deep linking
    source_content: str | None = None  # Original content this was derived from
    source_section: str | None = None  # Section ID where this FAQ was derived from

    def __post_init__(self) -> None:
        """Generate HTML ID if not provided."""
        if not self.html_id and self.question:
            self.html_id = self._generate_html_id(self.question)

    @staticmethod
    def _generate_html_id(question: str) -> str:
        """Generate kebab-case ID from question."""
        import re

        # Remove punctuation and convert to lowercase
        clean = re.sub(r"[^\w\s-]", "", question.lower())
        # Replace spaces with hyphens
        clean = re.sub(r"\s+", "-", clean.strip())
        # Remove leading/trailing hyphens
        clean = clean.strip("-")
        # Limit length
        return clean[:50]

    @property
    def word_count(self) -> int:
        """Get answer word count."""
        return len(self.answer.split())

    @property
    def is_valid_length(self) -> bool:
        """Check if answer is within 40-60 word range."""
        wc = self.word_count
        return 40 <= wc <= 60


@dataclass
class MetaTags:
    """Generated meta tags for SEO."""

    title: str  # 50-60 chars, front-loaded keyword
    description: str  # 120-158 chars, includes CTA

    # Pixel width estimates
    title_pixel_width: int = 0  # Should be <600px
    description_pixel_width: int = 0  # Should be <920px desktop

    # Validation
    keyword_in_title: bool = False
    keyword_in_description: bool = False
    h1_alignment_score: float = 0.0  # 0-1, how well title aligns with H1

    def __post_init__(self) -> None:
        """Calculate pixel widths if not provided."""
        if self.title_pixel_width == 0:
            self.title_pixel_width = self._estimate_pixel_width(self.title)
        if self.description_pixel_width == 0:
            self.description_pixel_width = self._estimate_pixel_width(self.description)

    @staticmethod
    def _estimate_pixel_width(text: str) -> int:
        """
        Estimate pixel width for SERP display.

        Uses approximate character width (Arial 16px average: ~8px per char).
        Some characters are wider (W, M) and some narrower (i, l, t).
        """
        if not text:
            return 0

        # Character width multipliers (relative to average)
        wide_chars = set("WMQOGDAB")
        narrow_chars = set("iltfj1!|")

        total_width = 0.0
        base_width = 8.0  # Average character width in pixels

        for char in text:
            if char in wide_chars:
                total_width += base_width * 1.4
            elif char in narrow_chars:
                total_width += base_width * 0.5
            elif char == " ":
                total_width += base_width * 0.4
            else:
                total_width += base_width

        return int(total_width)

    @property
    def title_length(self) -> int:
        """Get title character count."""
        return len(self.title)

    @property
    def description_length(self) -> int:
        """Get description character count."""
        return len(self.description)

    @property
    def is_title_safe(self) -> bool:
        """Check if title is within safe limits."""
        return 50 <= self.title_length <= 60 and self.title_pixel_width < 600

    @property
    def is_description_safe(self) -> bool:
        """Check if description is within safe limits."""
        return 120 <= self.description_length <= 158 and self.description_pixel_width < 920


@dataclass
class GuardrailViolation:
    """Record of a guardrail violation."""

    rule: str  # Which rule was violated
    severity: str  # "warning" or "blocked"
    message: str  # Description of the violation
    original_change: OptimizationChange | None = None  # The change that was blocked


@dataclass
class OptimizationResult:
    """Complete optimization output."""

    config: OptimizationConfig | None = None
    changes: list[OptimizationChange] = field(default_factory=list)
    faq_entries: list[FAQEntry] = field(default_factory=list)
    meta_tags: MetaTags | None = None

    # Score tracking
    original_geo_score: float = 0.0
    optimized_geo_score: float = 0.0

    # Safety tracking
    guardrail_warnings: list[GuardrailViolation] = field(default_factory=list)
    changes_blocked: list[GuardrailViolation] = field(default_factory=list)

    @property
    def geo_score(self) -> float:
        """Alias for optimized_geo_score for backward compatibility."""
        return self.optimized_geo_score

    @property
    def score_improvement(self) -> float:
        """Calculate absolute score improvement."""
        return self.optimized_geo_score - self.original_geo_score

    @property
    def score_improvement_percentage(self) -> float:
        """Calculate percentage improvement."""
        if self.original_geo_score == 0:
            return 0.0
        return (self.score_improvement / self.original_geo_score) * 100

    @property
    def total_changes(self) -> int:
        """Get total number of changes applied."""
        return len(self.changes)

    @property
    def changes_by_type(self) -> dict[str, int]:
        """Get count of changes by type."""
        counts: dict[str, int] = {}
        for change in self.changes:
            type_name = change.change_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    @property
    def total_blocked(self) -> int:
        """Get total number of blocked changes."""
        return len(self.changes_blocked)

    @property
    def total_warnings(self) -> int:
        """Get total number of warnings."""
        return len(self.guardrail_warnings)

    def get_changes_by_type(self, change_type: ChangeType) -> list[OptimizationChange]:
        """Get all changes of a specific type."""
        return [c for c in self.changes if c.change_type == change_type]

    def get_high_impact_changes(self, threshold: float = 2.0) -> list[OptimizationChange]:
        """Get changes with impact score above threshold."""
        return [c for c in self.changes if c.impact_score >= threshold]

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Optimization Result Summary",
            "=" * 40,
            f"GEO Score: {self.original_geo_score:.1f} → {self.optimized_geo_score:.1f} "
            f"(+{self.score_improvement:.1f}, {self.score_improvement_percentage:.1f}%)",
            "",
            f"Changes Applied: {self.total_changes}",
        ]

        for type_name, count in self.changes_by_type.items():
            lines.append(f"  - {type_name}: {count}")

        if self.total_blocked > 0:
            lines.append(f"\nChanges Blocked: {self.total_blocked}")

        if self.total_warnings > 0:
            lines.append(f"Warnings: {self.total_warnings}")

        if self.faq_entries:
            lines.append(f"\nFAQ Entries Generated: {len(self.faq_entries)}")

        if self.meta_tags:
            lines.append("\nMeta Tags Generated:")
            lines.append(f"  Title ({self.meta_tags.title_length} chars): {self.meta_tags.title}")
            lines.append(
                f"  Description ({self.meta_tags.description_length} chars): "
                f"{self.meta_tags.description[:80]}..."
            )

        return "\n".join(lines)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""

    # Execution status
    success: bool = True

    # AST results
    original_ast: Any = None  # DocumentAST before optimization
    optimized_ast: Any = None  # DocumentAST after optimization

    # Optimization result
    optimization_result: OptimizationResult | None = None

    # Output path
    output_path: Any = None  # Path | None

    # Change tracking
    change_map: dict | None = None

    # Processing metadata
    execution_time: float = 0.0

    # Status messages
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Backward compatibility fields
    input_path: str | None = None
    processing_time_seconds: float = 0.0
    highlight_count: int = 0

    @property
    def original_geo_score(self) -> float:
        """Get original GEO score."""
        if self.optimization_result:
            return self.optimization_result.original_geo_score
        return 0.0

    @property
    def optimized_geo_score(self) -> float:
        """Get optimized GEO score."""
        if self.optimization_result:
            return self.optimization_result.optimized_geo_score
        return 0.0

    @property
    def improvement(self) -> float:
        """Get absolute improvement."""
        if self.optimization_result:
            return self.optimization_result.score_improvement
        return 0.0

    @property
    def improvement_percentage(self) -> float:
        """Get percentage improvement."""
        if self.optimization_result:
            return self.optimization_result.score_improvement_percentage
        return 0.0

    @property
    def total_changes(self) -> int:
        """Get total changes."""
        if self.optimization_result:
            return self.optimization_result.total_changes
        return 0

    @property
    def blocked_changes(self) -> int:
        """Get blocked changes count."""
        if self.optimization_result:
            return self.optimization_result.total_blocked
        return 0

    def to_summary(self) -> str:
        """Generate complete pipeline summary."""
        lines = [
            "Pipeline Execution Summary",
            "=" * 40,
            f"Success: {self.success}",
            f"Output: {self.output_path}",
            f"Execution Time: {self.execution_time:.2f}s",
        ]

        if self.optimization_result:
            lines.append("")
            lines.append(self.optimization_result.to_summary())

        if self.highlight_count > 0:
            lines.append(f"\nHighlighted Changes: {self.highlight_count}")

        if self.warnings:
            lines.append(f"\nWarnings: {len(self.warnings)}")

        if self.errors:
            lines.append(f"\nErrors: {len(self.errors)}")

        return "\n".join(lines)

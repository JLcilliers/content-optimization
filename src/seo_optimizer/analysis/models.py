"""
Analysis Models - Data Structures for SEO Analysis

These models represent the output of the SEO analysis engine:
- GEO-Metric composite score
- Individual scorer outputs (SEO, Semantic, AI, Readability)
- Issues with severity classification
- Actionable recommendations

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IssueSeverity(str, Enum):
    """Severity levels for detected issues."""

    CRITICAL = "critical"  # Blocks ranking potential
    WARNING = "warning"  # Degrades performance
    INFO = "info"  # Minor optimization opportunity


class IssueCategory(str, Enum):
    """Categories of issues that can be detected."""

    STRUCTURE = "structure"
    KEYWORD = "keyword"
    ENTITY = "entity"
    READABILITY = "readability"
    AI_COMPATIBILITY = "ai_compatibility"
    REDUNDANCY = "redundancy"


@dataclass
class Issue:
    """A single issue detected during analysis."""

    category: IssueCategory
    severity: IssueSeverity
    message: str
    location: str | None = None  # e.g., "Heading 2: Introduction"
    current_value: str | None = None
    target_value: str | None = None
    fix_suggestion: str | None = None

    def __str__(self) -> str:
        """String representation of the issue."""
        prefix = f"[{self.severity.value.upper()}]"
        loc = f" at {self.location}" if self.location else ""
        return f"{prefix} {self.message}{loc}"


@dataclass
class EntityMatch:
    """A named entity found in the document."""

    text: str
    entity_type: str  # PERSON, ORG, PRODUCT, CONCEPT, LOCATION, EVENT
    start_char: int
    end_char: int
    confidence: float = 1.0

    @property
    def length(self) -> int:
        """Length of the entity text."""
        return self.end_char - self.start_char


@dataclass
class KeywordConfig:
    """Configuration for target keywords."""

    primary_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)
    semantic_entities: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate keyword configuration."""
        if not self.primary_keyword:
            raise ValueError("Primary keyword is required")


@dataclass
class KeywordAnalysis:
    """Analysis of keyword presence and placement."""

    primary_keyword: str | None = None
    primary_found: bool = False
    primary_locations: list[str] = field(default_factory=list)  # ["title", "h1", "first_100_words"]
    secondary_keywords: list[str] = field(default_factory=list)
    secondary_found: list[str] = field(default_factory=list)
    semantic_entities: list[str] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    keyword_density: float = 0.0  # Percentage
    passes_135_rule: bool = False

    @property
    def primary_placement_score(self) -> float:
        """Score based on where primary keyword appears (0-1)."""
        weights = {
            "title": 1.0,
            "h1": 0.9,
            "url_slug": 0.8,
            "first_100_words": 0.7,
            "h2_h3": 0.6,
            "body": 0.4,
            "alt_text": 0.3,
        }
        if not self.primary_locations:
            return 0.0
        return max(weights.get(loc, 0.3) for loc in self.primary_locations)


@dataclass
class HeadingAnalysis:
    """Analysis of document heading structure."""

    h1_count: int = 0
    h1_text: str | None = None
    hierarchy_valid: bool = True  # No skipped levels
    heading_density: float = 0.0  # Headings per 300 words
    headings_list: list[tuple[int, str]] = field(default_factory=list)  # [(level, text), ...]
    issues: list[str] = field(default_factory=list)

    @property
    def has_valid_h1(self) -> bool:
        """Check if document has exactly one H1."""
        return self.h1_count == 1


@dataclass
class SEOScore:
    """
    Traditional SEO metrics - 20% of total GEO score.

    Evaluates:
    - Keyword placement and density
    - Heading structure
    - Internal link readiness
    """

    keyword_score: float = 0.0  # 0-100
    heading_score: float = 0.0  # 0-100
    link_readiness_score: float = 0.0  # 0-100
    total: float = 0.0  # Weighted average 0-100
    keyword_analysis: KeywordAnalysis = field(default_factory=KeywordAnalysis)
    heading_analysis: HeadingAnalysis = field(default_factory=HeadingAnalysis)
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate total score if not set."""
        if self.total == 0.0 and any(
            [self.keyword_score, self.heading_score, self.link_readiness_score]
        ):
            self.total = (
                self.keyword_score * 0.4
                + self.heading_score * 0.4
                + self.link_readiness_score * 0.2
            )


@dataclass
class SemanticScore:
    """
    Semantic depth metrics - 30% of total GEO score.

    Evaluates:
    - Topic coverage via cosine similarity
    - Information gain (unique entities)
    - Entity density
    """

    topic_coverage: float = 0.0  # Cosine similarity 0-1
    information_gain: float = 0.0  # Unique entities ratio 0-1
    entity_density: float = 0.0  # Entities per section
    entity_saturation: bool = False  # True if >5% (over-optimization)
    total: float = 0.0  # 0-100
    entities_found: list[EntityMatch] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)  # Expected but not found
    redundant_sections: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (sec1, sec2, similarity)
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate total score if not set."""
        if self.total == 0.0 and any([self.topic_coverage, self.information_gain]):
            # Topic coverage is most important
            base_score = self.topic_coverage * 100 * 0.5
            # Information gain adds value
            base_score += self.information_gain * 100 * 0.3
            # Entity density contributes
            density_score = min(self.entity_density / 0.02, 1.0) * 100 * 0.2
            # Apply saturation penalty
            if self.entity_saturation:
                base_score *= 0.7
            self.total = base_score + density_score


@dataclass
class AIScore:
    """
    AI compatibility metrics - 30% of total GEO score.

    Evaluates:
    - Chunk clarity (self-contained segments)
    - BLUF compliance (Bottom Line Up Front)
    - Extraction friendliness (lists, tables)
    - Redundancy penalty
    """

    chunk_clarity: float = 0.0  # Ratio of self-contained chunks 0-1
    answer_completeness: float = 0.0  # BLUF test pass rate 0-1
    extraction_friendliness: float = 0.0  # List/table density 0-1
    redundancy_penalty: float = 0.0  # Sections with >0.90 similarity
    total: float = 0.0  # 0-100
    problematic_chunks: list[str] = field(default_factory=list)  # Chunks with pronoun issues
    redundant_sections: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (sec1, sec2, similarity)
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate total score if not set."""
        if self.total == 0.0 and any(
            [self.chunk_clarity, self.answer_completeness, self.extraction_friendliness]
        ):
            base_score = (
                self.chunk_clarity * 30
                + self.answer_completeness * 40
                + self.extraction_friendliness * 30
            )
            # Apply redundancy penalty
            self.total = base_score * (1 - self.redundancy_penalty)


@dataclass
class ReadabilityScore:
    """
    Readability & UX metrics - 20% of total GEO score.

    Evaluates:
    - Sentence length
    - Active voice usage
    - Flesch-Kincaid grade level
    """

    avg_sentence_length: float = 0.0  # Words per sentence
    active_voice_ratio: float = 0.0  # 0-1
    flesch_kincaid_grade: float = 0.0  # Grade level
    total: float = 0.0  # 0-100
    complex_sentences: list[str] = field(default_factory=list)  # Sentences > 25 words
    passive_sentences: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate total score if not set."""
        if self.total == 0.0 and self.avg_sentence_length > 0:
            # Sentence length score (optimal around 15-20 words)
            if self.avg_sentence_length <= 20:
                length_score = 100
            elif self.avg_sentence_length <= 25:
                length_score = 80
            elif self.avg_sentence_length <= 35:
                length_score = 60
            else:
                length_score = 40

            # Active voice score
            voice_score = self.active_voice_ratio * 100

            # Grade level score (optimal 8-12)
            if 8 <= self.flesch_kincaid_grade <= 12:
                grade_score = 100
            elif 6 <= self.flesch_kincaid_grade <= 14:
                grade_score = 80
            else:
                grade_score = 60

            self.total = length_score * 0.3 + voice_score * 0.4 + grade_score * 0.3


@dataclass
class GEOScore:
    """
    Composite GEO-Metric score.

    Formula:
    GEO = (0.20 × SEO) + (0.30 × Semantic) + (0.30 × AI) + (0.20 × Readability)
    """

    seo_score: SEOScore = field(default_factory=SEOScore)  # 20% weight
    semantic_score: SemanticScore = field(default_factory=SemanticScore)  # 30% weight
    ai_score: AIScore = field(default_factory=AIScore)  # 30% weight
    readability_score: ReadabilityScore = field(default_factory=ReadabilityScore)  # 20% weight
    total: float = 0.0  # Weighted composite 0-100
    all_issues: list[Issue] = field(default_factory=list)
    _confidence_rating: str = ""

    def __post_init__(self) -> None:
        """Calculate total score if not set."""
        if self.total == 0.0:
            self.total = (
                0.20 * self.seo_score.total
                + 0.30 * self.semantic_score.total
                + 0.30 * self.ai_score.total
                + 0.20 * self.readability_score.total
            )

        # Aggregate all issues
        if not self.all_issues:
            self.all_issues = (
                self.seo_score.issues
                + self.semantic_score.issues
                + self.ai_score.issues
                + self.readability_score.issues
            )

    @property
    def confidence_rating(self) -> str:
        """Get confidence rating based on total score."""
        if self._confidence_rating:
            return self._confidence_rating
        if self.total >= 90:
            return "Excellent"
        elif self.total >= 70:
            return "Good"
        elif self.total >= 40:
            return "Fair"
        else:
            return "Poor"

    @property
    def critical_issues(self) -> list[Issue]:
        """Get only critical issues."""
        return [i for i in self.all_issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def warning_issues(self) -> list[Issue]:
        """Get only warning issues."""
        return [i for i in self.all_issues if i.severity == IssueSeverity.WARNING]


@dataclass
class DocumentStats:
    """Statistics about the document."""

    word_count: int = 0
    paragraph_count: int = 0
    sentence_count: int = 0
    heading_count: int = 0
    list_count: int = 0
    table_count: int = 0
    link_count: int = 0
    image_count: int = 0

    @property
    def avg_paragraph_length(self) -> float:
        """Average words per paragraph."""
        if self.paragraph_count == 0:
            return 0.0
        return self.word_count / self.paragraph_count

    @property
    def avg_sentence_length(self) -> float:
        """Average words per sentence."""
        if self.sentence_count == 0:
            return 0.0
        return self.word_count / self.sentence_count


@dataclass
class AnalysisResult:
    """Complete analysis output."""

    document_stats: DocumentStats = field(default_factory=DocumentStats)
    geo_score: GEOScore = field(default_factory=GEOScore)
    recommendations: list[str] = field(default_factory=list)  # Prioritized action items
    before_after_comparison: dict[str, Any] | None = None  # If comparing versions

    @property
    def summary(self) -> str:
        """Generate a brief summary of the analysis."""
        return (
            f"GEO Score: {self.geo_score.total:.1f}/100 ({self.geo_score.confidence_rating})\n"
            f"  - SEO: {self.geo_score.seo_score.total:.1f}\n"
            f"  - Semantic: {self.geo_score.semantic_score.total:.1f}\n"
            f"  - AI Compatibility: {self.geo_score.ai_score.total:.1f}\n"
            f"  - Readability: {self.geo_score.readability_score.total:.1f}\n"
            f"Issues: {len(self.geo_score.critical_issues)} critical, "
            f"{len(self.geo_score.warning_issues)} warnings"
        )


@dataclass
class VersionComparison:
    """Comparison between original and optimized versions."""

    original_score: float
    optimized_score: float
    improvement: float
    key_changes: list[str] = field(default_factory=list)
    issues_fixed: list[str] = field(default_factory=list)
    new_issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate improvement if not set."""
        if self.improvement == 0.0:
            self.improvement = self.optimized_score - self.original_score

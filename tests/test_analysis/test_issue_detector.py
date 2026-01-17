"""
Tests for Issue Detector Module

Tests cross-cutting issue detection.
"""

import pytest

from seo_optimizer.analysis.issue_detector import (
    IssueDetector,
    IssueDetectorConfig,
    detect_issues,
)
from seo_optimizer.analysis.models import (
    AIScore,
    GEOScore,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordConfig,
    ReadabilityScore,
    SemanticScore,
    SEOScore,
)
from seo_optimizer.ingestion.models import (
    ContentNode,
    DocumentAST,
    DocumentMetadata,
    NodeType,
    PositionInfo,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector() -> IssueDetector:
    """Create an issue detector instance."""
    return IssueDetector()


@pytest.fixture
def simple_keywords() -> KeywordConfig:
    """Simple keyword configuration."""
    return KeywordConfig(
        primary_keyword="cloud computing",
        secondary_keywords=["AWS", "Azure"],
    )


@pytest.fixture
def empty_geo_score() -> GEOScore:
    """Empty GEO score with no issues."""
    return GEOScore(
        seo_score=SEOScore(total=75),
        semantic_score=SemanticScore(total=70),
        ai_score=AIScore(total=65),
        readability_score=ReadabilityScore(total=80),
        total=72.5,
        all_issues=[],
    )


def create_test_ast(
    full_text: str,
    nodes: list[ContentNode] | None = None,
) -> DocumentAST:
    """Helper to create test AST."""
    return DocumentAST(
        doc_id="test_doc",
        nodes=nodes or [],
        full_text=full_text,
        char_count=len(full_text),
        metadata=DocumentMetadata(),
    )


def create_heading_node(text: str, level: int, position: int = 0) -> ContentNode:
    """Helper to create heading node."""
    return ContentNode(
        node_id=f"h{level}_{position}",
        node_type=NodeType.HEADING,
        text_content=text,
        position=PositionInfo(
            position_id=f"pos_{position}",
            start_char=0,
            end_char=len(text),
        ),
        metadata={"level": level},
    )


def create_paragraph_node(text: str, position: int = 0) -> ContentNode:
    """Helper to create paragraph node."""
    return ContentNode(
        node_id=f"p_{position}",
        node_type=NodeType.PARAGRAPH,
        text_content=text,
        position=PositionInfo(
            position_id=f"pos_{position}",
            start_char=0,
            end_char=len(text),
        ),
    )


# =============================================================================
# Thin Content Tests
# =============================================================================


class TestThinContent:
    """Tests for thin content detection."""

    def test_thin_content_critical(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test critical issue for very thin content."""
        # Only 50 words
        text = " ".join(["word"] * 50)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score)

        thin_issues = [i for i in issues if "thin content" in i.message.lower()]
        assert len(thin_issues) > 0
        assert thin_issues[0].severity == IssueSeverity.CRITICAL

    def test_suboptimal_length_info(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test info issue for suboptimal length."""
        # 500 words - below optimal but not thin
        text = " ".join(["word"] * 500)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score)

        length_issues = [i for i in issues if "below optimal" in i.message.lower()]
        assert len(length_issues) > 0
        assert length_issues[0].severity == IssueSeverity.INFO

    def test_optimal_length_no_issue(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test no issue for optimal length."""
        # 1200 words - optimal
        text = " ".join(["word"] * 1200)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score)

        thin_issues = [i for i in issues if "thin" in i.message.lower() or "below optimal" in i.message.lower()]
        assert len(thin_issues) == 0


# =============================================================================
# FAQ Detection Tests
# =============================================================================


class TestFAQDetection:
    """Tests for missing FAQ section detection."""

    def test_missing_faq_issue(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test issue raised when FAQ is missing."""
        nodes = [
            create_heading_node("Introduction", level=1),
            create_heading_node("Features", level=2),
        ]
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text, nodes=nodes)
        issues = detector.detect_all(ast, empty_geo_score)

        faq_issues = [i for i in issues if "faq" in i.message.lower()]
        assert len(faq_issues) > 0

    def test_faq_present_no_issue(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test no issue when FAQ section exists."""
        nodes = [
            create_heading_node("Introduction", level=1),
            create_heading_node("Frequently Asked Questions", level=2),
        ]
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text, nodes=nodes)
        issues = detector.detect_all(ast, empty_geo_score)

        faq_issues = [i for i in issues if "faq" in i.message.lower()]
        assert len(faq_issues) == 0

    def test_faq_variant_detected(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test FAQ detection with variant headings."""
        nodes = [
            create_heading_node("Common Questions", level=2),
        ]
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text, nodes=nodes)
        issues = detector.detect_all(ast, empty_geo_score)

        faq_issues = [i for i in issues if "faq" in i.message.lower()]
        assert len(faq_issues) == 0  # "Common Questions" should match


# =============================================================================
# Keyword Stuffing Tests
# =============================================================================


class TestKeywordStuffing:
    """Tests for keyword stuffing detection."""

    def test_keyword_stuffing_detected(
        self,
        detector: IssueDetector,
        empty_geo_score: GEOScore,
        simple_keywords: KeywordConfig,
    ) -> None:
        """Test detection of keyword stuffing."""
        # Repeat keyword excessively
        text = "cloud computing " * 20 + " ".join(["word"] * 80)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score, simple_keywords)

        stuffing_issues = [i for i in issues if "stuffing" in i.message.lower()]
        assert len(stuffing_issues) > 0

    def test_normal_density_no_issue(
        self,
        detector: IssueDetector,
        empty_geo_score: GEOScore,
        simple_keywords: KeywordConfig,
    ) -> None:
        """Test no issue for normal keyword density."""
        # Use keyword 2 times in 100 words = 2% density
        text = "cloud computing " + " ".join(["word"] * 48) + " cloud computing " + " ".join(["word"] * 48)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score, simple_keywords)

        stuffing_issues = [i for i in issues if "stuffing" in i.message.lower()]
        assert len(stuffing_issues) == 0


# =============================================================================
# Meta Description Tests
# =============================================================================


class TestMetaDescription:
    """Tests for meta description issues."""

    def test_missing_meta_description(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test issue for missing meta description."""
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, empty_geo_score, meta_description=None)

        meta_issues = [i for i in issues if "meta description" in i.message.lower()]
        assert len(meta_issues) > 0

    def test_short_meta_description(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test issue for short meta description."""
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)
        issues = detector.detect_all(
            ast, empty_geo_score, meta_description="Too short"
        )

        meta_issues = [i for i in issues if "short" in i.message.lower() and "meta" in i.message.lower()]
        assert len(meta_issues) > 0

    def test_long_meta_description(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test issue for long meta description."""
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)
        long_meta = "x" * 200
        issues = detector.detect_all(ast, empty_geo_score, meta_description=long_meta)

        meta_issues = [i for i in issues if "long" in i.message.lower() and "meta" in i.message.lower()]
        assert len(meta_issues) > 0


# =============================================================================
# Structural Issues Tests
# =============================================================================


class TestStructuralIssues:
    """Tests for structural issue detection."""

    def test_no_headings_critical(
        self, detector: IssueDetector, empty_geo_score: GEOScore
    ) -> None:
        """Test critical issue for no headings."""
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text, nodes=[])
        issues = detector.detect_all(ast, empty_geo_score)

        heading_issues = [i for i in issues if "heading" in i.message.lower()]
        assert len(heading_issues) > 0


# =============================================================================
# Issue Aggregation Tests
# =============================================================================


class TestIssueAggregation:
    """Tests for issue aggregation and sorting."""

    def test_issues_sorted_by_severity(
        self, detector: IssueDetector
    ) -> None:
        """Test that issues are sorted by severity."""
        geo_score = GEOScore(
            all_issues=[
                Issue(IssueCategory.KEYWORD, IssueSeverity.INFO, "Info issue"),
                Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Critical issue"),
                Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Warning issue"),
            ]
        )
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, geo_score)

        # Critical should come first
        severity_order = [i.severity for i in issues[:3]]
        assert severity_order[0] == IssueSeverity.CRITICAL

    def test_duplicate_issues_removed(
        self, detector: IssueDetector
    ) -> None:
        """Test that duplicate issues are deduplicated."""
        geo_score = GEOScore(
            all_issues=[
                Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Same message"),
                Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Same message"),
            ]
        )
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)
        issues = detector.detect_all(ast, geo_score)

        same_messages = [i for i in issues if i.message == "Same message"]
        assert len(same_messages) <= 1


# =============================================================================
# Helper Methods Tests
# =============================================================================


class TestIssueFiltering:
    """Tests for issue filtering methods."""

    def test_get_critical_issues(self, detector: IssueDetector) -> None:
        """Test filtering critical issues."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Critical 1"),
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Warning 1"),
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Critical 2"),
        ]

        critical = detector.get_critical_issues(issues)

        assert len(critical) == 2
        assert all(i.severity == IssueSeverity.CRITICAL for i in critical)

    def test_get_issues_by_category(self, detector: IssueDetector) -> None:
        """Test filtering issues by category."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Keyword issue"),
            Issue(IssueCategory.STRUCTURE, IssueSeverity.WARNING, "Structure issue"),
            Issue(IssueCategory.KEYWORD, IssueSeverity.INFO, "Keyword info"),
        ]

        keyword_issues = detector.get_issues_by_category(issues, IssueCategory.KEYWORD)

        assert len(keyword_issues) == 2

    def test_format_issue_summary(self, detector: IssueDetector) -> None:
        """Test issue summary formatting."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Critical issue"),
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Warning issue"),
        ]

        summary = detector.format_issue_summary(issues)

        assert "2 total" in summary
        assert "1 Critical" in summary
        assert "1 Warning" in summary


# =============================================================================
# Configuration Tests
# =============================================================================


class TestIssueDetectorConfig:
    """Tests for detector configuration."""

    def test_custom_word_count(self) -> None:
        """Test custom word count thresholds."""
        config = IssueDetectorConfig(
            min_word_count=500,
            optimal_word_count=1500,
        )
        detector = IssueDetector(config)

        assert detector.config.min_word_count == 500

    def test_disable_faq_check(self) -> None:
        """Test disabling FAQ requirement."""
        config = IssueDetectorConfig(require_faq_section=False)
        detector = IssueDetector(config)
        geo_score = GEOScore()
        text = " ".join(["word"] * 400)
        ast = create_test_ast(text)

        issues = detector.detect_all(ast, geo_score)

        faq_issues = [i for i in issues if "faq" in i.message.lower()]
        assert len(faq_issues) == 0


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDetectIssuesFunction:
    """Tests for convenience function."""

    def test_detect_issues_basic(self) -> None:
        """Test basic issue detection function."""
        geo_score = GEOScore()
        text = " ".join(["word"] * 100)  # Thin content
        ast = create_test_ast(text)

        issues = detect_issues(ast, geo_score)

        assert len(issues) > 0
        assert any("thin" in i.message.lower() for i in issues)

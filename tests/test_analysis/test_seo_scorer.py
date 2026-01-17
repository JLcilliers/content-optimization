"""
Tests for SEO Scorer Module

Tests traditional SEO scoring (20% of GEO score).
"""

import pytest
from dataclasses import field

from seo_optimizer.analysis.seo_scorer import (
    SEOScorer,
    SEOScorerConfig,
    score_seo,
)
from seo_optimizer.analysis.models import KeywordConfig
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
def scorer() -> SEOScorer:
    """Create an SEO scorer instance."""
    return SEOScorer()


@pytest.fixture
def simple_keywords() -> KeywordConfig:
    """Simple keyword configuration."""
    return KeywordConfig(
        primary_keyword="cloud computing",
        secondary_keywords=["AWS", "Azure", "serverless"],
        semantic_entities=["data center", "virtualization", "scalability"],
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
# Basic Scoring Tests
# =============================================================================


class TestSEOScorerBasic:
    """Basic SEO scoring tests."""

    def test_score_empty_document(
        self, scorer: SEOScorer, simple_keywords: KeywordConfig
    ) -> None:
        """Test scoring an empty document."""
        ast = create_test_ast("")
        score = scorer.score(ast, simple_keywords)

        assert score.total >= 0
        assert score.keyword_score >= 0
        assert score.heading_score >= 0

    def test_score_returns_seo_score(
        self, scorer: SEOScorer, simple_keywords: KeywordConfig
    ) -> None:
        """Test that scoring returns SEOScore object."""
        ast = create_test_ast("This is about cloud computing and AWS services.")
        score = scorer.score(ast, simple_keywords)

        assert hasattr(score, "keyword_score")
        assert hasattr(score, "heading_score")
        assert hasattr(score, "link_readiness_score")
        assert hasattr(score, "total")

    def test_score_has_keyword_analysis(
        self, scorer: SEOScorer, simple_keywords: KeywordConfig
    ) -> None:
        """Test that keyword analysis is included."""
        ast = create_test_ast("Cloud computing is great for AWS users.")
        score = scorer.score(ast, simple_keywords)

        assert score.keyword_analysis is not None
        assert score.keyword_analysis.primary_keyword == "cloud computing"


# =============================================================================
# Keyword Scoring Tests
# =============================================================================


class TestKeywordScoring:
    """Tests for keyword-related scoring."""

    def test_primary_keyword_found(
        self, scorer: SEOScorer, simple_keywords: KeywordConfig
    ) -> None:
        """Test detection of primary keyword."""
        ast = create_test_ast("Cloud computing is revolutionizing the industry.")
        score = scorer.score(ast, simple_keywords)

        assert score.keyword_analysis.primary_found is True

    def test_primary_keyword_not_found(
        self, scorer: SEOScorer, simple_keywords: KeywordConfig
    ) -> None:
        """Test when primary keyword is missing."""
        ast = create_test_ast("Technology is changing rapidly.")
        score = scorer.score(ast, simple_keywords)

        assert score.keyword_analysis.primary_found is False

    def test_keyword_in_title(self, scorer: SEOScorer) -> None:
        """Test keyword detection in title."""
        keywords = KeywordConfig(primary_keyword="SEO")
        ast = create_test_ast("Learn about search engine optimization.")
        score = scorer.score(ast, keywords, title="SEO Best Practices Guide")

        assert "title" in score.keyword_analysis.primary_locations

    def test_keyword_in_h1(self, scorer: SEOScorer) -> None:
        """Test keyword detection in H1."""
        keywords = KeywordConfig(primary_keyword="python")
        h1_node = create_heading_node("Learn Python Programming", level=1)
        ast = create_test_ast(
            "Learn Python Programming. Python is great.",
            nodes=[h1_node],
        )
        score = scorer.score(ast, keywords)

        assert "h1" in score.keyword_analysis.primary_locations

    def test_keyword_in_first_100_words(self, scorer: SEOScorer) -> None:
        """Test keyword detection in first 100 words."""
        keywords = KeywordConfig(primary_keyword="testing")
        # Create text with keyword in first 100 words
        text = "Testing is important. " + "More content. " * 50
        ast = create_test_ast(text)
        score = scorer.score(ast, keywords)

        assert "first_100_words" in score.keyword_analysis.primary_locations

    def test_keyword_density_calculation(self, scorer: SEOScorer) -> None:
        """Test keyword density calculation."""
        keywords = KeywordConfig(primary_keyword="test")
        # 100 words with "test" appearing 2 times = 2% density
        text = "test " + "word " * 48 + "test " + "word " * 48
        ast = create_test_ast(text)
        score = scorer.score(ast, keywords)

        assert 1.5 < score.keyword_analysis.keyword_density < 2.5

    def test_secondary_keywords_found(self, scorer: SEOScorer) -> None:
        """Test detection of secondary keywords."""
        keywords = KeywordConfig(
            primary_keyword="cloud",
            secondary_keywords=["AWS", "Azure", "GCP"],
        )
        ast = create_test_ast("Cloud services include AWS and Azure platforms.")
        score = scorer.score(ast, keywords)

        assert "AWS" in score.keyword_analysis.secondary_found
        assert "Azure" in score.keyword_analysis.secondary_found
        assert "GCP" not in score.keyword_analysis.secondary_found

    def test_135_rule_pass(self, scorer: SEOScorer) -> None:
        """Test 1-3-5 rule compliance detection."""
        keywords = KeywordConfig(
            primary_keyword="cloud",
            secondary_keywords=["AWS", "Azure", "GCP"],
            semantic_entities=["server", "data", "network", "storage", "compute"],
        )
        text = "Cloud AWS Azure GCP server data network storage compute"
        ast = create_test_ast(text)
        score = scorer.score(ast, keywords)

        assert score.keyword_analysis.passes_135_rule is True

    def test_135_rule_fail(self, scorer: SEOScorer) -> None:
        """Test 1-3-5 rule failure."""
        keywords = KeywordConfig(
            primary_keyword="cloud",
            secondary_keywords=["AWS", "Azure", "GCP"],
            semantic_entities=["server", "data", "network", "storage", "compute"],
        )
        # Missing most keywords
        ast = create_test_ast("This text doesn't mention the keywords.")
        score = scorer.score(ast, keywords)

        assert score.keyword_analysis.passes_135_rule is False


# =============================================================================
# Heading Scoring Tests
# =============================================================================


class TestHeadingScoring:
    """Tests for heading-related scoring."""

    def test_single_h1_valid(self, scorer: SEOScorer) -> None:
        """Test single H1 is valid."""
        keywords = KeywordConfig(primary_keyword="test")
        h1 = create_heading_node("Test Heading", level=1)
        ast = create_test_ast("Test content", nodes=[h1])
        score = scorer.score(ast, keywords)

        assert score.heading_analysis.has_valid_h1 is True
        assert score.heading_analysis.h1_count == 1

    def test_multiple_h1_invalid(self, scorer: SEOScorer) -> None:
        """Test multiple H1s are invalid."""
        keywords = KeywordConfig(primary_keyword="test")
        h1_1 = create_heading_node("First H1", level=1, position=0)
        h1_2 = create_heading_node("Second H1", level=1, position=1)
        ast = create_test_ast("Test content", nodes=[h1_1, h1_2])
        score = scorer.score(ast, keywords)

        assert score.heading_analysis.has_valid_h1 is False
        assert score.heading_analysis.h1_count == 2

    def test_no_h1(self, scorer: SEOScorer) -> None:
        """Test document with no H1."""
        keywords = KeywordConfig(primary_keyword="test")
        h2 = create_heading_node("Just an H2", level=2)
        ast = create_test_ast("Test content", nodes=[h2])
        score = scorer.score(ast, keywords)

        assert score.heading_analysis.has_valid_h1 is False
        assert score.heading_analysis.h1_count == 0

    def test_heading_hierarchy_valid(self, scorer: SEOScorer) -> None:
        """Test valid heading hierarchy."""
        keywords = KeywordConfig(primary_keyword="test")
        nodes = [
            create_heading_node("H1", level=1, position=0),
            create_heading_node("H2", level=2, position=1),
            create_heading_node("H3", level=3, position=2),
        ]
        ast = create_test_ast("Test content", nodes=nodes)
        score = scorer.score(ast, keywords)

        assert score.heading_analysis.hierarchy_valid is True

    def test_heading_hierarchy_invalid_skip(self, scorer: SEOScorer) -> None:
        """Test invalid heading hierarchy (skipped level)."""
        keywords = KeywordConfig(primary_keyword="test")
        nodes = [
            create_heading_node("H1", level=1, position=0),
            create_heading_node("H3 (skipped H2)", level=3, position=1),
        ]
        ast = create_test_ast("Test content", nodes=nodes)
        score = scorer.score(ast, keywords)

        assert score.heading_analysis.hierarchy_valid is False


# =============================================================================
# Issue Detection Tests
# =============================================================================


class TestSEOIssueDetection:
    """Tests for SEO issue detection."""

    def test_missing_primary_keyword_issue(self, scorer: SEOScorer) -> None:
        """Test issue raised for missing primary keyword."""
        keywords = KeywordConfig(primary_keyword="specific term")
        ast = create_test_ast("This text doesn't contain the keyword.")
        score = scorer.score(ast, keywords)

        critical_issues = [i for i in score.issues if i.severity.value == "critical"]
        assert any("primary keyword" in i.message.lower() for i in critical_issues)

    def test_missing_h1_issue(self, scorer: SEOScorer) -> None:
        """Test issue raised for missing H1."""
        keywords = KeywordConfig(primary_keyword="test")
        ast = create_test_ast("No headings here.")
        score = scorer.score(ast, keywords)

        critical_issues = [i for i in score.issues if i.severity.value == "critical"]
        assert any("h1" in i.message.lower() for i in critical_issues)

    def test_keyword_not_in_first_100_words_warning(self, scorer: SEOScorer) -> None:
        """Test warning when keyword not in first 100 words."""
        keywords = KeywordConfig(primary_keyword="conclusion")
        # Keyword appears after first 100 words
        text = "word " * 150 + "conclusion"
        ast = create_test_ast(text)
        score = scorer.score(ast, keywords)

        warnings = [i for i in score.issues if i.severity.value == "warning"]
        assert any("first 100 words" in i.message.lower() for i in warnings)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSEOScorerConfig:
    """Tests for scorer configuration."""

    def test_custom_density_thresholds(self) -> None:
        """Test custom keyword density thresholds."""
        config = SEOScorerConfig(
            min_keyword_density=2.0,
            max_keyword_density=4.0,
        )
        scorer = SEOScorer(config)

        assert scorer.config.min_keyword_density == 2.0
        assert scorer.config.max_keyword_density == 4.0

    def test_custom_weights(self) -> None:
        """Test custom score weights."""
        config = SEOScorerConfig(
            keyword_weight=0.5,
            heading_weight=0.3,
            link_weight=0.2,
        )
        scorer = SEOScorer(config)

        assert scorer.config.keyword_weight == 0.5


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestScoreSEOFunction:
    """Tests for convenience function."""

    def test_score_seo_basic(self) -> None:
        """Test basic scoring function."""
        ast = create_test_ast("This is about python programming.")
        score = score_seo(ast, primary_keyword="python")

        assert score.keyword_analysis.primary_found is True

    def test_score_seo_with_secondary(self) -> None:
        """Test scoring with secondary keywords."""
        ast = create_test_ast("Python programming with Django and Flask frameworks.")
        score = score_seo(
            ast,
            primary_keyword="python",
            secondary_keywords=["Django", "Flask"],
        )

        assert "Django" in score.keyword_analysis.secondary_found
        assert "Flask" in score.keyword_analysis.secondary_found

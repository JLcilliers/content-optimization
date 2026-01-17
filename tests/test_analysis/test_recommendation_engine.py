"""
Tests for Recommendation Engine Module

Tests actionable recommendation generation.
"""

import pytest

from seo_optimizer.analysis.recommendation_engine import (
    Recommendation,
    RecommendationCategory,
    RecommendationEngine,
    RecommendationEngineConfig,
    RecommendationPriority,
    generate_recommendations,
)
from seo_optimizer.analysis.models import (
    AIScore,
    GEOScore,
    HeadingAnalysis,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordAnalysis,
    KeywordConfig,
    ReadabilityScore,
    SemanticScore,
    SEOScore,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine() -> RecommendationEngine:
    """Create a recommendation engine instance."""
    return RecommendationEngine()


@pytest.fixture
def sample_issues() -> list[Issue]:
    """Sample issues for testing."""
    return [
        Issue(
            category=IssueCategory.KEYWORD,
            severity=IssueSeverity.CRITICAL,
            message="Primary keyword not found in content",
            fix_suggestion="Add the primary keyword to the content",
        ),
        Issue(
            category=IssueCategory.STRUCTURE,
            severity=IssueSeverity.WARNING,
            message="Missing H1 heading",
            fix_suggestion="Add an H1 heading with the primary keyword",
        ),
        Issue(
            category=IssueCategory.READABILITY,
            severity=IssueSeverity.INFO,
            message="Average sentence length high",
            fix_suggestion="Break long sentences into shorter ones",
        ),
    ]


@pytest.fixture
def sample_geo_score() -> GEOScore:
    """Sample GEO score for testing."""
    return GEOScore(
        seo_score=SEOScore(
            keyword_score=50,
            heading_score=60,
            total=55,
            keyword_analysis=KeywordAnalysis(
                primary_keyword="test",
                primary_found=True,
                primary_locations=["body"],
            ),
            heading_analysis=HeadingAnalysis(h1_count=1),
        ),
        semantic_score=SemanticScore(
            topic_coverage=0.6,
            information_gain=0.4,
            total=55,
            missing_entities=["entity1", "entity2"],
        ),
        ai_score=AIScore(
            chunk_clarity=0.6,
            answer_completeness=0.5,
            extraction_friendliness=0.4,
            total=50,
        ),
        readability_score=ReadabilityScore(
            avg_sentence_length=25,
            active_voice_ratio=0.6,
            total=60,
        ),
        total=55,
    )


@pytest.fixture
def simple_keywords() -> KeywordConfig:
    """Simple keyword configuration."""
    return KeywordConfig(
        primary_keyword="test keyword",
        secondary_keywords=["secondary1", "secondary2"],
    )


# =============================================================================
# Recommendation Priority Tests
# =============================================================================


class TestRecommendationPriority:
    """Tests for recommendation priority enum."""

    def test_priority_values(self) -> None:
        """Test priority enum values."""
        assert RecommendationPriority.HIGH.value == "high"
        assert RecommendationPriority.MEDIUM.value == "medium"
        assert RecommendationPriority.LOW.value == "low"


class TestRecommendationCategory:
    """Tests for recommendation category enum."""

    def test_category_values(self) -> None:
        """Test category enum values."""
        assert RecommendationCategory.QUICK_WIN.value == "quick_win"
        assert RecommendationCategory.CONTENT_GAP.value == "content_gap"
        assert RecommendationCategory.OPTIMIZATION.value == "optimization"


# =============================================================================
# Recommendation Creation Tests
# =============================================================================


class TestRecommendation:
    """Tests for Recommendation dataclass."""

    def test_recommendation_creation(self) -> None:
        """Test creating a recommendation."""
        rec = Recommendation(
            title="Add H1 Heading",
            description="Include an H1 heading with the primary keyword",
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.STRUCTURE,
            impact="High - Essential for SEO",
            effort="Low - Quick fix",
        )

        assert rec.title == "Add H1 Heading"
        assert rec.priority == RecommendationPriority.HIGH

    def test_recommendation_str(self) -> None:
        """Test string representation."""
        rec = Recommendation(
            title="Fix Issue",
            description="Description here",
            priority=RecommendationPriority.HIGH,
            category=RecommendationCategory.OPTIMIZATION,
            impact="High",
            effort="Low",
        )

        assert "[HIGH]" in str(rec)
        assert "Fix Issue" in str(rec)


# =============================================================================
# Issue to Recommendation Tests
# =============================================================================


class TestIssueToRecommendation:
    """Tests for converting issues to recommendations."""

    def test_critical_issue_high_priority(
        self, engine: RecommendationEngine, sample_issues: list[Issue]
    ) -> None:
        """Test that critical issues become high priority recommendations."""
        critical_issue = sample_issues[0]  # CRITICAL severity
        geo_score = GEOScore()

        recommendations = engine.generate([critical_issue], geo_score)

        assert len(recommendations) > 0
        assert recommendations[0].priority == RecommendationPriority.HIGH

    def test_warning_issue_medium_priority(
        self, engine: RecommendationEngine, sample_issues: list[Issue]
    ) -> None:
        """Test that warning issues become medium priority."""
        warning_issue = sample_issues[1]  # WARNING severity
        geo_score = GEOScore()

        recommendations = engine.generate([warning_issue], geo_score)

        medium_recs = [r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]
        assert len(medium_recs) > 0

    def test_info_issue_low_priority(
        self, engine: RecommendationEngine, sample_issues: list[Issue]
    ) -> None:
        """Test that info issues become low priority."""
        info_issue = sample_issues[2]  # INFO severity
        geo_score = GEOScore()

        recommendations = engine.generate([info_issue], geo_score)

        low_recs = [r for r in recommendations if r.priority == RecommendationPriority.LOW]
        assert len(low_recs) > 0

    def test_issue_without_fix_suggestion_skipped(
        self, engine: RecommendationEngine
    ) -> None:
        """Test that issues without fix suggestions are skipped."""
        issue = Issue(
            category=IssueCategory.KEYWORD,
            severity=IssueSeverity.WARNING,
            message="Some issue",
            fix_suggestion=None,
        )
        geo_score = GEOScore()

        recommendations = engine.generate([issue], geo_score)

        # Should not create recommendation from this issue
        assert all(r.source_issue != issue for r in recommendations)


# =============================================================================
# Score Gap Recommendations Tests
# =============================================================================


class TestScoreGapRecommendations:
    """Tests for recommendations from score gaps."""

    def test_low_seo_score_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        """Test recommendation for low SEO score."""
        geo_score = GEOScore(
            seo_score=SEOScore(keyword_score=40, heading_score=45, total=42),
        )

        recommendations = engine.generate([], geo_score)

        # Should have recommendation for improving SEO
        seo_recs = [r for r in recommendations if "keyword" in r.title.lower() or "seo" in r.title.lower()]
        assert len(seo_recs) > 0

    def test_low_semantic_score_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        """Test recommendation for low semantic score."""
        geo_score = GEOScore(
            semantic_score=SemanticScore(
                topic_coverage=0.5,
                total=45,
                missing_entities=["entity1", "entity2", "entity3"],
            ),
        )

        recommendations = engine.generate([], geo_score)

        # Should have recommendation for topic coverage
        semantic_recs = [r for r in recommendations if "topic" in r.title.lower() or "coverage" in r.title.lower() or "entity" in r.title.lower()]
        assert len(semantic_recs) > 0

    def test_low_ai_score_recommendation(
        self, engine: RecommendationEngine
    ) -> None:
        """Test recommendation for low AI compatibility score."""
        geo_score = GEOScore(
            ai_score=AIScore(
                chunk_clarity=0.4,
                answer_completeness=0.5,
                extraction_friendliness=0.3,
                total=40,
            ),
        )

        recommendations = engine.generate([], geo_score)

        # Should have recommendations for AI compatibility
        ai_recs = [r for r in recommendations if "chunk" in r.title.lower() or "bluf" in r.title.lower() or "structured" in r.title.lower()]
        assert len(ai_recs) > 0


# =============================================================================
# Quick Wins Tests
# =============================================================================


class TestQuickWins:
    """Tests for quick win recommendations."""

    def test_keyword_in_title_quick_win(
        self, engine: RecommendationEngine, simple_keywords: KeywordConfig
    ) -> None:
        """Test quick win for adding keyword to title."""
        geo_score = GEOScore(
            seo_score=SEOScore(
                keyword_analysis=KeywordAnalysis(
                    primary_keyword="test",
                    primary_found=True,
                    primary_locations=["body"],  # Not in title
                ),
            ),
        )

        recommendations = engine.generate([], geo_score, simple_keywords)

        # Should have quick win for title
        title_recs = [r for r in recommendations if "title" in r.title.lower()]
        assert len(title_recs) > 0


# =============================================================================
# Sorting and Deduplication Tests
# =============================================================================


class TestSortingAndDedup:
    """Tests for recommendation sorting and deduplication."""

    def test_recommendations_sorted_by_priority(
        self, engine: RecommendationEngine, sample_issues: list[Issue]
    ) -> None:
        """Test that recommendations are sorted by priority."""
        geo_score = GEOScore()
        recommendations = engine.generate(sample_issues, geo_score)

        # High priority should come before medium and low
        priorities = [r.priority for r in recommendations]
        if len(priorities) >= 2:
            high_indices = [i for i, p in enumerate(priorities) if p == RecommendationPriority.HIGH]
            low_indices = [i for i, p in enumerate(priorities) if p == RecommendationPriority.LOW]
            if high_indices and low_indices:
                assert max(high_indices) < min(low_indices)

    def test_duplicate_recommendations_removed(
        self, engine: RecommendationEngine
    ) -> None:
        """Test that duplicate recommendations are removed."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, "Missing H1", "Add H1"),
            Issue(IssueCategory.STRUCTURE, IssueSeverity.WARNING, "Missing H1 heading", "Add H1"),
        ]
        geo_score = GEOScore()

        recommendations = engine.generate(issues, geo_score)

        # Similar titles should be deduplicated
        titles = [r.title.lower() for r in recommendations]
        assert len(titles) == len(set(titles))


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRecommendationEngineConfig:
    """Tests for engine configuration."""

    def test_max_recommendations(self) -> None:
        """Test maximum recommendations limit."""
        config = RecommendationEngineConfig(max_recommendations=3)
        engine = RecommendationEngine(config)

        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, f"Issue {i}", f"Fix {i}")
            for i in range(10)
        ]
        geo_score = GEOScore()

        recommendations = engine.generate(issues, geo_score)

        assert len(recommendations) <= 3

    def test_exclude_low_priority(self) -> None:
        """Test excluding low priority recommendations."""
        config = RecommendationEngineConfig(include_low_priority=False)
        engine = RecommendationEngine(config)

        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.INFO, "Low priority", "Fix it"),
        ]
        geo_score = GEOScore()

        recommendations = engine.generate(issues, geo_score)

        low_priority = [r for r in recommendations if r.priority == RecommendationPriority.LOW]
        assert len(low_priority) == 0


# =============================================================================
# Formatting Tests
# =============================================================================


class TestFormatting:
    """Tests for recommendation formatting."""

    def test_format_recommendations(self, engine: RecommendationEngine) -> None:
        """Test formatting recommendations as string."""
        recommendations = [
            Recommendation(
                title="Add H1",
                description="Include an H1 heading",
                priority=RecommendationPriority.HIGH,
                category=RecommendationCategory.STRUCTURE,
                impact="High",
                effort="Low",
            ),
        ]

        formatted = engine.format_recommendations(recommendations)

        assert "Add H1" in formatted
        assert "[HIGH]" in formatted
        assert "High" in formatted

    def test_format_empty_recommendations(self, engine: RecommendationEngine) -> None:
        """Test formatting empty recommendations."""
        formatted = engine.format_recommendations([])

        assert "No recommendations" in formatted

    def test_get_priority_recommendations(self, engine: RecommendationEngine) -> None:
        """Test filtering by priority."""
        recommendations = [
            Recommendation("High 1", "Desc", RecommendationPriority.HIGH, RecommendationCategory.OPTIMIZATION, "High", "Low"),
            Recommendation("Low 1", "Desc", RecommendationPriority.LOW, RecommendationCategory.OPTIMIZATION, "Low", "Low"),
            Recommendation("High 2", "Desc", RecommendationPriority.HIGH, RecommendationCategory.OPTIMIZATION, "High", "Low"),
        ]

        high_recs = engine.get_priority_recommendations(recommendations, RecommendationPriority.HIGH)

        assert len(high_recs) == 2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestGenerateRecommendationsFunction:
    """Tests for convenience function."""

    def test_generate_recommendations_basic(self) -> None:
        """Test basic recommendation generation."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Missing keyword", "Add keyword"),
        ]
        geo_score = GEOScore()

        recommendations = generate_recommendations(issues, geo_score)

        assert len(recommendations) > 0

    def test_generate_with_max_count(self) -> None:
        """Test with custom max count."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.WARNING, f"Issue {i}", f"Fix {i}")
            for i in range(10)
        ]
        geo_score = GEOScore()

        recommendations = generate_recommendations(issues, geo_score, max_count=5)

        assert len(recommendations) <= 5

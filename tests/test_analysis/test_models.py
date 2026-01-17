"""
Tests for Analysis Models

Tests all data structures used in the SEO analysis engine.
"""

import pytest

from seo_optimizer.analysis.models import (
    AIScore,
    AnalysisResult,
    DocumentStats,
    EntityMatch,
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
    VersionComparison,
)


# =============================================================================
# Issue Tests
# =============================================================================


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_severity_values(self) -> None:
        """Test severity enum values."""
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.INFO.value == "info"

    def test_severity_is_string(self) -> None:
        """Test that severity values can be used as strings."""
        # StrEnum comparison - value equals the string
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.WARNING.value == "warning"


class TestIssueCategory:
    """Tests for IssueCategory enum."""

    def test_category_values(self) -> None:
        """Test category enum values."""
        assert IssueCategory.STRUCTURE.value == "structure"
        assert IssueCategory.KEYWORD.value == "keyword"
        assert IssueCategory.ENTITY.value == "entity"
        assert IssueCategory.READABILITY.value == "readability"
        assert IssueCategory.AI_COMPATIBILITY.value == "ai_compatibility"
        assert IssueCategory.REDUNDANCY.value == "redundancy"


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self) -> None:
        """Test creating an issue."""
        issue = Issue(
            category=IssueCategory.KEYWORD,
            severity=IssueSeverity.WARNING,
            message="Missing primary keyword",
        )
        assert issue.category == IssueCategory.KEYWORD
        assert issue.severity == IssueSeverity.WARNING
        assert issue.message == "Missing primary keyword"

    def test_issue_with_all_fields(self) -> None:
        """Test issue with all optional fields."""
        issue = Issue(
            category=IssueCategory.STRUCTURE,
            severity=IssueSeverity.CRITICAL,
            message="Missing H1",
            location="Document header",
            current_value="0 H1 tags",
            target_value="1 H1 tag",
            fix_suggestion="Add an H1 heading",
        )
        assert issue.location == "Document header"
        assert issue.fix_suggestion == "Add an H1 heading"

    def test_issue_str_representation(self) -> None:
        """Test string representation of issue."""
        issue = Issue(
            category=IssueCategory.KEYWORD,
            severity=IssueSeverity.CRITICAL,
            message="Keyword missing",
            location="Title",
        )
        str_repr = str(issue)
        assert "[CRITICAL]" in str_repr
        assert "Keyword missing" in str_repr
        assert "Title" in str_repr


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntityMatch:
    """Tests for EntityMatch dataclass."""

    def test_entity_creation(self) -> None:
        """Test creating an entity match."""
        entity = EntityMatch(
            text="Google",
            entity_type="ORG",
            start_char=10,
            end_char=16,
        )
        assert entity.text == "Google"
        assert entity.entity_type == "ORG"
        assert entity.confidence == 1.0  # Default

    def test_entity_length_property(self) -> None:
        """Test entity length calculation."""
        entity = EntityMatch(
            text="Apple Inc.",
            entity_type="ORG",
            start_char=0,
            end_char=10,
        )
        assert entity.length == 10

    def test_entity_with_confidence(self) -> None:
        """Test entity with custom confidence."""
        entity = EntityMatch(
            text="product",
            entity_type="CONCEPT",
            start_char=50,
            end_char=57,
            confidence=0.8,
        )
        assert entity.confidence == 0.8


# =============================================================================
# Keyword Tests
# =============================================================================


class TestKeywordConfig:
    """Tests for KeywordConfig dataclass."""

    def test_keyword_config_creation(self) -> None:
        """Test creating keyword configuration."""
        config = KeywordConfig(
            primary_keyword="cloud computing",
            secondary_keywords=["AWS", "Azure"],
            semantic_entities=["data center", "virtualization"],
        )
        assert config.primary_keyword == "cloud computing"
        assert len(config.secondary_keywords) == 2
        assert len(config.semantic_entities) == 2

    def test_keyword_config_requires_primary(self) -> None:
        """Test that primary keyword is required."""
        with pytest.raises(ValueError, match="Primary keyword is required"):
            KeywordConfig(primary_keyword="")

    def test_keyword_config_defaults(self) -> None:
        """Test default empty lists."""
        config = KeywordConfig(primary_keyword="SEO")
        assert config.secondary_keywords == []
        assert config.semantic_entities == []


class TestKeywordAnalysis:
    """Tests for KeywordAnalysis dataclass."""

    def test_keyword_analysis_defaults(self) -> None:
        """Test default values."""
        analysis = KeywordAnalysis()
        assert analysis.primary_found is False
        assert analysis.keyword_density == 0.0
        assert analysis.passes_135_rule is False

    def test_primary_placement_score(self) -> None:
        """Test primary placement score calculation."""
        analysis = KeywordAnalysis(
            primary_locations=["title", "h1", "first_100_words"]
        )
        # Title has weight 1.0 (highest)
        assert analysis.primary_placement_score == 1.0

    def test_primary_placement_score_empty(self) -> None:
        """Test placement score with no locations."""
        analysis = KeywordAnalysis()
        assert analysis.primary_placement_score == 0.0


# =============================================================================
# Heading Tests
# =============================================================================


class TestHeadingAnalysis:
    """Tests for HeadingAnalysis dataclass."""

    def test_heading_analysis_defaults(self) -> None:
        """Test default values."""
        analysis = HeadingAnalysis()
        assert analysis.h1_count == 0
        assert analysis.hierarchy_valid is True

    def test_has_valid_h1_single(self) -> None:
        """Test valid H1 with exactly one."""
        analysis = HeadingAnalysis(h1_count=1)
        assert analysis.has_valid_h1 is True

    def test_has_valid_h1_multiple(self) -> None:
        """Test invalid H1 with multiple."""
        analysis = HeadingAnalysis(h1_count=3)
        assert analysis.has_valid_h1 is False

    def test_has_valid_h1_none(self) -> None:
        """Test invalid H1 with none."""
        analysis = HeadingAnalysis(h1_count=0)
        assert analysis.has_valid_h1 is False


# =============================================================================
# Score Tests
# =============================================================================


class TestSEOScore:
    """Tests for SEOScore dataclass."""

    def test_seo_score_defaults(self) -> None:
        """Test default values."""
        score = SEOScore()
        assert score.keyword_score == 0.0
        assert score.heading_score == 0.0
        assert score.total == 0.0

    def test_seo_score_auto_total(self) -> None:
        """Test automatic total calculation."""
        score = SEOScore(
            keyword_score=80,
            heading_score=70,
            link_readiness_score=60,
        )
        # 80*0.4 + 70*0.4 + 60*0.2 = 32 + 28 + 12 = 72
        assert score.total == 72.0


class TestSemanticScore:
    """Tests for SemanticScore dataclass."""

    def test_semantic_score_defaults(self) -> None:
        """Test default values."""
        score = SemanticScore()
        assert score.topic_coverage == 0.0
        assert score.entity_saturation is False

    def test_semantic_score_auto_calculation(self) -> None:
        """Test automatic total calculation."""
        score = SemanticScore(
            topic_coverage=0.85,
            information_gain=0.6,
            entity_density=0.02,
        )
        assert score.total > 0


class TestAIScore:
    """Tests for AIScore dataclass."""

    def test_ai_score_defaults(self) -> None:
        """Test default values."""
        score = AIScore()
        assert score.chunk_clarity == 0.0
        assert score.redundancy_penalty == 0.0

    def test_ai_score_auto_calculation(self) -> None:
        """Test automatic total calculation."""
        score = AIScore(
            chunk_clarity=0.8,
            answer_completeness=0.7,
            extraction_friendliness=0.6,
        )
        assert score.total > 0


class TestReadabilityScore:
    """Tests for ReadabilityScore dataclass."""

    def test_readability_score_defaults(self) -> None:
        """Test default values."""
        score = ReadabilityScore()
        assert score.avg_sentence_length == 0.0
        assert score.flesch_kincaid_grade == 0.0

    def test_readability_score_auto_calculation(self) -> None:
        """Test automatic total calculation."""
        score = ReadabilityScore(
            avg_sentence_length=18,
            active_voice_ratio=0.85,
            flesch_kincaid_grade=10,
        )
        assert score.total > 0


class TestGEOScore:
    """Tests for GEOScore dataclass."""

    def test_geo_score_defaults(self) -> None:
        """Test default values."""
        score = GEOScore()
        assert score.total == 0.0

    def test_geo_score_auto_calculation(self) -> None:
        """Test automatic total calculation from components."""
        seo = SEOScore(total=80)
        semantic = SemanticScore(total=75)
        ai = AIScore(total=70)
        readability = ReadabilityScore(total=85)

        geo = GEOScore(
            seo_score=seo,
            semantic_score=semantic,
            ai_score=ai,
            readability_score=readability,
        )
        # 0.20*80 + 0.30*75 + 0.30*70 + 0.20*85 = 16 + 22.5 + 21 + 17 = 76.5
        assert geo.total == 76.5

    def test_geo_confidence_rating_excellent(self) -> None:
        """Test excellent confidence rating."""
        geo = GEOScore(total=95)
        assert geo.confidence_rating == "Excellent"

    def test_geo_confidence_rating_good(self) -> None:
        """Test good confidence rating."""
        geo = GEOScore(total=75)
        assert geo.confidence_rating == "Good"

    def test_geo_confidence_rating_fair(self) -> None:
        """Test fair confidence rating."""
        geo = GEOScore(total=50)
        assert geo.confidence_rating == "Fair"

    def test_geo_confidence_rating_poor(self) -> None:
        """Test poor confidence rating."""
        geo = GEOScore(total=30)
        assert geo.confidence_rating == "Poor"

    def test_geo_critical_issues(self) -> None:
        """Test critical issues property."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Missing keyword"),
            Issue(IssueCategory.STRUCTURE, IssueSeverity.WARNING, "Low heading density"),
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "No H1"),
        ]
        geo = GEOScore(all_issues=issues)
        assert len(geo.critical_issues) == 2

    def test_geo_warning_issues(self) -> None:
        """Test warning issues property."""
        issues = [
            Issue(IssueCategory.KEYWORD, IssueSeverity.CRITICAL, "Missing keyword"),
            Issue(IssueCategory.STRUCTURE, IssueSeverity.WARNING, "Low heading density"),
            Issue(IssueCategory.READABILITY, IssueSeverity.WARNING, "Long sentences"),
        ]
        geo = GEOScore(all_issues=issues)
        assert len(geo.warning_issues) == 2


# =============================================================================
# Document Stats Tests
# =============================================================================


class TestDocumentStats:
    """Tests for DocumentStats dataclass."""

    def test_document_stats_defaults(self) -> None:
        """Test default values."""
        stats = DocumentStats()
        assert stats.word_count == 0
        assert stats.paragraph_count == 0

    def test_avg_paragraph_length(self) -> None:
        """Test average paragraph length calculation."""
        stats = DocumentStats(word_count=1000, paragraph_count=10)
        assert stats.avg_paragraph_length == 100.0

    def test_avg_paragraph_length_zero_paragraphs(self) -> None:
        """Test average with zero paragraphs."""
        stats = DocumentStats(word_count=100, paragraph_count=0)
        assert stats.avg_paragraph_length == 0.0

    def test_avg_sentence_length(self) -> None:
        """Test average sentence length calculation."""
        stats = DocumentStats(word_count=500, sentence_count=25)
        assert stats.avg_sentence_length == 20.0


# =============================================================================
# Analysis Result Tests
# =============================================================================


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_analysis_result_defaults(self) -> None:
        """Test default values."""
        result = AnalysisResult()
        assert result.recommendations == []

    def test_analysis_result_summary(self) -> None:
        """Test summary generation."""
        geo = GEOScore(
            seo_score=SEOScore(total=80),
            semantic_score=SemanticScore(total=75),
            ai_score=AIScore(total=70),
            readability_score=ReadabilityScore(total=85),
            total=76.5,
        )
        result = AnalysisResult(geo_score=geo)
        summary = result.summary
        assert "76.5" in summary
        assert "SEO" in summary
        assert "Semantic" in summary


# =============================================================================
# Version Comparison Tests
# =============================================================================


class TestVersionComparison:
    """Tests for VersionComparison dataclass."""

    def test_version_comparison_creation(self) -> None:
        """Test creating version comparison."""
        comparison = VersionComparison(
            original_score=60.0,
            optimized_score=80.0,
            improvement=20.0,
        )
        assert comparison.original_score == 60.0
        assert comparison.improvement == 20.0

    def test_version_comparison_auto_improvement(self) -> None:
        """Test automatic improvement calculation."""
        comparison = VersionComparison(
            original_score=50.0,
            optimized_score=75.0,
            improvement=0.0,  # Should be auto-calculated
        )
        # Auto-calculation only happens when improvement is exactly 0.0
        assert comparison.improvement == 25.0

    def test_version_comparison_with_changes(self) -> None:
        """Test comparison with key changes."""
        comparison = VersionComparison(
            original_score=55.0,
            optimized_score=78.0,
            improvement=23.0,
            key_changes=["Added FAQ section", "Improved heading structure"],
            issues_fixed=["Missing H1 heading"],
            new_issues=[],
        )
        assert len(comparison.key_changes) == 2
        assert len(comparison.issues_fixed) == 1

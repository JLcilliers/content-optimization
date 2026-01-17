"""
Tests for optimization models.

Tests:
- Configuration defaults and validation
- Change type enums
- Result dataclasses
- Serialization/deserialization
"""

import pytest

from seo_optimizer.optimization.models import (
    ChangeType,
    ContentType,
    FAQEntry,
    GuardrailViolation,
    MetaTags,
    OptimizationChange,
    OptimizationConfig,
    OptimizationMode,
    OptimizationResult,
    PipelineResult,
)


class TestOptimizationMode:
    """Tests for OptimizationMode enum."""

    def test_conservative_mode_value(self):
        """Test conservative mode has correct value."""
        assert OptimizationMode.CONSERVATIVE.value == "conservative"

    def test_balanced_mode_value(self):
        """Test balanced mode has correct value."""
        assert OptimizationMode.BALANCED.value == "balanced"

    def test_aggressive_mode_value(self):
        """Test aggressive mode has correct value."""
        assert OptimizationMode.AGGRESSIVE.value == "aggressive"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        mode = OptimizationMode("balanced")
        assert mode == OptimizationMode.BALANCED


class TestContentType:
    """Tests for ContentType enum."""

    def test_all_content_types_defined(self):
        """Test all expected content types are defined."""
        expected = ["article", "product", "service", "landing_page", "blog_post"]
        for ct in expected:
            assert ContentType(ct) is not None


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_keyword_change_type(self):
        """Test keyword change type."""
        assert ChangeType.KEYWORD.value == "keyword"

    def test_entity_change_type(self):
        """Test entity change type."""
        assert ChangeType.ENTITY.value == "entity"

    def test_structure_change_type(self):
        """Test structure change type."""
        assert ChangeType.STRUCTURE.value == "structure"

    def test_readability_change_type(self):
        """Test readability change type."""
        assert ChangeType.READABILITY.value == "readability"

    def test_meta_change_type(self):
        """Test meta change type."""
        assert ChangeType.META.value == "meta"


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_mode_is_balanced(self):
        """Test default mode is balanced."""
        config = OptimizationConfig()
        assert config.mode == OptimizationMode.BALANCED

    def test_default_keyword_density_thresholds(self):
        """Test default density thresholds."""
        config = OptimizationConfig()
        assert config.min_keyword_density == 1.0
        assert config.max_keyword_density == 2.5

    def test_custom_primary_keyword(self):
        """Test setting primary keyword."""
        config = OptimizationConfig(primary_keyword="SEO optimization")
        assert config.primary_keyword == "SEO optimization"

    def test_secondary_keywords_list(self):
        """Test secondary keywords list."""
        config = OptimizationConfig(
            secondary_keywords=["content", "marketing", "strategy"]
        )
        assert len(config.secondary_keywords) == 3

    def test_semantic_entities_list(self):
        """Test semantic entities list."""
        config = OptimizationConfig(semantic_entities=["BERT", "E-E-A-T"])
        assert "BERT" in config.semantic_entities

    def test_default_max_sentence_length(self):
        """Test default max sentence length."""
        config = OptimizationConfig()
        assert config.max_sentence_length == 25

    def test_default_max_faq_items(self):
        """Test default max FAQ items."""
        config = OptimizationConfig()
        assert config.max_faq_items == 5

    def test_feature_flags_defaults(self):
        """Test feature flag defaults."""
        config = OptimizationConfig()
        assert config.inject_keywords is True
        assert config.inject_entities is True
        assert config.improve_readability is True
        assert config.generate_faq is True

    def test_custom_mode(self):
        """Test custom mode setting."""
        config = OptimizationConfig(mode=OptimizationMode.AGGRESSIVE)
        assert config.mode == OptimizationMode.AGGRESSIVE


class TestOptimizationChange:
    """Tests for OptimizationChange dataclass."""

    def test_create_keyword_change(self):
        """Test creating a keyword change."""
        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="First paragraph",
            original="This is the text",
            optimized="This is the SEO text",
            reason="Added primary keyword",
            impact_score=3.5,
        )
        assert change.change_type == ChangeType.KEYWORD
        assert change.impact_score == 3.5

    def test_change_with_section_id(self):
        """Test change with section ID."""
        change = OptimizationChange(
            change_type=ChangeType.STRUCTURE,
            location="H2 heading",
            original="Title",
            optimized="SEO Title",
            reason="Added keyword",
            impact_score=2.0,
            section_id="section_1",
        )
        assert change.section_id == "section_1"

    def test_change_impact_score_range(self):
        """Test impact score is stored correctly."""
        change = OptimizationChange(
            change_type=ChangeType.READABILITY,
            location="Para 2",
            original="Old",
            optimized="New",
            reason="Improved",
            impact_score=5.0,
        )
        assert 0 <= change.impact_score <= 5.0


class TestFAQEntry:
    """Tests for FAQEntry dataclass."""

    def test_create_faq_entry(self):
        """Test creating FAQ entry."""
        faq = FAQEntry(
            question="What is SEO?",
            answer="SEO stands for Search Engine Optimization.",
            html_id="faq-what-is-seo",
        )
        assert faq.question.endswith("?")
        assert faq.html_id.startswith("faq-")

    def test_faq_with_source_section(self):
        """Test FAQ with source section."""
        faq = FAQEntry(
            question="How does it work?",
            answer="It works by analyzing content.",
            html_id="faq-how",
            source_section="section_2",
        )
        assert faq.source_section == "section_2"


class TestMetaTags:
    """Tests for MetaTags dataclass."""

    def test_create_meta_tags(self):
        """Test creating meta tags."""
        meta = MetaTags(
            title="SEO Guide 2024",
            description="Learn SEO best practices in this guide.",
        )
        assert meta.title is not None
        assert meta.description is not None

    def test_meta_with_pixel_widths(self):
        """Test meta tags with pixel widths."""
        meta = MetaTags(
            title="Short Title",
            description="Description text here.",
            title_pixel_width=200.0,
            description_pixel_width=400.0,
        )
        assert meta.title_pixel_width == 200.0
        assert meta.description_pixel_width == 400.0


class TestGuardrailViolation:
    """Tests for GuardrailViolation dataclass."""

    def test_create_violation(self):
        """Test creating a violation."""
        violation = GuardrailViolation(
            rule="keyword_density",
            message="Keyword density exceeds 5%",
            severity="warning",
        )
        assert violation.severity == "warning"

    def test_violation_severities(self):
        """Test different severity levels."""
        for severity in ["info", "warning", "error"]:
            violation = GuardrailViolation(
                rule="test", message="test", severity=severity
            )
            assert violation.severity == severity


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_empty_result(self):
        """Test empty optimization result."""
        result = OptimizationResult(changes=[])
        assert len(result.changes) == 0
        assert result.geo_score == 0.0

    def test_result_with_changes(self):
        """Test result with changes."""
        changes = [
            OptimizationChange(
                change_type=ChangeType.KEYWORD,
                location="P1",
                original="Old",
                optimized="New",
                reason="Improved",
                impact_score=2.0,
            )
        ]
        result = OptimizationResult(changes=changes, optimized_geo_score=75.0)
        assert len(result.changes) == 1
        assert result.geo_score == 75.0  # Property alias works

    def test_result_with_faq(self):
        """Test result with FAQ entries."""
        faq = FAQEntry(
            question="Q?", answer="A", html_id="faq-q"
        )
        result = OptimizationResult(changes=[], faq_entries=[faq])
        assert len(result.faq_entries) == 1


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_result(self):
        """Test successful pipeline result."""
        result = PipelineResult(success=True)
        assert result.success is True

    def test_failed_result_with_errors(self):
        """Test failed result with errors."""
        result = PipelineResult(
            success=False,
            errors=["File not found", "Parse error"],
        )
        assert result.success is False
        assert len(result.errors) == 2

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = PipelineResult(
            success=True,
            warnings=["Low keyword density"],
        )
        assert len(result.warnings) == 1

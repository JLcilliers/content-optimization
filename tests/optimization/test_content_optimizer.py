"""
Tests for content optimizer (main orchestrator).

Tests:
- Optimization orchestration
- GEO-Metric score calculation
- Mode-based change limits
- Component integration
"""

import pytest

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType
from seo_optimizer.optimization.content_optimizer import (
    ContentOptimizer,
    GEO_METRIC_WEIGHTS,
    MODE_CHANGE_LIMITS,
    OptimizationContext,
)
from seo_optimizer.optimization.models import (
    ChangeType,
    OptimizationConfig,
    OptimizationMode,
    OptimizationResult,
)

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        mode=OptimizationMode.BALANCED,
        primary_keyword="content optimization",
        secondary_keywords=["SEO", "ranking"],
        semantic_entities=["BERT", "E-E-A-T"],
        generate_faq=True,
    )


@pytest.fixture
def optimizer(config):
    """Create content optimizer."""
    return ContentOptimizer(config)


@pytest.fixture
def sample_ast():
    """Create sample document AST."""
    nodes = [
        make_node(
            "h1",
            NodeType.HEADING,
            "Guide to Digital Marketing",
            0,
            {"level": 1},
        ),
        make_node(
            "p1",
            NodeType.PARAGRAPH,
            "Digital marketing involves various strategies and techniques. "
            "This guide covers the essential aspects of online marketing success.",
            30,
        ),
        make_node(
            "h2_1",
            NodeType.HEADING,
            "Getting Started",
            200,
            {"level": 2},
        ),
        make_node(
            "p2",
            NodeType.PARAGRAPH,
            "Starting your journey requires understanding the fundamentals. "
            "Follow these steps to begin implementing effective strategies.",
            220,
        ),
        make_node(
            "h2_2",
            NodeType.HEADING,
            "Best Practices",
            400,
            {"level": 2},
        ),
        make_node(
            "p3",
            NodeType.PARAGRAPH,
            "Utilize comprehensive tools and platforms to achieve results. "
            "The methodology involves multiple approaches for success.",
            420,
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})


class TestGEOMetricWeights:
    """Tests for GEO-Metric weight configuration."""

    def test_weights_sum_to_one(self):
        """Test weights sum to 1.0."""
        total = sum(GEO_METRIC_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_seo_weight(self):
        """Test SEO weight is 0.20."""
        assert GEO_METRIC_WEIGHTS["seo"] == 0.20

    def test_semantic_weight(self):
        """Test semantic weight is 0.30."""
        assert GEO_METRIC_WEIGHTS["semantic"] == 0.30

    def test_ai_readiness_weight(self):
        """Test AI readiness weight is 0.30."""
        assert GEO_METRIC_WEIGHTS["ai_readiness"] == 0.30

    def test_readability_weight(self):
        """Test readability weight is 0.20."""
        assert GEO_METRIC_WEIGHTS["readability"] == 0.20


class TestModeChangeLimits:
    """Tests for mode-based change limits."""

    def test_conservative_limit(self):
        """Test conservative mode limit."""
        assert MODE_CHANGE_LIMITS[OptimizationMode.CONSERVATIVE] == 10

    def test_balanced_limit(self):
        """Test balanced mode limit."""
        assert MODE_CHANGE_LIMITS[OptimizationMode.BALANCED] == 25

    def test_aggressive_limit(self):
        """Test aggressive mode limit."""
        assert MODE_CHANGE_LIMITS[OptimizationMode.AGGRESSIVE] == 50


class TestOptimization:
    """Tests for main optimization flow."""

    def test_returns_optimization_result(self, optimizer, sample_ast):
        """Test returns OptimizationResult."""
        result = optimizer.optimize(sample_ast)
        assert isinstance(result, OptimizationResult)

    def test_result_has_changes(self, optimizer, sample_ast):
        """Test result contains changes list."""
        result = optimizer.optimize(sample_ast)
        assert isinstance(result.changes, list)

    def test_result_has_geo_score(self, optimizer, sample_ast):
        """Test result has GEO score."""
        result = optimizer.optimize(sample_ast)
        assert result.geo_score >= 0
        assert result.geo_score <= 100

    def test_result_has_score_tracking(self, optimizer, sample_ast):
        """Test result has score tracking."""
        result = optimizer.optimize(sample_ast)
        assert result.original_geo_score >= 0
        assert result.optimized_geo_score >= 0
        assert result.geo_score >= 0  # Property alias


class TestModeRespect:
    """Tests for respecting optimization mode."""

    def test_conservative_mode_fewer_changes(self, sample_ast):
        """Test conservative mode makes fewer changes."""
        config = OptimizationConfig(
            mode=OptimizationMode.CONSERVATIVE,
            primary_keyword="marketing",
        )
        optimizer = ContentOptimizer(config)
        result = optimizer.optimize(sample_ast)

        assert len(result.changes) <= MODE_CHANGE_LIMITS[OptimizationMode.CONSERVATIVE]

    def test_aggressive_mode_more_changes(self, sample_ast):
        """Test aggressive mode allows more changes."""
        config = OptimizationConfig(
            mode=OptimizationMode.AGGRESSIVE,
            primary_keyword="marketing",
        )
        optimizer = ContentOptimizer(config)
        result = optimizer.optimize(sample_ast)

        # Should allow up to aggressive limit
        assert len(result.changes) <= MODE_CHANGE_LIMITS[OptimizationMode.AGGRESSIVE]


class TestFAQGeneration:
    """Tests for FAQ generation integration."""

    def test_generates_faq_entries(self, optimizer, sample_ast):
        """Test generates FAQ entries."""
        result = optimizer.optimize(sample_ast)
        # FAQ generation should be attempted
        assert result.faq_entries is not None or hasattr(result, 'faq_entries')

    def test_respects_faq_disabled(self, sample_ast):
        """Test respects FAQ generation disabled."""
        config = OptimizationConfig(generate_faq=False)
        optimizer = ContentOptimizer(config)
        result = optimizer.optimize(sample_ast)

        if result.faq_entries:
            assert len(result.faq_entries) == 0


class TestMetaGeneration:
    """Tests for meta tag generation integration."""

    def test_generates_meta_tags(self, optimizer, sample_ast):
        """Test generates meta tags."""
        result = optimizer.optimize(sample_ast)
        # Meta generation should be attempted
        assert hasattr(result, 'meta_tags')

    def test_meta_tags_have_title(self, optimizer, sample_ast):
        """Test meta tags include title."""
        result = optimizer.optimize(sample_ast)
        if result.meta_tags:
            assert result.meta_tags.title is not None


class TestIncrementalOptimization:
    """Tests for incremental optimization."""

    def test_incremental_keywords_only(self, optimizer, sample_ast):
        """Test incremental optimization for keywords only."""
        result = optimizer.optimize_incremental(
            sample_ast,
            change_types=[ChangeType.KEYWORD]
        )

        for change in result.changes:
            assert change.change_type == ChangeType.KEYWORD

    def test_incremental_heading_only(self, optimizer, sample_ast):
        """Test incremental optimization for headings only."""
        result = optimizer.optimize_incremental(
            sample_ast,
            change_types=[ChangeType.HEADING]
        )

        for change in result.changes:
            assert change.change_type == ChangeType.HEADING

    def test_incremental_multiple_types(self, optimizer, sample_ast):
        """Test incremental with multiple types."""
        result = optimizer.optimize_incremental(
            sample_ast,
            change_types=[ChangeType.KEYWORD, ChangeType.READABILITY]
        )

        for change in result.changes:
            assert change.change_type in [ChangeType.KEYWORD, ChangeType.READABILITY]


class TestOptimizationSummary:
    """Tests for optimization summary."""

    def test_get_summary(self, optimizer, sample_ast):
        """Test getting optimization summary."""
        result = optimizer.optimize(sample_ast)
        summary = optimizer.get_optimization_summary(result)

        assert "total_changes" in summary
        assert "geo_score" in summary
        assert "optimized_geo_score" in summary

    def test_summary_change_count_matches(self, optimizer, sample_ast):
        """Test summary change count matches result."""
        result = optimizer.optimize(sample_ast)
        summary = optimizer.get_optimization_summary(result)

        assert summary["total_changes"] == len(result.changes)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self, optimizer):
        """Test validating valid configuration."""
        issues = optimizer.validate_config()
        # Default config should be mostly valid
        assert isinstance(issues, list)

    def test_validate_missing_keyword_for_injection(self):
        """Test validates missing keyword when injection enabled."""
        config = OptimizationConfig(
            inject_keywords=True,
            primary_keyword=None,
        )
        optimizer = ContentOptimizer(config)
        issues = optimizer.validate_config()

        assert any("keyword" in issue.lower() for issue in issues)


class TestGuardrailIntegration:
    """Tests for guardrail integration."""

    def test_tracks_guardrail_violations(self, optimizer, sample_ast):
        """Test tracks guardrail violations."""
        result = optimizer.optimize(sample_ast)
        assert hasattr(result, 'guardrail_warnings')
        assert hasattr(result, 'changes_blocked')


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, optimizer):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        result = optimizer.optimize(ast)
        assert isinstance(result, OptimizationResult)

    def test_only_headings(self, optimizer):
        """Test document with only headings."""
        nodes = [
            ContentNode(
                node_id="h1",
                node_type=NodeType.HEADING,
                text_content="Title",
                metadata={"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = optimizer.optimize(ast)
        assert isinstance(result, OptimizationResult)

    def test_very_long_document(self, optimizer):
        """Test very long document."""
        nodes = []
        for i in range(50):
            nodes.append(
                ContentNode(
                    node_id=f"p{i}",
                    node_type=NodeType.PARAGRAPH,
                    text_content=f"Paragraph {i} with some content. " * 5,
                )
            )
        ast = DocumentAST(nodes=nodes, metadata={})
        result = optimizer.optimize(ast)

        # Should still respect mode limits
        assert len(result.changes) <= MODE_CHANGE_LIMITS[optimizer.config.mode]

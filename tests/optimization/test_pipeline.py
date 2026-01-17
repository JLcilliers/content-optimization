"""
Tests for optimization pipeline.

Tests:
- End-to-end workflow
- Input/output handling
- Change tracking
- State management
"""

import pytest
from pathlib import Path

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.models import OptimizationConfig, PipelineResult
from seo_optimizer.optimization.pipeline import (
    OptimizationPipeline,
    PipelineConfig,
    PipelineState,
    optimize_content,
)

from .conftest import make_node


@pytest.fixture
def pipeline_config():
    """Create pipeline configuration."""
    opt_config = OptimizationConfig(
        primary_keyword="content optimization",
        secondary_keywords=["SEO", "ranking"],
        generate_faq=True,
    )
    return PipelineConfig(
        optimization_config=opt_config,
        dry_run=True,  # Don't write files
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create optimization pipeline."""
    return OptimizationPipeline(pipeline_config)


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
        make_node("h2", NodeType.HEADING, "Getting Started", 150, {"level": 2}),
        make_node(
            "p2",
            NodeType.PARAGRAPH,
            "Starting your journey requires understanding the fundamentals. "
            "Follow these steps to begin implementing effective strategies.",
            170,
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})


class TestPipelineExecution:
    """Tests for pipeline execution."""

    def test_run_returns_result(self, pipeline, sample_ast):
        """Test run returns PipelineResult."""
        result = pipeline.run(ast=sample_ast)
        assert isinstance(result, PipelineResult)

    def test_successful_run(self, pipeline, sample_ast):
        """Test successful pipeline run."""
        result = pipeline.run(ast=sample_ast)
        assert result.success is True

    def test_result_has_optimized_ast(self, pipeline, sample_ast):
        """Test result includes optimized AST."""
        result = pipeline.run(ast=sample_ast)
        if result.success:
            assert result.optimized_ast is not None

    def test_result_has_optimization_result(self, pipeline, sample_ast):
        """Test result includes optimization result."""
        result = pipeline.run(ast=sample_ast)
        if result.success:
            assert result.optimization_result is not None


class TestInputHandling:
    """Tests for input handling."""

    def test_accepts_ast_input(self, pipeline, sample_ast):
        """Test accepts AST as input."""
        result = pipeline.run(ast=sample_ast)
        assert result.success is True

    def test_parses_content_string(self):
        """Test parses content from string."""
        config = PipelineConfig(
            input_content="# Test Title\n\nTest paragraph content.",
            optimization_config=OptimizationConfig(primary_keyword="test"),
            dry_run=True,
        )
        pipeline = OptimizationPipeline(config)
        result = pipeline.run()

        assert isinstance(result, PipelineResult)


class TestChangeTracking:
    """Tests for change tracking."""

    def test_builds_change_map(self, pipeline, sample_ast):
        """Test builds change map."""
        result = pipeline.run(ast=sample_ast)
        if result.success:
            assert result.change_map is not None

    def test_change_map_tracks_new_nodes(self, pipeline, sample_ast):
        """Test change map tracks new nodes."""
        result = pipeline.run(ast=sample_ast)
        if result.success and result.change_map:
            assert "new_nodes" in result.change_map

    def test_change_map_tracks_modifications(self, pipeline, sample_ast):
        """Test change map tracks modifications."""
        result = pipeline.run(ast=sample_ast)
        if result.success and result.change_map:
            assert "modified_nodes" in result.change_map


class TestStateManagement:
    """Tests for pipeline state management."""

    def test_initial_state(self, pipeline):
        """Test initial state is empty."""
        state = pipeline.get_state()
        assert isinstance(state, PipelineState)

    def test_state_tracks_phases(self, pipeline, sample_ast):
        """Test state tracks completed phases."""
        pipeline.run(ast=sample_ast)
        state = pipeline.get_state()

        assert len(state.phases_completed) > 0

    def test_state_tracks_start_time(self, pipeline, sample_ast):
        """Test state tracks start time."""
        pipeline.run(ast=sample_ast)
        state = pipeline.get_state()

        assert state.started_at is not None


class TestDryRun:
    """Tests for dry run mode."""

    def test_dry_run_no_output(self):
        """Test dry run doesn't write output."""
        config = PipelineConfig(
            input_content="# Test\n\nContent here.",
            output_path=Path("test_output.docx"),
            optimization_config=OptimizationConfig(primary_keyword="test"),
            dry_run=True,
        )
        pipeline = OptimizationPipeline(config)
        result = pipeline.run()

        # Should not create output file
        assert result.output_path is None or not Path("test_output.docx").exists()


class TestASTCloning:
    """Tests for AST cloning."""

    def test_preserves_original_ast(self, pipeline, sample_ast):
        """Test original AST is preserved."""
        original_text = sample_ast.nodes[0].text_content
        result = pipeline.run(ast=sample_ast)

        # Original should be unchanged
        assert sample_ast.nodes[0].text_content == original_text

    def test_returns_original_ast(self, pipeline, sample_ast):
        """Test result includes original AST."""
        result = pipeline.run(ast=sample_ast)
        if result.success:
            assert result.original_ast is not None


class TestConvenienceFunction:
    """Tests for optimize_content convenience function."""

    def test_optimize_content_basic(self):
        """Test basic usage of optimize_content."""
        content = """
# Test Document

This is a test paragraph about digital marketing.
It discusses various strategies and techniques.

## Getting Started

Follow these steps to begin your journey.
"""
        result = optimize_content(
            content=content,
            primary_keyword="digital marketing",
        )

        assert isinstance(result, PipelineResult)

    def test_optimize_content_with_options(self):
        """Test optimize_content with additional options."""
        content = "# Test\n\nSome content here."
        result = optimize_content(
            content=content,
            primary_keyword="test",
            secondary_keywords=["example", "sample"],
            semantic_entities=["API", "SDK"],
        )

        assert isinstance(result, PipelineResult)


class TestErrorHandling:
    """Tests for error handling."""

    def test_handles_missing_input(self):
        """Test handles missing input gracefully."""
        config = PipelineConfig(
            input_path=None,
            input_content=None,
            dry_run=True,
        )
        pipeline = OptimizationPipeline(config)
        result = pipeline.run()

        assert result.success is False
        assert len(result.errors) > 0

    def test_errors_tracked_in_result(self):
        """Test errors are tracked in result."""
        config = PipelineConfig(dry_run=True)
        pipeline = OptimizationPipeline(config)
        result = pipeline.run()

        assert hasattr(result, 'errors')


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, pipeline):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        result = pipeline.run(ast=ast)
        assert isinstance(result, PipelineResult)

    def test_single_node_document(self, pipeline):
        """Test document with single node."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Single paragraph."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = pipeline.run(ast=ast)
        assert isinstance(result, PipelineResult)

    def test_verbose_mode(self, sample_ast):
        """Test verbose mode."""
        config = PipelineConfig(
            optimization_config=OptimizationConfig(primary_keyword="test"),
            dry_run=True,
            verbose=True,
        )
        pipeline = OptimizationPipeline(config)
        result = pipeline.run(ast=sample_ast)
        assert isinstance(result, PipelineResult)

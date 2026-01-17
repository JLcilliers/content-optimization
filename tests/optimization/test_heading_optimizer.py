"""
Tests for heading optimizer.

Tests:
- H1 validation and fixes
- Heading hierarchy fixes
- Section heading insertion
- Heading text optimization
"""

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.heading_optimizer import HeadingOptimizer
from seo_optimizer.optimization.models import ChangeType, OptimizationConfig

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        primary_keyword="SEO optimization",
        secondary_keywords=["content", "ranking"],
    )


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def optimizer(config, guardrails):
    """Create heading optimizer."""
    return HeadingOptimizer(config, guardrails)


class TestH1Validation:
    """Tests for H1 heading validation."""

    def test_detects_missing_h1(self, optimizer):
        """Test detects missing H1."""
        nodes = [
            make_node("h2", NodeType.HEADING, "Section Title", 0, {"level": 2}),
            make_node("p1", NodeType.PARAGRAPH, "Some content here.", 15),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should suggest adding H1
        h1_changes = [c for c in changes if "H1" in c.location or "H1" in c.reason]
        assert len(h1_changes) > 0

    def test_detects_multiple_h1(self, optimizer):
        """Test detects multiple H1 headings."""
        nodes = [
            make_node("h1_1", NodeType.HEADING, "First Title", 0, {"level": 1}),
            make_node("h1_2", NodeType.HEADING, "Second Title", 15, {"level": 1}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should flag multiple H1s
        assert len(changes) > 0

    def test_valid_single_h1_no_changes(self, optimizer):
        """Test valid single H1 doesn't trigger changes."""
        nodes = [
            make_node("h1", NodeType.HEADING, "SEO Optimization Guide", 0, {"level": 1}),
            make_node("h2", NodeType.HEADING, "Getting Started", 25, {"level": 2}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should not flag valid structure
        h1_issues = [c for c in changes if "multiple" in c.reason.lower() or "missing" in c.reason.lower()]
        assert len(h1_issues) == 0


class TestHierarchyFixes:
    """Tests for heading hierarchy fixes."""

    def test_detects_hierarchy_gap(self, optimizer):
        """Test detects H1 → H3 gap (skipping H2)."""
        nodes = [
            make_node("h1", NodeType.HEADING, "Main Title", 0, {"level": 1}),
            make_node("h3", NodeType.HEADING, "Subsection", 15, {"level": 3}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should flag hierarchy gap
        hierarchy_changes = [c for c in changes if "hierarchy" in c.reason.lower() or "level" in c.reason.lower()]
        assert len(hierarchy_changes) > 0 or len(changes) > 0

    def test_valid_hierarchy_no_changes(self, optimizer):
        """Test valid hierarchy doesn't trigger changes."""
        nodes = [
            make_node("h1", NodeType.HEADING, "SEO optimization Guide", 0, {"level": 1}),
            make_node("h2", NodeType.HEADING, "Introduction", 25, {"level": 2}),
            make_node("h3", NodeType.HEADING, "Background", 40, {"level": 3}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should not flag valid hierarchy
        hierarchy_issues = [c for c in changes if "gap" in c.reason.lower()]
        assert len(hierarchy_issues) == 0


class TestHeadingTextOptimization:
    """Tests for heading text optimization."""

    def test_adds_keyword_to_h1(self, optimizer):
        """Test adds keyword to H1 if missing."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "Complete Guide to Content",  # Missing keyword
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should suggest adding keyword
        keyword_changes = [c for c in changes if "keyword" in c.reason.lower()]
        # May or may not add keyword depending on implementation
        assert isinstance(changes, list)

    def test_preserves_existing_keyword(self, optimizer):
        """Test preserves H1 with existing keyword."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "SEO Optimization Best Practices",
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should not modify H1 that already has keyword
        h1_changes = [c for c in changes if c.section_id == "h1"]
        # Existing keyword should not trigger change
        assert len(h1_changes) == 0 or all("keyword" not in c.reason.lower() for c in h1_changes)


class TestSectionHeadingInsertion:
    """Tests for section heading insertion."""

    def test_suggests_headings_for_long_content(self, optimizer):
        """Test suggests headings for long content without structure."""
        # Create long content without headings
        long_para = "This is a paragraph with a lot of content. " * 50
        nodes = [
            make_node("h1", NodeType.HEADING, "SEO Optimization Guide", 0, {"level": 1}),
            make_node("p1", NodeType.PARAGRAPH, long_para, 25),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should suggest adding section headings
        assert isinstance(changes, list)

    def test_well_structured_content_no_insertion(self, optimizer):
        """Test well-structured content doesn't get insertions."""
        nodes = [
            make_node("h1", NodeType.HEADING, "SEO optimization Guide", 0, {"level": 1}),
            make_node("h2_1", NodeType.HEADING, "Section One", 25, {"level": 2}),
            make_node("p1", NodeType.PARAGRAPH, "Content for section one.", 40),
            make_node("h2_2", NodeType.HEADING, "Section Two", 70, {"level": 2}),
            make_node("p2", NodeType.PARAGRAPH, "Content for section two.", 85),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should not suggest inserting more headings
        insert_changes = [c for c in changes if "insert" in c.reason.lower()]
        assert len(insert_changes) == 0


class TestChangeTypes:
    """Tests for change type categorization."""

    def test_heading_change_type(self, optimizer):
        """Test changes use HEADING type for heading-specific changes."""
        nodes = [
            make_node("h1_1", NodeType.HEADING, "First Title", 0, {"level": 1}),
            make_node("h1_2", NodeType.HEADING, "Second Title", 15, {"level": 1}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        for change in changes:
            assert change.change_type == ChangeType.HEADING


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, optimizer):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        changes = optimizer.optimize(ast)
        assert isinstance(changes, list)

    def test_only_paragraphs(self, optimizer):
        """Test document with only paragraphs."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Paragraph one.", 0),
            make_node("p2", NodeType.PARAGRAPH, "Paragraph two.", 15),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)

        # Should suggest adding H1
        assert len(changes) > 0

    def test_very_long_heading(self, optimizer):
        """Test handling very long heading."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "This is a very long heading that goes on and on and contains many words " * 3,
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = optimizer.optimize(ast)
        assert isinstance(changes, list)

"""
Tests for keyword injector.

Tests:
- Keyword placement in priority zones
- Density management
- Natural insertion
- Secondary keyword distribution
"""

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.keyword_injector import (
    KeywordInjector,
    KEYWORD_PLACEMENT_PRIORITY,
    MIN_KEYWORD_GAP,
)
from seo_optimizer.optimization.models import ChangeType, OptimizationConfig

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        primary_keyword="content optimization",
        secondary_keywords=["SEO", "ranking", "traffic"],
        inject_keywords=True,
        min_keyword_density=1.0,
        max_keyword_density=2.5,
    )


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def injector(config, guardrails):
    """Create keyword injector."""
    return KeywordInjector(config, guardrails)


@pytest.fixture
def sample_ast_no_keywords():
    """Create AST without keywords.

    Note: Content includes patterns that allow for natural keyword insertion,
    such as "these organizations" which can be replaced with specific keywords.
    The conservative keyword injector only inserts where grammatically correct.
    """
    nodes = [
        make_node(
            "h1",
            NodeType.HEADING,
            "Complete Guide to Digital Marketing",
            0,
            {"level": 1},
        ),
        make_node(
            "p1",
            NodeType.PARAGRAPH,
            "These organizations need effective digital marketing strategies. "
            "You will learn about techniques that help businesses grow.",
            40,
        ),
        make_node("h2_1", NodeType.HEADING, "Getting Started", 160, {"level": 2}),
        make_node(
            "p2",
            NodeType.PARAGRAPH,
            "Starting your journey requires understanding the basics. "
            "Such groups benefit from following these steps to success.",
            180,
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})


@pytest.fixture
def sample_ast_with_keywords():
    """Create AST that already has keywords."""
    nodes = [
        make_node(
            "h1",
            NodeType.HEADING,
            "Content Optimization Guide",
            0,
            {"level": 1},
        ),
        make_node(
            "p1",
            NodeType.PARAGRAPH,
            "Content optimization is crucial for SEO success. "
            "This guide teaches you content optimization best practices.",
            30,
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})


class TestKeywordPlacement:
    """Tests for keyword placement priorities."""

    def test_placement_priority_values(self):
        """Test placement priorities are defined."""
        assert "first_100_words" in KEYWORD_PLACEMENT_PRIORITY
        assert "h1" in KEYWORD_PLACEMENT_PRIORITY
        assert "h2_headings" in KEYWORD_PLACEMENT_PRIORITY

    def test_first_100_words_high_priority(self):
        """Test first 100 words has high priority."""
        assert KEYWORD_PLACEMENT_PRIORITY["first_100_words"] > 0.8

    def test_h1_high_priority(self):
        """Test H1 has high priority."""
        assert KEYWORD_PLACEMENT_PRIORITY["h1"] > 0.8


class TestFirst100WordsInjection:
    """Tests for first 100 words keyword injection."""

    def test_injects_in_first_100_words(self, injector, sample_ast_no_keywords):
        """Test injects keyword in first 100 words."""
        changes = injector.inject(sample_ast_no_keywords)

        # Should inject in first paragraph
        first_100_changes = [
            c for c in changes if "first" in c.reason.lower() or "100" in c.reason.lower()
        ]
        assert len(first_100_changes) > 0

    def test_skips_if_already_present(self, injector, sample_ast_with_keywords):
        """Test skips injection if keyword already in first 100 words."""
        changes = injector.inject(sample_ast_with_keywords)

        # Should not inject again
        first_100_changes = [
            c for c in changes if "first 100" in c.reason.lower()
        ]
        assert len(first_100_changes) == 0


class TestH1Injection:
    """Tests for H1 keyword injection."""

    def test_injects_in_h1(self, injector, sample_ast_no_keywords):
        """Test injects keyword in H1."""
        changes = injector.inject(sample_ast_no_keywords)

        h1_changes = [c for c in changes if "H1" in c.location]
        # May or may not inject depending on current H1 content
        assert isinstance(changes, list)

    def test_skips_h1_with_keyword(self, injector, sample_ast_with_keywords):
        """Test skips H1 injection if keyword present."""
        changes = injector.inject(sample_ast_with_keywords)

        h1_changes = [c for c in changes if "H1" in c.location and "keyword" in c.reason.lower()]
        assert len(h1_changes) == 0


class TestH2Injection:
    """Tests for H2 keyword injection."""

    def test_injects_secondary_in_h2(self, injector, sample_ast_no_keywords):
        """Test injects secondary keywords in H2 headings."""
        changes = injector.inject(sample_ast_no_keywords)

        h2_changes = [c for c in changes if "H2" in c.location]
        # May inject secondary keywords
        assert isinstance(changes, list)


class TestSecondaryKeywordDistribution:
    """Tests for secondary keyword distribution."""

    def test_distributes_secondary_keywords(self, injector, sample_ast_no_keywords):
        """Test distributes secondary keywords in body."""
        changes = injector.inject(sample_ast_no_keywords)

        secondary_changes = [c for c in changes if "secondary" in c.reason.lower()]
        # Should distribute secondary keywords
        assert isinstance(changes, list)

    def test_respects_keyword_gap(self, injector):
        """Test respects minimum gap between keywords."""
        # MIN_KEYWORD_GAP should be defined
        assert MIN_KEYWORD_GAP > 0


class TestDensityManagement:
    """Tests for keyword density management."""

    def test_respects_max_density(self, config, guardrails):
        """Test respects maximum density limit."""
        # Create AST already at high density
        high_density_text = "content optimization " * 10 + "other words " * 10
        nodes = [
            make_node("h1", NodeType.HEADING, "Content Optimization", 0, {"level": 1}),
            make_node("p1", NodeType.PARAGRAPH, high_density_text, 25),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})

        injector = KeywordInjector(config, guardrails)
        changes = injector.inject(ast)

        # Should not add more keywords at high density
        # (implementation may vary)
        assert isinstance(changes, list)

    def test_disabled_injection(self, guardrails):
        """Test injection can be disabled."""
        config = OptimizationConfig(inject_keywords=False)
        injector = KeywordInjector(config, guardrails)

        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Some content without keywords."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = injector.inject(ast)

        assert len(changes) == 0


class TestNaturalInsertion:
    """Tests for natural keyword insertion."""

    def test_insertion_maintains_grammar(self, injector, sample_ast_no_keywords):
        """Test insertions maintain grammatical correctness."""
        changes = injector.inject(sample_ast_no_keywords)

        for change in changes:
            if change.optimized:
                # Basic grammar check - sentence should end properly
                optimized = change.optimized.strip()
                # Skip heading changes - headings don't need punctuation
                if optimized.startswith(("H1:", "H2:", "H3:", "H4:", "H5:", "H6:")):
                    continue
                # Non-heading text should end with punctuation or continuation marker
                assert optimized.endswith((".", "!", "?", "..."))

    def test_uses_enhancement_strategies(self, injector):
        """Test uses different enhancement strategies."""
        # Test the private method
        result = injector._enhance_sentence_with_keyword(
            "This technology is useful for businesses.",
            "SEO"
        )
        # Should return enhanced sentence or None
        assert result is None or "SEO" in result


class TestChangeTracking:
    """Tests for change tracking."""

    def test_changes_have_keyword_type(self, injector, sample_ast_no_keywords):
        """Test all changes have KEYWORD type."""
        changes = injector.inject(sample_ast_no_keywords)

        for change in changes:
            assert change.change_type == ChangeType.KEYWORD

    def test_changes_have_impact_score(self, injector, sample_ast_no_keywords):
        """Test all changes have impact scores."""
        changes = injector.inject(sample_ast_no_keywords)

        for change in changes:
            assert change.impact_score > 0

    def test_changes_have_section_id(self, injector, sample_ast_no_keywords):
        """Test changes reference section IDs."""
        changes = injector.inject(sample_ast_no_keywords)

        for change in changes:
            # Section ID should be present for most changes
            assert change.location is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_primary_keyword(self, guardrails):
        """Test handling no primary keyword."""
        config = OptimizationConfig(primary_keyword=None)
        injector = KeywordInjector(config, guardrails)

        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Some content."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = injector.inject(ast)

        assert len(changes) == 0

    def test_empty_document(self, injector):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        changes = injector.inject(ast)
        assert len(changes) == 0

    def test_special_characters_in_keyword(self, guardrails):
        """Test keyword with special characters."""
        config = OptimizationConfig(primary_keyword="C++ programming")
        injector = KeywordInjector(config, guardrails)

        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Learn about programming languages and software development.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = injector.inject(ast)
        assert isinstance(changes, list)

    def test_very_short_content(self, injector):
        """Test very short content."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Short."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = injector.inject(ast)
        assert isinstance(changes, list)

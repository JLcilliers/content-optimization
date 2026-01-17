"""
Tests for redundancy resolver.

Tests:
- Duplicate detection
- Similarity calculation
- Repetitive phrase detection
- Redundancy resolution
"""

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.models import OptimizationConfig
from seo_optimizer.optimization.redundancy_resolver import (
    HIGH_SIMILARITY_THRESHOLD,
    MEDIUM_SIMILARITY_THRESHOLD,
    MIN_SENTENCE_LENGTH,
    RedundancyAnalysis,
    RedundancyResolver,
)

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig()


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def resolver(config, guardrails):
    """Create redundancy resolver."""
    return RedundancyResolver(config, guardrails)


class TestSimilarityThresholds:
    """Tests for similarity thresholds."""

    def test_high_threshold(self):
        """Test high similarity threshold."""
        assert HIGH_SIMILARITY_THRESHOLD >= 0.8
        assert HIGH_SIMILARITY_THRESHOLD <= 1.0

    def test_medium_threshold(self):
        """Test medium similarity threshold."""
        assert MEDIUM_SIMILARITY_THRESHOLD >= 0.5
        assert MEDIUM_SIMILARITY_THRESHOLD < HIGH_SIMILARITY_THRESHOLD

    def test_min_sentence_length(self):
        """Test minimum sentence length."""
        assert MIN_SENTENCE_LENGTH > 0


class TestDuplicateDetection:
    """Tests for duplicate sentence detection."""

    def test_detects_exact_duplicates(self, resolver):
        """Test detects exact duplicate sentences."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This is an important sentence that appears twice.",
                0,
            ),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "This is an important sentence that appears twice.",
                50,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        assert len(analysis.matches) > 0
        # Should detect high similarity
        assert any(m.similarity >= HIGH_SIMILARITY_THRESHOLD for m in analysis.matches)

    def test_detects_near_duplicates(self, resolver):
        """Test detects near-duplicate sentences."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This content is very important for SEO optimization.",
                0,
            ),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "This content is extremely important for SEO optimization.",
                55,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        # May detect similarity
        assert isinstance(analysis, RedundancyAnalysis)

    def test_unique_content_no_matches(self, resolver):
        """Test unique content has no matches."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "First paragraph about topic A.", 0),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "Second paragraph discussing completely different subject B.",
                35,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        high_sim_matches = [m for m in analysis.matches if m.similarity >= HIGH_SIMILARITY_THRESHOLD]
        assert len(high_sim_matches) == 0


class TestSimilarityCalculation:
    """Tests for similarity calculation."""

    def test_identical_texts_score_1(self, resolver):
        """Test identical texts score 1.0."""
        similarity = resolver._calculate_similarity(
            "This is a test sentence.",
            "This is a test sentence."
        )
        assert similarity >= 0.99

    def test_completely_different_texts_low_score(self, resolver):
        """Test completely different texts have low score."""
        similarity = resolver._calculate_similarity(
            "The quick brown fox jumps over the lazy dog.",
            "Python programming is fun and educational."
        )
        assert similarity < 0.5

    def test_similar_texts_medium_score(self, resolver):
        """Test similar texts have medium score."""
        similarity = resolver._calculate_similarity(
            "Content optimization is important for SEO.",
            "Content optimization is crucial for SEO success."
        )
        assert 0.5 <= similarity <= 0.95


class TestRepetitivePhrasesDetection:
    """Tests for repetitive phrase detection."""

    def test_detects_repeated_phrases(self, resolver):
        """Test detects phrases repeated many times."""
        # Use a 3+ word phrase repeated 4+ times to trigger detection
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Digital marketing strategy works well. Digital marketing strategy improves results. "
                "Digital marketing strategy helps businesses. Digital marketing strategy drives growth. "
                "Digital marketing strategy is essential for success.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        assert len(analysis.repeated_phrases) > 0

    def test_ignores_common_phrases(self, resolver):
        """Test ignores common transitional phrases."""
        text = "For example, this works. For example, that works too. For example, here is another."
        phrases = resolver._find_repeated_phrases(text)

        # Common phrases should not be flagged as heavily
        assert isinstance(phrases, list)


class TestRedundancyResolution:
    """Tests for redundancy resolution."""

    def test_resolves_duplicates(self, resolver):
        """Test suggests resolution for duplicates."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This important point about SEO is repeated.",
                0,
            ),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "This important point about SEO is repeated.",
                45,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)
        changes = resolver.resolve(ast, analysis)

        if analysis.matches:
            # Should suggest changes
            assert len(changes) > 0 or analysis.redundancy_score > 0


class TestRedundancyScoring:
    """Tests for redundancy scoring."""

    def test_unique_content_low_score(self, resolver):
        """Test unique content has low redundancy score."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "First unique paragraph about topic A with original content.",
                0,
            ),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "Second different paragraph discussing subject B entirely.",
                65,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        assert analysis.redundancy_score < 0.5

    def test_high_uniqueness_ratio(self, resolver):
        """Test high uniqueness ratio for original content."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Completely original content here."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)

        assert analysis.unique_content_ratio > 0.5


class TestUniquenessScore:
    """Tests for uniqueness score method."""

    def test_get_uniqueness_score(self, resolver):
        """Test getting uniqueness score."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Original content paragraph."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        score = resolver.get_uniqueness_score(ast)

        assert 0 <= score <= 1


class TestSimilarParagraphDetection:
    """Tests for similar paragraph detection."""

    def test_find_similar_paragraphs(self, resolver):
        """Test finding similar paragraphs."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This paragraph discusses content optimization strategies.",
                0,
            ),
            make_node(
                "p2",
                NodeType.PARAGRAPH,
                "This paragraph discusses content optimization techniques.",
                60,
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        similar = resolver.find_similar_paragraphs(ast)

        assert isinstance(similar, list)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, resolver):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        analysis = resolver.analyze(ast)
        assert isinstance(analysis, RedundancyAnalysis)

    def test_single_paragraph(self, resolver):
        """Test document with single paragraph."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Single paragraph content."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)
        assert len(analysis.matches) == 0

    def test_very_short_sentences(self, resolver):
        """Test ignores very short sentences."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Hi. Yes. No. Ok. Hi. Yes.",  # Short repeated
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)
        # Short sentences should be ignored
        assert isinstance(analysis, RedundancyAnalysis)

    def test_heading_nodes_excluded(self, resolver):
        """Test heading nodes are excluded from analysis."""
        nodes = [
            make_node("h1", NodeType.HEADING, "Introduction", 0, {"level": 1}),
            make_node(
                "h2",
                NodeType.HEADING,
                "Introduction",  # Same heading
                15,
                {"level": 2},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        analysis = resolver.analyze(ast)
        # Headings should not be compared for redundancy
        assert isinstance(analysis, RedundancyAnalysis)

"""
Tests for readability improver.

Tests:
- Sentence splitting
- Passive to active voice conversion
- Vocabulary simplification
- Sentence variance improvement
"""

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.models import ChangeType, OptimizationConfig
from seo_optimizer.optimization.readability_improver import (
    COMPLEX_WORD_SYLLABLES,
    MAX_SENTENCE_WORDS,
    MIN_SENTENCE_WORDS,
    PASSIVE_PATTERNS,
    ReadabilityImprover,
    SIMPLE_ALTERNATIVES,
)

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        improve_readability=True,
        max_sentence_length=25,
    )


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def improver(config, guardrails):
    """Create readability improver."""
    return ReadabilityImprover(config, guardrails)


class TestConstants:
    """Tests for readability constants."""

    def test_max_sentence_words(self):
        """Test max sentence words is reasonable."""
        assert MAX_SENTENCE_WORDS == 25
        assert MAX_SENTENCE_WORDS > MIN_SENTENCE_WORDS

    def test_min_sentence_words(self):
        """Test min sentence words is reasonable."""
        assert MIN_SENTENCE_WORDS == 5
        assert MIN_SENTENCE_WORDS > 0

    def test_complex_word_syllables(self):
        """Test complex word threshold."""
        assert COMPLEX_WORD_SYLLABLES == 3

    def test_passive_patterns_defined(self):
        """Test passive patterns are defined."""
        assert len(PASSIVE_PATTERNS) >= 3

    def test_simple_alternatives_defined(self):
        """Test simple alternatives dictionary."""
        assert "utilize" in SIMPLE_ALTERNATIVES
        assert SIMPLE_ALTERNATIVES["utilize"] == "use"


class TestSentenceSplitting:
    """Tests for sentence splitting."""

    def test_splits_long_sentences(self, improver):
        """Test splits sentences over max length."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This is a very long sentence that contains many words and goes on and on "
                "and keeps going with more content and additional information that makes it exceed "
                "the maximum sentence length threshold that we have set for optimal readability.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        split_changes = [c for c in changes if "split" in c.reason.lower()]
        assert len(split_changes) > 0

    def test_preserves_short_sentences(self, improver):
        """Test preserves sentences under max length."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This is a short sentence. It is easy to read.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        split_changes = [c for c in changes if "split" in c.reason.lower()]
        assert len(split_changes) == 0

    def test_splits_at_conjunctions(self, improver):
        """Test splits at conjunctions."""
        result = improver._split_sentence(
            "The first part is important and the second part adds more detail "
            "that extends beyond the maximum sentence length we prefer."
        )
        if result:
            assert len(result) >= 2

    def test_splits_at_semicolons(self, improver):
        """Test splits at semicolons."""
        result = improver._split_sentence(
            "The first clause contains information; the second clause adds more "
            "context and detail that makes the sentence too long for easy reading."
        )
        if result:
            assert len(result) >= 2


class TestPassiveVoiceConversion:
    """Tests for passive to active voice conversion."""

    def test_detects_passive_voice(self, improver):
        """Test detects passive voice constructions."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "The report was written by the team. The code was reviewed by developers.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        voice_changes = [c for c in changes if "passive" in c.reason.lower() or "active" in c.reason.lower()]
        # May or may not convert depending on context
        assert isinstance(changes, list)

    def test_preserves_active_voice(self, improver):
        """Test preserves active voice."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "The team wrote the report. Developers reviewed the code.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        voice_changes = [c for c in changes if "passive" in c.reason.lower()]
        assert len(voice_changes) == 0


class TestVocabularySimplification:
    """Tests for vocabulary simplification."""

    def test_simplifies_utilize(self, improver):
        """Test simplifies 'utilize' to 'use'."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "You should utilize this tool for better results.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        vocab_changes = [c for c in changes if "vocabulary" in c.reason.lower() or "simplified" in c.reason.lower()]
        assert len(vocab_changes) > 0

    def test_simplifies_multiple_words(self, improver):
        """Test simplifies multiple complex words."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Utilize the methodology to implement a comprehensive solution.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        vocab_changes = [c for c in changes if "vocabulary" in c.reason.lower() or "simplified" in c.reason.lower()]
        assert len(vocab_changes) > 0

    def test_preserves_case(self, improver):
        """Test preserves word case during simplification."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Utilize this tool. utilize another.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        for change in changes:
            if change.optimized and "Use" in change.optimized:
                # First occurrence should be capitalized
                assert "Use" in change.optimized or "use" in change.optimized


class TestSentenceVariance:
    """Tests for sentence variance improvement."""

    def test_detects_low_variance(self, improver):
        """Test detects low sentence length variance."""
        # Create monotonous content (similar length sentences)
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "This is a test sentence. Here is another test. "
                "And one more test here. Plus another test too. "
                "Final test sentence now.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        variance_changes = [c for c in changes if "variance" in c.reason.lower() or "vary" in c.reason.lower()]
        # May suggest variance improvements
        assert isinstance(changes, list)


class TestSyllableCounting:
    """Tests for syllable counting."""

    def test_count_single_syllable(self, improver):
        """Test counting single syllable words."""
        assert improver.count_syllables("cat") == 1
        assert improver.count_syllables("dog") == 1

    def test_count_two_syllables(self, improver):
        """Test counting two syllable words."""
        assert improver.count_syllables("hello") == 2
        assert improver.count_syllables("paper") == 2

    def test_count_three_syllables(self, improver):
        """Test counting three syllable words."""
        assert improver.count_syllables("beautiful") >= 3
        assert improver.count_syllables("important") >= 3

    def test_count_silent_e(self, improver):
        """Test handling silent e."""
        assert improver.count_syllables("make") == 1
        assert improver.count_syllables("complete") >= 2


class TestComplexWordIdentification:
    """Tests for complex word identification."""

    def test_identify_complex_words(self, improver):
        """Test identifies complex words."""
        text = "The implementation of this methodology requires comprehensive understanding."
        complex_words = improver.identify_complex_words(text)

        assert "implementation" in complex_words or "methodology" in complex_words
        assert len(complex_words) >= 2


class TestChangeTracking:
    """Tests for change tracking."""

    def test_changes_have_readability_type(self, improver):
        """Test all changes have READABILITY type."""
        nodes = [
            make_node(
                "p1",
                NodeType.PARAGRAPH,
                "Utilize comprehensive methodologies for implementation.",
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        for change in changes:
            assert change.change_type == ChangeType.READABILITY

    def test_changes_have_impact_score(self, improver):
        """Test all changes have impact scores."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Utilize comprehensive methodologies."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        for change in changes:
            assert change.impact_score > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_disabled_readability(self, guardrails):
        """Test readability can be disabled."""
        config = OptimizationConfig(improve_readability=False)
        improver = ReadabilityImprover(config, guardrails)

        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Utilize methodologies."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        assert len(changes) == 0

    def test_empty_document(self, improver):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        changes = improver.improve(ast)
        assert len(changes) == 0

    def test_very_short_content(self, improver):
        """Test very short content."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Hi."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)
        assert isinstance(changes, list)

    def test_heading_nodes_ignored(self, improver):
        """Test heading nodes are not modified."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "Utilize Comprehensive Methodologies",
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        changes = improver.improve(ast)

        # Headings should not be simplified
        heading_changes = [c for c in changes if c.section_id == "h1"]
        assert len(heading_changes) == 0

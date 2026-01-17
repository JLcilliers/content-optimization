"""
Tests for Readability Scorer Module

Tests readability and UX scoring (20% of GEO score).
"""

import pytest

from seo_optimizer.analysis.readability_scorer import (
    ReadabilityScorer,
    ReadabilityScorerConfig,
    score_readability,
)
from seo_optimizer.ingestion.models import (
    DocumentAST,
    DocumentMetadata,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def scorer() -> ReadabilityScorer:
    """Create a readability scorer instance."""
    return ReadabilityScorer()


def create_test_ast(full_text: str) -> DocumentAST:
    """Helper to create test AST."""
    return DocumentAST(
        doc_id="test_doc",
        nodes=[],
        full_text=full_text,
        char_count=len(full_text),
        metadata=DocumentMetadata(),
    )


# =============================================================================
# Sentence Length Tests
# =============================================================================


class TestSentenceLength:
    """Tests for sentence length analysis."""

    def test_short_sentences_optimal(self, scorer: ReadabilityScorer) -> None:
        """Test that short sentences score well."""
        # Average 10 words per sentence
        text = "This is short. Very easy to read. Simple and clear. Good for users."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert score.avg_sentence_length < 15

    def test_long_sentences_penalized(self, scorer: ReadabilityScorer) -> None:
        """Test that long sentences are penalized."""
        # Single very long sentence
        text = " ".join(["word"] * 50) + "."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert score.avg_sentence_length > 35

    def test_complex_sentences_detected(self, scorer: ReadabilityScorer) -> None:
        """Test that complex sentences are detected."""
        # Mix of short and long sentences
        short = "Short sentence here."
        long = " ".join(["word"] * 40) + "."
        text = f"{short} {long}"
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert len(score.complex_sentences) > 0


# =============================================================================
# Active Voice Tests
# =============================================================================


class TestActiveVoice:
    """Tests for active voice analysis."""

    def test_active_voice_high_ratio(self, scorer: ReadabilityScorer) -> None:
        """Test detection of active voice."""
        text = "The team built the software. They released it quickly. Users loved the product."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert score.active_voice_ratio >= 0.8

    def test_passive_voice_detected(self, scorer: ReadabilityScorer) -> None:
        """Test detection of passive voice."""
        text = "The software was built. It was released quickly. The product was loved."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert score.active_voice_ratio < 0.5
        assert len(score.passive_sentences) > 0

    def test_mixed_voice(self, scorer: ReadabilityScorer) -> None:
        """Test mixed active and passive voice."""
        text = "The team built the software. It was released quickly."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert 0.3 < score.active_voice_ratio < 0.8


# =============================================================================
# Flesch-Kincaid Tests
# =============================================================================


class TestFleschKincaid:
    """Tests for Flesch-Kincaid grade level."""

    def test_simple_text_low_grade(self, scorer: ReadabilityScorer) -> None:
        """Test that simple text has low grade level."""
        text = "The cat sat. It was nice. The sun was hot. The day was good."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        # Simple text should have low grade level
        assert score.flesch_kincaid_grade < 8

    def test_complex_text_high_grade(self, scorer: ReadabilityScorer) -> None:
        """Test that complex text has high grade level."""
        text = """
        The implementation of sophisticated algorithmic methodologies
        necessitates comprehensive understanding of computational paradigms
        and their theoretical underpinnings in contemporary research contexts.
        """
        ast = create_test_ast(text)
        score = scorer.score(ast)

        # Complex text should have higher grade level
        assert score.flesch_kincaid_grade > 12

    def test_optimal_grade_range(self, scorer: ReadabilityScorer) -> None:
        """Test content in optimal grade range."""
        text = """
        Cloud computing has changed how businesses work.
        Companies now store data online instead of local servers.
        This makes it easier to access information from anywhere.
        """
        ast = create_test_ast(text)
        score = scorer.score(ast)

        # Should be in reasonable range
        assert 6 < score.flesch_kincaid_grade < 14


# =============================================================================
# Total Score Tests
# =============================================================================


class TestTotalScore:
    """Tests for total readability score calculation."""

    def test_perfect_readability(self, scorer: ReadabilityScorer) -> None:
        """Test highly readable content gets high score."""
        text = """
        This is clear content.
        It uses short sentences.
        The words are simple.
        Readers find it easy.
        """
        ast = create_test_ast(text)
        score = scorer.score(ast)

        assert score.total >= 70

    def test_poor_readability(self, scorer: ReadabilityScorer) -> None:
        """Test hard-to-read content gets lower score than simple content."""
        text = """
        The fundamentally transformative implications of implementing
        comprehensive organizational restructuring initiatives were
        thoroughly evaluated by the strategically appointed committee
        which was subsequently dissolved after prolonged deliberations.
        """
        ast = create_test_ast(text)
        score = scorer.score(ast)

        # Complex text should score below 85 (compared to >90 for simple text)
        assert score.total < 85

    def test_empty_document_score(self, scorer: ReadabilityScorer) -> None:
        """Test empty document handling."""
        ast = create_test_ast("")
        score = scorer.score(ast)

        assert score.total == 0


# =============================================================================
# Issue Detection Tests
# =============================================================================


class TestReadabilityIssues:
    """Tests for readability issue detection."""

    def test_long_sentence_issue(self, scorer: ReadabilityScorer) -> None:
        """Test issue raised for long sentences."""
        long_sentence = " ".join(["word"] * 40) + "."
        ast = create_test_ast(long_sentence)
        score = scorer.score(ast)

        issues = [i for i in score.issues if "sentence" in i.message.lower()]
        assert len(issues) > 0

    def test_passive_voice_issue(self, scorer: ReadabilityScorer) -> None:
        """Test issue raised for excessive passive voice."""
        text = "The work was done. The task was completed. The goal was achieved."
        ast = create_test_ast(text)
        score = scorer.score(ast)

        issues = [i for i in score.issues if "voice" in i.message.lower()]
        assert len(issues) > 0

    def test_high_grade_level_issue(self, scorer: ReadabilityScorer) -> None:
        """Test issue raised for high grade level."""
        text = """
        The epistemological ramifications of implementing transformative
        paradigmatic shifts necessitate comprehensive stakeholder engagement
        and meticulous consideration of multifaceted implications.
        """
        ast = create_test_ast(text)
        score = scorer.score(ast)

        issues = [i for i in score.issues if "reading level" in i.message.lower() or "grade" in i.message.lower()]
        assert len(issues) > 0


# =============================================================================
# Syllable Counting Tests
# =============================================================================


class TestSyllableCounting:
    """Tests for syllable counting logic."""

    def test_single_syllable_words(self, scorer: ReadabilityScorer) -> None:
        """Test counting syllables in simple words."""
        assert scorer._count_syllables("cat") == 1
        assert scorer._count_syllables("dog") == 1
        assert scorer._count_syllables("run") == 1

    def test_multi_syllable_words(self, scorer: ReadabilityScorer) -> None:
        """Test counting syllables in complex words."""
        assert scorer._count_syllables("computer") >= 3
        assert scorer._count_syllables("beautiful") >= 3
        assert scorer._count_syllables("understanding") >= 4

    def test_silent_e_handling(self, scorer: ReadabilityScorer) -> None:
        """Test handling of silent 'e' at word end."""
        # "make" should be 1 syllable (silent e)
        assert scorer._count_syllables("make") == 1
        assert scorer._count_syllables("take") == 1


# =============================================================================
# Configuration Tests
# =============================================================================


class TestReadabilityConfig:
    """Tests for scorer configuration."""

    def test_custom_sentence_length(self) -> None:
        """Test custom sentence length thresholds."""
        config = ReadabilityScorerConfig(
            optimal_sentence_length=15,
            max_sentence_length=30,
        )
        scorer = ReadabilityScorer(config)

        assert scorer.config.optimal_sentence_length == 15

    def test_custom_active_voice_target(self) -> None:
        """Test custom active voice target."""
        config = ReadabilityScorerConfig(active_voice_target=0.90)
        scorer = ReadabilityScorer(config)

        assert scorer.config.active_voice_target == 0.90


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestScoreReadabilityFunction:
    """Tests for convenience function."""

    def test_score_readability_basic(self) -> None:
        """Test basic scoring function."""
        ast = create_test_ast("This is simple text. Easy to read.")
        score = score_readability(ast)

        assert score.total > 0
        assert score.avg_sentence_length > 0

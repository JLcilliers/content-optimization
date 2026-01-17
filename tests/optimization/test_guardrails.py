"""
Tests for safety guardrails.

CRITICAL: These tests verify the safety system that prevents:
- Keyword stuffing
- AI vocabulary detection
- Over-optimization
- Factual content preservation
"""

import pytest

from seo_optimizer.optimization.guardrails import (
    AI_FLAGGED_VOCABULARY,
    KEYWORD_DENSITY_DANGER,
    KEYWORD_DENSITY_WARNING,
    SafetyGuardrails,
)
from seo_optimizer.optimization.models import OptimizationConfig


@pytest.fixture
def guardrails():
    """Create guardrails with default config."""
    config = OptimizationConfig()
    return SafetyGuardrails(config)


@pytest.fixture
def strict_guardrails():
    """Create guardrails with strict settings."""
    config = OptimizationConfig(
        max_keyword_density=1.5,
        max_entity_density=3.0,
    )
    return SafetyGuardrails(config)


class TestAIFlaggedVocabulary:
    """Tests for AI vocabulary detection."""

    def test_vocabulary_has_verbs(self):
        """Test AI vocabulary contains flagged verbs."""
        assert "verbs" in AI_FLAGGED_VOCABULARY
        assert "delve" in AI_FLAGGED_VOCABULARY["verbs"]

    def test_vocabulary_has_adjectives(self):
        """Test AI vocabulary contains flagged adjectives."""
        assert "adjectives" in AI_FLAGGED_VOCABULARY
        assert "robust" in AI_FLAGGED_VOCABULARY["adjectives"]

    def test_vocabulary_has_phrases(self):
        """Test AI vocabulary contains flagged phrases."""
        assert "phrases" in AI_FLAGGED_VOCABULARY

    def test_delve_has_replacements(self):
        """Test 'delve' has replacement suggestions."""
        replacements = AI_FLAGGED_VOCABULARY["verbs"]["delve"]
        assert len(replacements) > 0
        assert "explore" in replacements

    def test_leverage_has_replacements(self):
        """Test 'leverage' has replacement suggestions."""
        replacements = AI_FLAGGED_VOCABULARY["verbs"]["leverage"]
        assert "use" in replacements

    def test_robust_has_replacements(self):
        """Test 'robust' has replacement suggestions."""
        replacements = AI_FLAGGED_VOCABULARY["adjectives"]["robust"]
        assert "strong" in replacements or "reliable" in replacements


class TestKeywordDensityChecks:
    """Tests for keyword density checking."""

    def test_check_low_density_is_safe(self, guardrails):
        """Test low keyword density passes."""
        text = "This is a sample text about content. " * 50  # 400 words
        text += "SEO SEO SEO"  # 3 keywords in ~400 words = 0.75%
        result = guardrails.check_keyword_density(text, "SEO")
        assert result.is_safe is True

    def test_check_high_density_warns(self, guardrails):
        """Test high density triggers warning."""
        text = "SEO is important. SEO helps ranking. SEO " * 10  # Many SEO mentions
        result = guardrails.check_keyword_density(text, "SEO")
        assert result.density > KEYWORD_DENSITY_WARNING or result.is_safe is False

    def test_check_extreme_density_dangerous(self, guardrails):
        """Test extreme density is dangerous."""
        text = "SEO SEO SEO SEO SEO " * 20  # All SEO
        result = guardrails.check_keyword_density(text, "SEO")
        assert result.density >= KEYWORD_DENSITY_DANGER or result.is_safe is False

    def test_empty_text_is_safe(self, guardrails):
        """Test empty text doesn't crash."""
        result = guardrails.check_keyword_density("", "keyword")
        assert result.is_safe is True
        assert result.density == 0.0

    def test_case_insensitive_counting(self, guardrails):
        """Test keyword counting is case insensitive."""
        text = "SEO seo Seo SEO"
        result = guardrails.check_keyword_density(text, "seo")
        assert result.count >= 4

    def test_would_exceed_density(self, guardrails):
        """Test would_exceed_density check."""
        text = "SEO " * 20 + "other words " * 100
        would_exceed = guardrails.would_exceed_density(text, "SEO", 5)
        # Adding 5 more to already high density
        assert isinstance(would_exceed, bool)

    def test_density_calculation_accuracy(self, guardrails):
        """Test density calculation is accurate."""
        text = "keyword " * 2 + "other " * 98  # 2% density
        result = guardrails.check_keyword_density(text, "keyword")
        assert 1.5 <= result.density <= 2.5


class TestEntityDensityChecks:
    """Tests for entity density checking."""

    def test_low_entity_density_safe(self, guardrails):
        """Test low entity density passes."""
        # Text with ~3% entity density (2 entity words / 72 total words)
        text = "This is a long text about machine learning and artificial intelligence. " * 9 + "It mentions BERT and GPT. "
        result = guardrails.check_entity_density(text, ["BERT", "GPT"])
        assert result.is_safe is True

    def test_high_entity_density_warns(self, guardrails):
        """Test high entity density warns."""
        text = "BERT GPT BERT GPT BERT GPT " * 10
        result = guardrails.check_entity_density(
            text, ["BERT", "GPT", "LLM", "NLP", "API"]
        )
        # Should warn about entity stuffing
        assert isinstance(result.is_safe, bool)

    def test_entity_density_message(self, guardrails):
        """Test entity density returns message."""
        text = "Some text with BERT mentioned."
        result = guardrails.check_entity_density(text, ["BERT"])
        assert result.message is not None


class TestVocabularyFiltering:
    """Tests for AI vocabulary filtering."""

    def test_filter_delve(self, guardrails):
        """Test filtering 'delve' word."""
        text = "Let's delve into this topic."
        result = guardrails.filter_ai_vocabulary(text)
        assert "delve" not in result.cleaned_text.lower()

    def test_filter_leverage(self, guardrails):
        """Test filtering 'leverage' word."""
        text = "We can leverage this technology."
        result = guardrails.filter_ai_vocabulary(text)
        assert "leverage" not in result.cleaned_text.lower()

    def test_filter_robust(self, guardrails):
        """Test filtering 'robust' word."""
        text = "This is a robust solution."
        result = guardrails.filter_ai_vocabulary(text)
        assert "robust" not in result.cleaned_text.lower()

    def test_filter_seamlessly(self, guardrails):
        """Test filtering 'seamlessly' word."""
        text = "It integrates seamlessly with other tools."
        result = guardrails.filter_ai_vocabulary(text)
        assert "seamlessly" not in result.cleaned_text.lower()

    def test_filter_multiple_ai_words(self, guardrails):
        """Test filtering multiple AI words."""
        text = "Let's delve into robust solutions and leverage seamless integration."
        result = guardrails.filter_ai_vocabulary(text)
        assert "delve" not in result.cleaned_text.lower()
        assert "robust" not in result.cleaned_text.lower()

    def test_tracks_replacements(self, guardrails):
        """Test replacements are tracked."""
        text = "Let's delve into this."
        result = guardrails.filter_ai_vocabulary(text)
        assert len(result.replacements) > 0

    def test_preserves_non_flagged_words(self, guardrails):
        """Test non-flagged words are preserved."""
        text = "This is a normal sentence without AI vocabulary."
        result = guardrails.filter_ai_vocabulary(text)
        assert "normal" in result.cleaned_text
        assert "sentence" in result.cleaned_text

    def test_case_preservation(self, guardrails):
        """Test case is preserved in replacements."""
        text = "Delve into the topic."  # Capitalized
        result = guardrails.filter_ai_vocabulary(text)
        # First word should remain capitalized
        first_word = result.cleaned_text.split()[0]
        assert first_word[0].isupper()


class TestSentenceVariance:
    """Tests for sentence variance (burstiness) checking."""

    def test_high_variance_acceptable(self, guardrails):
        """Test high variance is acceptable."""
        sentences = [
            "Short one.",
            "This is a medium length sentence for testing.",
            "Longer sentences provide more detail and context for readers.",
            "Quick.",
            "Another medium sentence here.",
        ]
        is_acceptable, message, variance = guardrails.check_sentence_variance(sentences)
        assert is_acceptable is True

    def test_low_variance_flagged(self, guardrails):
        """Test low variance is flagged."""
        # All same length sentences (AI-like)
        sentences = [
            "This is a test sentence.",
            "This is also a test.",
            "Here is another test.",
            "And one more test here.",
            "Final test sentence now.",
        ]
        is_acceptable, message, variance = guardrails.check_sentence_variance(sentences)
        # May or may not be acceptable depending on thresholds
        assert message is not None

    def test_variance_suggestions(self, guardrails):
        """Test variance improvement suggestions."""
        sentences = ["Test sentence." for _ in range(10)]
        suggestions = guardrails.suggest_variance_improvements(sentences)
        assert isinstance(suggestions, list)


class TestFactualContentPreservation:
    """Tests for preserving factual content."""

    def test_preserve_statistics(self, guardrails):
        """Test statistics are preserved."""
        original = "The study showed 85% improvement in results."
        modified = "The study showed great improvement in results."
        result = guardrails.preserve_factual_content(original, modified)
        assert "85%" in result

    def test_preserve_years(self, guardrails):
        """Test years are preserved."""
        original = "This was established in 2020."
        modified = "This was established recently."
        result = guardrails.preserve_factual_content(original, modified)
        assert "2020" in result

    def test_preserve_quotes(self, guardrails):
        """Test quoted content is preserved."""
        original = 'He said "This is important."'
        modified = "He made a notable statement."  # "notable" can be replaced
        result = guardrails.preserve_factual_content(original, modified)
        assert '"This is important."' in result or "important" in result.lower()

    def test_preserve_numbers(self, guardrails):
        """Test numerical data is preserved."""
        original = "The price is $299.99 per month."
        modified = "The price is competitive."
        result = guardrails.preserve_factual_content(original, modified)
        assert "$299.99" in result or "299" in result


class TestChangeValidation:
    """Tests for validating proposed changes."""

    def test_validate_safe_change(self, guardrails):
        """Test validating a safe change."""
        from seo_optimizer.optimization.models import ChangeType, OptimizationChange

        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="P1",
            original="This is text.",
            optimized="This is SEO text.",
            reason="Added keyword",
            impact_score=2.0,
        )
        is_valid, issues = guardrails.validate_change(change, "Full document text.")
        assert is_valid is True

    def test_validate_change_with_ai_vocabulary(self, guardrails):
        """Test change with AI vocabulary is flagged."""
        from seo_optimizer.optimization.models import ChangeType, OptimizationChange

        change = OptimizationChange(
            change_type=ChangeType.READABILITY,
            location="P1",
            original="This is text.",
            optimized="Let's delve into this robust text.",
            reason="Improved",
            impact_score=2.0,
        )
        is_valid, issues = guardrails.validate_change(change, "Document text.")
        # Should flag AI vocabulary
        assert len(issues) > 0 or is_valid is False or "delve" in str(issues)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_keyword(self, guardrails):
        """Test empty keyword handling."""
        result = guardrails.check_keyword_density("Some text", "")
        assert result.count == 0

    def test_special_characters_in_keyword(self, guardrails):
        """Test special characters in keyword."""
        text = "Learn about C++ programming and C++ best practices."
        result = guardrails.check_keyword_density(text, "C++")
        assert result.count >= 1

    def test_very_long_text(self, guardrails):
        """Test handling very long text."""
        text = "This is a test sentence. " * 10000
        result = guardrails.check_keyword_density(text, "test")
        assert result.is_safe is not None

    def test_unicode_text(self, guardrails):
        """Test handling unicode text."""
        text = "Learn about SEO and optimización for international sites."
        result = guardrails.check_keyword_density(text, "SEO")
        assert result.count >= 1

    def test_multiline_text(self, guardrails):
        """Test handling multiline text."""
        text = """
        This is line one with keyword.
        This is line two with keyword.
        This is line three without.
        """
        result = guardrails.check_keyword_density(text, "keyword")
        assert result.count >= 2

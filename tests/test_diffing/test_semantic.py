"""
Tests for semantic similarity detection.

These tests ensure that semantically equivalent content
(rewording) is NOT highlighted as new content.
"""

import pytest

from seo_optimizer.diffing.semantic import (
    CONSERVATIVE_THRESHOLD,
    REDUNDANCY_THRESHOLD,
    SEMANTIC_EQUIVALENCE_THRESHOLD,
    SemanticMatcher,
    compute_batch_similarity,
    compute_semantic_similarity,
    get_embedding,
    is_redundant_content,
    is_semantic_match,
    is_semantically_equivalent,
)


class TestSemanticMatcherInit:
    """Tests for SemanticMatcher initialization."""

    def test_default_thresholds(self) -> None:
        """SemanticMatcher should use research-validated default thresholds."""
        matcher = SemanticMatcher()
        assert matcher.equivalence_threshold == SEMANTIC_EQUIVALENCE_THRESHOLD
        assert matcher.redundancy_threshold == REDUNDANCY_THRESHOLD

    def test_custom_thresholds(self) -> None:
        """SemanticMatcher should accept custom thresholds."""
        matcher = SemanticMatcher(
            equivalence_threshold=0.90,
            redundancy_threshold=0.95,
        )
        assert matcher.equivalence_threshold == 0.90
        assert matcher.redundancy_threshold == 0.95


class TestComputeSemanticSimilarity:
    """Tests for compute_semantic_similarity function."""

    @pytest.mark.slow
    def test_identical_text_similarity_is_one(self) -> None:
        """Identical text should have similarity of 1.0."""
        similarity = compute_semantic_similarity("Hello world", "Hello world")
        assert similarity == 1.0

    @pytest.mark.slow
    def test_empty_string_handling(self) -> None:
        """Empty strings should be handled gracefully."""
        # Empty vs non-empty should be 0
        similarity = compute_semantic_similarity("", "Hello world")
        assert similarity == 0.0

        # Both empty should be 1
        similarity = compute_semantic_similarity("", "")
        assert similarity == 1.0


class TestBatchSimilarity:
    """Tests for compute_batch_similarity function."""

    def test_batch_similarity_length_validation(self) -> None:
        """Lists must have same length."""
        with pytest.raises(ValueError, match="same length"):
            compute_batch_similarity(["a", "b"], ["c"])

    def test_batch_similarity_empty_lists(self) -> None:
        """Empty lists should return empty results."""
        result = compute_batch_similarity([], [])
        assert result == []


class TestIsSemanticMatch:
    """Tests for is_semantic_match function."""

    @pytest.mark.slow
    def test_is_semantic_match_returns_tuple(self) -> None:
        """is_semantic_match should return (bool, float) tuple."""
        is_match, score = is_semantic_match("Hello world", "Hello world")
        assert isinstance(is_match, bool)
        assert isinstance(score, float)
        assert is_match is True
        assert score == 1.0


class TestGetEmbedding:
    """Tests for get_embedding function."""

    @pytest.mark.slow
    def test_get_embedding_returns_list(self) -> None:
        """get_embedding should return a list of floats."""
        embedding = get_embedding("sample text")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


class TestResearchThresholds:
    """Tests to verify research-derived thresholds."""

    def test_semantic_equivalence_threshold(self) -> None:
        """Content_Scoring: >= 0.85 = semantically equivalent."""
        assert SEMANTIC_EQUIVALENCE_THRESHOLD == 0.85

    def test_redundancy_threshold(self) -> None:
        """Content_Scoring: > 0.90 = redundant content."""
        assert REDUNDANCY_THRESHOLD == 0.90

    def test_conservative_threshold(self) -> None:
        """Conservative mode floor at 0.80."""
        assert CONSERVATIVE_THRESHOLD == 0.80


class TestIsSemanticallyEquivalent:
    """Tests for is_semantically_equivalent function."""

    @pytest.mark.slow
    def test_identical_is_equivalent(self) -> None:
        """Identical text should be equivalent."""
        assert is_semantically_equivalent("Hello world", "Hello world") is True

    @pytest.mark.slow
    def test_very_different_not_equivalent(self) -> None:
        """Very different text should not be equivalent."""
        assert is_semantically_equivalent(
            "The cat sat on the mat",
            "Quantum physics explains the universe"
        ) is False


class TestIsRedundantContent:
    """Tests for is_redundant_content function."""

    @pytest.mark.slow
    def test_identical_is_redundant(self) -> None:
        """Identical text should be redundant."""
        assert is_redundant_content("Hello world", "Hello world") is True

    @pytest.mark.slow
    def test_very_different_not_redundant(self) -> None:
        """Very different text should not be redundant."""
        assert is_redundant_content(
            "The product helps users",
            "The weather is sunny today"
        ) is False

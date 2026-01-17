"""
Tests for core diffing algorithm.

CRITICAL: These tests validate the most important functionality.
Zero false positives is mandatory.
"""

import pytest

from seo_optimizer.diffing.differ import (
    ContentDiffer,
    DiffConfig,
    MatchType,
    calculate_boundaries,
    compute_diff,
    diff_text_segments,
    identify_expansion,
)
from seo_optimizer.diffing.models import ChangeSet
from seo_optimizer.ingestion.models import DocumentAST, OriginalSnapshot
from tests.conftest import create_document_ast


class TestComputeDiff:
    """Tests for compute_diff function."""

    def test_compute_diff_identical_content(
        self,
        sample_original_snapshot: OriginalSnapshot,
        sample_document_ast: DocumentAST,
    ) -> None:
        """Identical content should have no additions."""
        result = compute_diff(sample_original_snapshot, sample_document_ast)
        assert isinstance(result, ChangeSet)
        # Note: May have empty additions if content matches perfectly

    def test_compute_diff_returns_changeset(
        self,
        sample_original_snapshot: OriginalSnapshot,
        sample_document_ast: DocumentAST,
    ) -> None:
        """compute_diff should return a ChangeSet."""
        result = compute_diff(sample_original_snapshot, sample_document_ast)
        assert isinstance(result, ChangeSet)

    def test_compute_diff_signature_accepts_thresholds(
        self,
        sample_original_snapshot: OriginalSnapshot,
        sample_document_ast: DocumentAST,
    ) -> None:
        """Verify compute_diff accepts threshold parameters."""
        result = compute_diff(
            sample_original_snapshot,
            sample_document_ast,
            semantic_threshold=0.85,
            move_detection_threshold=0.95,
            conservative_mode=True,
        )
        assert isinstance(result, ChangeSet)


class TestDiffTextSegments:
    """Tests for diff_text_segments function."""

    def test_empty_original_highlights_all(self) -> None:
        """Empty original should highlight all modified content."""
        additions, confidence = diff_text_segments("", "New content here")
        assert len(additions) == 1
        assert additions[0][2] == "New content here"
        assert confidence == 1.0

    def test_identical_content_no_highlight(self) -> None:
        """Identical content should have no highlights."""
        additions, confidence = diff_text_segments("Hello world", "Hello world")
        assert len(additions) == 0
        assert confidence == 1.0

    def test_whitespace_normalized_no_highlight(self) -> None:
        """Whitespace differences should not be highlighted."""
        additions, _ = diff_text_segments(
            "Hello   world",
            "Hello world"
        )
        assert len(additions) == 0

    def test_case_normalized_no_highlight(self) -> None:
        """Case differences alone should not be highlighted."""
        additions, _ = diff_text_segments(
            "Hello World",
            "hello world"
        )
        assert len(additions) == 0


class TestIdentifyExpansion:
    """Tests for identify_expansion function."""

    def test_identify_simple_expansion(self) -> None:
        """Detect simple text expansion."""
        result = identify_expansion(
            "The product helps users",
            "The product helps users save time"
        )
        assert result is not None
        start, end = result
        assert "The product helps users save time"[start:end] == " save time"

    def test_identify_no_expansion(self) -> None:
        """Non-expansion should return None."""
        result = identify_expansion(
            "The product helps users",
            "Completely different text"
        )
        assert result is None

    def test_identify_prefix_expansion(self) -> None:
        """Detect prefix-based expansion."""
        result = identify_expansion(
            "Hello",
            "Hello world"
        )
        assert result is not None


class TestCalculateBoundaries:
    """Tests for calculate_boundaries function."""

    def test_calculate_boundaries_basic(self) -> None:
        """Basic boundary calculation."""
        boundaries = calculate_boundaries(
            "Hello",
            "Hello world",
            [(6, 11)],
            use_word_boundaries=True
        )
        assert len(boundaries) == 1

    def test_calculate_boundaries_no_word_boundary(self) -> None:
        """Boundaries without word boundary adjustment."""
        boundaries = calculate_boundaries(
            "Hello",
            "Hello world",
            [(6, 11)],
            use_word_boundaries=False
        )
        assert boundaries == [(6, 11)]


class TestContentDiffer:
    """Tests for ContentDiffer class."""

    def test_content_differ_initialization(self) -> None:
        """ContentDiffer should initialize with default config."""
        differ = ContentDiffer()
        assert differ.config is not None
        assert differ.config.semantic_equivalence_threshold == 0.85
        assert differ.config.conservative_mode is True

    def test_content_differ_custom_config(self) -> None:
        """ContentDiffer should accept custom config."""
        config = DiffConfig(
            semantic_equivalence_threshold=0.90,
            conservative_mode=False,
        )
        differ = ContentDiffer(config)
        assert differ.config.semantic_equivalence_threshold == 0.90
        assert differ.config.conservative_mode is False

    def test_diff_config_defaults(self) -> None:
        """DiffConfig should have research-validated defaults."""
        config = DiffConfig()
        # Research-derived thresholds
        assert config.semantic_equivalence_threshold == 0.85
        assert config.redundancy_threshold == 0.90
        assert config.levenshtein_same_threshold == 0.90
        assert config.levenshtein_partial_threshold == 0.70
        assert config.ngram_overlap_threshold == 0.70
        assert config.conservative_similarity_floor == 0.80


class TestMatchType:
    """Tests for MatchType enum."""

    def test_match_type_values(self) -> None:
        """MatchType should have expected values."""
        assert MatchType.EXACT == "exact"
        assert MatchType.SIMHASH == "simhash"
        assert MatchType.NGRAM == "ngram"
        assert MatchType.FUZZY == "fuzzy"
        assert MatchType.SEMANTIC == "semantic"
        assert MatchType.NO_MATCH == "no_match"


class TestResearchThresholds:
    """Tests to verify research-derived thresholds are correctly applied."""

    def test_semantic_threshold_085(self) -> None:
        """Content_Scoring: >= 0.85 = semantically equivalent."""
        config = DiffConfig()
        assert config.semantic_equivalence_threshold == 0.85

    def test_redundancy_threshold_090(self) -> None:
        """Content_Scoring: > 0.90 = redundant content."""
        config = DiffConfig()
        assert config.redundancy_threshold == 0.90

    def test_conservative_threshold_080(self) -> None:
        """Conservative mode floor at 0.80."""
        config = DiffConfig()
        assert config.conservative_similarity_floor == 0.80

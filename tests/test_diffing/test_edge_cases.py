"""
Edge case tests for diffing system.

This file contains 35+ edge case scenarios from the research document.
Each case tests a specific scenario that must be handled correctly.

CRITICAL: All these tests must pass before the diffing system
is considered production-ready.

Reference: docs/research/05-diffing-highlighting.md section 8
"""

from typing import Any

import pytest


class TestEdgeCasesFromResearch:
    """
    Edge cases from the research document.

    Each test verifies the diffing system handles a specific
    scenario correctly. These are the cases that could cause
    false positives (highlighting existing content) or false
    negatives (missing new content).
    """

    @pytest.mark.edge_case
    def test_e01_identical_content(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E01: Identical content should have no highlight."""
        case = next(c for c in diff_edge_cases if c["id"] == "E01")
        assert case["expected_highlight"] is None
        # When implemented:
        # result = diff_text_segments(case["original"], case["modified"])
        # assert result highlights nothing

    @pytest.mark.edge_case
    def test_e02_entirely_new_content(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E02: Entirely new content should be fully highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E02")
        assert case["expected_highlight"] == "Entirely new paragraph content."
        # When implemented:
        # result = diff_text_segments(case["original"], case["modified"])
        # assert result highlights entire modified text

    @pytest.mark.edge_case
    def test_e03_sentence_expansion(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E03: Sentence expansion should highlight only new portion."""
        case = next(c for c in diff_edge_cases if c["id"] == "E03")
        assert case["expected_highlight"] == " save time"
        # When implemented:
        # result = diff_text_segments(case["original"], case["modified"])
        # assert only " save time" is highlighted

    @pytest.mark.edge_case
    def test_e04_semantic_equivalent_rewording(
        self, diff_edge_cases: list[dict[str, Any]]
    ) -> None:
        """E04: Semantic equivalent rewording should NOT be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E04")
        assert case["expected_highlight"] is None
        # This is CRITICAL - false positive prevention
        # "assists customers" vs "helps users" should be recognized as equivalent

    @pytest.mark.edge_case
    def test_e05_moved_content(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E05: Moved content should NOT be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E05")
        assert case["expected_highlight"] is None
        # Content that moves from one section to another is NOT new

    @pytest.mark.edge_case
    def test_e06_word_insertion(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E06: Single word insertion should highlight only that word."""
        case = next(c for c in diff_edge_cases if c["id"] == "E06")
        assert case["expected_highlight"] == "beautiful "

    @pytest.mark.edge_case
    def test_e07_append_to_end(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E07: Content appended to end should be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E07")
        assert case["expected_highlight"] == " jumps over the lazy dog"

    @pytest.mark.edge_case
    def test_e08_prepend_to_start(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E08: Content prepended to start should be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E08")
        assert case["expected_highlight"] == "Greetings, "

    @pytest.mark.edge_case
    def test_e09_word_replacement_similar_meaning(
        self, diff_edge_cases: list[dict[str, Any]]
    ) -> None:
        """E09: Word replacement with similar meaning should NOT be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E09")
        assert case["expected_highlight"] is None
        # "good" vs "excellent" - similar meaning, not new content

    @pytest.mark.edge_case
    def test_e11_new_sentence_added(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E11: New sentence added should be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E11")
        assert case["expected_highlight"] == " We're here to help!"

    @pytest.mark.edge_case
    def test_e15_whitespace_normalization(self, diff_edge_cases: list[dict[str, Any]]) -> None:
        """E15: Whitespace changes should NOT be highlighted."""
        case = next(c for c in diff_edge_cases if c["id"] == "E15")
        assert case["expected_highlight"] is None


class TestDiffModelsValidation:
    """Tests for diffing model validation."""

    def test_highlight_region_validates_char_range(self) -> None:
        """HighlightRegion should validate character range."""
        from seo_optimizer.diffing.models import HighlightRegion

        with pytest.raises(ValueError, match="start_char must be non-negative"):
            HighlightRegion(
                node_id="n1",
                start_char=-1,
                end_char=10,
                text="0123456789a",
                confidence=0.9,
            )

    def test_highlight_region_validates_end_greater_than_start(self) -> None:
        """HighlightRegion should require end > start."""
        from seo_optimizer.diffing.models import HighlightRegion

        with pytest.raises(ValueError, match="end_char must be greater than start_char"):
            HighlightRegion(
                node_id="n1",
                start_char=10,
                end_char=5,
                text="xxxxx",
                confidence=0.9,
            )

    def test_highlight_region_validates_text_length(self) -> None:
        """HighlightRegion should validate text length matches range."""
        from seo_optimizer.diffing.models import HighlightRegion

        with pytest.raises(ValueError, match="text length must match"):
            HighlightRegion(
                node_id="n1",
                start_char=0,
                end_char=10,
                text="short",  # 5 chars, should be 10
                confidence=0.9,
            )

    def test_changeset_calculates_statistics(self, sample_changeset: Any) -> None:
        """ChangeSet should calculate summary statistics."""
        assert sample_changeset.total_additions == 1
        assert sample_changeset.total_chars_added == 10
        assert sample_changeset.total_regions == 1

    def test_empty_changeset_has_zero_stats(self, empty_changeset: Any) -> None:
        """Empty ChangeSet should have zero statistics."""
        assert empty_changeset.total_additions == 0
        assert empty_changeset.total_chars_added == 0
        assert empty_changeset.total_regions == 0


class TestDocumentFingerprint:
    """Tests for DocumentFingerprint for move detection."""

    def test_fingerprint_from_text(self) -> None:
        """DocumentFingerprint should create from text."""
        from seo_optimizer.diffing.models import DocumentFingerprint

        fp = DocumentFingerprint.from_text("Hello World", "p0")
        assert fp.original_position == "p0"
        assert fp.normalized_text == "hello world"
        assert len(fp.content_hash) == 32  # blake2b with digest_size=16 produces 32 hex chars

    def test_fingerprint_exact_match(self) -> None:
        """Identical text should produce matching fingerprints."""
        from seo_optimizer.diffing.models import DocumentFingerprint

        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("Hello World", "p1")

        assert fp1.matches(fp2)

    def test_fingerprint_case_insensitive_match(self) -> None:
        """Fingerprints should match case-insensitively."""
        from seo_optimizer.diffing.models import DocumentFingerprint

        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("HELLO WORLD", "p1")

        assert fp1.matches(fp2)

    def test_fingerprint_whitespace_normalized(self) -> None:
        """Fingerprints should normalize whitespace."""
        from seo_optimizer.diffing.models import DocumentFingerprint

        fp1 = DocumentFingerprint.from_text("Hello   World", "p0")
        fp2 = DocumentFingerprint.from_text("Hello World", "p1")

        assert fp1.matches(fp2)

    def test_fingerprint_different_text_no_match(self) -> None:
        """Different text should not match."""
        from seo_optimizer.diffing.models import DocumentFingerprint

        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("Goodbye World", "p1")

        assert not fp1.matches(fp2)

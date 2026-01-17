"""
Tests for content fingerprinting and move detection.

These tests ensure that moved content is correctly detected
and NOT highlighted as new content.
"""

import pytest

from seo_optimizer.diffing.fingerprint import (
    ContentFingerprinter,
    MoveDetector,
    NGramFingerprint,
    SimHashFingerprint,
    build_fingerprint_index,
)
from seo_optimizer.diffing.models import DocumentFingerprint


class TestContentFingerprinter:
    """Tests for ContentFingerprinter class."""

    def test_fingerprinter_initialization(self) -> None:
        """ContentFingerprinter should initialize with default settings."""
        fp = ContentFingerprinter()
        assert fp.simhash_bits == 64
        assert fp.ngram_size == 3

    def test_custom_initialization(self) -> None:
        """ContentFingerprinter should accept custom settings."""
        fp = ContentFingerprinter(simhash_bits=128, ngram_size=4)
        assert fp.simhash_bits == 128
        assert fp.ngram_size == 4


class TestSimHash:
    """Tests for SimHash fingerprinting."""

    def test_simhash_returns_fingerprint(self) -> None:
        """compute_simhash should return a SimHashFingerprint."""
        fp = ContentFingerprinter()
        result = fp.compute_simhash("Hello world", "p0")
        assert isinstance(result, SimHashFingerprint)
        assert len(result.hash_value) == fp.simhash_bits
        assert result.position_id == "p0"

    def test_simhash_identical_text(self) -> None:
        """Identical text should produce identical SimHash."""
        fp = ContentFingerprinter()
        hash1 = fp.compute_simhash("Hello world", "p0")
        hash2 = fp.compute_simhash("Hello world", "p1")
        assert hash1.hash_value == hash2.hash_value

    def test_simhash_similar_text(self) -> None:
        """Similar text should produce similar SimHash."""
        fp = ContentFingerprinter()
        hash1 = fp.compute_simhash("The product helps users", "p0")
        hash2 = fp.compute_simhash("The product helps customers", "p1")
        # Similar text should have high similarity
        similarity = hash1.similarity(hash2)
        assert similarity > 0.5

    def test_simhash_different_text(self) -> None:
        """Very different text should produce different SimHash."""
        fp = ContentFingerprinter()
        hash1 = fp.compute_simhash("Hello world", "p0")
        hash2 = fp.compute_simhash("Quantum physics explains the universe", "p1")
        # Different text should have low similarity
        similarity = hash1.similarity(hash2)
        assert similarity < 0.9

    def test_simhash_empty_text(self) -> None:
        """Empty text should return zero hash."""
        fp = ContentFingerprinter()
        result = fp.compute_simhash("", "p0")
        assert result.hash_value == "0" * fp.simhash_bits
        assert result.word_count == 0


class TestNGramFingerprint:
    """Tests for N-gram fingerprinting."""

    def test_ngram_returns_fingerprint(self) -> None:
        """compute_ngram_fingerprints should return an NGramFingerprint."""
        fp = ContentFingerprinter()
        result = fp.compute_ngram_fingerprints("Hello world", position_id="p0")
        assert isinstance(result, NGramFingerprint)
        assert result.n == fp.ngram_size
        assert result.position_id == "p0"

    def test_ngram_identical_text(self) -> None:
        """Identical text should have perfect Jaccard similarity."""
        fp = ContentFingerprinter()
        ng1 = fp.compute_ngram_fingerprints("Hello world", position_id="p0")
        ng2 = fp.compute_ngram_fingerprints("Hello world", position_id="p1")
        assert ng1.jaccard_similarity(ng2) == 1.0

    def test_ngram_overlap_similar_text(self) -> None:
        """Similar text should have high n-gram overlap."""
        fp = ContentFingerprinter()
        ng1 = fp.compute_ngram_fingerprints("The product helps users", position_id="p0")
        ng2 = fp.compute_ngram_fingerprints("The product helps customers", position_id="p1")
        # Similar text should have decent overlap
        overlap = ng1.jaccard_similarity(ng2)
        assert overlap > 0.3

    def test_ngram_overlap_different_text(self) -> None:
        """Different text should have low n-gram overlap."""
        fp = ContentFingerprinter()
        ng1 = fp.compute_ngram_fingerprints("Hello world", position_id="p0")
        ng2 = fp.compute_ngram_fingerprints("Quantum physics explains the universe", position_id="p1")
        overlap = ng1.jaccard_similarity(ng2)
        assert overlap < 0.5

    def test_ngram_empty_text(self) -> None:
        """Empty text should return empty n-gram set."""
        fp = ContentFingerprinter()
        result = fp.compute_ngram_fingerprints("", position_id="p0")
        assert len(result.ngram_hashes) == 0


class TestWordNGram:
    """Tests for word-level n-gram fingerprinting."""

    def test_word_ngram_returns_fingerprint(self) -> None:
        """compute_word_ngram_fingerprints should return an NGramFingerprint."""
        fp = ContentFingerprinter()
        result = fp.compute_word_ngram_fingerprints("Hello beautiful world", position_id="p0")
        assert isinstance(result, NGramFingerprint)

    def test_word_ngram_similarity(self) -> None:
        """Similar sentences should have word n-gram overlap."""
        fp = ContentFingerprinter()
        ng1 = fp.compute_word_ngram_fingerprints("The quick brown fox", position_id="p0")
        ng2 = fp.compute_word_ngram_fingerprints("The quick red fox", position_id="p1")
        # Should have some overlap
        overlap = ng1.jaccard_similarity(ng2)
        assert overlap > 0


class TestSimHashFingerprint:
    """Tests for SimHashFingerprint dataclass."""

    def test_hamming_distance_identical(self) -> None:
        """Identical hashes should have distance 0."""
        hash_val = "1010101010"
        fp1 = SimHashFingerprint(
            hash_value=hash_val,
            hash_bits=10,
            source_text="test",
            position_id="p0",
            word_count=1,
        )
        fp2 = SimHashFingerprint(
            hash_value=hash_val,
            hash_bits=10,
            source_text="test",
            position_id="p1",
            word_count=1,
        )
        assert fp1.hamming_distance(fp2) == 0

    def test_hamming_distance_one_bit(self) -> None:
        """One bit difference should have distance 1."""
        fp1 = SimHashFingerprint(
            hash_value="1010101010",
            hash_bits=10,
            source_text="test",
            position_id="p0",
            word_count=1,
        )
        fp2 = SimHashFingerprint(
            hash_value="1010101011",
            hash_bits=10,
            source_text="test",
            position_id="p1",
            word_count=1,
        )
        assert fp1.hamming_distance(fp2) == 1

    def test_similarity_calculation(self) -> None:
        """Similarity should be 1 - (distance / bits)."""
        fp1 = SimHashFingerprint(
            hash_value="1010101010",
            hash_bits=10,
            source_text="test",
            position_id="p0",
            word_count=1,
        )
        fp2 = SimHashFingerprint(
            hash_value="1010101011",
            hash_bits=10,
            source_text="test",
            position_id="p1",
            word_count=1,
        )
        # 1 bit difference in 10 bits = 0.9 similarity
        assert fp1.similarity(fp2) == 0.9


class TestNGramFingerprintDataclass:
    """Tests for NGramFingerprint dataclass."""

    def test_jaccard_similarity_identical(self) -> None:
        """Identical n-grams should have similarity 1.0."""
        ngrams = frozenset(["abc", "bcd", "cde"])
        fp1 = NGramFingerprint(ngram_hashes=ngrams, n=3, source_text="test", position_id="p0")
        fp2 = NGramFingerprint(ngram_hashes=ngrams, n=3, source_text="test", position_id="p1")
        assert fp1.jaccard_similarity(fp2) == 1.0

    def test_jaccard_similarity_empty(self) -> None:
        """Empty n-grams should have similarity 0.0."""
        fp1 = NGramFingerprint(ngram_hashes=frozenset(), n=3, source_text="", position_id="p0")
        fp2 = NGramFingerprint(ngram_hashes=frozenset(["abc"]), n=3, source_text="test", position_id="p1")
        assert fp1.jaccard_similarity(fp2) == 0.0


class TestMoveDetector:
    """Tests for MoveDetector class."""

    def test_move_detector_initialization(self) -> None:
        """MoveDetector should initialize with default thresholds."""
        detector = MoveDetector()
        assert detector.exact_match_threshold == 0.95
        assert detector.fuzzy_match_threshold == 0.70

    def test_custom_thresholds(self) -> None:
        """MoveDetector should accept custom thresholds."""
        detector = MoveDetector(exact_match_threshold=0.90, fuzzy_match_threshold=0.60)
        assert detector.exact_match_threshold == 0.90
        assert detector.fuzzy_match_threshold == 0.60


class TestDocumentFingerprint:
    """Tests for DocumentFingerprint model."""

    def test_fingerprint_from_text(self) -> None:
        """DocumentFingerprint should create from text."""
        fp = DocumentFingerprint.from_text("Hello World", "p0")
        assert fp.original_position == "p0"
        assert fp.normalized_text == "hello world"
        assert len(fp.content_hash) == 32

    def test_fingerprint_exact_match(self) -> None:
        """Identical text should produce matching fingerprints."""
        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("Hello World", "p1")
        assert fp1.matches(fp2)

    def test_fingerprint_case_insensitive_match(self) -> None:
        """Fingerprints should match case-insensitively."""
        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("HELLO WORLD", "p1")
        assert fp1.matches(fp2)

    def test_fingerprint_whitespace_normalized(self) -> None:
        """Fingerprints should normalize whitespace."""
        fp1 = DocumentFingerprint.from_text("Hello   World", "p0")
        fp2 = DocumentFingerprint.from_text("Hello World", "p1")
        assert fp1.matches(fp2)

    def test_fingerprint_different_text_no_match(self) -> None:
        """Different text should not match."""
        fp1 = DocumentFingerprint.from_text("Hello World", "p0")
        fp2 = DocumentFingerprint.from_text("Goodbye World", "p1")
        assert not fp1.matches(fp2)

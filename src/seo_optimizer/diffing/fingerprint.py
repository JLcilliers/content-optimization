"""
Content Fingerprinting - Detects moved content.

When content is relocated (moved from one section to another),
it should NOT be highlighted as new. This module uses content
fingerprinting to detect moved content.

Implements:
- SimHash for locality-sensitive hashing (similar texts produce similar hashes)
- N-gram fingerprinting for partial match detection
- Move detection across document positions

Per Content_Ingestion research: "Calculate a locally sensitive hash (SimHash)
or N-gram fingerprint for every text block"

Reference: docs/research/05-diffing-highlighting.md section 6
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from seo_optimizer.diffing.chunker import ContentChunk, normalize_text
from seo_optimizer.diffing.models import DocumentFingerprint

if TYPE_CHECKING:
    from seo_optimizer.ingestion.models import DocumentAST, OriginalSnapshot


@dataclass
class SimHashFingerprint:
    """
    SimHash-based fingerprint for locality-sensitive hashing.

    SimHash produces similar hashes for similar content, making it
    useful for detecting moved content with minor modifications.
    """

    # The SimHash value (binary string representation)
    hash_value: str

    # Number of bits in the hash
    hash_bits: int

    # Original text that was hashed
    source_text: str

    # Position identifier
    position_id: str

    # Word count for quick filtering
    word_count: int

    def hamming_distance(self, other: SimHashFingerprint) -> int:
        """
        Calculate Hamming distance between two SimHash values.

        Lower distance = more similar content.

        Args:
            other: Another SimHash fingerprint

        Returns:
            Number of differing bits
        """
        if len(self.hash_value) != len(other.hash_value):
            return max(len(self.hash_value), len(other.hash_value))

        return sum(a != b for a, b in zip(self.hash_value, other.hash_value, strict=False))

    def similarity(self, other: SimHashFingerprint) -> float:
        """
        Calculate similarity as a ratio (0.0 to 1.0).

        Args:
            other: Another SimHash fingerprint

        Returns:
            Similarity ratio (1.0 = identical, 0.0 = completely different)
        """
        distance = self.hamming_distance(other)
        return 1.0 - (distance / self.hash_bits)


@dataclass
class NGramFingerprint:
    """
    N-gram based fingerprint for partial matching.

    Stores a set of n-grams that can be used to calculate
    Jaccard similarity for partial match detection.
    """

    # Set of n-gram hashes
    ngram_hashes: frozenset[str]

    # N-gram size
    n: int

    # Original text
    source_text: str

    # Position identifier
    position_id: str

    # Total n-gram count
    ngram_count: int = field(default=0)

    def __post_init__(self) -> None:
        """Calculate n-gram count."""
        object.__setattr__(self, "ngram_count", len(self.ngram_hashes))

    def jaccard_similarity(self, other: NGramFingerprint) -> float:
        """
        Calculate Jaccard similarity between n-gram sets.

        Args:
            other: Another n-gram fingerprint

        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        if not self.ngram_hashes or not other.ngram_hashes:
            return 0.0

        intersection = len(self.ngram_hashes & other.ngram_hashes)
        union = len(self.ngram_hashes | other.ngram_hashes)

        if union == 0:
            return 0.0

        return intersection / union


class ContentFingerprinter:
    """
    Creates and compares content fingerprints.

    Per research: "SimHash deduplication" and "N-gram fingerprinting
    for partial matches"
    """

    def __init__(
        self,
        simhash_bits: int = 64,
        ngram_size: int = 3,
    ) -> None:
        """
        Initialize fingerprinter.

        Args:
            simhash_bits: Number of bits for SimHash (default 64)
            ngram_size: Size of n-grams (default 3 = trigrams)
        """
        self.simhash_bits = simhash_bits
        self.ngram_size = ngram_size

    def compute_simhash(self, text: str, position_id: str = "") -> SimHashFingerprint:
        """
        Compute SimHash for text content.

        SimHash is a locality-sensitive hash that produces similar
        hashes for similar content. This is key for detecting moved
        content with minor modifications.

        Algorithm:
        1. Tokenize text into features (words)
        2. Hash each feature
        3. Create weighted vector
        4. Collapse to binary hash

        Args:
            text: Text to fingerprint
            position_id: Position identifier

        Returns:
            SimHash fingerprint
        """
        normalized = normalize_text(text)
        words = normalized.split()

        if not words:
            return SimHashFingerprint(
                hash_value="0" * self.simhash_bits,
                hash_bits=self.simhash_bits,
                source_text=text,
                position_id=position_id,
                word_count=0,
            )

        # Initialize bit vectors
        bit_counts = [0] * self.simhash_bits

        # Process each word
        for word in words:
            # Hash the word to get a bit pattern
            word_hash = self._hash_to_bits(word)

            # Add/subtract based on bit values
            for i, bit in enumerate(word_hash):
                if bit == "1":
                    bit_counts[i] += 1
                else:
                    bit_counts[i] -= 1

        # Convert to binary string
        hash_value = "".join("1" if count > 0 else "0" for count in bit_counts)

        return SimHashFingerprint(
            hash_value=hash_value,
            hash_bits=self.simhash_bits,
            source_text=text,
            position_id=position_id,
            word_count=len(words),
        )

    def compute_ngram_fingerprints(
        self,
        text: str,
        n: int | None = None,
        position_id: str = "",
    ) -> NGramFingerprint:
        """
        Generate n-gram fingerprints for partial matching.

        Per research: "If a block's fingerprint appears across...
        use N-gram fingerprint for every text block"

        Args:
            text: Text to fingerprint
            n: N-gram size (default: self.ngram_size)
            position_id: Position identifier

        Returns:
            N-gram fingerprint with hash set
        """
        n = n or self.ngram_size
        normalized = normalize_text(text)

        # Generate character n-grams
        ngrams: set[str] = set()
        for i in range(len(normalized) - n + 1):
            ngram = normalized[i : i + n]
            # Hash the n-gram for compact storage
            ngram_hash = hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest()[:8]
            ngrams.add(ngram_hash)

        return NGramFingerprint(
            ngram_hashes=frozenset(ngrams),
            n=n,
            source_text=text,
            position_id=position_id,
        )

    def compute_word_ngram_fingerprints(
        self,
        text: str,
        n: int = 2,
        position_id: str = "",
    ) -> NGramFingerprint:
        """
        Generate word-level n-gram fingerprints.

        Uses word sequences instead of character sequences,
        better for semantic chunking.

        Args:
            text: Text to fingerprint
            n: Number of words per n-gram
            position_id: Position identifier

        Returns:
            Word n-gram fingerprint
        """
        normalized = normalize_text(text)
        words = normalized.split()

        ngrams: set[str] = set()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngram_hash = hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest()[:8]
            ngrams.add(ngram_hash)

        return NGramFingerprint(
            ngram_hashes=frozenset(ngrams),
            n=n,
            source_text=text,
            position_id=position_id,
        )

    def calculate_ngram_overlap(
        self,
        fp_a: NGramFingerprint,
        fp_b: NGramFingerprint,
    ) -> float:
        """
        Calculate overlap ratio between two n-gram fingerprints.

        Per research: "N-gram overlap > 70% → Likely same content"

        Args:
            fp_a: First fingerprint
            fp_b: Second fingerprint

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        return fp_a.jaccard_similarity(fp_b)

    def _hash_to_bits(self, text: str) -> str:
        """
        Hash text to a bit string of simhash_bits length.

        Args:
            text: Text to hash

        Returns:
            Binary string representation
        """
        # Use MD5 for consistent hashing (not for security)
        hash_bytes = hashlib.md5(text.encode(), usedforsecurity=False).digest()

        # Convert to binary and pad/truncate to desired length
        hash_int = int.from_bytes(hash_bytes, "big")
        binary = bin(hash_int)[2:].zfill(128)  # MD5 is 128 bits

        # Take first simhash_bits
        return binary[: self.simhash_bits]


class MoveDetector:
    """
    Detects moved content between original and optimized documents.

    Uses both exact hash matching and fuzzy matching to identify
    content that was relocated rather than newly created.
    """

    def __init__(
        self,
        exact_match_threshold: float = 0.95,
        fuzzy_match_threshold: float = 0.70,
    ) -> None:
        """
        Initialize move detector.

        Args:
            exact_match_threshold: Threshold for exact/near-exact matches
            fuzzy_match_threshold: Threshold for fuzzy matches (n-gram overlap)
        """
        self.exact_match_threshold = exact_match_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.fingerprinter = ContentFingerprinter()

    def build_fingerprint_index(
        self,
        snapshot: OriginalSnapshot,
    ) -> dict[str, DocumentFingerprint]:
        """
        Build an index of content fingerprints from the original document.

        Creates fingerprints for all content blocks that can be used
        to detect if content was moved.

        Args:
            snapshot: The original document snapshot

        Returns:
            Dictionary mapping content_hash to DocumentFingerprint
        """
        index: dict[str, DocumentFingerprint] = {}

        for position_id, text in snapshot.text_by_position:
            if not text.strip():
                continue

            fp = DocumentFingerprint.from_text(text, position_id)
            index[fp.content_hash] = fp

        return index

    def build_simhash_index(
        self,
        chunks: Iterable[ContentChunk],
    ) -> dict[str, SimHashFingerprint]:
        """
        Build SimHash index from content chunks.

        Args:
            chunks: Content chunks to index

        Returns:
            Dictionary mapping chunk_id to SimHash fingerprint
        """
        index: dict[str, SimHashFingerprint] = {}

        for chunk in chunks:
            fp = self.fingerprinter.compute_simhash(chunk.text, chunk.chunk_id)
            index[chunk.chunk_id] = fp

        return index

    def build_ngram_index(
        self,
        chunks: Iterable[ContentChunk],
    ) -> dict[str, NGramFingerprint]:
        """
        Build n-gram index from content chunks.

        Args:
            chunks: Content chunks to index

        Returns:
            Dictionary mapping chunk_id to n-gram fingerprint
        """
        index: dict[str, NGramFingerprint] = {}

        for chunk in chunks:
            fp = self.fingerprinter.compute_ngram_fingerprints(chunk.text, position_id=chunk.chunk_id)
            index[chunk.chunk_id] = fp

        return index

    def detect_moved_content(
        self,
        candidate_text: str,
        candidate_position: str,
        fingerprint_index: dict[str, DocumentFingerprint],
        threshold: float | None = None,
    ) -> tuple[bool, str | None]:
        """
        Detect if content was moved from another location.

        Checks if the candidate text matches any fingerprint in the
        original document at a different position.

        Args:
            candidate_text: The text being evaluated
            candidate_position: Position of the candidate in optimized document
            fingerprint_index: Index of original content fingerprints
            threshold: Similarity threshold for match (default: exact_match_threshold)

        Returns:
            Tuple of:
            - is_moved: True if content was moved
            - original_position: Position where content originally appeared, or None
        """
        threshold = threshold if threshold is not None else self.exact_match_threshold

        # Create fingerprint for candidate
        candidate_fp = DocumentFingerprint.from_text(candidate_text, candidate_position)

        # Check for exact hash match first
        if candidate_fp.content_hash in fingerprint_index:
            original_fp = fingerprint_index[candidate_fp.content_hash]
            if original_fp.original_position != candidate_position:
                return True, original_fp.original_position

        # Check for fuzzy matches
        for _content_hash, original_fp in fingerprint_index.items():
            if original_fp.original_position == candidate_position:
                continue

            if candidate_fp.matches(original_fp, threshold):
                return True, original_fp.original_position

        return False, None

    def detect_moved_chunk(
        self,
        candidate_chunk: ContentChunk,
        original_simhash_index: dict[str, SimHashFingerprint],
        original_ngram_index: dict[str, NGramFingerprint],
    ) -> tuple[bool, str | None, float]:
        """
        Detect if a chunk was moved from the original document.

        Uses both SimHash (for near-exact) and n-gram (for partial) matching.

        Args:
            candidate_chunk: The chunk to check
            original_simhash_index: SimHash index of original chunks
            original_ngram_index: N-gram index of original chunks

        Returns:
            Tuple of:
            - is_moved: True if content was moved
            - original_chunk_id: ID of matching original chunk, or None
            - similarity: Similarity score of the match
        """
        # Compute fingerprints for candidate
        candidate_simhash = self.fingerprinter.compute_simhash(
            candidate_chunk.text, candidate_chunk.chunk_id
        )
        candidate_ngram = self.fingerprinter.compute_ngram_fingerprints(
            candidate_chunk.text, position_id=candidate_chunk.chunk_id
        )

        best_match_id: str | None = None
        best_similarity = 0.0

        # Check SimHash matches (exact/near-exact)
        for orig_id, orig_simhash in original_simhash_index.items():
            # Skip if word counts are very different
            if orig_simhash.word_count > 0:
                word_ratio = min(candidate_simhash.word_count, orig_simhash.word_count) / max(
                    candidate_simhash.word_count, orig_simhash.word_count
                )
                if word_ratio < 0.7:
                    continue

            similarity = candidate_simhash.similarity(orig_simhash)
            if similarity >= self.exact_match_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = orig_id

        # If no SimHash match, try n-gram matching
        if best_match_id is None:
            for orig_id, orig_ngram in original_ngram_index.items():
                similarity = candidate_ngram.jaccard_similarity(orig_ngram)
                if similarity >= self.fuzzy_match_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = orig_id

        is_moved = best_match_id is not None
        return is_moved, best_match_id, best_similarity

    def find_all_moved_content(
        self,
        optimized: DocumentAST,
        fingerprint_index: dict[str, DocumentFingerprint],
        threshold: float | None = None,
    ) -> dict[str, str]:
        """
        Find all content in the optimized document that was moved.

        Scans the entire optimized document and identifies all content
        blocks that match original fingerprints at different positions.

        Args:
            optimized: The optimized document AST
            fingerprint_index: Index of original content fingerprints
            threshold: Similarity threshold for match

        Returns:
            Dictionary mapping optimized_position -> original_position
            for all moved content
        """
        threshold = threshold if threshold is not None else self.exact_match_threshold
        moved_content: dict[str, str] = {}

        for node in optimized.nodes:
            is_moved, original_pos = self.detect_moved_content(
                node.text_content,
                node.position.position_id,
                fingerprint_index,
                threshold,
            )

            if is_moved and original_pos is not None:
                moved_content[node.position.position_id] = original_pos

        return moved_content


def build_fingerprint_index(
    snapshot: OriginalSnapshot,
) -> dict[str, DocumentFingerprint]:
    """
    Build an index of content fingerprints from the original document.

    Convenience function wrapping MoveDetector.build_fingerprint_index.

    Args:
        snapshot: The original document snapshot

    Returns:
        Dictionary mapping content_hash to DocumentFingerprint
    """
    detector = MoveDetector()
    return detector.build_fingerprint_index(snapshot)


def detect_moved_content(
    candidate_text: str,
    candidate_position: str,
    fingerprint_index: dict[str, DocumentFingerprint],
    threshold: float = 0.95,
) -> tuple[bool, str | None]:
    """
    Detect if content was moved from another location.

    Convenience function wrapping MoveDetector.detect_moved_content.

    Args:
        candidate_text: The text being evaluated
        candidate_position: Position of the candidate in optimized document
        fingerprint_index: Index of original content fingerprints
        threshold: Similarity threshold for match (default 0.95)

    Returns:
        Tuple of:
        - is_moved: True if content was moved
        - original_position: Position where content originally appeared, or None
    """
    detector = MoveDetector(exact_match_threshold=threshold)
    return detector.detect_moved_content(
        candidate_text, candidate_position, fingerprint_index, threshold
    )


def find_all_moved_content(
    optimized: DocumentAST,
    fingerprint_index: dict[str, DocumentFingerprint],
    threshold: float = 0.95,
) -> dict[str, str]:
    """
    Find all content in the optimized document that was moved.

    Convenience function wrapping MoveDetector.find_all_moved_content.

    Args:
        optimized: The optimized document AST
        fingerprint_index: Index of original content fingerprints
        threshold: Similarity threshold for match

    Returns:
        Dictionary mapping optimized_position -> original_position
        for all moved content
    """
    detector = MoveDetector(exact_match_threshold=threshold)
    return detector.find_all_moved_content(optimized, fingerprint_index, threshold)

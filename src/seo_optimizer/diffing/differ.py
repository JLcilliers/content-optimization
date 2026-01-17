"""
Core Diff Algorithm - Main diffing logic for change detection.

This is the heart of the diffing system. It compares original content
to optimized content and determines what is genuinely NEW.

CRITICAL DESIGN PRINCIPLES:
1. Zero false positives: NEVER highlight existing content
2. When uncertain, DON'T highlight (conservative approach)
3. Semantic awareness: Rewording != new content
4. Move detection: Relocated content != new content

Implements the 3-Layer Hybrid Diff Algorithm:
- Layer 1: Fingerprint Matching (Fast, Exact) - SimHash + N-gram
- Layer 2: Fuzzy String Matching (Medium, Structural) - Levenshtein
- Layer 3: Semantic Embedding Matching (Slow, Semantic) - sentence-transformers

Reference: docs/research/05-diffing-highlighting.md
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from Levenshtein import ratio as levenshtein_ratio

from seo_optimizer.diffing.chunker import (
    ContentChunk,
    DocumentChunker,
    normalize_text,
)
from seo_optimizer.diffing.fingerprint import (
    ContentFingerprinter,
    MoveDetector,
    NGramFingerprint,
    SimHashFingerprint,
)
from seo_optimizer.diffing.models import (
    Addition,
    ChangeSet,
    HighlightRegion,
)
from seo_optimizer.diffing.semantic import SemanticMatcher

if TYPE_CHECKING:
    from seo_optimizer.ingestion.models import DocumentAST, OriginalSnapshot


class MatchType(str, Enum):
    """Type of match found between content."""

    EXACT = "exact"  # Exact fingerprint match
    SIMHASH = "simhash"  # SimHash match (near-exact)
    NGRAM = "ngram"  # N-gram overlap match
    FUZZY = "fuzzy"  # Levenshtein ratio match
    SEMANTIC = "semantic"  # Semantic embedding match
    NO_MATCH = "no_match"  # No match found


@dataclass
class MatchResult:
    """Result of attempting to match content against original."""

    match_type: MatchType
    similarity: float
    matched_chunk_id: str | None = None
    matched_text: str | None = None
    confidence: float = 1.0
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    @property
    def is_match(self) -> bool:
        """Whether this represents a successful match."""
        return self.match_type != MatchType.NO_MATCH


@dataclass
class DiffConfig:
    """Configuration with research-derived defaults."""

    # Semantic thresholds (from Content_Scoring research)
    semantic_equivalence_threshold: float = 0.85
    redundancy_threshold: float = 0.90

    # Fuzzy matching thresholds
    levenshtein_same_threshold: float = 0.90
    levenshtein_partial_threshold: float = 0.70

    # Fingerprint thresholds
    simhash_threshold: float = 0.95
    ngram_overlap_threshold: float = 0.70

    # Conservative mode
    conservative_mode: bool = True
    conservative_similarity_floor: float = 0.80

    # Performance settings
    enable_semantic_matching: bool = True
    max_chunks_for_semantic: int = 100


class ContentDiffer:
    """
    3-layer hybrid diff algorithm per research specifications.

    Priority: 0% false positives > catching all new content.

    The algorithm processes content through 3 layers:
    1. Fingerprint matching (fast, exact/near-exact detection)
    2. Fuzzy string matching (medium, structural similarity)
    3. Semantic matching (slow, meaning-based comparison)

    Only content that fails ALL layers is marked as genuinely new.
    """

    def __init__(self, config: DiffConfig | None = None) -> None:
        """
        Initialize the differ with configuration.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or DiffConfig()
        self.fingerprinter = ContentFingerprinter()
        self.semantic_matcher = SemanticMatcher(
            equivalence_threshold=self.config.semantic_equivalence_threshold,
            redundancy_threshold=self.config.redundancy_threshold,
        )
        self.chunker = DocumentChunker()
        self.move_detector = MoveDetector(
            exact_match_threshold=self.config.simhash_threshold,
            fuzzy_match_threshold=self.config.ngram_overlap_threshold,
        )

    def diff(
        self,
        original: OriginalSnapshot,
        optimized: DocumentAST,
    ) -> ChangeSet:
        """
        Main entry point: compute diff between original and optimized documents.

        Algorithm:
        1. Chunk both documents (heading-aligned)
        2. Build fingerprint index from original
        3. For each optimized chunk:
           a. Layer 1: Check fingerprint match (moved content)
           b. Layer 2: Check fuzzy string match (minor edits)
           c. Layer 3: Check semantic similarity (paraphrases)
        4. Only mark as Addition if fails ALL layers
        5. Compute precise character boundaries

        Args:
            original: Immutable snapshot of original document
            optimized: The optimized document AST

        Returns:
            ChangeSet containing all additions to highlight

        CRITICAL: The returned ChangeSet must have zero false positives.
                  When uncertain, the change should NOT be included.
        """
        # Extract text from original snapshot
        original_texts = [text for _, text in original.text_by_position if text.strip()]
        original_full_text = "\n".join(original_texts)

        # Chunk both documents
        original_chunks = self.chunker.chunk_from_text(original_full_text, doc_id="original")
        optimized_chunks = self.chunker.chunk_document(optimized)

        # Build fingerprint indices from original
        simhash_index = self.move_detector.build_simhash_index(original_chunks)
        ngram_index = self.move_detector.build_ngram_index(original_chunks)

        # Extract original chunk texts for semantic matching
        original_chunk_texts = [chunk.text for chunk in original_chunks]

        # Process each optimized chunk
        additions: list[Addition] = []

        for opt_chunk in optimized_chunks:
            # Run through 3-layer matching
            match_result = self._match_chunk(
                opt_chunk,
                original_chunks,
                simhash_index,
                ngram_index,
                original_chunk_texts,
            )

            if not match_result.is_match:
                # This is genuinely new content
                addition = self._create_addition(opt_chunk, match_result)
                additions.append(addition)

        # Generate changeset
        # Note: review_required is calculated in ChangeSet.__post_init__
        # based on additions with confidence < 0.7
        changeset = ChangeSet(
            changeset_id=f"cs_{uuid.uuid4().hex[:8]}",
            original_doc_id=original.doc_id,
            optimized_doc_id=optimized.doc_id,
            additions=additions,
        )

        return changeset

    def _match_chunk(
        self,
        chunk: ContentChunk,
        original_chunks: list[ContentChunk],
        simhash_index: dict[str, SimHashFingerprint],
        ngram_index: dict[str, NGramFingerprint],
        original_texts: list[str],
    ) -> MatchResult:
        """
        Run chunk through 3-layer matching algorithm.

        Args:
            chunk: The optimized chunk to match
            original_chunks: Original document chunks
            simhash_index: SimHash fingerprint index
            ngram_index: N-gram fingerprint index
            original_texts: Original chunk texts for semantic matching

        Returns:
            MatchResult indicating if/how content matched
        """
        # Layer 1: Fingerprint matching
        layer1_result = self._layer1_fingerprint_match(chunk, simhash_index, ngram_index)
        if layer1_result.is_match:
            return layer1_result

        # Layer 2: Fuzzy string matching
        layer2_result = self._layer2_fuzzy_match(chunk, original_chunks)
        if layer2_result.is_match and layer2_result.similarity >= self.config.levenshtein_same_threshold:
            return layer2_result

        # Layer 3: Semantic matching (if enabled and worthwhile)
        if self.config.enable_semantic_matching and len(original_texts) <= self.config.max_chunks_for_semantic:
            layer3_result = self._layer3_semantic_match(chunk, original_texts)
            if layer3_result.is_match:
                return layer3_result

            # Conservative mode: if Layer 2 partial match + Layer 3 similarity is borderline
            if (
                self.config.conservative_mode
                and self._apply_conservative_mode([layer2_result, layer3_result])
            ):
                return MatchResult(
                    match_type=MatchType.SEMANTIC,
                    similarity=max(layer2_result.similarity, layer3_result.similarity),
                    confidence=0.9,
                    metadata={"conservative_match": True},
                )

        # No match found
        return MatchResult(
            match_type=MatchType.NO_MATCH,
            similarity=0.0,
        )

    def _layer1_fingerprint_match(
        self,
        chunk: ContentChunk,
        simhash_index: dict[str, SimHashFingerprint],
        ngram_index: dict[str, NGramFingerprint],
    ) -> MatchResult:
        """
        Layer 1: SimHash + N-gram fingerprint matching.

        Per research: "SimHash exists in original → MOVED content (don't highlight)"

        Args:
            chunk: Chunk to check
            simhash_index: SimHash index from original
            ngram_index: N-gram index from original

        Returns:
            MatchResult with match_type and confidence
        """
        # Compute fingerprints for candidate
        candidate_simhash = self.fingerprinter.compute_simhash(chunk.text, chunk.chunk_id)
        candidate_ngram = self.fingerprinter.compute_ngram_fingerprints(
            chunk.text, position_id=chunk.chunk_id
        )

        # Check SimHash matches (near-exact)
        best_simhash_id: str | None = None
        best_simhash_similarity = 0.0

        for orig_id, orig_simhash in simhash_index.items():
            # Skip if word counts are very different
            if orig_simhash.word_count > 0 and candidate_simhash.word_count > 0:
                word_ratio = min(candidate_simhash.word_count, orig_simhash.word_count) / max(
                    candidate_simhash.word_count, orig_simhash.word_count
                )
                if word_ratio < 0.5:
                    continue

            similarity = candidate_simhash.similarity(orig_simhash)
            if similarity >= self.config.simhash_threshold and similarity > best_simhash_similarity:
                best_simhash_similarity = similarity
                best_simhash_id = orig_id

        if best_simhash_id is not None:
            return MatchResult(
                match_type=MatchType.SIMHASH,
                similarity=best_simhash_similarity,
                matched_chunk_id=best_simhash_id,
                confidence=best_simhash_similarity,
                metadata={"layer": 1, "method": "simhash"},
            )

        # Check N-gram overlap
        best_ngram_id: str | None = None
        best_ngram_similarity = 0.0

        for orig_id, orig_ngram in ngram_index.items():
            similarity = candidate_ngram.jaccard_similarity(orig_ngram)
            if similarity >= self.config.ngram_overlap_threshold and similarity > best_ngram_similarity:
                best_ngram_similarity = similarity
                best_ngram_id = orig_id

        if best_ngram_id is not None:
            return MatchResult(
                match_type=MatchType.NGRAM,
                similarity=best_ngram_similarity,
                matched_chunk_id=best_ngram_id,
                confidence=best_ngram_similarity,
                metadata={"layer": 1, "method": "ngram"},
            )

        return MatchResult(
            match_type=MatchType.NO_MATCH,
            similarity=max(best_simhash_similarity, best_ngram_similarity),
        )

    def _layer2_fuzzy_match(
        self,
        chunk: ContentChunk,
        original_chunks: list[ContentChunk],
    ) -> MatchResult:
        """
        Layer 2: Levenshtein ratio matching.

        Thresholds:
        - >= 0.90 = same content (minor edits)
        - 0.70-0.90 = partial match (needs semantic check)
        - < 0.70 = likely different

        Args:
            chunk: Chunk to check
            original_chunks: Original document chunks

        Returns:
            MatchResult
        """
        best_match_id: str | None = None
        best_match_text: str | None = None
        best_similarity = 0.0

        # Normalize candidate text
        candidate_normalized = normalize_text(chunk.text)

        for orig_chunk in original_chunks:
            orig_normalized = normalize_text(orig_chunk.text)

            # Quick length check
            len_ratio = min(len(candidate_normalized), len(orig_normalized)) / max(
                len(candidate_normalized), len(orig_normalized), 1
            )
            if len_ratio < 0.5:
                continue

            # Levenshtein ratio
            similarity = levenshtein_ratio(candidate_normalized, orig_normalized)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = orig_chunk.chunk_id
                best_match_text = orig_chunk.text

        # Determine match type based on threshold
        if best_similarity >= self.config.levenshtein_same_threshold:
            return MatchResult(
                match_type=MatchType.FUZZY,
                similarity=best_similarity,
                matched_chunk_id=best_match_id,
                matched_text=best_match_text,
                confidence=best_similarity,
                metadata={"layer": 2, "threshold_met": "same"},
            )
        elif best_similarity >= self.config.levenshtein_partial_threshold:
            return MatchResult(
                match_type=MatchType.FUZZY,
                similarity=best_similarity,
                matched_chunk_id=best_match_id,
                matched_text=best_match_text,
                confidence=best_similarity * 0.9,  # Slightly lower confidence for partial
                metadata={"layer": 2, "threshold_met": "partial"},
            )

        return MatchResult(
            match_type=MatchType.NO_MATCH,
            similarity=best_similarity,
        )

    def _layer3_semantic_match(
        self,
        chunk: ContentChunk,
        original_texts: list[str],
    ) -> MatchResult:
        """
        Layer 3: Embedding similarity.

        Per research: >= 0.85 = semantically equivalent

        Args:
            chunk: Chunk to check
            original_texts: Original chunk texts

        Returns:
            MatchResult
        """
        # Find best semantic match
        best_idx, best_similarity = self.semantic_matcher.find_best_match(
            chunk.text, original_texts
        )

        if best_similarity >= self.config.semantic_equivalence_threshold:
            return MatchResult(
                match_type=MatchType.SEMANTIC,
                similarity=best_similarity,
                matched_text=original_texts[best_idx] if best_idx >= 0 else None,
                confidence=best_similarity,
                metadata={"layer": 3},
            )

        return MatchResult(
            match_type=MatchType.NO_MATCH,
            similarity=best_similarity,
        )

    def _apply_conservative_mode(
        self,
        match_results: list[MatchResult],
    ) -> bool:
        """
        When uncertain, DON'T highlight.

        If any layer shows similarity >= 0.80, treat as existing content.

        Args:
            match_results: Results from various matching layers

        Returns:
            True if content should be treated as existing (don't highlight)
        """
        for result in match_results:
            if result.similarity >= self.config.conservative_similarity_floor:
                return True
        return False

    def _create_addition(
        self,
        chunk: ContentChunk,
        match_result: MatchResult,
    ) -> Addition:
        """
        Create an Addition object from a chunk that didn't match.

        Args:
            chunk: The new content chunk
            match_result: The matching result (should be NO_MATCH)

        Returns:
            Addition object for the changeset
        """
        # Calculate confidence based on how "different" the content is
        # Lower similarity to original = higher confidence it's new
        confidence = 1.0 - match_result.similarity

        # Create highlight region
        region = HighlightRegion(
            node_id=chunk.source_node_ids[0] if chunk.source_node_ids else chunk.chunk_id,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.text,
            confidence=confidence,
            reason="new_content",
        )

        return Addition(
            addition_id=f"add_{uuid.uuid4().hex[:8]}",
            node_ids=chunk.source_node_ids,
            highlight_regions=[region],
            total_text=chunk.text,
            confidence=confidence,
            addition_type="new",
            metadata={
                "chunk_id": chunk.chunk_id,
                "best_match_similarity": match_result.similarity,
            },
        )


def compute_diff(
    original: OriginalSnapshot,
    optimized: DocumentAST,
    *,
    semantic_threshold: float = 0.85,
    move_detection_threshold: float = 0.95,
    conservative_mode: bool = True,
) -> ChangeSet:
    """
    Compute the diff between original and optimized documents.

    This is the main entry point for the diffing system. It identifies
    all genuinely NEW content that should be highlighted green.

    Args:
        original: Immutable snapshot of original document
        optimized: The optimized document AST
        semantic_threshold: Similarity threshold for semantic matching (default 0.85)
        move_detection_threshold: Threshold for move detection (default 0.95)
        conservative_mode: If True, don't highlight low-confidence changes

    Returns:
        ChangeSet containing all additions to highlight

    CRITICAL: The returned ChangeSet must have zero false positives.
              When uncertain, the change should NOT be included.

    Example:
        >>> original_snapshot = create_snapshot(parse_docx("original.docx"))
        >>> optimized_ast = optimize(original_snapshot)
        >>> changes = compute_diff(original_snapshot, optimized_ast)
        >>> print(f"Found {len(changes.additions)} additions")
    """
    config = DiffConfig(
        semantic_equivalence_threshold=semantic_threshold,
        simhash_threshold=move_detection_threshold,
        conservative_mode=conservative_mode,
    )
    differ = ContentDiffer(config)
    return differ.diff(original, optimized)


def diff_text_segments(
    original_text: str,
    modified_text: str,
    *,
    semantic_threshold: float = 0.85,
) -> tuple[list[tuple[int, int, str]], float]:
    """
    Diff two text segments and identify new portions.

    Uses a hybrid algorithm:
    1. Character-level diff (difflib) for structural changes
    2. Semantic similarity (sentence-transformers) for rewording detection
    3. Levenshtein distance for precise boundary calculation

    Args:
        original_text: The original text content
        modified_text: The modified text content
        semantic_threshold: Similarity above which text is considered "same"

    Returns:
        Tuple of:
        - List of (start, end, text) for new content portions
        - Confidence score (0-1)

    Example:
        >>> new_portions, confidence = diff_text_segments(
        ...     "The product helps users.",
        ...     "The product helps users save time and money."
        ... )
        >>> # new_portions = [(24, 44, " save time and money")]
    """
    # Handle edge cases
    if not original_text.strip():
        # Original is empty, everything is new
        return [(0, len(modified_text), modified_text)], 1.0

    if not modified_text.strip():
        # Modified is empty, nothing to highlight
        return [], 1.0

    # Check for exact match
    if normalize_text(original_text) == normalize_text(modified_text):
        return [], 1.0

    # Check semantic similarity
    matcher = SemanticMatcher(equivalence_threshold=semantic_threshold)
    similarity = matcher.compute_similarity(original_text, modified_text)

    if similarity >= semantic_threshold:
        # Semantically equivalent - no new content
        return [], similarity

    # Use Levenshtein to find differences
    from Levenshtein import editops

    ops = editops(original_text, modified_text)

    # Group insertions into contiguous regions
    additions: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_end: int | None = None

    for op_type, _orig_pos, mod_pos in ops:
        if op_type == "insert":
            if current_start is None:
                current_start = mod_pos
                current_end = mod_pos + 1
            elif mod_pos == current_end:
                current_end = mod_pos + 1
            else:
                # Save current region and start new one
                if current_start is not None and current_end is not None:
                    text = modified_text[current_start:current_end]
                    additions.append((current_start, current_end, text))
                current_start = mod_pos
                current_end = mod_pos + 1
        else:
            # Not an insertion, close current region if exists
            if current_start is not None and current_end is not None:
                text = modified_text[current_start:current_end]
                additions.append((current_start, current_end, text))
                current_start = None
                current_end = None

    # Close final region
    if current_start is not None and current_end is not None:
        text = modified_text[current_start:current_end]
        additions.append((current_start, current_end, text))

    # Calculate confidence based on dissimilarity
    confidence = 1.0 - similarity

    return additions, confidence


def identify_expansion(
    original_text: str,
    modified_text: str,
) -> tuple[int, int] | None:
    """
    Identify if modified text is an expansion of original.

    An expansion is when the original text is preserved but additional
    content is appended or inserted.

    Args:
        original_text: The original text
        modified_text: The potentially expanded text

    Returns:
        (start, end) positions of the expansion, or None if not an expansion

    Example:
        >>> expansion = identify_expansion(
        ...     "The product helps users",
        ...     "The product helps users save time"
        ... )
        >>> # expansion = (23, 33)  # " save time" is the expansion
    """
    # Normalize for comparison
    original_normalized = normalize_text(original_text)
    modified_normalized = normalize_text(modified_text)

    # Check if modified contains original or starts with it
    if (
        original_normalized not in modified_normalized
        and not modified_normalized.startswith(original_normalized)
    ):
        return None

    # Find where original ends in modified
    original_pos = modified_text.lower().find(original_text.lower())
    if original_pos == -1:
        # Try with normalized text
        original_pos = 0

    # The expansion starts after the original content
    start = original_pos + len(original_text)
    end = len(modified_text)

    if start >= end:
        return None

    return start, end


def calculate_boundaries(
    original_text: str,
    modified_text: str,
    new_content_indices: list[tuple[int, int]],
    *,
    use_word_boundaries: bool = True,
) -> list[tuple[int, int]]:
    """
    Refine highlight boundaries to word or character level.

    Takes rough indices from the diff algorithm and refines them
    to clean boundaries suitable for DOCX highlighting.

    Args:
        original_text: Original text for reference
        modified_text: Modified text containing new content
        new_content_indices: Rough (start, end) indices from diff
        use_word_boundaries: If True, snap to word boundaries

    Returns:
        Refined (start, end) indices for highlighting

    Example:
        >>> boundaries = calculate_boundaries(
        ...     "Hello world",
        ...     "Hello beautiful world",
        ...     [(6, 16)],
        ...     use_word_boundaries=True
        ... )
        >>> # boundaries = [(6, 16)]  # "beautiful " cleanly bounded
    """
    refined: list[tuple[int, int]] = []

    for start, end in new_content_indices:
        if not use_word_boundaries:
            refined.append((start, end))
            continue

        # Adjust to word boundaries
        adjusted_start = start
        adjusted_end = end

        # Expand start backward to word boundary
        while adjusted_start > 0 and not modified_text[adjusted_start - 1].isspace():
            adjusted_start -= 1

        # Expand end forward to word boundary
        while adjusted_end < len(modified_text) and not modified_text[adjusted_end - 1].isspace():
            if adjusted_end < len(modified_text) and modified_text[adjusted_end].isspace():
                break
            adjusted_end += 1

        # Trim leading whitespace from highlight (but keep trailing)
        while adjusted_start < adjusted_end and modified_text[adjusted_start].isspace():
            adjusted_start += 1

        if adjusted_start < adjusted_end:
            refined.append((adjusted_start, adjusted_end))

    return refined

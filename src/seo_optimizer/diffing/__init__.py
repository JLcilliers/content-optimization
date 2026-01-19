"""
Diffing Module - CRITICAL: Change Detection and Highlighting

This is the most critical module in the entire system. The tool's value
proposition depends on accurate change detection:

    - ZERO false positives: Never highlight existing content
    - ZERO false negatives: Never miss genuinely new content

Key capabilities:
- Character-level diffing for precise boundaries
- Semantic similarity detection (rewording != new content)
- Move detection (relocated content != new content)
- Confidence scoring for human review triggers

Reference: docs/research/05-diffing-highlighting.md

Implements the 3-Layer Hybrid Diff Algorithm:
- Layer 1: Fingerprint Matching (Fast, Exact) - SimHash + N-gram
- Layer 2: Fuzzy String Matching (Medium, Structural) - Levenshtein
- Layer 3: Semantic Embedding Matching (Slow, Semantic) - sentence-transformers
"""

from seo_optimizer.diffing.chunker import (
    ContentChunk,
    DocumentChunker,
    extract_chunks_from_text,
    normalize_text,
)
from seo_optimizer.diffing.differ import (
    ContentDiffer,
    DiffConfig,
    MatchResult,
    MatchType,
    calculate_boundaries,
    compute_diff,
    diff_text_segments,
    identify_expansion,
)
from seo_optimizer.diffing.fingerprint import (
    ContentFingerprinter,
    MoveDetector,
    NGramFingerprint,
    SimHashFingerprint,
    build_fingerprint_index,
    detect_moved_content,
    find_all_moved_content,
)
from seo_optimizer.diffing.models import (
    Addition,
    ChangeSet,
    ChangeType,
    DiffConfidence,
    DiffResult,
    DocumentFingerprint,
    HighlightRegion,
    OptimizedContent,
    ParagraphContent,
    TextSegment,
)
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
    preload_model,
)

__all__ = [
    # Models
    "Addition",
    "ChangeSet",
    "ChangeType",
    "ContentChunk",
    "DiffConfidence",
    "DiffConfig",
    "DiffResult",
    "DocumentFingerprint",
    "HighlightRegion",
    "MatchResult",
    "MatchType",
    "NGramFingerprint",
    "OptimizedContent",
    "ParagraphContent",
    "SimHashFingerprint",
    "TextSegment",
    # Classes
    "ContentDiffer",
    "ContentFingerprinter",
    "DocumentChunker",
    "MoveDetector",
    "SemanticMatcher",
    # Core diff functions
    "compute_diff",
    "diff_text_segments",
    "identify_expansion",
    "calculate_boundaries",
    # Fingerprint functions
    "build_fingerprint_index",
    "detect_moved_content",
    "find_all_moved_content",
    # Semantic functions
    "compute_semantic_similarity",
    "compute_batch_similarity",
    "is_semantic_match",
    "is_semantically_equivalent",
    "is_redundant_content",
    "get_embedding",
    "preload_model",
    # Utility functions
    "normalize_text",
    "extract_chunks_from_text",
    # Constants
    "SEMANTIC_EQUIVALENCE_THRESHOLD",
    "REDUNDANCY_THRESHOLD",
    "CONSERVATIVE_THRESHOLD",
]

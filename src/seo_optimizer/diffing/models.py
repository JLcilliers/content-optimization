"""
Diffing Models - Data Structures for Change Detection

These models represent the output of the diffing system:
- What content is genuinely new
- Precise character positions for highlighting
- Confidence scores for human review triggers

Critical Design Principles:
1. Zero false positives: Only mark genuinely NEW content
2. Conservative approach: When uncertain, DON'T highlight
3. Precise boundaries: Character-level accuracy for highlighting
"""

from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
from typing import Any


class DiffConfidence(str, Enum):
    """Confidence level for a diff determination."""

    HIGH = "high"  # Confident this is correct (>0.9)
    MEDIUM = "medium"  # Reasonably confident (0.7-0.9)
    LOW = "low"  # Uncertain, may need review (<0.7)
    REVIEW_REQUIRED = "review_required"  # Must be reviewed by human


@dataclass
class HighlightRegion:
    """
    A region of text that should be highlighted as new content.

    This represents the precise character boundaries for applying
    green highlighting in the output DOCX.
    """

    # Which node in the AST this belongs to
    node_id: str

    # Position within the node's text content
    start_char: int
    end_char: int

    # The actual text to highlight
    text: str

    # Confidence that this is genuinely new
    confidence: float  # 0.0 to 1.0

    # Reason for highlighting (for debugging/audit)
    reason: str = "new_content"

    def __post_init__(self) -> None:
        """Validate the region."""
        if self.start_char < 0:
            raise ValueError("start_char must be non-negative")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("text length must match character range")


@dataclass
class Addition:
    """
    Represents a single new content addition.

    An Addition may span multiple HighlightRegions if it crosses
    formatting boundaries (e.g., partially bold text).
    """

    # Unique identifier for this addition
    addition_id: str

    # The node(s) this addition belongs to
    node_ids: list[str]

    # Regions to highlight
    highlight_regions: list[HighlightRegion]

    # Total text content added
    total_text: str

    # Overall confidence for this addition
    confidence: float

    # Classification
    addition_type: str = "new"  # new, expansion, inserted

    # Metadata for auditing
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_chars(self) -> int:
        """Total number of characters in this addition."""
        return sum(r.end_char - r.start_char for r in self.highlight_regions)

    @property
    def requires_review(self) -> bool:
        """Whether this addition should be reviewed by a human."""
        return self.confidence < 0.7


@dataclass
class DiffResult:
    """
    Result of comparing original content to a modified segment.

    Used during the diffing process to track individual comparisons.
    """

    # Position ID being compared
    position_id: str

    # Original text
    original_text: str

    # Modified text
    modified_text: str

    # Is this content semantically equivalent?
    is_semantic_match: bool

    # Semantic similarity score (0-1)
    similarity_score: float

    # Was this content moved from elsewhere?
    is_moved: bool

    # If content was expanded, what's the new portion?
    expansion_start: int | None = None
    expansion_end: int | None = None
    expansion_text: str | None = None

    # Confidence in this determination
    confidence: DiffConfidence = DiffConfidence.HIGH


@dataclass
class ChangeSet:
    """
    Complete set of changes detected between original and optimized documents.

    This is the primary output of the diffing system, containing all
    additions that should be highlighted green in the output.

    CRITICAL: Only additions with sufficient confidence should be applied.
    When in doubt, DON'T highlight (conservative approach).
    """

    # Unique identifier for this changeset
    changeset_id: str

    # Reference to original document
    original_doc_id: str

    # Reference to optimized document
    optimized_doc_id: str

    # All detected additions
    additions: list[Addition]

    # Additions that require human review (low confidence)
    review_required: list[Addition] = field(default_factory=list)

    # Summary statistics
    total_additions: int = 0
    total_chars_added: int = 0
    total_regions: int = 0

    # Overall confidence
    overall_confidence: float = 1.0

    # Any warnings or notes
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate summary statistics."""
        self.total_additions = len(self.additions)
        self.total_chars_added = sum(a.total_chars for a in self.additions)
        self.total_regions = sum(len(a.highlight_regions) for a in self.additions)

        # Review required items are low-confidence additions
        self.review_required = [a for a in self.additions if a.requires_review]

        # Overall confidence is minimum of all additions
        if self.additions:
            self.overall_confidence = min(a.confidence for a in self.additions)

    def get_high_confidence_additions(self) -> list[Addition]:
        """Get only additions with high confidence (safe to auto-apply)."""
        return [a for a in self.additions if a.confidence >= 0.85]

    def get_regions_for_node(self, node_id: str) -> list[HighlightRegion]:
        """Get all highlight regions for a specific node."""
        regions: list[HighlightRegion] = []
        for addition in self.additions:
            for region in addition.highlight_regions:
                if region.node_id == node_id:
                    regions.append(region)
        return sorted(regions, key=lambda r: r.start_char)


@dataclass(frozen=True)
class DocumentFingerprint:
    """
    Fingerprint for detecting moved content.

    Uses content hashing to identify when text has been relocated
    rather than newly created.
    """

    # Hash of the normalized text content
    content_hash: str

    # Position where this content originally appeared
    original_position: str

    # Normalized text (lowercased, whitespace normalized)
    normalized_text: str

    @classmethod
    def from_text(cls, text: str, position: str) -> "DocumentFingerprint":
        """Create a fingerprint from text content."""
        # Normalize: lowercase, collapse whitespace
        normalized = " ".join(text.lower().split())

        # Use blake2b for fast hashing
        content_hash = blake2b(normalized.encode(), digest_size=16).hexdigest()

        return cls(
            content_hash=content_hash,
            original_position=position,
            normalized_text=normalized,
        )

    def matches(self, other: "DocumentFingerprint", threshold: float = 0.95) -> bool:
        """
        Check if another fingerprint matches this one.

        Args:
            other: Another fingerprint to compare
            threshold: Similarity threshold (default 0.95 for near-exact match)

        Returns:
            True if fingerprints match above threshold
        """
        # Exact hash match
        if self.content_hash == other.content_hash:
            return True

        # For high threshold, require very similar text
        if threshold >= 0.95:
            # Simple character-level comparison for near-exact
            if len(self.normalized_text) == 0 or len(other.normalized_text) == 0:
                return False

            # Count matching characters
            shorter = min(len(self.normalized_text), len(other.normalized_text))
            longer = max(len(self.normalized_text), len(other.normalized_text))

            if shorter / longer < threshold:
                return False

            matching = sum(
                1
                for a, b in zip(self.normalized_text, other.normalized_text, strict=False)
                if a == b
            )
            return matching / longer >= threshold

        return False

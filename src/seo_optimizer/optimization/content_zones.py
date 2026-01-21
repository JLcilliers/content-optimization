"""
Content Zone Detection - Separates metadata from actual content.

This module prevents optimizers from modifying:
- URL fields
- Meta Title/Description
- Page headers like "Page Content Improvement"

Only BODY CONTENT should be optimized.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seo_optimizer.ingestion.models import ContentNode


class ContentZone(str, Enum):
    """Document zones for content classification."""

    METADATA = "metadata"  # URL, Meta Title, Meta Description, headers
    CONTENT = "content"    # Actual body paragraphs
    FAQ = "faq"           # FAQ section


# Patterns that indicate metadata, not content
METADATA_MARKERS = [
    "Page Content",
    "Page Content Improvement",
    "Page Content Optimization",
    "URL:",
    "Meta Title:",
    "Meta Description:",
    "Meta Information",
    "Document Information",
    "SEO Information",
]

# Regex patterns for metadata detection
METADATA_PATTERNS = [
    r"^https?://",           # URLs
    r"^www\.",               # URLs without protocol
    r"^\[?URL\]?:",          # URL label
    r"^\[?Meta\s",           # Meta labels
    r"^Page\s+Content",      # Page content headers
]


def is_metadata_node(node: ContentNode) -> bool:
    """
    Check if a node contains metadata rather than content.

    Args:
        node: ContentNode to check

    Returns:
        True if the node is metadata, False if it's content
    """
    text = node.text_content.strip()

    # Empty nodes are not content
    if not text:
        return True

    # Check for metadata markers
    text_lower = text.lower()
    for marker in METADATA_MARKERS:
        if text_lower.startswith(marker.lower()):
            return True

    # Check regex patterns
    for pattern in METADATA_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    # Check if it looks like a URL (contains URL with no other text)
    if re.search(r"^https?://\S+$", text):
        return True

    # Check if it's a structural label (very short, likely a header)
    if len(text) < 25:
        # Short text that looks like a label
        if text.endswith(":"):
            return True
        # All caps or title case short text
        if text.isupper() or (text.istitle() and " " not in text):
            return True

    return False


def is_url_field(node: ContentNode) -> bool:
    """
    Check if a node contains a URL field.

    Args:
        node: ContentNode to check

    Returns:
        True if the node is a URL field
    """
    text = node.text_content.strip()

    # Direct URL
    if re.match(r"^https?://", text):
        return True

    # URL with label
    if text.lower().startswith("url:"):
        return True

    # Contains URL pattern
    if "://" in text and len(text) < 200:
        # Likely a URL field, not a paragraph mentioning a URL
        return True

    return False


def should_skip_node(node: ContentNode) -> bool:
    """
    Determine if a node should be skipped during optimization.

    This is the main check used by optimizers to avoid modifying
    metadata fields.

    Args:
        node: ContentNode to check

    Returns:
        True if the node should NOT be optimized
    """
    return is_metadata_node(node) or is_url_field(node)


def get_content_zone(node: ContentNode) -> ContentZone:
    """
    Determine which zone a node belongs to.

    Args:
        node: ContentNode to classify

    Returns:
        ContentZone enum value
    """
    text = node.text_content.strip().lower()

    # Check for FAQ section
    if "frequently asked" in text or "faq" in text:
        return ContentZone.FAQ

    # Check for metadata
    if is_metadata_node(node):
        return ContentZone.METADATA

    return ContentZone.CONTENT


def filter_content_nodes(nodes: list) -> list:
    """
    Filter a list of nodes to only include actual content.

    Args:
        nodes: List of ContentNodes

    Returns:
        List of nodes that are actual content (not metadata)
    """
    return [node for node in nodes if not should_skip_node(node)]


def validate_insertion(original: str, modified: str) -> tuple[bool, str]:
    """
    Validate that a text insertion is grammatically sound.

    Args:
        original: Original text
        modified: Modified text

    Returns:
        Tuple of (is_valid, reason)
    """
    # Check for double periods
    if ".." in modified and ".." not in original:
        return False, "Double period detected"

    # Check for double commas
    if ",," in modified and ",," not in original:
        return False, "Double comma detected"

    # Check for period followed by comma (broken sentence)
    if ".," in modified and ".," not in original:
        return False, "Period followed by comma detected"

    # Check insertion doesn't break URLs
    if re.search(r"https?://", original):
        # Original contains a URL - it should not be modified
        if original != modified:
            return False, "URL should not be modified"

    # Check for broken URL patterns in the result
    if re.search(r"https?://[^/\s]*,", modified):
        return False, "URL appears corrupted by insertion"

    # Check for awkward comma insertions
    if re.search(r",\s+,", modified):
        return False, "Double comma with space detected"

    # CRITICAL: Reject known bad insertion patterns
    bad_patterns = [
        r",\s*especially\s+for\s+",  # ", especially for X" is grammatically wrong
        r",\s*particularly\s+for\s+",  # ", particularly for X"
        r",\s*specifically\s+for\s+",  # ", specifically for X"
        r"When working with [^,]+,\s+[a-z]",  # Awkward "When working with X, sentence"
    ]

    for pattern in bad_patterns:
        if re.search(pattern, modified) and not re.search(pattern, original):
            return False, f"Bad insertion pattern detected: {pattern}"

    # Check for redundant keyword insertion (same phrase appearing twice)
    words = modified.lower().split()
    for i in range(len(words) - 2):
        phrase = " ".join(words[i:i+3])
        # Check if this 3-word phrase appears elsewhere
        remaining = " ".join(words[:i] + words[i+3:])
        if phrase in remaining:
            return False, "Redundant phrase detected (potential keyword stuffing)"

    # Validate sentences don't start with lowercase after period
    sentences = re.split(r"(?<=[.!?])\s+", modified)
    for i, sentence in enumerate(sentences[1:], 1):  # Skip first
        if sentence and sentence[0].islower():
            # Check if it's actually a continuation (like "e.g.")
            if not re.search(r"\b(e\.g\.|i\.e\.|etc\.)\s*$", sentences[i-1]):
                # This might be okay in some cases, just warn
                pass

    return True, ""

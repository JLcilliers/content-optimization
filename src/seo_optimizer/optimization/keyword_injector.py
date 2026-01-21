"""
Keyword Injector - Strategic Keyword Placement

Responsibilities:
- Place primary keyword in priority zones
- Distribute secondary keywords naturally
- Maintain optimal density (1-2.5%)
- Avoid keyword stuffing

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .content_zones import filter_content_nodes, should_skip_node, validate_insertion
from .guardrails import SafetyGuardrails
from .models import ChangeType, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import KeywordAnalysis


# =============================================================================
# Keyword Placement Priorities
# =============================================================================

KEYWORD_PLACEMENT_PRIORITY = {
    "first_100_words": 0.9,  # High priority
    "h1": 0.85,  # High priority (if not already present)
    "h2_headings": 0.7,  # Medium priority
    "first_sentence_h2": 0.65,  # Medium priority
    "body_natural": 0.5,  # Low priority - natural distribution
}

# Minimum word gap between keyword mentions
MIN_KEYWORD_GAP = 100  # Words between mentions to avoid clustering


class KeywordInjector:
    """
    Strategically places keywords in content.

    Uses natural language insertion to avoid keyword stuffing
    while ensuring optimal placement in priority zones.
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the keyword injector."""
        self.config = config
        self.guardrails = guardrails

    def inject(
        self,
        ast: DocumentAST,
        keyword_analysis: KeywordAnalysis | None = None,
    ) -> list[OptimizationChange]:
        """
        Inject keywords strategically based on analysis.

        Args:
            ast: The document AST
            keyword_analysis: Pre-computed keyword analysis

        Returns:
            List of optimization changes
        """
        if not self.config.inject_keywords:
            return []

        if not self.config.primary_keyword:
            return []

        changes: list[OptimizationChange] = []

        # Check current density first
        full_text = ast.full_text
        density_check = self.guardrails.check_keyword_density(
            full_text, self.config.primary_keyword
        )

        if not density_check.is_safe:
            # Already over-optimized - don't inject more
            return []

        # Priority 1: First 100 words
        first_100_change = self._inject_in_first_100_words(ast)
        if first_100_change:
            changes.append(first_100_change)

        # Priority 2: H1 heading
        h1_change = self._inject_in_h1(ast)
        if h1_change:
            changes.append(h1_change)

        # Priority 3: H2 headings
        h2_changes = self._inject_in_h2_headings(ast)
        changes.extend(h2_changes)

        # Priority 4: Distribute secondary keywords
        if self.config.secondary_keywords:
            secondary_changes = self._distribute_secondary_keywords(ast)
            changes.extend(secondary_changes)

        # Priority 5: Natural body placement (if still under density)
        if len(changes) < self.config.max_changes_per_section:
            body_changes = self._inject_naturally_in_body(ast)
            changes.extend(body_changes)

        return changes

    def _inject_in_first_100_words(
        self, ast: DocumentAST
    ) -> OptimizationChange | None:
        """
        Ensure primary keyword appears in first 100 words.

        Args:
            ast: Document AST

        Returns:
            Change if injection was made, None otherwise
        """
        keyword = self.config.primary_keyword
        if not keyword:
            return None

        # Get first 100 words
        first_100 = self._get_first_n_words(ast.full_text, 100)

        if keyword.lower() in first_100.lower():
            return None  # Already present

        # Find first CONTENT paragraph to modify (skip metadata)
        for node in ast.nodes:
            if node.node_type == NodeType.PARAGRAPH and node.text_content.strip():
                # Skip metadata fields (URL, Meta Title, etc.)
                if should_skip_node(node):
                    continue

                original = node.text_content
                modified = self._insert_keyword_naturally(original, keyword, position="early")

                if modified and modified != original:
                    # Validate the insertion is grammatically sound
                    is_valid, reason = validate_insertion(original, modified)
                    if not is_valid:
                        continue  # Skip this insertion and try next paragraph

                    # Check if change passes guardrails
                    if self.guardrails.would_exceed_density(ast.full_text, keyword, 1):
                        return None

                    return OptimizationChange(
                        change_type=ChangeType.KEYWORD,
                        location="First paragraph",
                        original=original[:100] + "..." if len(original) > 100 else original,
                        optimized=modified[:100] + "..." if len(modified) > 100 else modified,
                        reason="Added primary keyword to first 100 words",
                        impact_score=4.0,
                        section_id=node.node_id,
                        full_original=original,
                        full_optimized=modified,
                    )

        return None

    def _inject_in_h1(self, ast: DocumentAST) -> OptimizationChange | None:
        """
        Add primary keyword to H1 if missing.

        Args:
            ast: Document AST

        Returns:
            Change if injection was made, None otherwise
        """
        keyword = self.config.primary_keyword
        if not keyword:
            return None

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                if level == 1:
                    if keyword.lower() not in node.text_content.lower():
                        # Keyword not in H1 - add it
                        original = node.text_content
                        modified = self._add_keyword_to_heading(original, keyword)

                        if modified and modified != original:
                            return OptimizationChange(
                                change_type=ChangeType.KEYWORD,
                                location="H1 Heading",
                                original=f"H1: {original}",
                                optimized=f"H1: {modified}",
                                reason="Added primary keyword to H1",
                                impact_score=3.5,
                                section_id=node.node_id,
                            )
                    break  # Only one H1

        return None

    def _inject_in_h2_headings(self, ast: DocumentAST) -> list[OptimizationChange]:
        """
        Add secondary keywords to H2 headings.

        Args:
            ast: Document AST

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []
        secondary = self.config.secondary_keywords

        if not secondary:
            return changes

        keyword_index = 0

        for node in ast.nodes:
            if node.node_type != NodeType.HEADING:
                continue

            level = node.metadata.get("level", 2)
            if level != 2:
                continue

            if keyword_index >= len(secondary):
                break

            # Check if any secondary keyword already in heading
            heading_lower = node.text_content.lower()
            keyword = secondary[keyword_index]

            if keyword.lower() not in heading_lower:
                original = node.text_content
                modified = self._add_keyword_to_heading(original, keyword)

                if modified and modified != original:
                    changes.append(
                        OptimizationChange(
                            change_type=ChangeType.KEYWORD,
                            location=f"H2: {original[:30]}...",
                            original=f"H2: {original}",
                            optimized=f"H2: {modified}",
                            reason=f"Added secondary keyword '{keyword}' to H2",
                            impact_score=2.0,
                            section_id=node.node_id,
                        )
                    )

            keyword_index += 1

        return changes

    def _distribute_secondary_keywords(
        self, ast: DocumentAST
    ) -> list[OptimizationChange]:
        """
        Distribute secondary keywords throughout body content.

        Args:
            ast: Document AST

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []
        secondary = self.config.secondary_keywords

        if not secondary:
            return changes

        # Track which keywords have been injected
        injected: set[str] = set()

        # Get only CONTENT paragraphs (exclude metadata like URL, Meta Title, etc.)
        paragraphs = [
            node for node in ast.nodes
            if node.node_type == NodeType.PARAGRAPH and not should_skip_node(node)
        ]

        # Distribute keywords across paragraphs
        for i, keyword in enumerate(secondary):
            if keyword.lower() in ast.full_text.lower():
                continue  # Already present

            # Choose a paragraph to inject into
            target_index = min(i * 2, len(paragraphs) - 1)
            if target_index < 0:
                continue

            node = paragraphs[target_index]
            original = node.text_content
            modified = self._insert_keyword_naturally(original, keyword, position="natural")

            if modified and modified != original:
                # Validate the insertion is grammatically sound
                is_valid, reason = validate_insertion(original, modified)
                if not is_valid:
                    continue  # Skip this insertion

                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.KEYWORD,
                        location=f"Paragraph {target_index + 1}",
                        original=original[:80] + "..." if len(original) > 80 else original,
                        optimized=modified[:80] + "..." if len(modified) > 80 else modified,
                        reason=f"Added secondary keyword '{keyword}'",
                        impact_score=1.5,
                        section_id=node.node_id,
                        full_original=original,
                        full_optimized=modified,
                    )
                )
                injected.add(keyword)

            if len(changes) >= 3:  # Limit secondary keyword injections
                break

        return changes

    def _inject_naturally_in_body(self, ast: DocumentAST) -> list[OptimizationChange]:
        """
        Inject primary keyword naturally in body paragraphs.

        Ensures proper distribution with MIN_KEYWORD_GAP words between mentions.
        """
        changes: list[OptimizationChange] = []
        keyword = self.config.primary_keyword

        if not keyword:
            return changes

        # Check current density
        density_check = self.guardrails.check_keyword_density(ast.full_text, keyword)
        if density_check.density >= self.config.min_keyword_density:
            return changes  # Already at or above minimum

        # Find CONTENT paragraphs where we could add mentions (exclude metadata)
        words_since_last = MIN_KEYWORD_GAP  # Allow first injection
        paragraphs = [
            node for node in ast.nodes
            if node.node_type == NodeType.PARAGRAPH and not should_skip_node(node)
        ]

        for node in paragraphs:
            word_count = len(node.text_content.split())

            # Check if keyword already in this paragraph
            if keyword.lower() in node.text_content.lower():
                words_since_last = 0
                continue

            words_since_last += word_count

            if words_since_last >= MIN_KEYWORD_GAP:
                # Safe to inject here
                if self.guardrails.would_exceed_density(ast.full_text, keyword, 1):
                    break

                original = node.text_content
                modified = self._insert_keyword_naturally(
                    original, keyword, position="natural"
                )

                if modified and modified != original:
                    # Validate the insertion is grammatically sound
                    is_valid, reason = validate_insertion(original, modified)
                    if not is_valid:
                        continue  # Skip this insertion

                    changes.append(
                        OptimizationChange(
                            change_type=ChangeType.KEYWORD,
                            location=f"Paragraph: {original[:30]}...",
                            original=original[:80] + "...",
                            optimized=modified[:80] + "...",
                            reason="Distributed primary keyword for optimal density",
                            impact_score=1.0,
                            section_id=node.node_id,
                            full_original=original,
                            full_optimized=modified,
                        )
                    )
                    words_since_last = 0

                    if len(changes) >= 2:  # Limit natural injections
                        break

        return changes

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_first_n_words(self, text: str, n: int) -> str:
        """Get first N words from text."""
        words = text.split()[:n]
        return " ".join(words)

    def _insert_keyword_naturally(
        self, sentence: str, keyword: str, position: str = "natural"
    ) -> str | None:
        """
        Insert keyword naturally into a sentence.

        Args:
            sentence: The sentence to modify
            keyword: The keyword to insert
            position: "early", "natural", or "late"

        Returns:
            Modified sentence or None if no good insertion point
        """
        if keyword.lower() in sentence.lower():
            return None  # Already contains keyword

        sentences = re.split(r"([.!?]+\s*)", sentence)
        if len(sentences) < 2:
            return None

        # Find a good insertion point
        if position == "early":
            # Try to modify first sentence
            first = sentences[0]
            modified = self._enhance_sentence_with_keyword(first, keyword)
            if modified:
                sentences[0] = modified
                return "".join(sentences)
        elif position == "late":
            # Modify last sentence
            for i in range(len(sentences) - 1, -1, -1):
                if sentences[i].strip() and not re.match(r"^[.!?]+\s*$", sentences[i]):
                    modified = self._enhance_sentence_with_keyword(sentences[i], keyword)
                    if modified:
                        sentences[i] = modified
                        return "".join(sentences)
        else:
            # Natural - find best fit
            for i, s in enumerate(sentences):
                if s.strip() and not re.match(r"^[.!?]+\s*$", s):
                    modified = self._enhance_sentence_with_keyword(s, keyword)
                    if modified:
                        sentences[i] = modified
                        return "".join(sentences)

        return None

    def _extract_topic_from_keyword(self, keyword: str) -> str:
        """
        Extract the core topic from a keyword phrase for natural insertion.

        Removes common verb prefixes to get a natural-sounding topic.

        Examples:
            "Running a booster club" → "booster clubs"
            "How to start a business" → "businesses"
            "Starting your own podcast" → "podcasts"

        Args:
            keyword: The keyword phrase

        Returns:
            The core topic suitable for natural sentence insertion
        """
        if not keyword:
            return keyword

        topic = keyword.strip()

        # Common verb prefixes to strip (order matters - longer phrases first)
        verb_prefixes = [
            "how to start a", "how to start an", "how to start",
            "how to create a", "how to create an", "how to create",
            "how to run a", "how to run an", "how to run",
            "how to manage a", "how to manage an", "how to manage",
            "how to build a", "how to build an", "how to build",
            "running a", "running an", "running your",
            "starting a", "starting an", "starting your",
            "creating a", "creating an", "creating your",
            "managing a", "managing an", "managing your",
            "building a", "building an", "building your",
            "what is a", "what is an", "what is",
            "what are", "why is", "why are",
            "when to", "where to", "who can",
        ]

        topic_lower = topic.lower()
        for prefix in verb_prefixes:
            if topic_lower.startswith(prefix):
                topic = topic[len(prefix):].strip()
                break

        # Handle "your own X" pattern
        if topic.lower().startswith("your own "):
            topic = topic[9:].strip()
        elif topic.lower().startswith("your "):
            topic = topic[5:].strip()
        elif topic.lower().startswith("own "):
            topic = topic[4:].strip()

        # Ensure we have something left
        if not topic or len(topic) < 3:
            return keyword  # Return original if extraction failed

        # Return plural form for more natural sentence flow
        # Simple pluralization
        if not topic.endswith("s"):
            if topic.endswith(("s", "x", "z", "ch", "sh")):
                topic = topic + "es"
            elif topic.endswith("y") and len(topic) > 1 and topic[-2] not in "aeiou":
                topic = topic[:-1] + "ies"
            else:
                topic = topic + "s"

        return topic

    def _enhance_sentence_with_keyword(
        self, sentence: str, keyword: str
    ) -> str | None:
        """
        Enhance a sentence by naturally incorporating keyword.

        Uses several strategies:
        1. Add as clarifying phrase
        2. Replace generic term with keyword
        3. Add as example or specification

        IMPORTANT: Extracts the topic from the keyword for natural phrasing.
        Example: "Running a booster club" → inserts "booster clubs" not the raw keyword.
        """
        # Extract the core topic for natural insertion
        topic = self._extract_topic_from_keyword(keyword)

        # Strategy 1: Add as clarifying phrase
        # "This technology" → "This technology, particularly {topic},"
        generic_terms = [
            "this technology",
            "this approach",
            "this method",
            "these tools",
            "this process",
            "these techniques",
            "the solution",
            "this strategy",
        ]

        sentence_lower = sentence.lower()
        for term in generic_terms:
            if term in sentence_lower:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                replacement = f"{term} (such as {topic})"
                return pattern.sub(replacement, sentence, count=1)

        # Strategy 2: Add as introductory context
        # If sentence is about the topic, add keyword as context
        topic_indicators = ["when", "if you", "to", "for", "with"]
        first_word = sentence.split()[0].lower() if sentence.split() else ""

        if first_word in topic_indicators:
            return f"When working with {topic}, {sentence[0].lower()}{sentence[1:]}"

        # Strategy 3: Add as specification at end
        # Note: When called from _insert_keyword_naturally, sentences are split
        # with punctuation captured separately, so we should NOT add a period here.
        # The original punctuation will be re-added during the join.
        if len(sentence) < 100 and not sentence.rstrip().endswith((",", ":", ";")):
            # Add as trailing context (no period - will be added by rejoin)
            clean_sentence = sentence.rstrip(".")
            return f"{clean_sentence}, especially for {topic}"

        return None

    def _add_keyword_to_heading(self, heading: str, keyword: str) -> str | None:
        """
        Add keyword to a heading naturally.

        Args:
            heading: The heading text
            keyword: The keyword to add

        Returns:
            Modified heading or None
        """
        if keyword.lower() in heading.lower():
            return None  # Already present

        # Check if heading is a question
        if heading.endswith("?"):
            # Add keyword as context
            clean = heading.rstrip("?")
            return f"{clean} for {keyword}?"

        # Check length - don't make too long
        max_length = 70 if "H1" in heading else 60

        # Try different patterns
        patterns = [
            f"{heading}: {keyword}",
            f"{keyword} - {heading}",
            f"{heading} for {keyword}",
        ]

        for pattern in patterns:
            if len(pattern) <= max_length:
                return pattern

        return None

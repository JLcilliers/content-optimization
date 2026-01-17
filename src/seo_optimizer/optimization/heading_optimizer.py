"""
Heading Optimizer - Document Structure Fixes

Responsibilities:
- Fix single H1 violations (multiple H1s, missing H1)
- Correct heading hierarchy (no skipped levels)
- Insert H2s every 300 words in long content blocks
- Optimize heading length (20-70 chars for H1, 3-8 words for H2)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType

from .guardrails import SafetyGuardrails
from .models import ChangeType, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import HeadingAnalysis


# =============================================================================
# Heading Constraints
# =============================================================================

H1_MIN_CHARS = 20
H1_MAX_CHARS = 70
H2_MIN_WORDS = 3
H2_MAX_WORDS = 8
WORDS_PER_SECTION = 300  # Insert heading every N words


class HeadingOptimizer:
    """
    Optimizes document heading structure.

    Fixes common heading issues:
    - Multiple H1 tags (demote extras to H2)
    - Missing H1 (promote or generate)
    - Skipped heading levels (H2 → H4)
    - Long content blocks without headings
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the heading optimizer."""
        self.config = config
        self.guardrails = guardrails

    def optimize(
        self,
        ast: DocumentAST,
        heading_analysis: HeadingAnalysis | None = None,
    ) -> list[OptimizationChange]:
        """
        Fix all heading issues in the document.

        Args:
            ast: The document AST
            heading_analysis: Pre-computed heading analysis (optional)

        Returns:
            List of optimization changes made
        """
        if not self.config.optimize_headings:
            return []

        changes: list[OptimizationChange] = []

        # Get current heading structure
        headings = self._extract_headings(ast)

        # Fix H1 issues first
        h1_changes = self._fix_h1_issues(ast, headings)
        changes.extend(h1_changes)

        # Re-extract after H1 fixes
        headings = self._extract_headings(ast)

        # Fix hierarchy gaps
        hierarchy_changes = self._fix_hierarchy_gaps(ast, headings)
        changes.extend(hierarchy_changes)

        # Add section headings for long blocks
        section_changes = self._insert_section_headings(ast)
        changes.extend(section_changes)

        # Optimize heading text
        text_changes = self._optimize_heading_text(ast)
        changes.extend(text_changes)

        return changes

    def _extract_headings(
        self, ast: DocumentAST
    ) -> list[tuple[ContentNode, int]]:
        """
        Extract all headings with their levels.

        Returns:
            List of (node, level) tuples
        """
        headings: list[tuple[ContentNode, int]] = []

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                headings.append((node, level))

        return headings

    def _fix_h1_issues(
        self,
        ast: DocumentAST,
        headings: list[tuple[ContentNode, int]],
    ) -> list[OptimizationChange]:
        """
        Fix H1 violations (multiple or missing).

        Args:
            ast: Document AST
            headings: Current heading list

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []

        # Find all H1s
        h1_headings = [(node, i) for i, (node, level) in enumerate(headings) if level == 1]

        if len(h1_headings) > 1:
            # Multiple H1s - keep the best one, demote others
            changes.extend(self._fix_multiple_h1(ast, h1_headings))
        elif len(h1_headings) == 0:
            # No H1 - promote first H2 or generate one
            changes.extend(self._fix_missing_h1(ast, headings))

        return changes

    def _fix_multiple_h1(
        self,
        ast: DocumentAST,
        h1_headings: list[tuple[ContentNode, int]],
    ) -> list[OptimizationChange]:
        """
        Handle multiple H1 tags by keeping best one and demoting others.

        Strategy:
        1. Keep H1 that contains primary keyword
        2. If none contain keyword, keep the first one
        3. Demote all others to H2
        """
        changes: list[OptimizationChange] = []

        # Find best H1 to keep
        best_index = 0
        if self.config.primary_keyword:
            keyword_lower = self.config.primary_keyword.lower()
            for i, (node, _) in enumerate(h1_headings):
                if keyword_lower in node.text_content.lower():
                    best_index = i
                    break

        # Demote all others
        for i, (node, _) in enumerate(h1_headings):
            if i != best_index:
                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.HEADING,
                        location=f"H1: {node.text_content[:30]}...",
                        original=f"H1: {node.text_content}",
                        optimized=f"H2: {node.text_content}",
                        reason="Demoted duplicate H1 to H2 (only one H1 allowed)",
                        impact_score=3.0,
                        section_id=node.node_id,
                    )
                )
                # Update the node metadata
                node.metadata["level"] = 2
                node.metadata["original_level"] = 1

        return changes

    def _fix_missing_h1(
        self,
        ast: DocumentAST,
        headings: list[tuple[ContentNode, int]],
    ) -> list[OptimizationChange]:
        """
        Handle missing H1 by promoting H2 or generating one.

        Strategy:
        1. If first heading is H2 with keyword, promote it
        2. Otherwise, generate H1 from keyword + content summary
        """
        changes: list[OptimizationChange] = []

        # Find first H2
        h2_headings = [(node, i) for i, (node, level) in enumerate(headings) if level == 2]

        if h2_headings:
            # Promote first H2 to H1
            first_h2, _ = h2_headings[0]

            changes.append(
                OptimizationChange(
                    change_type=ChangeType.HEADING,
                    location=f"H2: {first_h2.text_content[:30]}...",
                    original=f"H2: {first_h2.text_content}",
                    optimized=f"H1: {first_h2.text_content}",
                    reason="Promoted first H2 to H1 (document needs main heading)",
                    impact_score=4.0,
                    section_id=first_h2.node_id,
                )
            )
            # Update the node metadata
            first_h2.metadata["level"] = 1
            first_h2.metadata["original_level"] = 2
        else:
            # Generate H1 from keyword
            generated_h1 = self._generate_h1(ast)
            if generated_h1:
                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.HEADING,
                        location="Document start",
                        original="(No H1)",
                        optimized=f"H1: {generated_h1}",
                        reason="Generated missing H1 heading",
                        impact_score=5.0,
                    )
                )

        return changes

    def _generate_h1(self, ast: DocumentAST) -> str | None:
        """
        Generate an H1 heading from keyword and content.

        Args:
            ast: Document AST

        Returns:
            Generated H1 text or None
        """
        if not self.config.primary_keyword:
            return None

        keyword = self.config.primary_keyword

        # Try to extract topic from first paragraph
        first_para = None
        for node in ast.nodes:
            if node.node_type == NodeType.PARAGRAPH and node.text_content.strip():
                first_para = node.text_content
                break

        if first_para:
            # Extract key concept from first paragraph
            # Simple approach: use first sentence topic
            first_sentence = first_para.split(".")[0].strip()
            if len(first_sentence) > 10:
                # Try to create a descriptive H1
                if keyword.lower() in first_sentence.lower():
                    # Keyword already in first sentence - use as base
                    h1 = self._trim_to_h1_length(first_sentence)
                else:
                    # Combine keyword with topic indicator
                    h1 = f"{keyword}: A Complete Guide"
            else:
                h1 = f"{keyword}: A Complete Guide"
        else:
            # Fallback to simple keyword-based H1
            h1 = f"{keyword}: A Complete Guide"

        return h1

    def _trim_to_h1_length(self, text: str) -> str:
        """Trim text to valid H1 length (20-70 chars)."""
        if len(text) <= H1_MAX_CHARS:
            return text

        # Try to break at word boundary
        trimmed = text[:H1_MAX_CHARS]
        last_space = trimmed.rfind(" ")
        if last_space > H1_MIN_CHARS:
            return trimmed[:last_space]

        return trimmed

    def _fix_hierarchy_gaps(
        self,
        ast: DocumentAST,
        headings: list[tuple[ContentNode, int]],
    ) -> list[OptimizationChange]:
        """
        Fix skipped heading levels (e.g., H2 → H4).

        Strategy:
        - When level skips more than 1, adjust to sequential
        """
        changes: list[OptimizationChange] = []

        if len(headings) < 2:
            return changes

        prev_level = headings[0][1]

        for node, level in headings[1:]:
            # Check for downward skip (e.g., H2 → H4)
            if level > prev_level + 1:
                new_level = prev_level + 1
                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.HEADING,
                        location=f"H{level}: {node.text_content[:30]}...",
                        original=f"H{level}: {node.text_content}",
                        optimized=f"H{new_level}: {node.text_content}",
                        reason=f"Fixed hierarchy gap (H{prev_level} → H{level} → H{new_level})",
                        impact_score=2.0,
                        section_id=node.node_id,
                    )
                )
                node.metadata["level"] = new_level
                node.metadata["original_level"] = level
                level = new_level

            prev_level = level

        return changes

    def _insert_section_headings(self, ast: DocumentAST) -> list[OptimizationChange]:
        """
        Insert H2/H3 headings in long content blocks without structure.

        Strategy:
        - Find paragraphs that are >300 words from last heading
        - Generate appropriate heading from content
        """
        changes: list[OptimizationChange] = []

        words_since_heading = 0
        last_heading_level = 1

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                words_since_heading = 0
                last_heading_level = node.metadata.get("level", 2)
            elif node.node_type == NodeType.PARAGRAPH:
                word_count = len(node.text_content.split())
                words_since_heading += word_count

                if words_since_heading > WORDS_PER_SECTION:
                    # Need to insert a heading before this paragraph
                    new_level = min(last_heading_level + 1, 3)  # Max H3
                    heading_text = self._generate_section_heading(node.text_content)

                    if heading_text:
                        changes.append(
                            OptimizationChange(
                                change_type=ChangeType.STRUCTURE,
                                location=f"Before paragraph: {node.text_content[:30]}...",
                                original="(No heading)",
                                optimized=f"H{new_level}: {heading_text}",
                                reason=f"Added section heading ({words_since_heading} words since last heading)",
                                impact_score=2.5,
                                section_id=node.node_id,
                            )
                        )
                        words_since_heading = word_count

        return changes

    def _generate_section_heading(self, paragraph_text: str) -> str | None:
        """
        Generate a section heading from paragraph content.

        Args:
            paragraph_text: The paragraph to summarize

        Returns:
            Generated heading text or None
        """
        if not paragraph_text:
            return None

        # Extract first sentence as base
        first_sentence = paragraph_text.split(".")[0].strip()

        if len(first_sentence) < 10:
            return None

        # Try to convert to question format if appropriate
        question_words = ["what", "how", "why", "when", "where", "who"]
        first_word = first_sentence.split()[0].lower() if first_sentence.split() else ""

        if any(first_sentence.lower().startswith(f"{w} ") for w in question_words):
            # Already a question - use as heading
            heading = first_sentence
            if not heading.endswith("?"):
                heading += "?"
        else:
            # Extract key noun phrase
            heading = self._extract_topic_phrase(first_sentence)

        # Trim to appropriate length
        if heading and len(heading) > 60:
            heading = heading[:57] + "..."

        return heading

    def _extract_topic_phrase(self, sentence: str) -> str:
        """
        Extract the main topic phrase from a sentence.

        Simple heuristic approach without NLP.
        """
        # Remove common sentence starters
        starters = [
            "this is",
            "there are",
            "it is",
            "these are",
            "the main",
            "one of the",
            "in order to",
        ]

        sentence_lower = sentence.lower()
        for starter in starters:
            if sentence_lower.startswith(starter):
                sentence = sentence[len(starter) :].strip()
                break

        # Take first noun phrase (rough heuristic)
        words = sentence.split()
        if len(words) <= H2_MAX_WORDS:
            return sentence.capitalize()

        # Take first few words
        phrase = " ".join(words[:H2_MAX_WORDS])
        return phrase.capitalize()

    def _optimize_heading_text(self, ast: DocumentAST) -> list[OptimizationChange]:
        """
        Optimize heading text for length and keyword placement.

        - H1: 20-70 characters
        - H2: 3-8 words
        - Add keyword if missing from H1
        """
        changes: list[OptimizationChange] = []

        for node in ast.nodes:
            if node.node_type != NodeType.HEADING:
                continue

            level = node.metadata.get("level", 2)
            text = node.text_content

            if level == 1:
                # Check H1 constraints
                change = self._optimize_h1_text(node, text)
                if change:
                    changes.append(change)
            elif level == 2:
                # Check H2 constraints
                change = self._optimize_h2_text(node, text)
                if change:
                    changes.append(change)

        return changes

    def _optimize_h1_text(
        self, node: ContentNode, text: str
    ) -> OptimizationChange | None:
        """Optimize H1 heading text."""
        optimized = text

        # Check length
        if len(text) < H1_MIN_CHARS:
            # Too short - try to expand
            if self.config.primary_keyword and self.config.primary_keyword.lower() not in text.lower():
                optimized = f"{text}: {self.config.primary_keyword}"
        elif len(text) > H1_MAX_CHARS:
            # Too long - trim
            optimized = self._trim_to_h1_length(text)

        # Check for keyword
        if (
            self.config.primary_keyword
            and self.config.primary_keyword.lower() not in optimized.lower()
        ):
            # Add keyword if it fits
            with_keyword = f"{self.config.primary_keyword}: {optimized}"
            if len(with_keyword) <= H1_MAX_CHARS:
                optimized = with_keyword
            else:
                # Try putting keyword at end
                with_keyword = f"{optimized} - {self.config.primary_keyword}"
                if len(with_keyword) <= H1_MAX_CHARS:
                    optimized = with_keyword

        if optimized != text:
            return OptimizationChange(
                change_type=ChangeType.HEADING,
                location=f"H1: {text[:30]}...",
                original=f"H1: {text}",
                optimized=f"H1: {optimized}",
                reason="Optimized H1 for length and keyword placement",
                impact_score=3.0,
                section_id=node.node_id,
            )

        return None

    def _optimize_h2_text(
        self, node: ContentNode, text: str
    ) -> OptimizationChange | None:
        """Optimize H2 heading text."""
        words = text.split()
        word_count = len(words)

        if word_count < H2_MIN_WORDS or word_count > H2_MAX_WORDS:
            optimized = text

            if word_count < H2_MIN_WORDS:
                # Too short - try to expand
                # Add context word if possible
                if self.config.primary_keyword:
                    optimized = f"{text} for {self.config.primary_keyword}"
                else:
                    optimized = f"Understanding {text}"
            elif word_count > H2_MAX_WORDS:
                # Too long - trim to max words
                optimized = " ".join(words[:H2_MAX_WORDS])

            return OptimizationChange(
                change_type=ChangeType.HEADING,
                location=f"H2: {text[:30]}...",
                original=f"H2: {text}",
                optimized=f"H2: {optimized}",
                reason=f"Adjusted H2 word count ({word_count} → {len(optimized.split())})",
                impact_score=1.5,
                section_id=node.node_id,
            )

        # Check for question format opportunity
        if not text.endswith("?") and self._could_be_question(text):
            question = self._convert_to_question(text)
            if question and question != text:
                return OptimizationChange(
                    change_type=ChangeType.HEADING,
                    location=f"H2: {text[:30]}...",
                    original=f"H2: {text}",
                    optimized=f"H2: {question}",
                    reason="Converted heading to question format for better engagement",
                    impact_score=1.0,
                    section_id=node.node_id,
                )

        return None

    def _could_be_question(self, text: str) -> bool:
        """Check if heading could be converted to question format."""
        # Headings about "how to", "what is", etc. can be questions
        patterns = [
            r"^how to",
            r"^what (is|are)",
            r"^why",
            r"^benefits of",
            r"^advantages of",
            r"^types of",
            r"^ways to",
        ]

        text_lower = text.lower()
        return any(re.match(p, text_lower) for p in patterns)

    def _convert_to_question(self, text: str) -> str | None:
        """Convert heading statement to question format."""
        text_lower = text.lower()

        conversions = [
            (r"^how to (.+)", r"How Do You \1?"),
            (r"^what (is|are) (.+)", r"What \1 \2?"),
            (r"^benefits of (.+)", r"What Are the Benefits of \1?"),
            (r"^advantages of (.+)", r"What Are the Advantages of \1?"),
            (r"^types of (.+)", r"What Are the Types of \1?"),
            (r"^ways to (.+)", r"What Are the Best Ways to \1?"),
        ]

        for pattern, replacement in conversions:
            if re.match(pattern, text_lower):
                result = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                # Capitalize first letter of each word for title case
                return result.title()

        return None

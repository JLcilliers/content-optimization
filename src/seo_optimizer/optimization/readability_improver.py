"""
Readability Improver - UX Optimization

Responsibilities:
- Split complex sentences (>25 words)
- Convert passive to active voice
- Reduce reading grade level
- Improve sentence variance (burstiness)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType

from .guardrails import SafetyGuardrails
from .models import ChangeType, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import ReadabilityScore


# =============================================================================
# Readability Constants
# =============================================================================

# Sentence length thresholds
MAX_SENTENCE_WORDS = 25
OPTIMAL_SENTENCE_WORDS = 15
MIN_SENTENCE_WORDS = 5

# Complex word syllable threshold
COMPLEX_WORD_SYLLABLES = 3

# Common passive voice patterns
PASSIVE_PATTERNS = [
    (r"\b(is|are|was|were|been|being)\s+(\w+ed)\b", "passive"),
    (r"\b(has|have|had)\s+been\s+(\w+ed)\b", "passive_perfect"),
    (r"\b(will|would|could|should|may|might|must)\s+be\s+(\w+ed)\b", "passive_modal"),
]

# Simple word replacements for complex words
SIMPLE_ALTERNATIVES = {
    "utilize": "use",
    "implement": "set up",
    "facilitate": "help",
    "subsequent": "later",
    "additional": "more",
    "approximately": "about",
    "demonstrate": "show",
    "modification": "change",
    "significantly": "greatly",
    "functionality": "feature",
    "methodology": "method",
    "optimization": "improvement",
    "configuration": "setup",
    "initialization": "start",
    "implementation": "setup",
    "authentication": "login",
    "authorization": "permission",
    "comprehensive": "complete",
    "requirements": "needs",
    "prerequisites": "requirements",
    "consequently": "so",
    "nevertheless": "still",
    "notwithstanding": "despite",
    "aforementioned": "this",
    "hereinafter": "below",
}


class ReadabilityImprover:
    """
    Improves content readability while preserving meaning.

    Focuses on:
    - Shorter, clearer sentences
    - Active voice over passive
    - Simpler vocabulary
    - Natural sentence rhythm (burstiness)
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the readability improver."""
        self.config = config
        self.guardrails = guardrails

    def improve(
        self,
        ast: DocumentAST,
        readability_analysis: ReadabilityScore | None = None,
    ) -> list[OptimizationChange]:
        """
        Fix readability issues in the document.

        Args:
            ast: The document AST
            readability_analysis: Pre-computed readability analysis

        Returns:
            List of optimization changes
        """
        if not self.config.improve_readability:
            return []

        changes: list[OptimizationChange] = []

        # Process each paragraph
        for node in ast.nodes:
            if node.node_type != NodeType.PARAGRAPH:
                continue

            # Split complex sentences
            split_changes = self._split_complex_sentences(node)
            changes.extend(split_changes)

            # Convert passive to active voice
            voice_changes = self._convert_passive_to_active(node)
            changes.extend(voice_changes)

            # Simplify vocabulary
            vocab_changes = self._simplify_vocabulary(node)
            changes.extend(vocab_changes)

        # Improve sentence variance
        variance_changes = self._improve_sentence_variance(ast)
        changes.extend(variance_changes)

        return changes

    def _split_complex_sentences(
        self, node: ContentNode
    ) -> list[OptimizationChange]:
        """
        Split sentences that exceed max length.

        Args:
            node: Paragraph node to process

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []
        text = node.text_content

        # Extract sentences
        sentences = self._extract_sentences(text)
        modified_sentences: list[str] = []
        any_modified = False

        for sentence in sentences:
            word_count = len(sentence.split())

            if word_count > self.config.max_sentence_length:
                # Try to split this sentence
                split_result = self._split_sentence(sentence)

                if split_result and len(split_result) > 1:
                    modified_sentences.extend(split_result)
                    any_modified = True
                else:
                    modified_sentences.append(sentence)
            else:
                modified_sentences.append(sentence)

        if any_modified:
            modified_text = " ".join(modified_sentences)

            # Preserve factual content
            modified_text = self.guardrails.preserve_factual_content(text, modified_text)

            changes.append(
                OptimizationChange(
                    change_type=ChangeType.READABILITY,
                    location=f"Paragraph: {text[:30]}...",
                    original=text[:100] + "..." if len(text) > 100 else text,
                    optimized=modified_text[:100] + "..." if len(modified_text) > 100 else modified_text,
                    reason="Split complex sentences for better readability",
                    impact_score=2.0,
                    section_id=node.node_id,
                )
            )

        return changes

    def _split_sentence(self, sentence: str) -> list[str] | None:
        """
        Split a long sentence into shorter ones.

        Looks for natural break points:
        - Conjunctions (and, but, or)
        - Relative clauses (which, that, who)
        - Semicolons
        """
        word_count = len(sentence.split())

        if word_count <= MAX_SENTENCE_WORDS:
            return None

        # Try splitting at conjunctions
        conjunction_pattern = r",?\s+(and|but|or|however|therefore|moreover|furthermore)\s+"
        parts = re.split(conjunction_pattern, sentence, maxsplit=1)

        if len(parts) >= 3:
            first_part = parts[0].strip()
            conjunction = parts[1]
            second_part = parts[2].strip()

            # Ensure each part is substantial
            if len(first_part.split()) >= MIN_SENTENCE_WORDS and len(second_part.split()) >= MIN_SENTENCE_WORDS:
                # Capitalize second part and add period to first
                if not first_part.endswith((".", "!", "?")):
                    first_part += "."

                # Capitalize continuation appropriately
                if conjunction.lower() in ["and", "but", "or"]:
                    second_part = second_part[0].upper() + second_part[1:] if second_part else second_part
                else:
                    second_part = f"{conjunction.capitalize()} {second_part[0].lower()}{second_part[1:]}"

                if not second_part.endswith((".", "!", "?")):
                    second_part += "."

                return [first_part, second_part]

        # Try splitting at semicolons
        if ";" in sentence:
            parts = sentence.split(";", 1)
            if len(parts) == 2:
                first = parts[0].strip()
                second = parts[1].strip()

                if len(first.split()) >= MIN_SENTENCE_WORDS and len(second.split()) >= MIN_SENTENCE_WORDS:
                    if not first.endswith((".", "!", "?")):
                        first += "."
                    second = second[0].upper() + second[1:] if second else second
                    if not second.endswith((".", "!", "?")):
                        second += "."
                    return [first, second]

        # Try splitting at relative clauses
        relative_pattern = r",?\s+(which|who|that)\s+"
        parts = re.split(relative_pattern, sentence, maxsplit=1)

        if len(parts) >= 3:
            first_part = parts[0].strip()
            relative = parts[1]
            second_part = parts[2].strip()

            if len(first_part.split()) >= MIN_SENTENCE_WORDS:
                if not first_part.endswith((".", "!", "?")):
                    first_part += "."

                # Create a new sentence from the relative clause
                if relative.lower() == "which" or relative.lower() == "that":
                    second_part = f"This {second_part}"
                elif relative.lower() == "who":
                    second_part = f"They {second_part}"

                if not second_part.endswith((".", "!", "?")):
                    second_part += "."

                return [first_part, second_part]

        return None

    def _convert_passive_to_active(
        self, node: ContentNode
    ) -> list[OptimizationChange]:
        """
        Convert passive voice constructions to active voice.

        Args:
            node: Paragraph node to process

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []
        text = node.text_content

        # Track if we made any modifications
        modified_text = text

        for pattern, voice_type in PASSIVE_PATTERNS:
            matches = list(re.finditer(pattern, modified_text, re.IGNORECASE))

            for match in reversed(matches):  # Process in reverse to maintain positions
                passive_phrase = match.group(0)
                active = self._convert_to_active(passive_phrase, modified_text, match.start())

                if active and active != passive_phrase:
                    modified_text = modified_text[:match.start()] + active + modified_text[match.end():]

        if modified_text != text:
            # Preserve factual content
            modified_text = self.guardrails.preserve_factual_content(text, modified_text)

            changes.append(
                OptimizationChange(
                    change_type=ChangeType.READABILITY,
                    location=f"Paragraph: {text[:30]}...",
                    original=text[:100] + "...",
                    optimized=modified_text[:100] + "...",
                    reason="Converted passive voice to active voice",
                    impact_score=1.5,
                    section_id=node.node_id,
                )
            )

        return changes

    def _convert_to_active(
        self, passive_phrase: str, context: str, position: int
    ) -> str | None:
        """
        Convert a passive phrase to active voice.

        This is a simplified conversion that handles common patterns.
        """
        # Extract the verb
        match = re.match(r"(is|are|was|were|been|being)\s+(\w+ed)", passive_phrase, re.IGNORECASE)

        if not match:
            return None

        aux_verb = match.group(1).lower()
        past_participle = match.group(2)

        # Look for "by {agent}" after the passive phrase
        after_pos = position + len(passive_phrase)
        after_text = context[after_pos:after_pos + 50] if after_pos < len(context) else ""

        by_match = re.match(r"\s+by\s+(\w+(?:\s+\w+)?)", after_text, re.IGNORECASE)

        if by_match:
            agent = by_match.group(1)
            # Convert to active: "was written by John" → "John wrote"
            # This is simplified - a full implementation would handle tenses better
            active_verb = self._get_active_form(past_participle, aux_verb)
            return f"{agent} {active_verb}"

        # Without agent, just note that this could be improved
        # In practice, we'd need more context or would flag for review
        return None

    def _get_active_form(self, past_participle: str, aux_verb: str) -> str:
        """Get active verb form from past participle."""
        # Simple mapping - a full implementation would use a verb conjugation library
        if aux_verb in ["was", "is"]:
            # Past/present simple
            if past_participle.endswith("ied"):
                return past_participle[:-3] + "ies"
            elif past_participle.endswith("ed"):
                return past_participle[:-2] + "s"
        elif aux_verb in ["were", "are"]:
            if past_participle.endswith("ed"):
                return past_participle[:-1]  # Simple approximation

        return past_participle  # Fallback

    def _simplify_vocabulary(self, node: ContentNode) -> list[OptimizationChange]:
        """
        Replace complex words with simpler alternatives.

        Args:
            node: Paragraph node to process

        Returns:
            List of changes made
        """
        changes: list[OptimizationChange] = []
        text = node.text_content
        modified_text = text
        replacements_made: list[tuple[str, str]] = []

        for complex_word, simple_word in SIMPLE_ALTERNATIVES.items():
            pattern = re.compile(rf"\b{re.escape(complex_word)}\b", re.IGNORECASE)

            if pattern.search(modified_text):
                # Preserve case
                def replace_preserving_case(m: re.Match[str]) -> str:
                    original = m.group(0)
                    if original[0].isupper():
                        return simple_word.capitalize()
                    return simple_word

                modified_text = pattern.sub(replace_preserving_case, modified_text)
                replacements_made.append((complex_word, simple_word))

        if replacements_made:
            changes.append(
                OptimizationChange(
                    change_type=ChangeType.READABILITY,
                    location=f"Paragraph: {text[:30]}...",
                    original=text[:100] + "...",
                    optimized=modified_text[:100] + "...",
                    reason=f"Simplified vocabulary: {', '.join(f'{c}→{s}' for c, s in replacements_made[:3])}",
                    impact_score=1.0,
                    section_id=node.node_id,
                )
            )

        return changes

    def _improve_sentence_variance(
        self, ast: DocumentAST
    ) -> list[OptimizationChange]:
        """
        Improve sentence length variance for natural rhythm.

        AI-generated text tends to have monotonous sentence lengths.
        This adds variety to seem more human-written.
        """
        changes: list[OptimizationChange] = []

        # Extract all sentences from document
        all_sentences: list[str] = []
        for node in ast.nodes:
            if node.node_type == NodeType.PARAGRAPH:
                sentences = self._extract_sentences(node.text_content)
                all_sentences.extend(sentences)

        if len(all_sentences) < 5:
            return changes  # Not enough sentences to analyze

        # Check current variance
        is_acceptable, message, variance = self.guardrails.check_sentence_variance(all_sentences)

        if is_acceptable:
            return changes  # Variance is fine

        # Get suggestions for improvement
        suggestions = self.guardrails.suggest_variance_improvements(all_sentences)

        if suggestions:
            # Create a single advisory change
            changes.append(
                OptimizationChange(
                    change_type=ChangeType.READABILITY,
                    location="Document-wide",
                    original=f"Current variance: {variance:.2f}",
                    optimized=f"Suggested changes: {len(suggestions)}",
                    reason=message + ". Consider varying sentence lengths more.",
                    impact_score=1.5,
                )
            )

        return changes

    def _extract_sentences(self, text: str) -> list[str]:
        """
        Extract sentences from text.

        Args:
            text: The text to process

        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def count_syllables(self, word: str) -> int:
        """
        Count syllables in a word.

        Uses a simple heuristic approach.
        """
        word = word.lower()
        word = re.sub(r"[^a-z]", "", word)

        if not word:
            return 0

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                syllable_count += 1
            prev_is_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        # Adjust for -le ending
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1

        return max(1, syllable_count)

    def identify_complex_words(self, text: str) -> list[str]:
        """
        Find complex words (3+ syllables) in text.

        Args:
            text: The text to analyze

        Returns:
            List of complex words
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        complex_words = [w for w in words if self.count_syllables(w) >= COMPLEX_WORD_SYLLABLES]
        return complex_words

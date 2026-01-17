"""
Safety Guardrails - Over-Optimization Prevention System

CRITICAL MODULE: Prevents penalties from:
- Keyword stuffing (>3% density warning, >5% blocked)
- Entity stuffing (>5% density)
- AI-detectable vocabulary patterns
- Factual content modification
- Monotonous sentence rhythm

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import (
    GuardrailViolation,
    OptimizationChange,
    OptimizationConfig,
)

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import AnalysisResult


# =============================================================================
# Thresholds from Research
# =============================================================================

# Keyword density limits
KEYWORD_DENSITY_OPTIMAL = 1.5  # 1-2% is ideal
KEYWORD_DENSITY_WARNING = 3.0  # >3% triggers warning
KEYWORD_DENSITY_DANGER = 5.0  # >5% = keyword stuffing penalty

# Entity density limits
ENTITY_DENSITY_MAX = 5.0  # >5% = entity stuffing

# Sentence variance (burstiness)
MIN_SENTENCE_LENGTH = 5  # Words
MAX_SENTENCE_LENGTH = 35  # Words
SENTENCE_VARIANCE_MIN = 0.25  # Minimum std dev / mean ratio


# =============================================================================
# AI-Flagged Vocabulary (Negative Lexicon)
# =============================================================================

AI_FLAGGED_VOCABULARY = {
    "verbs": {
        "delve": ["explore", "examine", "look into", "investigate"],
        "leverage": ["use", "apply", "employ", "take advantage of"],
        "utilize": ["use", "apply", "employ"],
        "facilitate": ["help", "enable", "support", "make easier"],
        "embark": ["start", "begin", "launch", "undertake"],
        "underscore": ["highlight", "emphasize", "stress", "show"],
        "unleash": ["release", "enable", "unlock", "free"],
        "unlock": ["access", "gain", "achieve", "open"],
        "navigate": ["manage", "handle", "deal with", "work through"],
        "streamline": ["simplify", "improve", "speed up"],
        "optimize": ["improve", "enhance", "refine"],  # Only in excessive use
        "revolutionize": ["change", "transform", "improve"],
        "spearhead": ["lead", "direct", "head", "manage"],
        "foster": ["encourage", "support", "promote", "build"],
        "harness": ["use", "apply", "capture", "employ"],
    },
    "adjectives": {
        "robust": ["strong", "solid", "reliable", "durable"],
        "pivotal": ["key", "important", "central", "crucial"],
        "seamless": ["smooth", "easy", "simple", "effortless"],
        "seamlessly": ["smoothly", "easily", "simply", "well"],
        "intricate": ["complex", "detailed", "elaborate"],
        "multifaceted": ["complex", "varied", "diverse"],
        "cutting-edge": ["modern", "advanced", "latest", "new"],
        "dynamic": ["active", "changing", "flexible"],
        "bespoke": ["custom", "tailored", "personalized"],
        "holistic": ["complete", "comprehensive", "overall"],
        "groundbreaking": ["new", "innovative", "original"],
        "comprehensive": ["complete", "thorough", "full"],  # Overused
        "innovative": ["new", "creative", "original"],  # Overused
        "transformative": ["changing", "significant", "important"],
        "unparalleled": ["unique", "exceptional", "outstanding"],
        "meticulous": ["careful", "thorough", "detailed"],
    },
    "nouns": {
        "tapestry": ["mix", "blend", "combination", "variety"],
        "landscape": ["area", "field", "space", "market"],
        "realm": ["area", "field", "domain", "world"],
        "game-changer": ["important change", "breakthrough", "advance"],
        "treasure trove": ["collection", "source", "resource"],
        "paradigm shift": ["major change", "new approach", "shift"],
        "symphony": ["combination", "blend", "harmony"],
        "synergy": ["cooperation", "collaboration", "combination"],
        "cornerstone": ["foundation", "basis", "key part"],
        "culmination": ["result", "peak", "outcome"],
        "plethora": ["many", "lots of", "variety of"],
        "myriad": ["many", "numerous", "various"],
    },
    "phrases": {
        "it is important to note": ["note that", "importantly", "keep in mind"],
        "in conclusion": ["to summarize", "in summary", "finally"],
        "in today's digital world": ["today", "now", "currently"],
        "furthermore": ["also", "plus", "and", "additionally"],
        "moreover": ["also", "and", "plus"],
        "in this article": ["here", "below", "in this guide"],
        "without further ado": ["let's begin", "here's", "starting with"],
        "it goes without saying": ["clearly", "obviously", "of course"],
        "at the end of the day": ["ultimately", "finally", "in the end"],
        "first and foremost": ["first", "mainly", "primarily"],
        "last but not least": ["finally", "also", "and"],
        "needless to say": ["clearly", "obviously"],
        "that being said": ["however", "but", "still"],
    },
}


# =============================================================================
# Factual Preservation Patterns
# =============================================================================

PRESERVE_PATTERNS = [
    r"\d+\.?\d*%",  # Percentages: 5%, 3.5%
    r"\$[\d,]+\.?\d*",  # Currency: $1,000, $99.99
    r"€[\d,]+\.?\d*",  # Euro
    r"£[\d,]+\.?\d*",  # Pounds
    r"\d{4}",  # Years: 2024, 1999
    r'"[^"]{5,}"',  # Quoted text (5+ chars)
    r"'[^']{5,}'",  # Single-quoted text (5+ chars)
    r"\d+(?:,\d{3})+",  # Large numbers: 1,000,000
    r"\d+(?:\.\d+)?(?:\s*)?(?:million|billion|trillion)",  # Numbers with scale
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",  # Dates
    r"\d{1,2}/\d{1,2}/\d{2,4}",  # Date format: 01/15/2024
    r"\d{1,2}-\d{1,2}-\d{2,4}",  # Date format: 01-15-2024
]


# =============================================================================
# Safety Guardrails Class
# =============================================================================


@dataclass
class DensityCheck:
    """Result of a density check."""

    density: float
    is_safe: bool
    status: str  # "optimal", "warning", "danger"
    message: str
    count: int = 0  # Number of keyword occurrences


@dataclass
class VocabularyCheck:
    """Result of vocabulary filtering."""

    cleaned_text: str
    replacements: list[tuple[str, str]]  # (original, replacement)
    count_replaced: int


class SafetyGuardrails:
    """
    Safety system to prevent over-optimization penalties.

    This is the most critical module - it prevents:
    - Keyword stuffing penalties
    - AI detection flags
    - Factual drift
    - Monotonous content patterns
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize guardrails with configuration."""
        self.config = config
        self.warnings: list[GuardrailViolation] = []
        self.blocked: list[GuardrailViolation] = []

        # Compile preservation patterns for efficiency
        self._preserve_pattern = re.compile(
            "|".join(f"({p})" for p in PRESERVE_PATTERNS),
            re.IGNORECASE,
        )

    def reset(self) -> None:
        """Reset warnings and blocked lists."""
        self.warnings = []
        self.blocked = []

    # =========================================================================
    # Keyword Density Checks
    # =========================================================================

    def check_keyword_density(
        self, text: str, keyword: str, include_secondary: bool = False
    ) -> DensityCheck:
        """
        Check if keyword density is within safe limits.

        Args:
            text: The content text
            keyword: The keyword to check
            include_secondary: Include secondary keyword occurrences

        Returns:
            DensityCheck with status and message
        """
        if not text or not keyword:
            return DensityCheck(
                density=0.0, is_safe=True, status="optimal", message="No content or keyword", count=0
            )

        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # Count keyword occurrences
        keyword_count = text_lower.count(keyword_lower)

        # Calculate density
        word_count = len(text.split())
        if word_count == 0:
            return DensityCheck(
                density=0.0, is_safe=True, status="optimal", message="No words in text", count=0
            )

        keyword_words = len(keyword.split())
        density = (keyword_count * keyword_words / word_count) * 100

        # Determine status
        if density <= KEYWORD_DENSITY_OPTIMAL:
            return DensityCheck(
                density=density,
                is_safe=True,
                status="optimal",
                message=f"Keyword density ({density:.1f}%) is optimal",
                count=keyword_count,
            )
        elif density <= KEYWORD_DENSITY_WARNING:
            return DensityCheck(
                density=density,
                is_safe=True,
                status="optimal",
                message=f"Keyword density ({density:.1f}%) is acceptable",
                count=keyword_count,
            )
        elif density <= KEYWORD_DENSITY_DANGER:
            self.warnings.append(
                GuardrailViolation(
                    rule="keyword_density",
                    severity="warning",
                    message=f"High keyword density ({density:.1f}%) - risk of over-optimization",
                )
            )
            return DensityCheck(
                density=density,
                is_safe=True,
                status="warning",
                message=f"Warning: Keyword density ({density:.1f}%) is high",
                count=keyword_count,
            )
        else:
            self.blocked.append(
                GuardrailViolation(
                    rule="keyword_density",
                    severity="blocked",
                    message=f"Keyword stuffing detected ({density:.1f}%)",
                )
            )
            return DensityCheck(
                density=density,
                is_safe=False,
                status="danger",
                message=f"Blocked: Keyword stuffing ({density:.1f}%)",
                count=keyword_count,
            )

    def would_exceed_density(
        self, text: str, keyword: str, additional_count: int = 1
    ) -> bool:
        """
        Check if adding more keyword mentions would exceed safe density.

        Args:
            text: Current content
            keyword: The keyword
            additional_count: How many more mentions to add

        Returns:
            True if adding would exceed safe density
        """
        if not text or not keyword:
            return False

        text_lower = text.lower()
        keyword_lower = keyword.lower()

        current_count = text_lower.count(keyword_lower)
        word_count = len(text.split())

        if word_count == 0:
            return True

        keyword_words = len(keyword.split())
        new_density = ((current_count + additional_count) * keyword_words / word_count) * 100

        return new_density > self.config.max_keyword_density

    # =========================================================================
    # Entity Density Checks
    # =========================================================================

    def check_entity_density(
        self, text: str, entities: list[str]
    ) -> DensityCheck:
        """
        Check if entity density is within safe limits.

        Uses word-based density calculation (entity words / total words * 100).

        Args:
            text: The content text
            entities: List of entities to check

        Returns:
            DensityCheck with status and message
        """
        if not text or not entities:
            return DensityCheck(
                density=0.0, is_safe=True, status="optimal", message="No content or entities", count=0
            )

        text_lower = text.lower()
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            return DensityCheck(
                density=0.0, is_safe=True, status="optimal", message="No content", count=0
            )

        # Count entity occurrences and their word contribution
        total_entity_count = 0
        entity_word_count = 0
        for entity in entities:
            entity_lower = entity.lower()
            count = text_lower.count(entity_lower)
            total_entity_count += count
            entity_words = len(entity.split())
            entity_word_count += count * entity_words

        # Calculate density as percentage of words that are entity words
        density = (entity_word_count / total_words) * 100

        if density <= ENTITY_DENSITY_MAX:
            return DensityCheck(
                density=density,
                is_safe=True,
                status="optimal",
                message=f"Entity density ({density:.1f}%) is safe",
                count=total_entity_count,
            )
        else:
            self.warnings.append(
                GuardrailViolation(
                    rule="entity_density",
                    severity="warning",
                    message=f"High entity density ({density:.1f}%) - risk of entity stuffing",
                )
            )
            return DensityCheck(
                density=density,
                is_safe=False,
                status="danger",
                message=f"Entity stuffing risk ({density:.1f}%)",
                count=total_entity_count,
            )

    # =========================================================================
    # AI Vocabulary Filtering
    # =========================================================================

    def filter_ai_vocabulary(self, text: str) -> VocabularyCheck:
        """
        Remove/replace AI-flagged vocabulary with natural alternatives.

        Args:
            text: The text to filter

        Returns:
            VocabularyCheck with cleaned text and list of replacements
        """
        if not self.config.filter_ai_vocabulary:
            return VocabularyCheck(cleaned_text=text, replacements=[], count_replaced=0)

        cleaned = text
        replacements: list[tuple[str, str]] = []

        # Process phrases first (longer patterns)
        for phrase, alternatives in AI_FLAGGED_VOCABULARY["phrases"].items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(cleaned):
                replacement = alternatives[0]
                cleaned = pattern.sub(replacement, cleaned)
                replacements.append((phrase, replacement))

        # Process words by category
        for category in ["verbs", "adjectives", "nouns"]:
            for word, alternatives in AI_FLAGGED_VOCABULARY[category].items():
                # Match whole words only
                pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
                if pattern.search(cleaned):
                    # Preserve case of first letter
                    def replace_preserving_case(match: re.Match[str]) -> str:
                        original = match.group(0)
                        replacement = alternatives[0]
                        if original[0].isupper():
                            return replacement.capitalize()
                        return replacement

                    cleaned = pattern.sub(replace_preserving_case, cleaned)
                    replacements.append((word, alternatives[0]))

        return VocabularyCheck(
            cleaned_text=cleaned,
            replacements=replacements,
            count_replaced=len(replacements),
        )

    def get_replacement_word(self, ai_word: str) -> str | None:
        """
        Get a human-sounding replacement for an AI-flagged word.

        Args:
            ai_word: The flagged word

        Returns:
            Replacement word or None if not found
        """
        ai_word_lower = ai_word.lower()

        for category in ["verbs", "adjectives", "nouns"]:
            if ai_word_lower in AI_FLAGGED_VOCABULARY[category]:
                alternatives = AI_FLAGGED_VOCABULARY[category][ai_word_lower]
                return alternatives[0] if alternatives else None

        for phrase, alternatives in AI_FLAGGED_VOCABULARY["phrases"].items():
            if ai_word_lower in phrase.lower():
                return alternatives[0] if alternatives else None

        return None

    def contains_ai_vocabulary(self, text: str) -> list[str]:
        """
        Find all AI-flagged words in text.

        Args:
            text: Text to check

        Returns:
            List of flagged words found
        """
        found: list[str] = []
        text_lower = text.lower()

        # Check phrases
        for phrase in AI_FLAGGED_VOCABULARY["phrases"]:
            if phrase in text_lower:
                found.append(phrase)

        # Check words
        for category in ["verbs", "adjectives", "nouns"]:
            for word in AI_FLAGGED_VOCABULARY[category]:
                if re.search(rf"\b{re.escape(word)}\b", text_lower):
                    found.append(word)

        return found

    # =========================================================================
    # Sentence Variance (Burstiness)
    # =========================================================================

    def check_sentence_variance(self, sentences: list[str]) -> tuple[bool, str, float]:
        """
        Check if sentence lengths have sufficient variance (burstiness).

        AI-generated text tends to have monotonous rhythm;
        human text has natural variance in sentence lengths.

        Args:
            sentences: List of sentences

        Returns:
            Tuple of (is_acceptable, message, variance_ratio)
        """
        if len(sentences) < 3:
            return True, "Insufficient sentences for variance check", 0.0

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]

        # Filter out very short sentences (fragments)
        lengths = [l for l in lengths if l >= MIN_SENTENCE_LENGTH]

        if len(lengths) < 3:
            return True, "Insufficient valid sentences", 0.0

        mean_length = statistics.mean(lengths)
        if mean_length == 0:
            return True, "No valid sentence lengths", 0.0

        std_dev = statistics.stdev(lengths)
        variance_ratio = std_dev / mean_length

        if variance_ratio >= SENTENCE_VARIANCE_MIN:
            return True, f"Good sentence variance ({variance_ratio:.2f})", variance_ratio
        else:
            self.warnings.append(
                GuardrailViolation(
                    rule="sentence_variance",
                    severity="warning",
                    message=f"Low sentence variance ({variance_ratio:.2f}) - may appear AI-generated",
                )
            )
            return (
                False,
                f"Low sentence variance ({variance_ratio:.2f}) - needs more variety",
                variance_ratio,
            )

    def suggest_variance_improvements(self, sentences: list[str]) -> list[str]:
        """
        Suggest which sentences to split or combine for better variance.

        Args:
            sentences: List of sentences

        Returns:
            List of suggestions
        """
        suggestions: list[str] = []
        lengths = [(i, len(s.split()), s) for i, s in enumerate(sentences)]

        # Find sentences that are too similar in length
        mean_length = statistics.mean([l for _, l, _ in lengths]) if lengths else 0

        for i, length, sentence in lengths:
            if length > MAX_SENTENCE_LENGTH:
                suggestions.append(f"Split long sentence ({length} words): '{sentence[:50]}...'")
            elif abs(length - mean_length) < 2 and length > 10:
                # Sentence is very close to mean - might want variety
                short_preview = sentence[:40] + "..." if len(sentence) > 40 else sentence
                suggestions.append(
                    f"Consider varying sentence {i + 1} ({length} words near mean): '{short_preview}'"
                )

        return suggestions

    # =========================================================================
    # Factual Content Preservation
    # =========================================================================

    def extract_factual_content(self, text: str) -> list[tuple[str, int, int]]:
        """
        Extract factual content that should be preserved.

        Args:
            text: The text to analyze

        Returns:
            List of (content, start_pos, end_pos) tuples
        """
        facts: list[tuple[str, int, int]] = []

        for match in self._preserve_pattern.finditer(text):
            facts.append((match.group(0), match.start(), match.end()))

        return facts

    def preserve_factual_content(self, original: str, optimized: str) -> str:
        """
        Ensure statistics, quotes, and dates are unchanged.

        If any factual content was accidentally modified, restore it.

        Args:
            original: Original text
            optimized: Optimized text

        Returns:
            Text with factual content restored
        """
        if not self.config.preserve_statistics and not self.config.preserve_quotes:
            return optimized

        original_facts = self.extract_factual_content(original)

        if not original_facts:
            return optimized

        result = optimized

        # Patterns that often replace factual content
        vague_replacements = {
            "great": None,  # "85%" -> "great"
            "recently": None,  # "in 2020" -> "recently"
            "competitive": None,  # "$299.99" -> "competitive"
            "significant": None,
            "substantial": None,
            "considerable": None,
            "notable": None,
        }

        # Check each factual item
        for fact, fact_type, _ in original_facts:
            if fact not in result:
                # Try to restore the fact
                restored = False

                # Find context in original around the fact
                fact_idx = original.find(fact)
                if fact_idx >= 0:
                    # Get surrounding context (10 chars before and after)
                    start = max(0, fact_idx - 10)
                    end = min(len(original), fact_idx + len(fact) + 10)
                    context = original[start:end]

                    # Look for vague words in optimized that replaced this fact
                    for vague_word in vague_replacements:
                        if vague_word in result.lower():
                            # Replace the vague word with the fact
                            pattern = re.compile(rf'\b{vague_word}\b', re.IGNORECASE)
                            if pattern.search(result):
                                result = pattern.sub(fact, result, count=1)
                                restored = True
                                break

                if not restored:
                    # Couldn't restore - warn the user
                    self.warnings.append(
                        GuardrailViolation(
                            rule="factual_preservation",
                            severity="warning",
                            message=f"Factual content modified: '{fact}'",
                        )
                    )

        return result

    def validate_factual_integrity(self, original: str, optimized: str) -> list[str]:
        """
        Validate that factual content wasn't changed.

        Args:
            original: Original text
            optimized: Optimized text

        Returns:
            List of violation messages
        """
        violations: list[str] = []

        original_facts = set(f[0] for f in self.extract_factual_content(original))
        optimized_facts = set(f[0] for f in self.extract_factual_content(optimized))

        # Check for removed facts
        removed = original_facts - optimized_facts
        for fact in removed:
            violations.append(f"Factual content removed: '{fact}'")

        # Check for modified facts (same type, different value)
        # This is harder to detect without semantic understanding
        # For now, we just report removals

        return violations

    # =========================================================================
    # Change Validation
    # =========================================================================

    def validate_change(
        self, change: OptimizationChange, current_text: str
    ) -> tuple[bool, str]:
        """
        Final gate before applying any change.

        Args:
            change: The proposed change
            current_text: Current document text

        Returns:
            Tuple of (is_allowed, reason_if_blocked)
        """
        # Check if change would cause keyword stuffing
        if change.change_type.value == "keyword" and self.config.primary_keyword:
            # Count how many keyword instances the change adds
            original_count = change.original.lower().count(
                self.config.primary_keyword.lower()
            )
            optimized_count = change.optimized.lower().count(
                self.config.primary_keyword.lower()
            )
            additional = optimized_count - original_count

            if additional > 0 and self.would_exceed_density(
                current_text, self.config.primary_keyword, additional
            ):
                return False, "Would exceed keyword density limit"

        # Check for factual content modification
        if self.config.preserve_statistics or self.config.preserve_quotes:
            violations = self.validate_factual_integrity(
                change.original, change.optimized
            )
            if violations:
                return False, f"Factual content modified: {violations[0]}"

        # Check for AI vocabulary in optimized text
        if self.config.filter_ai_vocabulary:
            ai_words = self.contains_ai_vocabulary(change.optimized)
            if ai_words:
                return False, f"Contains AI-flagged vocabulary: {', '.join(ai_words[:3])}"

        return True, ""

    def check_over_optimization(
        self,
        original_analysis: AnalysisResult,
        optimized_analysis: AnalysisResult,
    ) -> list[str]:
        """
        Compare before/after to detect if optimization went too far.

        Args:
            original_analysis: Analysis of original content
            optimized_analysis: Analysis of optimized content

        Returns:
            List of warning messages
        """
        warnings: list[str] = []

        # Check keyword density increase
        if hasattr(original_analysis, "geo_score") and hasattr(
            optimized_analysis, "geo_score"
        ):
            orig_seo = original_analysis.geo_score.seo_score
            opt_seo = optimized_analysis.geo_score.seo_score

            if hasattr(orig_seo, "keyword_analysis") and hasattr(
                opt_seo, "keyword_analysis"
            ):
                orig_density = orig_seo.keyword_analysis.primary_density
                opt_density = opt_seo.keyword_analysis.primary_density

                if opt_density > KEYWORD_DENSITY_WARNING:
                    warnings.append(
                        f"Keyword density increased to {opt_density:.1f}% "
                        f"(was {orig_density:.1f}%)"
                    )

        # Check readability degradation
        if hasattr(original_analysis, "geo_score") and hasattr(
            optimized_analysis, "geo_score"
        ):
            orig_read = original_analysis.geo_score.readability_score.total
            opt_read = optimized_analysis.geo_score.readability_score.total

            if opt_read < orig_read - 10:  # More than 10 point drop
                warnings.append(
                    f"Readability decreased significantly: {orig_read:.1f} → {opt_read:.1f}"
                )

        return warnings

    # =========================================================================
    # Summary Methods
    # =========================================================================

    def get_all_violations(self) -> list[GuardrailViolation]:
        """Get all violations (warnings + blocked)."""
        return self.warnings + self.blocked

    def get_summary(self) -> str:
        """Get human-readable summary of all guardrail activity."""
        lines = ["Guardrail Summary", "=" * 40]

        if not self.warnings and not self.blocked:
            lines.append("No violations detected.")
            return "\n".join(lines)

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:  # Limit to first 10
                lines.append(f"  - [{w.rule}] {w.message}")

        if self.blocked:
            lines.append(f"\nBlocked ({len(self.blocked)}):")
            for b in self.blocked[:10]:  # Limit to first 10
                lines.append(f"  - [{b.rule}] {b.message}")

        return "\n".join(lines)

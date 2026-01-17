"""
Readability Scorer - Readability & UX Metrics (20% of GEO Score)

Evaluates:
- Average sentence length (optimal: 15-20 words)
- Active voice ratio
- Flesch-Kincaid grade level (optimal: 8-12)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from seo_optimizer.ingestion.models import DocumentAST

from .models import (
    Issue,
    IssueCategory,
    IssueSeverity,
    ReadabilityScore,
)

# Flesch-Kincaid thresholds
FK_GRADE_OPTIMAL_MIN = 8.0
FK_GRADE_OPTIMAL_MAX = 12.0
FK_GRADE_ACCEPTABLE_MIN = 6.0
FK_GRADE_ACCEPTABLE_MAX = 14.0

# Sentence length thresholds (words)
SENTENCE_LENGTH_OPTIMAL = 20
SENTENCE_LENGTH_ACCEPTABLE = 25
SENTENCE_LENGTH_MAX = 35

# Active voice target
ACTIVE_VOICE_TARGET = 0.80  # 80% active voice


@dataclass
class ReadabilityScorerConfig:
    """Configuration for readability scoring."""

    # Sentence length thresholds
    optimal_sentence_length: int = SENTENCE_LENGTH_OPTIMAL
    acceptable_sentence_length: int = SENTENCE_LENGTH_ACCEPTABLE
    max_sentence_length: int = SENTENCE_LENGTH_MAX

    # Flesch-Kincaid grade thresholds
    fk_grade_optimal_min: float = FK_GRADE_OPTIMAL_MIN
    fk_grade_optimal_max: float = FK_GRADE_OPTIMAL_MAX
    fk_grade_acceptable_min: float = FK_GRADE_ACCEPTABLE_MIN
    fk_grade_acceptable_max: float = FK_GRADE_ACCEPTABLE_MAX

    # Active voice target
    active_voice_target: float = ACTIVE_VOICE_TARGET

    # Score weights
    sentence_length_weight: float = 0.30
    active_voice_weight: float = 0.40
    grade_level_weight: float = 0.30


class ReadabilityScorer:
    """
    Scores content on readability and user experience metrics.

    Contributes 20% to the total GEO score.
    """

    # Passive voice patterns
    PASSIVE_PATTERNS = [
        # "to be" + past participle patterns
        r"\b(is|are|was|were|been|being|be)\s+(\w+ed)\b",
        r"\b(is|are|was|were|been|being|be)\s+(\w+en)\b",  # written, taken, etc.
        # "get" passive
        r"\b(gets?|got|getting)\s+(\w+ed)\b",
        # Common irregular past participles
        r"\b(is|are|was|were|been|being|be)\s+(made|done|said|taken|written|given|found|thought|told|known|shown|left|brought|put)\b",
    ]

    # Words that look passive but aren't (false positives)
    NOT_PASSIVE = {
        "interested", "excited", "bored", "tired", "relaxed",
        "concerned", "worried", "married", "related", "associated",
        "used", "supposed", "allowed", "required", "based",
    }

    def __init__(self, config: ReadabilityScorerConfig | None = None) -> None:
        """Initialize the readability scorer."""
        self.config = config or ReadabilityScorerConfig()

    def score(self, ast: DocumentAST) -> ReadabilityScore:
        """
        Calculate readability score for a document.

        Args:
            ast: The document AST

        Returns:
            ReadabilityScore with breakdown and issues
        """
        full_text = ast.full_text
        if not full_text or not full_text.strip():
            return ReadabilityScore(
                avg_sentence_length=0,
                active_voice_ratio=1.0,
                flesch_kincaid_grade=0,
                total=0,
            )

        # Extract sentences
        sentences = self._extract_sentences(full_text)

        # Calculate metrics
        avg_sentence_length = self._calculate_avg_sentence_length(sentences)
        active_voice_ratio, passive_sentences = self._calculate_active_voice_ratio(sentences)
        flesch_kincaid_grade = self._calculate_flesch_kincaid(full_text)

        # Find complex sentences
        complex_sentences = self._find_complex_sentences(sentences)

        # Calculate total score
        total = self._calculate_total_score(
            avg_sentence_length=avg_sentence_length,
            active_voice_ratio=active_voice_ratio,
            flesch_kincaid_grade=flesch_kincaid_grade,
        )

        # Collect issues
        issues = self._collect_issues(
            avg_sentence_length=avg_sentence_length,
            active_voice_ratio=active_voice_ratio,
            flesch_kincaid_grade=flesch_kincaid_grade,
            complex_sentences=complex_sentences,
            passive_sentences=passive_sentences,
        )

        return ReadabilityScore(
            avg_sentence_length=avg_sentence_length,
            active_voice_ratio=active_voice_ratio,
            flesch_kincaid_grade=flesch_kincaid_grade,
            total=total,
            complex_sentences=complex_sentences,
            passive_sentences=passive_sentences,
            issues=issues,
        )

    def _extract_sentences(self, text: str) -> list[str]:
        """
        Extract sentences from text.

        Args:
            text: The input text

        Returns:
            List of sentences
        """
        # Split on sentence boundaries
        # Handle common abbreviations to avoid false splits
        text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Inc|Ltd|Co)\.", r"\1<PERIOD>", text)
        text = re.sub(r"(\d)\.", r"\1<PERIOD>", text)  # Numbers like "1."

        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+\s*", text)

        # Restore periods
        sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]

        return sentences

    def _calculate_avg_sentence_length(self, sentences: list[str]) -> float:
        """
        Calculate average sentence length in words.

        Args:
            sentences: List of sentences

        Returns:
            Average words per sentence
        """
        if not sentences:
            return 0.0

        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def _calculate_active_voice_ratio(
        self, sentences: list[str]
    ) -> tuple[float, list[str]]:
        """
        Calculate ratio of active voice sentences.

        Args:
            sentences: List of sentences

        Returns:
            Tuple of (active_voice_ratio, passive_sentences)
        """
        if not sentences:
            return 1.0, []

        passive_sentences: list[str] = []

        for sentence in sentences:
            if self._is_passive_voice(sentence):
                passive_sentences.append(sentence)

        active_count = len(sentences) - len(passive_sentences)
        ratio = active_count / len(sentences) if sentences else 1.0

        return ratio, passive_sentences

    def _is_passive_voice(self, sentence: str) -> bool:
        """
        Check if a sentence is in passive voice.

        Args:
            sentence: The sentence to check

        Returns:
            True if passive voice detected
        """
        sentence_lower = sentence.lower()

        for pattern in self.PASSIVE_PATTERNS:
            match = re.search(pattern, sentence_lower)
            if match:
                # Check if it's a false positive
                participle = match.group(2) if len(match.groups()) > 1 else ""
                if participle not in self.NOT_PASSIVE:
                    return True

        return False

    def _calculate_flesch_kincaid(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.

        Formula:
        0.39 × (total words / total sentences) + 11.8 × (total syllables / total words) - 15.59

        Args:
            text: The input text

        Returns:
            Flesch-Kincaid grade level
        """
        sentences = self._extract_sentences(text)
        words = self._extract_words(text)

        if not sentences or not words:
            return 0.0

        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)

        if total_sentences == 0 or total_words == 0:
            return 0.0

        # Flesch-Kincaid Grade Level formula
        grade = (
            0.39 * (total_words / total_sentences)
            + 11.8 * (total_syllables / total_words)
            - 15.59
        )

        # Clamp to reasonable range
        return max(0, min(20, grade))

    def _extract_words(self, text: str) -> list[str]:
        """
        Extract words from text.

        Args:
            text: The input text

        Returns:
            List of words (alphabetic only)
        """
        # Extract only alphabetic words
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in words if len(w) > 0]

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word.

        Uses a heuristic approach based on vowel patterns.

        Args:
            word: The word to analyze

        Returns:
            Estimated syllable count
        """
        word = word.lower()
        if len(word) <= 2:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Adjust for silent 'e' at end
        if word.endswith("e") and count > 1:
            count -= 1

        # Adjust for common suffixes
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        # Ensure at least one syllable
        return max(1, count)

    def _find_complex_sentences(self, sentences: list[str]) -> list[str]:
        """
        Find sentences that exceed the complexity threshold.

        Args:
            sentences: List of sentences

        Returns:
            List of complex/long sentences
        """
        complex_sentences: list[str] = []

        for sentence in sentences:
            word_count = len(sentence.split())
            if word_count > self.config.max_sentence_length:
                complex_sentences.append(sentence)

        return complex_sentences

    def _calculate_total_score(
        self,
        avg_sentence_length: float,
        active_voice_ratio: float,
        flesch_kincaid_grade: float,
    ) -> float:
        """
        Calculate weighted total readability score.

        Args:
            avg_sentence_length: Average words per sentence
            active_voice_ratio: Ratio of active voice (0-1)
            flesch_kincaid_grade: FK grade level

        Returns:
            Total score (0-100)
        """
        # Sentence length score
        if avg_sentence_length <= self.config.optimal_sentence_length:
            length_score = 100.0
        elif avg_sentence_length <= self.config.acceptable_sentence_length:
            # Linear decrease from 100 to 80
            excess = avg_sentence_length - self.config.optimal_sentence_length
            max_excess = self.config.acceptable_sentence_length - self.config.optimal_sentence_length
            length_score = 100 - (excess / max_excess) * 20
        elif avg_sentence_length <= self.config.max_sentence_length:
            # Linear decrease from 80 to 60
            excess = avg_sentence_length - self.config.acceptable_sentence_length
            max_excess = self.config.max_sentence_length - self.config.acceptable_sentence_length
            length_score = 80 - (excess / max_excess) * 20
        else:
            # Linear decrease below 60
            excess = avg_sentence_length - self.config.max_sentence_length
            length_score = max(0, 60 - excess * 2)

        # Active voice score
        voice_score = active_voice_ratio * 100

        # Grade level score
        if self.config.fk_grade_optimal_min <= flesch_kincaid_grade <= self.config.fk_grade_optimal_max:
            grade_score = 100.0
        elif self.config.fk_grade_acceptable_min <= flesch_kincaid_grade <= self.config.fk_grade_acceptable_max:
            grade_score = 80.0
        elif flesch_kincaid_grade < self.config.fk_grade_acceptable_min:
            # Too simple
            grade_score = 70.0
        else:
            # Too complex
            excess = flesch_kincaid_grade - self.config.fk_grade_acceptable_max
            grade_score = max(0, 80 - excess * 5)

        # Weighted total
        total = (
            length_score * self.config.sentence_length_weight
            + voice_score * self.config.active_voice_weight
            + grade_score * self.config.grade_level_weight
        )

        return min(100, max(0, total))

    def _collect_issues(
        self,
        avg_sentence_length: float,
        active_voice_ratio: float,
        flesch_kincaid_grade: float,
        complex_sentences: list[str],
        passive_sentences: list[str],
    ) -> list[Issue]:
        """Collect readability issues."""
        issues: list[Issue] = []

        # Sentence length issues
        if avg_sentence_length > self.config.max_sentence_length:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Average sentence length ({avg_sentence_length:.1f} words) exceeds maximum ({self.config.max_sentence_length})",
                    current_value=f"{avg_sentence_length:.1f} words",
                    target_value=f"<= {self.config.optimal_sentence_length} words",
                    fix_suggestion="Break long sentences into shorter, focused statements",
                )
            )
        elif avg_sentence_length > self.config.acceptable_sentence_length:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Sentence length ({avg_sentence_length:.1f} words) above optimal ({self.config.optimal_sentence_length})",
                    current_value=f"{avg_sentence_length:.1f} words",
                    target_value=f"<= {self.config.optimal_sentence_length} words",
                    fix_suggestion="Consider shortening some sentences for better readability",
                )
            )

        # Report specific complex sentences
        for i, sentence in enumerate(complex_sentences[:3]):  # Top 3
            word_count = len(sentence.split())
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Complex sentence ({word_count} words)",
                    location=f"Sentence {i + 1}",
                    current_value=sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    fix_suggestion="Break into multiple sentences",
                )
            )

        # Active voice issues
        if active_voice_ratio < 0.5:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Low active voice usage ({active_voice_ratio:.0%})",
                    current_value=f"{active_voice_ratio:.0%}",
                    target_value=f">= {self.config.active_voice_target:.0%}",
                    fix_suggestion="Rewrite passive sentences using active voice",
                )
            )
        elif active_voice_ratio < self.config.active_voice_target:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Active voice ratio ({active_voice_ratio:.0%}) below target ({self.config.active_voice_target:.0%})",
                    current_value=f"{active_voice_ratio:.0%}",
                    target_value=f">= {self.config.active_voice_target:.0%}",
                    fix_suggestion="Convert some passive sentences to active voice",
                )
            )

        # Report specific passive sentences
        for i, sentence in enumerate(passive_sentences[:3]):  # Top 3
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message="Passive voice detected",
                    location=f"Sentence {i + 1}",
                    current_value=sentence[:80] + "..." if len(sentence) > 80 else sentence,
                    fix_suggestion="Rewrite using active voice",
                )
            )

        # Grade level issues
        if flesch_kincaid_grade > self.config.fk_grade_acceptable_max:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Reading level too high (Grade {flesch_kincaid_grade:.1f})",
                    current_value=f"Grade {flesch_kincaid_grade:.1f}",
                    target_value=f"Grade {self.config.fk_grade_optimal_min}-{self.config.fk_grade_optimal_max}",
                    fix_suggestion="Use simpler words and shorter sentences",
                )
            )
        elif flesch_kincaid_grade > self.config.fk_grade_optimal_max:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Reading level slightly high (Grade {flesch_kincaid_grade:.1f})",
                    current_value=f"Grade {flesch_kincaid_grade:.1f}",
                    target_value=f"Grade {self.config.fk_grade_optimal_min}-{self.config.fk_grade_optimal_max}",
                    fix_suggestion="Consider simplifying vocabulary where appropriate",
                )
            )
        elif flesch_kincaid_grade < self.config.fk_grade_acceptable_min:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Reading level may be too simple (Grade {flesch_kincaid_grade:.1f})",
                    current_value=f"Grade {flesch_kincaid_grade:.1f}",
                    target_value=f"Grade {self.config.fk_grade_optimal_min}-{self.config.fk_grade_optimal_max}",
                    fix_suggestion="Content may benefit from more sophisticated vocabulary",
                )
            )

        return issues


def score_readability(ast: DocumentAST) -> ReadabilityScore:
    """
    Convenience function to score readability without class instantiation.

    Args:
        ast: Document AST

    Returns:
        ReadabilityScore with breakdown and issues
    """
    scorer = ReadabilityScorer()
    return scorer.score(ast)

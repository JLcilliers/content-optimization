"""
Redundancy Resolver - Duplicate Content Detection and Resolution

Responsibilities:
- Detect duplicate/similar sentences
- Identify repetitive phrasing
- Consolidate redundant sections
- Maintain content uniqueness

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType

from .guardrails import SafetyGuardrails
from .models import ChangeType, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    pass


# =============================================================================
# Redundancy Detection Constants
# =============================================================================

# Similarity thresholds
HIGH_SIMILARITY_THRESHOLD = 0.85  # Nearly identical
MEDIUM_SIMILARITY_THRESHOLD = 0.70  # Very similar
LOW_SIMILARITY_THRESHOLD = 0.50  # Somewhat similar

# Minimum sentence length to consider for redundancy
MIN_SENTENCE_LENGTH = 30  # Characters

# Common phrases to ignore in duplication check
COMMON_TRANSITIONAL_PHRASES = [
    "in addition",
    "furthermore",
    "moreover",
    "however",
    "therefore",
    "for example",
    "in conclusion",
    "as mentioned",
    "as a result",
    "in other words",
]


@dataclass
class RedundancyMatch:
    """A pair of redundant content."""

    first_text: str
    second_text: str
    first_location: str
    second_location: str
    similarity: float
    redundancy_type: str  # "exact", "paraphrase", "repetitive"


@dataclass
class RedundancyAnalysis:
    """Result of redundancy analysis."""

    matches: list[RedundancyMatch] = field(default_factory=list)
    redundancy_score: float = 0.0  # 0-1, higher = more redundant
    unique_content_ratio: float = 1.0
    repeated_phrases: list[str] = field(default_factory=list)


class RedundancyResolver:
    """
    Detects and resolves content redundancy.

    Identifies:
    - Duplicate sentences
    - Repetitive phrasing
    - Over-used terms
    - Similar paragraphs
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the redundancy resolver."""
        self.config = config
        self.guardrails = guardrails

    def analyze(self, ast: DocumentAST) -> RedundancyAnalysis:
        """
        Analyze content for redundancy.

        Args:
            ast: Document AST

        Returns:
            RedundancyAnalysis with findings
        """
        analysis = RedundancyAnalysis()

        # Extract all sentences
        sentences = self._extract_all_sentences(ast)

        # Find duplicate sentences
        duplicates = self._find_duplicate_sentences(sentences)
        analysis.matches.extend(duplicates)

        # Compute full_text from nodes if not set
        full_text = ast.full_text
        if not full_text:
            full_text = " ".join(node.text_content for node in ast.nodes if node.text_content)

        # Find repetitive phrases
        analysis.repeated_phrases = self._find_repeated_phrases(full_text)

        # Calculate redundancy score
        analysis.redundancy_score = self._calculate_redundancy_score(
            len(duplicates), len(sentences), len(analysis.repeated_phrases)
        )

        # Calculate unique content ratio
        analysis.unique_content_ratio = 1.0 - analysis.redundancy_score

        return analysis

    def resolve(
        self,
        ast: DocumentAST,
        analysis: RedundancyAnalysis | None = None,
    ) -> list[OptimizationChange]:
        """
        Resolve redundancy issues.

        Args:
            ast: Document AST
            analysis: Pre-computed redundancy analysis

        Returns:
            List of optimization changes
        """
        if analysis is None:
            analysis = self.analyze(ast)

        changes: list[OptimizationChange] = []

        # Handle duplicate sentences
        duplicate_changes = self._resolve_duplicates(ast, analysis.matches)
        changes.extend(duplicate_changes)

        # Handle repetitive phrases
        phrase_changes = self._resolve_repeated_phrases(ast, analysis.repeated_phrases)
        changes.extend(phrase_changes)

        return changes

    def _extract_all_sentences(
        self, ast: DocumentAST
    ) -> list[tuple[str, str, str]]:
        """
        Extract all sentences with their locations.

        Args:
            ast: Document AST

        Returns:
            List of (sentence, node_id, context) tuples
        """
        sentences: list[tuple[str, str, str]] = []

        for node in ast.nodes:
            if node.node_type != NodeType.PARAGRAPH:
                continue

            text = node.text_content
            node_sentences = self._split_sentences(text)

            for sentence in node_sentences:
                if len(sentence) >= MIN_SENTENCE_LENGTH:
                    sentences.append((sentence, node.node_id, text[:50]))

        return sentences

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_duplicate_sentences(
        self, sentences: list[tuple[str, str, str]]
    ) -> list[RedundancyMatch]:
        """
        Find duplicate or very similar sentences.

        Args:
            sentences: List of (sentence, node_id, context) tuples

        Returns:
            List of redundancy matches
        """
        matches: list[RedundancyMatch] = []
        seen_pairs: set[tuple[int, int]] = set()

        for i, (sent1, node1, ctx1) in enumerate(sentences):
            for j, (sent2, node2, ctx2) in enumerate(sentences[i + 1 :], i + 1):
                # Skip if same node
                if node1 == node2:
                    continue

                # Skip if already processed
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))

                # Calculate similarity
                similarity = self._calculate_similarity(sent1, sent2)

                if similarity >= MEDIUM_SIMILARITY_THRESHOLD:
                    redundancy_type = "exact" if similarity >= HIGH_SIMILARITY_THRESHOLD else "paraphrase"

                    matches.append(
                        RedundancyMatch(
                            first_text=sent1,
                            second_text=sent2,
                            first_location=ctx1,
                            second_location=ctx2,
                            similarity=similarity,
                            redundancy_type=redundancy_type,
                        )
                    )

        return matches

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0-1
        """
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Use sequence matcher
        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _find_repeated_phrases(self, text: str) -> list[str]:
        """
        Find phrases that are repeated excessively.

        Args:
            text: Full document text

        Returns:
            List of repeated phrases
        """
        repeated: list[str] = []

        # Extract n-grams (3-5 words)
        words = text.lower().split()

        for n in range(3, 6):
            ngrams: dict[str, int] = {}

            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])

                # Skip common transitional phrases
                if any(
                    phrase in ngram for phrase in COMMON_TRANSITIONAL_PHRASES
                ):
                    continue

                # Skip if contains only common words
                if self._is_common_phrase(ngram):
                    continue

                ngrams[ngram] = ngrams.get(ngram, 0) + 1

            # Find excessive repetitions (3+ times for longer phrases)
            threshold = 3 if n >= 4 else 4

            for phrase, count in ngrams.items():
                if count >= threshold and phrase not in repeated:
                    repeated.append(phrase)

        return repeated[:10]  # Limit to top 10

    def _is_common_phrase(self, phrase: str) -> bool:
        """
        Check if phrase is too common to be considered redundant.

        Args:
            phrase: The phrase to check

        Returns:
            True if common
        """
        common_patterns = [
            r"^(the|a|an|this|that|these|those)\s+",
            r"^(is|are|was|were|be|been)\s+",
            r"^(and|or|but|so|yet)\s+",
            r"\s+(and|or|of|to|in|for|with|on|at)\s*$",
        ]

        for pattern in common_patterns:
            if re.search(pattern, phrase):
                return True

        return False

    def _calculate_redundancy_score(
        self, duplicate_count: int, total_sentences: int, phrase_count: int
    ) -> float:
        """
        Calculate overall redundancy score.

        Args:
            duplicate_count: Number of duplicate pairs
            total_sentences: Total sentence count
            phrase_count: Number of repeated phrases

        Returns:
            Score 0-1
        """
        if total_sentences == 0:
            return 0.0

        # Duplicate contribution (max 0.5)
        dup_score = min(duplicate_count / max(total_sentences / 10, 1), 1.0) * 0.5

        # Phrase contribution (max 0.5)
        phrase_score = min(phrase_count / 10, 1.0) * 0.5

        return min(dup_score + phrase_score, 1.0)

    def _resolve_duplicates(
        self, ast: DocumentAST, matches: list[RedundancyMatch]
    ) -> list[OptimizationChange]:
        """
        Generate changes to resolve duplicate content.

        Args:
            ast: Document AST
            matches: Redundancy matches

        Returns:
            List of changes
        """
        changes: list[OptimizationChange] = []

        for match in matches[:self.config.max_changes_per_section]:
            # Decide which to keep (keep first, modify second)
            if match.redundancy_type == "exact":
                # Suggest removal
                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.STRUCTURE,
                        location=f"Near: {match.second_location}...",
                        original=match.second_text[:80] + "...",
                        optimized="[Consider removing duplicate]",
                        reason=f"Duplicate content detected ({match.similarity:.0%} similar)",
                        impact_score=2.0,
                        full_original=match.second_text,
                        full_optimized="",  # Removal suggestion, no replacement text
                    )
                )
            else:
                # Suggest rewording
                reworded = self._reword_sentence(match.second_text, match.first_text)

                if reworded and reworded != match.second_text:
                    changes.append(
                        OptimizationChange(
                            change_type=ChangeType.READABILITY,
                            location=f"Near: {match.second_location}...",
                            original=match.second_text[:80] + "...",
                            optimized=reworded[:80] + "...",
                            reason=f"Reworded to reduce similarity ({match.similarity:.0%} similar)",
                            impact_score=1.5,
                            full_original=match.second_text,
                            full_optimized=reworded,
                        )
                    )

        return changes

    def _reword_sentence(self, sentence: str, similar_to: str) -> str | None:
        """
        Reword a sentence to reduce similarity.

        Args:
            sentence: Sentence to reword
            similar_to: Sentence it's similar to

        Returns:
            Reworded sentence or None
        """
        # Simple rewording strategies

        # Strategy 1: Change sentence structure (active/passive)
        if re.search(r"\b(is|are|was|were)\s+\w+ed\b", sentence):
            # Try converting from passive to active
            # This is simplified - real implementation would use NLP
            pass

        # Strategy 2: Use synonyms for key words
        # Find words that are in both sentences
        words1 = set(self._normalize_text(sentence).split())
        words2 = set(self._normalize_text(similar_to).split())
        common = words1 & words2

        if len(common) > len(words1) * 0.5:
            # Too much overlap - suggest significant rewrite
            return None

        # Strategy 3: Change connector words
        connectors = {
            "however": "yet",
            "therefore": "as a result",
            "furthermore": "additionally",
            "moreover": "also",
            "because": "since",
        }

        result = sentence
        for old, new in connectors.items():
            pattern = rf"\b{old}\b"
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, new, result, count=1, flags=re.IGNORECASE)
                break

        return result if result != sentence else None

    def _resolve_repeated_phrases(
        self, ast: DocumentAST, phrases: list[str]
    ) -> list[OptimizationChange]:
        """
        Generate changes to resolve repeated phrases.

        Args:
            ast: Document AST
            phrases: Repeated phrases

        Returns:
            List of changes
        """
        changes: list[OptimizationChange] = []

        for phrase in phrases[:3]:  # Limit to top 3 repetitive phrases
            count = ast.full_text.lower().count(phrase)

            if count >= 3:
                changes.append(
                    OptimizationChange(
                        change_type=ChangeType.READABILITY,
                        location="Document-wide",
                        original=f"'{phrase}' (appears {count} times)",
                        optimized="Consider varying this phrase",
                        reason="Repetitive phrasing reduces content quality",
                        impact_score=1.0,
                    )
                )

        return changes

    def get_uniqueness_score(self, ast: DocumentAST) -> float:
        """
        Get content uniqueness score.

        Args:
            ast: Document AST

        Returns:
            Uniqueness score 0-1 (higher = more unique)
        """
        analysis = self.analyze(ast)
        return analysis.unique_content_ratio

    def find_similar_paragraphs(
        self, ast: DocumentAST, threshold: float = MEDIUM_SIMILARITY_THRESHOLD
    ) -> list[tuple[ContentNode, ContentNode, float]]:
        """
        Find paragraphs that are similar to each other.

        Args:
            ast: Document AST
            threshold: Similarity threshold

        Returns:
            List of (para1, para2, similarity) tuples
        """
        similar: list[tuple[ContentNode, ContentNode, float]] = []
        paragraphs = [n for n in ast.nodes if n.node_type == NodeType.PARAGRAPH]

        for i, para1 in enumerate(paragraphs):
            for para2 in paragraphs[i + 1 :]:
                similarity = self._calculate_similarity(
                    para1.text_content, para2.text_content
                )

                if similarity >= threshold:
                    similar.append((para1, para2, similarity))

        return similar

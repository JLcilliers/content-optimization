"""
AI Scorer - AI Compatibility Metrics (30% of GEO Score)

Evaluates:
- Chunk clarity (self-contained segments, minimal pronoun dependencies)
- BLUF compliance (Bottom Line Up Front - answer first)
- Extraction friendliness (lists, tables, structured data)
- Redundancy penalty (>0.90 similarity sections)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from seo_optimizer.diffing.semantic import SemanticMatcher
from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .models import (
    AIScore,
    Issue,
    IssueCategory,
    IssueSeverity,
)

# Pronouns that indicate context dependency (not self-contained)
CONTEXT_DEPENDENT_PRONOUNS = {
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "them", "their", "his", "her",
    "here", "there", "above", "below", "previous", "following",
}

# Sentence starters that indicate good BLUF compliance
BLUF_POSITIVE_STARTERS = [
    "the answer is",
    "in short",
    "simply put",
    "the key point is",
    "the main",
    "to summarize",
    "the bottom line",
    "here's what",
    "the quick answer",
]

# Redundancy threshold
REDUNDANCY_THRESHOLD = 0.90


@dataclass
class AIScorerConfig:
    """Configuration for AI compatibility scoring."""

    # Chunk clarity thresholds
    max_pronoun_ratio: float = 0.10  # Max 10% pronoun tokens
    min_chunk_length: int = 50  # Minimum characters for a valid chunk

    # BLUF thresholds
    bluf_check_first_sentences: int = 2  # Check first N sentences

    # Extraction friendliness targets
    min_lists_for_bonus: int = 1
    min_tables_for_bonus: int = 1

    # Redundancy
    redundancy_threshold: float = REDUNDANCY_THRESHOLD

    # Score weights
    chunk_clarity_weight: float = 0.30
    answer_completeness_weight: float = 0.40
    extraction_friendliness_weight: float = 0.30


class AIScorer:
    """
    Scores content on AI/LLM compatibility metrics.

    Contributes 30% to the total GEO score.

    Focuses on:
    1. Can AI extract this as a standalone chunk?
    2. Does it answer questions directly (BLUF)?
    3. Is it structured for easy extraction?
    4. Is there redundant content that wastes context?
    """

    def __init__(
        self,
        config: AIScorerConfig | None = None,
        semantic_matcher: SemanticMatcher | None = None,
    ) -> None:
        """Initialize the AI scorer."""
        self.config = config or AIScorerConfig()
        self._semantic_matcher = semantic_matcher

    @property
    def semantic_matcher(self) -> SemanticMatcher:
        """Lazy load the semantic matcher."""
        if self._semantic_matcher is None:
            self._semantic_matcher = SemanticMatcher()
        return self._semantic_matcher

    def score(self, ast: DocumentAST) -> AIScore:
        """
        Calculate AI compatibility score for a document.

        Args:
            ast: The document AST

        Returns:
            AIScore with breakdown and issues
        """
        # Analyze chunk clarity
        chunk_clarity, problematic_chunks = self._analyze_chunk_clarity(ast)

        # Analyze BLUF compliance
        answer_completeness = self._analyze_bluf_compliance(ast)

        # Analyze extraction friendliness
        extraction_friendliness = self._analyze_extraction_friendliness(ast)

        # Detect redundant sections
        redundant_sections = self._detect_redundancy(ast)
        redundancy_penalty = self._calculate_redundancy_penalty(redundant_sections)

        # Calculate total score
        total = self._calculate_total_score(
            chunk_clarity=chunk_clarity,
            answer_completeness=answer_completeness,
            extraction_friendliness=extraction_friendliness,
            redundancy_penalty=redundancy_penalty,
        )

        # Collect issues
        issues = self._collect_issues(
            chunk_clarity=chunk_clarity,
            answer_completeness=answer_completeness,
            extraction_friendliness=extraction_friendliness,
            problematic_chunks=problematic_chunks,
            redundant_sections=redundant_sections,
            _ast=ast,
        )

        return AIScore(
            chunk_clarity=chunk_clarity,
            answer_completeness=answer_completeness,
            extraction_friendliness=extraction_friendliness,
            redundancy_penalty=redundancy_penalty,
            total=total,
            problematic_chunks=problematic_chunks,
            redundant_sections=redundant_sections,
            issues=issues,
        )

    def _analyze_chunk_clarity(
        self, ast: DocumentAST
    ) -> tuple[float, list[str]]:
        """
        Analyze whether content chunks are self-contained.

        A self-contained chunk:
        - Doesn't rely heavily on pronouns referring to previous context
        - Makes sense when read in isolation
        - Has clear subject references

        Args:
            ast: The document AST

        Returns:
            Tuple of (clarity_score 0-1, list of problematic chunk texts)
        """
        chunks = self._extract_chunks(ast)
        if not chunks:
            return 1.0, []

        problematic: list[str] = []
        clear_count = 0.0

        for chunk in chunks:
            if len(chunk) < self.config.min_chunk_length:
                continue

            pronoun_ratio = self._calculate_pronoun_ratio(chunk)

            if pronoun_ratio > self.config.max_pronoun_ratio:
                # Check if first sentence starts with a pronoun
                first_sentence = self._get_first_sentence(chunk)
                if self._starts_with_pronoun(first_sentence):
                    problematic.append(chunk[:100] + "..." if len(chunk) > 100 else chunk)
                else:
                    # High pronouns but doesn't start with one - less severe
                    clear_count += 0.5
            else:
                clear_count += 1

        total_chunks = len([c for c in chunks if len(c) >= self.config.min_chunk_length])
        if total_chunks == 0:
            return 1.0, []

        clarity_score = clear_count / total_chunks
        return clarity_score, problematic

    def _extract_chunks(self, ast: DocumentAST) -> list[str]:
        """Extract content chunks from AST."""
        chunks: list[str] = []

        for node in ast.nodes:
            if (
                node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]
                and node.text_content
                and node.text_content.strip()
            ):
                chunks.append(node.text_content.strip())

        return chunks

    def _calculate_pronoun_ratio(self, text: str) -> float:
        """Calculate ratio of context-dependent pronouns to total words."""
        words = text.lower().split()
        if not words:
            return 0.0

        pronoun_count = sum(1 for w in words if w.strip(".,!?") in CONTEXT_DEPENDENT_PRONOUNS)
        return pronoun_count / len(words)

    def _get_first_sentence(self, text: str) -> str:
        """Extract the first sentence from text."""
        # Simple sentence boundary detection
        match = re.match(r"^[^.!?]+[.!?]", text)
        if match:
            return match.group(0)
        return text[:100] if len(text) > 100 else text

    def _starts_with_pronoun(self, text: str) -> bool:
        """Check if text starts with a context-dependent pronoun."""
        words = text.lower().split()
        if not words:
            return False
        first_word = words[0].strip(".,!?\"'")
        return first_word in CONTEXT_DEPENDENT_PRONOUNS

    def _analyze_bluf_compliance(self, ast: DocumentAST) -> float:
        """
        Analyze BLUF (Bottom Line Up Front) compliance.

        Checks if sections answer questions directly rather than
        building up to the answer.

        Args:
            ast: The document AST

        Returns:
            BLUF compliance score (0-1)
        """
        sections = self._get_sections(ast)
        if not sections:
            return 0.5  # No clear sections to evaluate

        bluf_compliant = 0

        for section_heading, section_content in sections:
            if self._section_is_bluf_compliant(section_heading, section_content):
                bluf_compliant += 1

        return bluf_compliant / len(sections) if sections else 0.5

    def _get_sections(self, ast: DocumentAST) -> list[tuple[str, str]]:
        """
        Extract sections as (heading, content) pairs.

        Returns:
            List of (heading_text, section_content) tuples
        """
        sections: list[tuple[str, str]] = []
        current_heading = "Introduction"
        current_content: list[str] = []

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                # Save previous section
                if current_content:
                    sections.append((current_heading, " ".join(current_content)))
                current_heading = node.text_content
                current_content = []
            elif node.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]:
                if node.text_content:
                    current_content.append(node.text_content)

        # Add final section
        if current_content:
            sections.append((current_heading, " ".join(current_content)))

        return sections

    def _section_is_bluf_compliant(self, heading: str, content: str) -> bool:
        """
        Check if a section follows BLUF principle.

        A BLUF-compliant section:
        - Starts with the key point/answer
        - Doesn't use excessive preamble
        - Gets to the point within first 1-2 sentences
        """
        if not content:
            return True  # Empty sections are fine

        first_sentences = self._get_first_n_sentences(
            content, self.config.bluf_check_first_sentences
        )

        # Check for positive BLUF indicators
        first_text_lower = first_sentences.lower()
        for starter in BLUF_POSITIVE_STARTERS:
            if starter in first_text_lower:
                return True

        # Check for negative patterns (excessive preamble)
        preamble_patterns = [
            r"^before we (begin|start|dive)",
            r"^in order to understand",
            r"^to fully appreciate",
            r"^let('s| us) (start|begin) (by|with)",
            r"^first, (let's|we need to|we should)",
        ]

        for pattern in preamble_patterns:
            if re.search(pattern, first_text_lower):
                return False

        # Check if heading question is answered in first sentences
        if self._is_question_heading(heading):
            # For question headings, check if answer is provided directly
            return self._contains_direct_answer(first_sentences)

        # Default: assume compliant if no red flags
        return True

    def _get_first_n_sentences(self, text: str, n: int) -> str:
        """Get first N sentences from text."""
        sentences = re.split(r"[.!?]+\s*", text)
        return ". ".join(sentences[:n]) + "." if sentences else text

    def _is_question_heading(self, heading: str) -> bool:
        """Check if heading is a question."""
        return heading.strip().endswith("?") or heading.lower().startswith(
            ("what ", "how ", "why ", "when ", "where ", "who ", "which ", "can ", "does ", "is ", "are ")
        )

    def _contains_direct_answer(self, text: str) -> bool:
        """Check if text contains a direct answer."""
        # Look for answer patterns
        answer_patterns = [
            r"^yes[,.]",
            r"^no[,.]",
            r"^the answer is",
            r"^you (can|should|need)",
            r"^it (is|means|refers)",
            r"^this (is|means|refers)",
            r"^[\w\s]+ is (the|a|an) ",  # Definition pattern with article
            r"^[\w\s]+ (is|are|means|refers) ",  # General definition pattern
        ]

        text_lower = text.lower().strip()
        return any(re.search(pattern, text_lower) for pattern in answer_patterns)

    def _analyze_extraction_friendliness(self, ast: DocumentAST) -> float:
        """
        Analyze how easy it is to extract structured information.

        Looks for:
        - Lists (bulleted, numbered)
        - Tables
        - Clear section structure
        - Code blocks

        Args:
            ast: The document AST

        Returns:
            Extraction friendliness score (0-1)
        """
        score = 0.0
        total_nodes = len(ast.nodes)

        if total_nodes == 0:
            return 0.0

        # Count structural elements
        list_count = len([n for n in ast.nodes if n.node_type == NodeType.LIST])
        table_count = len([n for n in ast.nodes if n.node_type == NodeType.TABLE])
        heading_count = len([n for n in ast.nodes if n.node_type == NodeType.HEADING])

        # Calculate ratios
        word_count = len(ast.full_text.split()) if ast.full_text else 1

        # Lists contribute up to 30%
        if list_count >= 3:
            score += 0.30
        elif list_count >= self.config.min_lists_for_bonus:
            score += 0.15 + (list_count / 6) * 0.15

        # Tables contribute up to 25%
        if table_count >= 2:
            score += 0.25
        elif table_count >= self.config.min_tables_for_bonus:
            score += 0.15

        # Heading structure contributes up to 25%
        headings_per_300 = (heading_count / (word_count / 300)) if word_count > 300 else heading_count
        if headings_per_300 >= 1.5:
            score += 0.25
        elif headings_per_300 >= 1.0:
            score += 0.20
        elif headings_per_300 >= 0.5:
            score += 0.10

        # Short, focused paragraphs contribute up to 20%
        paragraphs = [n for n in ast.nodes if n.node_type == NodeType.PARAGRAPH]
        if paragraphs:
            avg_para_length = sum(len(p.text_content.split()) for p in paragraphs) / len(paragraphs)
            if avg_para_length <= 75:  # Short, scannable paragraphs
                score += 0.20
            elif avg_para_length <= 100:
                score += 0.15
            elif avg_para_length <= 150:
                score += 0.10

        return min(1.0, score)

    def _detect_redundancy(
        self, ast: DocumentAST
    ) -> list[tuple[str, str, float]]:
        """
        Detect redundant sections with >0.90 similarity.

        Args:
            ast: The document AST

        Returns:
            List of (section1_id, section2_id, similarity) tuples
        """
        redundant: list[tuple[str, str, float]] = []

        # Get content nodes with substantial text
        content_nodes = [
            n for n in ast.nodes
            if n.node_type in [NodeType.PARAGRAPH, NodeType.LIST_ITEM]
            and len(n.text_content) > 50
        ]

        if len(content_nodes) < 2:
            return redundant

        # Compare pairs (limit to avoid O(n^2) for large docs)
        max_comparisons = 50
        comparison_count = 0

        for i in range(len(content_nodes)):
            for j in range(i + 1, len(content_nodes)):
                if comparison_count >= max_comparisons:
                    break

                node_i = content_nodes[i]
                node_j = content_nodes[j]

                try:
                    similarity = self.semantic_matcher.compute_similarity(
                        node_i.text_content, node_j.text_content
                    )

                    if similarity > self.config.redundancy_threshold:
                        redundant.append((
                            node_i.node_id,
                            node_j.node_id,
                            float(similarity),
                        ))
                    comparison_count += 1
                except Exception:
                    continue

            if comparison_count >= max_comparisons:
                break

        return redundant

    def _calculate_redundancy_penalty(
        self, redundant_sections: list[tuple[str, str, float]]
    ) -> float:
        """
        Calculate redundancy penalty factor.

        Args:
            redundant_sections: List of redundant section pairs

        Returns:
            Penalty factor (0-1, where 1 means severe penalty)
        """
        if not redundant_sections:
            return 0.0

        # Each redundant pair adds to penalty
        # Cap at 0.30 (30% max penalty)
        penalty = min(len(redundant_sections) * 0.05, 0.30)
        return penalty

    def _calculate_total_score(
        self,
        chunk_clarity: float,
        answer_completeness: float,
        extraction_friendliness: float,
        redundancy_penalty: float,
    ) -> float:
        """
        Calculate weighted total AI score.

        Args:
            chunk_clarity: Chunk clarity score (0-1)
            answer_completeness: BLUF compliance score (0-1)
            extraction_friendliness: Extraction friendliness score (0-1)
            redundancy_penalty: Redundancy penalty (0-1)

        Returns:
            Total score (0-100)
        """
        base_score = (
            chunk_clarity * 100 * self.config.chunk_clarity_weight
            + answer_completeness * 100 * self.config.answer_completeness_weight
            + extraction_friendliness * 100 * self.config.extraction_friendliness_weight
        )

        # Apply redundancy penalty
        penalized_score = base_score * (1 - redundancy_penalty)

        return min(100, max(0, penalized_score))

    def _collect_issues(
        self,
        chunk_clarity: float,
        answer_completeness: float,
        extraction_friendliness: float,
        problematic_chunks: list[str],
        redundant_sections: list[tuple[str, str, float]],
        _ast: DocumentAST,
    ) -> list[Issue]:
        """Collect AI compatibility issues."""
        issues: list[Issue] = []

        # Chunk clarity issues
        if chunk_clarity < 0.5:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Low chunk clarity ({chunk_clarity:.0%}) - content relies heavily on context",
                    current_value=f"{chunk_clarity:.0%}",
                    target_value=">= 70%",
                    fix_suggestion="Replace pronouns with explicit references; make sections self-contained",
                )
            )
        elif chunk_clarity < 0.7:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Moderate chunk clarity ({chunk_clarity:.0%})",
                    current_value=f"{chunk_clarity:.0%}",
                    target_value=">= 70%",
                    fix_suggestion="Reduce pronoun usage in paragraph openings",
                )
            )

        # Specific problematic chunks
        for i, chunk in enumerate(problematic_chunks[:3]):  # Report top 3
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.INFO,
                    message="Paragraph starts with context-dependent reference",
                    location=f"Paragraph {i + 1}",
                    current_value=chunk[:50] + "..." if len(chunk) > 50 else chunk,
                    fix_suggestion="Start with the subject rather than a pronoun",
                )
            )

        # BLUF compliance issues
        if answer_completeness < 0.5:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Low BLUF compliance ({answer_completeness:.0%}) - sections don't answer directly",
                    current_value=f"{answer_completeness:.0%}",
                    target_value=">= 70%",
                    fix_suggestion="Put the key point/answer at the beginning of each section",
                )
            )
        elif answer_completeness < 0.7:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Moderate BLUF compliance ({answer_completeness:.0%})",
                    current_value=f"{answer_completeness:.0%}",
                    target_value=">= 70%",
                    fix_suggestion="Consider leading with the conclusion before the explanation",
                )
            )

        # Extraction friendliness issues
        if extraction_friendliness < 0.3:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Low extraction friendliness ({extraction_friendliness:.0%})",
                    current_value=f"{extraction_friendliness:.0%}",
                    target_value=">= 50%",
                    fix_suggestion="Add lists, tables, or structured formatting to improve scannability",
                )
            )
        elif extraction_friendliness < 0.5:
            issues.append(
                Issue(
                    category=IssueCategory.AI_COMPATIBILITY,
                    severity=IssueSeverity.INFO,
                    message=f"Moderate extraction friendliness ({extraction_friendliness:.0%})",
                    current_value=f"{extraction_friendliness:.0%}",
                    target_value=">= 50%",
                    fix_suggestion="Consider adding bullet points or numbered lists where appropriate",
                )
            )

        # Redundancy issues
        for sec1, sec2, sim in redundant_sections[:3]:
            issues.append(
                Issue(
                    category=IssueCategory.REDUNDANCY,
                    severity=IssueSeverity.WARNING,
                    message=f"Redundant content wastes AI context ({sim:.0%} similar)",
                    location=f"{sec1} and {sec2}",
                    current_value=f"{sim:.0%} similarity",
                    target_value="< 90% similarity",
                    fix_suggestion="Consolidate duplicate information or differentiate content",
                )
            )

        return issues


def score_ai_compatibility(ast: DocumentAST) -> AIScore:
    """
    Convenience function to score AI compatibility without class instantiation.

    Args:
        ast: Document AST

    Returns:
        AIScore with breakdown and issues
    """
    scorer = AIScorer()
    return scorer.score(ast)

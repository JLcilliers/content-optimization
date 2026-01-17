"""
Semantic Scorer - Semantic Depth Metrics (30% of GEO Score)

Evaluates:
- Topic coverage via cosine similarity (embeddings)
- Information gain (unique entities ratio)
- Entity density
- Redundancy detection (>0.90 similarity = duplicate)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
Thresholds:
- >= 0.85 cosine similarity = "Excellent coverage"
- > 0.90 similarity between sections = Redundancy penalty
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from seo_optimizer.diffing.semantic import SemanticMatcher
from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .entity_extractor import EntityExtractor
from .models import (
    EntityMatch,
    Issue,
    IssueCategory,
    IssueSeverity,
    SemanticScore,
)

if TYPE_CHECKING:
    pass


# Research-derived thresholds
TOPIC_COVERAGE_EXCELLENT = 0.85
TOPIC_COVERAGE_GOOD = 0.70
TOPIC_COVERAGE_FAIR = 0.50

REDUNDANCY_THRESHOLD = 0.90  # > 0.90 similarity = redundant sections

ENTITY_SATURATION_THRESHOLD = 0.05  # > 5% entity density = over-optimization


@dataclass
class SemanticScorerConfig:
    """Configuration for semantic scoring."""

    # Topic coverage thresholds
    excellent_coverage: float = TOPIC_COVERAGE_EXCELLENT
    good_coverage: float = TOPIC_COVERAGE_GOOD
    fair_coverage: float = TOPIC_COVERAGE_FAIR

    # Redundancy threshold
    redundancy_threshold: float = REDUNDANCY_THRESHOLD

    # Entity saturation threshold
    entity_saturation_threshold: float = ENTITY_SATURATION_THRESHOLD

    # Score weights
    topic_coverage_weight: float = 0.50
    information_gain_weight: float = 0.30
    entity_density_weight: float = 0.20


class SemanticScorer:
    """
    Scores content on semantic depth and topical coverage.

    Contributes 30% to the total GEO score.
    """

    def __init__(
        self,
        config: SemanticScorerConfig | None = None,
        entity_extractor: EntityExtractor | None = None,
        semantic_matcher: SemanticMatcher | None = None,
    ) -> None:
        """
        Initialize the semantic scorer.

        Args:
            config: Scoring configuration
            entity_extractor: Entity extractor instance (lazy loaded if None)
            semantic_matcher: Semantic matcher instance (lazy loaded if None)
        """
        self.config = config or SemanticScorerConfig()
        self._entity_extractor = entity_extractor
        self._semantic_matcher = semantic_matcher

    @property
    def entity_extractor(self) -> EntityExtractor:
        """Lazy load the entity extractor."""
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor()
        return self._entity_extractor

    @property
    def semantic_matcher(self) -> SemanticMatcher:
        """Lazy load the semantic matcher."""
        if self._semantic_matcher is None:
            self._semantic_matcher = SemanticMatcher()
        return self._semantic_matcher

    def score(
        self,
        ast: DocumentAST,
        topic_description: str | None = None,
        expected_entities: list[str] | None = None,
        competitor_content: list[str] | None = None,
    ) -> SemanticScore:
        """
        Calculate semantic score for a document.

        Args:
            ast: The document AST
            topic_description: Description of the expected topic/theme
            expected_entities: List of expected semantic entities
            competitor_content: Content from competitor pages for comparison

        Returns:
            SemanticScore with breakdown and issues
        """
        full_text = ast.full_text

        # Extract entities
        entities = self.entity_extractor.extract_entities_with_concepts(full_text)
        unique_entities = self.entity_extractor.get_unique_entities(entities)

        # Calculate topic coverage
        topic_coverage = self._calculate_topic_coverage(
            full_text, topic_description, expected_entities
        )

        # Calculate information gain
        information_gain = self._calculate_information_gain(
            unique_entities, expected_entities, competitor_content
        )

        # Calculate entity density
        entity_density = self._calculate_entity_density(full_text, entities)
        entity_saturation = entity_density > self.config.entity_saturation_threshold

        # Detect redundant sections
        redundant_sections = self._detect_redundancy(ast)

        # Find missing expected entities
        missing_entities: list[str] = []
        if expected_entities:
            _, missing_entities = self.entity_extractor.match_expected_entities(
                entities, expected_entities
            )

        # Calculate total score
        total = self._calculate_total_score(
            topic_coverage=topic_coverage,
            information_gain=information_gain,
            entity_density=entity_density,
            entity_saturation=entity_saturation,
            redundancy_count=len(redundant_sections),
        )

        # Collect issues
        issues = self._collect_issues(
            topic_coverage=topic_coverage,
            information_gain=information_gain,
            entity_density=entity_density,
            entity_saturation=entity_saturation,
            missing_entities=missing_entities,
            redundant_sections=redundant_sections,
        )

        return SemanticScore(
            topic_coverage=topic_coverage,
            information_gain=information_gain,
            entity_density=entity_density,
            entity_saturation=entity_saturation,
            total=total,
            entities_found=entities,
            missing_entities=missing_entities,
            redundant_sections=redundant_sections,
            issues=issues,
        )

    def _calculate_topic_coverage(
        self,
        text: str,
        topic_description: str | None,
        expected_entities: list[str] | None,
    ) -> float:
        """
        Calculate topic coverage score using semantic similarity.

        Args:
            text: The document text
            topic_description: Expected topic description
            expected_entities: Expected entities to cover

        Returns:
            Coverage score (0.0 to 1.0)
        """
        if not topic_description and not expected_entities:
            # No reference provided - assume good coverage
            return 0.75

        reference_text = topic_description or ""
        if expected_entities:
            reference_text += " " + " ".join(expected_entities)

        if not reference_text.strip():
            return 0.75

        try:
            similarity = self.semantic_matcher.compute_similarity(text, reference_text)
            return float(similarity)
        except Exception:
            # Fallback to simple overlap check
            return self._simple_coverage_check(text, expected_entities or [])

    def _simple_coverage_check(self, text: str, expected: list[str]) -> float:
        """Simple fallback coverage check using word overlap."""
        if not expected:
            return 0.75

        text_lower = text.lower()
        found = sum(1 for e in expected if e.lower() in text_lower)
        return found / len(expected) if expected else 0.75

    def _calculate_information_gain(
        self,
        unique_entities: list[EntityMatch],
        expected_entities: list[str] | None,
        competitor_content: list[str] | None,
    ) -> float:
        """
        Calculate information gain score.

        This measures how much unique/valuable information the content provides.

        Args:
            unique_entities: Unique entities found in the document
            expected_entities: Expected entities
            competitor_content: Competitor content for comparison

        Returns:
            Information gain score (0.0 to 1.0)
        """
        if not unique_entities:
            return 0.0

        # Calculate base entity richness
        entity_richness = min(len(unique_entities) / 20, 1.0)  # Cap at 20 entities

        # Calculate coverage of expected entities
        expected_coverage = 0.0
        if expected_entities:
            found, _ = self.entity_extractor.match_expected_entities(
                unique_entities, expected_entities
            )
            expected_coverage = len(found) / len(expected_entities)

        # Calculate unique vs competitor
        unique_vs_competitor = 1.0
        if competitor_content:
            competitor_entities: set[str] = set()
            for content in competitor_content:
                comp_entities = self.entity_extractor.extract_entities(content)
                for e in comp_entities:
                    competitor_entities.add(e.text.lower())

            # Find entities unique to this content
            unique_to_content = [
                e for e in unique_entities
                if e.text.lower() not in competitor_entities
            ]
            unique_vs_competitor = (
                len(unique_to_content) / len(unique_entities)
                if unique_entities else 0.0
            )

        # Weighted average
        if expected_entities:
            return entity_richness * 0.3 + expected_coverage * 0.5 + unique_vs_competitor * 0.2
        else:
            return entity_richness * 0.5 + unique_vs_competitor * 0.5

    def _calculate_entity_density(
        self, text: str, entities: list[EntityMatch]
    ) -> float:
        """
        Calculate entity density as ratio of entity characters to total.

        Args:
            text: The document text
            entities: All extracted entities

        Returns:
            Entity density ratio (0.0 to 1.0)
        """
        if not text or not entities:
            return 0.0

        total_chars = len(text)
        entity_chars = sum(e.length for e in entities)

        return entity_chars / total_chars if total_chars > 0 else 0.0

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
            if n.node_type in [NodeType.PARAGRAPH, NodeType.HEADING]
            and len(n.text_content) > 50  # Skip very short content
        ]

        if len(content_nodes) < 2:
            return redundant

        # Compare each pair of nodes
        for i in range(len(content_nodes)):
            for j in range(i + 1, len(content_nodes)):
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
                except Exception:
                    # Skip if similarity computation fails
                    continue

        return redundant

    def _calculate_total_score(
        self,
        topic_coverage: float,
        information_gain: float,
        entity_density: float,
        entity_saturation: bool,
        redundancy_count: int,
    ) -> float:
        """
        Calculate weighted total semantic score.

        Args:
            topic_coverage: Topic coverage score (0-1)
            information_gain: Information gain score (0-1)
            entity_density: Entity density ratio (0-1)
            entity_saturation: Whether entity density is too high
            redundancy_count: Number of redundant section pairs

        Returns:
            Total score (0-100)
        """
        # Base score from components
        coverage_score = topic_coverage * 100 * self.config.topic_coverage_weight
        gain_score = information_gain * 100 * self.config.information_gain_weight

        # Entity density score (optimal around 2%)
        if entity_density < 0.01:
            density_score = (entity_density / 0.01) * 50
        elif entity_density <= 0.03:
            density_score = 100
        else:
            # Gradually decrease for over-saturation
            density_score = max(0, 100 - (entity_density - 0.03) * 1000)

        density_score *= self.config.entity_density_weight

        total = coverage_score + gain_score + density_score

        # Apply saturation penalty
        if entity_saturation:
            total *= 0.85  # 15% penalty

        # Apply redundancy penalty (5% per redundant pair, max 30%)
        redundancy_penalty = min(redundancy_count * 0.05, 0.30)
        total *= (1 - redundancy_penalty)

        return min(100, max(0, total))

    def _collect_issues(
        self,
        topic_coverage: float,
        information_gain: float,
        entity_density: float,
        entity_saturation: bool,
        missing_entities: list[str],
        redundant_sections: list[tuple[str, str, float]],
    ) -> list[Issue]:
        """Collect semantic-related issues."""
        issues: list[Issue] = []

        # Topic coverage issues
        if topic_coverage < self.config.fair_coverage:
            issues.append(
                Issue(
                    category=IssueCategory.ENTITY,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Poor topic coverage ({topic_coverage:.0%})",
                    current_value=f"{topic_coverage:.0%}",
                    target_value=f">= {self.config.excellent_coverage:.0%}",
                    fix_suggestion="Add more topic-relevant content and semantic entities",
                )
            )
        elif topic_coverage < self.config.good_coverage:
            issues.append(
                Issue(
                    category=IssueCategory.ENTITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Moderate topic coverage ({topic_coverage:.0%})",
                    current_value=f"{topic_coverage:.0%}",
                    target_value=f">= {self.config.excellent_coverage:.0%}",
                    fix_suggestion="Expand content with more related concepts and entities",
                )
            )
        elif topic_coverage < self.config.excellent_coverage:
            issues.append(
                Issue(
                    category=IssueCategory.ENTITY,
                    severity=IssueSeverity.INFO,
                    message=f"Good topic coverage ({topic_coverage:.0%}), room for improvement",
                    current_value=f"{topic_coverage:.0%}",
                    target_value=f">= {self.config.excellent_coverage:.0%}",
                    fix_suggestion="Consider adding more depth with additional semantic entities",
                )
            )

        # Missing entities
        if missing_entities:
            if len(missing_entities) > 3:
                issues.append(
                    Issue(
                        category=IssueCategory.ENTITY,
                        severity=IssueSeverity.WARNING,
                        message=f"Missing {len(missing_entities)} expected semantic entities",
                        current_value=", ".join(missing_entities[:5]) + ("..." if len(missing_entities) > 5 else ""),
                        fix_suggestion="Incorporate the missing entities naturally into the content",
                    )
                )
            else:
                issues.append(
                    Issue(
                        category=IssueCategory.ENTITY,
                        severity=IssueSeverity.INFO,
                        message=f"Missing {len(missing_entities)} expected semantic entities",
                        current_value=", ".join(missing_entities),
                        fix_suggestion="Consider adding these entities to improve coverage",
                    )
                )

        # Entity saturation
        if entity_saturation:
            issues.append(
                Issue(
                    category=IssueCategory.ENTITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Entity saturation detected ({entity_density:.1%} density)",
                    current_value=f"{entity_density:.1%}",
                    target_value=f"< {self.config.entity_saturation_threshold:.0%}",
                    fix_suggestion="Content may be over-optimized; consider more natural language",
                )
            )

        # Redundancy issues
        if redundant_sections:
            for sec1, sec2, sim in redundant_sections[:3]:  # Report top 3
                issues.append(
                    Issue(
                        category=IssueCategory.REDUNDANCY,
                        severity=IssueSeverity.WARNING,
                        message=f"Redundant content detected ({sim:.0%} similarity)",
                        location=f"{sec1} and {sec2}",
                        current_value=f"{sim:.0%} similarity",
                        target_value="< 90% similarity",
                        fix_suggestion="Consolidate or differentiate redundant sections",
                    )
                )

        # Information gain
        if information_gain < 0.3:
            issues.append(
                Issue(
                    category=IssueCategory.ENTITY,
                    severity=IssueSeverity.WARNING,
                    message="Low information gain - content may lack unique value",
                    current_value=f"{information_gain:.0%}",
                    target_value=">= 50%",
                    fix_suggestion="Add unique insights, data, or expert perspectives",
                )
            )

        return issues


def score_semantic(
    ast: DocumentAST,
    topic_description: str | None = None,
    expected_entities: list[str] | None = None,
) -> SemanticScore:
    """
    Convenience function to score semantics without class instantiation.

    Args:
        ast: Document AST
        topic_description: Description of expected topic
        expected_entities: List of expected semantic entities

    Returns:
        SemanticScore with breakdown and issues
    """
    scorer = SemanticScorer()
    return scorer.score(ast, topic_description, expected_entities)

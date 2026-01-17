"""
SEO Scorer - Traditional SEO Metrics (20% of GEO Score)

Evaluates:
- Keyword placement and density (1-3-5 rule)
- Heading structure (H1 uniqueness, hierarchy)
- Internal link readiness

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass

from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .models import (
    HeadingAnalysis,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordAnalysis,
    KeywordConfig,
    SEOScore,
)

# Optimal density range (as percentage)
KEYWORD_DENSITY_MIN = 1.0
KEYWORD_DENSITY_MAX = 3.0
KEYWORD_DENSITY_OPTIMAL = 2.0

# Heading density target (headings per 300 words)
HEADING_DENSITY_TARGET = 1.0

# Weight for internal link readiness component
LINK_WEIGHT = 0.2


@dataclass
class SEOScorerConfig:
    """Configuration for SEO scoring."""

    # Keyword density thresholds
    min_keyword_density: float = KEYWORD_DENSITY_MIN
    max_keyword_density: float = KEYWORD_DENSITY_MAX
    optimal_keyword_density: float = KEYWORD_DENSITY_OPTIMAL

    # Heading requirements
    require_single_h1: bool = True
    require_valid_hierarchy: bool = True
    heading_density_per_300_words: float = HEADING_DENSITY_TARGET

    # Link thresholds
    min_internal_links: int = 2
    max_internal_links_ratio: float = 0.05  # 5% of word count

    # Score weights
    keyword_weight: float = 0.4
    heading_weight: float = 0.4
    link_weight: float = 0.2


class SEOScorer:
    """
    Scores content on traditional SEO metrics.

    Contributes 20% to the total GEO score.
    """

    def __init__(self, config: SEOScorerConfig | None = None) -> None:
        """Initialize with optional configuration."""
        self.config = config or SEOScorerConfig()

    def score(
        self,
        ast: DocumentAST,
        keywords: KeywordConfig,
        title: str | None = None,
        url_slug: str | None = None,
    ) -> SEOScore:
        """
        Calculate SEO score for a document.

        Args:
            ast: The document AST
            keywords: Target keyword configuration
            title: Page title (meta title)
            url_slug: URL slug for the page

        Returns:
            SEOScore with breakdown and issues
        """
        # Analyze keywords
        keyword_analysis = self._analyze_keywords(
            ast=ast,
            keywords=keywords,
            title=title,
            url_slug=url_slug,
        )
        keyword_score = self._calculate_keyword_score(keyword_analysis)

        # Analyze headings
        heading_analysis = self._analyze_headings(ast)
        heading_score = self._calculate_heading_score(heading_analysis, ast)

        # Analyze link readiness
        link_score = self._calculate_link_score(ast)

        # Collect issues
        issues = self._collect_issues(keyword_analysis, heading_analysis, ast)

        # Calculate weighted total
        total = (
            keyword_score * self.config.keyword_weight
            + heading_score * self.config.heading_weight
            + link_score * self.config.link_weight
        )

        return SEOScore(
            keyword_score=keyword_score,
            heading_score=heading_score,
            link_readiness_score=link_score,
            total=total,
            keyword_analysis=keyword_analysis,
            heading_analysis=heading_analysis,
            issues=issues,
        )

    def _analyze_keywords(
        self,
        ast: DocumentAST,
        keywords: KeywordConfig,
        title: str | None = None,
        url_slug: str | None = None,
    ) -> KeywordAnalysis:
        """
        Analyze keyword presence and placement.

        Implements the 1-3-5 rule:
        - 1 primary keyword
        - 3 secondary keywords
        - 5 semantic entities
        """
        full_text = ast.full_text.lower()
        word_count = len(full_text.split())

        primary = keywords.primary_keyword.lower()
        primary_found = primary in full_text

        # Find primary keyword locations
        primary_locations: list[str] = []

        # Check title
        if title and primary in title.lower():
            primary_locations.append("title")

        # Check H1
        h1_nodes = [n for n in ast.nodes if n.node_type == NodeType.HEADING and n.metadata.get("level") == 1]
        for h1 in h1_nodes:
            if primary in h1.text_content.lower():
                primary_locations.append("h1")
                break

        # Check URL slug
        if url_slug and primary in url_slug.lower().replace("-", " "):
            primary_locations.append("url_slug")

        # Check first 100 words
        first_100_words = " ".join(full_text.split()[:100])
        if primary in first_100_words:
            primary_locations.append("first_100_words")

        # Check H2/H3
        subheading_nodes = [
            n for n in ast.nodes
            if n.node_type == NodeType.HEADING and n.metadata.get("level", 0) in [2, 3]
        ]
        for sub in subheading_nodes:
            if primary in sub.text_content.lower():
                primary_locations.append("h2_h3")
                break

        # Check body
        if primary in full_text:
            primary_locations.append("body")

        # Check alt text (if available in metadata)
        # This would require image alt text extraction from AST

        # Calculate keyword density
        primary_count = full_text.count(primary)
        primary_word_count = len(primary.split())
        keyword_density = (primary_count * primary_word_count / word_count * 100) if word_count > 0 else 0

        # Check secondary keywords
        secondary_found: list[str] = []
        for sec in keywords.secondary_keywords:
            if sec.lower() in full_text:
                secondary_found.append(sec)

        # Check semantic entities
        entities_found: list[str] = []
        for entity in keywords.semantic_entities:
            if entity.lower() in full_text:
                entities_found.append(entity)

        # Check 1-3-5 rule compliance
        passes_135_rule = (
            primary_found
            and len(secondary_found) >= min(3, len(keywords.secondary_keywords))
            and len(entities_found) >= min(5, len(keywords.semantic_entities))
        )

        return KeywordAnalysis(
            primary_keyword=keywords.primary_keyword,
            primary_found=primary_found,
            primary_locations=primary_locations,
            secondary_keywords=keywords.secondary_keywords,
            secondary_found=secondary_found,
            semantic_entities=keywords.semantic_entities,
            entities_found=entities_found,
            keyword_density=keyword_density,
            passes_135_rule=passes_135_rule,
        )

    def _calculate_keyword_score(self, analysis: KeywordAnalysis) -> float:
        """Calculate keyword component score (0-100)."""
        score = 0.0

        # Primary keyword presence (30 points)
        if analysis.primary_found:
            score += 30

        # Primary keyword placement (30 points)
        placement_score = analysis.primary_placement_score
        score += placement_score * 30

        # Secondary keywords (20 points)
        if analysis.secondary_keywords:
            sec_ratio = len(analysis.secondary_found) / len(analysis.secondary_keywords)
            score += sec_ratio * 20
        else:
            score += 20  # No secondary keywords specified

        # Keyword density (20 points)
        density = analysis.keyword_density
        if self.config.min_keyword_density <= density <= self.config.max_keyword_density:
            # Optimal range
            score += 20
        elif density < self.config.min_keyword_density:
            # Too low
            score += (density / self.config.min_keyword_density) * 15
        else:
            # Too high (over-optimization)
            over = density - self.config.max_keyword_density
            penalty = min(over * 5, 20)
            score += max(0, 20 - penalty)

        return min(100, score)

    def _analyze_headings(self, ast: DocumentAST) -> HeadingAnalysis:
        """Analyze document heading structure."""
        heading_nodes = [n for n in ast.nodes if n.node_type == NodeType.HEADING]

        h1_nodes = [n for n in heading_nodes if n.metadata.get("level") == 1]
        h1_count = len(h1_nodes)
        h1_text = h1_nodes[0].text_content if h1_nodes else None

        # Build headings list
        headings_list: list[tuple[int, str]] = []
        for node in heading_nodes:
            level = node.metadata.get("level", 1)
            headings_list.append((level, node.text_content))

        # Check hierarchy validity (no skipped levels)
        hierarchy_valid = True
        issues: list[str] = []
        prev_level = 0

        for level, text in headings_list:
            if prev_level > 0 and level > prev_level + 1:
                hierarchy_valid = False
                issues.append(f"Skipped heading level: H{prev_level} to H{level} at '{text[:30]}...'")
            prev_level = level

        # Calculate heading density
        word_count = len(ast.full_text.split()) if ast.full_text else 0
        heading_density = (len(heading_nodes) / (word_count / 300)) if word_count > 0 else 0

        if h1_count == 0:
            issues.append("Missing H1 heading")
        elif h1_count > 1:
            issues.append(f"Multiple H1 headings found ({h1_count})")

        return HeadingAnalysis(
            h1_count=h1_count,
            h1_text=h1_text,
            hierarchy_valid=hierarchy_valid,
            heading_density=heading_density,
            headings_list=headings_list,
            issues=issues,
        )

    def _calculate_heading_score(self, analysis: HeadingAnalysis, _ast: DocumentAST) -> float:
        """Calculate heading component score (0-100)."""
        score = 0.0

        # Single H1 (40 points)
        if analysis.has_valid_h1:
            score += 40
        elif analysis.h1_count > 1:
            score += 20  # Multiple H1s is better than none

        # Valid hierarchy (30 points)
        if analysis.hierarchy_valid:
            score += 30
        else:
            score += 15  # Partial credit for having structure

        # Heading density (30 points)
        # Target: 1 heading per 300 words
        if analysis.heading_density >= 0.8:
            score += 30
        elif analysis.heading_density >= 0.5:
            score += 20
        elif analysis.heading_density > 0:
            score += 10

        return min(100, score)

    def _calculate_link_score(self, ast: DocumentAST) -> float:
        """
        Calculate internal link readiness score (0-100).

        This measures how well the content is structured for internal linking.
        """
        score = 0.0
        word_count = len(ast.full_text.split()) if ast.full_text else 0

        # Count existing link indicators
        link_count = 0
        for node in ast.nodes:
            # Check for anchor references in node content
            # This is a simplified check - actual link detection would depend on AST structure
            if node.metadata.get("has_links"):
                link_count += node.metadata.get("link_count", 1)

        # Bonus for content length (more content = more linking opportunities)
        if word_count >= 1000:
            score += 30
        elif word_count >= 500:
            score += 20
        elif word_count >= 300:
            score += 10

        # Bonus for heading structure (good anchor targets)
        heading_count = len([n for n in ast.nodes if n.node_type == NodeType.HEADING])
        if heading_count >= 5:
            score += 30
        elif heading_count >= 3:
            score += 20
        elif heading_count >= 1:
            score += 10

        # Bonus for lists (good for scannable content)
        list_count = len([n for n in ast.nodes if n.node_type == NodeType.LIST])
        if list_count >= 2:
            score += 20
        elif list_count >= 1:
            score += 10

        # Existing links bonus
        if link_count >= self.config.min_internal_links:
            score += 20
        elif link_count > 0:
            score += 10

        return min(100, score)

    def _collect_issues(
        self,
        keyword_analysis: KeywordAnalysis,
        heading_analysis: HeadingAnalysis,
        ast: DocumentAST,
    ) -> list[Issue]:
        """Collect all SEO issues."""
        issues: list[Issue] = []

        # Keyword issues
        if not keyword_analysis.primary_found:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Primary keyword '{keyword_analysis.primary_keyword}' not found in content",
                    fix_suggestion=f"Add the primary keyword '{keyword_analysis.primary_keyword}' to the content, ideally in the first 100 words",
                )
            )

        if keyword_analysis.primary_found and "title" not in keyword_analysis.primary_locations:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message="Primary keyword missing from page title",
                    fix_suggestion="Include the primary keyword in the page title",
                )
            )

        if keyword_analysis.primary_found and "h1" not in keyword_analysis.primary_locations:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message="Primary keyword missing from H1 heading",
                    fix_suggestion="Include the primary keyword in the H1 heading",
                )
            )

        if keyword_analysis.primary_found and "first_100_words" not in keyword_analysis.primary_locations:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message="Primary keyword not in first 100 words",
                    fix_suggestion="Move the primary keyword to appear within the first 100 words",
                )
            )

        # Density issues
        if keyword_analysis.keyword_density < self.config.min_keyword_density:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.INFO,
                    message=f"Keyword density ({keyword_analysis.keyword_density:.1f}%) is below optimal ({self.config.min_keyword_density}%)",
                    current_value=f"{keyword_analysis.keyword_density:.1f}%",
                    target_value=f"{self.config.optimal_keyword_density}%",
                    fix_suggestion="Add more natural mentions of the primary keyword",
                )
            )
        elif keyword_analysis.keyword_density > self.config.max_keyword_density:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message=f"Keyword density ({keyword_analysis.keyword_density:.1f}%) exceeds maximum ({self.config.max_keyword_density}%)",
                    current_value=f"{keyword_analysis.keyword_density:.1f}%",
                    target_value=f"{self.config.max_keyword_density}%",
                    fix_suggestion="Reduce keyword usage to avoid over-optimization penalty",
                )
            )

        # Secondary keyword issues
        if keyword_analysis.secondary_keywords:
            missing_secondary = set(keyword_analysis.secondary_keywords) - set(keyword_analysis.secondary_found)
            if len(missing_secondary) > len(keyword_analysis.secondary_keywords) / 2:
                issues.append(
                    Issue(
                        category=IssueCategory.KEYWORD,
                        severity=IssueSeverity.INFO,
                        message=f"Missing {len(missing_secondary)} of {len(keyword_analysis.secondary_keywords)} secondary keywords",
                        current_value=", ".join(keyword_analysis.secondary_found[:3]) + ("..." if len(keyword_analysis.secondary_found) > 3 else ""),
                        target_value=", ".join(list(missing_secondary)[:3]) + ("..." if len(missing_secondary) > 3 else ""),
                        fix_suggestion="Incorporate missing secondary keywords naturally into the content",
                    )
                )

        # Heading issues
        if heading_analysis.h1_count == 0:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.CRITICAL,
                    message="Missing H1 heading",
                    fix_suggestion="Add an H1 heading that includes the primary keyword",
                )
            )
        elif heading_analysis.h1_count > 1:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.WARNING,
                    message=f"Multiple H1 headings found ({heading_analysis.h1_count})",
                    fix_suggestion="Use only one H1 heading per page",
                )
            )

        if not heading_analysis.hierarchy_valid:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.WARNING,
                    message="Heading hierarchy has skipped levels",
                    fix_suggestion="Ensure headings follow sequential order (H1 > H2 > H3)",
                )
            )

        # Heading density
        if heading_analysis.heading_density < 0.5:
            word_count = len(ast.full_text.split()) if ast.full_text else 0
            if word_count > 600:
                issues.append(
                    Issue(
                        category=IssueCategory.STRUCTURE,
                        severity=IssueSeverity.INFO,
                        message="Low heading density - content may be difficult to scan",
                        current_value=f"{heading_analysis.heading_density:.2f} per 300 words",
                        target_value="1.0 per 300 words",
                        fix_suggestion="Add more subheadings to break up content",
                    )
                )

        return issues


def score_seo(
    ast: DocumentAST,
    primary_keyword: str,
    secondary_keywords: list[str] | None = None,
    semantic_entities: list[str] | None = None,
    title: str | None = None,
    url_slug: str | None = None,
) -> SEOScore:
    """
    Convenience function to score SEO without class instantiation.

    Args:
        ast: Document AST
        primary_keyword: The main target keyword
        secondary_keywords: Optional secondary keywords
        semantic_entities: Optional semantic entities
        title: Page title
        url_slug: URL slug

    Returns:
        SEOScore with breakdown and issues
    """
    keywords = KeywordConfig(
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords or [],
        semantic_entities=semantic_entities or [],
    )
    scorer = SEOScorer()
    return scorer.score(ast, keywords, title, url_slug)

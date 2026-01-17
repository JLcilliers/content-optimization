"""
Recommendation Engine - Actionable Fixes

Generates prioritized, actionable recommendations based on:
- Detected issues
- GEO score components
- Content gaps
- Best practices

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import (
    GEOScore,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordConfig,
)


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationCategory(str, Enum):
    """Categories for recommendations."""

    QUICK_WIN = "quick_win"  # Easy to implement, high impact
    CONTENT_GAP = "content_gap"  # Missing content
    OPTIMIZATION = "optimization"  # Improve existing content
    STRUCTURE = "structure"  # Structural improvements
    TECHNICAL = "technical"  # Technical SEO


@dataclass
class Recommendation:
    """A single actionable recommendation."""

    title: str
    description: str
    priority: RecommendationPriority
    category: RecommendationCategory
    impact: str  # Expected impact description
    effort: str  # Effort estimation
    source_issue: Issue | None = None

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.priority.value.upper()}] {self.title}"


@dataclass
class RecommendationEngineConfig:
    """Configuration for recommendation generation."""

    max_recommendations: int = 10
    include_low_priority: bool = True
    include_quick_wins: bool = True


class RecommendationEngine:
    """
    Generates prioritized, actionable recommendations.

    Takes detected issues and GEO score to produce
    a ranked list of improvements.
    """

    def __init__(self, config: RecommendationEngineConfig | None = None) -> None:
        """Initialize the recommendation engine."""
        self.config = config or RecommendationEngineConfig()

    def generate(
        self,
        issues: list[Issue],
        geo_score: GEOScore,
        keywords: KeywordConfig | None = None,
    ) -> list[Recommendation]:
        """
        Generate prioritized recommendations.

        Args:
            issues: List of detected issues
            geo_score: The computed GEO score
            keywords: Target keyword configuration

        Returns:
            List of recommendations, sorted by priority and impact
        """
        recommendations: list[Recommendation] = []

        # Generate recommendations from issues
        recommendations.extend(self._from_issues(issues))

        # Generate recommendations from score gaps
        recommendations.extend(self._from_score_gaps(geo_score))

        # Generate quick wins
        if self.config.include_quick_wins:
            recommendations.extend(self._generate_quick_wins(geo_score, keywords))

        # Deduplicate and sort
        recommendations = self._deduplicate(recommendations)
        recommendations = self._sort_by_priority(recommendations)

        # Limit to max recommendations
        if not self.config.include_low_priority:
            recommendations = [
                r for r in recommendations
                if r.priority != RecommendationPriority.LOW
            ]

        return recommendations[: self.config.max_recommendations]

    def _from_issues(self, issues: list[Issue]) -> list[Recommendation]:
        """Generate recommendations from detected issues."""
        recommendations: list[Recommendation] = []

        for issue in issues:
            rec = self._issue_to_recommendation(issue)
            if rec:
                recommendations.append(rec)

        return recommendations

    def _issue_to_recommendation(self, issue: Issue) -> Recommendation | None:
        """Convert an issue to a recommendation."""
        if not issue.fix_suggestion:
            return None

        # Determine priority based on severity
        priority_map = {
            IssueSeverity.CRITICAL: RecommendationPriority.HIGH,
            IssueSeverity.WARNING: RecommendationPriority.MEDIUM,
            IssueSeverity.INFO: RecommendationPriority.LOW,
        }
        priority = priority_map.get(issue.severity, RecommendationPriority.LOW)

        # Determine category based on issue category
        category_map = {
            IssueCategory.STRUCTURE: RecommendationCategory.STRUCTURE,
            IssueCategory.KEYWORD: RecommendationCategory.OPTIMIZATION,
            IssueCategory.ENTITY: RecommendationCategory.CONTENT_GAP,
            IssueCategory.READABILITY: RecommendationCategory.OPTIMIZATION,
            IssueCategory.AI_COMPATIBILITY: RecommendationCategory.OPTIMIZATION,
            IssueCategory.REDUNDANCY: RecommendationCategory.OPTIMIZATION,
        }
        category = category_map.get(issue.category, RecommendationCategory.OPTIMIZATION)

        # Estimate impact and effort
        impact, effort = self._estimate_impact_effort(issue)

        return Recommendation(
            title=self._generate_title(issue),
            description=issue.fix_suggestion,
            priority=priority,
            category=category,
            impact=impact,
            effort=effort,
            source_issue=issue,
        )

    def _generate_title(self, issue: Issue) -> str:
        """Generate a concise title from an issue."""
        # Map common issue patterns to clear titles
        message_lower = issue.message.lower()

        if "missing h1" in message_lower:
            return "Add H1 Heading"
        elif "multiple h1" in message_lower:
            return "Fix Multiple H1 Headings"
        elif "primary keyword" in message_lower and "not found" in message_lower:
            return "Add Primary Keyword"
        elif "keyword density" in message_lower:
            if "below" in message_lower or "low" in message_lower:
                return "Increase Keyword Density"
            else:
                return "Reduce Keyword Density"
        elif "missing faq" in message_lower:
            return "Add FAQ Section"
        elif "thin content" in message_lower:
            return "Expand Content"
        elif "passive voice" in message_lower:
            return "Use Active Voice"
        elif "sentence length" in message_lower:
            return "Shorten Sentences"
        elif "reading level" in message_lower:
            return "Simplify Language"
        elif "topic coverage" in message_lower:
            return "Improve Topic Coverage"
        elif "chunk clarity" in message_lower:
            return "Make Sections Self-Contained"
        elif "bluf" in message_lower:
            return "Lead with Key Points"
        elif "redundant" in message_lower:
            return "Remove Duplicate Content"
        elif "meta description" in message_lower:
            return "Optimize Meta Description"
        elif "heading hierarchy" in message_lower:
            return "Fix Heading Structure"
        else:
            # Fallback: extract first few words
            words = issue.message.split()[:5]
            return " ".join(words).title()

    def _estimate_impact_effort(self, issue: Issue) -> tuple[str, str]:
        """Estimate impact and effort for an issue fix."""
        # High impact issues
        high_impact_keywords = [
            "primary keyword", "h1", "thin content", "missing",
            "topic coverage", "critical",
        ]
        # Low effort fixes
        low_effort_keywords = [
            "add", "include", "meta description", "alt text",
        ]

        message_lower = issue.message.lower()

        # Determine impact
        is_high_impact = any(kw in message_lower for kw in high_impact_keywords)
        if issue.severity == IssueSeverity.CRITICAL:
            impact = "High - Critical for SEO performance"
        elif is_high_impact:
            impact = "High - Significant ranking factor"
        elif issue.severity == IssueSeverity.WARNING:
            impact = "Medium - Improves content quality"
        else:
            impact = "Low - Minor optimization"

        # Determine effort
        is_low_effort = any(kw in message_lower for kw in low_effort_keywords)
        if is_low_effort and "expand" not in message_lower:
            effort = "Low - Quick fix"
        elif "rewrite" in message_lower or "restructure" in message_lower:
            effort = "High - Requires content rewrite"
        elif "thin content" in message_lower or "expand" in message_lower:
            effort = "High - Requires new content"
        else:
            effort = "Medium - Moderate editing"

        return impact, effort

    def _from_score_gaps(self, geo_score: GEOScore) -> list[Recommendation]:
        """Generate recommendations from score component gaps."""
        recommendations: list[Recommendation] = []
        threshold = 70  # Score threshold for generating recommendations

        # SEO Score gaps
        if geo_score.seo_score.total < threshold:
            if geo_score.seo_score.keyword_score < 60:
                recommendations.append(
                    Recommendation(
                        title="Improve Keyword Optimization",
                        description="Focus on primary keyword placement in title, H1, first paragraph, and throughout content.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.OPTIMIZATION,
                        impact="High - Direct ranking factor",
                        effort="Medium - Content review needed",
                    )
                )
            if geo_score.seo_score.heading_score < 60:
                recommendations.append(
                    Recommendation(
                        title="Restructure Headings",
                        description="Add proper H1-H3 hierarchy with keyword-rich subheadings every 300 words.",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.STRUCTURE,
                        impact="Medium - Improves scannability and SEO",
                        effort="Medium - Structural changes",
                    )
                )

        # Semantic Score gaps
        if geo_score.semantic_score.total < threshold:
            if geo_score.semantic_score.topic_coverage < 0.7:
                recommendations.append(
                    Recommendation(
                        title="Deepen Topic Coverage",
                        description="Add more related concepts, entities, and supporting information to establish topical authority.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.CONTENT_GAP,
                        impact="High - Establishes expertise",
                        effort="High - Research and writing",
                    )
                )
            if geo_score.semantic_score.missing_entities:
                recommendations.append(
                    Recommendation(
                        title="Add Missing Semantic Entities",
                        description=f"Include these missing entities: {', '.join(geo_score.semantic_score.missing_entities[:5])}",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.CONTENT_GAP,
                        impact="Medium - Improves relevance signals",
                        effort="Low - Targeted additions",
                    )
                )

        # AI Score gaps
        if geo_score.ai_score.total < threshold:
            if geo_score.ai_score.chunk_clarity < 0.7:
                recommendations.append(
                    Recommendation(
                        title="Improve Section Independence",
                        description="Make each section understandable on its own by replacing pronouns with specific references.",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.OPTIMIZATION,
                        impact="Medium - Better AI extraction",
                        effort="Medium - Careful editing",
                    )
                )
            if geo_score.ai_score.answer_completeness < 0.7:
                recommendations.append(
                    Recommendation(
                        title="Adopt BLUF Writing Style",
                        description="Put the key point or answer at the beginning of each section before the explanation.",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.OPTIMIZATION,
                        impact="Medium - Better AI summarization",
                        effort="Medium - Content restructuring",
                    )
                )
            if geo_score.ai_score.extraction_friendliness < 0.5:
                recommendations.append(
                    Recommendation(
                        title="Add Structured Elements",
                        description="Include bullet points, numbered lists, and tables to make content more extractable.",
                        priority=RecommendationPriority.LOW,
                        category=RecommendationCategory.QUICK_WIN,
                        impact="Medium - Improves scannability",
                        effort="Low - Easy formatting",
                    )
                )

        # Readability Score gaps
        if geo_score.readability_score.total < threshold:
            if geo_score.readability_score.avg_sentence_length > 25:
                recommendations.append(
                    Recommendation(
                        title="Shorten Sentences",
                        description="Break long sentences into shorter, clearer statements. Target 15-20 words per sentence.",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.OPTIMIZATION,
                        impact="Medium - Better readability",
                        effort="Medium - Editing required",
                    )
                )
            if geo_score.readability_score.active_voice_ratio < 0.7:
                recommendations.append(
                    Recommendation(
                        title="Increase Active Voice",
                        description="Rewrite passive sentences using active voice for more engaging content.",
                        priority=RecommendationPriority.LOW,
                        category=RecommendationCategory.OPTIMIZATION,
                        impact="Medium - More engaging content",
                        effort="Medium - Careful rewriting",
                    )
                )

        return recommendations

    def _generate_quick_wins(
        self,
        geo_score: GEOScore,
        keywords: KeywordConfig | None,
    ) -> list[Recommendation]:
        """Generate quick win recommendations."""
        quick_wins: list[Recommendation] = []

        # Check for easy SEO wins
        if keywords and geo_score.seo_score.keyword_analysis:
            kw_analysis = geo_score.seo_score.keyword_analysis

            if kw_analysis.primary_found and "title" not in kw_analysis.primary_locations:
                quick_wins.append(
                    Recommendation(
                        title="Add Keyword to Title",
                        description=f"Include '{keywords.primary_keyword}' in the page title for immediate SEO boost.",
                        priority=RecommendationPriority.HIGH,
                        category=RecommendationCategory.QUICK_WIN,
                        impact="High - Direct ranking signal",
                        effort="Low - One edit",
                    )
                )

            if kw_analysis.primary_found and "first_100_words" not in kw_analysis.primary_locations:
                quick_wins.append(
                    Recommendation(
                        title="Move Keyword to Opening",
                        description="Mention the primary keyword within the first 100 words of content.",
                        priority=RecommendationPriority.MEDIUM,
                        category=RecommendationCategory.QUICK_WIN,
                        impact="Medium - Relevance signal",
                        effort="Low - Minor edit",
                    )
                )

        # Check for structural quick wins
        if geo_score.seo_score.heading_analysis:
            heading_analysis = geo_score.seo_score.heading_analysis

            if heading_analysis.heading_density < 0.8:
                quick_wins.append(
                    Recommendation(
                        title="Add More Subheadings",
                        description="Add H2/H3 headings every 200-300 words to break up content.",
                        priority=RecommendationPriority.LOW,
                        category=RecommendationCategory.QUICK_WIN,
                        impact="Medium - Better UX and SEO",
                        effort="Low - Easy additions",
                    )
                )

        return quick_wins

    def _deduplicate(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Remove duplicate recommendations."""
        seen_titles: set[str] = set()
        unique: list[Recommendation] = []

        for rec in recommendations:
            title_key = rec.title.lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(rec)

        return unique

    def _sort_by_priority(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Sort recommendations by priority and impact."""
        priority_order = {
            RecommendationPriority.HIGH: 0,
            RecommendationPriority.MEDIUM: 1,
            RecommendationPriority.LOW: 2,
        }

        # Secondary sort by category (quick wins first within priority)
        category_order = {
            RecommendationCategory.QUICK_WIN: 0,
            RecommendationCategory.CONTENT_GAP: 1,
            RecommendationCategory.OPTIMIZATION: 2,
            RecommendationCategory.STRUCTURE: 3,
            RecommendationCategory.TECHNICAL: 4,
        }

        return sorted(
            recommendations,
            key=lambda r: (
                priority_order.get(r.priority, 3),
                category_order.get(r.category, 5),
            ),
        )

    def format_recommendations(
        self, recommendations: list[Recommendation]
    ) -> str:
        """
        Format recommendations as a readable list.

        Args:
            recommendations: List of recommendations

        Returns:
            Formatted string
        """
        if not recommendations:
            return "No recommendations at this time."

        lines = ["Recommendations:", ""]

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. [{rec.priority.value.upper()}] {rec.title}")
            lines.append(f"   {rec.description}")
            lines.append(f"   Impact: {rec.impact}")
            lines.append(f"   Effort: {rec.effort}")
            lines.append("")

        return "\n".join(lines)

    def get_priority_recommendations(
        self,
        recommendations: list[Recommendation],
        priority: RecommendationPriority,
    ) -> list[Recommendation]:
        """Get recommendations filtered by priority."""
        return [r for r in recommendations if r.priority == priority]


def generate_recommendations(
    issues: list[Issue],
    geo_score: GEOScore,
    keywords: KeywordConfig | None = None,
    max_count: int = 10,
) -> list[Recommendation]:
    """
    Convenience function to generate recommendations.

    Args:
        issues: List of detected issues
        geo_score: Computed GEO score
        keywords: Target keyword configuration
        max_count: Maximum recommendations to return

    Returns:
        List of prioritized recommendations
    """
    config = RecommendationEngineConfig(max_recommendations=max_count)
    engine = RecommendationEngine(config)
    return engine.generate(issues, geo_score, keywords)

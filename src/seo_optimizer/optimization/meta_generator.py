"""
Meta Generator - Title and Description Optimization

Responsibilities:
- Generate SEO-optimized meta titles
- Generate compelling meta descriptions
- Enforce pixel width constraints
- Include primary keywords naturally
- Create compelling CTAs

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .guardrails import SafetyGuardrails
from .models import ChangeType, MetaTags, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import ContentAnalysisResult


# =============================================================================
# Meta Tag Constants
# =============================================================================

# Pixel width constraints (based on Google SERP display)
MAX_TITLE_PIXELS = 600
MAX_DESCRIPTION_PIXELS = 920

# Character approximations (average for common fonts)
CHAR_WIDTH_TITLE = 9.5  # Average pixels per character in title
CHAR_WIDTH_DESCRIPTION = 6.0  # Average pixels per character in description

# Derived character limits
MAX_TITLE_CHARS = int(MAX_TITLE_PIXELS / CHAR_WIDTH_TITLE)  # ~63 chars
MAX_DESCRIPTION_CHARS = int(MAX_DESCRIPTION_PIXELS / CHAR_WIDTH_DESCRIPTION)  # ~153 chars

# Minimum lengths for quality
MIN_TITLE_CHARS = 30
MIN_DESCRIPTION_CHARS = 70

# CTA templates for descriptions
CTA_TEMPLATES = [
    "Learn more",
    "Discover how",
    "Find out why",
    "Get started today",
    "See how",
    "Read our guide",
    "Explore",
]

# Title templates by content type
TITLE_TEMPLATES = {
    "guide": "{topic}: Complete Guide [{year}]",
    "howto": "How to {topic} - Step by Step Guide",
    "list": "{count} Best {topic} in {year}",
    "review": "{topic} Review: Pros, Cons & Verdict",
    "comparison": "{topic} vs {alternative}: Which is Better?",
    "default": "{topic} | {brand}",
}

# Description templates
DESCRIPTION_TEMPLATES = {
    "guide": "Learn everything about {topic} in this comprehensive guide. {benefit}. {cta}.",
    "howto": "Step-by-step instructions for {topic}. {benefit}. {cta} and start today.",
    "list": "Discover the {count} best {topic} options. {benefit}. {cta} to find your perfect match.",
    "informational": "{topic} explained in detail. {benefit}. {cta} to learn more.",
    "default": "Everything you need to know about {topic}. {benefit}. {cta}.",
}


@dataclass
class MetaGenerationResult:
    """Result of meta tag generation."""

    meta_tags: MetaTags | None = None
    title_change: OptimizationChange | None = None
    description_change: OptimizationChange | None = None
    validation_issues: list[str] | None = None


class MetaGenerator:
    """
    Generates optimized meta titles and descriptions.

    Creates SEO-friendly meta tags that:
    - Include target keywords naturally
    - Stay within pixel width constraints
    - Are compelling to click
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the meta generator."""
        self.config = config
        self.guardrails = guardrails

    def generate(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None = None,
        existing_title: str | None = None,
        existing_description: str | None = None,
    ) -> MetaGenerationResult:
        """
        Generate optimized meta tags.

        Args:
            ast: Document AST
            analysis: Pre-computed content analysis
            existing_title: Current meta title if any
            existing_description: Current meta description if any

        Returns:
            MetaGenerationResult with generated tags
        """
        result = MetaGenerationResult()
        result.validation_issues = []

        # Extract content context
        context = self._extract_content_context(ast, analysis)

        # Generate or optimize title
        new_title = self._generate_title(context, existing_title)

        # Generate or optimize description
        new_description = self._generate_description(context, existing_description)

        # Apply AI vocabulary filter
        if new_title:
            filter_result = self.guardrails.filter_ai_vocabulary(new_title)
            new_title = filter_result.cleaned_text

        if new_description:
            filter_result = self.guardrails.filter_ai_vocabulary(new_description)
            new_description = filter_result.cleaned_text

        # Validate and truncate
        new_title = self._optimize_title_length(new_title)
        new_description = self._optimize_description_length(new_description)

        # Create result
        result.meta_tags = MetaTags(
            title=new_title,
            description=new_description,
            title_pixel_width=self._estimate_pixel_width(new_title, "title"),
            description_pixel_width=self._estimate_pixel_width(new_description, "description"),
        )

        # Create change records
        if existing_title and new_title != existing_title:
            result.title_change = OptimizationChange(
                change_type=ChangeType.META,
                location="Meta Title",
                original=existing_title,
                optimized=new_title,
                reason="Optimized title for SEO and SERP display",
                impact_score=3.5,
            )

        if existing_description and new_description != existing_description:
            result.description_change = OptimizationChange(
                change_type=ChangeType.META,
                location="Meta Description",
                original=existing_description,
                optimized=new_description,
                reason="Optimized description for CTR and keyword placement",
                impact_score=3.0,
            )

        # Validate
        result.validation_issues = self._validate_meta_tags(result.meta_tags)

        return result

    def _extract_content_context(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
    ) -> dict:
        """
        Extract context for meta generation.

        Args:
            ast: Document AST
            analysis: Content analysis

        Returns:
            Context dictionary
        """
        context: dict = {
            "topic": "",
            "content_type": "default",
            "benefit": "",
            "brand": self.config.brand_name or "Your Guide",
            "year": "2026",
            "count": "",
            "alternative": "",
        }

        # Get topic from primary keyword or H1
        if self.config.primary_keyword:
            context["topic"] = self.config.primary_keyword

        # Find H1 for topic if not set
        if not context["topic"]:
            for node in ast.nodes:
                if node.node_type == NodeType.HEADING:
                    level = node.metadata.get("level", 2)
                    if level == 1:
                        context["topic"] = node.text_content.strip()
                        break

        # Detect content type from structure
        context["content_type"] = self._detect_content_type(ast)

        # Extract benefit statement
        context["benefit"] = self._extract_benefit(ast)

        # Count items for list content
        if context["content_type"] == "list":
            context["count"] = self._count_list_items(ast)

        return context

    def _detect_content_type(self, ast: DocumentAST) -> str:
        """
        Detect content type from document structure.

        Args:
            ast: Document AST

        Returns:
            Content type string
        """
        full_text = ast.full_text.lower()
        h1_text = ""

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                if level == 1:
                    h1_text = node.text_content.lower()
                    break

        # Check for how-to content
        if re.search(r"\bhow\s+to\b", h1_text) or "step" in full_text[:500]:
            return "howto"

        # Check for list content
        if re.search(r"\b\d+\s+(best|top|ways|tips|reasons)\b", h1_text):
            return "list"

        # Check for guide content
        if re.search(r"\b(guide|tutorial|complete)\b", h1_text):
            return "guide"

        # Check for review content
        if re.search(r"\b(review|pros|cons|verdict)\b", h1_text):
            return "review"

        # Check for comparison content
        if re.search(r"\bvs\b|\bversus\b|\bcompare\b", h1_text):
            return "comparison"

        return "default"

    def _extract_benefit(self, ast: DocumentAST) -> str:
        """
        Extract a benefit statement from content.

        Args:
            ast: Document AST

        Returns:
            Benefit statement
        """
        # Look for benefit indicators in paragraphs
        for node in ast.nodes:
            if node.node_type != NodeType.PARAGRAPH:
                continue

            text = node.text_content

            # Check for benefit patterns
            patterns = [
                r"helps?\s+(?:you\s+)?(\w+(?:\s+\w+){2,6})",
                r"enables?\s+(?:you\s+to\s+)?(\w+(?:\s+\w+){2,6})",
                r"makes?\s+(?:it\s+)?(?:easy|easier)\s+to\s+(\w+(?:\s+\w+){2,5})",
                r"saves?\s+(?:you\s+)?(\w+(?:\s+\w+){1,4})",
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    benefit = match.group(0).strip()
                    # Capitalize first letter
                    return benefit[0].upper() + benefit[1:]

        # Default benefit based on content type
        return "Get the information you need"

    def _count_list_items(self, ast: DocumentAST) -> str:
        """
        Count list items or H2 headings for list content.

        Args:
            ast: Document AST

        Returns:
            Count as string
        """
        h2_count = 0

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                if level == 2:
                    h2_count += 1

        return str(h2_count) if h2_count >= 3 else "Top"

    def _generate_title(
        self, context: dict, existing_title: str | None
    ) -> str:
        """
        Generate optimized meta title.

        Args:
            context: Content context
            existing_title: Existing title if any

        Returns:
            Optimized title
        """
        topic = context.get("topic", "")
        content_type = context.get("content_type", "default")

        # Check if existing title is acceptable
        if existing_title:
            # Validate existing title
            if self._is_title_acceptable(existing_title, topic):
                return existing_title

        # Generate new title from template
        template = TITLE_TEMPLATES.get(content_type, TITLE_TEMPLATES["default"])

        title = template.format(
            topic=topic,
            brand=context.get("brand", ""),
            year=context.get("year", "2026"),
            count=context.get("count", ""),
            alternative=context.get("alternative", ""),
        )

        # Clean up empty placeholders
        title = re.sub(r"\s*\|\s*$", "", title)
        title = re.sub(r"\[\]", "", title)
        title = re.sub(r"\s+", " ", title).strip()

        # Ensure keyword is included
        if topic and topic.lower() not in title.lower():
            # Prepend topic
            title = f"{topic}: {title}"

        return title

    def _is_title_acceptable(self, title: str, keyword: str) -> bool:
        """
        Check if existing title is acceptable.

        Args:
            title: The title to check
            keyword: Primary keyword

        Returns:
            True if acceptable
        """
        # Check length
        if len(title) < MIN_TITLE_CHARS or len(title) > MAX_TITLE_CHARS + 10:
            return False

        # Check keyword presence
        if keyword and keyword.lower() not in title.lower():
            return False

        return True

    def _generate_description(
        self, context: dict, existing_description: str | None
    ) -> str:
        """
        Generate optimized meta description.

        Args:
            context: Content context
            existing_description: Existing description if any

        Returns:
            Optimized description
        """
        topic = context.get("topic", "")
        content_type = context.get("content_type", "default")

        # Check if existing description is acceptable
        if existing_description:
            if self._is_description_acceptable(existing_description, topic):
                return existing_description

        # Generate from template
        template = DESCRIPTION_TEMPLATES.get(content_type, DESCRIPTION_TEMPLATES["default"])

        # Select appropriate CTA
        cta = self._select_cta(content_type)

        description = template.format(
            topic=topic,
            benefit=context.get("benefit", "Get the information you need"),
            cta=cta,
            count=context.get("count", "best"),
        )

        # Clean up
        description = re.sub(r"\s+", " ", description).strip()

        return description

    def _is_description_acceptable(self, description: str, keyword: str) -> bool:
        """
        Check if existing description is acceptable.

        Args:
            description: The description to check
            keyword: Primary keyword

        Returns:
            True if acceptable
        """
        # Check length
        if len(description) < MIN_DESCRIPTION_CHARS or len(description) > MAX_DESCRIPTION_CHARS + 20:
            return False

        # Check keyword presence
        if keyword and keyword.lower() not in description.lower():
            return False

        # Check for CTA presence
        has_cta = any(cta.lower() in description.lower() for cta in CTA_TEMPLATES)
        if not has_cta:
            return False

        return True

    def _select_cta(self, content_type: str) -> str:
        """
        Select appropriate CTA for content type.

        Args:
            content_type: Type of content

        Returns:
            CTA string
        """
        cta_map = {
            "guide": "Read our guide",
            "howto": "Get started today",
            "list": "Discover",
            "review": "See our verdict",
            "comparison": "Find out which is better",
            "default": "Learn more",
        }

        return cta_map.get(content_type, "Learn more")

    def _optimize_title_length(self, title: str) -> str:
        """
        Optimize title to fit pixel constraints.

        Args:
            title: The title to optimize

        Returns:
            Optimized title
        """
        pixel_width = self._estimate_pixel_width(title, "title")

        if pixel_width <= MAX_TITLE_PIXELS:
            return title

        # Need to truncate
        # Try removing brand first
        if "|" in title:
            parts = title.split("|")
            shortened = parts[0].strip()
            if self._estimate_pixel_width(shortened, "title") <= MAX_TITLE_PIXELS:
                return shortened

        # Truncate at word boundary
        while self._estimate_pixel_width(title, "title") > MAX_TITLE_PIXELS:
            words = title.split()
            if len(words) <= 3:
                break
            title = " ".join(words[:-1])

        # Add ellipsis if truncated
        if not title.endswith("..."):
            title = title.rstrip(".,!?") + "..."

        return title

    def _optimize_description_length(self, description: str) -> str:
        """
        Optimize description to fit pixel constraints.

        Args:
            description: The description to optimize

        Returns:
            Optimized description
        """
        pixel_width = self._estimate_pixel_width(description, "description")

        if pixel_width <= MAX_DESCRIPTION_PIXELS:
            return description

        # Truncate at sentence boundary
        sentences = re.split(r"(?<=[.!?])\s+", description)
        result: list[str] = []
        current_width = 0

        for sentence in sentences:
            sentence_width = self._estimate_pixel_width(sentence, "description")
            if current_width + sentence_width <= MAX_DESCRIPTION_PIXELS:
                result.append(sentence)
                current_width += sentence_width + 6  # Account for space
            else:
                break

        if result:
            return " ".join(result)

        # Last resort: truncate at word boundary
        words = description.split()
        while self._estimate_pixel_width(" ".join(words), "description") > MAX_DESCRIPTION_PIXELS:
            if len(words) <= 5:
                break
            words = words[:-1]

        return " ".join(words) + "..."

    def _estimate_pixel_width(self, text: str, tag_type: str) -> float:
        """
        Estimate pixel width of text.

        Uses character-based approximation.
        In production, use font metrics library.

        Args:
            text: The text to measure
            tag_type: "title" or "description"

        Returns:
            Estimated pixel width
        """
        if not text:
            return 0.0

        char_width = CHAR_WIDTH_TITLE if tag_type == "title" else CHAR_WIDTH_DESCRIPTION

        # Adjust for character types
        total_width = 0.0

        for char in text:
            if char.isupper():
                total_width += char_width * 1.2  # Uppercase is wider
            elif char in "mwMW":
                total_width += char_width * 1.3  # Wide characters
            elif char in "ilIL.,!|":
                total_width += char_width * 0.5  # Narrow characters
            elif char == " ":
                total_width += char_width * 0.4  # Space
            else:
                total_width += char_width

        return total_width

    def _validate_meta_tags(self, meta_tags: MetaTags | None) -> list[str]:
        """
        Validate meta tags against best practices.

        Args:
            meta_tags: The meta tags to validate

        Returns:
            List of validation issues
        """
        issues: list[str] = []

        if not meta_tags:
            issues.append("No meta tags generated")
            return issues

        # Validate title
        if meta_tags.title:
            title_len = len(meta_tags.title)
            if title_len < MIN_TITLE_CHARS:
                issues.append(f"Title too short ({title_len} chars, min {MIN_TITLE_CHARS})")
            if meta_tags.title_pixel_width and meta_tags.title_pixel_width > MAX_TITLE_PIXELS:
                issues.append(f"Title may be truncated in SERP ({meta_tags.title_pixel_width:.0f}px > {MAX_TITLE_PIXELS}px)")

            # Check for keyword
            if self.config.primary_keyword:
                if self.config.primary_keyword.lower() not in meta_tags.title.lower():
                    issues.append("Primary keyword not in title")
        else:
            issues.append("No title generated")

        # Validate description
        if meta_tags.description:
            desc_len = len(meta_tags.description)
            if desc_len < MIN_DESCRIPTION_CHARS:
                issues.append(f"Description too short ({desc_len} chars, min {MIN_DESCRIPTION_CHARS})")
            if meta_tags.description_pixel_width and meta_tags.description_pixel_width > MAX_DESCRIPTION_PIXELS:
                issues.append(f"Description may be truncated ({meta_tags.description_pixel_width:.0f}px > {MAX_DESCRIPTION_PIXELS}px)")

            # Check for CTA
            has_cta = any(cta.lower() in meta_tags.description.lower() for cta in CTA_TEMPLATES)
            if not has_cta:
                issues.append("Description lacks call-to-action")
        else:
            issues.append("No description generated")

        return issues

    def generate_og_tags(
        self, meta_tags: MetaTags, url: str | None = None, image_url: str | None = None
    ) -> dict[str, str]:
        """
        Generate Open Graph tags.

        Args:
            meta_tags: Base meta tags
            url: Page URL
            image_url: OG image URL

        Returns:
            Dict of OG tag name to value
        """
        og_tags: dict[str, str] = {}

        if meta_tags.title:
            og_tags["og:title"] = meta_tags.title
            og_tags["twitter:title"] = meta_tags.title

        if meta_tags.description:
            og_tags["og:description"] = meta_tags.description
            og_tags["twitter:description"] = meta_tags.description

        og_tags["og:type"] = "article"

        if url:
            og_tags["og:url"] = url

        if image_url:
            og_tags["og:image"] = image_url
            og_tags["twitter:image"] = image_url
            og_tags["twitter:card"] = "summary_large_image"
        else:
            og_tags["twitter:card"] = "summary"

        return og_tags

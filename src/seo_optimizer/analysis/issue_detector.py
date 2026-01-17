"""
Issue Detector - Cross-Cutting Problem Detection

Aggregates issues from all scorers and detects additional problems:
- Thin content (< 300 words)
- Missing FAQ section
- Keyword stuffing
- Content freshness indicators
- Structural problems

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .models import (
    GEOScore,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordConfig,
)

# Content thresholds
MIN_WORD_COUNT = 300
OPTIMAL_WORD_COUNT = 1000
MAX_WORD_COUNT = 5000

# Keyword stuffing threshold
KEYWORD_STUFFING_THRESHOLD = 5.0  # >5% density = stuffing


@dataclass
class IssueDetectorConfig:
    """Configuration for issue detection."""

    # Content length thresholds
    min_word_count: int = MIN_WORD_COUNT
    optimal_word_count: int = OPTIMAL_WORD_COUNT
    max_word_count: int = MAX_WORD_COUNT

    # Keyword stuffing threshold
    keyword_stuffing_threshold: float = KEYWORD_STUFFING_THRESHOLD

    # Structural requirements
    require_faq_section: bool = True
    require_meta_description: bool = True
    require_images: bool = True


class IssueDetector:
    """
    Detects cross-cutting issues that span multiple scoring categories.

    This class aggregates issues from individual scorers and adds
    detection for additional problems.
    """

    # FAQ section indicators
    FAQ_PATTERNS = [
        r"frequently\s+asked\s+questions?",
        r"\bfaq\b",
        r"common\s+questions?",
        r"questions?\s+and\s+answers?",
        r"\bq\s*&\s*a\b",
    ]

    def __init__(self, config: IssueDetectorConfig | None = None) -> None:
        """Initialize the issue detector."""
        self.config = config or IssueDetectorConfig()

    def detect_all(
        self,
        ast: DocumentAST,
        geo_score: GEOScore,
        keywords: KeywordConfig | None = None,
        meta_description: str | None = None,
    ) -> list[Issue]:
        """
        Detect all issues in a document.

        Combines scorer issues with cross-cutting issue detection.

        Args:
            ast: The document AST
            geo_score: The computed GEO score (contains scorer issues)
            keywords: Target keyword configuration
            meta_description: Page meta description

        Returns:
            Complete list of issues, sorted by severity
        """
        issues: list[Issue] = []

        # Collect scorer issues
        issues.extend(geo_score.all_issues)

        # Detect additional cross-cutting issues
        issues.extend(self._detect_thin_content(ast))
        issues.extend(self._detect_missing_faq(ast))
        issues.extend(self._detect_keyword_stuffing(ast, keywords))
        issues.extend(self._detect_meta_issues(ast, meta_description, keywords))
        issues.extend(self._detect_structural_issues(ast))
        issues.extend(self._detect_freshness_issues(ast))

        # Sort by severity (critical first)
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.WARNING: 1,
            IssueSeverity.INFO: 2,
        }
        issues.sort(key=lambda i: severity_order.get(i.severity, 3))

        # Deduplicate similar issues
        issues = self._deduplicate_issues(issues)

        return issues

    def _detect_thin_content(self, ast: DocumentAST) -> list[Issue]:
        """Detect thin content issues."""
        issues: list[Issue] = []
        word_count = len(ast.full_text.split()) if ast.full_text else 0

        if word_count < self.config.min_word_count:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.CRITICAL,
                    message=f"Thin content: only {word_count} words",
                    current_value=f"{word_count} words",
                    target_value=f">= {self.config.min_word_count} words",
                    fix_suggestion="Expand content with more depth, examples, and supporting information",
                )
            )
        elif word_count < self.config.optimal_word_count:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.INFO,
                    message=f"Content length ({word_count} words) below optimal",
                    current_value=f"{word_count} words",
                    target_value=f"~{self.config.optimal_word_count} words",
                    fix_suggestion="Consider adding more depth or supporting sections",
                )
            )
        elif word_count > self.config.max_word_count:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.INFO,
                    message=f"Content may be too long ({word_count} words)",
                    current_value=f"{word_count} words",
                    target_value=f"<= {self.config.max_word_count} words",
                    fix_suggestion="Consider splitting into multiple focused articles",
                )
            )

        return issues

    def _detect_missing_faq(self, ast: DocumentAST) -> list[Issue]:
        """Detect missing FAQ section."""
        if not self.config.require_faq_section:
            return []

        issues: list[Issue] = []
        has_faq = False

        # Check headings for FAQ patterns
        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                text_lower = node.text_content.lower()
                for pattern in self.FAQ_PATTERNS:
                    if re.search(pattern, text_lower):
                        has_faq = True
                        break
            if has_faq:
                break

        if not has_faq:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.INFO,
                    message="Missing FAQ section",
                    fix_suggestion="Add a Frequently Asked Questions section with common queries",
                )
            )

        return issues

    def _detect_keyword_stuffing(
        self, ast: DocumentAST, keywords: KeywordConfig | None
    ) -> list[Issue]:
        """Detect keyword stuffing (over-optimization)."""
        issues: list[Issue] = []

        if not keywords or not keywords.primary_keyword:
            return issues

        full_text = ast.full_text.lower() if ast.full_text else ""
        word_count = len(full_text.split())

        if word_count == 0:
            return issues

        # Calculate primary keyword density
        primary = keywords.primary_keyword.lower()
        primary_count = full_text.count(primary)
        primary_word_count = len(primary.split())
        density = (primary_count * primary_word_count / word_count) * 100

        if density > self.config.keyword_stuffing_threshold:
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message=f"Keyword stuffing detected ({density:.1f}% density)",
                    current_value=f"{density:.1f}%",
                    target_value=f"< {self.config.keyword_stuffing_threshold}%",
                    fix_suggestion="Reduce keyword frequency to avoid SEO penalty",
                )
            )

        # Check for unnatural keyword proximity
        if self._has_keyword_clustering(full_text, primary):
            issues.append(
                Issue(
                    category=IssueCategory.KEYWORD,
                    severity=IssueSeverity.WARNING,
                    message="Unnatural keyword clustering detected",
                    fix_suggestion="Distribute keyword mentions more naturally throughout content",
                )
            )

        return issues

    def _has_keyword_clustering(self, text: str, keyword: str) -> bool:
        """
        Check for unnatural keyword clustering.

        Looks for multiple occurrences within a small window.
        """
        words = text.split()
        window_size = 50  # Words

        if len(words) < window_size:
            return False

        for i in range(len(words) - window_size):
            window = " ".join(words[i : i + window_size])
            count = window.count(keyword)
            if count >= 4:  # 4+ occurrences in 50 words
                return True

        return False

    def _detect_meta_issues(
        self,
        _ast: DocumentAST,
        meta_description: str | None,
        keywords: KeywordConfig | None,
    ) -> list[Issue]:
        """Detect meta-related issues."""
        issues: list[Issue] = []

        if not self.config.require_meta_description:
            return issues

        if not meta_description:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.WARNING,
                    message="Missing meta description",
                    fix_suggestion="Add a compelling meta description (150-160 characters)",
                )
            )
        else:
            # Check meta description length
            desc_length = len(meta_description)
            if desc_length < 120:
                issues.append(
                    Issue(
                        category=IssueCategory.STRUCTURE,
                        severity=IssueSeverity.INFO,
                        message=f"Meta description too short ({desc_length} characters)",
                        current_value=f"{desc_length} characters",
                        target_value="150-160 characters",
                        fix_suggestion="Expand meta description with more compelling copy",
                    )
                )
            elif desc_length > 160:
                issues.append(
                    Issue(
                        category=IssueCategory.STRUCTURE,
                        severity=IssueSeverity.INFO,
                        message=f"Meta description too long ({desc_length} characters)",
                        current_value=f"{desc_length} characters",
                        target_value="150-160 characters",
                        fix_suggestion="Shorten meta description to prevent truncation in SERPs",
                    )
                )

            # Check for keyword in meta description
            if (
                keywords
                and keywords.primary_keyword
                and keywords.primary_keyword.lower() not in meta_description.lower()
            ):
                issues.append(
                    Issue(
                        category=IssueCategory.KEYWORD,
                        severity=IssueSeverity.INFO,
                        message="Primary keyword missing from meta description",
                        fix_suggestion="Include the primary keyword in the meta description",
                    )
                )

        return issues

    def _detect_structural_issues(self, ast: DocumentAST) -> list[Issue]:
        """Detect structural issues in the document."""
        issues: list[Issue] = []

        # Check for heading structure
        headings = [n for n in ast.nodes if n.node_type == NodeType.HEADING]

        if not headings:
            issues.append(
                Issue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.CRITICAL,
                    message="No headings found in content",
                    fix_suggestion="Add H1, H2, and H3 headings to structure your content",
                )
            )

        # Check for images
        if self.config.require_images:
            has_images = any(
                n.metadata.get("has_images", False)
                or n.node_type == NodeType.IMAGE
                for n in ast.nodes
            )
            if not has_images:
                issues.append(
                    Issue(
                        category=IssueCategory.STRUCTURE,
                        severity=IssueSeverity.INFO,
                        message="No images found in content",
                        fix_suggestion="Add relevant images with descriptive alt text",
                    )
                )

        # Check paragraph lengths
        paragraphs = [n for n in ast.nodes if n.node_type == NodeType.PARAGRAPH]
        long_paragraphs = [
            p for p in paragraphs
            if len(p.text_content.split()) > 150
        ]

        if long_paragraphs:
            issues.append(
                Issue(
                    category=IssueCategory.READABILITY,
                    severity=IssueSeverity.INFO,
                    message=f"{len(long_paragraphs)} paragraph(s) exceed 150 words",
                    fix_suggestion="Break long paragraphs into smaller, more scannable blocks",
                )
            )

        return issues

    def _detect_freshness_issues(self, ast: DocumentAST) -> list[Issue]:
        """Detect content freshness indicators."""
        issues: list[Issue] = []
        full_text = ast.full_text.lower() if ast.full_text else ""

        # Look for outdated year references
        outdated_patterns = [
            r"\b20(1[0-9]|2[0-2])\b",  # Years 2010-2022
            r"last year",
            r"this year",
            r"recently updated",
        ]

        for pattern in outdated_patterns:
            if re.search(pattern, full_text):
                issues.append(
                    Issue(
                        category=IssueCategory.STRUCTURE,
                        severity=IssueSeverity.INFO,
                        message="Content may contain outdated date references",
                        fix_suggestion="Review and update date-specific information",
                    )
                )
                break

        return issues

    def _deduplicate_issues(self, issues: list[Issue]) -> list[Issue]:
        """Remove duplicate or very similar issues."""
        seen_messages: set[str] = set()
        unique_issues: list[Issue] = []

        for issue in issues:
            # Create a simplified key for deduplication
            key = f"{issue.category.value}:{issue.message[:50]}"
            if key not in seen_messages:
                seen_messages.add(key)
                unique_issues.append(issue)

        return unique_issues

    def get_critical_issues(self, issues: list[Issue]) -> list[Issue]:
        """Get only critical severity issues."""
        return [i for i in issues if i.severity == IssueSeverity.CRITICAL]

    def get_warning_issues(self, issues: list[Issue]) -> list[Issue]:
        """Get only warning severity issues."""
        return [i for i in issues if i.severity == IssueSeverity.WARNING]

    def get_info_issues(self, issues: list[Issue]) -> list[Issue]:
        """Get only info severity issues."""
        return [i for i in issues if i.severity == IssueSeverity.INFO]

    def get_issues_by_category(
        self, issues: list[Issue], category: IssueCategory
    ) -> list[Issue]:
        """Get issues filtered by category."""
        return [i for i in issues if i.category == category]

    def format_issue_summary(self, issues: list[Issue]) -> str:
        """
        Create a formatted summary of issues.

        Args:
            issues: List of issues

        Returns:
            Formatted string summary
        """
        if not issues:
            return "No issues detected."

        critical = self.get_critical_issues(issues)
        warnings = self.get_warning_issues(issues)
        info = self.get_info_issues(issues)

        lines = [
            f"Issues Found: {len(issues)} total",
            f"  - {len(critical)} Critical",
            f"  - {len(warnings)} Warnings",
            f"  - {len(info)} Info",
            "",
        ]

        if critical:
            lines.append("Critical Issues:")
            for issue in critical[:5]:  # Top 5
                lines.append(f"  - {issue.message}")

        if warnings:
            lines.append("\nWarnings:")
            for issue in warnings[:5]:  # Top 5
                lines.append(f"  - {issue.message}")

        return "\n".join(lines)


def detect_issues(
    ast: DocumentAST,
    geo_score: GEOScore,
    keywords: KeywordConfig | None = None,
) -> list[Issue]:
    """
    Convenience function to detect issues without class instantiation.

    Args:
        ast: Document AST
        geo_score: Computed GEO score
        keywords: Target keyword configuration

    Returns:
        List of all detected issues
    """
    detector = IssueDetector()
    return detector.detect_all(ast, geo_score, keywords)

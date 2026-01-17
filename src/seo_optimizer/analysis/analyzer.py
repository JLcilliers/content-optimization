"""
Analyzer - Main Orchestrator for SEO Analysis

Orchestrates all scoring components to produce:
- GEO-Metric composite score
- Component breakdowns
- Issue detection
- Actionable recommendations

GEO Formula:
GEO = (0.20 × SEO) + (0.30 × Semantic) + (0.30 × AI) + (0.20 × Readability)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from seo_optimizer.ingestion.docx_parser import parse_docx
from seo_optimizer.ingestion.models import DocumentAST

from .ai_scorer import AIScorer
from .entity_extractor import EntityExtractor
from .issue_detector import IssueDetector
from .models import (
    AnalysisResult,
    DocumentStats,
    GEOScore,
    KeywordConfig,
    VersionComparison,
)
from .readability_scorer import ReadabilityScorer
from .recommendation_engine import Recommendation, RecommendationEngine
from .semantic_scorer import SemanticScorer
from .seo_scorer import SEOScorer

# GEO Score weights (from research)
SEO_WEIGHT = 0.20
SEMANTIC_WEIGHT = 0.30
AI_WEIGHT = 0.30
READABILITY_WEIGHT = 0.20


@dataclass
class AnalyzerConfig:
    """Configuration for the content analyzer."""

    # Component weights
    seo_weight: float = SEO_WEIGHT
    semantic_weight: float = SEMANTIC_WEIGHT
    ai_weight: float = AI_WEIGHT
    readability_weight: float = READABILITY_WEIGHT

    # Feature flags
    enable_semantic_analysis: bool = True
    enable_ai_analysis: bool = True
    enable_entity_extraction: bool = True

    # Output options
    max_recommendations: int = 10
    include_detailed_issues: bool = True


class ContentAnalyzer:
    """
    Main orchestrator for SEO content analysis.

    Coordinates all scoring components to produce a comprehensive
    analysis result with the GEO-Metric score.
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """
        Initialize the content analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or AnalyzerConfig()

        # Initialize scorers (lazy loaded)
        self._seo_scorer: SEOScorer | None = None
        self._semantic_scorer: SemanticScorer | None = None
        self._ai_scorer: AIScorer | None = None
        self._readability_scorer: ReadabilityScorer | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._issue_detector: IssueDetector | None = None
        self._recommendation_engine: RecommendationEngine | None = None

    @property
    def seo_scorer(self) -> SEOScorer:
        """Lazy load SEO scorer."""
        if self._seo_scorer is None:
            self._seo_scorer = SEOScorer()
        return self._seo_scorer

    @property
    def semantic_scorer(self) -> SemanticScorer:
        """Lazy load semantic scorer."""
        if self._semantic_scorer is None:
            self._semantic_scorer = SemanticScorer()
        return self._semantic_scorer

    @property
    def ai_scorer(self) -> AIScorer:
        """Lazy load AI scorer."""
        if self._ai_scorer is None:
            self._ai_scorer = AIScorer()
        return self._ai_scorer

    @property
    def readability_scorer(self) -> ReadabilityScorer:
        """Lazy load readability scorer."""
        if self._readability_scorer is None:
            self._readability_scorer = ReadabilityScorer()
        return self._readability_scorer

    @property
    def entity_extractor(self) -> EntityExtractor:
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            self._entity_extractor = EntityExtractor()
        return self._entity_extractor

    @property
    def issue_detector(self) -> IssueDetector:
        """Lazy load issue detector."""
        if self._issue_detector is None:
            self._issue_detector = IssueDetector()
        return self._issue_detector

    @property
    def recommendation_engine(self) -> RecommendationEngine:
        """Lazy load recommendation engine."""
        if self._recommendation_engine is None:
            self._recommendation_engine = RecommendationEngine()
        return self._recommendation_engine

    def analyze(
        self,
        ast: DocumentAST,
        keywords: KeywordConfig,
        title: str | None = None,
        url_slug: str | None = None,
        meta_description: str | None = None,
        topic_description: str | None = None,
        expected_entities: list[str] | None = None,
    ) -> AnalysisResult:
        """
        Perform comprehensive content analysis.

        Args:
            ast: The document AST (from parsing)
            keywords: Target keyword configuration
            title: Page title (meta title)
            url_slug: URL slug for the page
            meta_description: Meta description
            topic_description: Description of expected topic
            expected_entities: List of expected semantic entities

        Returns:
            Complete AnalysisResult with GEO score and recommendations
        """
        # Calculate document stats
        doc_stats = self._calculate_stats(ast)

        # Run all scorers
        seo_score = self.seo_scorer.score(ast, keywords, title, url_slug)

        if self.config.enable_semantic_analysis:
            semantic_score = self.semantic_scorer.score(
                ast, topic_description, expected_entities or keywords.semantic_entities
            )
        else:
            semantic_score = self.semantic_scorer.score(ast)

        if self.config.enable_ai_analysis:
            ai_score = self.ai_scorer.score(ast)
        else:
            from .models import AIScore
            ai_score = AIScore(total=75)  # Default score if disabled

        readability_score = self.readability_scorer.score(ast)

        # Calculate composite GEO score
        geo_total = (
            self.config.seo_weight * seo_score.total
            + self.config.semantic_weight * semantic_score.total
            + self.config.ai_weight * ai_score.total
            + self.config.readability_weight * readability_score.total
        )

        geo_score = GEOScore(
            seo_score=seo_score,
            semantic_score=semantic_score,
            ai_score=ai_score,
            readability_score=readability_score,
            total=geo_total,
        )

        # Detect issues
        all_issues = self.issue_detector.detect_all(
            ast, geo_score, keywords, meta_description
        )

        # Update GEO score with all issues
        geo_score.all_issues = all_issues

        # Generate recommendations
        recommendations = self.recommendation_engine.generate(
            all_issues, geo_score, keywords
        )

        return AnalysisResult(
            document_stats=doc_stats,
            geo_score=geo_score,
            recommendations=[self._format_recommendation(r) for r in recommendations],
        )

    def analyze_file(
        self,
        file_path: Path | str,
        primary_keyword: str,
        secondary_keywords: list[str] | None = None,
        semantic_entities: list[str] | None = None,
        title: str | None = None,
        url_slug: str | None = None,
        meta_description: str | None = None,
    ) -> AnalysisResult:
        """
        Analyze a DOCX file directly.

        Convenience method that handles parsing and analysis.

        Args:
            file_path: Path to the DOCX file
            primary_keyword: The main target keyword
            secondary_keywords: Optional secondary keywords
            semantic_entities: Optional semantic entities
            title: Page title
            url_slug: URL slug
            meta_description: Meta description

        Returns:
            Complete AnalysisResult
        """
        # Parse the document
        path = Path(file_path) if isinstance(file_path, str) else file_path
        ast = parse_docx(path)

        # Create keyword config
        keywords = KeywordConfig(
            primary_keyword=primary_keyword,
            secondary_keywords=secondary_keywords or [],
            semantic_entities=semantic_entities or [],
        )

        return self.analyze(
            ast=ast,
            keywords=keywords,
            title=title,
            url_slug=url_slug,
            meta_description=meta_description,
        )

    def compare_versions(
        self,
        original_ast: DocumentAST,
        optimized_ast: DocumentAST,
        keywords: KeywordConfig,
        title: str | None = None,
    ) -> VersionComparison:
        """
        Compare original and optimized document versions.

        Args:
            original_ast: AST of the original document
            optimized_ast: AST of the optimized document
            keywords: Target keyword configuration
            title: Page title

        Returns:
            VersionComparison with improvement metrics
        """
        # Analyze both versions
        original_result = self.analyze(original_ast, keywords, title)
        optimized_result = self.analyze(optimized_ast, keywords, title)

        # Calculate improvement
        original_score = original_result.geo_score.total
        optimized_score = optimized_result.geo_score.total
        improvement = optimized_score - original_score

        # Identify key changes
        key_changes = self._identify_key_changes(original_result, optimized_result)

        # Identify fixed issues
        original_issues = {i.message for i in original_result.geo_score.all_issues}
        optimized_issues = {i.message for i in optimized_result.geo_score.all_issues}
        issues_fixed = list(original_issues - optimized_issues)
        new_issues = list(optimized_issues - original_issues)

        return VersionComparison(
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=improvement,
            key_changes=key_changes,
            issues_fixed=issues_fixed,
            new_issues=new_issues,
        )

    def _calculate_stats(self, ast: DocumentAST) -> DocumentStats:
        """Calculate document statistics."""
        full_text = ast.full_text or ""
        words = full_text.split()

        # Count sentences (simple split)
        sentences = [s for s in full_text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        # Count structural elements from AST
        from seo_optimizer.ingestion.models import NodeType

        paragraph_count = len([n for n in ast.nodes if n.node_type == NodeType.PARAGRAPH])
        heading_count = len([n for n in ast.nodes if n.node_type == NodeType.HEADING])
        list_count = len([n for n in ast.nodes if n.node_type == NodeType.LIST])
        table_count = len([n for n in ast.nodes if n.node_type == NodeType.TABLE])

        # Count links and images from AST nodes
        link_count = sum(
            1 for n in ast.nodes if n.node_type == NodeType.HYPERLINK
        )
        image_count = sum(
            1 for n in ast.nodes if n.node_type == NodeType.IMAGE
        )

        return DocumentStats(
            word_count=len(words),
            paragraph_count=paragraph_count,
            sentence_count=len(sentences),
            heading_count=heading_count,
            list_count=list_count,
            table_count=table_count,
            link_count=link_count,
            image_count=image_count,
        )

    def _format_recommendation(self, rec: Recommendation) -> str:
        """Format a recommendation as a string."""
        return f"[{rec.priority.value.upper()}] {rec.title}: {rec.description}"

    def _identify_key_changes(
        self, original: AnalysisResult, optimized: AnalysisResult
    ) -> list[str]:
        """Identify key changes between versions."""
        changes: list[str] = []

        # Compare word counts
        orig_words = original.document_stats.word_count
        opt_words = optimized.document_stats.word_count
        if opt_words > orig_words * 1.2:
            changes.append(f"Content expanded by {opt_words - orig_words} words")
        elif opt_words < orig_words * 0.8:
            changes.append(f"Content reduced by {orig_words - opt_words} words")

        # Compare SEO scores
        seo_diff = optimized.geo_score.seo_score.total - original.geo_score.seo_score.total
        if abs(seo_diff) > 5:
            direction = "improved" if seo_diff > 0 else "decreased"
            changes.append(f"SEO score {direction} by {abs(seo_diff):.1f} points")

        # Compare semantic scores
        sem_diff = optimized.geo_score.semantic_score.total - original.geo_score.semantic_score.total
        if abs(sem_diff) > 5:
            direction = "improved" if sem_diff > 0 else "decreased"
            changes.append(f"Semantic depth {direction} by {abs(sem_diff):.1f} points")

        # Compare AI scores
        ai_diff = optimized.geo_score.ai_score.total - original.geo_score.ai_score.total
        if abs(ai_diff) > 5:
            direction = "improved" if ai_diff > 0 else "decreased"
            changes.append(f"AI compatibility {direction} by {abs(ai_diff):.1f} points")

        # Compare readability scores
        read_diff = optimized.geo_score.readability_score.total - original.geo_score.readability_score.total
        if abs(read_diff) > 5:
            direction = "improved" if read_diff > 0 else "decreased"
            changes.append(f"Readability {direction} by {abs(read_diff):.1f} points")

        # Check structural changes
        orig_headings = original.document_stats.heading_count
        opt_headings = optimized.document_stats.heading_count
        if opt_headings > orig_headings:
            changes.append(f"Added {opt_headings - orig_headings} heading(s)")

        orig_lists = original.document_stats.list_count
        opt_lists = optimized.document_stats.list_count
        if opt_lists > orig_lists:
            changes.append(f"Added {opt_lists - orig_lists} list(s)")

        return changes


def analyze_content(
    ast: DocumentAST,
    primary_keyword: str,
    secondary_keywords: list[str] | None = None,
    semantic_entities: list[str] | None = None,
    title: str | None = None,
) -> AnalysisResult:
    """
    Convenience function to analyze content without class instantiation.

    Args:
        ast: Document AST
        primary_keyword: Main target keyword
        secondary_keywords: Optional secondary keywords
        semantic_entities: Optional semantic entities
        title: Page title

    Returns:
        Complete AnalysisResult
    """
    keywords = KeywordConfig(
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords or [],
        semantic_entities=semantic_entities or [],
    )

    analyzer = ContentAnalyzer()
    return analyzer.analyze(ast, keywords, title)


def analyze_docx(
    file_path: Path | str,
    primary_keyword: str,
    secondary_keywords: list[str] | None = None,
) -> AnalysisResult:
    """
    Convenience function to analyze a DOCX file.

    Args:
        file_path: Path to the DOCX file
        primary_keyword: Main target keyword
        secondary_keywords: Optional secondary keywords

    Returns:
        Complete AnalysisResult
    """
    analyzer = ContentAnalyzer()
    return analyzer.analyze_file(
        file_path=file_path,
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords,
    )

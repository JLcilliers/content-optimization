"""
SEO Analysis Engine

Core intelligence module for content analysis and optimization scoring.

Components:
- GEO Score: Composite metric (SEO 20% + Semantic 30% + AI 30% + Readability 20%)
- Entity Extraction: NER using spaCy
- Issue Detection: Cross-cutting problem identification
- Recommendation Engine: Prioritized, actionable fixes

Usage:
    from seo_optimizer.analysis import ContentAnalyzer, KeywordConfig

    analyzer = ContentAnalyzer()
    keywords = KeywordConfig(
        primary_keyword="cloud computing",
        secondary_keywords=["AWS", "Azure", "serverless"],
        semantic_entities=["data center", "virtualization", "scalability"],
    )
    result = analyzer.analyze(ast, keywords)
    print(result.summary)
"""

from .ai_scorer import AIScorer, AIScorerConfig, score_ai_compatibility
from .analyzer import (
    AnalyzerConfig,
    ContentAnalyzer,
    analyze_content,
    analyze_docx,
)
from .entity_extractor import EntityExtractor, extract_entities_simple
from .issue_detector import IssueDetector, IssueDetectorConfig, detect_issues
from .models import (
    AIScore,
    AnalysisResult,
    DocumentStats,
    EntityMatch,
    GEOScore,
    HeadingAnalysis,
    Issue,
    IssueCategory,
    IssueSeverity,
    KeywordAnalysis,
    KeywordConfig,
    ReadabilityScore,
    SemanticScore,
    SEOScore,
    VersionComparison,
)
from .readability_scorer import ReadabilityScorer, ReadabilityScorerConfig, score_readability
from .recommendation_engine import (
    Recommendation,
    RecommendationCategory,
    RecommendationEngine,
    RecommendationEngineConfig,
    RecommendationPriority,
    generate_recommendations,
)
from .semantic_scorer import SemanticScorer, SemanticScorerConfig, score_semantic
from .seo_scorer import SEOScorer, SEOScorerConfig, score_seo

__all__ = [
    # Core Analyzer
    "ContentAnalyzer",
    "AnalyzerConfig",
    "analyze_content",
    "analyze_docx",
    # Models
    "GEOScore",
    "SEOScore",
    "SemanticScore",
    "AIScore",
    "ReadabilityScore",
    "Issue",
    "IssueCategory",
    "IssueSeverity",
    "EntityMatch",
    "KeywordConfig",
    "KeywordAnalysis",
    "HeadingAnalysis",
    "DocumentStats",
    "AnalysisResult",
    "VersionComparison",
    # Entity Extractor
    "EntityExtractor",
    "extract_entities_simple",
    # SEO Scorer
    "SEOScorer",
    "SEOScorerConfig",
    "score_seo",
    # Semantic Scorer
    "SemanticScorer",
    "SemanticScorerConfig",
    "score_semantic",
    # AI Scorer
    "AIScorer",
    "AIScorerConfig",
    "score_ai_compatibility",
    # Readability Scorer
    "ReadabilityScorer",
    "ReadabilityScorerConfig",
    "score_readability",
    # Issue Detector
    "IssueDetector",
    "IssueDetectorConfig",
    "detect_issues",
    # Recommendation Engine
    "RecommendationEngine",
    "RecommendationEngineConfig",
    "Recommendation",
    "RecommendationCategory",
    "RecommendationPriority",
    "generate_recommendations",
]

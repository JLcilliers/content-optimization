"""
SEO Content Optimization Engine

This module provides comprehensive content optimization capabilities:
- Keyword injection and density management
- Entity enrichment for semantic depth
- Heading structure optimization
- Readability improvements
- FAQ generation
- Meta tag optimization
- Redundancy resolution

Example usage:
    from seo_optimizer.optimization import (
        ContentOptimizer,
        OptimizationConfig,
        OptimizationMode,
    )

    config = OptimizationConfig(
        mode=OptimizationMode.BALANCED,
        primary_keyword="content optimization",
        secondary_keywords=["SEO", "AI content"],
        semantic_entities=["BERT", "E-E-A-T"],
    )

    optimizer = ContentOptimizer(config)
    result = optimizer.optimize(document_ast)

    print(f"GEO Score: {result.geo_score}")
    print(f"Changes: {len(result.changes)}")
"""

from .content_optimizer import ContentOptimizer, OptimizationContext
from .entity_enricher import EntityEnricher
from .faq_generator import FAQGenerationResult, FAQGenerator
from .guardrails import (
    AI_FLAGGED_VOCABULARY,
    DensityCheck,
    SafetyGuardrails,
    VocabularyCheck,
)
from .heading_optimizer import HeadingOptimizer
from .keyword_injector import KeywordInjector
from .meta_generator import MetaGenerationResult, MetaGenerator
from .models import (
    ChangeType,
    ContentType,
    FAQEntry,
    GuardrailViolation,
    MetaTags,
    OptimizationChange,
    OptimizationConfig,
    OptimizationMode,
    OptimizationResult,
    PipelineResult,
)
from .pipeline import OptimizationPipeline, PipelineConfig, PipelineState, optimize_content
from .readability_improver import ReadabilityImprover
from .redundancy_resolver import (
    RedundancyAnalysis,
    RedundancyMatch,
    RedundancyResolver,
)

__all__ = [
    # Main classes
    "ContentOptimizer",
    "OptimizationPipeline",
    # Configuration
    "OptimizationConfig",
    "OptimizationMode",
    "ContentType",
    "PipelineConfig",
    # Results
    "OptimizationResult",
    "OptimizationChange",
    "ChangeType",
    "PipelineResult",
    "FAQEntry",
    "MetaTags",
    # Specialized optimizers
    "HeadingOptimizer",
    "KeywordInjector",
    "EntityEnricher",
    "ReadabilityImprover",
    "FAQGenerator",
    "MetaGenerator",
    "RedundancyResolver",
    # Guardrails
    "SafetyGuardrails",
    "GuardrailViolation",
    "AI_FLAGGED_VOCABULARY",
    "DensityCheck",
    "VocabularyCheck",
    # Supporting types
    "OptimizationContext",
    "PipelineState",
    "FAQGenerationResult",
    "MetaGenerationResult",
    "RedundancyAnalysis",
    "RedundancyMatch",
    # Convenience functions
    "optimize_content",
]

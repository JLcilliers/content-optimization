"""
Content Optimizer - Main Orchestrator

Responsibilities:
- Coordinate all optimization components
- Apply optimizations in correct order
- Aggregate results and changes
- Compute GEO-Metric scores
- Respect mode-specific constraints

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import DocumentAST

from .entity_enricher import EntityEnricher
from .faq_generator import FAQGenerationResult, FAQGenerator
from .guardrails import SafetyGuardrails
from .heading_optimizer import HeadingOptimizer
from .keyword_injector import KeywordInjector
from .meta_generator import MetaGenerationResult, MetaGenerator
from .models import (
    ChangeType,
    OptimizationChange,
    OptimizationConfig,
    OptimizationMode,
    OptimizationResult,
)
from .readability_improver import ReadabilityImprover
from .redundancy_resolver import RedundancyAnalysis, RedundancyResolver

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import ContentAnalysisResult


# =============================================================================
# GEO-Metric Weights
# =============================================================================

GEO_METRIC_WEIGHTS = {
    "seo": 0.20,
    "semantic": 0.30,
    "ai_readiness": 0.30,
    "readability": 0.20,
}

# Mode-specific change limits
MODE_CHANGE_LIMITS = {
    OptimizationMode.CONSERVATIVE: 10,
    OptimizationMode.BALANCED: 25,
    OptimizationMode.AGGRESSIVE: 50,
}


@dataclass
class OptimizationContext:
    """Context accumulated during optimization."""

    total_changes: int = 0
    changes_by_type: dict[ChangeType, int] = field(default_factory=dict)
    faq_result: FAQGenerationResult | None = None
    meta_result: MetaGenerationResult | None = None
    redundancy_analysis: RedundancyAnalysis | None = None
    guardrail_violations: list[str] = field(default_factory=list)


class ContentOptimizer:
    """
    Main orchestrator for content optimization.

    Coordinates all optimization components:
    - Heading structure optimization
    - Keyword injection
    - Entity enrichment
    - Readability improvement
    - FAQ generation
    - Meta tag generation
    - Redundancy resolution

    Computes final GEO-Metric score combining:
    - SEO score (20%)
    - Semantic depth (30%)
    - AI readiness (30%)
    - Readability (20%)
    """

    def __init__(self, config: OptimizationConfig | None = None) -> None:
        """
        Initialize the content optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.guardrails = SafetyGuardrails(self.config)

        # Initialize all components
        self.heading_optimizer = HeadingOptimizer(self.config, self.guardrails)
        self.keyword_injector = KeywordInjector(self.config, self.guardrails)
        self.entity_enricher = EntityEnricher(self.config, self.guardrails)
        self.readability_improver = ReadabilityImprover(self.config, self.guardrails)
        self.faq_generator = FAQGenerator(self.config, self.guardrails)
        self.meta_generator = MetaGenerator(self.config, self.guardrails)
        self.redundancy_resolver = RedundancyResolver(self.config, self.guardrails)

    def optimize(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None = None,
    ) -> OptimizationResult:
        """
        Run full optimization on content.

        Args:
            ast: Document AST to optimize
            analysis: Pre-computed content analysis

        Returns:
            OptimizationResult with all changes and metrics
        """
        context = OptimizationContext()
        all_changes: list[OptimizationChange] = []

        # Get change limit based on mode
        max_changes = MODE_CHANGE_LIMITS.get(self.config.mode, 25)

        # Phase 1: Structure optimization (headings)
        if self._should_continue(context, max_changes):
            heading_changes = self._optimize_headings(ast, analysis, context)
            all_changes.extend(heading_changes)

        # Phase 2: Keyword optimization
        if self._should_continue(context, max_changes):
            keyword_changes = self._optimize_keywords(ast, analysis, context)
            all_changes.extend(keyword_changes)

        # Phase 3: Entity enrichment
        if self._should_continue(context, max_changes):
            entity_changes = self._optimize_entities(ast, analysis, context)
            all_changes.extend(entity_changes)

        # Phase 4: Readability improvement
        if self._should_continue(context, max_changes):
            readability_changes = self._optimize_readability(ast, analysis, context)
            all_changes.extend(readability_changes)

        # Phase 5: Redundancy resolution
        if self._should_continue(context, max_changes):
            redundancy_changes = self._resolve_redundancy(ast, context)
            all_changes.extend(redundancy_changes)

        # Phase 6: FAQ generation (if needed)
        faq_result = self._generate_faq(ast, analysis)
        context.faq_result = faq_result

        # Phase 7: Meta tag generation
        meta_result = self._generate_meta(ast, analysis)
        context.meta_result = meta_result

        # Compute scores
        geo_score = self._compute_geo_score(ast, analysis, context)

        # Build result
        result = OptimizationResult(
            config=self.config,
            changes=all_changes,
            faq_entries=faq_result.faqs if faq_result else [],
            meta_tags=meta_result.meta_tags if meta_result else None,
            optimized_geo_score=geo_score,
            guardrail_warnings=[v for v in context.guardrail_violations if v.severity == "warning"],
            changes_blocked=[v for v in context.guardrail_violations if v.severity == "blocked"],
        )

        return result

    def _should_continue(
        self, context: OptimizationContext, max_changes: int
    ) -> bool:
        """
        Check if optimization should continue.

        Args:
            context: Current optimization context
            max_changes: Maximum allowed changes

        Returns:
            True if should continue
        """
        return context.total_changes < max_changes

    def _optimize_headings(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> list[OptimizationChange]:
        """
        Run heading optimization.

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            List of changes
        """
        heading_analysis = None
        if analysis and hasattr(analysis, "seo_score"):
            heading_analysis = getattr(analysis.seo_score, "heading_analysis", None)

        changes = self.heading_optimizer.optimize(ast, heading_analysis)

        # Update context
        for change in changes:
            context.total_changes += 1
            context.changes_by_type[change.change_type] = (
                context.changes_by_type.get(change.change_type, 0) + 1
            )

        return changes

    def _optimize_keywords(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> list[OptimizationChange]:
        """
        Run keyword optimization.

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            List of changes
        """
        keyword_analysis = None
        if analysis and hasattr(analysis, "seo_score"):
            keyword_analysis = getattr(analysis.seo_score, "keyword_analysis", None)

        changes = self.keyword_injector.inject(ast, keyword_analysis)

        # Update context
        for change in changes:
            context.total_changes += 1
            context.changes_by_type[change.change_type] = (
                context.changes_by_type.get(change.change_type, 0) + 1
            )

        return changes

    def _optimize_entities(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> list[OptimizationChange]:
        """
        Run entity enrichment.

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            List of changes
        """
        semantic_analysis = None
        if analysis and hasattr(analysis, "semantic_score"):
            semantic_analysis = analysis.semantic_score

        changes = self.entity_enricher.enrich(ast, semantic_analysis)

        # Update context
        for change in changes:
            context.total_changes += 1
            context.changes_by_type[change.change_type] = (
                context.changes_by_type.get(change.change_type, 0) + 1
            )

        return changes

    def _optimize_readability(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> list[OptimizationChange]:
        """
        Run readability optimization.

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            List of changes
        """
        readability_analysis = None
        if analysis and hasattr(analysis, "readability_score"):
            readability_analysis = analysis.readability_score

        changes = self.readability_improver.improve(ast, readability_analysis)

        # Update context
        for change in changes:
            context.total_changes += 1
            context.changes_by_type[change.change_type] = (
                context.changes_by_type.get(change.change_type, 0) + 1
            )

        return changes

    def _resolve_redundancy(
        self,
        ast: DocumentAST,
        context: OptimizationContext,
    ) -> list[OptimizationChange]:
        """
        Run redundancy resolution.

        Args:
            ast: Document AST
            context: Optimization context

        Returns:
            List of changes
        """
        # First analyze
        analysis = self.redundancy_resolver.analyze(ast)
        context.redundancy_analysis = analysis

        # Then resolve
        changes = self.redundancy_resolver.resolve(ast, analysis)

        # Update context
        for change in changes:
            context.total_changes += 1
            context.changes_by_type[change.change_type] = (
                context.changes_by_type.get(change.change_type, 0) + 1
            )

        return changes

    def _generate_faq(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
    ) -> FAQGenerationResult:
        """
        Generate FAQ section.

        Args:
            ast: Document AST
            analysis: Content analysis

        Returns:
            FAQ generation result
        """
        return self.faq_generator.generate(ast, analysis)

    def _generate_meta(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
    ) -> MetaGenerationResult:
        """
        Generate meta tags.

        Args:
            ast: Document AST
            analysis: Content analysis

        Returns:
            Meta generation result
        """
        # Try to get existing meta from AST metadata
        if hasattr(ast.metadata, 'title'):
            existing_title = ast.metadata.title
            existing_description = getattr(ast.metadata, 'description', None)
        else:
            # Fallback for dict metadata
            existing_title = ast.metadata.get("title") if isinstance(ast.metadata, dict) else None
            existing_description = ast.metadata.get("description") if isinstance(ast.metadata, dict) else None

        return self.meta_generator.generate(
            ast,
            analysis,
            existing_title=existing_title,
            existing_description=existing_description,
        )

    def _compute_geo_score(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> float:
        """
        Compute the GEO-Metric score.

        Formula: (0.20 × SEO) + (0.30 × Semantic) + (0.30 × AI) + (0.20 × Readability)

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            GEO score 0-100
        """
        component_scores = self._compute_component_scores(ast, analysis, context)

        geo_score = (
            GEO_METRIC_WEIGHTS["seo"] * component_scores.get("seo", 0.0)
            + GEO_METRIC_WEIGHTS["semantic"] * component_scores.get("semantic", 0.0)
            + GEO_METRIC_WEIGHTS["ai_readiness"] * component_scores.get("ai_readiness", 0.0)
            + GEO_METRIC_WEIGHTS["readability"] * component_scores.get("readability", 0.0)
        )

        return round(geo_score, 2)

    def _compute_component_scores(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
        context: OptimizationContext,
    ) -> dict[str, float]:
        """
        Compute individual component scores.

        Args:
            ast: Document AST
            analysis: Content analysis
            context: Optimization context

        Returns:
            Dict of component scores
        """
        scores: dict[str, float] = {
            "seo": 50.0,
            "semantic": 50.0,
            "ai_readiness": 50.0,
            "readability": 50.0,
        }

        # Use analysis scores if available
        if analysis:
            if hasattr(analysis, "seo_score") and analysis.seo_score:
                scores["seo"] = getattr(analysis.seo_score, "overall_score", 50.0)

            if hasattr(analysis, "semantic_score") and analysis.semantic_score:
                scores["semantic"] = getattr(analysis.semantic_score, "overall_score", 50.0)

            if hasattr(analysis, "ai_readiness_score") and analysis.ai_readiness_score:
                scores["ai_readiness"] = getattr(analysis.ai_readiness_score, "overall_score", 50.0)

            if hasattr(analysis, "readability_score") and analysis.readability_score:
                scores["readability"] = getattr(analysis.readability_score, "overall_score", 50.0)

        # Adjust based on optimizations made
        change_counts = context.changes_by_type

        # SEO score boost from keyword/heading changes
        keyword_changes = change_counts.get(ChangeType.KEYWORD, 0)
        structure_changes = change_counts.get(ChangeType.STRUCTURE, 0)
        if keyword_changes > 0 or structure_changes > 0:
            scores["seo"] = min(100.0, scores["seo"] + (keyword_changes + structure_changes) * 2)

        # Semantic score boost from entity changes
        entity_changes = change_counts.get(ChangeType.ENTITY, 0)
        if entity_changes > 0:
            scores["semantic"] = min(100.0, scores["semantic"] + entity_changes * 3)

        # Readability score boost from readability changes
        readability_changes = change_counts.get(ChangeType.READABILITY, 0)
        if readability_changes > 0:
            scores["readability"] = min(100.0, scores["readability"] + readability_changes * 2)

        # AI readiness boost from FAQ and structure
        if context.faq_result and context.faq_result.faqs:
            scores["ai_readiness"] = min(100.0, scores["ai_readiness"] + len(context.faq_result.faqs) * 3)

        # Uniqueness penalty from redundancy
        if context.redundancy_analysis:
            redundancy_penalty = context.redundancy_analysis.redundancy_score * 10
            scores["readability"] = max(0.0, scores["readability"] - redundancy_penalty)

        return scores

    def optimize_incremental(
        self,
        ast: DocumentAST,
        change_types: list[ChangeType],
        analysis: ContentAnalysisResult | None = None,
    ) -> OptimizationResult:
        """
        Run optimization for specific change types only.

        Useful for targeted improvements.

        Args:
            ast: Document AST
            change_types: Types of changes to make
            analysis: Content analysis

        Returns:
            OptimizationResult
        """
        context = OptimizationContext()
        all_changes: list[OptimizationChange] = []

        if ChangeType.HEADING in change_types or ChangeType.STRUCTURE in change_types:
            changes = self._optimize_headings(ast, analysis, context)
            all_changes.extend(changes)

        if ChangeType.KEYWORD in change_types:
            changes = self._optimize_keywords(ast, analysis, context)
            all_changes.extend(changes)

        if ChangeType.ENTITY in change_types:
            changes = self._optimize_entities(ast, analysis, context)
            all_changes.extend(changes)

        if ChangeType.READABILITY in change_types:
            changes = self._optimize_readability(ast, analysis, context)
            all_changes.extend(changes)

        # Compute scores
        geo_score = self._compute_geo_score(ast, analysis, context)

        return OptimizationResult(
            config=self.config,
            changes=all_changes,
            optimized_geo_score=geo_score,
        )

    def get_optimization_summary(
        self, result: OptimizationResult
    ) -> dict:
        """
        Get human-readable optimization summary.

        Args:
            result: Optimization result

        Returns:
            Summary dictionary
        """
        changes_by_type: dict[str, int] = {}
        for change in result.changes:
            type_name = change.change_type.value
            changes_by_type[type_name] = changes_by_type.get(type_name, 0) + 1

        return {
            "total_changes": len(result.changes),
            "changes_by_type": changes_by_type,
            "geo_score": result.geo_score,
            "original_geo_score": result.original_geo_score,
            "optimized_geo_score": result.optimized_geo_score,
            "faq_count": len(result.faq_entries) if result.faq_entries else 0,
            "has_meta_tags": result.meta_tags is not None,
            "guardrail_warnings": len(result.guardrail_warnings) if result.guardrail_warnings else 0,
            "changes_blocked": len(result.changes_blocked) if result.changes_blocked else 0,
        }

    def validate_config(self) -> list[str]:
        """
        Validate optimization configuration.

        Returns:
            List of validation issues
        """
        issues: list[str] = []

        # Check keyword configuration
        if self.config.inject_keywords and not self.config.primary_keyword:
            issues.append("Keyword injection enabled but no primary keyword specified")

        # Check density thresholds
        if self.config.max_keyword_density <= self.config.min_keyword_density:
            issues.append("Max keyword density must be greater than min")

        # Check entity configuration
        if self.config.inject_entities and not self.config.semantic_entities:
            issues.append("Entity injection enabled but no entities specified")

        # Check FAQ configuration
        if self.config.generate_faq and self.config.max_faq_items < 1:
            issues.append("FAQ generation enabled but max items is 0")

        return issues

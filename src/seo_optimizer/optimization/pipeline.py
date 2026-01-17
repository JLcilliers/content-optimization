"""
Optimization Pipeline - End-to-End Integration

Responsibilities:
- Orchestrate full optimization workflow
- Connect analysis → optimization → output
- Handle document transformation
- Track all changes for highlighting

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType

from .content_optimizer import ContentOptimizer
from .models import (
    FAQEntry,
    OptimizationChange,
    OptimizationConfig,
    OptimizationResult,
    PipelineResult,
)

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import ContentAnalysisResult


logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the optimization pipeline."""

    # Input configuration
    input_path: Path | None = None
    input_content: str | None = None

    # Output configuration
    output_path: Path | None = None
    output_format: str = "docx"  # docx, html, markdown

    # Optimization configuration
    optimization_config: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )

    # Processing options
    skip_analysis: bool = False
    dry_run: bool = False  # Preview changes without applying

    # Logging
    verbose: bool = False


@dataclass
class PipelineState:
    """Tracks pipeline execution state."""

    started_at: datetime | None = None
    completed_at: datetime | None = None
    current_phase: str = ""
    phases_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class OptimizationPipeline:
    """
    End-to-end optimization pipeline.

    Workflow:
    1. Input: Read DOCX/content
    2. Parse: Create DocumentAST
    3. Analyze: Run SEO analysis (optional)
    4. Optimize: Apply optimizations
    5. Transform: Apply changes to AST
    6. Output: Generate optimized DOCX

    All new content is tracked for green highlighting.
    """

    def __init__(self, config: PipelineConfig | OptimizationConfig | None = None) -> None:
        """
        Initialize the pipeline.

        Args:
            config: Pipeline or Optimization configuration
        """
        # Accept either PipelineConfig or OptimizationConfig for convenience
        if isinstance(config, OptimizationConfig):
            self.config = PipelineConfig(optimization_config=config)
        else:
            self.config = config or PipelineConfig()
        self.state = PipelineState()
        self.optimizer = ContentOptimizer(self.config.optimization_config)

    def optimize(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None = None,
    ) -> PipelineResult:
        """
        Convenience method to optimize an AST directly.

        Args:
            ast: DocumentAST to optimize
            analysis: Optional pre-computed analysis

        Returns:
            PipelineResult with optimization outputs
        """
        return self.run(ast=ast, analysis=analysis)

    def run(
        self,
        ast: DocumentAST | None = None,
        analysis: ContentAnalysisResult | None = None,
    ) -> PipelineResult:
        """
        Run the full optimization pipeline.

        Args:
            ast: Pre-parsed DocumentAST (or None to parse from input)
            analysis: Pre-computed analysis (or None to analyze)

        Returns:
            PipelineResult with all outputs
        """
        self.state = PipelineState(started_at=datetime.now())

        try:
            # Phase 1: Input
            self._set_phase("input")
            if ast is None:
                ast = self._load_and_parse_input()

            if ast is None:
                self.state.errors.append("No input content provided")
                return self._create_error_result()

            # Store original content
            original_ast = self._clone_ast(ast)

            # Phase 2: Analysis (optional)
            self._set_phase("analysis")
            if not self.config.skip_analysis and analysis is None:
                analysis = self._run_analysis(ast)

            # Phase 3: Optimization
            self._set_phase("optimization")
            optimization_result = self.optimizer.optimize(ast, analysis)

            # Phase 4: Transformation
            self._set_phase("transformation")
            optimized_ast = self._apply_changes(ast, optimization_result)

            # Phase 5: Change tracking
            self._set_phase("change_tracking")
            change_map = self._build_change_map(
                original_ast, optimized_ast, optimization_result
            )

            # Phase 6: Output (unless dry run)
            self._set_phase("output")
            output_path = None
            if not self.config.dry_run and self.config.output_path:
                output_path = self._generate_output(optimized_ast, change_map)

            # Complete
            self.state.completed_at = datetime.now()
            self._set_phase("complete")

            return PipelineResult(
                success=True,
                original_ast=original_ast,
                optimized_ast=optimized_ast,
                optimization_result=optimization_result,
                output_path=output_path,
                change_map=change_map,
                execution_time=(
                    self.state.completed_at - self.state.started_at
                ).total_seconds(),
                warnings=self.state.warnings,
            )

        except Exception as e:
            logger.exception("Pipeline error")
            self.state.errors.append(str(e))
            return self._create_error_result()

    def _set_phase(self, phase: str) -> None:
        """Update current phase."""
        if self.state.current_phase:
            self.state.phases_completed.append(self.state.current_phase)
        self.state.current_phase = phase

        if self.config.verbose:
            logger.info(f"Pipeline phase: {phase}")

    def _load_and_parse_input(self) -> DocumentAST | None:
        """
        Load and parse input content.

        Returns:
            DocumentAST or None
        """
        if self.config.input_content:
            # Parse from string content
            return self._parse_text_content(self.config.input_content)

        if self.config.input_path:
            # Load from file
            path = Path(self.config.input_path)

            if not path.exists():
                self.state.errors.append(f"Input file not found: {path}")
                return None

            # Read based on extension
            if path.suffix.lower() == ".docx":
                return self._load_docx(path)
            elif path.suffix.lower() in [".md", ".markdown"]:
                return self._load_markdown(path)
            elif path.suffix.lower() in [".txt", ".html"]:
                return self._load_text(path)
            else:
                self.state.errors.append(f"Unsupported file type: {path.suffix}")
                return None

        return None

    def _parse_text_content(self, content: str) -> DocumentAST:
        """
        Parse plain text content into AST.

        Args:
            content: Text content

        Returns:
            DocumentAST
        """
        nodes: list[ContentNode] = []
        node_id = 0

        # Simple paragraph-based parsing
        paragraphs = content.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if it's a heading (starts with #)
            if para.startswith("#"):
                level = len(para) - len(para.lstrip("#"))
                text = para.lstrip("#").strip()

                nodes.append(
                    ContentNode(
                        node_id=f"node_{node_id}",
                        node_type=NodeType.HEADING,
                        text_content=text,
                        metadata={"level": level},
                    )
                )
            else:
                nodes.append(
                    ContentNode(
                        node_id=f"node_{node_id}",
                        node_type=NodeType.PARAGRAPH,
                        text_content=para,
                    )
                )

            node_id += 1

        return DocumentAST(nodes=nodes, metadata={})

    def _load_docx(self, path: Path) -> DocumentAST | None:
        """Load DOCX file (placeholder - uses docx_io module)."""
        # In full implementation, this would use seo_optimizer.io.docx_io
        self.state.warnings.append("DOCX loading: Using placeholder parser")
        content = path.read_text(encoding="utf-8", errors="ignore")
        return self._parse_text_content(content)

    def _load_markdown(self, path: Path) -> DocumentAST | None:
        """Load Markdown file."""
        content = path.read_text(encoding="utf-8")
        return self._parse_text_content(content)

    def _load_text(self, path: Path) -> DocumentAST | None:
        """Load plain text file."""
        content = path.read_text(encoding="utf-8")
        return self._parse_text_content(content)

    def _run_analysis(
        self, ast: DocumentAST
    ) -> ContentAnalysisResult | None:
        """
        Run content analysis.

        Args:
            ast: Document AST

        Returns:
            Analysis result or None
        """
        # In full implementation, this would use seo_optimizer.analysis
        # For now, return None and let optimizer use defaults
        self.state.warnings.append("Analysis: Using default analysis")
        return None

    def _clone_ast(self, ast: DocumentAST) -> DocumentAST:
        """
        Create a deep copy of the AST.

        Args:
            ast: AST to clone

        Returns:
            Cloned AST
        """
        from dataclasses import asdict

        cloned_nodes = []

        for node in ast.nodes:
            cloned_node = ContentNode(
                node_id=node.node_id,
                node_type=node.node_type,
                text_content=node.text_content,
                children=list(node.children) if node.children else [],
                metadata=dict(node.metadata) if node.metadata else {},
            )
            cloned_nodes.append(cloned_node)

        # Handle metadata - convert DocumentMetadata to dict if needed
        if hasattr(ast.metadata, '__dataclass_fields__'):
            metadata = asdict(ast.metadata)
        elif isinstance(ast.metadata, dict):
            metadata = dict(ast.metadata)
        else:
            metadata = {}

        return DocumentAST(
            nodes=cloned_nodes,
            metadata=metadata,
        )

    def _apply_changes(
        self,
        ast: DocumentAST,
        result: OptimizationResult,
    ) -> DocumentAST:
        """
        Apply optimization changes to AST.

        Creates a new AST with changes applied.

        Args:
            ast: Original AST
            result: Optimization result with changes

        Returns:
            Modified AST
        """
        # Clone to avoid modifying original
        modified_ast = self._clone_ast(ast)

        # Group changes by section
        changes_by_section: dict[str, list[OptimizationChange]] = {}

        for change in result.changes:
            section_id = change.section_id or "document"
            if section_id not in changes_by_section:
                changes_by_section[section_id] = []
            changes_by_section[section_id].append(change)

        # Apply changes to nodes
        for node in modified_ast.nodes:
            node_changes = changes_by_section.get(node.node_id, [])

            for change in node_changes:
                if change.original and change.optimized:
                    # Simple text replacement
                    # In production, would use more sophisticated matching
                    if change.original in node.text_content:
                        node.text_content = node.text_content.replace(
                            change.original, change.optimized, 1
                        )

        # Add FAQ section if generated
        if result.faq_entries:
            faq_nodes = self._create_faq_nodes(result.faq_entries)
            modified_ast.nodes.extend(faq_nodes)

        # Update metadata with meta tags
        if result.meta_tags:
            if hasattr(modified_ast.metadata, 'title'):
                modified_ast.metadata.title = result.meta_tags.title
                modified_ast.metadata.description = result.meta_tags.description
            elif isinstance(modified_ast.metadata, dict):
                modified_ast.metadata["title"] = result.meta_tags.title
                modified_ast.metadata["description"] = result.meta_tags.description

        return modified_ast

    def _create_faq_nodes(
        self, faq_entries: list[FAQEntry]
    ) -> list[ContentNode]:
        """
        Create AST nodes for FAQ section.

        Args:
            faq_entries: FAQ entries to add

        Returns:
            List of content nodes
        """
        nodes: list[ContentNode] = []

        # Add FAQ heading
        nodes.append(
            ContentNode(
                node_id="faq_heading",
                node_type=NodeType.HEADING,
                text_content="Frequently Asked Questions",
                metadata={"level": 2, "is_new": True},
            )
        )

        # Add each FAQ
        for i, faq in enumerate(faq_entries):
            # Question as H3
            nodes.append(
                ContentNode(
                    node_id=f"faq_q_{i}",
                    node_type=NodeType.HEADING,
                    text_content=faq.question,
                    metadata={"level": 3, "is_new": True, "html_id": faq.html_id},
                )
            )

            # Answer as paragraph
            nodes.append(
                ContentNode(
                    node_id=f"faq_a_{i}",
                    node_type=NodeType.PARAGRAPH,
                    text_content=faq.answer,
                    metadata={"is_new": True},
                )
            )

        return nodes

    def _build_change_map(
        self,
        original: DocumentAST,
        optimized: DocumentAST,
        result: OptimizationResult,
    ) -> dict[str, Any]:
        """
        Build map of all changes for highlighting.

        Args:
            original: Original AST
            optimized: Optimized AST
            result: Optimization result

        Returns:
            Change map for highlighting
        """
        change_map: dict[str, Any] = {
            "new_nodes": [],
            "modified_nodes": [],
            "text_insertions": [],
            "faq_section": None,
            "meta_changes": {},
        }

        # Track new nodes (like FAQ)
        original_ids = {n.node_id for n in original.nodes}

        for node in optimized.nodes:
            if node.node_id not in original_ids:
                change_map["new_nodes"].append({
                    "node_id": node.node_id,
                    "type": node.node_type.value,
                    "content": node.text_content,
                })

                if node.node_id.startswith("faq_"):
                    if change_map["faq_section"] is None:
                        change_map["faq_section"] = {
                            "start_node": node.node_id,
                            "entries": [],
                        }

        # Track text modifications from changes
        for change in result.changes:
            if change.original and change.optimized and change.original != change.optimized:
                change_map["text_insertions"].append({
                    "section_id": change.section_id,
                    "original": change.original,
                    "new": change.optimized,
                    "type": change.change_type.value,
                })

                if change.section_id:
                    change_map["modified_nodes"].append(change.section_id)

        # Track meta changes
        if result.meta_tags:
            change_map["meta_changes"] = {
                "title": result.meta_tags.title,
                "description": result.meta_tags.description,
            }

        return change_map

    def _generate_output(
        self,
        ast: DocumentAST,
        change_map: dict[str, Any],
    ) -> Path | None:
        """
        Generate output file with highlighting.

        Args:
            ast: Optimized AST
            change_map: Change tracking map

        Returns:
            Output path or None
        """
        if not self.config.output_path:
            return None

        output_path = Path(self.config.output_path)

        # Generate based on format
        if self.config.output_format == "docx":
            # In full implementation, use seo_optimizer.io.docx_io
            self.state.warnings.append("DOCX output: Using placeholder generator")
            self._generate_markdown_output(ast, change_map, output_path)

        elif self.config.output_format == "markdown":
            self._generate_markdown_output(ast, change_map, output_path)

        elif self.config.output_format == "html":
            self._generate_html_output(ast, change_map, output_path)

        return output_path

    def _generate_markdown_output(
        self,
        ast: DocumentAST,
        change_map: dict[str, Any],
        path: Path,
    ) -> None:
        """Generate Markdown output."""
        lines: list[str] = []
        new_node_ids = {n["node_id"] for n in change_map.get("new_nodes", [])}

        for node in ast.nodes:
            is_new = node.node_id in new_node_ids or node.metadata.get("is_new", False)

            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                prefix = "#" * level
                text = node.text_content

                if is_new:
                    text = f"**[NEW]** {text}"

                lines.append(f"{prefix} {text}")
                lines.append("")

            elif node.node_type == NodeType.PARAGRAPH:
                text = node.text_content

                if is_new:
                    text = f"**[NEW]** {text}"

                lines.append(text)
                lines.append("")

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_html_output(
        self,
        ast: DocumentAST,
        change_map: dict[str, Any],
        path: Path,
    ) -> None:
        """Generate HTML output with highlighting."""
        lines: list[str] = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<style>",
            ".new-content { background-color: #90EE90; }",  # Light green
            ".modified { background-color: #FFFFE0; }",  # Light yellow
            "</style>",
            "</head>",
            "<body>",
        ]

        new_node_ids = {n["node_id"] for n in change_map.get("new_nodes", [])}
        modified_ids = set(change_map.get("modified_nodes", []))

        for node in ast.nodes:
            is_new = node.node_id in new_node_ids or node.metadata.get("is_new", False)
            is_modified = node.node_id in modified_ids

            css_class = ""
            if is_new:
                css_class = ' class="new-content"'
            elif is_modified:
                css_class = ' class="modified"'

            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", 2)
                html_id = node.metadata.get("html_id", "")
                id_attr = f' id="{html_id}"' if html_id else ""
                lines.append(f"<h{level}{id_attr}{css_class}>{node.text_content}</h{level}>")

            elif node.node_type == NodeType.PARAGRAPH:
                lines.append(f"<p{css_class}>{node.text_content}</p>")

        lines.extend(["</body>", "</html>"])

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")

    def _create_error_result(self) -> PipelineResult:
        """Create error result."""
        return PipelineResult(
            success=False,
            errors=self.state.errors,
            warnings=self.state.warnings,
        )

    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state


def optimize_content(
    content: str,
    primary_keyword: str,
    secondary_keywords: list[str] | None = None,
    semantic_entities: list[str] | None = None,
    **config_kwargs: Any,
) -> PipelineResult:
    """
    Convenience function for quick optimization.

    Args:
        content: Content to optimize
        primary_keyword: Primary target keyword
        secondary_keywords: Secondary keywords
        semantic_entities: Semantic entities to include
        **config_kwargs: Additional config options

    Returns:
        PipelineResult
    """
    opt_config = OptimizationConfig(
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords or [],
        semantic_entities=semantic_entities or [],
        **config_kwargs,
    )

    pipeline_config = PipelineConfig(
        input_content=content,
        optimization_config=opt_config,
        dry_run=True,  # Don't write output
    )

    pipeline = OptimizationPipeline(pipeline_config)
    return pipeline.run()

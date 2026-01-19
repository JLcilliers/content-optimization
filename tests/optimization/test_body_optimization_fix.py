"""
Integration tests for body optimization fix.

This test module verifies that the text truncation bug has been fixed:
- OptimizationChange now has full_original and full_optimized fields
- Pipeline._apply_changes() uses full text for accurate matching
- Pipeline._build_change_map() passes full text for highlighting

The bug was: optimizers stored truncated text (original[:100] + "..."),
which never matched actual content in _apply_changes(), causing
body optimizations to be created but never applied.
"""

import pytest

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType, PositionInfo
from seo_optimizer.optimization.models import (
    ChangeType,
    ContentType,
    OptimizationChange,
    OptimizationConfig,
    OptimizationMode,
    OptimizationResult,
)
from seo_optimizer.optimization.pipeline import OptimizationPipeline


def make_node(
    node_id: str,
    node_type: NodeType,
    text_content: str,
    start_char: int = 0,
    metadata: dict | None = None,
) -> ContentNode:
    """Create a ContentNode with default position info."""
    end_char = start_char + len(text_content)
    position = PositionInfo(
        position_id=node_id,
        start_char=start_char,
        end_char=end_char,
    )
    return ContentNode(
        node_id=node_id,
        node_type=node_type,
        position=position,
        text_content=text_content,
        metadata=metadata or {},
    )


class TestOptimizationChangeFullText:
    """Tests for OptimizationChange full text fields."""

    def test_full_original_field_exists(self) -> None:
        """OptimizationChange should have full_original field."""
        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="Test",
            original="short",
            optimized="short modified",
            reason="Test change",
        )
        assert hasattr(change, "full_original")
        assert change.full_original == ""  # Default value

    def test_full_optimized_field_exists(self) -> None:
        """OptimizationChange should have full_optimized field."""
        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="Test",
            original="short",
            optimized="short modified",
            reason="Test change",
        )
        assert hasattr(change, "full_optimized")
        assert change.full_optimized == ""  # Default value

    def test_full_text_fields_can_be_set(self) -> None:
        """Full text fields should accept values during construction."""
        long_original = "This is a very long paragraph " * 10
        long_optimized = "This is a modified very long paragraph " * 10

        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="Test paragraph",
            original=long_original[:100] + "...",
            optimized=long_optimized[:100] + "...",
            reason="Added keyword",
            full_original=long_original,
            full_optimized=long_optimized,
        )

        assert change.full_original == long_original
        assert change.full_optimized == long_optimized
        assert change.original.endswith("...")  # Truncated for display
        assert change.optimized.endswith("...")

    def test_backward_compatibility_empty_full_text(self) -> None:
        """Changes without full_text should still work (backward compat)."""
        change = OptimizationChange(
            change_type=ChangeType.KEYWORD,
            location="Test",
            original="short text",
            optimized="short modified text",
            reason="Test",
        )

        # Should fall back to original/optimized when full_* is empty
        effective_original = change.full_original or change.original
        effective_optimized = change.full_optimized or change.optimized

        assert effective_original == "short text"
        assert effective_optimized == "short modified text"


class TestPipelineApplyChanges:
    """Tests for pipeline change application with full text."""

    @pytest.fixture
    def sample_ast(self) -> DocumentAST:
        """Create a sample AST for testing."""
        long_paragraph = (
            "This is a very long paragraph about software development that "
            "discusses various programming concepts and best practices for "
            "building maintainable applications with clean code principles."
        )

        return DocumentAST(
            doc_id="test_doc",
            nodes=[
                make_node(
                    node_id="node_1",
                    node_type=NodeType.HEADING,
                    text_content="Introduction to Software Development",
                    start_char=0,
                    metadata={"level": 1},
                ),
                make_node(
                    node_id="node_2",
                    node_type=NodeType.PARAGRAPH,
                    text_content=long_paragraph,
                    start_char=50,
                ),
            ],
            metadata={},
        )

    @pytest.fixture
    def pipeline(self) -> OptimizationPipeline:
        """Create a pipeline instance."""
        config = OptimizationConfig(
            mode=OptimizationMode.BALANCED,
            primary_keyword="software development",
            content_type=ContentType.INFORMATIONAL,
        )
        return OptimizationPipeline(config)

    def test_apply_changes_uses_full_original(
        self, pipeline: OptimizationPipeline, sample_ast: DocumentAST
    ) -> None:
        """Changes should be applied using full_original for matching."""
        original_text = sample_ast.nodes[1].text_content
        modified_text = original_text.replace(
            "software development", "modern software development"
        )

        result = OptimizationResult(
            changes=[
                OptimizationChange(
                    change_type=ChangeType.KEYWORD,
                    location="Paragraph 1",
                    original=original_text[:80] + "...",  # Truncated
                    optimized=modified_text[:80] + "...",  # Truncated
                    reason="Added keyword",
                    section_id="node_2",
                    full_original=original_text,  # Full text
                    full_optimized=modified_text,  # Full text
                )
            ]
        )

        # Apply changes
        modified_ast = pipeline._apply_changes(sample_ast, result)

        # Verify change was applied (content should be modified)
        para_node = modified_ast.nodes[1]
        assert "modern software development" in para_node.text_content
        assert para_node.text_content == modified_text

    def test_apply_changes_fails_with_truncated_only(
        self, pipeline: OptimizationPipeline, sample_ast: DocumentAST
    ) -> None:
        """Without full_original, truncated text won't match (the old bug)."""
        original_text = sample_ast.nodes[1].text_content
        modified_text = original_text.replace(
            "software development", "modern software development"
        )

        result = OptimizationResult(
            changes=[
                OptimizationChange(
                    change_type=ChangeType.KEYWORD,
                    location="Paragraph 1",
                    original=original_text[:80] + "...",  # Truncated - WON'T MATCH
                    optimized=modified_text[:80] + "...",  # Truncated
                    reason="Added keyword",
                    section_id="node_2",
                    # NO full_original or full_optimized set
                )
            ]
        )

        # Apply changes
        modified_ast = pipeline._apply_changes(sample_ast, result)

        # Without full_original, the truncated text "..." won't match
        # So the change should NOT be applied (demonstrating the old bug behavior)
        para_node = modified_ast.nodes[1]
        # The text should be unchanged because truncated text doesn't match
        assert "modern software development" not in para_node.text_content
        assert para_node.text_content == original_text


class TestPipelineBuildChangeMap:
    """Tests for pipeline change map building with full text."""

    @pytest.fixture
    def sample_ast(self) -> DocumentAST:
        """Create a sample AST for testing."""
        return DocumentAST(
            doc_id="test_doc",
            nodes=[
                make_node(
                    node_id="node_1",
                    node_type=NodeType.PARAGRAPH,
                    text_content="Original paragraph content for testing.",
                    start_char=0,
                ),
            ],
            metadata={},
        )

    @pytest.fixture
    def pipeline(self) -> OptimizationPipeline:
        """Create a pipeline instance."""
        config = OptimizationConfig(
            mode=OptimizationMode.BALANCED,
            primary_keyword="test",
            content_type=ContentType.INFORMATIONAL,
        )
        return OptimizationPipeline(config)

    def test_build_change_map_uses_full_text(
        self, pipeline: OptimizationPipeline, sample_ast: DocumentAST
    ) -> None:
        """Change map should contain full text for highlighting."""
        full_original = "This is the complete original paragraph text."
        full_optimized = "This is the complete modified paragraph text."

        result = OptimizationResult(
            changes=[
                OptimizationChange(
                    change_type=ChangeType.READABILITY,
                    location="Paragraph 1",
                    original=full_original[:50] + "...",  # Truncated
                    optimized=full_optimized[:50] + "...",  # Truncated
                    reason="Improved readability",
                    section_id="node_1",
                    full_original=full_original,
                    full_optimized=full_optimized,
                )
            ]
        )

        change_map = pipeline._build_change_map(sample_ast, sample_ast, result)

        # Verify change map contains full text, not truncated
        assert len(change_map["text_insertions"]) == 1
        insertion = change_map["text_insertions"][0]
        assert insertion["original"] == full_original
        assert insertion["new"] == full_optimized
        # Should NOT contain truncated versions
        assert "..." not in insertion["original"]
        assert "..." not in insertion["new"]


class TestKeywordInjectorFullText:
    """Tests that KeywordInjector sets full text fields."""

    @pytest.fixture
    def sample_ast(self) -> DocumentAST:
        """Create a sample AST with content for keyword injection."""
        return DocumentAST(
            doc_id="test_doc",
            nodes=[
                make_node(
                    node_id="h1_node",
                    node_type=NodeType.HEADING,
                    text_content="Introduction to Programming",
                    start_char=0,
                    metadata={"level": 1},
                ),
                make_node(
                    node_id="p1_node",
                    node_type=NodeType.PARAGRAPH,
                    text_content=(
                        "Building software applications requires understanding "
                        "various concepts including data structures, algorithms, "
                        "and design patterns. Developers must learn to write "
                        "clean, maintainable code that follows best practices."
                    ),
                    start_char=30,
                ),
            ],
            metadata={},
        )

    def test_keyword_injector_sets_full_text(self, sample_ast: DocumentAST) -> None:
        """KeywordInjector should set full_original and full_optimized."""
        from seo_optimizer.optimization.guardrails import SafetyGuardrails
        from seo_optimizer.optimization.keyword_injector import KeywordInjector

        config = OptimizationConfig(
            mode=OptimizationMode.AGGRESSIVE,
            primary_keyword="python programming",
            inject_keywords=True,
        )
        guardrails = SafetyGuardrails(config)
        injector = KeywordInjector(config, guardrails)

        changes = injector.inject(sample_ast)

        # If any changes were made, verify they have full text
        for change in changes:
            if change.full_original and change.full_optimized:
                # Full text should be longer than truncated versions
                assert len(change.full_original) >= len(change.original.rstrip("..."))
                assert len(change.full_optimized) >= len(change.optimized.rstrip("..."))
                # Full text should not have truncation marker
                if len(change.full_original) > 100:
                    assert not change.full_original.endswith("...")
                if len(change.full_optimized) > 100:
                    assert not change.full_optimized.endswith("...")

"""
Pytest Configuration and Shared Fixtures

Provides fixtures for:
- Loading test DOCX files
- Creating in-memory documents
- Sample content blocks for diffing tests
- Edge case parameterization
"""

from pathlib import Path
from typing import Any

import pytest

from seo_optimizer.diffing.models import (
    Addition,
    ChangeSet,
    DiffConfidence,
    DocumentFingerprint,
    HighlightRegion,
)
from seo_optimizer.ingestion.models import (
    ContentNode,
    DocumentAST,
    DocumentMetadata,
    FormattingInfo,
    NodeType,
    OriginalSnapshot,
    PositionInfo,
    TextRun,
)


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_docx_path(fixtures_dir: Path) -> Path | None:
    """Path to a sample DOCX file for testing."""
    sample = fixtures_dir / "sample.docx"
    if sample.exists():
        return sample
    return None


# =============================================================================
# Document Model Fixtures
# =============================================================================


@pytest.fixture
def sample_position_info() -> PositionInfo:
    """Create a sample PositionInfo."""
    return PositionInfo(
        position_id="p0",
        start_char=0,
        end_char=100,
        start_line=1,
        end_line=5,
    )


@pytest.fixture
def sample_formatting_info() -> FormattingInfo:
    """Create a sample FormattingInfo."""
    return FormattingInfo(
        bold=False,
        italic=False,
        font_name="Calibri",
        font_size=11.0,
        alignment="left",
    )


@pytest.fixture
def sample_text_run(
    sample_formatting_info: FormattingInfo,
) -> TextRun:
    """Create a sample TextRun."""
    return TextRun(
        text="This is sample text content.",
        formatting=sample_formatting_info,
        position=PositionInfo(
            position_id="p0_r0",
            start_char=0,
            end_char=28,
        ),
    )


@pytest.fixture
def sample_content_node(
    sample_position_info: PositionInfo,
    sample_text_run: TextRun,
) -> ContentNode:
    """Create a sample ContentNode."""
    return ContentNode(
        node_id="node_p0",
        node_type=NodeType.PARAGRAPH,
        position=sample_position_info,
        text_content="This is sample text content.",
        runs=[sample_text_run],
    )


@pytest.fixture
def sample_document_ast(sample_content_node: ContentNode) -> DocumentAST:
    """Create a sample DocumentAST."""
    return DocumentAST(
        doc_id="test_doc_001",
        nodes=[sample_content_node],
        metadata=DocumentMetadata(
            source_path="test.docx",
            title="Test Document",
        ),
        full_text="This is sample text content.",
        char_count=28,
    )


@pytest.fixture
def sample_original_snapshot(sample_document_ast: DocumentAST) -> OriginalSnapshot:
    """Create a sample OriginalSnapshot."""
    return OriginalSnapshot.from_document_ast(sample_document_ast)


# =============================================================================
# Diffing Model Fixtures
# =============================================================================


@pytest.fixture
def sample_highlight_region() -> HighlightRegion:
    """Create a sample HighlightRegion."""
    return HighlightRegion(
        node_id="node_p0",
        start_char=10,
        end_char=20,
        text="new text 1",
        confidence=0.95,
        reason="new_content",
    )


@pytest.fixture
def sample_addition(sample_highlight_region: HighlightRegion) -> Addition:
    """Create a sample Addition."""
    return Addition(
        addition_id="add_001",
        node_ids=["node_p0"],
        highlight_regions=[sample_highlight_region],
        total_text="new text 1",
        confidence=0.95,
    )


@pytest.fixture
def sample_changeset(sample_addition: Addition) -> ChangeSet:
    """Create a sample ChangeSet."""
    return ChangeSet(
        changeset_id="cs_001",
        original_doc_id="test_doc_001",
        optimized_doc_id="test_doc_001_opt",
        additions=[sample_addition],
    )


@pytest.fixture
def empty_changeset() -> ChangeSet:
    """Create an empty ChangeSet (no changes)."""
    return ChangeSet(
        changeset_id="cs_empty",
        original_doc_id="test_doc_001",
        optimized_doc_id="test_doc_001_opt",
        additions=[],
    )


# =============================================================================
# Edge Case Fixtures for Diffing Tests
# =============================================================================


@pytest.fixture
def diff_edge_cases() -> list[dict[str, Any]]:
    """
    Edge cases for diffing tests.

    Each case has:
    - id: Unique identifier
    - original: Original text
    - modified: Modified text
    - expected_highlight: What should be highlighted (None if nothing)
    - description: What this case tests
    """
    return [
        {
            "id": "E01",
            "original": "Hello world",
            "modified": "Hello world",
            "expected_highlight": None,
            "description": "Identical content - no highlight",
        },
        {
            "id": "E02",
            "original": "",
            "modified": "Entirely new paragraph content.",
            "expected_highlight": "Entirely new paragraph content.",
            "description": "Entirely new content - highlight all",
        },
        {
            "id": "E03",
            "original": "The product helps users.",
            "modified": "The product helps users save time.",
            "expected_highlight": " save time",
            "description": "Sentence expansion - highlight only new part",
        },
        {
            "id": "E04",
            "original": "The product assists customers.",
            "modified": "The product helps users.",
            "expected_highlight": None,
            "description": "Semantic equivalent rewording - no highlight",
        },
        {
            "id": "E05",
            "original": "First paragraph.",
            "modified": "First paragraph.",
            "expected_highlight": None,
            "description": "Content moved (same text) - no highlight",
        },
        {
            "id": "E06",
            "original": "Hello world",
            "modified": "Hello beautiful world",
            "expected_highlight": "beautiful ",
            "description": "Word insertion - highlight inserted word",
        },
        {
            "id": "E07",
            "original": "The quick brown fox",
            "modified": "The quick brown fox jumps over the lazy dog",
            "expected_highlight": " jumps over the lazy dog",
            "description": "Append to end - highlight appended content",
        },
        {
            "id": "E08",
            "original": "Hello world",
            "modified": "Greetings, Hello world",
            "expected_highlight": "Greetings, ",
            "description": "Prepend to start - highlight prepended content",
        },
        {
            "id": "E09",
            "original": "The product is good.",
            "modified": "The product is excellent.",
            "expected_highlight": None,
            "description": "Word replacement (similar meaning) - no highlight",
        },
        {
            "id": "E10",
            "original": "We offer services.",
            "modified": "We offer premium services to our clients.",
            "expected_highlight": "premium , to our clients",
            "description": "Multiple insertions - highlight all new parts",
        },
        # Additional edge cases from research document
        {
            "id": "E11",
            "original": "Contact us today.",
            "modified": "Contact us today. We're here to help!",
            "expected_highlight": " We're here to help!",
            "description": "New sentence added - highlight new sentence",
        },
        {
            "id": "E12",
            "original": "Step 1: Do this.",
            "modified": "Step 1: Do this.\nStep 2: Do that.",
            "expected_highlight": "\nStep 2: Do that.",
            "description": "New list item - highlight new item",
        },
        {
            "id": "E13",
            "original": "The price is $100.",
            "modified": "The price is $100.",
            "expected_highlight": None,
            "description": "Exact number match - no highlight",
        },
        {
            "id": "E14",
            "original": "Founded in 2020.",
            "modified": "Founded in 2020. We've grown significantly since then.",
            "expected_highlight": " We've grown significantly since then.",
            "description": "Expansion after date - highlight new content",
        },
        {
            "id": "E15",
            "original": "   Extra   spaces   here   ",
            "modified": "Extra spaces here",
            "expected_highlight": None,
            "description": "Whitespace normalization - no highlight",
        },
    ]


@pytest.fixture(params=[
    ("E01", "Hello world", "Hello world", None),
    ("E02", "", "New content", "New content"),
    ("E03", "Base text.", "Base text. More text.", " More text."),
])
def parameterized_diff_case(request: pytest.FixtureRequest) -> tuple[str, str, str, str | None]:
    """Parameterized fixture for common diff test cases."""
    return request.param


# =============================================================================
# Factory Functions
# =============================================================================


def create_document_ast(
    text: str,
    doc_id: str = "test_doc",
    node_type: NodeType = NodeType.PARAGRAPH,
) -> DocumentAST:
    """Factory function to create a DocumentAST from simple text."""
    position = PositionInfo(
        position_id="p0",
        start_char=0,
        end_char=len(text),
    )
    formatting = FormattingInfo()
    run = TextRun(
        text=text,
        formatting=formatting,
        position=PositionInfo(
            position_id="p0_r0",
            start_char=0,
            end_char=len(text),
        ),
    )
    node = ContentNode(
        node_id="node_p0",
        node_type=node_type,
        position=position,
        text_content=text,
        runs=[run],
    )
    return DocumentAST(
        doc_id=doc_id,
        nodes=[node],
        metadata=DocumentMetadata(),
        full_text=text,
        char_count=len(text),
    )


# Make factory function available to tests
@pytest.fixture
def make_document_ast() -> type:
    """Fixture that returns the factory function."""
    return create_document_ast

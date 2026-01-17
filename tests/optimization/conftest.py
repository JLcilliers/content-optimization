"""
Shared fixtures for optimization tests.
"""

import pytest

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType, PositionInfo


def make_node(
    node_id: str,
    node_type: NodeType,
    text_content: str,
    start_char: int = 0,
    metadata: dict = None,
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


@pytest.fixture
def make_content_node():
    """Fixture that returns a node factory function."""
    char_offset = [0]  # Mutable to track position

    def _make_node(
        node_id: str,
        node_type: NodeType,
        text_content: str,
        metadata: dict = None,
    ) -> ContentNode:
        position = PositionInfo(
            position_id=node_id,
            start_char=char_offset[0],
            end_char=char_offset[0] + len(text_content),
        )
        char_offset[0] += len(text_content) + 1  # +1 for separator
        return ContentNode(
            node_id=node_id,
            node_type=node_type,
            position=position,
            text_content=text_content,
            metadata=metadata or {},
        )

    return _make_node


@pytest.fixture
def simple_ast(make_content_node):
    """Create a simple document AST for testing."""
    nodes = [
        make_content_node(
            "h1",
            NodeType.HEADING,
            "Test Document Title",
            {"level": 1},
        ),
        make_content_node(
            "p1",
            NodeType.PARAGRAPH,
            "This is a test paragraph with some content.",
        ),
        make_content_node(
            "h2",
            NodeType.HEADING,
            "Section Two",
            {"level": 2},
        ),
        make_content_node(
            "p2",
            NodeType.PARAGRAPH,
            "Another paragraph with more content for testing.",
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})

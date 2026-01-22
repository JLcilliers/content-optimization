"""
Markdown Writer - Convert DocumentAST back to markdown with change indicators.

This module converts the optimized DocumentAST back into markdown format,
with optional markers to indicate new/changed content.
"""

from dataclasses import dataclass
from typing import Literal

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType


@dataclass
class MarkdownWriterConfig:
    """Configuration for markdown writer."""

    # How to indicate new content
    # 'comments' - Use HTML comments <!-- NEW --> ... <!-- /NEW -->
    # 'bold' - Wrap new content in **bold**
    # 'highlight' - Use ==highlight== markers (some markdown flavors)
    # 'none' - No special marking
    new_content_indicator: Literal["comments", "bold", "highlight", "none"] = "comments"

    # Whether to add a summary header with change statistics
    include_change_summary: bool = True

    # Heading style: 'atx' (#, ##) or 'setext' (underlines)
    heading_style: Literal["atx", "setext"] = "atx"

    # List marker for bullet lists
    bullet_marker: str = "-"

    # Add blank lines between elements
    add_blank_lines: bool = True


class MarkdownWriter:
    """
    Convert DocumentAST to markdown format.

    Supports marking new content with various indicators.
    """

    def __init__(self, config: MarkdownWriterConfig | None = None):
        """Initialize writer with optional config."""
        self.config = config or MarkdownWriterConfig()

    def write(
        self,
        ast: DocumentAST,
        change_map: dict[str, str] | None = None,
    ) -> str:
        """
        Convert DocumentAST to markdown string.

        Args:
            ast: DocumentAST to convert
            change_map: Optional dict mapping node_ids to change types
                       (e.g., {"p5": "new", "h2": "modified"})

        Returns:
            Markdown string representation
        """
        lines: list[str] = []

        # Add change summary header if enabled and there are changes
        if self.config.include_change_summary and change_map:
            new_count = sum(1 for v in change_map.values() if v == "new")
            modified_count = sum(1 for v in change_map.values() if v == "modified")

            if new_count > 0 or modified_count > 0:
                lines.append("<!-- OPTIMIZATION SUMMARY")
                if new_count > 0:
                    lines.append(f"New content additions: {new_count}")
                if modified_count > 0:
                    lines.append(f"Modified sections: {modified_count}")
                lines.append("-->")
                lines.append("")

        # Convert nodes
        for node in ast.nodes:
            node_lines = self._convert_node(node, change_map)
            lines.extend(node_lines)
            if self.config.add_blank_lines:
                lines.append("")

        # Clean up extra blank lines at end
        while lines and lines[-1] == "":
            lines.pop()

        return "\n".join(lines)

    def _convert_node(
        self,
        node: ContentNode,
        change_map: dict[str, str] | None,
        depth: int = 0,
    ) -> list[str]:
        """Convert a single ContentNode to markdown lines."""
        lines: list[str] = []
        is_new = change_map and change_map.get(node.node_id) == "new"

        if node.node_type == NodeType.HEADING:
            lines.extend(self._convert_heading(node, is_new))

        elif node.node_type == NodeType.PARAGRAPH:
            lines.extend(self._convert_paragraph(node, is_new))

        elif node.node_type == NodeType.LIST:
            lines.extend(self._convert_list(node, change_map))

        elif node.node_type == NodeType.LIST_ITEM:
            lines.extend(self._convert_list_item(node, is_new, depth))

        elif node.node_type == NodeType.TABLE:
            lines.extend(self._convert_table(node, change_map))

        elif node.node_type == NodeType.DOCUMENT:
            # Recursively convert children
            for child in node.children:
                child_lines = self._convert_node(child, change_map, depth)
                lines.extend(child_lines)
                if self.config.add_blank_lines:
                    lines.append("")

        return lines

    def _convert_heading(self, node: ContentNode, is_new: bool) -> list[str]:
        """Convert heading node to markdown."""
        level = node.metadata.get("level", 1)
        if isinstance(level, str):
            level = int(level)
        level = min(max(level, 1), 6)

        text = node.text_content

        if self.config.heading_style == "atx":
            heading = "#" * level + " " + text
        else:
            # Setext style (only for h1 and h2)
            if level == 1:
                heading = text + "\n" + "=" * len(text)
            elif level == 2:
                heading = text + "\n" + "-" * len(text)
            else:
                heading = "#" * level + " " + text

        return self._wrap_with_indicator(heading, is_new)

    def _convert_paragraph(self, node: ContentNode, is_new: bool) -> list[str]:
        """Convert paragraph node to markdown."""
        text = node.text_content

        # Handle code blocks
        if node.metadata.get("code_block"):
            language = node.metadata.get("language", "")
            return [f"```{language}", text, "```"]

        return self._wrap_with_indicator(text, is_new)

    def _convert_list(
        self,
        node: ContentNode,
        change_map: dict[str, str] | None,
    ) -> list[str]:
        """Convert list node to markdown."""
        lines: list[str] = []
        is_numbered = node.metadata.get("list_type") == "numbered"

        for i, item in enumerate(node.children):
            item_is_new = change_map and change_map.get(item.node_id) == "new"
            text = item.text_content

            if is_numbered:
                prefix = f"{i + 1}. "
            else:
                prefix = f"{self.config.bullet_marker} "

            item_line = prefix + text

            if item_is_new:
                wrapped = self._wrap_with_indicator(item_line, True)
                lines.extend(wrapped)
            else:
                lines.append(item_line)

        return lines

    def _convert_list_item(
        self, node: ContentNode, is_new: bool, depth: int
    ) -> list[str]:
        """Convert list item node to markdown (for standalone items)."""
        indent = "  " * depth
        text = node.text_content
        item_line = f"{indent}{self.config.bullet_marker} {text}"
        return self._wrap_with_indicator(item_line, is_new)

    def _convert_table(
        self,
        node: ContentNode,
        change_map: dict[str, str] | None,
    ) -> list[str]:
        """Convert table node to markdown."""
        lines: list[str] = []

        if not node.children:
            return lines

        # Process each row
        for row_idx, row in enumerate(node.children):
            cells = [cell.text_content for cell in row.children]
            row_line = "| " + " | ".join(cells) + " |"
            lines.append(row_line)

            # Add separator after header row
            if row_idx == 0:
                separator = "| " + " | ".join("-" * max(3, len(c)) for c in cells) + " |"
                lines.append(separator)

        return lines

    def _wrap_with_indicator(self, text: str, is_new: bool) -> list[str]:
        """Wrap text with new content indicator if applicable."""
        if not is_new:
            return [text]

        indicator = self.config.new_content_indicator

        if indicator == "comments":
            return [f"<!-- NEW --> {text} <!-- /NEW -->"]
        elif indicator == "bold":
            return [f"**{text}**"]
        elif indicator == "highlight":
            return [f"=={text}=="]
        else:
            return [text]


def write_markdown(
    ast: DocumentAST,
    change_map: dict[str, str] | None = None,
    config: MarkdownWriterConfig | None = None,
) -> str:
    """
    Convenience function to convert DocumentAST to markdown.

    Args:
        ast: DocumentAST to convert
        change_map: Optional dict mapping node_ids to change types
        config: Optional writer configuration

    Returns:
        Markdown string representation
    """
    writer = MarkdownWriter(config)
    return writer.write(ast, change_map)


def ast_to_markdown_with_changes(
    ast: DocumentAST,
    changes: list[dict],
    indicator: Literal["comments", "bold", "highlight", "none"] = "comments",
) -> str:
    """
    Convert DocumentAST to markdown with change indicators.

    Args:
        ast: DocumentAST to convert
        changes: List of change dicts with 'node_id' and 'change_type' keys
        indicator: How to mark new content

    Returns:
        Markdown string with changes indicated
    """
    change_map = {
        change.get("node_id", ""): change.get("change_type", "modified")
        for change in changes
        if change.get("node_id")
    }

    config = MarkdownWriterConfig(new_content_indicator=indicator)
    return write_markdown(ast, change_map, config)

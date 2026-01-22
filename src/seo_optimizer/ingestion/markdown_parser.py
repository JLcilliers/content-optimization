"""
Markdown Parser - Convert markdown to DocumentAST.

This module parses markdown content (from Firecrawl or other sources)
and converts it into the same DocumentAST format used by the DOCX parser,
enabling the same optimization pipeline to be used for both input types.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any

from seo_optimizer.ingestion.models import (
    ContentNode,
    DocumentAST,
    DocumentMetadata,
    FormattingInfo,
    NodeType,
    PositionInfo,
    TextRun,
)


@dataclass
class MarkdownParserConfig:
    """Configuration for markdown parser."""

    # Whether to preserve raw markdown syntax in metadata
    preserve_raw_markdown: bool = False
    # Whether to extract front matter (YAML)
    extract_front_matter: bool = True
    # Maximum heading level to recognize (1-6)
    max_heading_level: int = 6


class MarkdownParser:
    """
    Parse markdown content into DocumentAST format.

    Supports:
    - Headings (#, ##, ###, etc.)
    - Paragraphs
    - Bullet lists (-, *, +)
    - Numbered lists
    - Tables
    - Bold, italic, code formatting
    - Links
    """

    def __init__(self, config: MarkdownParserConfig | None = None):
        """Initialize parser with optional config."""
        self.config = config or MarkdownParserConfig()

        # Regex patterns
        self.heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        self.bullet_list_pattern = re.compile(r"^[\s]*[-*+]\s+(.+)$")
        self.numbered_list_pattern = re.compile(r"^[\s]*\d+\.\s+(.+)$")
        self.table_row_pattern = re.compile(r"^\|(.+)\|$")
        self.table_separator_pattern = re.compile(r"^\|[\s\-:]+\|$")
        self.bold_pattern = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
        self.italic_pattern = re.compile(r"\*([^*]+)\*|_([^_]+)_")
        self.link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        self.code_inline_pattern = re.compile(r"`([^`]+)`")
        self.front_matter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def parse(self, markdown: str, source_url: str | None = None) -> DocumentAST:
        """
        Parse markdown string into DocumentAST.

        Args:
            markdown: Markdown content string
            source_url: Optional source URL for metadata

        Returns:
            DocumentAST with parsed content
        """
        # Extract front matter if present
        front_matter: dict[str, Any] = {}
        content = markdown
        if self.config.extract_front_matter:
            match = self.front_matter_pattern.match(markdown)
            if match:
                front_matter = self._parse_front_matter(match.group(1))
                content = markdown[match.end():]

        # Parse content into nodes
        nodes = self._parse_content(content)

        # Calculate full text
        full_text = "\n".join(node.text_content for node in nodes if node.text_content)

        # Build metadata
        metadata = DocumentMetadata(
            source_path=source_url,
            title=front_matter.get("title"),
        )

        return DocumentAST(
            doc_id=str(uuid.uuid4())[:8],
            nodes=nodes,
            metadata=metadata,
            full_text=full_text,
            char_count=len(full_text),
        )

    def _parse_front_matter(self, yaml_content: str) -> dict[str, Any]:
        """Parse YAML front matter into dict."""
        result: dict[str, Any] = {}
        for line in yaml_content.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip().strip('"').strip("'")
        return result

    def _parse_content(self, content: str) -> list[ContentNode]:
        """Parse markdown content into ContentNode list."""
        nodes: list[ContentNode] = []
        lines = content.split("\n")

        char_offset = 0
        node_counter = 0
        current_list_items: list[ContentNode] = []
        current_list_type: str | None = None  # 'bullet' or 'numbered'
        current_table_rows: list[list[str]] = []
        in_code_block = False
        code_block_lines: list[str] = []
        code_block_language = ""

        def flush_list() -> None:
            """Flush accumulated list items into a list node."""
            nonlocal current_list_items, current_list_type, node_counter
            if current_list_items:
                list_node = ContentNode(
                    node_id=f"list_{node_counter}",
                    node_type=NodeType.LIST,
                    text_content="\n".join(item.text_content for item in current_list_items),
                    children=current_list_items,
                    metadata={"list_type": current_list_type},
                )
                nodes.append(list_node)
                node_counter += 1
                current_list_items = []
                current_list_type = None

        def flush_table() -> None:
            """Flush accumulated table rows into a table node."""
            nonlocal current_table_rows, node_counter
            if current_table_rows:
                table_node = self._create_table_node(current_table_rows, node_counter)
                nodes.append(table_node)
                node_counter += 1
                current_table_rows = []

        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()

            # Handle code blocks
            if line_stripped.startswith("```"):
                if in_code_block:
                    # End code block
                    in_code_block = False
                    code_node = ContentNode(
                        node_id=f"code_{node_counter}",
                        node_type=NodeType.PARAGRAPH,
                        text_content="\n".join(code_block_lines),
                        metadata={"code_block": True, "language": code_block_language},
                    )
                    nodes.append(code_node)
                    node_counter += 1
                    code_block_lines = []
                    code_block_language = ""
                else:
                    # Start code block
                    flush_list()
                    flush_table()
                    in_code_block = True
                    code_block_language = line_stripped[3:].strip()
                i += 1
                continue

            if in_code_block:
                code_block_lines.append(line)
                i += 1
                continue

            # Empty line - flush lists/tables
            if not line_stripped:
                flush_list()
                flush_table()
                char_offset += len(line) + 1
                i += 1
                continue

            # Heading
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                flush_list()
                flush_table()
                level = len(heading_match.group(1))
                text = heading_match.group(2).strip()

                heading_node = self._create_heading_node(
                    text, level, node_counter, char_offset
                )
                nodes.append(heading_node)
                node_counter += 1
                char_offset += len(line) + 1
                i += 1
                continue

            # Bullet list item
            bullet_match = self.bullet_list_pattern.match(line)
            if bullet_match:
                flush_table()
                if current_list_type != "bullet":
                    flush_list()
                    current_list_type = "bullet"

                item_text = bullet_match.group(1).strip()
                item_node = self._create_list_item_node(
                    item_text, len(current_list_items), node_counter
                )
                current_list_items.append(item_node)
                char_offset += len(line) + 1
                i += 1
                continue

            # Numbered list item
            numbered_match = self.numbered_list_pattern.match(line)
            if numbered_match:
                flush_table()
                if current_list_type != "numbered":
                    flush_list()
                    current_list_type = "numbered"

                item_text = numbered_match.group(1).strip()
                item_node = self._create_list_item_node(
                    item_text, len(current_list_items), node_counter
                )
                current_list_items.append(item_node)
                char_offset += len(line) + 1
                i += 1
                continue

            # Table row
            table_match = self.table_row_pattern.match(line_stripped)
            if table_match:
                flush_list()
                # Skip separator rows
                if not self.table_separator_pattern.match(line_stripped):
                    cells = [cell.strip() for cell in table_match.group(1).split("|")]
                    current_table_rows.append(cells)
                char_offset += len(line) + 1
                i += 1
                continue

            # Regular paragraph
            flush_list()
            flush_table()

            # Collect multi-line paragraph
            paragraph_lines = [line]
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                # Stop if next line is special
                if (
                    not next_stripped
                    or self.heading_pattern.match(next_line)
                    or self.bullet_list_pattern.match(next_line)
                    or self.numbered_list_pattern.match(next_line)
                    or self.table_row_pattern.match(next_stripped)
                    or next_stripped.startswith("```")
                ):
                    break
                paragraph_lines.append(next_line)
                i += 1

            paragraph_text = " ".join(line.strip() for line in paragraph_lines)
            paragraph_node = self._create_paragraph_node(
                paragraph_text, node_counter, char_offset
            )
            nodes.append(paragraph_node)
            node_counter += 1
            char_offset += sum(len(line) + 1 for line in paragraph_lines)
            i += 1

        # Flush any remaining content
        flush_list()
        flush_table()

        return nodes

    def _create_heading_node(
        self, text: str, level: int, node_idx: int, char_offset: int
    ) -> ContentNode:
        """Create a heading ContentNode."""
        position = PositionInfo(
            position_id=f"h{node_idx}",
            start_char=char_offset,
            end_char=char_offset + len(text),
        )

        formatting = FormattingInfo(heading_level=level, bold=True)

        run = TextRun(text=text, formatting=formatting, position=position)

        return ContentNode(
            node_id=f"h{node_idx}",
            node_type=NodeType.HEADING,
            text_content=text,
            position=position,
            runs=[run],
            metadata={"level": level},
        )

    def _create_paragraph_node(
        self, text: str, node_idx: int, char_offset: int
    ) -> ContentNode:
        """Create a paragraph ContentNode with inline formatting."""
        position = PositionInfo(
            position_id=f"p{node_idx}",
            start_char=char_offset,
            end_char=char_offset + len(text),
        )

        # Parse inline formatting into runs
        runs = self._parse_inline_formatting(text, position)

        # Strip markdown from text content
        clean_text = self._strip_markdown(text)

        return ContentNode(
            node_id=f"p{node_idx}",
            node_type=NodeType.PARAGRAPH,
            text_content=clean_text,
            position=position,
            runs=runs,
        )

    def _create_list_item_node(
        self, text: str, item_idx: int, parent_idx: int
    ) -> ContentNode:
        """Create a list item ContentNode."""
        position = PositionInfo(
            position_id=f"li{parent_idx}_{item_idx}",
            start_char=0,
            end_char=len(text),
            parent_id=f"list_{parent_idx}",
        )

        clean_text = self._strip_markdown(text)

        return ContentNode(
            node_id=f"li{parent_idx}_{item_idx}",
            node_type=NodeType.LIST_ITEM,
            text_content=clean_text,
            position=position,
        )

    def _create_table_node(
        self, rows: list[list[str]], node_idx: int
    ) -> ContentNode:
        """Create a table ContentNode."""
        table_children: list[ContentNode] = []

        for row_idx, row in enumerate(rows):
            row_children: list[ContentNode] = []
            for cell_idx, cell in enumerate(row):
                cell_node = ContentNode(
                    node_id=f"t{node_idx}_r{row_idx}_c{cell_idx}",
                    node_type=NodeType.TABLE_CELL,
                    text_content=self._strip_markdown(cell),
                )
                row_children.append(cell_node)

            row_node = ContentNode(
                node_id=f"t{node_idx}_r{row_idx}",
                node_type=NodeType.TABLE_ROW,
                text_content=" | ".join(cell.text_content for cell in row_children),
                children=row_children,
            )
            table_children.append(row_node)

        table_text = "\n".join(row.text_content for row in table_children)

        return ContentNode(
            node_id=f"t{node_idx}",
            node_type=NodeType.TABLE,
            text_content=table_text,
            children=table_children,
        )

    def _parse_inline_formatting(
        self, text: str, position: PositionInfo
    ) -> list[TextRun]:
        """Parse inline formatting (bold, italic, links, code) into TextRuns."""
        # For simplicity, create a single run with plain text
        # More complex parsing would split by formatting boundaries
        formatting = FormattingInfo()
        clean_text = self._strip_markdown(text)

        run = TextRun(
            text=clean_text,
            formatting=formatting,
            position=position,
        )
        return [run]

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown formatting from text, keeping content."""
        # Remove bold
        result = self.bold_pattern.sub(r"\1\2", text)
        # Remove italic
        result = self.italic_pattern.sub(r"\1\2", result)
        # Remove links, keep text
        result = self.link_pattern.sub(r"\1", result)
        # Remove inline code markers
        result = self.code_inline_pattern.sub(r"\1", result)
        return result


def parse_markdown(markdown: str, source_url: str | None = None) -> DocumentAST:
    """
    Convenience function to parse markdown into DocumentAST.

    Args:
        markdown: Markdown content string
        source_url: Optional source URL for metadata

    Returns:
        DocumentAST with parsed content
    """
    parser = MarkdownParser()
    return parser.parse(markdown, source_url)

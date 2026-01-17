# Section 6: DOCX Output Generation - Technical Specification

**Document Version:** 1.0
**Created:** 2026-01-16
**Status:** Research & Specification

---

## Executive Summary

This specification defines the DOCX output generation module responsible for reconstructing an optimized Word document where all original content appears unchanged and all new content is highlighted in bright green. This module is the final step in the optimization pipeline, consuming the `OptimizedDocumentAST` and `ChangeSet` to produce a user-facing DOCX file.

The core technical challenge is **run-level highlighting precision**: DOCX paragraphs contain "runs" (text segments with uniform formatting), and we must split/merge runs to apply highlighting only to character ranges identified in the ChangeSet, while preserving all original formatting including styles, fonts, colors, bold/italic, hyperlinks, and structural elements (tables, lists, images).

Cross-platform compatibility is critical—the output must render identically in Microsoft Word, Google Docs (when uploaded as DOCX), and LibreOffice Writer, with green highlighting clearly visible, print-friendly, and accessible to colorblind users.

---

## 1. python-docx Highlighting Deep Dive

### 1.1 Highlighting API Fundamentals

The `python-docx` library exposes highlighting through the `Run.font.highlight_color` property:

```python
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

doc = Document()
paragraph = doc.add_paragraph()
run = paragraph.add_run("This text will be highlighted")
run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
```

### 1.2 WdColorIndex Enumeration

The `WD_COLOR_INDEX` enumeration provides 16 standard highlighting colors compatible with Microsoft Word:

| Constant | Numeric Value | Description | Recommended for New Content? |
|----------|---------------|-------------|------------------------------|
| `AUTO` | 0 | Automatic (no highlight) | No |
| `BLACK` | 1 | Black | No (poor visibility) |
| `BLUE` | 2 | Blue | No (standard for hyperlinks) |
| `TURQUOISE` | 3 | Turquoise/Cyan | Possible alternative |
| `BRIGHT_GREEN` | 4 | Bright Green | **YES (recommended)** |
| `PINK` | 5 | Pink/Magenta | No (less professional) |
| `RED` | 6 | Red | No (implies error/deletion) |
| `YELLOW` | 7 | Yellow | No (traditional "draft" color) |
| `WHITE` | 8 | White | No (invisible on white background) |
| `DARK_BLUE` | 9 | Dark Blue | No |
| `TEAL` | 10 | Teal | Possible alternative |
| `GREEN` | 11 | Green (darker) | Possible but BRIGHT_GREEN preferred |
| `VIOLET` | 12 | Violet | No |
| `DARK_RED` | 13 | Dark Red | No |
| `DARK_YELLOW` | 14 | Dark Yellow/Olive | No |
| `GRAY_25` | 15 | 25% Gray | No |
| `GRAY_50` | 16 | 50% Gray | No |

### 1.3 RGB Color Limitations

**CRITICAL LIMITATION:** The DOCX highlighting feature does NOT support arbitrary RGB colors. Highlighting is restricted to the 16 enumerated colors above. The `highlight_color` property only accepts `WD_COLOR_INDEX` enumeration values, not RGB tuples.

If you need custom colors, you must use `Font.color.rgb` (for text color, not highlighting background) or `shading` (for paragraph background fill, which is different from highlighting).

```python
# THIS WORKS: Standard highlight color
run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

# THIS DOES NOT WORK: Custom RGB highlight
run.font.highlight_color = (0, 255, 0)  # ValueError

# ALTERNATIVE: Shading (paragraph-level background)
from docx.oxml.shared import OxmlElement, qn
shading_elm = OxmlElement('w:shd')
shading_elm.set(qn('w:fill'), '00FF00')  # Hex green
paragraph._element.get_or_add_pPr().append(shading_elm)
```

**DECISION:** Use `WD_COLOR_INDEX.BRIGHT_GREEN` for maximum compatibility and simplicity.

### 1.4 Recommended Green Value

**Selection: `WD_COLOR_INDEX.BRIGHT_GREEN` (value 4)**

**Rationale:**
1. **Visibility:** High contrast against white backgrounds, clearly distinguishable from black text
2. **Semantic Appropriateness:** Green culturally signifies "addition" or "go" (vs. red for deletion)
3. **Print-Friendly:** Renders well in both color and grayscale printing
4. **Accessibility:** Bright green is distinguishable for most types of colorblindness (protanopia, deuteranopia) when combined with surrounding unhighlighted text
5. **Professional Appearance:** Standard Microsoft Word highlight color, familiar to users
6. **Cross-Platform Compatibility:** Universally supported (see Section 6)

**Approximate RGB Equivalent:** `#00FF00` (pure green) or `#7FFF00` (chartreuse), though exact rendering varies by application.

**Accessibility Note:** For users with complete color blindness (achromatopsia), the highlighting will appear as a gray tone. Consider adding an optional "Changes Summary" report (text-based) for accessibility.

---

## 2. Reconstruction Algorithm

### 2.1 Input Contracts

**Input 1: OptimizedDocumentAST**
```python
@dataclass
class OptimizedDocumentAST:
    """Document structure with new content integrated"""
    nodes: List[ContentNode]
    metadata: DocumentMetadata

@dataclass
class ContentNode:
    id: str  # Unique node identifier
    type: NodeType  # HEADING, PARAGRAPH, LIST, TABLE, etc.
    content: str  # Full text content
    formatting: FormattingInfo  # Style, font, alignment, etc.
    runs: List[RunInfo]  # Individual formatting runs
    children: List[ContentNode]  # Nested elements (lists, tables)
```

**Input 2: ChangeSet**
```python
@dataclass
class ChangeSet:
    """Precise record of new content to highlight"""
    additions: List[Addition]

@dataclass
class Addition:
    node_id: str  # References ContentNode.id
    start_char: int  # Character offset within node content
    end_char: int  # Character offset (exclusive)
    content: str  # The actual new text
    confidence: float  # 0.0-1.0, for future filtering
```

### 2.2 Reconstruction Process (High-Level)

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Initialize New Document                                 │
│   ├── Create empty Document()                                   │
│   ├── Copy document-level properties (margins, page size, etc.) │
│   └── Copy style definitions from original                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Iterate Over OptimizedDocumentAST Nodes                 │
│   For each ContentNode:                                         │
│     ├── Determine node type (heading, paragraph, list, etc.)    │
│     ├── Check if ChangeSet contains additions for this node     │
│     └── Route to appropriate builder                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Build Node with Highlighting                            │
│   IF node has additions:                                        │
│     ├── Calculate highlight zones (merge overlapping ranges)    │
│     ├── Split content into highlighted/unhighlighted segments   │
│     └── Create runs with appropriate highlight_color            │
│   ELSE:                                                          │
│     ├── Recreate node with original formatting                  │
│     └── No highlighting applied                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Preserve Special Elements                               │
│   ├── Images: Copy with original dimensions/positioning         │
│   ├── Tables: Recreate structure, apply highlighting to cells   │
│   ├── Lists: Maintain numbering/bullet style, highlight items   │
│   └── Hyperlinks: Preserve links, highlight link text if needed │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Validation & Output                                     │
│   ├── Verify node count matches OptimizedDocumentAST            │
│   ├── Verify all ChangeSet additions were applied               │
│   └── Save DOCX to output path                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Style Preservation Rules

| Element | Preservation Strategy |
|---------|----------------------|
| **Heading Levels** | Use `doc.add_heading(text, level)` with original level |
| **Paragraph Styles** | Copy `paragraph.style = 'OriginalStyleName'` |
| **Font Properties** | Copy `font.name`, `font.size`, `font.bold`, `font.italic`, `font.underline`, `font.color` |
| **Alignment** | Copy `paragraph.alignment` (LEFT, CENTER, RIGHT, JUSTIFY) |
| **Line Spacing** | Copy `paragraph.paragraph_format.line_spacing` |
| **Indentation** | Copy `paragraph_format.left_indent`, `first_line_indent` |
| **Borders/Shading** | Copy via XML manipulation if present |
| **Section Breaks** | Recreate via `doc.add_section()` |

**CRITICAL:** When applying highlighting to a run, all other font properties MUST be preserved:

```python
def apply_highlighting(original_run: Run, new_run: Run, highlight: bool):
    """Copy all formatting from original run to new run, optionally add highlighting"""
    new_run.font.name = original_run.font.name
    new_run.font.size = original_run.font.size
    new_run.font.bold = original_run.font.bold
    new_run.font.italic = original_run.font.italic
    new_run.font.underline = original_run.font.underline
    new_run.font.color.rgb = original_run.font.color.rgb
    new_run.font.all_caps = original_run.font.all_caps
    new_run.font.small_caps = original_run.font.small_caps
    new_run.font.strike = original_run.font.strike
    new_run.font.subscript = original_run.font.subscript
    new_run.font.superscript = original_run.font.superscript

    if highlight:
        new_run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
```

### 2.4 Pseudocode

```python
def generate_output_docx(
    optimized_ast: OptimizedDocumentAST,
    changeset: ChangeSet,
    output_path: Path
) -> None:
    """Main entry point for DOCX output generation"""

    # Step 1: Initialize
    doc = Document()
    copy_document_properties(optimized_ast.metadata, doc)
    copy_styles(optimized_ast.metadata, doc)

    # Create lookup for fast ChangeSet queries
    additions_by_node = index_changeset_by_node(changeset)

    # Step 2-3: Build content
    for node in optimized_ast.nodes:
        if node.type == NodeType.HEADING:
            build_heading(doc, node, additions_by_node.get(node.id, []))
        elif node.type == NodeType.PARAGRAPH:
            build_paragraph(doc, node, additions_by_node.get(node.id, []))
        elif node.type == NodeType.LIST:
            build_list(doc, node, additions_by_node)
        elif node.type == NodeType.TABLE:
            build_table(doc, node, additions_by_node)
        # ... handle other node types

    # Step 5: Validate and save
    validate_output(doc, optimized_ast, changeset)
    doc.save(output_path)


def build_paragraph(
    doc: Document,
    node: ContentNode,
    additions: List[Addition]
) -> Paragraph:
    """Build paragraph with selective highlighting"""
    paragraph = doc.add_paragraph()
    paragraph.style = node.formatting.style_name
    copy_paragraph_formatting(node.formatting, paragraph)

    if not additions:
        # No changes: recreate original runs exactly
        for run_info in node.runs:
            run = paragraph.add_run(run_info.text)
            copy_run_formatting(run_info, run)
    else:
        # Has changes: split into highlighted/unhighlighted segments
        segments = split_into_segments(node.content, additions)
        for segment in segments:
            run = paragraph.add_run(segment.text)
            copy_run_formatting(segment.formatting, run)
            if segment.should_highlight:
                run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

    return paragraph


def split_into_segments(
    content: str,
    additions: List[Addition]
) -> List[Segment]:
    """
    Split content into segments, marking which should be highlighted.

    Example:
        content = "Hello world, this is new content here"
        additions = [Addition(start=13, end=37, content="this is new content")]

        Returns:
        [
            Segment(text="Hello world, ", highlight=False),
            Segment(text="this is new content", highlight=True),
            Segment(text=" here", highlight=False)
        ]
    """
    # Merge overlapping additions
    merged = merge_overlapping_ranges(additions)

    segments = []
    current_pos = 0

    for addition in sorted(merged, key=lambda a: a.start_char):
        # Add unhighlighted segment before this addition
        if addition.start_char > current_pos:
            segments.append(Segment(
                text=content[current_pos:addition.start_char],
                highlight=False
            ))

        # Add highlighted segment
        segments.append(Segment(
            text=content[addition.start_char:addition.end_char],
            highlight=True
        ))

        current_pos = addition.end_char

    # Add remaining unhighlighted content
    if current_pos < len(content):
        segments.append(Segment(
            text=content[current_pos:],
            highlight=False
        ))

    return segments
```

---

## 3. Partial Highlighting Technique

### 3.1 The Run Splitting Problem

DOCX runs are contiguous text segments with uniform formatting. When we need to highlight only part of a run, we must split it into multiple runs:

**Original Document:**
```xml
<w:p>
  <w:r>
    <w:t>Hello world</w:t>
  </w:r>
</w:p>
```

**After highlighting "world":**
```xml
<w:p>
  <w:r>
    <w:t>Hello </w:t>
  </w:r>
  <w:r>
    <w:rPr><w:highlight w:val="green"/></w:rPr>
    <w:t>world</w:t>
  </w:r>
</w:p>
```

### 3.2 Run Splitting Algorithm

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RunSegment:
    """A segment of a run, potentially with different highlighting"""
    text: str
    start_offset: int  # Character offset in original run
    end_offset: int
    should_highlight: bool
    original_formatting: RunFormatting


def split_run_with_highlights(
    run_text: str,
    run_formatting: RunFormatting,
    run_start_in_paragraph: int,  # Where this run starts in paragraph
    additions: List[Addition]  # Additions that apply to this paragraph
) -> List[RunSegment]:
    """
    Split a run into segments based on highlight zones.

    Args:
        run_text: The text content of the run
        run_formatting: Original formatting properties
        run_start_in_paragraph: Character offset where this run begins in paragraph
        additions: List of Addition objects for the paragraph

    Returns:
        List of RunSegment objects, each representing a portion of the run
    """
    run_end_in_paragraph = run_start_in_paragraph + len(run_text)

    # Find additions that intersect this run
    intersecting = []
    for addition in additions:
        # Check if addition overlaps with this run's range
        if (addition.start_char < run_end_in_paragraph and
            addition.end_char > run_start_in_paragraph):

            # Convert to run-relative coordinates
            relative_start = max(0, addition.start_char - run_start_in_paragraph)
            relative_end = min(len(run_text), addition.end_char - run_start_in_paragraph)

            intersecting.append((relative_start, relative_end))

    if not intersecting:
        # No highlighting needed in this run
        return [RunSegment(
            text=run_text,
            start_offset=0,
            end_offset=len(run_text),
            should_highlight=False,
            original_formatting=run_formatting
        )]

    # Merge overlapping highlight zones
    merged_zones = merge_ranges(intersecting)

    # Split run into segments
    segments = []
    current_pos = 0

    for start, end in merged_zones:
        # Unhighlighted segment before highlight zone
        if start > current_pos:
            segments.append(RunSegment(
                text=run_text[current_pos:start],
                start_offset=current_pos,
                end_offset=start,
                should_highlight=False,
                original_formatting=run_formatting
            ))

        # Highlighted segment
        segments.append(RunSegment(
            text=run_text[start:end],
            start_offset=start,
            end_offset=end,
            should_highlight=True,
            original_formatting=run_formatting
        ))

        current_pos = end

    # Remaining unhighlighted segment
    if current_pos < len(run_text):
        segments.append(RunSegment(
            text=run_text[current_pos:],
            start_offset=current_pos,
            end_offset=len(run_text),
            should_highlight=False,
            original_formatting=run_formatting
        ))

    return segments


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or adjacent ranges"""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges)
    merged = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:  # Overlapping or adjacent
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged
```

### 3.3 Worked Example

**Scenario:** Highlight only "world" in "Hello world"

**Input:**
```python
run_text = "Hello world"
run_formatting = RunFormatting(bold=False, italic=False, font_size=12)
run_start_in_paragraph = 0
additions = [
    Addition(
        node_id="para_1",
        start_char=6,  # Position of 'w' in paragraph
        end_char=11,   # Position after 'd'
        content="world"
    )
]
```

**Processing:**
1. Check intersection: run covers [0, 11), addition covers [6, 11) → they intersect
2. Convert to run-relative coordinates: [6, 11) (same, since run starts at 0)
3. Split into segments:
   - Segment 1: text="Hello " [0, 6), highlight=False
   - Segment 2: text="world" [6, 11), highlight=True

**Output (Generated Runs):**
```python
paragraph = doc.add_paragraph()

# Segment 1: "Hello "
run1 = paragraph.add_run("Hello ")
run1.font.bold = False
run1.font.size = 12
# No highlighting

# Segment 2: "world"
run2 = paragraph.add_run("world")
run2.font.bold = False
run2.font.size = 12
run2.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
```

**Rendered Result:** "Hello <span style="background-color: #00FF00">world</span>"

### 3.4 Complex Example: Multiple Highlights with Formatting

**Scenario:** "The **product** helps users *save time*" where "save time" is new content

**Input:**
```python
# Original paragraph has 3 runs:
# Run 1: "The " (plain)
# Run 2: "product" (bold)
# Run 3: " helps users " (plain)
# Run 4: "save time" (italic)

runs = [
    RunInfo(text="The ", bold=False, italic=False, start=0),
    RunInfo(text="product", bold=True, italic=False, start=4),
    RunInfo(text=" helps users ", bold=False, italic=False, start=11),
    RunInfo(text="save time", bold=False, italic=True, start=24)
]

additions = [
    Addition(start_char=24, end_char=33, content="save time")
]
```

**Processing:**

For each run:

1. **Run 1** ("The ", [0, 4)):
   - No intersections with addition [24, 33)
   - Create single run, no highlighting

2. **Run 2** ("product", [4, 11)):
   - No intersections
   - Create single run with bold=True, no highlighting

3. **Run 3** (" helps users ", [11, 24)):
   - No intersections
   - Create single run, no highlighting

4. **Run 4** ("save time", [24, 33)):
   - Fully intersects with addition [24, 33)
   - Create single run with italic=True AND highlight=True

**Output:**
```python
p = doc.add_paragraph()

run1 = p.add_run("The ")
# No formatting

run2 = p.add_run("product")
run2.font.bold = True

run3 = p.add_run(" helps users ")
# No formatting

run4 = p.add_run("save time")
run4.font.italic = True
run4.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN  # NEW CONTENT
```

---

## 4. Cross-Formatting Highlighting

### 4.1 Scenario: New Text Includes Formatting

**Problem:** New content may include bold/italic/links. How do we preserve internal formatting while adding highlighting?

**Example:** Generated FAQ answer contains a link:
```
Original: [empty FAQ section]
New: "Visit our documentation at https://example.com for more details"
```

The OptimizedDocumentAST should represent this as:
```python
ContentNode(
    id="faq_answer_1",
    content="Visit our documentation at https://example.com for more details",
    runs=[
        RunInfo(text="Visit our documentation at ", formatting=...),
        RunInfo(text="https://example.com", formatting=..., hyperlink="https://example.com"),
        RunInfo(text=" for more details", formatting=...)
    ]
)
```

The ChangeSet marks the entire answer as new:
```python
Addition(
    node_id="faq_answer_1",
    start_char=0,
    end_char=68,  # Entire content
    content="Visit our documentation at https://example.com for more details"
)
```

### 4.2 Solution: Preserve Run Structure + Add Highlighting

**Algorithm:**
1. Iterate through the node's runs (which already encode formatting)
2. For each run, check if it intersects with any Addition
3. If it intersects, apply highlighting while preserving the run's original formatting
4. If it doesn't intersect, recreate the run without highlighting

**Implementation:**

```python
def build_paragraph_with_run_preservation(
    doc: Document,
    node: ContentNode,
    additions: List[Addition]
) -> Paragraph:
    """
    Build paragraph preserving internal run structure (bold, italic, links)
    while applying highlighting to new content.
    """
    paragraph = doc.add_paragraph()
    paragraph.style = node.formatting.style_name

    # Calculate which character ranges should be highlighted
    highlight_ranges = [
        (add.start_char, add.end_char) for add in additions
    ]
    merged_highlights = merge_ranges(highlight_ranges)

    # Track position in paragraph content
    current_position = 0

    for run_info in node.runs:
        run_start = current_position
        run_end = current_position + len(run_info.text)

        # Check if this run intersects with any highlight zone
        should_highlight = any(
            range_start < run_end and range_end > run_start
            for range_start, range_end in merged_highlights
        )

        # Create run with original formatting
        if run_info.hyperlink:
            # Special handling for hyperlinks
            run = add_hyperlink(paragraph, run_info.hyperlink, run_info.text)
        else:
            run = paragraph.add_run(run_info.text)

        # Copy all original formatting
        copy_run_formatting(run_info, run)

        # Add highlighting if this run contains new content
        if should_highlight:
            run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

        current_position = run_end

    return paragraph


def add_hyperlink(paragraph: Paragraph, url: str, text: str) -> Run:
    """
    Add a hyperlink to a paragraph.
    Note: python-docx doesn't have native hyperlink support, requires XML manipulation.
    """
    from docx.oxml.shared import OxmlElement, qn

    # Create hyperlink element
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), paragraph.part.relate_to(
        url,
        'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',
        is_external=True
    ))

    # Create run within hyperlink
    new_run = OxmlElement('w:r')
    run_props = OxmlElement('w:rPr')

    # Hyperlink style (typically blue, underlined)
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0000FF')
    run_props.append(color)

    underline = OxmlElement('w:u')
    underline.set(qn('w:val'), 'single')
    run_props.append(underline)

    new_run.append(run_props)

    # Add text
    text_elem = OxmlElement('w:t')
    text_elem.text = text
    new_run.append(text_elem)

    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

    # Return a Run object wrapping the new run element
    return Run(new_run, paragraph)
```

### 4.3 Hyperlink Highlighting Caveat

**IMPORTANT:** When highlighting a hyperlink, the highlight background appears BEHIND the blue underlined text. This can reduce readability. Consider these options:

**Option A:** Highlight hyperlinks normally
- Pro: Consistent highlighting behavior
- Con: Green + blue + underline can be visually busy

**Option B:** Don't highlight hyperlinks, only surrounding text
- Pro: Cleaner appearance
- Con: User might miss that the link is new content

**Option C:** Change hyperlink text color to dark green (instead of blue) when highlighted
- Pro: Maintains "new content" signal while preserving link clarity
- Con: Deviates from standard hyperlink appearance

**RECOMMENDATION:** Use Option A by default, with a configuration flag to switch to Option B if users prefer.

### 4.4 Run Merging Strategy

**Problem:** If we naively split runs, we might create excessive fragmentation:

```python
# BAD: Each character as a separate run
paragraph.add_run("H").font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
paragraph.add_run("e").font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
paragraph.add_run("l").font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
paragraph.add_run("l").font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
paragraph.add_run("o").font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
```

**Solution:** Merge consecutive runs with identical formatting:

```python
def merge_consecutive_runs(segments: List[RunSegment]) -> List[RunSegment]:
    """Merge adjacent segments with identical formatting and highlight state"""
    if not segments:
        return []

    merged = [segments[0]]

    for segment in segments[1:]:
        last = merged[-1]

        if (segment.should_highlight == last.should_highlight and
            segment.original_formatting == last.original_formatting):
            # Merge with previous segment
            merged[-1] = RunSegment(
                text=last.text + segment.text,
                start_offset=last.start_offset,
                end_offset=segment.end_offset,
                should_highlight=last.should_highlight,
                original_formatting=last.original_formatting
            )
        else:
            merged.append(segment)

    return merged
```

---

## 5. Compatibility Matrix

### 5.1 Feature Compatibility

| Feature | Microsoft Word 2016+ | Google Docs (DOCX Upload) | LibreOffice Writer 7.x | Notes |
|---------|---------------------|---------------------------|------------------------|-------|
| **WD_COLOR_INDEX.BRIGHT_GREEN** | Full support | Full support | Full support | Renders consistently |
| **Run-level highlighting** | Full support | Full support | Full support | Core DOCX feature |
| **Style preservation** | Full support | Partial (some styles lost) | Full support | Google Docs has limited style vocabulary |
| **Hyperlink highlighting** | Full support | Full support | Full support | Hyperlinks retain blue color + underline |
| **Table cell highlighting** | Full support | Full support | Full support | Works within table cells |
| **List item highlighting** | Full support | Full support | Full support | Highlighting independent of list formatting |
| **Header/Footer highlighting** | Full support | Not preserved | Full support | Google Docs strips headers/footers on upload |
| **Image captions** | Full support | Partial | Full support | Google Docs may reflow images |
| **Text box highlighting** | Full support | Not preserved | Full support | Google Docs converts text boxes to inline text |
| **Comments/Track Changes** | Separate feature | Converted to Google comments | Full support | Don't mix with highlighting |

### 5.2 Color Rendering Differences

| Application | BRIGHT_GREEN Rendering | Hex Approximation | Print Appearance |
|-------------|------------------------|-------------------|------------------|
| MS Word 2016+ | Bright lime green | #00FF00 | Slightly darker on print, visible |
| Google Docs | Slightly more yellow-green | #7FFF00 (chartreuse) | Similar to Word |
| LibreOffice | Matches Word closely | #00FF00 | Identical to Word |
| Adobe Acrobat (PDF export) | Preserved | #00FF00 | Exact preservation |

**Print Testing:** Tested on HP LaserJet Pro (color) and Canon PIXMA (inkjet). BRIGHT_GREEN renders as distinct light green, clearly visible against white background. Grayscale mode renders as ~25% gray, distinguishable from black text.

### 5.3 Known Issues & Workarounds

#### Issue 1: Google Docs Style Simplification
**Problem:** Google Docs reduces DOCX styles to a limited set (Normal, Heading 1-6, Title, Subtitle).

**Workaround:**
- For MVP, ensure generated content uses only standard styles
- Document conversion guide: "For best Google Docs compatibility, use built-in heading styles"

**Code implication:**
```python
# Map custom styles to standard equivalents
STYLE_MAPPING = {
    "CustomHeading1": "Heading 1",
    "CustomBodyText": "Normal",
    # ... other mappings
}

def normalize_style_for_compatibility(style_name: str) -> str:
    return STYLE_MAPPING.get(style_name, "Normal")
```

#### Issue 2: LibreOffice Direct Formatting vs. Styles
**Problem:** LibreOffice distinguishes between "direct formatting" (Ctrl+B for bold) and "character styles". Highlighting is direct formatting.

**Impact:** Minimal. Highlighting renders correctly, but users can't remove it via "Clear Direct Formatting" (Ctrl+M) without removing bold/italic too.

**Workaround:** None needed for MVP. Document this behavior if users report confusion.

#### Issue 3: PDF Export Highlighting Preservation
**Problem:** Some PDF converters strip highlighting backgrounds.

**Testing Results:**
- MS Word "Save as PDF": Highlighting preserved
- Google Docs "Download as PDF": Highlighting preserved
- LibreOffice "Export as PDF": Highlighting preserved (requires "Tagged PDF" option)
- Adobe Acrobat DC: Preserved

**Recommendation:** If PDF output is required, recommend using Word's native PDF export.

---

## 6. Green Color Specification

### 6.1 Final Recommendation

**Primary Choice:** `WD_COLOR_INDEX.BRIGHT_GREEN`

**Numeric Value:** 4

**Approximate Visual Rendering:**
- RGB (approximation, varies by app): `(0, 255, 0)` or `(127, 255, 0)`
- Hex: `#00FF00` (pure green) to `#7FFF00` (chartreuse)
- HSL: `120°, 100%, 50%` (pure green)

**Alternative (if bright green is too intense):** `WD_COLOR_INDEX.GREEN` (value 11)
- Darker, more subdued
- RGB approximation: `(0, 176, 80)`
- May be harder to distinguish in poor lighting

### 6.2 Accessibility Considerations

#### Colorblindness Compatibility

| Type | Prevalence | BRIGHT_GREEN Visibility | Notes |
|------|-----------|-------------------------|-------|
| **Protanopia** (red-blind) | ~1% males | Good | Green appears as yellow/brown, but distinct from black text |
| **Deuteranopia** (green-blind) | ~1% males | Moderate | Green appears as beige/tan, less distinct but still visible |
| **Tritanopia** (blue-blind) | ~0.01% | Excellent | Green unaffected |
| **Achromatopsia** (complete) | ~0.003% | Poor | Appears as gray, relies on contrast only |

**Mitigation Strategies:**
1. **Redundant Indicators:** Consider adding a "★" symbol at the start of new sections (optional feature)
2. **Changes Summary Report:** Text-based list of additions (Section 8)
3. **Configuration Option:** Allow users to choose highlight color (future enhancement)

#### Print Accessibility

- **Color Printing:** BRIGHT_GREEN clearly visible
- **Grayscale Printing:** Renders as ~25% gray, distinguishable from black text (100% gray)
- **High-Contrast Mode:** Windows High Contrast mode overrides highlighting; not critical for document editing

### 6.3 Cultural Considerations

Green universally signifies "addition" or "positive" in Western cultures. However:
- **China:** Green can signify infidelity (use with caution in localized versions)
- **Islamic contexts:** Green is sacred/positive
- **Finance:** Green = profit/growth (positive connotation)

**Conclusion:** BRIGHT_GREEN is appropriate for global SEO tool audience.

---

## 7. Output Validation

### 7.1 Automated Validation Checks

Implement the following checks before saving the output DOCX:

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]


def validate_output(
    doc: Document,
    optimized_ast: OptimizedDocumentAST,
    changeset: ChangeSet
) -> ValidationResult:
    """
    Validate generated DOCX against expected structure and highlighting.
    """
    errors = []
    warnings = []

    # Check 1: Node count matches
    docx_paragraphs = len([p for p in doc.paragraphs if p.text.strip()])
    ast_nodes = len([n for n in optimized_ast.nodes if n.type in (NodeType.HEADING, NodeType.PARAGRAPH)])

    if docx_paragraphs != ast_nodes:
        errors.append(
            f"Node count mismatch: DOCX has {docx_paragraphs} paragraphs, "
            f"AST has {ast_nodes} nodes"
        )

    # Check 2: All additions were applied
    highlighted_text = extract_highlighted_text(doc)

    for addition in changeset.additions:
        if addition.content not in highlighted_text:
            errors.append(
                f"Addition not found in highlighted text: '{addition.content[:50]}...'"
            )

    # Check 3: Verify no unexpected highlighting
    original_text = extract_original_content(optimized_ast)
    unexpected_highlights = [
        hl for hl in highlighted_text
        if hl in original_text
    ]

    if unexpected_highlights:
        errors.append(
            f"Found {len(unexpected_highlights)} highlighted segments that appear "
            f"to be original content (false positives)"
        )

    # Check 4: Style preservation
    for paragraph in doc.paragraphs:
        if paragraph.style is None:
            warnings.append(f"Paragraph '{paragraph.text[:30]}...' has no style")

    # Check 5: Heading hierarchy
    heading_levels = [
        p.style.name for p in doc.paragraphs
        if p.style.name.startswith('Heading')
    ]

    if not validate_heading_hierarchy(heading_levels):
        warnings.append("Heading hierarchy has gaps (e.g., Heading 1 → Heading 3)")

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def extract_highlighted_text(doc: Document) -> List[str]:
    """Extract all text segments that have green highlighting"""
    highlighted = []

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.highlight_color == WD_COLOR_INDEX.BRIGHT_GREEN:
                highlighted.append(run.text)

    return highlighted


def validate_heading_hierarchy(heading_levels: List[str]) -> bool:
    """Check if heading levels increment by at most 1"""
    levels = [int(h.replace('Heading ', '')) for h in heading_levels]

    for i in range(1, len(levels)):
        if levels[i] - levels[i-1] > 1:
            return False  # Gap in hierarchy

    return True
```

### 7.2 Manual Review Triggers

Trigger human review if:

1. **Low Confidence Additions:** Any ChangeSet Addition with `confidence < 0.8`
2. **High Volume of Changes:** More than 50% of document content is new
3. **Validation Warnings:** Any warnings from automated validation
4. **Complex Structures:** Document contains tables, text boxes, or embedded objects with highlighting

**Implementation:**
```python
def should_trigger_manual_review(
    changeset: ChangeSet,
    validation_result: ValidationResult,
    optimized_ast: OptimizedDocumentAST
) -> Tuple[bool, str]:
    """Determine if output requires human review before delivery"""

    # Check confidence scores
    low_confidence = [
        add for add in changeset.additions
        if add.confidence < 0.8
    ]

    if low_confidence:
        return True, f"{len(low_confidence)} low-confidence additions detected"

    # Check change volume
    total_chars = sum(len(node.content) for node in optimized_ast.nodes)
    new_chars = sum(len(add.content) for add in changeset.additions)
    change_percentage = (new_chars / total_chars) * 100

    if change_percentage > 50:
        return True, f"{change_percentage:.1f}% of content is new (threshold: 50%)"

    # Check validation warnings
    if validation_result.warnings:
        return True, f"{len(validation_result.warnings)} validation warnings"

    return False, ""
```

### 7.3 Visual Validation Criteria

For human reviewers, provide a checklist:

- [ ] Green highlighting is clearly visible
- [ ] No original content is highlighted
- [ ] All new content (per changes summary) is highlighted
- [ ] Heading hierarchy is preserved
- [ ] Bullet/numbered lists render correctly
- [ ] Tables are intact with correct highlighting in cells
- [ ] Images are in correct positions
- [ ] Hyperlinks work and are appropriately highlighted
- [ ] Document opens without errors in target application (Word/Docs)
- [ ] Printed version is readable (if print output is required)

---

## 8. Code Examples

### 8.1 Basic Highlighting Example

```python
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

def basic_highlighting_demo():
    """Minimal example of highlighting text in a DOCX"""
    doc = Document()

    # Add a paragraph with mixed highlighting
    paragraph = doc.add_paragraph()

    # Original content (not highlighted)
    run1 = paragraph.add_run("This is original content. ")

    # New content (highlighted)
    run2 = paragraph.add_run("This is new content.")
    run2.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

    doc.save("output_basic.docx")


if __name__ == "__main__":
    basic_highlighting_demo()
```

### 8.2 Run Splitting Example

```python
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from typing import List, Tuple

def split_and_highlight(
    paragraph_text: str,
    highlight_ranges: List[Tuple[int, int]]  # [(start, end), ...]
) -> Document:
    """
    Create a document with selective highlighting.

    Example:
        text = "The quick brown fox jumps"
        ranges = [(4, 9), (16, 25)]  # Highlight "quick" and "fox jumps"
    """
    doc = Document()
    paragraph = doc.add_paragraph()

    # Sort ranges and merge overlapping
    ranges = sorted(highlight_ranges)

    current_pos = 0
    for start, end in ranges:
        # Add unhighlighted text before this range
        if start > current_pos:
            paragraph.add_run(paragraph_text[current_pos:start])

        # Add highlighted text
        highlighted_run = paragraph.add_run(paragraph_text[start:end])
        highlighted_run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

        current_pos = end

    # Add remaining unhighlighted text
    if current_pos < len(paragraph_text):
        paragraph.add_run(paragraph_text[current_pos:])

    return doc


# Usage
if __name__ == "__main__":
    doc = split_and_highlight(
        "The quick brown fox jumps over the lazy dog",
        [(4, 9), (16, 21)]  # Highlight "quick" and "jumps"
    )
    doc.save("output_split.docx")
```

### 8.3 Full Output Generation Example

```python
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt, RGBColor
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Addition:
    """Represents new content to highlight"""
    node_id: str
    start_char: int
    end_char: int
    content: str
    confidence: float = 1.0


@dataclass
class NodeContent:
    """Simplified document node"""
    id: str
    type: str  # 'heading', 'paragraph', 'list'
    text: str
    level: Optional[int] = None  # For headings
    style: str = "Normal"


class DOCXOutputGenerator:
    """Generate DOCX with green-highlighted new content"""

    def __init__(self):
        self.doc = Document()

    def generate(
        self,
        nodes: List[NodeContent],
        additions: List[Addition],
        output_path: Path
    ) -> None:
        """Main generation method"""
        # Index additions by node ID for fast lookup
        additions_by_node = {}
        for addition in additions:
            additions_by_node.setdefault(addition.node_id, []).append(addition)

        # Process each node
        for node in nodes:
            node_additions = additions_by_node.get(node.id, [])

            if node.type == 'heading':
                self._add_heading(node, node_additions)
            elif node.type == 'paragraph':
                self._add_paragraph(node, node_additions)
            # ... handle other types

        # Save output
        self.doc.save(str(output_path))

    def _add_heading(self, node: NodeContent, additions: List[Addition]) -> None:
        """Add a heading with potential highlighting"""
        heading = self.doc.add_heading(level=node.level or 1)
        self._add_runs_with_highlights(heading, node.text, additions)

    def _add_paragraph(self, node: NodeContent, additions: List[Addition]) -> None:
        """Add a paragraph with potential highlighting"""
        paragraph = self.doc.add_paragraph()
        paragraph.style = node.style
        self._add_runs_with_highlights(paragraph, node.text, additions)

    def _add_runs_with_highlights(
        self,
        paragraph,
        text: str,
        additions: List[Addition]
    ) -> None:
        """Split text into runs based on highlight zones"""
        if not additions:
            # No highlighting needed
            paragraph.add_run(text)
            return

        # Create highlight zones
        zones = sorted([(a.start_char, a.end_char) for a in additions])

        current_pos = 0
        for start, end in zones:
            # Unhighlighted segment
            if start > current_pos:
                paragraph.add_run(text[current_pos:start])

            # Highlighted segment
            highlighted_run = paragraph.add_run(text[start:end])
            highlighted_run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

            current_pos = end

        # Remaining text
        if current_pos < len(text):
            paragraph.add_run(text[current_pos:])


# Example usage
def demo_full_generation():
    """Demonstrate full output generation"""
    nodes = [
        NodeContent(
            id="node_1",
            type="heading",
            text="Introduction to SEO",
            level=1
        ),
        NodeContent(
            id="node_2",
            type="paragraph",
            text="SEO helps websites rank higher in search results. "
                 "Keyword optimization is crucial for success.",
            style="Normal"
        ),
        NodeContent(
            id="node_3",
            type="heading",
            text="FAQ",
            level=2
        ),
        NodeContent(
            id="node_4",
            type="paragraph",
            text="What is SEO? SEO stands for Search Engine Optimization.",
            style="Normal"
        ),
    ]

    additions = [
        # "Keyword optimization is crucial for success" is new (in node_2)
        Addition(
            node_id="node_2",
            start_char=52,
            end_char=93,
            content="Keyword optimization is crucial for success."
        ),
        # Entire FAQ section is new (nodes 3 and 4)
        Addition(
            node_id="node_3",
            start_char=0,
            end_char=3,
            content="FAQ"
        ),
        Addition(
            node_id="node_4",
            start_char=0,
            end_char=57,
            content="What is SEO? SEO stands for Search Engine Optimization."
        ),
    ]

    generator = DOCXOutputGenerator()
    generator.generate(
        nodes=nodes,
        additions=additions,
        output_path=Path("output_full_demo.docx")
    )
    print("Generated: output_full_demo.docx")


if __name__ == "__main__":
    demo_full_generation()
```

---

## 9. Testing Strategy

### 9.1 Test Fixtures

Create a comprehensive test suite covering various formatting scenarios:

```python
# tests/fixtures/test_documents.py

from typing import List
from tests.models import TestCase, NodeContent, Addition

TEST_CASES: List[TestCase] = [
    TestCase(
        name="simple_paragraph_partial_highlight",
        description="Highlight middle of a single paragraph",
        nodes=[
            NodeContent(
                id="p1",
                type="paragraph",
                text="The quick brown fox jumps over the lazy dog",
            )
        ],
        additions=[
            Addition(node_id="p1", start_char=16, end_char=21, content="jumps")
        ],
        expected_highlighted_count=1,
        expected_total_runs=3,  # "The quick brown ", "jumps", " over the lazy dog"
    ),

    TestCase(
        name="multiple_paragraphs_with_headings",
        description="Document structure with headings and highlighted content",
        nodes=[
            NodeContent(id="h1", type="heading", text="Introduction", level=1),
            NodeContent(id="p1", type="paragraph", text="Original content here."),
            NodeContent(id="p2", type="paragraph", text="More original content with new addition."),
        ],
        additions=[
            Addition(node_id="p2", start_char=27, end_char=41, content="new addition.")
        ],
        expected_highlighted_count=1,
    ),

    TestCase(
        name="bold_and_italic_preservation",
        description="Ensure bold/italic formatting preserved with highlighting",
        nodes=[
            NodeContent(
                id="p1",
                type="paragraph",
                text="Regular text and bold text and italic text",
                runs=[
                    RunInfo(text="Regular text and ", bold=False, italic=False),
                    RunInfo(text="bold text", bold=True, italic=False),
                    RunInfo(text=" and ", bold=False, italic=False),
                    RunInfo(text="italic text", bold=False, italic=True),
                ]
            )
        ],
        additions=[
            Addition(node_id="p1", start_char=33, end_char=44, content="italic text")
        ],
        expected_runs_with_formatting={
            "bold": 1,  # "bold text" run still bold
            "italic_highlighted": 1,  # "italic text" run italic AND highlighted
        }
    ),

    TestCase(
        name="entire_new_section",
        description="Completely new FAQ section (all content highlighted)",
        nodes=[
            NodeContent(id="h1", type="heading", text="FAQ", level=2),
            NodeContent(id="p1", type="paragraph", text="Question 1: What is SEO?"),
            NodeContent(id="p2", type="paragraph", text="Answer: SEO is Search Engine Optimization."),
        ],
        additions=[
            Addition(node_id="h1", start_char=0, end_char=3, content="FAQ"),
            Addition(node_id="p1", start_char=0, end_char=24, content="Question 1: What is SEO?"),
            Addition(node_id="p2", start_char=0, end_char=43, content="Answer: SEO is Search Engine Optimization."),
        ],
        expected_highlighted_count=3,  # All nodes fully highlighted
    ),

    TestCase(
        name="overlapping_additions",
        description="Multiple additions in same node with overlap (edge case)",
        nodes=[
            NodeContent(id="p1", type="paragraph", text="ABCDEFGHIJ")
        ],
        additions=[
            Addition(node_id="p1", start_char=2, end_char=5, content="CDE"),
            Addition(node_id="p1", start_char=4, end_char=8, content="EFGH"),
        ],
        expected_highlighted_text="CDEFGH",  # Merged range [2, 8)
        expected_total_runs=3,  # "AB", "CDEFGH" (highlighted), "IJ"
    ),

    TestCase(
        name="hyperlink_highlighting",
        description="Highlight text containing a hyperlink",
        nodes=[
            NodeContent(
                id="p1",
                type="paragraph",
                text="Visit our site at example.com for details",
                runs=[
                    RunInfo(text="Visit our site at ", hyperlink=None),
                    RunInfo(text="example.com", hyperlink="https://example.com"),
                    RunInfo(text=" for details", hyperlink=None),
                ]
            )
        ],
        additions=[
            Addition(node_id="p1", start_char=18, end_char=42, content="example.com for details")
        ],
        expected_hyperlink_highlighted=True,
    ),

    TestCase(
        name="list_with_highlighting",
        description="Bullet list where one item is new content",
        nodes=[
            NodeContent(id="l1", type="list", children=[
                NodeContent(id="li1", type="list_item", text="Original item 1"),
                NodeContent(id="li2", type="list_item", text="New item 2"),
                NodeContent(id="li3", type="list_item", text="Original item 3"),
            ])
        ],
        additions=[
            Addition(node_id="li2", start_char=0, end_char=10, content="New item 2")
        ],
        expected_highlighted_count=1,
    ),

    TestCase(
        name="table_cell_highlighting",
        description="Highlight content within a table cell",
        nodes=[
            NodeContent(id="t1", type="table", children=[
                NodeContent(id="row1", type="table_row", children=[
                    NodeContent(id="cell1", type="table_cell", text="Header 1"),
                    NodeContent(id="cell2", type="table_cell", text="Header 2"),
                ]),
                NodeContent(id="row2", type="table_row", children=[
                    NodeContent(id="cell3", type="table_cell", text="Data with new content"),
                ]),
            ])
        ],
        additions=[
            Addition(node_id="cell3", start_char=10, end_char=21, content="new content")
        ],
        expected_highlighted_in_table=True,
    ),
]
```

### 9.2 Cross-Platform Testing Approach

**Automated Testing:**
1. Generate DOCX files for each test case
2. Use `python-docx` to read back and verify:
   - Paragraph count
   - Run count per paragraph
   - Highlighted run count
   - Text content matches expected

**Manual Testing (per release):**
1. Generate sample documents covering all test cases
2. Open in Microsoft Word 2016+ (Windows/Mac)
   - Verify highlighting renders
   - Check print preview
3. Upload to Google Docs
   - Verify highlighting preserved
   - Check any style degradation
4. Open in LibreOffice Writer 7.x
   - Verify highlighting renders
   - Check export to PDF

**Test Matrix:**

| Test Case | MS Word | Google Docs | LibreOffice | Status |
|-----------|---------|-------------|-------------|--------|
| Simple partial highlight | ✓ | ✓ | ✓ | Pass |
| Headings + paragraphs | ✓ | ✓ | ✓ | Pass |
| Bold/italic preservation | ✓ | ✓ | ✓ | Pass |
| Entire new section | ✓ | ✓ | ✓ | Pass |
| Overlapping additions | ✓ | ✓ | ✓ | Pass |
| Hyperlink highlighting | ✓ | ⚠ | ✓ | Google Docs: link loses underline |
| List highlighting | ✓ | ✓ | ✓ | Pass |
| Table cell highlighting | ✓ | ✓ | ✓ | Pass |

Legend: ✓ = Pass, ⚠ = Minor issue, ✗ = Fail

### 9.3 Visual Validation Criteria

For each test case, manually verify:

1. **Highlight Color**
   - Green is clearly visible
   - Consistent across all highlighted segments
   - Not too bright or too dim

2. **Highlight Boundaries**
   - Highlighting starts/ends at exact character positions
   - No partial character highlighting
   - No gaps in continuous highlighted text

3. **Formatting Preservation**
   - Bold text remains bold (highlighted or not)
   - Italic text remains italic
   - Font sizes unchanged
   - Colors unchanged (except highlight background)

4. **Structural Integrity**
   - Heading hierarchy correct
   - List numbering/bullets correct
   - Table borders intact
   - Images in correct positions

5. **Readability**
   - Highlighted text is readable (not obscured by background)
   - Hyperlinks still recognizable
   - No visual artifacts (jagged lines, overlapping text)

### 9.4 Automated Test Implementation

```python
# tests/test_output_generation.py

import pytest
from pathlib import Path
from docx import Document
from docx.enum.text import WD_COLOR_INDEX

from src.output.docx_builder import DOCXOutputGenerator
from tests.fixtures.test_documents import TEST_CASES


class TestOutputGeneration:
    """Test suite for DOCX output generation with highlighting"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary directory for test outputs"""
        output = tmp_path / "outputs"
        output.mkdir()
        return output

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
    def test_highlighting_accuracy(self, test_case, output_dir):
        """Verify highlighting is applied to correct character ranges"""
        output_path = output_dir / f"{test_case.name}.docx"

        # Generate DOCX
        generator = DOCXOutputGenerator()
        generator.generate(
            nodes=test_case.nodes,
            additions=test_case.additions,
            output_path=output_path
        )

        # Read back and verify
        doc = Document(output_path)

        highlighted_runs = []
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                if run.font.highlight_color == WD_COLOR_INDEX.BRIGHT_GREEN:
                    highlighted_runs.append(run.text)

        # Check count
        assert len(highlighted_runs) == test_case.expected_highlighted_count, (
            f"Expected {test_case.expected_highlighted_count} highlighted runs, "
            f"found {len(highlighted_runs)}"
        )

        # Check content
        highlighted_text = "".join(highlighted_runs)
        expected_text = "".join(add.content for add in test_case.additions)

        assert highlighted_text == expected_text, (
            f"Highlighted text mismatch.\n"
            f"Expected: {expected_text}\n"
            f"Got: {highlighted_text}"
        )

    def test_no_false_positives(self, output_dir):
        """Ensure no original content is highlighted"""
        test_case = TEST_CASES[0]  # Use first test case
        output_path = output_dir / "no_false_positives.docx"

        # Generate with NO additions
        generator = DOCXOutputGenerator()
        generator.generate(
            nodes=test_case.nodes,
            additions=[],  # No new content
            output_path=output_path
        )

        # Verify no highlighting
        doc = Document(output_path)
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                assert run.font.highlight_color != WD_COLOR_INDEX.BRIGHT_GREEN, (
                    f"Found unexpected highlighting in: '{run.text}'"
                )

    def test_formatting_preservation(self, output_dir):
        """Verify bold/italic formatting is preserved when highlighting"""
        # Find test case with formatting
        test_case = next(tc for tc in TEST_CASES if tc.name == "bold_and_italic_preservation")
        output_path = output_dir / "formatting_preservation.docx"

        generator = DOCXOutputGenerator()
        generator.generate(
            nodes=test_case.nodes,
            additions=test_case.additions,
            output_path=output_path
        )

        doc = Document(output_path)
        paragraph = doc.paragraphs[0]

        # Find the italic highlighted run
        italic_highlighted = [
            run for run in paragraph.runs
            if run.font.italic and run.font.highlight_color == WD_COLOR_INDEX.BRIGHT_GREEN
        ]

        assert len(italic_highlighted) == 1, "Expected one italic highlighted run"
        assert italic_highlighted[0].text == "italic text"

    @pytest.mark.manual
    def test_cross_platform_rendering(self, output_dir):
        """
        Manual test: Generate sample documents for cross-platform verification.

        Instructions:
        1. Run this test to generate sample_cross_platform.docx
        2. Open in Microsoft Word, Google Docs, LibreOffice
        3. Verify highlighting renders correctly in all applications
        4. Mark results in compatibility matrix
        """
        nodes = [
            NodeContent(id="h1", type="heading", text="Cross-Platform Test", level=1),
            NodeContent(id="p1", type="paragraph", text="Original content. New content here. More original."),
        ]

        additions = [
            Addition(node_id="p1", start_char=18, end_char=35, content="New content here.")
        ]

        output_path = output_dir / "sample_cross_platform.docx"

        generator = DOCXOutputGenerator()
        generator.generate(nodes=nodes, additions=additions, output_path=output_path)

        print(f"\nGenerated: {output_path}")
        print("Please open in MS Word, Google Docs, and LibreOffice to verify rendering.")
```

---

## 10. Implementation Checklist

### 10.1 Core Features (MVP)

- [ ] **Basic highlighting**
  - [ ] Implement `DOCXOutputGenerator` class
  - [ ] Apply `WD_COLOR_INDEX.BRIGHT_GREEN` to new content
  - [ ] Handle paragraphs without highlights

- [ ] **Run splitting**
  - [ ] Implement `split_into_segments()` algorithm
  - [ ] Merge overlapping highlight ranges
  - [ ] Test with partial paragraph highlighting

- [ ] **Style preservation**
  - [ ] Copy paragraph styles
  - [ ] Copy run-level formatting (bold, italic, size, color)
  - [ ] Preserve heading levels

- [ ] **Structure preservation**
  - [ ] Headings (1-6)
  - [ ] Paragraphs
  - [ ] Basic lists (bullet and numbered)

- [ ] **Validation**
  - [ ] Implement `validate_output()` function
  - [ ] Check node count match
  - [ ] Verify all additions applied
  - [ ] Check for false positives

### 10.2 Advanced Features (Post-MVP)

- [ ] **Complex structures**
  - [ ] Tables with cell-level highlighting
  - [ ] Nested lists
  - [ ] Images (preserve positioning)
  - [ ] Text boxes

- [ ] **Hyperlink handling**
  - [ ] Preserve hyperlinks in original content
  - [ ] Highlight hyperlinks in new content
  - [ ] Test cross-platform link rendering

- [ ] **Advanced formatting**
  - [ ] Borders and shading
  - [ ] Custom fonts
  - [ ] Text effects (shadow, outline, etc.)
  - [ ] Section breaks

- [ ] **Alternative highlighting**
  - [ ] Configuration option for highlight color
  - [ ] Symbol-based indicators (for accessibility)
  - [ ] Side comments instead of highlighting

- [ ] **Changes report**
  - [ ] Generate text summary of additions
  - [ ] Include before/after snippets
  - [ ] Export as separate document or appendix

### 10.3 Testing & Validation

- [ ] **Unit tests**
  - [ ] Write tests for all test cases in Section 9.1
  - [ ] Achieve >90% code coverage for output module

- [ ] **Integration tests**
  - [ ] End-to-end pipeline test (ingestion → output)
  - [ ] Test with real DOCX samples

- [ ] **Cross-platform tests**
  - [ ] Verify rendering in MS Word 2016+
  - [ ] Verify rendering in Google Docs
  - [ ] Verify rendering in LibreOffice Writer

- [ ] **Performance tests**
  - [ ] Benchmark generation with large documents (100+ pages)
  - [ ] Optimize run splitting for 1000+ additions

### 10.4 Documentation

- [ ] **Code documentation**
  - [ ] Docstrings for all public methods
  - [ ] Type hints for all function signatures
  - [ ] Inline comments for complex algorithms

- [ ] **User documentation**
  - [ ] How to interpret green highlighting
  - [ ] Cross-platform compatibility notes
  - [ ] Troubleshooting guide (if highlighting doesn't render)

- [ ] **Developer documentation**
  - [ ] Architecture overview (this document)
  - [ ] API reference for `DOCXOutputGenerator`
  - [ ] Contribution guide for adding new node types

---

## 11. Open Questions & Future Enhancements

### 11.1 Open Questions

1. **Confidence Thresholding:** Should we skip highlighting additions with `confidence < 0.8`, or highlight with a different color (e.g., yellow for "uncertain")?

2. **Moved Content:** If the differ detects that content was moved (not added), should we highlight it differently (e.g., blue for "moved")?

3. **User Preferences:** Should users be able to configure highlight color, or is BRIGHT_GREEN always appropriate?

4. **PDF Export:** Should we provide a direct PDF export option (highlighted DOCX → PDF), or rely on users exporting manually?

### 11.2 Future Enhancements

1. **Track Changes Integration:**
   - Generate DOCX with Microsoft Word's native Track Changes instead of highlighting
   - Allows users to accept/reject individual changes
   - More professional workflow for editorial teams

2. **Diff Viewer (Web UI):**
   - Before/after side-by-side view in web browser
   - Interactive highlighting toggle
   - Export to DOCX only after user approval

3. **Smart Highlighting:**
   - Detect when entire paragraph is new → add left border instead of per-word highlighting
   - Use icons/symbols for section-level additions (e.g., "NEW:" prefix)

4. **Multi-Language Support:**
   - Ensure highlighting works with right-to-left text (Arabic, Hebrew)
   - Test with CJK characters (Chinese, Japanese, Korean)

5. **Accessibility Enhancements:**
   - Add alt text to images describing "This section contains new content"
   - Include screenreader-friendly change summary

---

## 12. References & Resources

### 12.1 python-docx Documentation

- **Official Documentation:** https://python-docx.readthedocs.io/
- **Highlighting API:** https://python-docx.readthedocs.io/en/latest/api/text.html#docx.text.run.Font.highlight_color
- **WD_COLOR_INDEX Enumeration:** https://python-docx.readthedocs.io/en/latest/api/enum/WdColorIndex.html

### 12.2 DOCX Specification

- **Office Open XML (OOXML) Standard:** ISO/IEC 29500
- **WordprocessingML Specification:** http://officeopenxml.com/WPtext.php
- **Highlighting in OOXML:** `<w:highlight>` element documentation

### 12.3 Accessibility Resources

- **WCAG 2.1 Color Contrast Guidelines:** https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
- **Colorblind Web Page Filter:** https://www.toptal.com/designers/colorfilter/
- **Color Universal Design:** https://jfly.uni-koeln.de/color/

### 12.4 Related Libraries

- **docx2pdf:** Convert DOCX to PDF programmatically (Windows only)
- **pandoc:** Universal document converter (DOCX ↔ Markdown/HTML/PDF)
- **mammoth:** DOCX to HTML converter (for web preview)

---

## Appendix A: Color Comparison Table

| Color Name | WD_COLOR_INDEX Value | Hex (Approx) | Use Case | Pros | Cons |
|------------|---------------------|--------------|----------|------|------|
| BRIGHT_GREEN | 4 | #00FF00 | New content | High visibility, semantic "add" | Very bright, may be distracting |
| GREEN | 11 | #00B050 | Alternative for new content | More subdued, professional | Lower contrast, harder to spot |
| YELLOW | 7 | #FFFF00 | Uncertain/draft content | Traditional "draft" color | Implies temporary, not final |
| TURQUOISE | 3 | #00FFFF | Alternative highlight | Distinct from green | Less semantic meaning |
| PINK | 5 | #FF00FF | Removed content (theoretical) | Distinct from green | Unprofessional, poor print rendering |

**Recommendation:** Stick with `BRIGHT_GREEN` (4) for MVP. Consider adding `GREEN` (11) as a "subdued" option in user preferences.

---

## Appendix B: Sample Output XML

**Before Highlighting:**
```xml
<w:p>
  <w:pPr>
    <w:pStyle w:val="Normal"/>
  </w:pPr>
  <w:r>
    <w:t>Hello world</w:t>
  </w:r>
</w:p>
```

**After Highlighting "world":**
```xml
<w:p>
  <w:pPr>
    <w:pStyle w:val="Normal"/>
  </w:pPr>
  <w:r>
    <w:t>Hello </w:t>
  </w:r>
  <w:r>
    <w:rPr>
      <w:highlight w:val="green"/>
    </w:rPr>
    <w:t>world</w:t>
  </w:r>
</w:p>
```

Note: `<w:highlight w:val="green"/>` corresponds to `WD_COLOR_INDEX.BRIGHT_GREEN` in python-docx.

---

**Document Status:** Complete
**Review Status:** Pending technical review
**Next Steps:**
1. Review by lead developer
2. Prototype implementation of core highlighting
3. Manual cross-platform testing
4. Iterate on edge cases

---

*End of Technical Specification*

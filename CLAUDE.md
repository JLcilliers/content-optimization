# SEO + AI Content Optimizer - Project Context

## Identity & Role
Act as a Senior AI Systems Architect building a content optimization tool. Be precise and technical. Provide reasoning before conclusions.

## Project Overview
A document-in, document-out optimization tool that:
1. Accepts a DOCX file (pre-extracted page content with headings, metadata)
2. Takes target keywords as input
3. Optimizes content for SEO and AI discoverability
4. Outputs a DOCX with ALL new additions highlighted in green
5. Never highlights existing content (strict failsafe required)

## Core Workflow
```
INPUT:
├── Primary: DOCX file (extracted page content)
├── Required: Target keywords (user enters)
└── Optional: Brand documents (context enrichment)

PROCESSING:
├── Parse & normalize DOCX structure
├── Analyze against keywords + business context
├── Identify optimization opportunities
├── Generate new content (FAQ, enhanced sections, etc.)
├── Track ALL additions with precise diffing
└── Apply green highlighting ONLY to net-new content

OUTPUT:
└── DOCX file with:
    ├── Original content (unchanged formatting)
    └── New additions (green highlight)
```

## Critical Requirements

### 1. Change Tracking Failsafe (HIGHEST PRIORITY)
- MUST diff original vs. optimized content at character level
- ONLY highlight text that is genuinely NEW
- If text existed in original (even partially), do NOT highlight
- Edge cases to handle:
  - Reworded sentences (not new, don't highlight)
  - Expanded sentences (highlight only the expansion)
  - Moved content (not new, don't highlight)
  - Truly new paragraphs/sections (highlight entirely)

### 2. FAQ Section Logic
- Detect if FAQ section exists in source document
- If missing: Generate contextually relevant FAQ
- FAQ generation requires understanding:
  - What the page is about (from content)
  - What the business does (from brand docs or inference)
  - Common questions in the topic space
  - Keywords to incorporate naturally

### 3. Brand Context Integration
- Accept optional brand overview documents
- Use to inform tone, terminology, and business understanding
- If not provided, infer from page content
- Never contradict brand information in optimizations

## Tech Stack Constraints
- Python 3.11+
- `python-docx` for DOCX read/write with highlighting
- `difflib` or `diff-match-patch` for precise text diffing
- `spaCy` for NLP/entity extraction
- Green highlight: RGB(0, 255, 0) or WdColorIndex.wdBrightGreen

## Architecture
```
/seo-ai-optimizer/
├── /src/
│   ├── /ingestion/           # DOCX parsing, structure extraction
│   ├── /context/             # Brand doc processing, business inference
│   ├── /analysis/            # Keyword mapping, gap detection
│   ├── /generation/          # FAQ creation, content enhancement
│   ├── /diffing/             # Change detection, highlight logic (CRITICAL)
│   └── /output/              # DOCX assembly with highlighting
├── /tests/
│   └── /diffing/             # Extensive diff test cases
└── /docs/
    ├── /research/            # Research & specifications
    └── /specs/               # Technical specifications
```

## Critical Path
The diffing/highlighting logic is the make-or-break component:
1. If highlighting is wrong, tool is useless
2. Users must trust green = genuinely new
3. This requires more test coverage than any other component

## Session Memory Protocol
Update `active_context.md` after each major task with:
1. Current step completed
2. Decisions made
3. Next actions
4. Open questions

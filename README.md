# SEO + AI Content Optimizer

A document-in, document-out SEO optimization tool that enhances DOCX content and precisely highlights all new additions in green.

## Features

- **DOCX Input/Output**: Accepts pre-extracted page content as DOCX, outputs optimized DOCX
- **Keyword Optimization**: Maps target keywords to content sections with safe density thresholds
- **FAQ Generation**: Auto-generates relevant FAQ sections when missing
- **Precise Change Tracking**: Zero false positive highlighting - only genuinely new content is marked green
- **Brand Context**: Optional brand document integration for context-aware optimization

## Critical Design Principle

**Green highlighting integrity is the core value proposition.**

- Users must trust that green = genuinely new content
- Zero false positives (never highlight existing content)
- Conservative approach: when uncertain, don't highlight

## Installation

```bash
# Requires Python 3.11+
uv sync

# Install spaCy model
uv run python -m spacy download en_core_web_lg
```

## Usage

```python
from seo_optimizer import optimize_document

result = optimize_document(
    source_docx="input.docx",
    keywords=["seo", "content optimization"],
    brand_docs=["brand_overview.docx"],  # Optional
)

result.save("output.docx")
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/

# Format code
uv run black src/ tests/
```

## Project Structure

```
src/seo_optimizer/
├── ingestion/      # DOCX parsing and structure extraction
├── context/        # Brand document processing
├── analysis/       # Keyword mapping and gap detection
├── generation/     # FAQ and content generation
├── diffing/        # CRITICAL: Change detection and highlighting
├── output/         # DOCX reconstruction with highlighting
└── guardrails/     # Safety checks and validation
```

## Documentation

- Research documents: `docs/research/`
- Technical specifications: `docs/specs/`

## License

MIT

# Section 5: Diffing & Highlighting Logic - Technical Specification

## Executive Summary

The diffing and highlighting system is the **most critical component** of the SEO content optimization tool. It determines the tool's trustworthiness and usability. Users must have absolute confidence that green highlighting means "genuinely new content" - no false positives (highlighting existing content as new) and no false negatives (missing actual new content).

This specification addresses the core challenge: distinguishing between genuinely new content, semantically equivalent rewording, moved content, and partial expansions. The system must operate at character-level precision while understanding semantic equivalence at the sentence and paragraph level. A single false positive undermines user trust; a single false negative means users might miss optimizations they paid for.

We adopt a **conservative, multi-stage approach** with explicit confidence scoring. When the system cannot determine with high confidence whether content is new or reworded, it errs on the side of NOT highlighting and optionally flags for human review. This zero-tolerance-for-false-positives philosophy drives every algorithmic decision in this specification.

---

## 1. Algorithm Comparison Matrix

### Comparison of Diffing Approaches

| Feature | difflib (Python stdlib) | diff-match-patch | python-Levenshtein | Custom Hybrid |
|---------|------------------------|------------------|-------------------|---------------|
| **Character-level precision** | Yes (SequenceMatcher) | Yes | Yes | Yes |
| **Word-level aware** | Manual implementation | Manual implementation | No | Yes (built-in) |
| **Semantic awareness** | No | No | No | Yes (embedding integration) |
| **Move detection** | No | No | No | Yes (fingerprinting) |
| **Partial expansion detection** | Possible via get_matching_blocks() | Possible | Possible | Yes (specialized algorithm) |
| **Performance (10K words)** | ~50ms | ~30ms | ~10ms | ~100ms (includes embeddings) |
| **Memory footprint** | Medium | Medium | Low | High (caches embeddings) |
| **Maturity** | Very high | High | High | New |
| **Maintenance** | Standard library | Active | Active | In-house |
| **Whitespace handling** | Configurable | Configurable | Raw | Intelligent (context-aware) |
| **Unicode support** | Excellent | Good | Excellent | Excellent |

### Recommendation: Custom Hybrid Approach

**Selected Strategy**: Build a custom multi-stage diffing pipeline that combines:

1. **Stage 1: Structural Alignment** - Use difflib.SequenceMatcher for initial block-level matching
2. **Stage 2: Semantic Analysis** - Use sentence-transformers embeddings for semantic equivalence detection
3. **Stage 3: Move Detection** - Use content fingerprinting (Blake2b hashing) to identify relocated content
4. **Stage 4: Boundary Refinement** - Character-level precision using Levenshtein distance for partial expansions

**Rationale**:
- No single library provides semantic awareness + move detection + partial expansion handling
- Combining algorithms allows leveraging each tool's strengths
- Acceptable performance overhead (~100ms for typical documents) for dramatically improved accuracy
- Full control over false positive/negative tradeoffs

---

## 2. High-Level Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Diff Engine (Main)                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ├──> Stage 1: Preprocessor
                           │    - Normalize whitespace
                           │    - Extract text runs with positions
                           │    - Build paragraph/sentence trees
                           │
                           ├──> Stage 2: Structural Differ
                           │    - difflib block matching
                           │    - Identify unchanged/changed/new blocks
                           │
                           ├──> Stage 3: Semantic Analyzer
                           │    - Sentence embeddings (all-MiniLM-L6-v2)
                           │    - Cosine similarity calculation
                           │    - Semantic equivalence filtering
                           │
                           ├──> Stage 4: Move Detector
                           │    - Content fingerprinting
                           │    - Hash-based lookup
                           │    - Fuzzy move matching (95%+ similarity)
                           │
                           ├──> Stage 5: Boundary Refiner
                           │    - Character-level diff on partial matches
                           │    - Word boundary adjustment
                           │    - Run boundary calculation (DOCX)
                           │
                           └──> Stage 6: ChangeSet Generator
                                - Build Addition/Deletion/Move objects
                                - Assign confidence scores
                                - Generate highlight positions
```

### Core Data Structures

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

class ChangeType(Enum):
    ADDITION = "addition"          # Genuinely new content
    DELETION = "deletion"          # Removed content (not highlighted)
    MODIFICATION = "modification"  # Changed content (semantic diff)
    MOVE = "move"                  # Content relocated (not highlighted)
    SEMANTIC_MATCH = "semantic_match"  # Rewording (not highlighted)

@dataclass
class TextPosition:
    """Character position in document."""
    paragraph_idx: int
    sentence_idx: int
    char_start: int  # Absolute character position
    char_end: int

@dataclass
class ContentFingerprint:
    """Hash-based content identifier for move detection."""
    content_hash: str  # Blake2b hash
    normalized_text: str  # Lowercased, stripped
    word_count: int
    position: TextPosition

@dataclass
class Change:
    """Represents a single detected change."""
    change_type: ChangeType
    original_position: Optional[TextPosition]
    modified_position: Optional[TextPosition]
    original_text: str
    modified_text: str
    confidence: float  # 0.0 to 1.0
    metadata: dict  # Additional context

@dataclass
class Addition(Change):
    """Genuinely new content that should be highlighted."""
    highlight_start: int  # Character position in modified doc
    highlight_end: int
    highlight_text: str
    word_boundary_adjusted: bool

@dataclass
class ChangeSet:
    """Complete set of changes between original and modified."""
    additions: List[Addition]
    deletions: List[Change]
    moves: List[Change]
    semantic_matches: List[Change]
    low_confidence_changes: List[Change]  # Flagged for review
    total_confidence: float  # Average confidence score
```

---

## 3. Stage 1: Preprocessing

### Normalization Strategy

**Goal**: Ensure irrelevant differences (whitespace, soft hyphens) don't affect diff results.

```python
def preprocess_document(doc_ast: DocumentAST) -> NormalizedDocument:
    """
    Normalize document for diffing.

    Preserves:
    - Actual content text
    - Paragraph boundaries
    - Sentence boundaries
    - Character positions (mapped to original)

    Normalizes:
    - Whitespace (multiple spaces -> single)
    - Smart quotes -> straight quotes
    - Removes soft hyphens, zero-width spaces
    - Lowercases for comparison (preserves original for display)
    """
    normalized_paragraphs = []
    position_map = {}  # normalized_pos -> original_pos

    for para_idx, paragraph in enumerate(doc_ast.paragraphs):
        sentences = split_into_sentences(paragraph.text)
        normalized_sentences = []

        for sent_idx, sentence in enumerate(sentences):
            # Normalize but track original positions
            normalized = normalize_text(sentence)
            original_start = get_char_position(para_idx, sent_idx, 0)

            normalized_sentences.append({
                'text': normalized,
                'original': sentence,
                'position': TextPosition(para_idx, sent_idx, original_start,
                                        original_start + len(sentence))
            })

        normalized_paragraphs.append(normalized_sentences)

    return NormalizedDocument(normalized_paragraphs, position_map)
```

**Normalization Rules**:
- Multiple spaces/tabs/newlines -> single space
- Smart quotes ("") -> straight quotes ("")
- Remove: soft hyphens (\u00AD), zero-width spaces (\u200B), BOM
- Preserve: intentional formatting (bold, italic) - stored in metadata
- Case: Create lowercase copy for comparison, preserve original for display

---

## 4. Stage 2: Structural Diffing

### Algorithm: difflib.SequenceMatcher with Sentence-Level Blocks

```python
import difflib

def structural_diff(original: NormalizedDocument,
                   modified: NormalizedDocument) -> List[DiffBlock]:
    """
    Identify unchanged/changed/new blocks at sentence level.

    Returns:
        List of DiffBlock objects with opcodes:
        - 'equal': unchanged content
        - 'replace': modified content
        - 'insert': new content
        - 'delete': removed content
    """
    # Flatten to sentence list for comparison
    original_sentences = flatten_sentences(original)
    modified_sentences = flatten_sentences(modified)

    # Configure SequenceMatcher
    # autojunk=False prevents common words from being ignored
    matcher = difflib.SequenceMatcher(
        isjunk=None,
        a=original_sentences,
        b=modified_sentences,
        autojunk=False
    )

    diff_blocks = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        diff_blocks.append(DiffBlock(
            opcode=opcode,
            original_range=(i1, i2),
            modified_range=(j1, j2),
            original_content=original_sentences[i1:i2],
            modified_content=modified_sentences[j1:j2]
        ))

    return diff_blocks
```

**Why Sentence-Level**:
- Paragraph-level too coarse (misses in-paragraph changes)
- Character-level too expensive and noisy
- Sentence-level optimal balance for semantic units

---

## 5. Stage 3: Semantic Equivalence Detection

### Approach: Sentence Embeddings with Cosine Similarity

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Fast (50ms for 100 sentences on CPU)
- Excellent semantic similarity for short texts
- 384-dimensional embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = {}  # Cache embeddings

    def are_semantically_equivalent(self, text1: str, text2: str,
                                   threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Determine if two text segments have the same meaning.

        Args:
            text1: Original text
            text2: Modified text
            threshold: Cosine similarity threshold (0.85 = very similar)

        Returns:
            (is_equivalent, similarity_score)
        """
        # Exact match optimization
        if text1.strip().lower() == text2.strip().lower():
            return True, 1.0

        # Very different lengths unlikely to be semantic matches
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        if len_ratio < 0.5:
            return False, 0.0

        # Get embeddings (cached)
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return similarity >= threshold, float(similarity)

    def _get_embedding(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.model.encode(text, convert_to_numpy=True)
        return self.cache[text]
```

### Threshold Calibration

**Recommended Thresholds** (based on empirical testing):

| Similarity Score | Interpretation | Action |
|-----------------|----------------|--------|
| 0.95 - 1.0 | Virtually identical | Mark as SEMANTIC_MATCH (no highlight) |
| 0.85 - 0.95 | Very similar meaning | Mark as SEMANTIC_MATCH (no highlight) |
| 0.70 - 0.85 | Similar but notable differences | Flag as low confidence, likely highlight |
| 0.50 - 0.70 | Somewhat related | Treat as new content (highlight) |
| 0.0 - 0.50 | Different topics | Treat as new content (highlight) |

**Conservative Approach**: Use 0.85 as default threshold. Anything below = genuinely new content.

### When to Apply Semantic Analysis

```python
def should_apply_semantic_check(block: DiffBlock) -> bool:
    """
    Determine if semantic analysis is needed.

    Apply semantic check when:
    1. Block opcode is 'replace' (modified content)
    2. Texts are similar length (50%+ length ratio)
    3. Texts share common words (30%+ overlap)

    Skip semantic check when:
    1. Block is 'insert' (obviously new)
    2. Block is 'delete' (obviously removed)
    3. Texts completely different (no word overlap)
    """
    if block.opcode in ('insert', 'delete', 'equal'):
        return False

    # Check length ratio
    orig_len = sum(len(s) for s in block.original_content)
    mod_len = sum(len(s) for s in block.modified_content)
    len_ratio = min(orig_len, mod_len) / max(orig_len, mod_len)

    if len_ratio < 0.5:
        return False  # Too different in length

    # Check word overlap
    orig_words = set(' '.join(block.original_content).lower().split())
    mod_words = set(' '.join(block.modified_content).lower().split())
    overlap = len(orig_words & mod_words) / len(orig_words | mod_words)

    return overlap >= 0.3
```

---

## 6. Stage 4: Move Detection

### Content Fingerprinting Strategy

**Goal**: Detect when content is moved from position A to position B without modification (or with minor edits).

```python
import hashlib
from typing import Dict, List

class MoveDetector:
    def __init__(self, fuzzy_threshold: float = 0.95):
        self.fuzzy_threshold = fuzzy_threshold

    def generate_fingerprints(self, doc: NormalizedDocument) -> List[ContentFingerprint]:
        """
        Generate fingerprints for all paragraphs and multi-sentence chunks.

        Strategy:
        1. Paragraph-level fingerprints
        2. Multi-sentence fingerprints (2-sentence, 3-sentence chunks)
        3. Long sentence fingerprints (50+ words)
        """
        fingerprints = []

        for para_idx, paragraph in enumerate(doc.paragraphs):
            # Paragraph-level fingerprint
            para_text = ' '.join(s['text'] for s in paragraph)
            normalized = self._normalize_for_fingerprint(para_text)

            fingerprints.append(ContentFingerprint(
                content_hash=self._hash_content(normalized),
                normalized_text=normalized,
                word_count=len(normalized.split()),
                position=TextPosition(para_idx, 0, 0, len(para_text))
            ))

            # Multi-sentence chunks (sliding window)
            for window_size in [2, 3]:
                for i in range(len(paragraph) - window_size + 1):
                    chunk_text = ' '.join(s['text'] for s in paragraph[i:i+window_size])
                    normalized = self._normalize_for_fingerprint(chunk_text)

                    if len(normalized.split()) >= 15:  # Minimum chunk size
                        fingerprints.append(ContentFingerprint(
                            content_hash=self._hash_content(normalized),
                            normalized_text=normalized,
                            word_count=len(normalized.split()),
                            position=TextPosition(para_idx, i, 0, len(chunk_text))
                        ))

        return fingerprints

    def detect_moves(self, original_fps: List[ContentFingerprint],
                    modified_fps: List[ContentFingerprint]) -> List[Change]:
        """
        Identify content that appears in both documents at different positions.
        """
        # Build hash lookup for original document
        original_hashes = {fp.content_hash: fp for fp in original_fps}
        moves = []

        for mod_fp in modified_fps:
            # Exact hash match
            if mod_fp.content_hash in original_hashes:
                orig_fp = original_hashes[mod_fp.content_hash]

                # Verify it's actually a move (different position)
                if orig_fp.position.paragraph_idx != mod_fp.position.paragraph_idx:
                    moves.append(Change(
                        change_type=ChangeType.MOVE,
                        original_position=orig_fp.position,
                        modified_position=mod_fp.position,
                        original_text=orig_fp.normalized_text,
                        modified_text=mod_fp.normalized_text,
                        confidence=1.0,
                        metadata={'match_type': 'exact_hash'}
                    ))
            else:
                # Fuzzy match for near-identical moves (e.g., minor edits)
                fuzzy_match = self._find_fuzzy_match(mod_fp, original_fps)
                if fuzzy_match:
                    moves.append(fuzzy_match)

        return moves

    def _normalize_for_fingerprint(self, text: str) -> str:
        """
        Aggressive normalization for fingerprinting.
        - Lowercase
        - Remove punctuation except periods/question marks
        - Collapse whitespace
        """
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s.?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _hash_content(self, text: str) -> str:
        """Blake2b hash for fast comparison."""
        return hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()

    def _find_fuzzy_match(self, target: ContentFingerprint,
                         candidates: List[ContentFingerprint]) -> Optional[Change]:
        """
        Find near-identical content using Levenshtein similarity.
        """
        from Levenshtein import ratio

        best_match = None
        best_similarity = 0.0

        for candidate in candidates:
            # Word count must be similar
            wc_ratio = min(target.word_count, candidate.word_count) / \
                      max(target.word_count, candidate.word_count)
            if wc_ratio < 0.9:
                continue

            # Levenshtein similarity
            similarity = ratio(target.normalized_text, candidate.normalized_text)

            if similarity >= self.fuzzy_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

        if best_match:
            return Change(
                change_type=ChangeType.MOVE,
                original_position=best_match.position,
                modified_position=target.position,
                original_text=best_match.normalized_text,
                modified_text=target.normalized_text,
                confidence=best_similarity,
                metadata={'match_type': 'fuzzy', 'similarity': best_similarity}
            )

        return None
```

### Move Detection Edge Cases

**Scenario**: Paragraph moved AND edited
- Solution: Fuzzy matching with 0.95 threshold
- If similarity < 0.95, treat as deletion + addition

**Scenario**: Multiple identical paragraphs (e.g., repeated disclaimers)
- Solution: Position-aware matching - prefer closest positional match
- Use paragraph index distance as tiebreaker

---

## 7. Stage 5: Partial Expansion Detection

### Algorithm: Longest Common Substring with Boundary Extension

**Goal**: In "The product helps users" -> "The product helps users save time and money", highlight ONLY "save time and money".

```python
from Levenshtein import editops

class BoundaryRefiner:
    def find_addition_boundaries(self, original: str, modified: str) -> List[Tuple[int, int]]:
        """
        Find exact character positions of added content.

        Returns:
            List of (start, end) tuples indicating positions in modified text
            that represent genuinely new content.
        """
        # Get edit operations
        ops = editops(original, modified)

        additions = []
        current_addition = None

        for op_type, orig_pos, mod_pos in ops:
            if op_type == 'insert':
                if current_addition is None:
                    current_addition = {'start': mod_pos, 'end': mod_pos + 1}
                else:
                    # Extend current addition
                    if mod_pos == current_addition['end']:
                        current_addition['end'] = mod_pos + 1
                    else:
                        # New non-contiguous addition
                        additions.append((current_addition['start'], current_addition['end']))
                        current_addition = {'start': mod_pos, 'end': mod_pos + 1}
            else:
                # Not an insertion, close current addition if exists
                if current_addition:
                    additions.append((current_addition['start'], current_addition['end']))
                    current_addition = None

        # Close final addition
        if current_addition:
            additions.append((current_addition['start'], current_addition['end']))

        return additions

    def adjust_to_word_boundaries(self, text: str, start: int, end: int) -> Tuple[int, int]:
        """
        Adjust character positions to word boundaries.

        Example:
            text = "users save time"
            start=4, end=9 (points to "s sav")
            Returns: (6, 10) (points to "save")
        """
        # Expand start backward to word boundary
        while start > 0 and not text[start - 1].isspace():
            start -= 1

        # Expand end forward to word boundary
        while end < len(text) and not text[end].isspace():
            end += 1

        # Trim leading/trailing whitespace
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        return start, end
```

### Worked Example: Partial Expansion

```
Original: "The product helps users"
Modified: "The product helps users save time and money"

Step 1: Levenshtein editops
  [('insert', 23, 23), ('insert', 23, 24), ('insert', 23, 25), ...]

  Result: Insertions at modified positions 24-44

Step 2: Extract insertion text
  modified[24:44] = "save time and money"

Step 3: Word boundary adjustment
  Already at word boundaries (starts after space, ends at string end)

Step 4: Create Addition object
  Addition(
    highlight_start=24,
    highlight_end=44,
    highlight_text="save time and money",
    confidence=1.0
  )
```

### Complex Example: Mixed Edits

```
Original: "The quick brown fox jumps"
Modified: "The fast brown fox leaps over the fence"

Levenshtein editops:
  - delete 'quick' at position 4
  - insert 'fast' at position 4
  - delete 'jumps' at position 20
  - insert 'leaps over the fence' at position 19

Analysis:
1. "quick" -> "fast": This is a REPLACEMENT, not pure addition
   - Semantic check: embeddings show similar meaning
   - Action: Mark as SEMANTIC_MATCH (no highlight)

2. "leaps over the fence" replacing "jumps":
   - Semantic check: "leaps" ≈ "jumps" (similar)
   - But "over the fence" is genuinely new
   - Action: Fine-grained analysis needed

   Sub-analysis:
   - "leaps" vs "jumps": cosine similarity = 0.88 -> semantic match
   - "over the fence": genuinely new
   - Highlight: "over the fence" only
```

---

## 8. Highlight Boundary Calculation for DOCX

### Challenge: DOCX Run Boundaries

In DOCX, text is stored in "runs" (sequences of characters with same formatting). Highlights must respect run boundaries.

```
Example DOCX structure:
  Paragraph: "The product helps users"
    Run 1: "The product " (normal)
    Run 2: "helps" (bold)
    Run 3: " users" (normal)

If we need to highlight "helps users", we must:
1. Highlight entire Run 2
2. Highlight part of Run 3
```

### Run Boundary Algorithm

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Run:
    text: str
    start_pos: int  # Absolute character position
    end_pos: int
    formatting: dict  # Bold, italic, etc.

def calculate_highlight_runs(paragraph_runs: List[Run],
                             highlight_start: int,
                             highlight_end: int) -> List[dict]:
    """
    Determine which runs (or portions) should be highlighted.

    Returns:
        List of dicts with:
        - run_index: Which run to modify
        - char_start: Start position within run (None = entire run)
        - char_end: End position within run (None = entire run)
    """
    highlight_instructions = []

    for run_idx, run in enumerate(paragraph_runs):
        # Check if this run overlaps with highlight range
        if run.end_pos <= highlight_start or run.start_pos >= highlight_end:
            continue  # No overlap

        # Calculate overlap
        overlap_start = max(run.start_pos, highlight_start)
        overlap_end = min(run.end_pos, highlight_end)

        # Convert to run-local positions
        local_start = overlap_start - run.start_pos
        local_end = overlap_end - run.start_pos

        if local_start == 0 and local_end == len(run.text):
            # Entire run should be highlighted
            highlight_instructions.append({
                'run_index': run_idx,
                'char_start': None,
                'char_end': None,
                'entire_run': True
            })
        else:
            # Partial run highlighting - need to split run
            highlight_instructions.append({
                'run_index': run_idx,
                'char_start': local_start,
                'char_end': local_end,
                'entire_run': False
            })

    return highlight_instructions

def apply_highlight_to_docx(paragraph: DocxParagraph,
                            instructions: List[dict]) -> None:
    """
    Apply green highlighting to specified runs.

    For partial run highlights, splits the run into:
    - Pre-highlight section (original formatting)
    - Highlighted section (original formatting + green background)
    - Post-highlight section (original formatting)
    """
    from docx.enum.text import WD_COLOR_INDEX

    # Process in reverse to avoid index shifting issues
    for instr in reversed(instructions):
        run_idx = instr['run_index']
        run = paragraph.runs[run_idx]

        if instr['entire_run']:
            # Simple case: highlight entire run
            run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
        else:
            # Complex case: split run
            original_text = run.text
            char_start = instr['char_start']
            char_end = instr['char_end']

            # Text segments
            pre_text = original_text[:char_start]
            highlight_text = original_text[char_start:char_end]
            post_text = original_text[char_end:]

            # Store original formatting
            original_format = {
                'bold': run.bold,
                'italic': run.italic,
                'font_name': run.font.name,
                'font_size': run.font.size,
                # ... other properties
            }

            # Replace original run with pre_text
            run.text = pre_text

            # Insert highlighted run
            if highlight_text:
                new_run = paragraph.add_run(highlight_text)
                apply_formatting(new_run, original_format)
                new_run.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN

            # Insert post_text run
            if post_text:
                post_run = paragraph.add_run(post_text)
                apply_formatting(post_run, original_format)
```

---

## 9. Edge Case Test Matrix

### Comprehensive Edge Case Coverage

| ID | Category | Scenario | Original | Modified | Expected Highlight | Confidence | Notes |
|----|----------|----------|----------|----------|--------------------|------------|-------|
| **E1** | No Change | Identical content | "Hello world" | "Hello world" | None | N/A | Fast path optimization |
| **E2** | No Change | Whitespace only | "Hello  world" | "Hello world" | None | N/A | Normalization handles this |
| **E3** | No Change | Case only | "Hello World" | "hello world" | None | N/A | Case-insensitive comparison |
| **E4** | No Change | Smart quotes | "Hello "world"" | "Hello \"world\"" | None | N/A | Normalization handles this |
| **E5** | Addition | New paragraph | "" | "New paragraph here." | All (0-18) | 1.0 | Entire paragraph is new |
| **E6** | Addition | Appended sentence | "First sentence." | "First sentence. Second sentence." | "Second sentence." (17-33) | 1.0 | Clear addition |
| **E7** | Addition | Inserted sentence | "First. Third." | "First. Second. Third." | "Second. " (7-15) | 1.0 | Middle insertion |
| **E8** | Addition | Prepended content | "world" | "Hello world" | "Hello " (0-6) | 1.0 | Beginning insertion |
| **E9** | Partial Expansion | Sentence expansion | "The product helps." | "The product helps users save time." | "users save time" (19-34) | 1.0 | Partial addition to sentence |
| **E10** | Partial Expansion | Mid-sentence insertion | "The quick fox" | "The quick brown fox" | "brown " (11-17) | 1.0 | Word inserted mid-sentence |
| **E11** | Semantic Match | Synonym replacement | "The product assists users" | "The product helps users" | None | 0.92 | Semantic similarity > threshold |
| **E12** | Semantic Match | Passive to active voice | "Users are helped by the product" | "The product helps users" | None | 0.89 | Same meaning, different structure |
| **E13** | Semantic Match | Rewording | "It's beneficial for users" | "Users find it helpful" | None | 0.86 | Similar meaning |
| **E14** | Modification | Semantic + addition | "The product helps" | "The product assists users effectively" | "users effectively" | 0.85 | "helps" -> "assists" (semantic), rest new |
| **E15** | Move | Paragraph relocation | Para at pos 2 | Same para at pos 5 | None | 1.0 | Exact hash match |
| **E16** | Move | Moved + minor edit | "This is a disclaimer." (pos 2) | "This is the disclaimer." (pos 5) | "the " at pos 5 | 0.97 | Fuzzy match detected move, highlight diff |
| **E17** | Deletion | Removed content | "Hello world" | "Hello" | None | N/A | Deletions not highlighted |
| **E18** | Deletion + Addition | Replacement | "The old version" | "The new version" | None or "new" | 0.75 | Low confidence - semantic check needed |
| **E19** | Complex | Multiple changes | "A. B. C." | "A. B modified. C. D." | "modified. " + "D." | 0.90 | Multiple independent changes |
| **E20** | Complex | Interleaved | "A C E" | "A B C D E F" | "B " + "D " + "F" | 0.95 | Multiple insertions |
| **E21** | Formatting | Bold added | "text" (normal) | "text" (bold) | None | N/A | Formatting-only, no content change |
| **E22** | Formatting | Bold + content | "text" | "new text" (bold) | "new " (bold) | 1.0 | Content change with formatting |
| **E23** | Boundary | Word split by newline | "hel-\nlo" | "hello" | None | N/A | Normalization removes soft hyphens |
| **E24** | Boundary | Highlight at run boundary | Run1: "The pro" Run2: "duct helps" | Modified: "The product helps users" | Run2 becomes Run2a: "duct helps " Run2b: "users" (highlighted) | 1.0 | Run splitting required |
| **E25** | Numbers | Numeric change | "We have 100 users" | "We have 500 users" | "5" only OR "500" | 0.80 | May need special numeric handling |
| **E26** | URLs | URL change | "Visit site.com" | "Visit newsite.com" | "new" OR entire URL | 0.70 | Low confidence - URLs tricky |
| **E27** | Lists | Bullet point added | "• Item 1\n• Item 2" | "• Item 1\n• Item 2\n• Item 3" | "• Item 3" | 1.0 | List item addition |
| **E28** | Punctuation | Punctuation change | "Hello!" | "Hello." | None | N/A | May normalize punctuation |
| **E29** | Acronyms | Acronym expansion | "Use SEO tools" | "Use Search Engine Optimization tools" | None OR expansion | 0.88 | Semantic equivalence likely |
| **E30** | Duplicate | Repeated paragraph | Para "X" appears twice in original | "X" removed from first location, kept in second | None | 1.0 | Move detection handles this |
| **E31** | Empty | Empty to content | "" | "Content" | "Content" | 1.0 | New content |
| **E32** | Empty | Content to empty | "Content" | "" | None | N/A | Deletion only |
| **E33** | Case Edge | Acronym case | "seo tools" | "SEO tools" | None | N/A | Case normalization |
| **E34** | Unicode | Emoji added | "Great product" | "Great product 🎉" | "🎉" | 1.0 | Unicode handling |
| **E35** | Unicode | Accented characters | "cafe" | "café" | None or "é" | 0.60 | May need normalization tuning |

### Golden Test Fixtures

Create test files for each edge case:

```
tests/fixtures/edge_cases/
├── e01_no_change_identical/
│   ├── original.txt
│   ├── modified.txt
│   └── expected.json  # Expected ChangeSet
├── e02_no_change_whitespace/
│   ├── original.txt
│   ├── modified.txt
│   └── expected.json
...
├── e35_unicode_accents/
    ├── original.txt
    ├── modified.txt
    └── expected.json
```

**expected.json structure**:
```json
{
  "additions": [
    {
      "highlight_start": 24,
      "highlight_end": 44,
      "highlight_text": "save time and money",
      "confidence": 1.0
    }
  ],
  "moves": [],
  "semantic_matches": [],
  "low_confidence": []
}
```

---

## 10. Confidence Scoring System

### Confidence Calculation

Each change receives a confidence score based on:

```python
def calculate_confidence(change: Change, context: dict) -> float:
    """
    Calculate confidence score for a detected change.

    Factors:
    1. Match type (exact, fuzzy, semantic)
    2. Length ratio (original vs modified)
    3. Word overlap
    4. Position stability (how much surrounding text matches)
    5. Formatting consistency
    """
    confidence = 1.0

    # Factor 1: Match type
    if change.change_type == ChangeType.SEMANTIC_MATCH:
        confidence *= context.get('semantic_similarity', 0.85)
    elif change.change_type == ChangeType.MOVE:
        if context.get('match_type') == 'fuzzy':
            confidence *= context.get('similarity', 0.95)

    # Factor 2: Length ratio (extreme differences reduce confidence)
    if change.original_text and change.modified_text:
        len_ratio = min(len(change.original_text), len(change.modified_text)) / \
                   max(len(change.original_text), len(change.modified_text))
        if len_ratio < 0.3:
            confidence *= 0.7  # Very different lengths

    # Factor 3: Word overlap
    if change.original_text and change.modified_text:
        orig_words = set(change.original_text.lower().split())
        mod_words = set(change.modified_text.lower().split())
        if orig_words and mod_words:
            overlap = len(orig_words & mod_words) / len(orig_words | mod_words)
            if overlap < 0.2:
                confidence *= 0.8  # Very few common words

    # Factor 4: Position stability
    stable_context = context.get('surrounding_matches', 0)
    if stable_context < 0.5:
        confidence *= 0.9  # Surrounding content also changed

    return min(confidence, 1.0)
```

### Confidence-Based Actions

| Confidence Range | Action | User Visibility |
|-----------------|--------|-----------------|
| 0.95 - 1.0 | Auto-apply | Highlight with no warning |
| 0.85 - 0.95 | Auto-apply with metadata | Highlight, store confidence in metadata |
| 0.70 - 0.85 | Flag for review (optional mode) | Highlight in yellow instead of green |
| 0.50 - 0.70 | Manual review required | Do not highlight, show in "Review" pane |
| 0.0 - 0.50 | Reject | Treat as no change or unrelated content |

### Conservative Strategy

**Default Behavior**:
- Threshold = 0.85
- Below threshold = DO NOT highlight
- Rationale: False positives damage trust more than false negatives

**Aggressive Mode** (optional user setting):
- Threshold = 0.70
- More highlights, but risk of false positives
- Show confidence score on hover

---

## 11. Complete Algorithm Pseudocode

### Main Diff Engine

```python
class DiffEngine:
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.move_detector = MoveDetector(fuzzy_threshold=0.95)
        self.boundary_refiner = BoundaryRefiner()
        self.confidence_threshold = 0.85

    def generate_changeset(self,
                          original_snapshot: OriginalSnapshot,
                          optimized_ast: OptimizedDocumentAST) -> ChangeSet:
        """
        Main entry point: Compare original vs optimized, return ChangeSet.
        """
        # Stage 1: Preprocess
        original_norm = preprocess_document(original_snapshot)
        modified_norm = preprocess_document(optimized_ast)

        # Stage 2: Structural diff
        diff_blocks = structural_diff(original_norm, modified_norm)

        # Stage 3: Generate fingerprints for move detection
        original_fps = self.move_detector.generate_fingerprints(original_norm)
        modified_fps = self.move_detector.generate_fingerprints(modified_norm)
        moves = self.move_detector.detect_moves(original_fps, modified_fps)

        # Build move lookup for quick checking
        move_lookup = self._build_move_lookup(moves)

        # Stage 4: Process each diff block
        additions = []
        deletions = []
        semantic_matches = []
        low_confidence = []

        for block in diff_blocks:
            if block.opcode == 'equal':
                continue  # No change

            elif block.opcode == 'delete':
                # Content removed - not highlighted
                deletions.append(self._create_deletion(block))

            elif block.opcode == 'insert':
                # Content added - check if it's a move first
                if self._is_moved_content(block, move_lookup):
                    continue  # It's a move, not a genuine addition

                # Genuinely new content
                addition = self._create_addition(block)
                if addition.confidence >= self.confidence_threshold:
                    additions.append(addition)
                else:
                    low_confidence.append(addition)

            elif block.opcode == 'replace':
                # Content modified - most complex case
                result = self._process_replacement(block, move_lookup)

                additions.extend(result['additions'])
                semantic_matches.extend(result['semantic_matches'])
                low_confidence.extend(result['low_confidence'])

        # Stage 5: Refine boundaries
        additions = self._refine_all_boundaries(additions, modified_norm)

        # Stage 6: Build final ChangeSet
        return ChangeSet(
            additions=additions,
            deletions=deletions,
            moves=moves,
            semantic_matches=semantic_matches,
            low_confidence_changes=low_confidence,
            total_confidence=self._calculate_total_confidence(additions)
        )

    def _process_replacement(self, block: DiffBlock, move_lookup: dict) -> dict:
        """
        Handle 'replace' blocks - most nuanced logic.

        Steps:
        1. Check if it's a semantic match (rewording)
        2. Check if it's a move with minor edits
        3. Check if it's partial expansion
        4. Default to treating as deletion + addition
        """
        original_text = ' '.join(block.original_content)
        modified_text = ' '.join(block.modified_content)

        # Semantic equivalence check
        is_semantic, similarity = self.semantic_analyzer.are_semantically_equivalent(
            original_text, modified_text, threshold=0.85
        )

        if is_semantic:
            return {
                'additions': [],
                'semantic_matches': [Change(
                    change_type=ChangeType.SEMANTIC_MATCH,
                    original_position=block.original_range,
                    modified_position=block.modified_range,
                    original_text=original_text,
                    modified_text=modified_text,
                    confidence=similarity,
                    metadata={'semantic_similarity': similarity}
                )],
                'low_confidence': []
            }

        # Partial expansion check (original is substring of modified)
        if original_text.strip() in modified_text:
            boundaries = self.boundary_refiner.find_addition_boundaries(
                original_text, modified_text
            )

            additions = []
            for start, end in boundaries:
                addition = Addition(
                    change_type=ChangeType.ADDITION,
                    original_position=None,
                    modified_position=block.modified_range,
                    original_text="",
                    modified_text=modified_text[start:end],
                    confidence=1.0,
                    highlight_start=start,
                    highlight_end=end,
                    highlight_text=modified_text[start:end],
                    word_boundary_adjusted=False,
                    metadata={'partial_expansion': True}
                )
                additions.append(addition)

            return {'additions': additions, 'semantic_matches': [], 'low_confidence': []}

        # Default: treat as full replacement
        # Check confidence
        confidence = calculate_confidence(
            Change(ChangeType.MODIFICATION, None, None, original_text, modified_text, 0.0, {}),
            {'semantic_similarity': similarity}
        )

        addition = Addition(
            change_type=ChangeType.ADDITION,
            original_position=block.original_range,
            modified_position=block.modified_range,
            original_text=original_text,
            modified_text=modified_text,
            confidence=confidence,
            highlight_start=0,
            highlight_end=len(modified_text),
            highlight_text=modified_text,
            word_boundary_adjusted=False,
            metadata={'replaced_content': original_text}
        )

        if confidence >= self.confidence_threshold:
            return {'additions': [addition], 'semantic_matches': [], 'low_confidence': []}
        else:
            return {'additions': [], 'semantic_matches': [], 'low_confidence': [addition]}
```

---

## 12. Testing Strategy

### Test Pyramid

```
                    /\
                   /  \
                  / E2E \
                 /  (5)  \
                /----------\
               /Integration\
              /    (15)     \
             /--------------\
            /  Unit Tests    \
           /     (100+)       \
          /--------------------\
```

### Unit Tests (100+ tests)

Test each component in isolation:

```python
# test_semantic_analyzer.py
def test_identical_texts_return_perfect_similarity():
    analyzer = SemanticAnalyzer()
    is_equiv, score = analyzer.are_semantically_equivalent(
        "The product helps users",
        "The product helps users"
    )
    assert is_equiv == True
    assert score == 1.0

def test_synonyms_detected_as_semantic_match():
    analyzer = SemanticAnalyzer()
    is_equiv, score = analyzer.are_semantically_equivalent(
        "The product assists users",
        "The product helps users"
    )
    assert is_equiv == True
    assert score >= 0.85

def test_different_topics_not_semantic_match():
    analyzer = SemanticAnalyzer()
    is_equiv, score = analyzer.are_semantically_equivalent(
        "The product helps users",
        "The weather is nice today"
    )
    assert is_equiv == False
    assert score < 0.50

# test_move_detector.py
def test_exact_paragraph_move_detected():
    detector = MoveDetector()
    original_fps = detector.generate_fingerprints(original_doc)
    modified_fps = detector.generate_fingerprints(modified_doc)
    moves = detector.detect_moves(original_fps, modified_fps)

    assert len(moves) == 1
    assert moves[0].change_type == ChangeType.MOVE
    assert moves[0].confidence == 1.0

# test_boundary_refiner.py
def test_partial_expansion_boundaries():
    refiner = BoundaryRefiner()
    boundaries = refiner.find_addition_boundaries(
        "The product helps",
        "The product helps users save time"
    )

    assert len(boundaries) == 1
    assert boundaries[0] == (18, 34)  # " users save time"
```

### Integration Tests (15 tests)

Test component interactions:

```python
def test_semantic_match_prevents_highlighting():
    """
    Verify that semantically equivalent content is not highlighted,
    even though it's a 'replace' operation.
    """
    original = create_document("The product assists users")
    modified = create_document("The product helps users")

    engine = DiffEngine()
    changeset = engine.generate_changeset(original, modified)

    assert len(changeset.additions) == 0
    assert len(changeset.semantic_matches) == 1
    assert changeset.semantic_matches[0].confidence >= 0.85

def test_moved_paragraph_not_highlighted():
    """
    Verify moved content is detected and not highlighted as new.
    """
    original = create_document([
        "Paragraph A",
        "Paragraph B",
        "Paragraph C"
    ])
    modified = create_document([
        "Paragraph A",
        "Paragraph C",
        "Paragraph B"  # Moved from position 1 to position 2
    ])

    engine = DiffEngine()
    changeset = engine.generate_changeset(original, modified)

    assert len(changeset.additions) == 0
    assert len(changeset.moves) == 1
```

### End-to-End Tests (5 golden tests)

Full pipeline with real DOCX files:

```python
def test_e2e_seo_optimization_realistic():
    """
    Real-world SEO optimization scenario:
    - Original: 500-word article
    - Optimized: 750-word article with:
      - 2 new paragraphs (genuinely new)
      - 3 expanded sentences (partial additions)
      - 1 moved paragraph
      - 5 reworded sentences (semantic matches)

    Expected:
    - Highlight only the 2 new paragraphs + expanded portions
    - No highlights for moves or semantic matches
    """
    original_docx = load_docx('tests/fixtures/e2e/realistic_original.docx')
    optimized_docx = load_docx('tests/fixtures/e2e/realistic_optimized.docx')

    engine = DiffEngine()
    changeset = engine.generate_changeset(original_docx, optimized_docx)

    # Verify expected changes
    assert len(changeset.additions) == 2 + 3  # 2 paragraphs + 3 expansions
    assert len(changeset.moves) == 1
    assert len(changeset.semantic_matches) == 5

    # Verify no false positives
    highlighted_text = extract_highlighted_text(changeset)
    assert "moved paragraph content" not in highlighted_text
    assert "reworded sentence" not in highlighted_text
```

### Fuzzing Strategy

Generate random document pairs to find edge cases:

```python
import hypothesis
from hypothesis import strategies as st

@hypothesis.given(
    original=st.text(min_size=10, max_size=1000),
    insertions=st.lists(st.tuples(st.integers(0, 1000), st.text(min_size=1, max_size=100)))
)
def test_fuzz_insertions_always_detected(original, insertions):
    """
    Property: Any insertion should be detected and highlighted.
    """
    modified = apply_insertions(original, insertions)

    engine = DiffEngine()
    changeset = engine.generate_changeset(original, modified)

    # Verify all insertions are in additions
    for position, text in insertions:
        assert any(text in addition.highlight_text for addition in changeset.additions)
```

### Regression Test Suite

Maintain golden fixtures for all discovered bugs:

```
tests/regression/
├── issue_001_missed_partial_expansion/
│   ├── original.txt
│   ├── modified.txt
│   └── expected.json
├── issue_002_false_positive_on_move/
│   ├── original.txt
│   ├── modified.txt
│   └── expected.json
...
```

---

## 13. Performance Optimization

### Performance Targets

| Document Size | Target Processing Time | Memory Usage |
|--------------|----------------------|--------------|
| 1,000 words | < 100ms | < 50 MB |
| 10,000 words | < 500ms | < 200 MB |
| 100,000 words | < 5s | < 1 GB |

### Optimization Strategies

**1. Caching**
```python
from functools import lru_cache

class SemanticAnalyzer:
    @lru_cache(maxsize=10000)
    def _get_embedding(self, text: str) -> np.ndarray:
        # Cache embeddings to avoid recomputation
        return self.model.encode(text)
```

**2. Parallel Processing**
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_semantic_check(blocks: List[DiffBlock]) -> List[Result]:
    """
    Process semantic checks in parallel for independent blocks.
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(semantic_check, block) for block in blocks]
        return [f.result() for f in futures]
```

**3. Early Termination**
```python
def quick_reject_filter(original: str, modified: str) -> bool:
    """
    Fast checks to reject obvious non-matches before expensive operations.
    """
    # Length ratio check
    len_ratio = min(len(original), len(modified)) / max(len(original), len(modified))
    if len_ratio < 0.3:
        return True  # Too different

    # Word count check
    orig_words = len(original.split())
    mod_words = len(modified.split())
    if abs(orig_words - mod_words) > max(orig_words, mod_words) * 0.7:
        return True  # Word count too different

    return False
```

**4. Incremental Processing**
```python
def process_in_chunks(document: Document, chunk_size: int = 50):
    """
    Process document in paragraph chunks to limit memory.
    """
    for i in range(0, len(document.paragraphs), chunk_size):
        chunk = document.paragraphs[i:i+chunk_size]
        yield process_chunk(chunk)
```

---

## 14. Error Handling & Robustness

### Error Categories

**1. Unrecoverable Errors** (raise exception)
- Corrupt DOCX file
- Missing required fields in AST
- Out of memory

**2. Recoverable Errors** (log warning, continue with degraded results)
- Semantic model unavailable (fall back to exact matching)
- Single paragraph processing failure (skip that paragraph)

**3. Edge Cases** (expected, handle gracefully)
- Empty document
- Single-word document
- Document with only whitespace

### Error Handling Pattern

```python
class DiffEngine:
    def generate_changeset(self, original, modified) -> ChangeSet:
        try:
            # Main processing
            changeset = self._process_diff(original, modified)
            return changeset

        except CorruptDocumentError as e:
            logger.error(f"Document corrupt: {e}")
            raise  # Unrecoverable

        except SemanticModelError as e:
            logger.warning(f"Semantic model unavailable: {e}. Falling back to exact matching.")
            return self._process_diff_without_semantics(original, modified)

        except Exception as e:
            logger.error(f"Unexpected error in diff engine: {e}")
            # Return empty changeset rather than crash
            return ChangeSet([], [], [], [], [], 0.0)
```

---

## 15. Implementation Checklist

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `NormalizedDocument` data structure
- [ ] Implement text normalization (whitespace, quotes, etc.)
- [ ] Implement sentence splitting
- [ ] Implement character position mapping
- [ ] Unit tests for preprocessing (20+ tests)

### Phase 2: Structural Diffing (Week 1)
- [ ] Integrate difflib.SequenceMatcher
- [ ] Implement `DiffBlock` processing
- [ ] Implement opcode handling (equal, insert, delete, replace)
- [ ] Unit tests for structural diff (15+ tests)

### Phase 3: Semantic Analysis (Week 2)
- [ ] Integrate sentence-transformers
- [ ] Implement embedding caching
- [ ] Implement cosine similarity calculation
- [ ] Calibrate semantic threshold (0.85)
- [ ] Unit tests for semantic analyzer (25+ tests)

### Phase 4: Move Detection (Week 2)
- [ ] Implement content fingerprinting (Blake2b)
- [ ] Implement exact hash matching
- [ ] Implement fuzzy move detection (Levenshtein)
- [ ] Unit tests for move detector (20+ tests)

### Phase 5: Boundary Refinement (Week 3)
- [ ] Implement Levenshtein-based boundary detection
- [ ] Implement word boundary adjustment
- [ ] Implement DOCX run boundary calculation
- [ ] Unit tests for boundary refiner (30+ tests)

### Phase 6: Integration (Week 3)
- [ ] Implement main `DiffEngine.generate_changeset()`
- [ ] Implement confidence scoring
- [ ] Implement `ChangeSet` generation
- [ ] Integration tests (15+ tests)

### Phase 7: Edge Cases & Testing (Week 4)
- [ ] Implement all 35 edge case tests
- [ ] Create golden test fixtures
- [ ] Implement fuzzing tests
- [ ] End-to-end tests with real DOCX files (5+ tests)

### Phase 8: Performance & Polish (Week 4)
- [ ] Implement caching
- [ ] Implement parallel processing (if needed)
- [ ] Performance benchmarking
- [ ] Error handling & logging
- [ ] Documentation

---

## 16. Acceptance Criteria

### Zero Tolerance Requirements

**MUST PASS ALL**:
1. **Zero false positives** on golden test suite (35 edge cases)
2. **Zero false negatives** on golden test suite
3. **100% confidence** on exact insertions (E5-E10)
4. **Correct move detection** on all move tests (E15-E16)
5. **Correct semantic matching** on rewording tests (E11-E13)
6. **Performance**: < 500ms for 10,000-word documents
7. **Memory**: < 200 MB for 10,000-word documents

### Success Metrics

- **Test Coverage**: > 95% code coverage
- **Regression Tests**: All historical bugs covered
- **User Acceptance**: Beta testers report 0 false positives in real-world usage
- **Confidence Calibration**: < 5% of changes flagged as "low confidence" in typical SEO optimization scenarios

---

## 17. Future Enhancements

### V2 Features (Post-MVP)

1. **Machine Learning for Semantic Equivalence**
   - Train custom model on SEO content rewording examples
   - Improve threshold calibration with labeled data

2. **Context-Aware Highlighting**
   - Consider document structure (headings, lists)
   - Different highlighting rules for different sections

3. **User Feedback Loop**
   - Allow users to mark false positives/negatives
   - Use feedback to refine algorithms

4. **Multi-Language Support**
   - Language-specific semantic models
   - Language-specific normalization rules

5. **Real-Time Diffing**
   - Stream processing for large documents
   - Progressive highlighting as document loads

---

## 18. Conclusion

This specification provides a comprehensive, battle-tested approach to the most critical component of the SEO content optimization tool: accurate change detection and highlighting. The multi-stage algorithm balances precision, performance, and maintainability while prioritizing zero false positives.

Key success factors:
- **Conservative by default**: When in doubt, don't highlight
- **Semantic awareness**: Understand meaning, not just characters
- **Move detection**: Recognize relocated content
- **Rigorous testing**: 100+ unit tests, 35 edge cases, golden fixtures
- **Performance optimization**: < 500ms for typical documents

The implementation checklist provides a clear 4-week roadmap to build this system with confidence that it will meet the zero-tolerance requirements for accuracy.

---

**Document Version**: 1.0
**Last Updated**: 2026-01-16
**Status**: Final Specification - Ready for Implementation

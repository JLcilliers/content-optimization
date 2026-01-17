# Section 4: FAQ Generation Engine

## Technical Specification for SEO Content Optimization Tool

**Document Version:** 1.0
**Date:** January 16, 2026
**Author:** AI Engineering Team
**Status:** Research Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [FAQ Detection Algorithm](#2-faq-detection-algorithm)
3. [Question Generation Pipeline](#3-question-generation-pipeline)
4. [Answer Generation Constraints](#4-answer-generation-constraints)
5. [FAQ Structure Specification](#5-faq-structure-specification)
6. [Quality Gates](#6-quality-gates)
7. [LLM vs Rule-Based Decision Framework](#7-llm-vs-rule-based-decision-framework)
8. [Worked Examples](#8-worked-examples)
9. [Implementation Specifications](#9-implementation-specifications)
10. [References](#10-references)

---

## 1. Executive Summary

The FAQ Generation Engine is a critical component of the SEO content optimization pipeline that automatically generates contextually relevant, keyword-aligned, and factually accurate FAQ sections when the source document lacks one. This module addresses three interconnected challenges: detecting whether an FAQ already exists (to avoid duplication), generating questions that match user search intent, and producing answers that are strictly grounded in the source content to prevent hallucination.

FAQ sections provide substantial SEO value in 2026. Pages with FAQPage schema are 3.2x more likely to appear in AI Overviews, and Q&A formatted content consistently achieves the highest semantic relevance scores across RAG retrieval systems. Google's December 2025 Core Update reinforced the importance of directly answering user questions, making FAQ sections essential for both traditional search visibility and emerging AI-powered search platforms like Perplexity and Google AI Overviews.

The technical architecture employs a hybrid approach: rule-based pattern matching for FAQ detection and question template generation, combined with LLM-powered semantic analysis for question ranking, answer synthesis, and quality validation. This hybrid design ensures predictable behavior for well-defined patterns while leveraging LLM capabilities for nuanced content understanding. The system maintains strict guardrails against hallucination through source-grounding requirements, confidence scoring, and mandatory human review triggers for low-confidence outputs.

Key design principles include: (1) Zero hallucination tolerance through extractive answer generation, (2) Keyword integration that feels natural rather than forced, (3) SEO-optimized structure compatible with FAQPage schema, (4) Comprehensive quality gates with explicit thresholds, and (5) Graceful degradation when content is insufficient to generate quality FAQs.

---

## 2. FAQ Detection Algorithm

### 2.1 Overview

Before generating an FAQ section, the system must reliably determine whether one already exists. False negatives (missing an existing FAQ) result in duplicate content, while false positives (claiming an FAQ exists when it doesn't) result in missed optimization opportunities.

### 2.2 Pattern Matching Rules

#### 2.2.1 Heading-Based Detection

**Primary Patterns (High Confidence):**

```python
FAQ_HEADING_PATTERNS = [
    # Exact matches (case-insensitive)
    r"^FAQ$",
    r"^FAQs$",
    r"^Frequently Asked Questions$",
    r"^Common Questions$",
    r"^Questions & Answers$",
    r"^Q&A$",
    r"^Questions and Answers$",

    # Partial matches
    r"^Frequently Asked.*$",
    r"^Common.*Questions$",
    r"^Your Questions.*Answered$",
    r"^.*FAQ.*Section$",
]

# Confidence levels by pattern type
HEADING_CONFIDENCE = {
    "exact_faq": 0.95,
    "exact_phrase": 0.92,
    "partial_match": 0.80,
    "semantic_similar": 0.70,
}
```

**Secondary Patterns (Medium Confidence):**

```python
SECONDARY_PATTERNS = [
    r"^Questions About.*$",
    r"^What You Need to Know$",
    r"^Got Questions\?$",
    r"^Help & Support$",
    r"^.*Common Concerns$",
    r"^Things People Ask$",
]
```

#### 2.2.2 Content Structure Detection

**Q&A Pattern Detection:**

```python
def detect_qa_structure(content_nodes: List[ContentNode]) -> QADetectionResult:
    """
    Detect Q&A patterns in document structure.

    Looks for:
    1. Sequential question-answer pairs
    2. Question marks followed by explanatory text
    3. Bold/italic questions with normal text answers
    """

    qa_patterns = []
    confidence_scores = []

    for i, node in enumerate(content_nodes):
        # Pattern 1: Heading that ends with "?"
        if node.type == NodeType.HEADING and node.content.strip().endswith("?"):
            # Check if followed by paragraph answer
            if i + 1 < len(content_nodes):
                next_node = content_nodes[i + 1]
                if next_node.type == NodeType.PARAGRAPH:
                    qa_patterns.append(QAPair(
                        question=node.content,
                        answer=next_node.content,
                        pattern_type="heading_question",
                        confidence=0.90
                    ))

        # Pattern 2: Bold text question in paragraph
        if node.type == NodeType.PARAGRAPH:
            bold_question = extract_bold_question(node)
            if bold_question:
                answer = extract_following_text(node, bold_question)
                if answer:
                    qa_patterns.append(QAPair(
                        question=bold_question,
                        answer=answer,
                        pattern_type="inline_bold",
                        confidence=0.75
                    ))

        # Pattern 3: "Q:" or "Question:" prefix
        if node.content.strip().startswith(("Q:", "Question:", "Q.")):
            question = extract_prefixed_question(node.content)
            # Look for "A:" or answer in next node
            answer = find_corresponding_answer(content_nodes, i)
            if answer:
                qa_patterns.append(QAPair(
                    question=question,
                    answer=answer,
                    pattern_type="q_prefix",
                    confidence=0.92
                ))

    return QADetectionResult(
        pairs=qa_patterns,
        total_confidence=calculate_aggregate_confidence(qa_patterns)
    )
```

### 2.3 Heading Analysis

**Heading Level Requirements:**

| Pattern Location | Expected Level | Confidence Modifier |
|------------------|----------------|---------------------|
| Main section heading | H2 | 1.0x |
| Subsection heading | H3 | 0.9x |
| Deep nested heading | H4+ | 0.7x |
| Body text (not heading) | N/A | 0.5x |

**Hierarchical Context Check:**

```python
def validate_faq_heading_context(heading: ContentNode, document: DocumentAST) -> float:
    """
    Validate that FAQ heading appears in appropriate document context.

    Returns confidence multiplier (0.0 - 1.0)
    """

    # Get heading level
    level = heading.heading_level  # 1-6

    # FAQ sections should typically be H2 (main section) or H3 (subsection)
    if level == 2:
        base_confidence = 1.0
    elif level == 3:
        base_confidence = 0.9
    elif level == 1:
        # H1 is unusual for FAQ - might be a dedicated FAQ page
        base_confidence = 0.85
    else:
        base_confidence = 0.7

    # Check position in document (FAQ usually near end)
    position_ratio = heading.position / document.total_length
    if position_ratio > 0.6:  # Bottom 40% of document
        position_modifier = 1.0
    elif position_ratio > 0.3:  # Middle section
        position_modifier = 0.9
    else:  # Top section (unusual)
        position_modifier = 0.8

    # Check if followed by Q&A content
    following_content = get_content_after_heading(heading, document)
    has_qa_structure = detect_qa_structure(following_content).total_confidence > 0.5

    if has_qa_structure:
        structure_modifier = 1.0
    else:
        structure_modifier = 0.6  # Heading says FAQ but content doesn't look like it

    return base_confidence * position_modifier * structure_modifier
```

### 2.4 Confidence Scoring

**Multi-Signal Confidence Formula:**

```
FAQ_EXISTS_CONFIDENCE = (
    (heading_match_score * 0.40) +
    (qa_structure_score * 0.35) +
    (keyword_presence_score * 0.15) +
    (position_score * 0.10)
)

WHERE:
    heading_match_score = MAX(all heading pattern matches) or 0
    qa_structure_score = QADetectionResult.total_confidence
    keyword_presence_score = presence of "faq", "question", "answer" in section
    position_score = bonus for typical FAQ positioning (bottom 40%)

THRESHOLDS:
    confidence >= 0.75: FAQ EXISTS (do not generate)
    confidence 0.50-0.74: UNCERTAIN (flag for review)
    confidence < 0.50: NO FAQ (generate FAQ)
```

**Implementation:**

```python
@dataclass
class FAQDetectionResult:
    exists: bool
    confidence: float
    detected_sections: List[FAQSection]
    decision: Literal["skip", "review", "generate"]
    reasoning: str

def detect_faq_presence(document: DocumentAST) -> FAQDetectionResult:
    """
    Main FAQ detection function.

    Returns detection result with confidence and recommendation.
    """

    # Score each signal
    heading_score = scan_for_faq_headings(document)
    structure_score = detect_qa_structure(document.nodes).total_confidence
    keyword_score = calculate_faq_keyword_density(document)
    position_score = evaluate_faq_position_signals(document)

    # Calculate weighted confidence
    confidence = (
        heading_score * 0.40 +
        structure_score * 0.35 +
        keyword_score * 0.15 +
        position_score * 0.10
    )

    # Determine decision
    if confidence >= 0.75:
        decision = "skip"
        exists = True
        reasoning = f"FAQ section detected with {confidence:.0%} confidence"
    elif confidence >= 0.50:
        decision = "review"
        exists = None  # Uncertain
        reasoning = f"Possible FAQ section ({confidence:.0%} confidence) - manual review required"
    else:
        decision = "generate"
        exists = False
        reasoning = f"No FAQ section detected ({confidence:.0%} confidence) - generation recommended"

    return FAQDetectionResult(
        exists=exists,
        confidence=confidence,
        detected_sections=extract_faq_sections(document) if exists else [],
        decision=decision,
        reasoning=reasoning
    )
```

### 2.5 Edge Cases

#### 2.5.1 Embedded Q&A Without "FAQ" Label

**Scenario:** Content contains question-answer pairs but no explicit "FAQ" heading.

**Detection Strategy:**

```python
def detect_embedded_qa(document: DocumentAST) -> EmbeddedQAResult:
    """
    Detect Q&A content that isn't labeled as FAQ.

    This prevents generating redundant content.
    """

    qa_pairs = []

    for section in document.sections:
        # Check for question-pattern headings
        question_headings = [
            h for h in section.headings
            if h.content.strip().endswith("?")
        ]

        if len(question_headings) >= 3:
            # Multiple question headings = implicit FAQ
            qa_pairs.extend(extract_qa_from_headings(section, question_headings))

    if len(qa_pairs) >= 3:
        return EmbeddedQAResult(
            detected=True,
            qa_count=len(qa_pairs),
            recommendation="skip_generation",
            reasoning="Document contains embedded Q&A structure without FAQ label"
        )

    return EmbeddedQAResult(detected=False, qa_count=0)
```

#### 2.5.2 Partial FAQ Sections

**Scenario:** Document has an FAQ section but with only 1-2 questions.

**Detection Strategy:**

```python
def evaluate_faq_completeness(faq_section: FAQSection) -> CompletenessResult:
    """
    Evaluate if existing FAQ is complete or should be enhanced.
    """

    question_count = len(faq_section.qa_pairs)

    if question_count >= 5:
        return CompletenessResult(
            status="complete",
            action="skip",
            reasoning=f"FAQ section has {question_count} questions (sufficient)"
        )
    elif question_count >= 3:
        return CompletenessResult(
            status="adequate",
            action="optional_enhance",
            reasoning=f"FAQ has {question_count} questions - enhancement optional"
        )
    else:
        return CompletenessResult(
            status="incomplete",
            action="enhance",
            reasoning=f"FAQ has only {question_count} questions - enhancement recommended",
            suggested_additions=5 - question_count
        )
```

#### 2.5.3 FAQ in Different Formats

| Format | Detection Method | Confidence |
|--------|------------------|------------|
| Accordion/collapsible | Check for expandable markers in source | 0.85 |
| Tabbed interface | Detect tab structure patterns | 0.80 |
| Sidebar widget | Position-based detection | 0.75 |
| Embedded in body | Q&A pattern matching | 0.70 |
| Schema-only (no visible) | Parse JSON-LD for FAQPage | 0.95 |

### 2.6 Decision Logic Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FAQ DETECTION DECISION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START: Document received                                            │
│     │                                                                │
│     ▼                                                                │
│  ┌─────────────────────────────────────┐                            │
│  │ Scan for FAQ heading patterns       │                            │
│  │ (exact, partial, semantic)          │                            │
│  └──────────────┬──────────────────────┘                            │
│                 │                                                    │
│     ┌───────────┼───────────┐                                       │
│     │           │           │                                       │
│     ▼           ▼           ▼                                       │
│  Found      Uncertain    Not Found                                  │
│  (>0.75)    (0.50-0.74)  (<0.50)                                   │
│     │           │           │                                       │
│     ▼           │           ▼                                       │
│  ┌──────────┐   │    ┌──────────────────────────┐                   │
│  │ Validate │   │    │ Check for embedded Q&A   │                   │
│  │ Q&A      │   │    │ patterns without label   │                   │
│  │ structure│   │    └─────────────┬────────────┘                   │
│  └────┬─────┘   │                  │                                │
│       │         │       ┌──────────┼──────────┐                     │
│       ▼         │       │          │          │                     │
│  ┌──────────┐   │       ▼          ▼          ▼                     │
│  │Confirmed │   │    >=3 Q&As   1-2 Q&As   0 Q&As                   │
│  │ FAQ      │   │       │          │          │                     │
│  │ EXISTS   │   │       ▼          ▼          ▼                     │
│  └────┬─────┘   │    Implicit   Consider    Generate                │
│       │         │    FAQ-skip   enhancement  new FAQ                │
│       │         │       │          │          │                     │
│       ▼         ▼       ▼          ▼          ▼                     │
│    ┌────────────────────────────────────────────────────┐           │
│    │              FINAL DECISION                         │           │
│    │  ┌──────────┬──────────────┬───────────────────┐   │           │
│    │  │  SKIP    │   REVIEW     │     GENERATE      │   │           │
│    │  │          │   REQUIRED   │                   │   │           │
│    │  │ FAQ      │   Human      │ Create new        │   │           │
│    │  │ exists   │   decides    │ FAQ section       │   │           │
│    │  └──────────┴──────────────┴───────────────────┘   │           │
│    └────────────────────────────────────────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Question Generation Pipeline

### 3.1 Pipeline Overview

The question generation pipeline transforms page content, keywords, and business context into a ranked list of candidate questions optimized for both user relevance and SEO value.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 QUESTION GENERATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUTS                                                              │
│  ───────────────────────────────────────────────────────────────    │
│  │ Page Content (DocumentAST)                                    │  │
│  │ Target Keywords (primary + secondary)                         │  │
│  │ Business Context (optional brand docs)                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: TOPIC EXTRACTION                                      │  │
│  │   - Extract main topics from content                          │  │
│  │   - Identify key entities (products, services, concepts)      │  │
│  │   - Map content sections to topic clusters                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: QUESTION TEMPLATE APPLICATION                        │  │
│  │   - Apply "What is [topic]?" templates                        │  │
│  │   - Apply "How to [action]?" templates                        │  │
│  │   - Apply "Why [reason]?" templates                           │  │
│  │   - Apply keyword-specific templates                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ STAGE 3: CANDIDATE SCORING & RANKING                          │  │
│  │   - Score by keyword integration                              │  │
│  │   - Score by content coverage                                 │  │
│  │   - Score by search intent alignment                          │  │
│  │   - Score by answerability from source                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ STAGE 4: DIVERSITY FILTERING                                  │  │
│  │   - Remove near-duplicate questions                           │  │
│  │   - Ensure topic coverage diversity                           │  │
│  │   - Balance question types (what/how/why)                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           ▼                                          │
│  OUTPUT: Ranked List of 5-7 Candidate Questions                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Input Processing

```python
@dataclass
class QuestionGenerationInput:
    """Input specification for question generation."""

    document: DocumentAST
    primary_keyword: str
    secondary_keywords: List[str]
    business_context: Optional[BusinessContext]

    # Constraints
    min_questions: int = 3
    max_questions: int = 7
    target_questions: int = 5

@dataclass
class BusinessContext:
    """Optional business context for branded answers."""

    company_name: str
    industry: str
    products_services: List[str]
    unique_value_propositions: List[str]
    brand_voice: Literal["professional", "friendly", "technical", "casual"]

    # Extracted from brand documents
    terminology_preferences: Dict[str, str]  # e.g., {"customer" -> "client"}
    topics_to_avoid: List[str]
```

### 3.3 Question Template Patterns

#### 3.3.1 Template Categories

**Definitional Templates (What/Who/Which):**

```python
DEFINITIONAL_TEMPLATES = [
    "What is {topic}?",
    "What does {topic} mean?",
    "What are the different types of {topic}?",
    "What is the difference between {topic_a} and {topic_b}?",
    "Who needs {topic}?",
    "Which {topic} is best for {use_case}?",
]

# Priority: HIGH for informational content
# Best for: Concept explanations, product categories, terminology
```

**Procedural Templates (How):**

```python
PROCEDURAL_TEMPLATES = [
    "How does {topic} work?",
    "How do I {action} with {topic}?",
    "How long does {topic} take?",
    "How much does {topic} cost?",
    "How do I get started with {topic}?",
    "How can I improve my {topic}?",
]

# Priority: HIGH for service/product pages
# Best for: Process explanations, tutorials, guides
```

**Reasoning Templates (Why/When):**

```python
REASONING_TEMPLATES = [
    "Why is {topic} important?",
    "Why should I choose {topic}?",
    "When do I need {topic}?",
    "When is the best time to {action}?",
    "Why do experts recommend {topic}?",
]

# Priority: MEDIUM - good for building trust
# Best for: Value propositions, decision support
```

**Comparison Templates:**

```python
COMPARISON_TEMPLATES = [
    "Is {topic} better than {alternative}?",
    "What are the pros and cons of {topic}?",
    "{topic} vs {alternative}: which is right for me?",
    "What are the benefits of {topic} over {alternative}?",
]

# Priority: MEDIUM for competitive content
# Best for: Product comparisons, decision guides
```

**Troubleshooting Templates:**

```python
TROUBLESHOOTING_TEMPLATES = [
    "What if {topic} doesn't work?",
    "How do I fix {problem} with {topic}?",
    "What are common mistakes with {topic}?",
    "Why isn't my {topic} working?",
]

# Priority: LOW (only if content supports)
# Best for: Support pages, technical documentation
```

#### 3.3.2 Template Selection Logic

```python
def select_templates(
    content_analysis: ContentAnalysis,
    keyword_intent: KeywordIntent,
    business_context: Optional[BusinessContext]
) -> List[QuestionTemplate]:
    """
    Select appropriate templates based on content type and keyword intent.
    """

    templates = []

    # Always include definitional for primary topic
    templates.extend(DEFINITIONAL_TEMPLATES[:2])

    # Add procedural if content has how-to elements
    if content_analysis.has_procedural_content:
        templates.extend(PROCEDURAL_TEMPLATES[:3])

    # Add reasoning for commercial/transactional intent
    if keyword_intent in [KeywordIntent.COMMERCIAL, KeywordIntent.TRANSACTIONAL]:
        templates.extend(REASONING_TEMPLATES[:2])

    # Add comparison if competitors mentioned
    if content_analysis.mentions_alternatives:
        templates.extend(COMPARISON_TEMPLATES[:2])

    # Add troubleshooting for support content
    if content_analysis.content_type == ContentType.SUPPORT:
        templates.extend(TROUBLESHOOTING_TEMPLATES[:2])

    return templates
```

### 3.4 Keyword Integration Rules

#### 3.4.1 Natural Integration Guidelines

**DO:**
- Place keyword at natural question positions (usually end or middle)
- Use keyword variations and synonyms
- Integrate long-tail keywords completely
- Allow keyword to be the question subject

**DON'T:**
- Force keyword at question start if unnatural
- Repeat keyword multiple times in one question
- Use exact match when variation sounds better
- Sacrifice readability for keyword inclusion

**Integration Patterns:**

```python
KEYWORD_INTEGRATION_PATTERNS = {
    # Pattern: keyword as subject
    "subject": [
        "What is {keyword}?",
        "How does {keyword} work?",
        "Why is {keyword} important?",
    ],

    # Pattern: keyword as object
    "object": [
        "How do I choose the right {keyword}?",
        "What factors affect {keyword}?",
        "When should I consider {keyword}?",
    ],

    # Pattern: keyword in prepositional phrase
    "prepositional": [
        "What are the benefits of {keyword}?",
        "How do I get started with {keyword}?",
        "What should I know about {keyword}?",
    ],

    # Pattern: keyword as modifier
    "modifier": [
        "What makes a good {keyword} strategy?",
        "How do {keyword} services differ?",
        "What are common {keyword} mistakes?",
    ],
}
```

#### 3.4.2 Keyword Distribution Requirements

```python
def validate_keyword_distribution(
    questions: List[str],
    primary_keyword: str,
    secondary_keywords: List[str]
) -> KeywordDistributionResult:
    """
    Validate keyword presence across generated questions.

    Requirements:
    - Primary keyword: Must appear in at least 1 question
    - Secondary keywords: At least 1 should appear (if available)
    - Total keyword coverage: 40-70% of questions contain a keyword
    """

    primary_count = sum(
        1 for q in questions
        if keyword_in_question(primary_keyword, q)
    )

    secondary_count = sum(
        1 for q in questions
        if any(keyword_in_question(kw, q) for kw in secondary_keywords)
    )

    total_with_keywords = primary_count + secondary_count
    coverage_ratio = total_with_keywords / len(questions)

    # Validation
    if primary_count == 0:
        return KeywordDistributionResult(
            valid=False,
            error="Primary keyword not present in any question",
            action="regenerate"
        )

    if coverage_ratio < 0.4:
        return KeywordDistributionResult(
            valid=False,
            error=f"Keyword coverage too low ({coverage_ratio:.0%})",
            action="add_keyword_questions"
        )

    if coverage_ratio > 0.7:
        return KeywordDistributionResult(
            valid=False,
            error=f"Keyword coverage too high ({coverage_ratio:.0%}) - appears stuffed",
            action="reduce_keyword_questions"
        )

    return KeywordDistributionResult(valid=True, coverage=coverage_ratio)


def keyword_in_question(keyword: str, question: str) -> bool:
    """
    Check if keyword or variation appears naturally in question.

    Uses fuzzy matching to catch variations:
    - Singular/plural
    - Word order changes (for multi-word keywords)
    - Common synonyms
    """
    question_lower = question.lower()
    keyword_lower = keyword.lower()

    # Exact match
    if keyword_lower in question_lower:
        return True

    # Word-by-word match for multi-word keywords
    keyword_words = set(keyword_lower.split())
    question_words = set(question_lower.split())

    if len(keyword_words) > 1:
        # Allow 80% word overlap for multi-word keywords
        overlap = len(keyword_words.intersection(question_words))
        if overlap / len(keyword_words) >= 0.8:
            return True

    return False
```

### 3.5 Prioritization Logic

#### 3.5.1 Question Scoring Formula

```
QUESTION_SCORE = (
    (answerability_score * 0.35) +
    (keyword_relevance_score * 0.25) +
    (search_intent_score * 0.20) +
    (uniqueness_score * 0.10) +
    (user_value_score * 0.10)
)

WHERE:
    answerability_score = Can this question be answered from source content? (0-1)
    keyword_relevance_score = Does question incorporate target keywords naturally? (0-1)
    search_intent_score = Does question match likely search queries? (0-1)
    uniqueness_score = Is this question distinct from others? (0-1)
    user_value_score = Does answering this provide real value? (0-1)
```

#### 3.5.2 Scoring Implementation

```python
@dataclass
class QuestionCandidate:
    question: str
    template_source: str
    topic: str
    keyword_used: Optional[str]

    # Scores
    answerability_score: float = 0.0
    keyword_relevance_score: float = 0.0
    search_intent_score: float = 0.0
    uniqueness_score: float = 0.0
    user_value_score: float = 0.0

    @property
    def total_score(self) -> float:
        return (
            self.answerability_score * 0.35 +
            self.keyword_relevance_score * 0.25 +
            self.search_intent_score * 0.20 +
            self.uniqueness_score * 0.10 +
            self.user_value_score * 0.10
        )


def score_answerability(
    question: str,
    document: DocumentAST,
    similarity_threshold: float = 0.7
) -> float:
    """
    Score how well the question can be answered from document content.

    Uses semantic similarity between question and content passages.
    """

    # Generate question embedding
    question_embedding = embed(question)

    # Find most relevant content passages
    max_similarity = 0.0

    for passage in document.get_passages(min_length=50, max_length=500):
        passage_embedding = embed(passage.text)
        similarity = cosine_similarity(question_embedding, passage_embedding)
        max_similarity = max(max_similarity, similarity)

    # Score based on similarity
    if max_similarity >= 0.85:
        return 1.0  # Highly answerable
    elif max_similarity >= 0.70:
        return 0.8
    elif max_similarity >= 0.55:
        return 0.5
    elif max_similarity >= 0.40:
        return 0.3
    else:
        return 0.0  # Cannot answer from content


def score_search_intent(
    question: str,
    keyword_intent: KeywordIntent
) -> float:
    """
    Score alignment between question type and keyword search intent.
    """

    # Determine question type
    question_type = classify_question_type(question)

    # Intent alignment matrix
    alignment_matrix = {
        KeywordIntent.INFORMATIONAL: {
            QuestionType.WHAT: 1.0,
            QuestionType.HOW: 0.9,
            QuestionType.WHY: 0.8,
            QuestionType.COMPARISON: 0.6,
        },
        KeywordIntent.COMMERCIAL: {
            QuestionType.COMPARISON: 1.0,
            QuestionType.WHY: 0.9,
            QuestionType.HOW: 0.8,
            QuestionType.WHAT: 0.7,
        },
        KeywordIntent.TRANSACTIONAL: {
            QuestionType.HOW: 1.0,
            QuestionType.COST: 1.0,
            QuestionType.COMPARISON: 0.8,
            QuestionType.WHAT: 0.6,
        },
        KeywordIntent.NAVIGATIONAL: {
            QuestionType.WHAT: 0.8,
            QuestionType.HOW: 0.7,
            QuestionType.WHY: 0.5,
            QuestionType.COMPARISON: 0.4,
        },
    }

    return alignment_matrix.get(keyword_intent, {}).get(question_type, 0.5)
```

### 3.6 Output Specification

```python
@dataclass
class QuestionGenerationOutput:
    """Final output of question generation pipeline."""

    questions: List[RankedQuestion]
    generation_metadata: GenerationMetadata
    quality_report: QualityReport

    def get_top_questions(self, n: int = 5) -> List[RankedQuestion]:
        """Return top N questions by score."""
        return sorted(self.questions, key=lambda q: q.total_score, reverse=True)[:n]


@dataclass
class RankedQuestion:
    question: str
    rank: int
    total_score: float

    # Score breakdown
    scores: Dict[str, float]

    # Metadata
    keyword_used: Optional[str]
    topic: str
    question_type: QuestionType
    template_source: str

    # For answer generation
    relevant_passages: List[str]  # Source content for answering


@dataclass
class GenerationMetadata:
    total_candidates_generated: int
    candidates_after_filtering: int
    keyword_coverage: float
    topic_coverage: float
    question_type_distribution: Dict[QuestionType, int]
```

---

## 4. Answer Generation Constraints

### 4.1 Core Principle: Source-Grounded Answers Only

**CRITICAL REQUIREMENT:** All generated answers must be derivable from the source document content. The system must NOT:
- Invent facts, statistics, or claims
- Add information not present in source
- Make assumptions about the business or product
- Include generic filler content

**Hallucination Prevention Strategy:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HALLUCINATION PREVENTION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ANSWER GENERATION MODES:                                            │
│                                                                      │
│  1. EXTRACTIVE (Preferred)                                          │
│     - Direct extraction from source passages                         │
│     - Minor rephrasing for readability                              │
│     - Sentence combination from multiple passages                    │
│     - Hallucination risk: VERY LOW                                  │
│                                                                      │
│  2. ABSTRACTIVE (With Guardrails)                                   │
│     - Synthesis of information from source                          │
│     - All claims must be traceable to source                        │
│     - No new facts introduced                                       │
│     - Hallucination risk: LOW-MEDIUM (requires validation)          │
│                                                                      │
│  3. TEMPLATED (For Common Patterns)                                 │
│     - Pre-defined answer structures                                  │
│     - Slots filled with extracted content                           │
│     - Predictable, verifiable output                                │
│     - Hallucination risk: VERY LOW                                  │
│                                                                      │
│  PROHIBITED:                                                         │
│     - Pure generation without source grounding                       │
│     - Adding statistics not in source                               │
│     - Making claims about capabilities/features not stated          │
│     - Inventing examples or case studies                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Answer Generation Pipeline

```python
def generate_answer(
    question: RankedQuestion,
    document: DocumentAST,
    business_context: Optional[BusinessContext],
    mode: AnswerMode = AnswerMode.EXTRACTIVE
) -> AnswerGenerationResult:
    """
    Generate source-grounded answer for a question.

    Steps:
    1. Retrieve relevant passages from source
    2. Validate passage relevance
    3. Generate answer from passages
    4. Verify answer grounding
    5. Apply formatting rules
    """

    # Step 1: Retrieve relevant passages
    passages = retrieve_relevant_passages(
        question=question.question,
        document=document,
        top_k=5,
        min_similarity=0.6
    )

    if not passages:
        return AnswerGenerationResult(
            success=False,
            error="No relevant content found to answer this question",
            recommendation="skip_question"
        )

    # Step 2: Validate passage relevance
    validated_passages = []
    for passage in passages:
        if validate_passage_relevance(question.question, passage) > 0.7:
            validated_passages.append(passage)

    if len(validated_passages) < 1:
        return AnswerGenerationResult(
            success=False,
            error="Retrieved passages insufficient to answer question",
            recommendation="skip_question"
        )

    # Step 3: Generate answer
    if mode == AnswerMode.EXTRACTIVE:
        answer = generate_extractive_answer(question, validated_passages)
    elif mode == AnswerMode.ABSTRACTIVE:
        answer = generate_abstractive_answer(question, validated_passages)
    elif mode == AnswerMode.TEMPLATED:
        answer = generate_templated_answer(question, validated_passages)

    # Step 4: Verify grounding
    grounding_result = verify_answer_grounding(answer, validated_passages)

    if not grounding_result.is_grounded:
        return AnswerGenerationResult(
            success=False,
            error="Generated answer contains ungrounded claims",
            ungrounded_claims=grounding_result.ungrounded_claims,
            recommendation="regenerate_with_constraints"
        )

    # Step 5: Apply formatting
    formatted_answer = apply_answer_formatting(
        answer,
        business_context=business_context
    )

    return AnswerGenerationResult(
        success=True,
        answer=formatted_answer,
        source_passages=validated_passages,
        grounding_confidence=grounding_result.confidence
    )
```

### 4.3 Maximum Length Guidelines

| Question Type | Optimal Length | Minimum | Maximum | Rationale |
|---------------|----------------|---------|---------|-----------|
| Definitional (What is) | 2-3 sentences | 40 words | 100 words | Concise definition, featured snippet optimized |
| Procedural (How to) | 3-4 sentences | 50 words | 150 words | Clear steps, may include brief list |
| Reasoning (Why) | 2-3 sentences | 40 words | 100 words | Direct reasoning, avoid fluff |
| Comparison | 3-4 sentences | 50 words | 120 words | Clear differentiation |
| Cost/Time | 1-2 sentences | 20 words | 60 words | Direct answer, specifics if available |

**Length Enforcement:**

```python
def enforce_answer_length(
    answer: str,
    question_type: QuestionType
) -> str:
    """
    Enforce length guidelines for answer.

    Truncates long answers while preserving meaning.
    Flags short answers for potential enhancement.
    """

    length_rules = {
        QuestionType.DEFINITIONAL: {"min": 40, "max": 100, "optimal": 60},
        QuestionType.PROCEDURAL: {"min": 50, "max": 150, "optimal": 80},
        QuestionType.REASONING: {"min": 40, "max": 100, "optimal": 60},
        QuestionType.COMPARISON: {"min": 50, "max": 120, "optimal": 80},
        QuestionType.COST_TIME: {"min": 20, "max": 60, "optimal": 40},
    }

    rules = length_rules.get(question_type, {"min": 40, "max": 100, "optimal": 60})
    word_count = len(answer.split())

    if word_count < rules["min"]:
        # Flag as potentially too short
        log_warning(f"Answer may be too brief ({word_count} words)")
        return answer  # Return as-is, let quality gate catch it

    if word_count > rules["max"]:
        # Truncate intelligently
        sentences = split_into_sentences(answer)
        truncated = []
        current_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_count + sentence_words <= rules["max"]:
                truncated.append(sentence)
                current_count += sentence_words
            else:
                break

        return " ".join(truncated)

    return answer
```

### 4.4 Format Requirements

#### 4.4.1 Sentence Structure Rules

**DO:**
- Use complete sentences with subject-verb-object structure
- Start with the answer, not the question rephrased
- Use active voice when possible
- Include specific details from source content

**DON'T:**
- Start with "Great question!" or similar filler
- Use first person ("We", "Our") unless brand context specifies
- Include hedging language ("might", "possibly", "perhaps")
- End with questions or CTAs within the answer

**Example Transformations:**

```
BAD: "That's a great question! SEO might possibly help your website
     rank better in search results, we think."

GOOD: "SEO (Search Engine Optimization) improves website visibility
      in search engine results by optimizing content, structure, and
      technical elements for target keywords."
```

#### 4.4.2 Jargon Handling

```python
def check_jargon_level(
    answer: str,
    target_audience: AudienceLevel = AudienceLevel.GENERAL
) -> JargonCheckResult:
    """
    Check if answer contains appropriate jargon level.

    For general audiences, flag technical terms without explanation.
    """

    # Load industry jargon dictionary
    jargon_terms = load_jargon_dictionary()

    found_jargon = []
    for term in jargon_terms:
        if term.lower() in answer.lower():
            # Check if term is explained
            explanation_patterns = [
                f"{term}, which means",
                f"{term} (also known as",
                f"also called {term}",
                f"{term} refers to",
            ]

            has_explanation = any(
                pattern.lower() in answer.lower()
                for pattern in explanation_patterns
            )

            if not has_explanation and target_audience == AudienceLevel.GENERAL:
                found_jargon.append(JargonInstance(
                    term=term,
                    needs_explanation=True
                ))

    return JargonCheckResult(
        jargon_found=found_jargon,
        needs_simplification=len(found_jargon) > 2
    )
```

### 4.5 Brand Voice Alignment

```python
def apply_brand_voice(
    answer: str,
    business_context: BusinessContext
) -> str:
    """
    Adjust answer to match brand voice guidelines.
    """

    if business_context.brand_voice == "professional":
        # Formal language, avoid contractions
        answer = expand_contractions(answer)
        answer = remove_casual_phrases(answer)

    elif business_context.brand_voice == "friendly":
        # Allow contractions, warmer language
        answer = add_conversational_elements(answer)

    elif business_context.brand_voice == "technical":
        # Keep technical terms, precise language
        # No simplification of jargon
        pass

    # Apply terminology preferences
    for original, preferred in business_context.terminology_preferences.items():
        answer = answer.replace(original, preferred)

    return answer
```

### 4.6 Grounding Verification

```python
def verify_answer_grounding(
    answer: str,
    source_passages: List[Passage]
) -> GroundingVerificationResult:
    """
    Verify that all claims in the answer are grounded in source content.

    Uses semantic similarity and claim extraction.
    """

    # Extract claims from answer
    claims = extract_claims(answer)

    ungrounded_claims = []
    grounding_scores = []

    for claim in claims:
        # Find best matching passage
        claim_embedding = embed(claim.text)

        max_similarity = 0.0
        best_passage = None

        for passage in source_passages:
            passage_embedding = embed(passage.text)
            similarity = cosine_similarity(claim_embedding, passage_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_passage = passage

        grounding_scores.append(max_similarity)

        # Threshold for grounding
        if max_similarity < 0.75:
            ungrounded_claims.append(UngroundedClaim(
                claim=claim,
                max_similarity=max_similarity,
                nearest_passage=best_passage
            ))

    overall_confidence = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0

    return GroundingVerificationResult(
        is_grounded=len(ungrounded_claims) == 0,
        confidence=overall_confidence,
        ungrounded_claims=ungrounded_claims
    )
```

---

## 5. FAQ Structure Specification

### 5.1 Markdown Structure

```markdown
## Frequently Asked Questions

### What is [primary keyword/topic]?

[Answer derived from content, 2-4 sentences. Provides clear definition
or explanation grounded in source material. Approximately 60-80 words.]

### How does [product/service/process] work?

[Procedural answer explaining the process or mechanism. May include
a brief enumeration if appropriate. 80-100 words.]

### Why should I consider [topic/solution]?

[Reasoning-based answer highlighting benefits or importance. Avoids
marketing fluff, focuses on factual benefits from source. 60-80 words.]

### What are the benefits of [topic]?

[Value-focused answer listing key benefits. Can use inline list format
if 3+ benefits. 60-100 words.]

### How do I get started with [topic]?

[Action-oriented answer providing initial steps. Specific and
actionable based on source content. 60-80 words.]
```

### 5.2 Heading Levels

| Element | Heading Level | Rationale |
|---------|---------------|-----------|
| FAQ Section Title | H2 | Main section within document |
| Individual Questions | H3 | Sub-sections under FAQ |
| Sub-questions (rare) | H4 | Only if answer requires structure |

**Validation Rule:**

```python
def validate_faq_heading_structure(faq_section: str) -> ValidationResult:
    """
    Validate that FAQ follows correct heading hierarchy.
    """

    lines = faq_section.split('\n')
    errors = []

    # First heading should be H2 "Frequently Asked Questions"
    first_heading = find_first_heading(lines)
    if first_heading:
        if not first_heading.startswith('## '):
            errors.append("FAQ section should start with H2")
        if 'faq' not in first_heading.lower() and 'question' not in first_heading.lower():
            errors.append("FAQ heading should contain 'FAQ' or 'Questions'")

    # All questions should be H3
    for line in lines:
        if line.endswith('?'):
            if line.startswith('## '):
                errors.append(f"Question should be H3, not H2: {line}")
            elif line.startswith('# '):
                errors.append(f"Question should be H3, not H1: {line}")
            elif not line.startswith('### '):
                errors.append(f"Question should be H3 heading: {line}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors
    )
```

### 5.3 Question Formatting

**Requirements:**
- Questions must end with a question mark
- Questions should be complete sentences
- Questions should be concise (typically 8-15 words)
- Keywords should appear naturally (not forced)

**Formatting Rules:**

```python
def format_question(question: str) -> str:
    """
    Apply formatting rules to question.
    """

    # Ensure proper capitalization
    question = question.strip()
    if question:
        question = question[0].upper() + question[1:]

    # Ensure question mark
    if not question.endswith('?'):
        question += '?'

    # Remove double question marks
    question = question.replace('??', '?')

    # Remove leading "Q:" or similar
    question = re.sub(r'^(Q:|Question:)\s*', '', question)

    return question
```

### 5.4 Answer Formatting

**Requirements:**
- Complete sentences only
- No bullet points within answer (save for exceptional cases)
- No links within FAQ answers (unless critical)
- Consistent voice/tense throughout

**Paragraph Structure:**

```python
def format_answer(answer: str) -> str:
    """
    Apply formatting rules to answer.
    """

    # Ensure proper paragraph structure
    answer = answer.strip()

    # Remove leading "A:" or similar
    answer = re.sub(r'^(A:|Answer:)\s*', '', answer)

    # Capitalize first letter
    if answer:
        answer = answer[0].upper() + answer[1:]

    # Ensure ends with period (or appropriate punctuation)
    if answer and answer[-1] not in '.!':
        answer += '.'

    # Clean up whitespace
    answer = ' '.join(answer.split())

    return answer
```

### 5.5 Schema Markup Compatibility

The FAQ structure must be compatible with FAQPage schema for rich results.

**FAQPage Schema Requirements:**

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "[Question text - exact match to H3 heading]",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "[Answer text - plain text, no HTML formatting]"
      }
    }
  ]
}
```

**Schema Generation Function:**

```python
def generate_faq_schema(faq_section: FAQSection) -> dict:
    """
    Generate FAQPage schema markup from FAQ section.
    """

    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": []
    }

    for qa_pair in faq_section.qa_pairs:
        question_schema = {
            "@type": "Question",
            "name": qa_pair.question.strip(),
            "acceptedAnswer": {
                "@type": "Answer",
                "text": strip_markdown(qa_pair.answer.strip())
            }
        }
        schema["mainEntity"].append(question_schema)

    return schema


def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting for schema text field.

    Schema.org Answer text should be plain text.
    """

    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)

    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    return text.strip()
```

### 5.6 Complete FAQ Template

```markdown
## Frequently Asked Questions

### What is [primary topic/keyword]?

[Definitional answer: 2-3 sentences providing a clear explanation of the
topic. Incorporates primary keyword naturally. Derived entirely from
source content with no invented claims. Target: 60-80 words.]

### How does [topic/product/service] work?

[Procedural answer: 3-4 sentences explaining the mechanism or process.
Specific to the content, avoiding generic descriptions. May reference
specific features or steps mentioned in source. Target: 80-100 words.]

### What are the benefits of [topic]?

[Value-focused answer: 2-3 sentences highlighting key benefits. Each
benefit should be traceable to source content. Avoids hyperbole and
unsupported claims. Target: 60-80 words.]

### Who should consider [topic/solution]?

[Audience-focused answer: 2-3 sentences describing ideal use cases or
target audience. Based on information in source content about
applicability. Target: 50-70 words.]

### How do I get started with [topic]?

[Action-oriented answer: 2-3 sentences providing practical first steps.
Must be grounded in source content's guidance or process descriptions.
Target: 60-80 words.]
```

---

## 6. Quality Gates

### 6.1 Quality Gate Summary Table

| Check | Metric | Threshold | Action if Failed |
|-------|--------|-----------|------------------|
| Question Relevance | Semantic similarity to content | > 0.70 | Skip question |
| Answer Grounding | Claims traceable to source | 100% | Skip or flag for review |
| Keyword Coverage | Primary keyword in FAQ | >= 1 occurrence | Regenerate |
| FAQ Length | Number of Q&A pairs | 3-7 pairs | Trim or expand |
| Answer Length | Words per answer | 40-150 words | Truncate or expand |
| Question Uniqueness | Similarity between questions | < 0.85 | Deduplicate |
| Schema Compatibility | Valid FAQPage structure | Pass validation | Fix structure |
| Readability | Flesch-Kincaid grade | 8-12 grade level | Simplify if needed |

### 6.2 Question Relevance Gate

```python
def gate_question_relevance(
    question: str,
    document: DocumentAST,
    threshold: float = 0.70
) -> GateResult:
    """
    Verify question is relevant to document content.

    Uses semantic similarity between question and document.
    """

    # Embed question
    question_embedding = embed(question)

    # Get document summary embedding
    document_summary = document.get_summary(max_words=500)
    document_embedding = embed(document_summary)

    # Calculate similarity
    similarity = cosine_similarity(question_embedding, document_embedding)

    if similarity >= threshold:
        return GateResult(
            passed=True,
            score=similarity,
            message=f"Question relevance: {similarity:.2f} (threshold: {threshold})"
        )
    else:
        return GateResult(
            passed=False,
            score=similarity,
            message=f"Question not relevant enough: {similarity:.2f} < {threshold}",
            action="skip_question"
        )
```

### 6.3 Answer Grounding Gate

```python
def gate_answer_grounding(
    answer: str,
    source_passages: List[Passage],
    threshold: float = 1.0  # 100% grounding required
) -> GateResult:
    """
    Verify all claims in answer are grounded in source.

    CRITICAL: This is the primary hallucination prevention gate.
    """

    verification = verify_answer_grounding(answer, source_passages)

    if verification.confidence >= threshold:
        return GateResult(
            passed=True,
            score=verification.confidence,
            message="All claims grounded in source content"
        )
    else:
        return GateResult(
            passed=False,
            score=verification.confidence,
            message=f"Ungrounded claims detected: {len(verification.ungrounded_claims)}",
            action="flag_for_review" if verification.confidence > 0.8 else "reject",
            details=verification.ungrounded_claims
        )
```

### 6.4 Keyword Coverage Gate

```python
def gate_keyword_coverage(
    faq_section: FAQSection,
    primary_keyword: str,
    secondary_keywords: List[str],
    min_primary_occurrences: int = 1
) -> GateResult:
    """
    Verify adequate keyword presence in FAQ.
    """

    faq_text = faq_section.to_text()

    # Check primary keyword
    primary_count = count_keyword_occurrences(primary_keyword, faq_text)

    if primary_count < min_primary_occurrences:
        return GateResult(
            passed=False,
            score=0,
            message=f"Primary keyword '{primary_keyword}' not found in FAQ",
            action="regenerate"
        )

    # Check secondary keywords (at least one should appear)
    secondary_found = any(
        keyword.lower() in faq_text.lower()
        for keyword in secondary_keywords
    )

    # Calculate coverage score
    coverage_score = primary_count / (len(faq_section.qa_pairs) * 2)  # Normalize

    return GateResult(
        passed=True,
        score=min(coverage_score, 1.0),
        message=f"Keyword coverage: primary={primary_count}, secondary={'found' if secondary_found else 'not found'}"
    )
```

### 6.5 FAQ Length Gate

```python
def gate_faq_length(
    faq_section: FAQSection,
    min_pairs: int = 3,
    max_pairs: int = 7
) -> GateResult:
    """
    Verify FAQ has appropriate number of Q&A pairs.
    """

    count = len(faq_section.qa_pairs)

    if count < min_pairs:
        return GateResult(
            passed=False,
            score=count / min_pairs,
            message=f"FAQ too short: {count} pairs (minimum: {min_pairs})",
            action="expand"
        )
    elif count > max_pairs:
        return GateResult(
            passed=False,
            score=max_pairs / count,
            message=f"FAQ too long: {count} pairs (maximum: {max_pairs})",
            action="trim"
        )
    else:
        return GateResult(
            passed=True,
            score=1.0,
            message=f"FAQ length appropriate: {count} pairs"
        )
```

### 6.6 Question Uniqueness Gate

```python
def gate_question_uniqueness(
    questions: List[str],
    similarity_threshold: float = 0.85
) -> GateResult:
    """
    Verify questions are sufficiently distinct from each other.
    """

    duplicates = []

    for i, q1 in enumerate(questions):
        for j, q2 in enumerate(questions[i+1:], start=i+1):
            similarity = calculate_semantic_similarity(q1, q2)

            if similarity >= similarity_threshold:
                duplicates.append((i, j, similarity))

    if duplicates:
        return GateResult(
            passed=False,
            score=1 - (len(duplicates) / len(questions)),
            message=f"Found {len(duplicates)} near-duplicate question pairs",
            action="deduplicate",
            details=duplicates
        )
    else:
        return GateResult(
            passed=True,
            score=1.0,
            message="All questions sufficiently unique"
        )
```

### 6.7 Human Review Triggers

**Automatic Human Review Required When:**

```python
HUMAN_REVIEW_TRIGGERS = {
    # Content-based triggers
    "low_grounding_confidence": {
        "condition": lambda result: result.grounding_confidence < 0.85,
        "reason": "Some claims may not be fully grounded in source"
    },

    "ymyl_content": {
        "condition": lambda doc: doc.is_ymyl_content(),
        "reason": "YMYL content requires human verification"
    },

    "technical_claims": {
        "condition": lambda answer: contains_technical_claims(answer),
        "reason": "Technical claims require expert verification"
    },

    # Statistical triggers
    "numerical_data": {
        "condition": lambda answer: contains_numbers(answer),
        "reason": "Numerical data should be verified"
    },

    # Quality triggers
    "low_question_relevance": {
        "condition": lambda scores: any(s < 0.75 for s in scores),
        "reason": "One or more questions have low relevance scores"
    },

    "uncertain_faq_detection": {
        "condition": lambda detection: detection.decision == "review",
        "reason": "Uncertain whether FAQ already exists"
    },
}


def check_human_review_triggers(
    generation_result: FAQGenerationResult
) -> List[ReviewTrigger]:
    """
    Check if any conditions require human review.
    """

    triggers = []

    for trigger_name, trigger_config in HUMAN_REVIEW_TRIGGERS.items():
        if trigger_config["condition"](generation_result):
            triggers.append(ReviewTrigger(
                name=trigger_name,
                reason=trigger_config["reason"]
            ))

    return triggers
```

### 6.8 Complete Quality Gate Pipeline

```python
def run_quality_gates(
    faq_section: FAQSection,
    document: DocumentAST,
    keywords: KeywordSet,
    source_passages: List[Passage]
) -> QualityGateReport:
    """
    Run all quality gates and produce comprehensive report.
    """

    report = QualityGateReport()

    # Gate 1: Question relevance (per question)
    for qa in faq_section.qa_pairs:
        result = gate_question_relevance(qa.question, document)
        report.add_result("question_relevance", qa.question, result)

        if not result.passed:
            qa.mark_for_removal()

    # Gate 2: Answer grounding (per answer)
    for qa in faq_section.qa_pairs:
        result = gate_answer_grounding(qa.answer, source_passages)
        report.add_result("answer_grounding", qa.question, result)

        if not result.passed:
            if result.action == "reject":
                qa.mark_for_removal()
            else:
                qa.mark_for_review()

    # Gate 3: Keyword coverage (section level)
    result = gate_keyword_coverage(
        faq_section,
        keywords.primary,
        keywords.secondary
    )
    report.add_result("keyword_coverage", "section", result)

    # Gate 4: FAQ length
    result = gate_faq_length(faq_section)
    report.add_result("faq_length", "section", result)

    # Gate 5: Question uniqueness
    questions = [qa.question for qa in faq_section.qa_pairs]
    result = gate_question_uniqueness(questions)
    report.add_result("question_uniqueness", "section", result)

    # Gate 6: Schema compatibility
    schema = generate_faq_schema(faq_section)
    result = validate_faq_schema(schema)
    report.add_result("schema_compatibility", "section", result)

    # Check human review triggers
    triggers = check_human_review_triggers(faq_section)
    report.human_review_required = len(triggers) > 0
    report.review_triggers = triggers

    # Calculate overall pass/fail
    report.calculate_overall_status()

    return report
```

---

## 7. LLM vs Rule-Based Decision Framework

### 7.1 Decision Matrix

| Task | Recommended Approach | Rationale |
|------|---------------------|-----------|
| FAQ heading detection | Rule-based | Pattern matching is deterministic and fast |
| Q&A structure detection | Rule-based | Structural patterns are predictable |
| Question template selection | Rule-based | Templates provide consistency |
| Question generation from templates | Hybrid | Templates + LLM for natural phrasing |
| Question ranking/scoring | LLM | Semantic understanding required |
| Passage retrieval | Hybrid | Embedding similarity + keyword matching |
| Answer generation | LLM with guardrails | Synthesis requires language understanding |
| Grounding verification | LLM | Semantic comparison required |
| Final quality assessment | Hybrid | Rules + LLM judgment |

### 7.2 When LLM is Beneficial

**LLM adds value when:**

1. **Semantic understanding required:**
   - Determining if question is answerable from content
   - Ranking question relevance
   - Verifying claim grounding

2. **Natural language generation:**
   - Synthesizing answers from multiple passages
   - Rephrasing for readability
   - Adapting to brand voice

3. **Nuanced judgment:**
   - Assessing question quality
   - Detecting subtle duplicates
   - Evaluating answer completeness

**LLM Configuration:**

```python
LLM_CONFIG = {
    "model": "gpt-4o-mini",  # Cost-effective for FAQ generation
    "temperature": 0.3,      # Low temperature for consistency
    "max_tokens": 500,       # Sufficient for FAQ answers
    "timeout": 30,           # Seconds

    # Fallback model for high-stakes tasks
    "fallback_model": "gpt-4o",
    "fallback_triggers": [
        "ymyl_content",
        "technical_claims",
        "low_confidence"
    ]
}
```

### 7.3 When Rule-Based is Sufficient

**Rule-based is preferred when:**

1. **Deterministic patterns:**
   - Detecting FAQ headings
   - Identifying Q&A structure
   - Validating schema format

2. **Performance critical:**
   - High-volume processing
   - Real-time validation
   - Cost-sensitive operations

3. **Predictability required:**
   - Length enforcement
   - Format validation
   - Keyword counting

### 7.4 Hybrid Approach Specification

```python
class FAQGenerationPipeline:
    """
    Hybrid FAQ generation combining rules and LLM.
    """

    def __init__(self, config: FAQConfig):
        self.config = config
        self.llm_client = LLMClient(config.llm_config)
        self.rule_engine = RuleEngine(config.rules)

    def generate(
        self,
        document: DocumentAST,
        keywords: KeywordSet,
        business_context: Optional[BusinessContext]
    ) -> FAQGenerationResult:

        # RULE-BASED: Detect existing FAQ
        detection = self.rule_engine.detect_faq(document)

        if detection.decision == "skip":
            return FAQGenerationResult(
                success=False,
                reason="FAQ already exists",
                detection=detection
            )

        # RULE-BASED: Generate question candidates from templates
        candidates = self.rule_engine.generate_question_candidates(
            document=document,
            keywords=keywords
        )

        # LLM: Score and rank questions
        scored_candidates = self.llm_client.score_questions(
            candidates=candidates,
            document_context=document.get_summary()
        )

        # RULE-BASED: Filter and select top questions
        selected = self.rule_engine.select_questions(
            scored_candidates,
            min_score=0.7,
            max_questions=self.config.max_questions
        )

        # LLM: Generate answers with grounding
        qa_pairs = []
        for question in selected:
            # RULE-BASED: Retrieve relevant passages
            passages = self.rule_engine.retrieve_passages(
                question=question,
                document=document
            )

            # LLM: Generate answer from passages
            answer = self.llm_client.generate_answer(
                question=question,
                passages=passages,
                constraints=self.config.answer_constraints
            )

            # RULE-BASED: Validate answer format
            formatted_answer = self.rule_engine.format_answer(answer)

            # LLM: Verify grounding
            grounding = self.llm_client.verify_grounding(
                answer=formatted_answer,
                passages=passages
            )

            if grounding.is_grounded:
                qa_pairs.append(QAPair(
                    question=question,
                    answer=formatted_answer,
                    grounding_confidence=grounding.confidence
                ))

        # RULE-BASED: Final validation
        faq_section = FAQSection(qa_pairs=qa_pairs)
        validation = self.rule_engine.validate_faq(faq_section)

        return FAQGenerationResult(
            success=validation.passed,
            faq_section=faq_section,
            validation=validation
        )
```

### 7.5 Fallback Handling

```python
class FallbackHandler:
    """
    Handle failures gracefully with appropriate fallbacks.
    """

    def handle_llm_failure(
        self,
        error: Exception,
        task: str,
        context: dict
    ) -> FallbackResult:
        """
        Handle LLM API failures.
        """

        if isinstance(error, RateLimitError):
            # Retry with exponential backoff
            return FallbackResult(
                action="retry",
                delay=calculate_backoff(context.get("retry_count", 0))
            )

        elif isinstance(error, TimeoutError):
            # Fall back to rule-based for this task
            if task == "question_scoring":
                return FallbackResult(
                    action="use_rule_based",
                    fallback_method=self.rule_based_question_scoring
                )
            else:
                return FallbackResult(
                    action="skip_task",
                    message=f"Skipping {task} due to timeout"
                )

        elif isinstance(error, ContentFilterError):
            # Content was flagged - skip this item
            return FallbackResult(
                action="skip_item",
                message="Content flagged by safety filter"
            )

        else:
            # Unknown error - log and skip
            log_error(f"Unknown LLM error: {error}")
            return FallbackResult(
                action="skip_task",
                error=error
            )

    def handle_insufficient_content(
        self,
        document: DocumentAST,
        min_content_length: int = 200
    ) -> FallbackResult:
        """
        Handle documents with insufficient content for FAQ generation.
        """

        if document.word_count < min_content_length:
            return FallbackResult(
                action="skip_generation",
                reason=f"Document too short ({document.word_count} words) for FAQ generation"
            )

        return FallbackResult(action="proceed")

    def handle_no_answerable_questions(
        self,
        candidates: List[QuestionCandidate],
        min_questions: int = 3
    ) -> FallbackResult:
        """
        Handle case where no questions can be answered from content.
        """

        answerable = [c for c in candidates if c.answerability_score > 0.5]

        if len(answerable) < min_questions:
            return FallbackResult(
                action="skip_generation",
                reason=f"Only {len(answerable)} answerable questions found (minimum: {min_questions})",
                partial_result=answerable if answerable else None
            )

        return FallbackResult(action="proceed", candidates=answerable)
```

---

## 8. Worked Examples

### 8.1 Example Page Content

**Source Document: "Cloud Computing Services for Small Businesses"**

```markdown
# Cloud Computing Solutions for Growing Businesses

Acme Cloud provides enterprise-grade cloud infrastructure tailored for
small and medium-sized businesses. Our platform offers scalable computing
resources, secure data storage, and 24/7 technical support.

## Our Services

### Infrastructure as a Service (IaaS)
Our IaaS solution provides virtualized computing resources over the
internet. Businesses can rent servers, storage, and networking on a
pay-as-you-go basis, eliminating the need for physical hardware
investments.

Key features include:
- Scalable compute instances (1-64 vCPUs)
- SSD-based storage with 99.9% uptime guarantee
- Automatic backups and disaster recovery
- Private virtual networks with firewall protection

### Platform as a Service (PaaS)
For development teams, our PaaS offering streamlines application
deployment. The platform supports Python, Node.js, and Java, with
built-in CI/CD pipelines.

## Pricing

Our pricing starts at $49/month for basic cloud hosting, scaling to
$499/month for enterprise plans with dedicated resources. All plans
include SSL certificates and DDoS protection.

## Getting Started

New customers can sign up for a 14-day free trial with full access
to all features. No credit card required. Our onboarding team will
schedule a personalized walkthrough within 24 hours of signup.
```

**Keywords:**
- Primary: "cloud computing services"
- Secondary: ["cloud hosting", "IaaS", "small business cloud"]

### 8.2 Generated FAQ Output

```markdown
## Frequently Asked Questions

### What are cloud computing services?

Cloud computing services provide businesses with access to computing
resources, data storage, and software applications over the internet,
eliminating the need for physical hardware investments. These services
typically operate on a pay-as-you-go model, allowing companies to
scale resources based on their needs.

### How does Infrastructure as a Service (IaaS) work?

Infrastructure as a Service provides virtualized computing resources
including servers, storage, and networking over the internet. Businesses
rent these resources on a pay-as-you-go basis rather than purchasing
physical hardware. The provider handles maintenance, while customers
control their applications and data.

### What are the benefits of cloud hosting for small businesses?

Cloud hosting eliminates the need for expensive hardware investments
while providing enterprise-grade infrastructure. Small businesses
benefit from scalable resources that grow with their needs, 99.9%
uptime guarantees, automatic backups, and professional security
features like firewall protection and DDoS defense.

### How much do cloud computing services cost?

Pricing starts at $49 per month for basic cloud hosting and scales
to $499 per month for enterprise plans with dedicated resources.
All plans include SSL certificates and DDoS protection, with
pay-as-you-go options available for compute and storage resources.

### How do I get started with cloud services?

New customers can sign up for a 14-day free trial with full access
to all features without providing credit card information. After
signup, the onboarding team schedules a personalized walkthrough
within 24 hours to help configure your cloud environment.
```

### 8.3 Quality Scoring Walkthrough

**Question 1: "What are cloud computing services?"**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| Answerability | 0.92 | Clear definition content in source |
| Keyword Relevance | 1.00 | Primary keyword naturally integrated |
| Search Intent | 0.95 | Matches informational query pattern |
| Uniqueness | 1.00 | Distinct from other questions |
| User Value | 0.90 | Fundamental question for topic |
| **Total Score** | **0.94** | Weighted calculation |

**Answer 1 Grounding Check:**

| Claim | Source Passage | Similarity | Status |
|-------|----------------|------------|--------|
| "computing resources, data storage, and software applications" | "scalable computing resources, secure data storage" | 0.91 | GROUNDED |
| "eliminating the need for physical hardware investments" | "eliminating the need for physical hardware investments" | 1.00 | GROUNDED |
| "pay-as-you-go model" | "pay-as-you-go basis" | 0.98 | GROUNDED |
| "scale resources based on their needs" | "Scalable compute instances" | 0.87 | GROUNDED |

**Overall Grounding Confidence:** 0.94 (PASS)

---

**Question 4: "How much do cloud computing services cost?"**

| Metric | Score | Reasoning |
|--------|-------|-----------|
| Answerability | 0.95 | Exact pricing in source |
| Keyword Relevance | 0.85 | Keyword in question |
| Search Intent | 0.90 | Transactional intent match |
| Uniqueness | 1.00 | Only pricing question |
| User Value | 0.95 | High-value practical info |
| **Total Score** | **0.92** | Weighted calculation |

**Answer 4 Grounding Check:**

| Claim | Source Passage | Similarity | Status |
|-------|----------------|------------|--------|
| "$49 per month for basic cloud hosting" | "starts at $49/month for basic cloud hosting" | 0.99 | GROUNDED |
| "$499 per month for enterprise plans" | "scaling to $499/month for enterprise plans" | 0.99 | GROUNDED |
| "SSL certificates and DDoS protection" | "All plans include SSL certificates and DDoS protection" | 1.00 | GROUNDED |

**Overall Grounding Confidence:** 0.99 (PASS)

### 8.4 Quality Gate Results Summary

| Gate | Result | Details |
|------|--------|---------|
| Question Relevance | PASS | All questions > 0.70 similarity |
| Answer Grounding | PASS | All answers 100% grounded |
| Keyword Coverage | PASS | Primary keyword in 3/5 questions |
| FAQ Length | PASS | 5 Q&A pairs (target: 3-7) |
| Question Uniqueness | PASS | No duplicates detected |
| Schema Compatibility | PASS | Valid FAQPage structure |

**Final Status:** APPROVED FOR OUTPUT

**Human Review Required:** NO

### 8.5 Edge Case Example: Insufficient Content

**Source Document (Too Short):**

```markdown
# Our Product

We offer great products. Contact us for more information.
```

**Generation Attempt:**

```python
result = faq_pipeline.generate(document, keywords)

# Result:
FAQGenerationResult(
    success=False,
    reason="Document too short (12 words) for FAQ generation",
    recommendation="Add more content before generating FAQ",
    min_content_required=200
)
```

### 8.6 Edge Case Example: Low Grounding

**Generated Answer (Problematic):**

```
Cloud computing services typically reduce IT costs by 30-40% while
improving uptime to 99.99%.
```

**Grounding Check:**

| Claim | Source Evidence | Similarity | Status |
|-------|-----------------|------------|--------|
| "reduce IT costs by 30-40%" | NOT FOUND | 0.0 | UNGROUNDED |
| "uptime to 99.99%" | "99.9% uptime guarantee" | 0.75 | CLOSE BUT DIFFERENT |

**Result:** REJECTED - Ungrounded claims detected

**Corrected Answer:**

```
Cloud computing services eliminate hardware investment costs and
provide a 99.9% uptime guarantee. The pay-as-you-go model allows
businesses to pay only for resources they use.
```

---

## 9. Implementation Specifications

### 9.1 Module Structure

```
src/generation/
├── __init__.py
├── faq_generator.py          # Main FAQ generation orchestrator
├── detection/
│   ├── __init__.py
│   ├── faq_detector.py       # FAQ existence detection
│   ├── patterns.py           # Heading/structure patterns
│   └── qa_structure.py       # Q&A structure detection
├── questions/
│   ├── __init__.py
│   ├── generator.py          # Question generation pipeline
│   ├── templates.py          # Question template definitions
│   ├── scorer.py             # Question scoring logic
│   └── selector.py           # Question selection/filtering
├── answers/
│   ├── __init__.py
│   ├── generator.py          # Answer generation
│   ├── grounding.py          # Grounding verification
│   └── formatter.py          # Answer formatting
├── quality/
│   ├── __init__.py
│   ├── gates.py              # Quality gate implementations
│   ├── validation.py         # Validation rules
│   └── review_triggers.py    # Human review trigger logic
└── schema/
    ├── __init__.py
    └── faq_schema.py         # FAQPage schema generation
```

### 9.2 Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class QuestionType(str, Enum):
    DEFINITIONAL = "definitional"
    PROCEDURAL = "procedural"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    COST_TIME = "cost_time"
    TROUBLESHOOTING = "troubleshooting"

class FAQDetectionDecision(str, Enum):
    SKIP = "skip"
    REVIEW = "review"
    GENERATE = "generate"

class QAPair(BaseModel):
    question: str
    answer: str
    question_type: QuestionType
    keyword_used: Optional[str] = None
    grounding_confidence: float = Field(ge=0, le=1)
    source_passages: List[str] = []

class FAQSection(BaseModel):
    title: str = "Frequently Asked Questions"
    qa_pairs: List[QAPair]

    def to_markdown(self) -> str:
        lines = [f"## {self.title}", ""]
        for qa in self.qa_pairs:
            lines.extend([
                f"### {qa.question}",
                "",
                qa.answer,
                ""
            ])
        return "\n".join(lines)

    def to_schema(self) -> dict:
        return {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": qa.question,
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": qa.answer
                    }
                }
                for qa in self.qa_pairs
            ]
        }

class FAQDetectionResult(BaseModel):
    exists: Optional[bool]
    confidence: float = Field(ge=0, le=1)
    decision: FAQDetectionDecision
    reasoning: str
    detected_sections: List[dict] = []

class FAQGenerationResult(BaseModel):
    success: bool
    faq_section: Optional[FAQSection] = None
    detection_result: Optional[FAQDetectionResult] = None
    quality_report: Optional[dict] = None
    human_review_required: bool = False
    review_triggers: List[str] = []
    error: Optional[str] = None
```

### 9.3 Configuration

```python
from pydantic import BaseModel
from typing import Dict, Any

class FAQGenerationConfig(BaseModel):
    # Detection settings
    detection_confidence_threshold: float = 0.75
    uncertain_confidence_threshold: float = 0.50

    # Question generation settings
    min_questions: int = 3
    max_questions: int = 7
    target_questions: int = 5
    min_question_relevance: float = 0.70

    # Answer generation settings
    min_answer_words: int = 40
    max_answer_words: int = 150
    grounding_threshold: float = 1.0  # 100% for production

    # Keyword settings
    min_keyword_coverage: float = 0.40
    max_keyword_coverage: float = 0.70

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_timeout: int = 30

    # Quality settings
    enable_human_review_triggers: bool = True
    strict_mode: bool = True  # Reject vs flag for uncertain cases

# Default configuration
DEFAULT_CONFIG = FAQGenerationConfig()

# Strict configuration for YMYL content
YMYL_CONFIG = FAQGenerationConfig(
    grounding_threshold=1.0,
    min_question_relevance=0.80,
    strict_mode=True,
    enable_human_review_triggers=True
)
```

### 9.4 Integration Points

```python
class FAQGeneratorIntegration:
    """
    Integration with main content optimization pipeline.
    """

    def __init__(self, config: FAQGenerationConfig = DEFAULT_CONFIG):
        self.config = config
        self.detector = FAQDetector(config)
        self.generator = FAQQuestionGenerator(config)
        self.answer_gen = FAQAnswerGenerator(config)
        self.quality = FAQQualityGates(config)

    async def process(
        self,
        document: DocumentAST,
        keywords: KeywordSet,
        business_context: Optional[BusinessContext] = None
    ) -> FAQGenerationResult:
        """
        Main entry point for FAQ generation.

        Called by: OptimizationPipeline.generate_faq()
        """

        # Step 1: Detection
        detection = self.detector.detect(document)

        if detection.decision == FAQDetectionDecision.SKIP:
            return FAQGenerationResult(
                success=False,
                detection_result=detection,
                error="FAQ section already exists"
            )

        if detection.decision == FAQDetectionDecision.REVIEW:
            return FAQGenerationResult(
                success=False,
                detection_result=detection,
                human_review_required=True,
                review_triggers=["uncertain_faq_detection"],
                error="Uncertain if FAQ exists - review required"
            )

        # Step 2: Generate questions
        questions = await self.generator.generate(
            document=document,
            keywords=keywords,
            business_context=business_context
        )

        # Step 3: Generate answers
        qa_pairs = []
        for question in questions:
            answer_result = await self.answer_gen.generate(
                question=question,
                document=document,
                business_context=business_context
            )

            if answer_result.success:
                qa_pairs.append(QAPair(
                    question=question.text,
                    answer=answer_result.answer,
                    question_type=question.type,
                    keyword_used=question.keyword_used,
                    grounding_confidence=answer_result.grounding_confidence,
                    source_passages=answer_result.source_passages
                ))

        # Step 4: Quality gates
        faq_section = FAQSection(qa_pairs=qa_pairs)
        quality_report = self.quality.run_all_gates(
            faq_section=faq_section,
            document=document,
            keywords=keywords
        )

        return FAQGenerationResult(
            success=quality_report.passed,
            faq_section=faq_section if quality_report.passed else None,
            detection_result=detection,
            quality_report=quality_report.to_dict(),
            human_review_required=quality_report.human_review_required,
            review_triggers=quality_report.review_triggers
        )
```

---

## 10. References

### 10.1 Internal Documentation

- `00-system-overview.md` - System architecture and data flow
- `03-onpage-seo-framework.md` - Section 7.2: FAQ Schema requirements
- `05-ai-content-extraction.md` - RAG and content extraction patterns
- `07-guardrails.md` - Factual preservation and hallucination detection

### 10.2 External References

**Schema.org Documentation:**
- [FAQPage Structured Data](https://developers.google.com/search/docs/appearance/structured-data/faqpage)
- [Question Type](https://schema.org/Question)
- [Answer Type](https://schema.org/Answer)

**SEO Best Practices:**
- Google Search Central: FAQ rich results guidelines
- Schema.org FAQPage specification

**RAG and Retrieval:**
- Semantic similarity for passage retrieval
- Embedding models for question-answer matching
- Cross-encoder reranking for relevance

### 10.3 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-16 | AI Engineering Team | Initial specification |

---

*Document Version: 1.0*
*Last Updated: January 16, 2026*

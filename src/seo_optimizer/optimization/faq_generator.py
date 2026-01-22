"""
FAQ Generator - Contextual Question/Answer Creation

Responsibilities:
- Detect if FAQ section exists
- Generate relevant questions from content
- Create BLUF answers (40-60 words)
- Generate FAQ schema markup
- Ensure answers are self-contained

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import DocumentAST, NodeType

from .content_zones import should_skip_node
from .guardrails import SafetyGuardrails
from .models import FAQEntry, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import ContentAnalysisResult


# =============================================================================
# FAQ Generation Constants
# =============================================================================

# Answer length constraints (BLUF methodology)
MIN_ANSWER_WORDS = 40
MAX_ANSWER_WORDS = 60
OPTIMAL_ANSWER_WORDS = 50

# Question patterns for generation
QUESTION_PATTERNS = {
    "what": [
        "What is {topic}?",
        "What are the benefits of {topic}?",
        "What does {topic} involve?",
        "What should you know about {topic}?",
    ],
    "how": [
        "How does {topic} work?",
        "How can you use {topic}?",
        "How do you get started with {topic}?",
        "How is {topic} different from alternatives?",
    ],
    "why": [
        "Why is {topic} important?",
        "Why should you consider {topic}?",
        "Why do businesses use {topic}?",
    ],
    "when": [
        "When should you use {topic}?",
        "When is {topic} most effective?",
    ],
    "who": [
        "Who can benefit from {topic}?",
        "Who should consider {topic}?",
    ],
}

# FAQ section detection patterns
FAQ_SECTION_PATTERNS = [
    r"(?i)\b(faq|faqs|frequently\s+asked\s+questions)\b",
    r"(?i)\b(questions?\s+(?:and\s+)?answers?|q\s*&\s*a)\b",
    r"(?i)\b(common\s+questions|popular\s+questions)\b",
]

# Regex pattern to strip [H1], [H2], [H3] markers from text
HEADING_MARKER_PATTERN = re.compile(r"\[H[123]\]\s*", re.IGNORECASE)


@dataclass
class FAQGenerationResult:
    """Result of FAQ generation."""

    faqs: list[FAQEntry] = field(default_factory=list)
    has_existing_faq: bool = False
    existing_faq_count: int = 0
    generated_count: int = 0
    schema_markup: str | None = None


class FAQGenerator:
    """
    Generates contextual FAQ sections from content.

    Creates questions that users would naturally ask about
    the content topic, with BLUF-style concise answers.
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the FAQ generator."""
        self.config = config
        self.guardrails = guardrails

    def generate(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None = None,
    ) -> FAQGenerationResult:
        """
        Generate FAQ section for content.

        Args:
            ast: The document AST
            analysis: Pre-computed content analysis

        Returns:
            FAQGenerationResult with generated FAQs
        """
        if not self.config.generate_faq:
            return FAQGenerationResult()

        result = FAQGenerationResult()

        # Check for existing FAQ section
        existing_faq = self._detect_existing_faq(ast)
        result.has_existing_faq = existing_faq["exists"]
        result.existing_faq_count = existing_faq["count"]

        if result.has_existing_faq and not self.config.enhance_existing_faq:
            # FAQ exists and we shouldn't modify it
            return result

        # Extract topic and context from content
        topic_info = self._extract_topic_info(ast, analysis)

        # Generate questions based on content
        questions = self._generate_questions(topic_info)

        # Generate answers for each question with STRICT validation
        h1_title = topic_info.get("h1_title", "")

        for question in questions[: self.config.max_faq_items * 2]:  # Generate more, filter later
            if len(result.faqs) >= self.config.max_faq_items:
                break  # Stop once we have enough valid FAQs

            answer = self._generate_answer(question, topic_info, ast)

            if answer:
                # Apply AI vocabulary filter
                filter_result = self.guardrails.filter_ai_vocabulary(answer)
                answer = filter_result.cleaned_text

                # CRITICAL: Validate FAQ answer quality BEFORE adding
                is_valid, validation_reason = validate_faq_answer(
                    answer,
                    question,
                    source_title=h1_title,
                )

                if not is_valid:
                    # Skip this FAQ and try the next question
                    continue

                faq_entry = FAQEntry(
                    question=question,
                    answer=answer,
                    html_id=self._generate_html_id(question),
                    source_content=topic_info.get("primary_section", ""),
                )
                result.faqs.append(faq_entry)

        result.generated_count = len(result.faqs)

        # Generate schema markup
        if result.faqs:
            result.schema_markup = self._generate_schema_markup(result.faqs)

        return result

    def _detect_existing_faq(self, ast: DocumentAST) -> dict:
        """
        Detect if FAQ section already exists in content.

        Args:
            ast: Document AST

        Returns:
            Dict with existence status and count
        """
        result = {"exists": False, "count": 0, "section_id": None}

        for node in ast.nodes:
            text = node.text_content.lower()

            # Check headings for FAQ indicators
            if node.node_type == NodeType.HEADING:
                for pattern in FAQ_SECTION_PATTERNS:
                    if re.search(pattern, text):
                        result["exists"] = True
                        result["section_id"] = node.node_id
                        break

            # Count question-like paragraphs after FAQ heading
            if result["exists"] and node.node_type == NodeType.PARAGRAPH:
                if text.strip().endswith("?"):
                    result["count"] += 1

        return result

    def _extract_topic_info(
        self,
        ast: DocumentAST,
        analysis: ContentAnalysisResult | None,
    ) -> dict:
        """
        Extract topic information from content.

        Args:
            ast: Document AST
            analysis: Content analysis

        Returns:
            Dict with topic information
        """
        info: dict = {
            "primary_topic": "",
            "secondary_topics": [],
            "key_points": [],
            "entities": [],
            "primary_section": "",
            "content_summary": "",  # Full text for context
            "section_content": {},  # Map of H2 -> following paragraphs
            "h1_title": "",  # Track H1 title to exclude from FAQ answers
        }

        # Use primary keyword if available
        if self.config.primary_keyword:
            info["primary_topic"] = self.config.primary_keyword

        # Use secondary keywords as secondary topics
        if self.config.secondary_keywords:
            info["secondary_topics"] = list(self.config.secondary_keywords)

        # Extract H1 title and use as primary topic if no keyword specified
        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", node.metadata.get("heading_level", 2))
                if level == 1:
                    # Strip [H*] markers from heading text
                    h1_text = self._strip_heading_markers(node.text_content)
                    info["h1_title"] = h1_text  # Always track H1 for exclusion
                    if not info["primary_topic"]:
                        info["primary_topic"] = h1_text
                        info["primary_section"] = node.node_id
                    break

        # Extract key points from H2 headings AND build section content map
        # IMPORTANT: Skip metadata nodes (URL, Meta Title, etc.)
        current_h2 = None
        all_paragraphs: list[str] = []

        for node in ast.nodes:
            # Skip metadata fields - only process actual content
            if should_skip_node(node):
                continue

            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", node.metadata.get("heading_level", 2))
                if level == 2:
                    # Strip [H*] markers from heading text
                    heading_text = self._strip_heading_markers(node.text_content)
                    info["key_points"].append(heading_text)
                    current_h2 = heading_text
                    info["section_content"][current_h2] = []

            elif node.node_type == NodeType.PARAGRAPH:
                # Strip any [H*] markers that might appear in paragraphs
                para_text = self._strip_heading_markers(node.text_content)
                # Additional check: skip paragraphs that look like metadata
                if para_text and len(para_text) > 20:
                    # Exclude paragraphs that contain URL patterns or metadata labels
                    if not self._is_metadata_paragraph(para_text):
                        all_paragraphs.append(para_text)
                        if current_h2 and current_h2 in info["section_content"]:
                            info["section_content"][current_h2].append(para_text)

        # Build content summary from first few CONTENT paragraphs only
        info["content_summary"] = " ".join(all_paragraphs[:5])

        # Use semantic entities if available
        if self.config.semantic_entities:
            info["entities"] = list(self.config.semantic_entities)

        # Use analysis data if available
        if analysis:
            if hasattr(analysis, "semantic_score") and analysis.semantic_score:
                if analysis.semantic_score.entities:
                    info["entities"].extend(
                        [e.text for e in analysis.semantic_score.entities[:5]]
                    )

        return info

    def _is_metadata_paragraph(self, text: str) -> bool:
        """
        Check if a paragraph contains metadata rather than content.

        Args:
            text: Paragraph text to check

        Returns:
            True if the paragraph is metadata
        """
        text_lower = text.lower().strip()

        # Check for URL patterns
        if re.match(r"^https?://", text):
            return True

        # Check for metadata labels
        metadata_labels = [
            "url:", "meta title:", "meta description:",
            "page content", "seo information", "document information",
            "extracted at:", "source:", "document:"
        ]
        for label in metadata_labels:
            if text_lower.startswith(label):
                return True

        # Check if it's primarily a URL (contains URL with minimal other text)
        if "://" in text and len(text) < 200:
            # Check ratio of URL to other content
            url_match = re.search(r"https?://\S+", text)
            if url_match:
                url_len = len(url_match.group())
                if url_len > len(text) * 0.5:  # URL is >50% of content
                    return True

        return False

    def _is_heading_content(self, text: str, original_text: str | None = None) -> bool:
        """
        Check if text appears to be a heading rather than body content.

        Headings are typically short, lack ending punctuation, and may
        have originally contained [H*] markers.

        Args:
            text: The cleaned text to check
            original_text: The original text before stripping markers (optional)

        Returns:
            True if the text appears to be a heading
        """
        # Check if original text had heading markers
        if original_text and HEADING_MARKER_PATTERN.match(original_text):
            return True

        text = text.strip()

        # Short text without sentence-ending punctuation is likely a heading
        if len(text) < 100 and not text.endswith((".", "!", "?")):
            # Additional check: headings often have title-like patterns
            # e.g., "Everything You Need to Know About X"
            title_patterns = [
                r"everything you need to know",
                r"guide to",
                r"introduction to",
                r"overview of",
                r"what you need to know",
                r"complete guide",
                r"ultimate guide",
            ]
            for pattern in title_patterns:
                if re.search(pattern, text.lower()):
                    return True

        return False

    def _strip_heading_markers(self, text: str) -> str:
        """
        Strip [H1], [H2], [H3] markers from text.

        These markers are from the document extraction and should not
        appear in FAQ answers.

        Args:
            text: Text that may contain heading markers

        Returns:
            Cleaned text without heading markers
        """
        return HEADING_MARKER_PATTERN.sub("", text).strip()

    def _extract_topic_from_keyword(self, keyword: str) -> str:
        """
        Extract the core topic from a keyword phrase.

        Removes common verb prefixes to get a natural-sounding topic for FAQ questions.

        Examples:
            "Running a booster club" → "booster club"
            "How to start a business" → "business"
            "Starting your own podcast" → "podcast"
            "Creating effective content" → "effective content"

        Args:
            keyword: The keyword phrase (e.g., primary_keyword from config)

        Returns:
            The core topic (noun phrase) suitable for FAQ questions
        """
        if not keyword:
            return ""

        topic = keyword.strip()

        # Common verb prefixes to strip (order matters - longer phrases first)
        verb_prefixes = [
            "how to start a", "how to start an", "how to start",
            "how to create a", "how to create an", "how to create",
            "how to run a", "how to run an", "how to run",
            "how to manage a", "how to manage an", "how to manage",
            "how to build a", "how to build an", "how to build",
            "running a", "running an", "running your",
            "starting a", "starting an", "starting your",
            "creating a", "creating an", "creating your",
            "managing a", "managing an", "managing your",
            "building a", "building an", "building your",
            "what is a", "what is an", "what is",
            "what are", "why is", "why are",
            "when to", "where to", "who can",
        ]

        topic_lower = topic.lower()
        for prefix in verb_prefixes:
            if topic_lower.startswith(prefix):
                # Remove prefix and clean up
                topic = topic[len(prefix):].strip()
                break

        # Handle "your own X" pattern
        if topic.lower().startswith("your own "):
            topic = topic[9:].strip()
        elif topic.lower().startswith("your "):
            topic = topic[5:].strip()
        elif topic.lower().startswith("own "):
            topic = topic[4:].strip()

        # Ensure we have something left
        if not topic or len(topic) < 3:
            return keyword  # Return original if extraction failed

        return topic

    def _is_likely_plural(self, word: str) -> bool:
        """
        Check if a word/phrase appears to already be in plural form.

        Handles common English plural patterns. Not perfect but catches most cases.

        Args:
            word: The word or phrase to check

        Returns:
            True if the word appears to be plural
        """
        if not word:
            return False

        word = word.strip().lower()

        # Get the last word (for multi-word phrases like "booster clubs")
        last_word = word.split()[-1] if word else ""
        if not last_word:
            return False

        # Common singular words that end in 's' (not actually plural)
        singular_s_words = {
            "business", "class", "glass", "grass", "mass", "pass", "boss",
            "loss", "cross", "dress", "stress", "process", "progress",
            "success", "access", "address", "express", "compress", "congress",
            "analysis", "basis", "crisis", "diagnosis", "emphasis", "hypothesis",
            "thesis", "synopsis", "paralysis", "news", "lens", "means",
            "series", "species", "politics", "economics", "physics", "mathematics",
            "insurance", "compliance", "guidance", "assistance", "performance",
        }
        if last_word in singular_s_words:
            return False

        # Check for common plural endings
        # Words ending in -ies (from -y → -ies)
        if last_word.endswith("ies") and len(last_word) > 4:
            return True

        # Words ending in -es (from -s, -x, -z, -ch, -sh)
        if last_word.endswith("es") and len(last_word) > 3:
            # Check if the base form would end in s, x, z, ch, sh
            base = last_word[:-2]
            if base.endswith(("s", "x", "z", "ch", "sh")):
                return True
            # Also -oes plurals like "heroes", "potatoes"
            if base.endswith("o"):
                return True

        # Words ending in -s (regular plurals) - but not -ss or -us or -is
        if last_word.endswith("s") and not last_word.endswith(("ss", "us", "is", "ness", "less")):
            # Check it's not a verb (simple heuristic: verbs often follow patterns)
            # Most regular plural nouns: just end in -s after a non-s letter
            if len(last_word) > 3 and last_word[-2] not in "s":
                return True

        return False

    def _singularize(self, word: str) -> str:
        """
        Convert a plural word to singular form.

        Simple singularization for common English patterns.

        Args:
            word: The word to singularize

        Returns:
            The singular form of the word
        """
        if not word:
            return word

        word = word.strip()
        word_lower = word.lower()

        # Get the last word for multi-word phrases
        words = word.split()
        if not words:
            return word

        last_word = words[-1]
        last_lower = last_word.lower()

        # Check common singular words that look plural
        singular_s_words = {
            "business", "class", "glass", "grass", "mass", "pass", "boss",
            "loss", "cross", "dress", "stress", "process", "progress",
            "success", "access", "address", "express", "compress", "congress",
            "insurance", "compliance", "guidance", "assistance", "performance",
        }
        if last_lower in singular_s_words:
            return word  # Already singular

        # Handle -ies → -y
        if last_lower.endswith("ies") and len(last_lower) > 4:
            singular_last = last_word[:-3] + "y"
            words[-1] = singular_last
            return " ".join(words)

        # Handle -es → remove es (for -s, -x, -z, -ch, -sh, -o bases)
        if last_lower.endswith("es") and len(last_lower) > 3:
            base = last_lower[:-2]
            if base.endswith(("s", "x", "z", "ch", "sh", "o")):
                singular_last = last_word[:-2]
                words[-1] = singular_last
                return " ".join(words)

        # Handle regular -s plurals
        if last_lower.endswith("s") and not last_lower.endswith(("ss", "us", "is")):
            singular_last = last_word[:-1]
            words[-1] = singular_last
            return " ".join(words)

        return word

    def _get_topic_with_article(self, topic: str, plural: bool = False) -> str:
        """
        Get the topic with an appropriate article ("a" or "an") for natural phrasing.

        CRITICAL: Detects if topic is already plural to avoid double-pluralization.

        Examples:
            "booster club" → "a booster club"
            "booster clubs" → "booster clubs" (already plural, no article needed)
            "organization" → "an organization"
            "booster club" (plural=True) → "booster clubs"
            "booster clubs" (plural=True) → "booster clubs" (already plural)

        Args:
            topic: The topic noun phrase
            plural: If True, return plural form without article

        Returns:
            Topic with appropriate article or plural form
        """
        if not topic:
            return topic

        topic = topic.strip()
        is_already_plural = self._is_likely_plural(topic)

        if plural:
            # If already plural, return as-is to avoid "clubses" errors
            if is_already_plural:
                return topic

            # Simple pluralization (handles common cases)
            if topic.endswith(("s", "x", "z", "ch", "sh")):
                return topic + "es"
            elif topic.endswith("y") and len(topic) > 1 and topic[-2] not in "aeiou":
                return topic[:-1] + "ies"
            else:
                return topic + "s"

        # For singular with article - need to singularize if already plural
        if is_already_plural:
            topic = self._singularize(topic)

        # Check if topic already has an article
        topic_lower = topic.lower()
        if topic_lower.startswith(("a ", "an ", "the ")):
            return topic

        # Determine article based on first letter sound
        first_char = topic_lower[0] if topic_lower else ""
        vowels = "aeiou"

        # Special cases for vowel sounds
        if first_char in vowels:
            return f"an {topic}"
        else:
            return f"a {topic}"

    def _generate_questions(self, topic_info: dict) -> list[str]:
        """
        Generate relevant questions for the topic.

        Uses the extracted topic (not raw keyword) for natural-sounding questions.
        Example: "Running a booster club" → "What is a booster club?"
                 NOT "What is Running a booster club?"

        Args:
            topic_info: Extracted topic information

        Returns:
            List of question strings
        """
        questions: list[str] = []
        raw_keyword = topic_info.get("primary_topic", "")

        if not raw_keyword:
            return questions

        # Extract the core topic from the keyword for natural-sounding questions
        topic = self._extract_topic_from_keyword(raw_keyword)

        # Get topic variants for different question types
        topic_with_article = self._get_topic_with_article(topic)  # "a booster club"
        topic_plural = self._get_topic_with_article(topic, plural=True)  # "booster clubs"

        # Generate questions with proper grammar
        # Use article version for singular definitions, plural for general benefits
        natural_questions = [
            f"What is {topic_with_article}?",  # "What is a booster club?"
            f"How does {topic_with_article} work?",  # "How does a booster club work?"
            f"Why are {topic_plural} important?",  # "Why are booster clubs important?"
            f"When should you consider {topic_with_article}?",  # "When should you consider a booster club?"
            f"Who can benefit from {topic_plural}?",  # "Who can benefit from booster clubs?"
        ]

        questions.extend(natural_questions)

        # Add topic-specific questions from key points
        for point in topic_info.get("key_points", [])[:3]:
            # Generate question from heading
            question = self._heading_to_question(point, topic)
            if question and question not in questions:
                questions.append(question)

        # Add entity-based questions
        for entity in topic_info.get("entities", [])[:2]:
            question = f"How does {entity} relate to {topic_plural}?"
            if question not in questions:
                questions.append(question)

        return questions

    def _heading_to_question(self, heading: str, topic: str) -> str | None:
        """
        Convert a heading to a question.

        Args:
            heading: The heading text
            topic: The main topic

        Returns:
            Question string or None
        """
        heading = heading.strip()

        # Skip if already a question
        if heading.endswith("?"):
            return heading

        # Remove common heading prefixes
        heading = re.sub(r"^(how to|what is|why|when)\s+", "", heading, flags=re.I)

        # Generate question based on heading structure
        heading_lower = heading.lower()

        if "benefit" in heading_lower or "advantage" in heading_lower:
            return f"What are the benefits of {topic}?"
        elif "feature" in heading_lower:
            return f"What features does {topic} offer?"
        elif "cost" in heading_lower or "price" in heading_lower:
            return f"How much does {topic} cost?"
        elif "start" in heading_lower or "begin" in heading_lower:
            return f"How do you get started with {topic}?"
        elif "best" in heading_lower or "top" in heading_lower:
            return f"What are the best practices for {topic}?"
        elif "work" in heading_lower:
            return f"How does {topic} work?"
        else:
            # Generic question from heading
            return f"What should you know about {heading}?"

    def _generate_answer(
        self, question: str, topic_info: dict, ast: DocumentAST
    ) -> str | None:
        """
        Generate a BLUF-style answer for a question.

        Answers are:
        - 40-60 words
        - Self-contained (understandable without question)
        - Front-loaded with key information

        Args:
            question: The question to answer
            topic_info: Topic information
            ast: Document AST for context

        Returns:
            Answer string or None
        """
        raw_keyword = topic_info.get("primary_topic", "")
        if not raw_keyword:
            return None

        # Extract the core topic for natural-sounding answers
        # "Running a booster club" → "booster club" (or "a booster club")
        topic = self._extract_topic_from_keyword(raw_keyword)

        # Extract relevant content for the answer (passing topic_info for section mapping)
        relevant_content = self._find_relevant_content(question, ast, topic_info)

        # Generate answer based on question type
        question_lower = question.lower()

        if question_lower.startswith("what is"):
            answer = self._generate_definition_answer(topic, relevant_content)
        elif question_lower.startswith("what are the benefits"):
            answer = self._generate_benefits_answer(topic, relevant_content)
        elif question_lower.startswith("how does"):
            answer = self._generate_explanation_answer(topic, relevant_content)
        elif question_lower.startswith("how can") or question_lower.startswith("how do you"):
            answer = self._generate_howto_answer(topic, relevant_content)
        elif question_lower.startswith("why"):
            answer = self._generate_reasoning_answer(topic, relevant_content)
        elif question_lower.startswith("when"):
            answer = self._generate_timing_answer(topic, relevant_content)
        elif question_lower.startswith("who"):
            answer = self._generate_audience_answer(topic, relevant_content)
        else:
            answer = self._generate_generic_answer(topic, question, relevant_content)

        # Ensure answer meets word count requirements
        if answer:
            answer = self._adjust_answer_length(answer)

        return answer

    def _find_relevant_content(self, question: str, ast: DocumentAST, topic_info: dict | None = None) -> str:
        """
        Find content relevant to answering the question.

        Uses section content map for better matching, falls back to paragraph search.
        Filters out heading content and metadata to ensure only body text is used.

        CRITICAL: Excludes H1 title content to prevent FAQ answers from containing
        title fragments like "everything you need to know about X".

        Args:
            question: The question
            ast: Document AST
            topic_info: Optional topic info with section content map

        Returns:
            Relevant content string (body paragraphs only, no headings)
        """
        # Extract H1 title for filtering (must NOT appear in FAQ answers)
        h1_title = ""
        h1_words = set()
        if topic_info:
            h1_title = topic_info.get("h1_title", "").lower()
            # Extract significant words from H1 for fuzzy matching
            h1_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", h1_title))
            h1_words -= {"what", "about", "know", "need", "your", "this", "that", "with", "from", "have", "will"}

        # Extract key terms from question
        question_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        )
        # Remove common question words
        question_words -= {"what", "does", "should", "know", "about", "which", "where", "when", "that", "this", "with", "from", "have", "will", "would", "could"}

        def _contains_h1_content(text: str) -> bool:
            """Check if text contains H1 title content (which must be excluded)."""
            if not h1_title:
                return False
            text_lower = text.lower()
            # Check for exact H1 title
            if h1_title in text_lower:
                return True
            # Check for significant H1 word overlap (more than 60% match)
            if h1_words:
                text_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text_lower))
                overlap = len(h1_words & text_words)
                if overlap > len(h1_words) * 0.6:
                    return True
            # Check for title-like patterns
            title_patterns = [
                r"everything you need to know",
                r"complete guide to",
                r"ultimate guide to",
                r":\s*everything",
            ]
            for pattern in title_patterns:
                if re.search(pattern, text_lower):
                    return True
            return False

        # First, try to find matching section from topic_info
        if topic_info and "section_content" in topic_info:
            section_content = topic_info.get("section_content", {})
            for heading, paragraphs in section_content.items():
                heading_lower = heading.lower()
                # Check if heading matches question keywords
                heading_score = sum(1 for word in question_words if word in heading_lower)
                if heading_score > 0 and paragraphs:
                    # Filter out any heading-like content and H1 content from paragraphs
                    valid_paragraphs = [
                        p for p in paragraphs
                        if not self._is_heading_content(p)
                        and not _contains_h1_content(p)
                        and len(p) > 50
                    ]
                    if valid_paragraphs:
                        return " ".join(valid_paragraphs[:2])

            # Also check content_summary as fallback
            content_summary = topic_info.get("content_summary", "")
            if content_summary and len(content_summary) > 50:
                # Ensure content_summary doesn't contain heading-like or H1 content
                if not self._is_heading_content(content_summary) and not _contains_h1_content(content_summary):
                    return content_summary

        relevant_paragraphs: list[tuple[int, str]] = []

        for node in ast.nodes:
            if node.node_type != NodeType.PARAGRAPH:
                continue

            # Skip metadata nodes
            if should_skip_node(node):
                continue

            original_text = node.text_content

            # Strip [H*] markers from content
            text = self._strip_heading_markers(original_text)
            if not text or len(text) < 50:  # Increased minimum length
                continue

            # Skip metadata content
            if self._is_metadata_paragraph(text):
                continue

            # Skip heading-like content (including text that had [H*] markers)
            if self._is_heading_content(text, original_text):
                continue

            # CRITICAL: Skip content that contains H1 title fragments
            if _contains_h1_content(text):
                continue

            # Require proper sentence structure (ends with punctuation)
            if not text.rstrip().endswith((".", "!", "?")):
                continue

            text_lower = text.lower()

            # Score relevance by term overlap
            score = sum(1 for word in question_words if word in text_lower)

            if score > 0:
                relevant_paragraphs.append((score, text))

        # Sort by relevance and take top content
        relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)

        if relevant_paragraphs:
            # Content already has markers stripped
            return " ".join([p[1] for p in relevant_paragraphs[:2]])

        # Final fallback: use any substantial CONTENT paragraph (not metadata/heading/H1)
        for node in ast.nodes:
            if node.node_type == NodeType.PARAGRAPH and len(node.text_content) > 50:
                original_text = node.text_content
                text = self._strip_heading_markers(original_text)

                if (not should_skip_node(node)
                    and not self._is_metadata_paragraph(text)
                    and not self._is_heading_content(text, original_text)
                    and not _contains_h1_content(text)
                    and text.rstrip().endswith((".", "!", "?"))):
                    return text

        return ""

    def _generate_definition_answer(self, topic: str, context: str) -> str:
        """
        Generate a definition-style answer using actual content.

        IMPORTANT: Uses complete sentences from context, not fragments.
        The answer should be self-contained and grammatically correct.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks - we prefer skipping the FAQ
        over generating generic filler.

        Args:
            topic: The topic being defined (e.g., "booster club")
            context: Relevant content from the document

        Returns:
            A complete, grammatical answer string, or empty string if insufficient content
        """
        topic_with_article = self._get_topic_with_article(topic)

        # Extract COMPLETE sentences from context (not fragments)
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            # Use complete sentences directly - they're already grammatical
            first_sentence = complete_sentences[0]

            # Check if the first sentence already defines the topic
            first_lower = first_sentence.lower()
            if topic.lower() in first_lower and ("is" in first_lower or "are" in first_lower):
                # Sentence already defines the topic - use directly
                answer = first_sentence
                if len(complete_sentences) > 1:
                    answer += " " + complete_sentences[1]
                if len(complete_sentences) > 2 and len(answer.split()) < 45:
                    answer += " " + complete_sentences[2]
            else:
                # Just use the content sentences without generic framing
                answer = first_sentence
                if len(complete_sentences) > 1:
                    answer += " " + complete_sentences[1]
                if len(complete_sentences) > 2 and len(answer.split()) < 45:
                    answer += " " + complete_sentences[2]

            return answer

        elif context and len(context) > 50:
            # Try to use complete sentences from context
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_benefits_answer(self, topic: str, context: str) -> str:
        """
        Generate a benefits-focused answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            # Use complete sentences to form a coherent answer
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            # Try to use complete sentences from context
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_explanation_answer(self, topic: str, context: str) -> str:
        """
        Generate an explanation-style answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_howto_answer(self, topic: str, context: str) -> str:
        """
        Generate a how-to style answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_reasoning_answer(self, topic: str, context: str) -> str:
        """
        Generate a reasoning-style answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_timing_answer(self, topic: str, context: str) -> str:
        """
        Generate a timing-focused answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_audience_answer(self, topic: str, context: str) -> str:
        """
        Generate an audience-focused answer using complete sentences from content.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _generate_generic_answer(
        self, topic: str, question: str, context: str
    ) -> str:
        """
        Generate an answer based on actual content when pattern doesn't match.

        CRITICAL: Returns empty string if no real content is available.
        NO hardcoded template fallbacks.
        """
        # Extract complete sentences from context
        complete_sentences = self._extract_key_info(context, max_phrases=3)

        if complete_sentences and len(complete_sentences) >= 1:
            answer = complete_sentences[0]
            if len(complete_sentences) > 1:
                answer += " " + complete_sentences[1]
            if len(complete_sentences) > 2 and len(answer.split()) < 45:
                answer += " " + complete_sentences[2]
            return answer

        elif context and len(context) > 50:
            first_sentences = self._get_first_sentences(context, 3)
            if first_sentences and len(first_sentences.split()) >= 30:
                return first_sentences

        # NO FALLBACK - return empty string if insufficient real content
        return ""

    def _get_first_sentences(self, text: str, count: int = 2) -> str:
        """
        Extract first N COMPLETE sentences from text.

        Only returns sentences that:
        - End with proper punctuation (., !, ?)
        - Are at least 20 characters long
        - Don't appear to be headings or fragments

        Args:
            text: Source text to extract from
            count: Number of sentences to extract

        Returns:
            String containing the complete sentences
        """
        if not text:
            return ""

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Filter to only complete sentences
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Must end with proper punctuation
            if not sentence.endswith((".", "!", "?")):
                continue
            # Must be substantial (not just a fragment)
            if len(sentence) < 20:
                continue
            # Skip if it looks like a heading/title
            if self._is_heading_content(sentence):
                continue
            complete_sentences.append(sentence)

        # Take first N complete sentences
        selected = complete_sentences[:count]

        if not selected:
            return ""

        # Join and clean
        result = " ".join(selected)
        return result.strip()

    def _extract_key_info(self, text: str, max_phrases: int = 3) -> list[str]:
        """
        Extract complete, meaningful sentences from text for FAQ answers.

        IMPORTANT: Returns COMPLETE sentences, not fragments. This prevents
        gibberish FAQ answers like "booster club is booster clubs: everything".

        Args:
            text: Source text to extract from
            max_phrases: Maximum number of sentences to return

        Returns:
            List of complete, self-contained sentences
        """
        if not text:
            return []

        # Split into sentences properly
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        complete_sentences: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()

            # Must be a complete sentence with proper ending
            if not sentence.endswith((".", "!", "?")):
                continue

            # Must be substantial (at least 30 chars to be meaningful)
            if len(sentence) < 30:
                continue

            # Skip heading-like content (short without proper sentence structure)
            if self._is_heading_content(sentence):
                continue

            # Skip sentences that look like title fragments
            sentence_lower = sentence.lower()
            title_patterns = [
                "everything you need to know",
                "complete guide",
                "ultimate guide",
                ": everything",
                "all you need",
            ]
            if any(pattern in sentence_lower for pattern in title_patterns):
                continue

            # Skip very short sentences that are likely fragments
            word_count = len(sentence.split())
            if word_count < 6:
                continue

            complete_sentences.append(sentence)

            if len(complete_sentences) >= max_phrases:
                break

        return complete_sentences

    def _extract_benefits(self, text: str) -> list[str]:
        """Extract benefit phrases from text."""
        benefits: list[str] = []

        # Look for benefit indicators
        patterns = [
            r"(?:benefit|advantage|help|improve|increase|enhance|reduce|save)\w*\s+\w+(?:\s+\w+){0,3}",
            r"(?:better|faster|easier|more efficient|more effective)\s+\w+(?:\s+\w+){0,2}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            benefits.extend(matches[:2])

        # Clean up
        return [b.strip().lower() for b in benefits if len(b) > 10][:3]

    def _extract_processes(self, text: str) -> list[str]:
        """Extract process descriptions from text."""
        processes: list[str] = []

        patterns = [
            r"(?:works by|operates by|functions by|processes)\s+\w+(?:\s+\w+){0,5}",
            r"(?:analyzes?|generates?|creates?|produces?)\s+\w+(?:\s+\w+){0,3}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            processes.extend(matches[:2])

        return [p.strip().lower() for p in processes if len(p) > 10][:2]

    def _extract_steps(self, text: str) -> list[str]:
        """Extract step-like instructions from text."""
        steps: list[str] = []

        patterns = [
            r"(?:first|then|next|finally|step\s+\d+)\s*[,:]\s*\w+(?:\s+\w+){0,5}",
            r"(?:start|begin|create|set up|configure)\s+\w+(?:\s+\w+){0,4}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend(matches[:3])

        return [s.strip().lower() for s in steps if len(s) > 10][:3]

    def _extract_reasons(self, text: str) -> list[str]:
        """Extract reasoning phrases from text."""
        reasons: list[str] = []

        patterns = [
            r"(?:because|since|as|due to)\s+\w+(?:\s+\w+){0,5}",
            r"(?:important|essential|critical|vital)\s+(?:for|to|in)\s+\w+(?:\s+\w+){0,3}",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            reasons.extend(matches[:2])

        return [r.strip().lower() for r in reasons if len(r) > 10][:2]

    def _adjust_answer_length(self, answer: str) -> str | None:
        """
        Adjust answer to meet word count requirements.

        Target: 40-60 words (BLUF methodology)

        CRITICAL: Does NOT add filler content. If answer is too short and cannot
        be improved with real content, returns None so the FAQ is skipped.

        Args:
            answer: The answer to adjust

        Returns:
            Adjusted answer, or None if answer is too short
        """
        if not answer:
            return None

        words = answer.split()
        word_count = len(words)

        if word_count < MIN_ANSWER_WORDS:
            # Too short - REJECT instead of adding filler
            # The validation will catch this anyway, but we fail fast here
            return None

        elif word_count > MAX_ANSWER_WORDS:
            # Too long - truncate at sentence boundary
            sentences = re.split(r"(?<=[.!?])\s+", answer)
            result: list[str] = []
            current_count = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())
                if current_count + sentence_words <= MAX_ANSWER_WORDS:
                    result.append(sentence)
                    current_count += sentence_words
                else:
                    break

            if result:
                answer = " ".join(result)
            else:
                # Can't fit even one sentence - reject
                return None

        # Ensure proper ending
        answer = answer.strip()
        if not answer.endswith((".", "!", "?")):
            answer += "."

        return answer

    def _generate_html_id(self, question: str) -> str:
        """
        Generate a unique HTML ID for FAQ entry.

        Args:
            question: The FAQ question

        Returns:
            HTML-safe ID string
        """
        # Create slug from question
        slug = question.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)  # Remove punctuation
        slug = re.sub(r"\s+", "-", slug)  # Replace spaces with hyphens
        slug = slug[:50]  # Limit length

        # Add hash for uniqueness
        hash_suffix = hashlib.md5(question.encode()).hexdigest()[:6]

        return f"faq-{slug}-{hash_suffix}"

    def _generate_schema_markup(self, faqs: list[FAQEntry]) -> str:
        """
        Generate JSON-LD FAQ schema markup.

        Args:
            faqs: List of FAQ entries

        Returns:
            JSON-LD schema string
        """
        import json

        schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [],
        }

        for faq in faqs:
            qa_item = {
                "@type": "Question",
                "name": faq.question,
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq.answer,
                },
            }
            schema["mainEntity"].append(qa_item)

        return json.dumps(schema, indent=2)

    def validate_faq_quality(self, faqs: list[FAQEntry]) -> list[dict]:
        """
        Validate FAQ quality against standards.

        Args:
            faqs: List of FAQ entries

        Returns:
            List of validation issues
        """
        issues: list[dict] = []

        for i, faq in enumerate(faqs):
            # Check question format
            if not faq.question.endswith("?"):
                issues.append({
                    "faq_index": i,
                    "issue": "question_format",
                    "message": "Question should end with question mark",
                })

            # Check answer length
            word_count = len(faq.answer.split())
            if word_count < MIN_ANSWER_WORDS:
                issues.append({
                    "faq_index": i,
                    "issue": "answer_too_short",
                    "message": f"Answer has {word_count} words, minimum is {MIN_ANSWER_WORDS}",
                })
            elif word_count > MAX_ANSWER_WORDS:
                issues.append({
                    "faq_index": i,
                    "issue": "answer_too_long",
                    "message": f"Answer has {word_count} words, maximum is {MAX_ANSWER_WORDS}",
                })

            # Check HTML ID
            if not faq.html_id or not faq.html_id.startswith("faq-"):
                issues.append({
                    "faq_index": i,
                    "issue": "invalid_html_id",
                    "message": "HTML ID should be present and start with 'faq-'",
                })

            # Check self-containment (answer mentions topic, not just "it")
            answer_lower = faq.answer.lower()
            first_words = answer_lower.split()[:3]
            if first_words and first_words[0] in ["it", "this", "that", "they"]:
                issues.append({
                    "faq_index": i,
                    "issue": "not_self_contained",
                    "message": "Answer should be self-contained, avoid starting with pronouns",
                })

        return issues


# =============================================================================
# FAQ Quality Gate - CRITICAL for FAQ answer validation
# =============================================================================


# Banned phrases that indicate non-self-contained or filler answers
BANNED_FAQ_PHRASES = [
    # Reference-dependent phrases (requires context)
    "this article",
    "as mentioned",
    "see above",
    "click here",
    "the following",
    "as discussed",
    "in this guide",
    "in this post",
    "as we mentioned",
    "as stated above",
    "as explained above",
    "read more",
    "learn more",
    "find out more",
    "check out",
    # Generic filler phrases (low information density)
    "it is important to note",
    "understanding these aspects",
    "this approach helps ensure",
    "anyone affected by the topic",
    "for making informed decisions",
    "this is important",
    "is important because",
    "comprehensive process",
    "various aspects",
    "several key",
    "numerous benefits",
    "many advantages",
    "wide range of",
    "plays a crucial role",
    "plays an important role",
    "is essential for",
    "is crucial for",
    "is vital for",
    "stakeholders and participants",
    "proper engagement",
    # Title-like patterns (H1 contamination)
    "everything you need to know",
    "complete guide to",
    "ultimate guide to",
    "all you need to know",
    "comprehensive overview",
    # Circular/meaningless definitions
    "refers to a specific type",
    "is a type of organization that serves specific purposes",
    "involves examining their various components",
    "encompasses several important aspects",
]

# Additional phrases that indicate incoherent content
INCOHERENT_PATTERNS = [
    r":\s*everything\b",  # Colon followed by "everything"
    r"\b\w+\s+is\s+\w+s\s*:\s*\w+",  # "X is Xs: Y" pattern (gibberish)
    r"\b(\w+)\s+\1\s+\1\b",  # Triple word repetition
    r"^[a-z]",  # Starts with lowercase (sentence fragment)
]


def validate_faq_answer(
    answer: str,
    question: str,
    source_title: str | None = None,
    min_words: int = MIN_ANSWER_WORDS,
    max_words: int = MAX_ANSWER_WORDS,
) -> tuple[bool, str]:
    """
    Quality gate for FAQ answers - CRITICAL validation function.

    This is the primary validation gate that ensures FAQ answers meet
    quality standards before being included in the output.

    Validation checks:
    1. Length: 40-60 words (configurable via min/max)
    2. Fragment detection: Must have subject and verb
    3. Self-containment: No dangling references
    4. Title contamination: Answer shouldn't start with title
    5. Question echo: Answer shouldn't repeat the question
    6. Incoherent patterns: No gibberish constructions
    7. Sentence completeness: All sentences must be complete

    Args:
        answer: The FAQ answer to validate
        question: The FAQ question (for echo detection)
        source_title: Optional document title (for contamination check)
        min_words: Minimum word count (default 40)
        max_words: Maximum word count (default 60)

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not answer or not answer.strip():
        return False, "Empty answer"

    answer = answer.strip()
    answer_lower = answer.lower()

    # 1. Length check
    word_count = len(answer.split())
    if word_count < min_words:
        return False, f"Answer too short ({word_count} words, minimum is {min_words})"
    if word_count > max_words:
        return False, f"Answer too long ({word_count} words, maximum is {max_words})"

    # 2. Fragment check - must have subject and verb
    has_valid_structure = _check_sentence_structure(answer)
    if not has_valid_structure:
        return False, "Answer appears to be a fragment (no subject-verb structure)"

    # 3. Self-containment check - no banned phrases
    for phrase in BANNED_FAQ_PHRASES:
        if phrase in answer_lower:
            return False, f"Answer contains non-self-contained phrase: '{phrase}'"

    # Check for pronoun-only subjects at start
    first_words = answer_lower.split()[:3]
    if first_words:
        # Don't allow starting with bare pronouns that require context
        if first_words[0] in ["it", "this", "that", "they", "he", "she", "we"]:
            # Unless followed by "is" + noun (e.g., "This is a tool that...")
            if len(first_words) >= 2 and first_words[1] != "is":
                return False, "Answer starts with context-dependent pronoun"

    # 4. Title contamination check
    if source_title:
        title_lower = source_title.lower()
        # Check if answer starts with title content
        answer_first_50 = answer_lower[:min(50, len(answer_lower))]
        if title_lower in answer_first_50:
            return False, "Answer starts with document title (title contamination)"

        # Check for significant title word overlap at the start
        title_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", title_lower))
        title_words -= {"what", "about", "know", "need", "your", "this", "that", "with"}

        if title_words:
            answer_start_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", answer_first_50))
            overlap = len(title_words & answer_start_words)
            if overlap > len(title_words) * 0.6:
                return False, "Answer contains too much title content at start"

    # 5. Question echo check - answer shouldn't repeat the question
    if question:
        question_lower = question.lower().rstrip("?")
        question_words = question_lower.split()[:5]
        answer_first_words = answer_lower.split()[:5]

        # Count word overlap
        overlap = sum(1 for w in answer_first_words if w in question_words)
        if overlap > 3:
            return False, "Answer echoes the question too closely"

    # 6. Must end with proper punctuation
    if not answer.rstrip().endswith((".", "!", "?")):
        return False, "Answer doesn't end with proper punctuation"

    # 7. Must have at least 2 complete sentences for substance
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    complete_sentences = [s for s in sentences if s.strip() and len(s.split()) >= 5]
    if len(complete_sentences) < 2:
        return False, "Answer should have at least 2 complete sentences"

    # 8. Check for incoherent patterns (gibberish)
    for pattern in INCOHERENT_PATTERNS:
        if re.search(pattern, answer):
            return False, f"Answer contains incoherent pattern"

    # 9. Check each sentence is grammatically complete (ends properly, starts capitalized)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Sentence should start with uppercase (unless after abbreviation)
        if sentence and sentence[0].islower():
            # Check if previous sentence ended with an abbreviation
            if not re.search(r"\b(?:e\.g|i\.e|etc|vs|dr|mr|ms|mrs)\s*$", answer_lower[:answer_lower.find(sentence.lower())], re.IGNORECASE):
                return False, "Sentence starts with lowercase (possible fragment)"

        # Check for incomplete thoughts (ending mid-sentence markers)
        if sentence.rstrip().endswith((",", ";", ":")):
            return False, "Sentence ends with incomplete marker"

        # Check for sentence fragments that look like titles/headings
        if len(sentence.split()) < 6 and not sentence.endswith((".", "!", "?")):
            return False, "Short sentence fragment detected"

    # 10. Check for word repetition patterns indicating incoherence
    words = answer_lower.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and words[i] not in ("very", "had", "that"):
            return False, f"Repeated word detected: '{words[i]}'"

    return True, ""


def _check_sentence_structure(text: str) -> bool:
    """
    Check if text has valid sentence structure (subject + verb).

    Uses spaCy if available, otherwise uses heuristic.

    Args:
        text: Text to check

    Returns:
        True if text has valid sentence structure
    """
    # Try spaCy for accurate analysis
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)

            has_subject = False
            has_verb = False

            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl"):
                    has_subject = True
                if token.pos_ == "VERB":
                    has_verb = True

            return has_subject and has_verb

        except OSError:
            pass  # spaCy model not found, use heuristic
    except ImportError:
        pass  # spaCy not installed, use heuristic

    # Heuristic fallback: check for common verb patterns
    # This is less accurate but works without spaCy
    text_lower = text.lower()

    # Common verb patterns (expanded to catch more verbs)
    verb_patterns = [
        r"\b(is|are|was|were|will|would|can|could|should|have|has|had)\b",
        r"\b(provides?|offers?|includes?|involves?|requires?|means?)\b",
        r"\b(helps?|makes?|creates?|allows?|enables?|supports?)\b",
        r"\b(refers?|describes?|explains?|defines?|represents?)\b",
        r"\b(delivers?|eliminates?|gains?|access|transforms?)\b",
        r"\b(protects?|covers?|shields?|safeguards?|insures?)\b",
        r"\b(operates?|functions?|works?|runs?|manages?)\b",
        r"\b(raises?|funds?|purchases?|buys?|sells?)\b",
        r"\b(starts?|begins?|establishes?|forms?|organizes?)\b",
        r"\b(need|needs|needed|want|wants|wanted|get|gets|got)\b",
    ]

    has_verb = any(re.search(pattern, text_lower) for pattern in verb_patterns)

    # Check for subject by looking for nouns/pronouns at sentence start
    words = text.split()
    first_word_original = words[0] if words else ""
    first_word_lower = first_word_original.lower()

    subject_indicators = [
        "a", "an", "the", "this", "that", "these", "those",
        "it", "they", "we", "he", "she", "i", "you",
    ]

    # Subject exists if first word is:
    # - a common subject indicator (articles, pronouns)
    # - a capitalized word (likely a proper noun or start of sentence)
    has_subject = (
        first_word_lower in subject_indicators or
        (first_word_original and first_word_original[0].isupper())
    )

    return has_verb and has_subject


def filter_valid_faqs(
    faqs: list[FAQEntry],
    source_title: str | None = None,
    min_valid: int = 3,
) -> list[FAQEntry]:
    """
    Filter FAQs to only include those passing quality gate.

    Args:
        faqs: List of FAQ entries to filter
        source_title: Optional document title for contamination check
        min_valid: Minimum number of valid FAQs to return

    Returns:
        List of valid FAQ entries
    """
    valid_faqs: list[FAQEntry] = []

    for faq in faqs:
        is_valid, reason = validate_faq_answer(
            faq.answer,
            faq.question,
            source_title=source_title,
        )

        if is_valid:
            valid_faqs.append(faq)

    # If we don't have enough valid FAQs, log warning
    if len(valid_faqs) < min_valid:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Only {len(valid_faqs)} FAQs passed quality gate "
            f"(minimum {min_valid} requested)"
        )

    return valid_faqs

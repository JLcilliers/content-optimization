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

        # Generate answers for each question
        for question in questions[: self.config.max_faq_items]:
            answer = self._generate_answer(question, topic_info, ast)

            if answer:
                # Apply AI vocabulary filter
                filter_result = self.guardrails.filter_ai_vocabulary(answer)
                answer = filter_result.cleaned_text

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
        }

        # Use primary keyword if available
        if self.config.primary_keyword:
            info["primary_topic"] = self.config.primary_keyword

        # Use secondary keywords as secondary topics
        if self.config.secondary_keywords:
            info["secondary_topics"] = list(self.config.secondary_keywords)

        # Extract from H1 if no keyword specified
        if not info["primary_topic"]:
            for node in ast.nodes:
                if node.node_type == NodeType.HEADING:
                    level = node.metadata.get("level", node.metadata.get("heading_level", 2))
                    if level == 1:
                        info["primary_topic"] = node.text_content.strip()
                        info["primary_section"] = node.node_id
                        break

        # Extract key points from H2 headings AND build section content map
        current_h2 = None
        all_paragraphs: list[str] = []

        for node in ast.nodes:
            if node.node_type == NodeType.HEADING:
                level = node.metadata.get("level", node.metadata.get("heading_level", 2))
                if level == 2:
                    heading_text = node.text_content.strip()
                    info["key_points"].append(heading_text)
                    current_h2 = heading_text
                    info["section_content"][current_h2] = []

            elif node.node_type == NodeType.PARAGRAPH:
                para_text = node.text_content.strip()
                if para_text and len(para_text) > 20:
                    all_paragraphs.append(para_text)
                    if current_h2 and current_h2 in info["section_content"]:
                        info["section_content"][current_h2].append(para_text)

        # Build content summary from first few paragraphs
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

    def _generate_questions(self, topic_info: dict) -> list[str]:
        """
        Generate relevant questions for the topic.

        Args:
            topic_info: Extracted topic information

        Returns:
            List of question strings
        """
        questions: list[str] = []
        topic = topic_info.get("primary_topic", "")

        if not topic:
            return questions

        # Generate questions from patterns
        question_types = ["what", "how", "why", "when", "who"]

        for q_type in question_types:
            patterns = QUESTION_PATTERNS.get(q_type, [])
            if patterns:
                # Use first pattern for each type
                question = patterns[0].format(topic=topic)
                questions.append(question)

        # Add topic-specific questions from key points
        for point in topic_info.get("key_points", [])[:3]:
            # Generate question from heading
            question = self._heading_to_question(point, topic)
            if question and question not in questions:
                questions.append(question)

        # Add entity-based questions
        for entity in topic_info.get("entities", [])[:2]:
            question = f"How does {entity} relate to {topic}?"
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
        topic = topic_info.get("primary_topic", "")
        if not topic:
            return None

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

        Args:
            question: The question
            ast: Document AST
            topic_info: Optional topic info with section content map

        Returns:
            Relevant content string
        """
        # Extract key terms from question
        question_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", question.lower())
        )
        # Remove common question words
        question_words -= {"what", "does", "should", "know", "about", "which", "where", "when", "that", "this", "with", "from", "have", "will", "would", "could"}

        # First, try to find matching section from topic_info
        if topic_info and "section_content" in topic_info:
            section_content = topic_info.get("section_content", {})
            for heading, paragraphs in section_content.items():
                heading_lower = heading.lower()
                # Check if heading matches question keywords
                heading_score = sum(1 for word in question_words if word in heading_lower)
                if heading_score > 0 and paragraphs:
                    return " ".join(paragraphs[:2])

            # Also check content_summary as fallback
            content_summary = topic_info.get("content_summary", "")
            if content_summary and len(content_summary) > 50:
                return content_summary

        relevant_paragraphs: list[tuple[int, str]] = []

        for node in ast.nodes:
            if node.node_type != NodeType.PARAGRAPH:
                continue

            text = node.text_content
            if not text or len(text) < 30:
                continue

            text_lower = text.lower()

            # Score relevance by term overlap
            score = sum(1 for word in question_words if word in text_lower)

            if score > 0:
                relevant_paragraphs.append((score, text))

        # Sort by relevance and take top content
        relevant_paragraphs.sort(key=lambda x: x[0], reverse=True)

        if relevant_paragraphs:
            return " ".join([p[1] for p in relevant_paragraphs[:2]])

        # Final fallback: use any substantial paragraph
        for node in ast.nodes:
            if node.node_type == NodeType.PARAGRAPH and len(node.text_content) > 50:
                return node.text_content

        return ""

    def _generate_definition_answer(self, topic: str, context: str) -> str:
        """Generate a definition-style answer using actual content."""
        # Extract key information from context
        key_info = self._extract_key_info(context, max_phrases=3)

        if key_info and len(key_info) >= 2:
            answer = f"{topic} is {key_info[0]}. {key_info[1].capitalize()}."
            if len(key_info) > 2:
                answer += f" {key_info[2].capitalize()}."
        elif context and len(context) > 50:
            # Use first sentence from context directly
            first_sentence = self._get_first_sentences(context, 2)
            answer = f"{topic} refers to {first_sentence.lower() if not first_sentence[0].isupper() else first_sentence}"
        else:
            # Minimal fallback - at least name the topic clearly
            answer = (
                f"{topic} is a concept that involves specific practices and approaches. "
                f"Understanding {topic} requires examining its core components and applications. "
                f"This topic covers important aspects that affect how the process works."
            )

        return answer

    def _generate_benefits_answer(self, topic: str, context: str) -> str:
        """Generate a benefits-focused answer using actual content."""
        benefits = self._extract_benefits(context)

        if benefits and len(benefits) >= 2:
            answer = f"The main benefits of {topic} include {benefits[0]} and {benefits[1]}."
            if len(benefits) > 2:
                answer += f" Additionally, {benefits[2]}."
        elif context and len(context) > 50:
            # Extract from context directly
            key_points = self._get_first_sentences(context, 2)
            answer = f"The key benefits of {topic} are described as follows: {key_points}"
        else:
            answer = (
                f"The benefits of {topic} depend on how it is implemented and used. "
                f"Proper application of {topic} principles can lead to better outcomes. "
                f"Understanding these benefits helps in making informed decisions."
            )

        return answer

    def _generate_explanation_answer(self, topic: str, context: str) -> str:
        """Generate an explanation-style answer using actual content."""
        processes = self._extract_processes(context)

        if processes:
            answer = f"{topic} works by {processes[0]}."
            if len(processes) > 1:
                answer += f" The process involves {processes[1]}."
        elif context and len(context) > 50:
            # Use context to explain
            explanation = self._get_first_sentences(context, 2)
            answer = f"{topic} involves a process where {explanation.lower() if explanation else 'specific steps are followed'}."
        else:
            answer = (
                f"{topic} works through a series of defined steps and procedures. "
                f"The approach requires understanding the underlying principles involved. "
                f"Each component of {topic} contributes to the overall process."
            )

        return answer

    def _generate_howto_answer(self, topic: str, context: str) -> str:
        """Generate a how-to style answer using actual content."""
        steps = self._extract_steps(context)

        if steps and len(steps) >= 2:
            answer = f"To get started with {topic}, first {steps[0]}. Then, {steps[1]}."
            if len(steps) > 2:
                answer += f" Finally, {steps[2]}."
        elif context and len(context) > 50:
            # Use context for guidance
            guidance = self._get_first_sentences(context, 2)
            answer = f"Getting started with {topic} involves: {guidance}"
        else:
            answer = (
                f"Getting started with {topic} requires following specific steps. "
                f"Begin by understanding the basic requirements and prerequisites. "
                f"Then proceed with the implementation according to established guidelines."
            )

        return answer

    def _generate_reasoning_answer(self, topic: str, context: str) -> str:
        """Generate a reasoning-style answer using actual content."""
        reasons = self._extract_reasons(context)

        if reasons and len(reasons) >= 2:
            answer = f"{topic} is important because {reasons[0]}. Furthermore, {reasons[1]}."
        elif context and len(context) > 50:
            # Extract reasoning from context
            reasoning = self._get_first_sentences(context, 2)
            answer = f"{topic} matters because {reasoning.lower() if reasoning else 'it addresses key needs'}."
        else:
            answer = (
                f"{topic} is important for several reasons related to its core purpose. "
                f"Understanding why {topic} matters helps in proper implementation. "
                f"The significance becomes clear when examining its practical applications."
            )

        return answer

    def _generate_timing_answer(self, topic: str, context: str) -> str:
        """Generate a timing-focused answer using actual content."""
        if context and len(context) > 50:
            timing_info = self._get_first_sentences(context, 2)
            return (
                f"The appropriate time to consider {topic} depends on specific circumstances. "
                f"{timing_info} Understanding when to apply {topic} principles ensures better results."
            )
        return (
            f"The timing for {topic} depends on the specific situation and requirements. "
            f"Generally, {topic} should be considered when the relevant conditions are met. "
            f"Planning ahead and understanding the prerequisites helps determine the right timing."
        )

    def _generate_audience_answer(self, topic: str, context: str) -> str:
        """Generate an audience-focused answer using actual content."""
        if context and len(context) > 50:
            audience_info = self._get_first_sentences(context, 2)
            return (
                f"People interested in {topic} typically include individuals seeking specific outcomes. "
                f"{audience_info} Anyone affected by the topic can benefit from understanding it better."
            )
        return (
            f"People who can benefit from understanding {topic} include individuals directly involved in related activities. "
            f"Information about {topic} is valuable for anyone making decisions in this area. "
            f"Both newcomers and experienced individuals can gain insights from learning about {topic}."
        )

    def _generate_generic_answer(
        self, topic: str, question: str, context: str
    ) -> str:
        """Generate an answer based on actual content when pattern doesn't match."""
        if context and len(context) > 50:
            content_summary = self._get_first_sentences(context, 3)
            return (
                f"Regarding {topic}: {content_summary} "
                f"This information helps address key aspects of the topic."
            )
        return (
            f"Understanding {topic} involves examining its various components and implications. "
            f"The topic encompasses several important aspects worth considering. "
            f"Further exploration of {topic} reveals its relevance to the broader context."
        )

    def _get_first_sentences(self, text: str, count: int = 2) -> str:
        """Extract first N sentences from text."""
        if not text:
            return ""

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        # Take first N sentences
        selected = sentences[:count]

        # Join and clean
        result = " ".join(selected)

        # Ensure it ends properly
        result = result.strip()
        if result and not result.endswith((".", "!", "?")):
            result += "."

        return result

    def _extract_key_info(self, text: str, max_phrases: int = 3) -> list[str]:
        """Extract key phrases from text."""
        if not text:
            return []

        # Simple extraction - in production, use NLP
        sentences = re.split(r"[.!?]+", text)
        phrases: list[str] = []

        for sentence in sentences[:5]:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Extract a meaningful phrase
                words = sentence.split()[:8]
                phrase = " ".join(words).lower()
                if phrase and len(phrase) > 10:
                    phrases.append(phrase)

        return phrases[:max_phrases]

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

    def _adjust_answer_length(self, answer: str) -> str:
        """
        Adjust answer to meet word count requirements.

        Target: 40-60 words (BLUF methodology)

        Args:
            answer: The answer to adjust

        Returns:
            Adjusted answer
        """
        words = answer.split()
        word_count = len(words)

        if word_count < MIN_ANSWER_WORDS:
            # Too short - need to expand with meaningful additions
            answer = answer.rstrip(".")

            # Add expansion sentences until we meet minimum
            expansion_sentences = [
                "This approach helps ensure consistent results across different use cases.",
                "Understanding these aspects provides a solid foundation for making informed decisions.",
                "Proper implementation leads to better outcomes and more effective results.",
            ]

            expansion_idx = 0
            while word_count < MIN_ANSWER_WORDS and expansion_idx < len(expansion_sentences):
                answer += ". " + expansion_sentences[expansion_idx]
                word_count = len(answer.split())
                expansion_idx += 1

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

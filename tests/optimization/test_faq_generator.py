"""
Tests for FAQ generator.

CRITICAL: Tests for:
- Question generation format
- BLUF answer length (40-60 words)
- HTML ID generation
- Self-contained answers
- FAQ schema generation
"""

import json

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.faq_generator import (
    FAQGenerator,
    FAQGenerationResult,
    MAX_ANSWER_WORDS,
    MIN_ANSWER_WORDS,
)
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.models import OptimizationConfig

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        primary_keyword="content optimization",
        secondary_keywords=["SEO", "AI content"],
        semantic_entities=["BERT", "E-E-A-T"],
        generate_faq=True,
        max_faq_items=5,
    )


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def faq_generator(config, guardrails):
    """Create FAQ generator."""
    return FAQGenerator(config, guardrails)


@pytest.fixture
def sample_ast():
    """Create sample document AST."""
    nodes = [
        make_node(
            "h1",
            NodeType.HEADING,
            "Complete Guide to Content Optimization",
            0,
            {"level": 1},
        ),
        make_node(
            "p1",
            NodeType.PARAGRAPH,
            "Content optimization is the process of improving your content "
            "to make it more effective for search engines and users. It involves "
            "keyword research, content structure, and readability improvements.",
            40,
        ),
        make_node(
            "h2_1",
            NodeType.HEADING,
            "Benefits of Content Optimization",
            250,
            {"level": 2},
        ),
        make_node(
            "p2",
            NodeType.PARAGRAPH,
            "Optimizing your content helps improve search rankings, "
            "increases organic traffic, and enhances user engagement. These benefits "
            "lead to better conversion rates and business growth.",
            290,
        ),
        make_node(
            "h2_2",
            NodeType.HEADING,
            "How to Get Started",
            480,
            {"level": 2},
        ),
        make_node(
            "p3",
            NodeType.PARAGRAPH,
            "Start by analyzing your current content performance. "
            "Then identify target keywords and optimize your headings, meta tags, "
            "and body content. Regular monitoring ensures continued improvement.",
            510,
        ),
    ]
    return DocumentAST(nodes=nodes, metadata={})


class TestFAQGeneration:
    """Tests for FAQ generation."""

    def test_generates_faq_entries(self, faq_generator, sample_ast):
        """Test FAQ entries are generated."""
        result = faq_generator.generate(sample_ast)
        assert isinstance(result, FAQGenerationResult)
        assert len(result.faqs) > 0

    def test_respects_max_faq_items(self, config, guardrails, sample_ast):
        """Test respects max FAQ items limit."""
        config.max_faq_items = 3
        generator = FAQGenerator(config, guardrails)
        result = generator.generate(sample_ast)
        assert len(result.faqs) <= 3

    def test_disabled_faq_generation(self, guardrails, sample_ast):
        """Test FAQ generation can be disabled."""
        config = OptimizationConfig(generate_faq=False)
        generator = FAQGenerator(config, guardrails)
        result = generator.generate(sample_ast)
        assert len(result.faqs) == 0


class TestQuestionFormat:
    """Tests for question format."""

    def test_questions_end_with_question_mark(self, faq_generator, sample_ast):
        """Test all questions end with ?."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            assert faq.question.endswith("?"), f"Question doesn't end with ?: {faq.question}"

    def test_questions_start_with_question_word(self, faq_generator, sample_ast):
        """Test questions start with question words."""
        result = faq_generator.generate(sample_ast)
        question_starters = ["what", "how", "why", "when", "who", "which", "where"]
        for faq in result.faqs:
            first_word = faq.question.lower().split()[0]
            assert first_word in question_starters, f"Question doesn't start with question word: {faq.question}"

    def test_questions_include_topic(self, faq_generator, sample_ast):
        """Test questions reference the topic."""
        result = faq_generator.generate(sample_ast)
        # At least some questions should mention the topic
        topic_mentions = sum(
            1 for faq in result.faqs
            if "content" in faq.question.lower() or "optimization" in faq.question.lower()
        )
        assert topic_mentions > 0

    def test_questions_are_natural_language(self, faq_generator, sample_ast):
        """Test questions are natural language."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            # Questions should have multiple words
            assert len(faq.question.split()) >= 4
            # No weird punctuation
            assert not faq.question.startswith("-")
            assert not faq.question.startswith("*")


class TestAnswerLength:
    """Tests for BLUF answer length (40-60 words)."""

    def test_answers_meet_minimum_length(self, faq_generator, sample_ast):
        """Test answers meet minimum word count."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            word_count = len(faq.answer.split())
            assert word_count >= MIN_ANSWER_WORDS, (
                f"Answer too short ({word_count} words): {faq.answer[:50]}..."
            )

    def test_answers_meet_maximum_length(self, faq_generator, sample_ast):
        """Test answers don't exceed maximum word count."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            word_count = len(faq.answer.split())
            assert word_count <= MAX_ANSWER_WORDS + 5, (  # Small buffer
                f"Answer too long ({word_count} words): {faq.answer[:50]}..."
            )

    def test_answers_in_optimal_range(self, faq_generator, sample_ast):
        """Test most answers are in optimal range."""
        result = faq_generator.generate(sample_ast)
        in_range = 0
        for faq in result.faqs:
            word_count = len(faq.answer.split())
            if MIN_ANSWER_WORDS <= word_count <= MAX_ANSWER_WORDS:
                in_range += 1

        # At least 50% should be in optimal range
        if result.faqs:
            assert in_range / len(result.faqs) >= 0.5


class TestSelfContainedAnswers:
    """Tests for self-contained answers."""

    def test_answers_dont_start_with_pronouns(self, faq_generator, sample_ast):
        """Test answers don't start with 'it', 'this', 'that'."""
        result = faq_generator.generate(sample_ast)
        problematic_starters = ["it", "this", "that", "they", "these", "those"]

        for faq in result.faqs:
            first_word = faq.answer.lower().split()[0] if faq.answer else ""
            assert first_word not in problematic_starters, (
                f"Answer starts with pronoun: {faq.answer[:50]}..."
            )

    def test_answers_are_complete_sentences(self, faq_generator, sample_ast):
        """Test answers are complete sentences."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            # Answers should end with proper punctuation
            assert faq.answer.rstrip().endswith((".", "!", "?")), (
                f"Answer doesn't end with punctuation: {faq.answer[-30:]}"
            )

    def test_answers_mention_topic_entity(self, faq_generator, sample_ast):
        """Test answers reference the topic."""
        result = faq_generator.generate(sample_ast)
        # Most answers should mention the main topic
        topic_mentions = sum(
            1 for faq in result.faqs
            if "content" in faq.answer.lower() or "optimization" in faq.answer.lower()
        )
        if result.faqs:
            assert topic_mentions / len(result.faqs) >= 0.3  # At least 30%


class TestHTMLIDGeneration:
    """Tests for HTML ID generation."""

    def test_html_ids_are_generated(self, faq_generator, sample_ast):
        """Test HTML IDs are generated for all FAQs."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            assert faq.html_id is not None
            assert len(faq.html_id) > 0

    def test_html_ids_start_with_faq(self, faq_generator, sample_ast):
        """Test HTML IDs start with 'faq-'."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            assert faq.html_id.startswith("faq-"), f"Invalid HTML ID: {faq.html_id}"

    def test_html_ids_are_valid(self, faq_generator, sample_ast):
        """Test HTML IDs are valid (no spaces, lowercase)."""
        result = faq_generator.generate(sample_ast)
        for faq in result.faqs:
            assert " " not in faq.html_id
            assert faq.html_id == faq.html_id.lower()

    def test_html_ids_are_unique(self, faq_generator, sample_ast):
        """Test HTML IDs are unique."""
        result = faq_generator.generate(sample_ast)
        ids = [faq.html_id for faq in result.faqs]
        assert len(ids) == len(set(ids)), "Duplicate HTML IDs found"

    def test_html_id_generation_method(self, faq_generator):
        """Test HTML ID generation method directly."""
        html_id = faq_generator._generate_html_id("What is content optimization?")
        assert html_id.startswith("faq-")
        assert "what" in html_id.lower()
        assert " " not in html_id


class TestSchemaMarkup:
    """Tests for FAQ schema generation."""

    def test_schema_is_generated(self, faq_generator, sample_ast):
        """Test schema markup is generated."""
        result = faq_generator.generate(sample_ast)
        assert result.schema_markup is not None

    def test_schema_is_valid_json(self, faq_generator, sample_ast):
        """Test schema markup is valid JSON."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup:
            schema = json.loads(result.schema_markup)
            assert isinstance(schema, dict)

    def test_schema_has_correct_type(self, faq_generator, sample_ast):
        """Test schema has FAQPage type."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup:
            schema = json.loads(result.schema_markup)
            assert schema.get("@type") == "FAQPage"

    def test_schema_has_context(self, faq_generator, sample_ast):
        """Test schema has @context."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup:
            schema = json.loads(result.schema_markup)
            assert "@context" in schema
            assert "schema.org" in schema["@context"]

    def test_schema_has_main_entity(self, faq_generator, sample_ast):
        """Test schema has mainEntity array."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup:
            schema = json.loads(result.schema_markup)
            assert "mainEntity" in schema
            assert isinstance(schema["mainEntity"], list)

    def test_schema_entries_have_question_type(self, faq_generator, sample_ast):
        """Test schema entries have Question type."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup and result.faqs:
            schema = json.loads(result.schema_markup)
            for entry in schema["mainEntity"]:
                assert entry.get("@type") == "Question"

    def test_schema_entries_have_accepted_answer(self, faq_generator, sample_ast):
        """Test schema entries have acceptedAnswer."""
        result = faq_generator.generate(sample_ast)
        if result.schema_markup and result.faqs:
            schema = json.loads(result.schema_markup)
            for entry in schema["mainEntity"]:
                assert "acceptedAnswer" in entry
                assert entry["acceptedAnswer"].get("@type") == "Answer"


class TestExistingFAQDetection:
    """Tests for detecting existing FAQ sections."""

    def test_detects_faq_heading(self, faq_generator):
        """Test detects FAQ heading."""
        nodes = [
            make_node(
                "h2",
                NodeType.HEADING,
                "Frequently Asked Questions",
                0,
                {"level": 2},
            ),
            make_node("p1", NodeType.PARAGRAPH, "What is this about?", 30),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = faq_generator.generate(ast)
        assert result.has_existing_faq is True

    def test_detects_faq_abbreviation(self, faq_generator):
        """Test detects FAQ abbreviation."""
        nodes = [
            make_node("h2", NodeType.HEADING, "FAQ", 0, {"level": 2}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = faq_generator.generate(ast)
        assert result.has_existing_faq is True

    def test_counts_existing_questions(self, faq_generator):
        """Test counts existing FAQ questions."""
        nodes = [
            make_node("h2", NodeType.HEADING, "FAQ", 0, {"level": 2}),
            make_node("p1", NodeType.PARAGRAPH, "What is this?", 5),
            make_node("p2", NodeType.PARAGRAPH, "How does it work?", 20),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = faq_generator.generate(ast)
        assert result.existing_faq_count >= 2


class TestFAQQualityValidation:
    """Tests for FAQ quality validation."""

    def test_validate_good_faq(self, faq_generator):
        """Test validating a good FAQ."""
        from seo_optimizer.optimization.models import FAQEntry

        faq = FAQEntry(
            question="What is content optimization?",
            answer="Content optimization is the comprehensive process of improving your content "
            "to achieve better search engine rankings and enhanced user engagement. It "
            "involves strategic keyword research, thoughtful content structure improvements, and "
            "readability enhancements to create more effective and valuable content for your audience. "
            "This process helps businesses reach their target customers more effectively online.",
            html_id="faq-what-is-content-optimization-abc123",
        )
        issues = faq_generator.validate_faq_quality([faq])
        # Should have few or no issues
        critical_issues = [i for i in issues if "too_short" in i.get("issue", "") or "too_long" in i.get("issue", "")]
        assert len(critical_issues) == 0

    def test_validate_short_answer(self, faq_generator):
        """Test validation catches short answers."""
        from seo_optimizer.optimization.models import FAQEntry

        faq = FAQEntry(
            question="What is SEO?",
            answer="SEO is search engine optimization.",
            html_id="faq-seo",
        )
        issues = faq_generator.validate_faq_quality([faq])
        # Should flag short answer
        short_issues = [i for i in issues if "short" in i.get("issue", "")]
        assert len(short_issues) > 0

    def test_validate_missing_question_mark(self, faq_generator):
        """Test validation catches missing question mark."""
        from seo_optimizer.optimization.models import FAQEntry

        faq = FAQEntry(
            question="What is content optimization",  # Missing ?
            answer="Content optimization involves improving your content for "
            "better search rankings and user engagement through various techniques.",
            html_id="faq-test",
        )
        issues = faq_generator.validate_faq_quality([faq])
        format_issues = [i for i in issues if "format" in i.get("issue", "")]
        assert len(format_issues) > 0

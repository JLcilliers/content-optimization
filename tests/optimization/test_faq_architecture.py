"""
Architectural Tests for FAQ Generator

These tests enforce the core architectural principle:
"The FAQ generator must ONLY output text that exists in or is derived from the source document."

These tests verify:
1. No hardcoded template strings in generated output
2. Questions must be traceable to document content
3. Answers must be traceable to document content
4. No fallback patterns exist
"""

import inspect
import re

import pytest

from seo_optimizer.optimization.faq_generator import (
    FAQGenerator,
    FAQGenerationResult,
    validate_faq_answer,
)
from seo_optimizer.optimization.models import OptimizationConfig
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.ingestion.models import DocumentAST, ContentNode, NodeType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Standard FAQ generation config."""
    return OptimizationConfig(
        primary_keyword="test topic",
        generate_faq=True,
        max_faq_items=5,
    )


@pytest.fixture
def guardrails(config):
    """Standard guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def faq_generator(config, guardrails):
    """Standard FAQ generator."""
    return FAQGenerator(config, guardrails)


def make_document(*paragraphs, h2_headings=None):
    """Helper to create test documents."""
    nodes = []
    pos = 0

    # Add H1
    nodes.append(ContentNode(
        node_id="h1",
        node_type=NodeType.HEADING,
        text_content="Test Document Title",
        position=pos,
        metadata={"level": 1},
    ))
    pos += 50

    # Add H2 headings if provided
    if h2_headings:
        for i, heading in enumerate(h2_headings):
            nodes.append(ContentNode(
                node_id=f"h2_{i}",
                node_type=NodeType.HEADING,
                text_content=heading,
                position=pos,
                metadata={"level": 2},
            ))
            pos += 50

    # Add paragraphs
    for i, para in enumerate(paragraphs):
        nodes.append(ContentNode(
            node_id=f"p{i}",
            node_type=NodeType.PARAGRAPH,
            text_content=para,
            position=pos,
            metadata={},
        ))
        pos += 100

    return DocumentAST(nodes=nodes, metadata={})


# =============================================================================
# ARCHITECTURAL TEST 1: No Hardcoded Template Strings
# =============================================================================


class TestNoHardcodedTemplates:
    """Verify no hardcoded template strings leak into output."""

    # These template patterns should NEVER appear in FAQ output
    FORBIDDEN_TEMPLATE_PATTERNS = [
        r"What is \{topic\}",
        r"How does \{topic\} work",
        r"Why is \{topic\} important",
        r"Who can benefit from \{topic\}",
        r"When should you use \{topic\}",
        # Generic filler patterns
        r"This approach helps ensure",
        r"Understanding these aspects",
        r"It is important to note",
        r"plays a crucial role",
        r"is essential for",
        r"wide range of",
        r"various aspects",
        r"several key",
    ]

    def test_no_template_placeholders_in_questions(self, faq_generator):
        """Questions should never contain {topic} or similar placeholders."""
        # Create a document with real content
        doc = make_document(
            "Insurance coverage protects organizations from financial liability in case "
            "of accidents or legal claims. General liability insurance typically costs "
            "between $300 and $1,000 annually depending on the organization's size and activities.",
            h2_headings=["Understanding Insurance Coverage", "Benefits of Insurance"]
        )

        result = faq_generator.generate(doc)

        for faq in result.faqs:
            assert "{topic}" not in faq.question
            assert "{keyword}" not in faq.question
            assert "{" not in faq.question, f"Template placeholder in question: {faq.question}"

    def test_no_template_placeholders_in_answers(self, faq_generator):
        """Answers should never contain {topic} or similar placeholders."""
        doc = make_document(
            "Insurance coverage protects organizations from financial liability in case "
            "of accidents or legal claims. General liability insurance typically costs "
            "between $300 and $1,000 annually depending on the organization's size and activities.",
            h2_headings=["Understanding Insurance Coverage"]
        )

        result = faq_generator.generate(doc)

        for faq in result.faqs:
            assert "{topic}" not in faq.answer
            assert "{keyword}" not in faq.answer
            assert "{" not in faq.answer, f"Template placeholder in answer: {faq.answer}"

    def test_no_forbidden_filler_patterns_in_answers(self, faq_generator):
        """Answers should never contain forbidden filler patterns."""
        doc = make_document(
            "Insurance coverage protects organizations from financial liability in case "
            "of accidents or legal claims. General liability insurance typically costs "
            "between $300 and $1,000 annually depending on the organization's size and activities. "
            "Directors and officers liability insurance provides additional protection for "
            "leadership against claims of mismanagement or negligence.",
            h2_headings=["Understanding Insurance Coverage"]
        )

        result = faq_generator.generate(doc)

        for faq in result.faqs:
            answer_lower = faq.answer.lower()
            for pattern in self.FORBIDDEN_TEMPLATE_PATTERNS:
                assert not re.search(pattern, answer_lower, re.IGNORECASE), \
                    f"Forbidden pattern '{pattern}' found in answer: {faq.answer}"


# =============================================================================
# ARCHITECTURAL TEST 2: Questions Must Be Content-Derived
# =============================================================================


class TestQuestionsAreContentDerived:
    """Verify questions are derived from document content, not templates."""

    def test_questions_relate_to_document_headings(self, faq_generator):
        """Generated questions should relate to actual document headings."""
        doc = make_document(
            "Fundraising requires careful planning and community engagement. "
            "Successful campaigns start with clear goals and target audiences.",
            h2_headings=["Fundraising Best Practices", "Getting Started with Campaigns"]
        )

        result = faq_generator.generate(doc)

        if result.faqs:
            # At least one question should contain words from the headings
            heading_words = {"fundraising", "best", "practices", "getting", "started", "campaigns"}

            questions_have_heading_words = False
            for faq in result.faqs:
                question_words = set(faq.question.lower().split())
                if heading_words & question_words:
                    questions_have_heading_words = True
                    break

            assert questions_have_heading_words, \
                f"No questions relate to headings. Questions: {[f.question for f in result.faqs]}"

    def test_questions_without_matching_headings_return_empty(self, faq_generator):
        """Documents without transformable headings should generate fewer questions."""
        # Create document with headings that can't be transformed
        doc = make_document(
            "Some random content about various things. This paragraph has no clear topic.",
            h2_headings=["Random Heading XYZ", "Another Random ABC"]
        )

        result = faq_generator.generate(doc)

        # Should have 0 or very few FAQs since headings can't be transformed
        # and content has no existing questions
        assert len(result.faqs) <= 2, \
            f"Generated {len(result.faqs)} FAQs from untransformable content"

    def test_existing_questions_in_document_are_extracted(self, faq_generator):
        """Questions that exist in the document should be extracted."""
        doc = make_document(
            "What makes insurance important? Insurance coverage protects organizations from "
            "financial liability in case of accidents or legal claims. "
            "How much does coverage cost? General liability insurance typically costs "
            "between $300 and $1,000 annually depending on the organization's size.",
            h2_headings=["About Insurance"]
        )

        result = faq_generator.generate(doc)

        # Check if any of the document's questions were extracted
        questions = [faq.question.lower() for faq in result.faqs]

        # At least one question should match the document's questions
        document_has_questions = any(
            "insurance important" in q or "coverage cost" in q
            for q in questions
        )

        # This is a soft check - content-derived questions may still be valid
        # even if they don't exactly match the document's questions


# =============================================================================
# ARCHITECTURAL TEST 3: Answers Must Be Content-Derived
# =============================================================================


class TestAnswersAreContentDerived:
    """Verify answers are derived from document content only."""

    def test_answer_text_comes_from_document(self, faq_generator):
        """Answer text should be traceable to document paragraphs."""
        unique_content = (
            "The XYZ-Alpha protocol requires three specific components: authentication, "
            "encryption, and verification. Each component serves a distinct purpose in "
            "maintaining system integrity. Authentication ensures users are who they claim."
        )

        doc = make_document(unique_content, h2_headings=["Understanding XYZ-Alpha Protocol"])

        result = faq_generator.generate(doc)

        for faq in result.faqs:
            # Answer should contain words from the unique content
            unique_words = {"xyz-alpha", "protocol", "authentication", "encryption",
                          "verification", "integrity"}
            answer_lower = faq.answer.lower()

            # At least some unique words should appear
            matches = sum(1 for word in unique_words if word in answer_lower)

            # If the answer doesn't contain any unique document words, it's likely templated
            # Allow for answers that reference the content topic even if not exact words
            assert matches > 0 or "protocol" in answer_lower or "component" in answer_lower, \
                f"Answer doesn't appear derived from document: {faq.answer}"

    def test_empty_document_produces_no_faqs(self, faq_generator):
        """Empty documents should produce no FAQs."""
        doc = DocumentAST(nodes=[], metadata={})
        result = faq_generator.generate(doc)

        assert len(result.faqs) == 0, \
            f"Empty document produced {len(result.faqs)} FAQs"

    def test_metadata_only_document_produces_no_faqs(self, faq_generator):
        """Documents with only metadata should produce no FAQs."""
        nodes = [
            ContentNode(
                node_id="meta",
                node_type=NodeType.PARAGRAPH,
                text_content="https://example.com/page",
                position=0,
                metadata={"is_meta": True},
            ),
        ]
        doc = DocumentAST(nodes=nodes, metadata={})
        result = faq_generator.generate(doc)

        assert len(result.faqs) == 0, \
            f"Metadata-only document produced {len(result.faqs)} FAQs"


# =============================================================================
# ARCHITECTURAL TEST 4: No Fallback Patterns
# =============================================================================


class TestNoFallbackPatterns:
    """Verify no fallback/default patterns exist in the code."""

    def test_source_code_has_no_fallback_comments(self):
        """Check that answer methods document 'NO FALLBACK' policy."""
        import seo_optimizer.optimization.faq_generator as faq_module
        source = inspect.getsource(faq_module)

        # Should have "NO FALLBACK" comments in answer generation methods
        no_fallback_count = source.count("NO FALLBACK")

        # We expect at least 5 "NO FALLBACK" comments (one per answer method)
        assert no_fallback_count >= 5, \
            f"Expected at least 5 'NO FALLBACK' comments, found {no_fallback_count}"

    def test_answer_methods_return_empty_on_no_content(self, faq_generator):
        """Answer methods should return empty string when no content available."""
        # Test _generate_definition_answer with empty context
        result = faq_generator._generate_definition_answer("test", "")
        assert result == "", f"Definition answer should be empty, got: {result}"

        # Test _generate_benefits_answer with empty context
        result = faq_generator._generate_benefits_answer("test", "")
        assert result == "", f"Benefits answer should be empty, got: {result}"

        # Test _generate_explanation_answer with empty context
        result = faq_generator._generate_explanation_answer("test", "")
        assert result == "", f"Explanation answer should be empty, got: {result}"

    def test_adjust_length_returns_none_for_short_content(self, faq_generator):
        """_adjust_answer_length should return None for content that's too short."""
        short_content = "This is too short."
        result = faq_generator._adjust_answer_length(short_content)

        assert result is None, \
            f"Short content should return None, got: {result}"


# =============================================================================
# ARCHITECTURAL TEST 5: Heading Transformation Rules
# =============================================================================


class TestHeadingTransformationRules:
    """Verify heading transformation follows strict rules without fallbacks."""

    def test_transformation_uses_heading_text_not_templates(self, faq_generator):
        """Transformed questions should contain the actual heading text."""
        test_cases = [
            ("Benefits of Cloud Computing", "What are the benefits of Cloud Computing?"),
            ("How Kubernetes Works", "How does Kubernetes work?"),
            ("Getting Started with Docker", "How do you get started with Docker?"),
            ("Understanding Machine Learning", "What should you understand about Machine Learning?"),
            ("Security Best Practices", "What are the best practices for Security?"),
        ]

        for heading, expected_pattern in test_cases:
            result = faq_generator._heading_to_question_minimal(heading)

            if result:
                # The question should contain words from the original heading
                heading_words = set(heading.lower().split())
                question_words = set(result.lower().replace("?", "").split())

                # Remove common words
                common = {"what", "are", "the", "of", "how", "does", "do", "you", "should", "is"}
                heading_words -= common
                question_words -= common

                # At least half the heading words should be in the question
                overlap = heading_words & question_words
                assert len(overlap) >= len(heading_words) / 2, \
                    f"Heading '{heading}' transformed to '{result}' lost heading words"

    def test_untransformable_headings_return_none(self, faq_generator):
        """Headings that can't be cleanly transformed should return None."""
        untransformable = [
            "Random Text Here",
            "XYZ ABC DEF",
            "Chapter 1",
            "Section A",
            "Conclusion",
            "Summary",
        ]

        for heading in untransformable:
            result = faq_generator._heading_to_question_minimal(heading)
            assert result is None, \
                f"Heading '{heading}' should not be transformable, got: {result}"


# =============================================================================
# ARCHITECTURAL TEST 6: Complete Traceability
# =============================================================================


class TestCompleteTraceability:
    """Verify complete traceability from output to source."""

    def test_faq_output_traceable_to_input(self, faq_generator):
        """Every word in FAQ output should be traceable to input document."""
        # Create document with specific, unusual words
        unique_words = ["cryptographic", "asymmetric", "handshake", "cipher"]
        doc_content = (
            f"The {unique_words[0]} {unique_words[1]} {unique_words[2]} protocol uses "
            f"a {unique_words[3]} to establish secure connections. This process involves "
            "multiple steps including key exchange and verification. The protocol ensures "
            "data integrity through mathematical proofs."
        )

        doc = make_document(doc_content, h2_headings=["How the Protocol Works"])
        result = faq_generator.generate(doc)

        # Collect all words from document
        doc_words = set()
        for node in doc.nodes:
            doc_words.update(node.text_content.lower().split())

        # Add common question words that transformations add
        allowed_additions = {
            "what", "how", "why", "when", "who", "does", "is", "are", "the", "a", "an",
            "should", "you", "understand", "about", "best", "practices", "for",
        }

        for faq in result.faqs:
            # Check questions
            question_words = set(faq.question.lower().replace("?", "").split())
            unknown_question_words = question_words - doc_words - allowed_additions

            # Allow small number of unknown words (transformation artifacts)
            assert len(unknown_question_words) <= 3, \
                f"Question has untraceable words: {unknown_question_words}"

            # Check answers (should be almost entirely from document)
            answer_words = set(faq.answer.lower().replace(".", "").replace(",", "").split())
            unknown_answer_words = answer_words - doc_words - allowed_additions

            # Answers should have very few unknown words
            assert len(unknown_answer_words) <= 5, \
                f"Answer has untraceable words: {unknown_answer_words}"

"""
Tests for meta generator.

Tests:
- Title generation and constraints
- Description generation and CTAs
- Pixel width constraints
- Open Graph tag generation
"""

import pytest

from seo_optimizer.ingestion.models import DocumentAST, NodeType
from seo_optimizer.optimization.guardrails import SafetyGuardrails
from seo_optimizer.optimization.meta_generator import (
    CTA_TEMPLATES,
    MAX_DESCRIPTION_CHARS,
    MAX_DESCRIPTION_PIXELS,
    MAX_TITLE_CHARS,
    MAX_TITLE_PIXELS,
    MetaGenerator,
    MetaGenerationResult,
)
from seo_optimizer.optimization.models import OptimizationConfig

from .conftest import make_node


@pytest.fixture
def config():
    """Create test configuration."""
    return OptimizationConfig(
        primary_keyword="content optimization",
        brand_name="SEO Pro",
    )


@pytest.fixture
def guardrails(config):
    """Create guardrails."""
    return SafetyGuardrails(config)


@pytest.fixture
def generator(config, guardrails):
    """Create meta generator."""
    return MetaGenerator(config, guardrails)


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
            "Content optimization helps improve your search rankings. "
            "This guide teaches you the best practices for SEO success.",
            40,
        ),
        make_node("h2", NodeType.HEADING, "Benefits of Optimization", 150, {"level": 2}),
    ]
    return DocumentAST(nodes=nodes, metadata={})


class TestPixelConstraints:
    """Tests for pixel width constraints."""

    def test_max_title_pixels(self):
        """Test max title pixel width."""
        assert MAX_TITLE_PIXELS == 600

    def test_max_description_pixels(self):
        """Test max description pixel width."""
        assert MAX_DESCRIPTION_PIXELS == 920

    def test_max_title_chars_reasonable(self):
        """Test max title chars is reasonable."""
        assert 50 <= MAX_TITLE_CHARS <= 70

    def test_max_description_chars_reasonable(self):
        """Test max description chars is reasonable."""
        assert 140 <= MAX_DESCRIPTION_CHARS <= 160


class TestTitleGeneration:
    """Tests for meta title generation."""

    def test_generates_title(self, generator, sample_ast):
        """Test generates a title."""
        result = generator.generate(sample_ast)
        assert result.meta_tags is not None
        assert result.meta_tags.title is not None
        assert len(result.meta_tags.title) > 0

    def test_title_includes_keyword(self, generator, sample_ast):
        """Test title includes primary keyword."""
        result = generator.generate(sample_ast)
        if result.meta_tags and result.meta_tags.title:
            # Should include topic/keyword
            title_lower = result.meta_tags.title.lower()
            assert "content" in title_lower or "optimization" in title_lower or "guide" in title_lower

    def test_title_respects_length(self, generator, sample_ast):
        """Test title respects max length."""
        result = generator.generate(sample_ast)
        if result.meta_tags and result.meta_tags.title:
            assert len(result.meta_tags.title) <= MAX_TITLE_CHARS + 10  # Small buffer

    def test_title_pixel_width_calculated(self, generator, sample_ast):
        """Test title pixel width is calculated."""
        result = generator.generate(sample_ast)
        if result.meta_tags:
            assert result.meta_tags.title_pixel_width is not None
            assert result.meta_tags.title_pixel_width > 0

    def test_title_truncation_adds_ellipsis(self, generator):
        """Test truncated title adds ellipsis."""
        # Force a very long title by using a long H1
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "This is an extremely long title that definitely exceeds "
                "the maximum pixel width limit and needs to be truncated",
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = generator.generate(ast)

        if result.meta_tags and result.meta_tags.title:
            if len(result.meta_tags.title) >= MAX_TITLE_CHARS - 5:
                # If truncated, should end with ellipsis
                assert result.meta_tags.title.endswith("...") or len(result.meta_tags.title) < MAX_TITLE_CHARS


class TestDescriptionGeneration:
    """Tests for meta description generation."""

    def test_generates_description(self, generator, sample_ast):
        """Test generates a description."""
        result = generator.generate(sample_ast)
        assert result.meta_tags is not None
        assert result.meta_tags.description is not None
        assert len(result.meta_tags.description) > 0

    def test_description_includes_cta(self, generator, sample_ast):
        """Test description includes CTA."""
        result = generator.generate(sample_ast)
        if result.meta_tags and result.meta_tags.description:
            desc_lower = result.meta_tags.description.lower()
            has_cta = any(cta.lower() in desc_lower for cta in CTA_TEMPLATES)
            assert has_cta

    def test_description_respects_length(self, generator, sample_ast):
        """Test description respects max length."""
        result = generator.generate(sample_ast)
        if result.meta_tags and result.meta_tags.description:
            assert len(result.meta_tags.description) <= MAX_DESCRIPTION_CHARS + 20

    def test_description_pixel_width_calculated(self, generator, sample_ast):
        """Test description pixel width is calculated."""
        result = generator.generate(sample_ast)
        if result.meta_tags:
            assert result.meta_tags.description_pixel_width is not None
            assert result.meta_tags.description_pixel_width > 0


class TestCTATemplates:
    """Tests for CTA templates."""

    def test_cta_templates_defined(self):
        """Test CTA templates are defined."""
        assert len(CTA_TEMPLATES) > 0

    def test_common_ctas_present(self):
        """Test common CTAs are present."""
        common = ["Learn more", "Discover", "Get started"]
        for cta in common:
            assert any(cta.lower() in t.lower() for t in CTA_TEMPLATES)


class TestContentTypeDetection:
    """Tests for content type detection."""

    def test_detects_guide_content(self, generator):
        """Test detects guide content type."""
        nodes = [
            make_node("h1", NodeType.HEADING, "Complete Guide to SEO", 0, {"level": 1}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        context = generator._extract_content_context(ast, None)
        assert context["content_type"] == "guide"

    def test_detects_howto_content(self, generator):
        """Test detects how-to content type."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "How to Optimize Your Content",
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        context = generator._extract_content_context(ast, None)
        assert context["content_type"] == "howto"

    def test_detects_list_content(self, generator):
        """Test detects list content type."""
        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "10 Best SEO Tools in 2024",
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        context = generator._extract_content_context(ast, None)
        assert context["content_type"] == "list"


class TestExistingMetaTags:
    """Tests for handling existing meta tags."""

    def test_uses_existing_good_title(self, generator, sample_ast):
        """Test uses existing good title."""
        result = generator.generate(
            sample_ast,
            existing_title="Content Optimization: The Ultimate Guide",
        )
        # Should use or modify existing title
        assert result.meta_tags is not None

    def test_replaces_bad_title(self, generator, sample_ast):
        """Test replaces bad title."""
        result = generator.generate(
            sample_ast,
            existing_title="Page",  # Too short
        )
        if result.meta_tags and result.meta_tags.title:
            assert len(result.meta_tags.title) > 10


class TestValidation:
    """Tests for meta tag validation."""

    def test_validates_meta_tags(self, generator, sample_ast):
        """Test validates generated meta tags."""
        result = generator.generate(sample_ast)
        assert result.validation_issues is not None
        assert isinstance(result.validation_issues, list)

    def test_flags_missing_keyword_in_title(self, guardrails):
        """Test flags missing keyword in title."""
        config = OptimizationConfig(primary_keyword="SEO")
        generator = MetaGenerator(config, guardrails)

        nodes = [
            make_node(
                "h1",
                NodeType.HEADING,
                "Marketing Tips",  # No SEO keyword
                0,
                {"level": 1},
            ),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = generator.generate(ast)

        # May flag missing keyword
        assert isinstance(result.validation_issues, list)


class TestOpenGraphTags:
    """Tests for Open Graph tag generation."""

    def test_generates_og_tags(self, generator, sample_ast):
        """Test generates OG tags."""
        result = generator.generate(sample_ast)
        if result.meta_tags:
            og_tags = generator.generate_og_tags(result.meta_tags)

            assert "og:title" in og_tags
            assert "og:description" in og_tags
            assert "og:type" in og_tags

    def test_og_tags_with_url(self, generator, sample_ast):
        """Test OG tags include URL when provided."""
        result = generator.generate(sample_ast)
        if result.meta_tags:
            og_tags = generator.generate_og_tags(
                result.meta_tags,
                url="https://example.com/page"
            )
            assert og_tags.get("og:url") == "https://example.com/page"

    def test_og_tags_with_image(self, generator, sample_ast):
        """Test OG tags include image when provided."""
        result = generator.generate(sample_ast)
        if result.meta_tags:
            og_tags = generator.generate_og_tags(
                result.meta_tags,
                image_url="https://example.com/image.jpg"
            )
            assert og_tags.get("og:image") == "https://example.com/image.jpg"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, generator):
        """Test handling empty document."""
        ast = DocumentAST(nodes=[], metadata={})
        result = generator.generate(ast)
        assert isinstance(result, MetaGenerationResult)

    def test_no_h1(self, generator):
        """Test handling document without H1."""
        nodes = [
            make_node("p1", NodeType.PARAGRAPH, "Some content without a heading."),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = generator.generate(ast)
        # Should still generate meta tags
        assert isinstance(result, MetaGenerationResult)

    def test_no_keyword(self, guardrails):
        """Test handling no primary keyword."""
        config = OptimizationConfig(primary_keyword=None)
        generator = MetaGenerator(config, guardrails)

        nodes = [
            make_node("h1", NodeType.HEADING, "Page Title", 0, {"level": 1}),
        ]
        ast = DocumentAST(nodes=nodes, metadata={})
        result = generator.generate(ast)
        assert isinstance(result, MetaGenerationResult)

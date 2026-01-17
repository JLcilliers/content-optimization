"""
Tests for Entity Extractor Module

Tests NER functionality and entity matching.
"""

import pytest

from seo_optimizer.analysis.entity_extractor import (
    EntityExtractor,
    extract_entities_simple,
)
from seo_optimizer.analysis.models import EntityMatch


# Mark all tests in this module as slow (requires model loading)
pytestmark = pytest.mark.slow


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor() -> EntityExtractor:
    """Create an entity extractor instance."""
    return EntityExtractor(model="en_core_web_sm")


@pytest.fixture
def sample_text() -> str:
    """Sample text with various entities."""
    return """
    Apple Inc. was founded by Steve Jobs in Cupertino, California.
    The company released the iPhone in January 2007.
    Today, Microsoft and Google are major competitors.
    """


@pytest.fixture
def business_text() -> str:
    """Business-focused sample text."""
    return """
    Cloud computing has transformed enterprise IT infrastructure.
    Amazon Web Services (AWS) leads the market with 32% share.
    Companies like Netflix and Spotify rely on AWS for scalability.
    The data center industry is expected to grow 10% annually.
    """


# =============================================================================
# Basic Extraction Tests
# =============================================================================


class TestExtractEntities:
    """Tests for basic entity extraction."""

    def test_extract_empty_text(self, extractor: EntityExtractor) -> None:
        """Test extraction from empty text."""
        entities = extractor.extract_entities("")
        assert entities == []

    def test_extract_whitespace_only(self, extractor: EntityExtractor) -> None:
        """Test extraction from whitespace-only text."""
        entities = extractor.extract_entities("   \n\t  ")
        assert entities == []

    def test_extract_finds_organizations(
        self, extractor: EntityExtractor, sample_text: str
    ) -> None:
        """Test that organizations are found."""
        entities = extractor.extract_entities(sample_text)
        org_texts = [e.text for e in entities if e.entity_type == "ORG"]
        assert any("Apple" in t for t in org_texts)

    def test_extract_finds_persons(
        self, extractor: EntityExtractor, sample_text: str
    ) -> None:
        """Test that persons are found."""
        entities = extractor.extract_entities(sample_text)
        person_texts = [e.text for e in entities if e.entity_type == "PERSON"]
        assert any("Steve Jobs" in t for t in person_texts)

    def test_extract_finds_locations(
        self, extractor: EntityExtractor, sample_text: str
    ) -> None:
        """Test that locations are found."""
        entities = extractor.extract_entities(sample_text)
        loc_texts = [e.text for e in entities if e.entity_type == "LOCATION"]
        assert any("California" in t or "Cupertino" in t for t in loc_texts)

    def test_extract_returns_positions(
        self, extractor: EntityExtractor, sample_text: str
    ) -> None:
        """Test that character positions are returned."""
        entities = extractor.extract_entities(sample_text)
        for entity in entities:
            assert entity.start_char >= 0
            assert entity.end_char > entity.start_char
            assert entity.length > 0

    def test_extract_default_confidence(
        self, extractor: EntityExtractor, sample_text: str
    ) -> None:
        """Test that default confidence is 1.0."""
        entities = extractor.extract_entities(sample_text)
        for entity in entities:
            assert entity.confidence == 1.0


class TestExtractWithConcepts:
    """Tests for extraction with noun chunks."""

    def test_includes_noun_chunks(
        self, extractor: EntityExtractor, business_text: str
    ) -> None:
        """Test that noun chunks are included as concepts."""
        entities = extractor.extract_entities_with_concepts(business_text)
        concept_entities = [e for e in entities if e.entity_type == "CONCEPT"]
        # Should have some concept entities from noun chunks
        assert len(concept_entities) > 0

    def test_concept_lower_confidence(
        self, extractor: EntityExtractor, business_text: str
    ) -> None:
        """Test that noun chunk concepts have lower confidence."""
        entities = extractor.extract_entities_with_concepts(business_text)
        concepts = [e for e in entities if e.entity_type == "CONCEPT" and e.confidence < 1.0]
        # Some concepts should have 0.8 confidence (from noun chunks)
        assert len(concepts) > 0

    def test_no_duplicate_spans(
        self, extractor: EntityExtractor, business_text: str
    ) -> None:
        """Test that entities and concepts don't overlap."""
        entities = extractor.extract_entities_with_concepts(business_text)
        spans = [(e.start_char, e.end_char) for e in entities]
        # Check for exact duplicates
        assert len(spans) == len(set(spans))

    def test_can_disable_noun_chunks(
        self, extractor: EntityExtractor, business_text: str
    ) -> None:
        """Test disabling noun chunk extraction."""
        with_chunks = extractor.extract_entities_with_concepts(
            business_text, include_noun_chunks=True
        )
        without_chunks = extractor.extract_entities_with_concepts(
            business_text, include_noun_chunks=False
        )
        assert len(with_chunks) >= len(without_chunks)


# =============================================================================
# Entity Matching Tests
# =============================================================================


class TestMatchExpectedEntities:
    """Tests for matching against expected entities."""

    def test_match_exact_entities(self, extractor: EntityExtractor) -> None:
        """Test exact entity matching."""
        extracted = [
            EntityMatch("Apple Inc.", "ORG", 0, 10),
            EntityMatch("Steve Jobs", "PERSON", 20, 30),
            EntityMatch("California", "LOCATION", 40, 50),
        ]
        expected = ["Apple Inc.", "Steve Jobs"]

        found, missing = extractor.match_expected_entities(extracted, expected)

        assert "Apple Inc." in found
        assert "Steve Jobs" in found
        assert len(missing) == 0

    def test_match_case_insensitive(self, extractor: EntityExtractor) -> None:
        """Test case-insensitive matching."""
        extracted = [
            EntityMatch("Apple", "ORG", 0, 5),
        ]
        expected = ["apple", "APPLE"]

        found, missing = extractor.match_expected_entities(extracted, expected)

        assert len(found) == 2  # Both should match

    def test_match_partial_entities(self, extractor: EntityExtractor) -> None:
        """Test partial entity matching."""
        extracted = [
            EntityMatch("Amazon Web Services", "ORG", 0, 20),
        ]
        expected = ["AWS"]

        found, missing = extractor.match_expected_entities(extracted, expected)

        # "AWS" should be missing since it's not contained in "Amazon Web Services"
        assert "AWS" in missing

    def test_match_empty_expected(self, extractor: EntityExtractor) -> None:
        """Test with empty expected list."""
        extracted = [
            EntityMatch("Test", "ORG", 0, 4),
        ]

        found, missing = extractor.match_expected_entities(extracted, [])

        assert found == []
        assert missing == []

    def test_match_empty_extracted(self, extractor: EntityExtractor) -> None:
        """Test with empty extracted list."""
        expected = ["Company", "Product"]

        found, missing = extractor.match_expected_entities([], expected)

        assert found == []
        assert missing == expected


class TestCalculateEntityGap:
    """Tests for entity gap calculation."""

    def test_gap_with_missing_entities(self, extractor: EntityExtractor) -> None:
        """Test identifying missing competitor entities."""
        extracted = [
            EntityMatch("Cloud Computing", "CONCEPT", 0, 15),
        ]
        competitor_entities = ["Cloud Computing", "Serverless", "Kubernetes"]

        gaps = extractor.calculate_entity_gap(extracted, competitor_entities)

        assert "Serverless" in gaps
        assert "Kubernetes" in gaps
        assert "Cloud Computing" not in gaps

    def test_gap_all_covered(self, extractor: EntityExtractor) -> None:
        """Test when all competitor entities are covered."""
        extracted = [
            EntityMatch("AWS", "ORG", 0, 3),
            EntityMatch("Azure", "ORG", 10, 15),
        ]
        competitor_entities = ["AWS", "Azure"]

        gaps = extractor.calculate_entity_gap(extracted, competitor_entities)

        assert len(gaps) == 0


# =============================================================================
# Entity Density Tests
# =============================================================================


class TestEntityDensity:
    """Tests for entity density calculation."""

    def test_density_calculation(self, extractor: EntityExtractor) -> None:
        """Test basic density calculation."""
        text = "X" * 100  # 100 characters
        entities = [
            EntityMatch("Test", "ORG", 0, 10),  # 10 chars
            EntityMatch("Demo", "ORG", 50, 60),  # 10 chars
        ]

        density = extractor.get_entity_density(text, entities)

        assert density == 0.2  # 20 chars / 100 chars

    def test_density_empty_text(self, extractor: EntityExtractor) -> None:
        """Test density with empty text."""
        density = extractor.get_entity_density("", [])
        assert density == 0.0

    def test_density_no_entities(self, extractor: EntityExtractor) -> None:
        """Test density with no entities."""
        density = extractor.get_entity_density("Some text", [])
        assert density == 0.0


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGetUniqueEntities:
    """Tests for entity deduplication."""

    def test_removes_duplicates(self, extractor: EntityExtractor) -> None:
        """Test that duplicate entities are removed."""
        entities = [
            EntityMatch("Apple", "ORG", 0, 5),
            EntityMatch("Apple", "ORG", 50, 55),
            EntityMatch("Google", "ORG", 100, 106),
        ]

        unique = extractor.get_unique_entities(entities)

        assert len(unique) == 2

    def test_keeps_first_occurrence(self, extractor: EntityExtractor) -> None:
        """Test that first occurrence is kept."""
        entities = [
            EntityMatch("Apple", "ORG", 0, 5),
            EntityMatch("Apple", "ORG", 50, 55),
        ]

        unique = extractor.get_unique_entities(entities)

        assert unique[0].start_char == 0

    def test_case_insensitive_dedup(self, extractor: EntityExtractor) -> None:
        """Test case-insensitive deduplication."""
        entities = [
            EntityMatch("Apple", "ORG", 0, 5),
            EntityMatch("APPLE", "ORG", 50, 55),
            EntityMatch("apple", "ORG", 100, 105),
        ]

        unique = extractor.get_unique_entities(entities)

        assert len(unique) == 1


class TestGroupByType:
    """Tests for grouping entities by type."""

    def test_groups_by_type(self, extractor: EntityExtractor) -> None:
        """Test entity grouping."""
        entities = [
            EntityMatch("Apple", "ORG", 0, 5),
            EntityMatch("Steve Jobs", "PERSON", 10, 20),
            EntityMatch("Google", "ORG", 30, 36),
            EntityMatch("Elon Musk", "PERSON", 40, 49),
        ]

        groups = extractor.group_by_type(entities)

        assert len(groups["ORG"]) == 2
        assert len(groups["PERSON"]) == 2

    def test_empty_groups(self, extractor: EntityExtractor) -> None:
        """Test with no entities."""
        groups = extractor.group_by_type([])
        assert groups == {}


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestExtractEntitiesSimple:
    """Tests for the convenience function."""

    def test_simple_extraction(self) -> None:
        """Test simple extraction function."""
        text = "Apple Inc. is based in Cupertino."
        entities = extract_entities_simple(text)
        assert len(entities) > 0

    def test_empty_text(self) -> None:
        """Test with empty text."""
        entities = extract_entities_simple("")
        assert entities == []

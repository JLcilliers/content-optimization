"""
Entity Extractor - NER and Entity Analysis

Uses spaCy for Named Entity Recognition to:
- Extract entities with positions and confidence
- Match against expected semantic entities
- Calculate entity gaps vs competitor content

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.language import Language

from .models import EntityMatch

# Lazy loading of spaCy model
_nlp_model: Language | None = None


def _get_nlp(model: str = "en_core_web_sm") -> Language:
    """
    Lazy load the spaCy model.

    Uses en_core_web_sm by default for faster loading.
    Use en_core_web_lg for better accuracy.
    """
    global _nlp_model
    if _nlp_model is None:
        import spacy

        try:
            _nlp_model = spacy.load(model)
        except OSError:
            # Fallback to small model if requested model not available
            try:
                _nlp_model = spacy.load("en_core_web_sm")
            except OSError as e:
                raise RuntimeError(
                    "No spaCy model found. Install with: python -m spacy download en_core_web_sm"
                ) from e
    return _nlp_model


class EntityExtractor:
    """
    Extracts named entities from text using spaCy NER.

    Supports entity types:
    - PERSON: People, including fictional characters
    - ORG: Organizations, companies, agencies
    - PRODUCT: Products, objects, vehicles
    - GPE: Geopolitical entities (countries, cities, states)
    - LOC: Non-GPE locations
    - EVENT: Named events (hurricanes, battles, sports events)
    - WORK_OF_ART: Titles of books, songs, etc.
    - LAW: Named documents made into laws
    - CONCEPT: Custom entity type for key concepts (via noun chunks)
    """

    # Entity types we care about for SEO/content analysis
    RELEVANT_ENTITY_TYPES = {
        "PERSON",
        "ORG",
        "PRODUCT",
        "GPE",
        "LOC",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "NORP",  # Nationalities, religious/political groups
        "FAC",  # Facilities (buildings, airports)
        "MONEY",
        "PERCENT",
        "DATE",
        "TIME",
        "QUANTITY",
    }

    # Map spaCy types to our simplified types
    TYPE_MAPPING = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "PRODUCT": "PRODUCT",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "FAC": "LOCATION",
        "EVENT": "EVENT",
        "WORK_OF_ART": "PRODUCT",
        "LAW": "CONCEPT",
        "NORP": "ORG",
        "MONEY": "CONCEPT",
        "PERCENT": "CONCEPT",
        "DATE": "CONCEPT",
        "TIME": "CONCEPT",
        "QUANTITY": "CONCEPT",
    }

    def __init__(self, model: str = "en_core_web_sm") -> None:
        """
        Initialize the entity extractor.

        Args:
            model: spaCy model name. Options:
                - en_core_web_sm (faster, less accurate)
                - en_core_web_md (balanced)
                - en_core_web_lg (slower, more accurate)
        """
        self._model_name = model
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        """Lazy load the spaCy model."""
        if self._nlp is None:
            self._nlp = _get_nlp(self._model_name)
        return self._nlp

    def extract_entities(self, text: str) -> list[EntityMatch]:
        """
        Extract all named entities from text.

        Args:
            text: The text to analyze

        Returns:
            List of EntityMatch objects with text, type, positions, confidence
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        entities: list[EntityMatch] = []

        for ent in doc.ents:
            if ent.label_ in self.RELEVANT_ENTITY_TYPES:
                entity_type = self.TYPE_MAPPING.get(ent.label_, "CONCEPT")
                entities.append(
                    EntityMatch(
                        text=ent.text,
                        entity_type=entity_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=1.0,  # spaCy doesn't provide confidence scores
                    )
                )

        return entities

    def extract_entities_with_concepts(
        self, text: str, include_noun_chunks: bool = True
    ) -> list[EntityMatch]:
        """
        Extract named entities plus important noun chunks as concepts.

        This provides richer semantic coverage by including key noun phrases
        that aren't formal named entities but are topically important.

        Args:
            text: The text to analyze
            include_noun_chunks: Whether to include noun chunks as CONCEPT type

        Returns:
            List of EntityMatch objects
        """
        entities = self.extract_entities(text)

        if not include_noun_chunks:
            return entities

        if not text or not text.strip():
            return entities

        doc = self.nlp(text)

        # Get positions of existing entities to avoid duplicates
        existing_spans = {(e.start_char, e.end_char) for e in entities}

        for chunk in doc.noun_chunks:
            # Skip short chunks (articles, pronouns, etc.)
            if len(chunk.text.split()) < 2:
                continue

            # Skip if overlaps with existing entity
            if (chunk.start_char, chunk.end_char) in existing_spans:
                continue

            # Skip if chunk is contained within an existing entity
            is_contained = any(
                chunk.start_char >= e.start_char and chunk.end_char <= e.end_char
                for e in entities
            )
            if is_contained:
                continue

            entities.append(
                EntityMatch(
                    text=chunk.text,
                    entity_type="CONCEPT",
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    confidence=0.8,  # Lower confidence for noun chunks
                )
            )

        # Sort by position
        entities.sort(key=lambda e: e.start_char)
        return entities

    def match_expected_entities(
        self, extracted: list[EntityMatch], expected: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        Match extracted entities against expected semantic entities.

        Uses fuzzy matching to handle variations in naming.

        Args:
            extracted: List of extracted EntityMatch objects
            expected: List of expected entity strings

        Returns:
            Tuple of (found_entities, missing_entities)
        """
        if not expected:
            return [], []

        # Normalize extracted entity texts for matching
        extracted_texts = {self._normalize(e.text) for e in extracted}
        extracted_texts_lower = {t.lower() for t in extracted_texts}

        found: list[str] = []
        missing: list[str] = []

        for expected_entity in expected:
            normalized = self._normalize(expected_entity)

            # Exact match (case-insensitive)
            if normalized.lower() in extracted_texts_lower:
                found.append(expected_entity)
                continue

            # Check if expected is contained in any extracted
            is_found = False
            for ext_text in extracted_texts:
                if self._fuzzy_match(normalized, ext_text):
                    found.append(expected_entity)
                    is_found = True
                    break

            if not is_found:
                missing.append(expected_entity)

        return found, missing

    def calculate_entity_gap(
        self, extracted: list[EntityMatch], competitor_entities: list[str]
    ) -> list[str]:
        """
        Identify entities present in competitors but missing from content.

        This helps identify content gaps that could improve SEO.

        Args:
            extracted: Entities extracted from the analyzed content
            competitor_entities: Entities found in competitor content

        Returns:
            List of entities missing from the content
        """
        _, missing = self.match_expected_entities(extracted, competitor_entities)
        return missing

    def get_entity_density(self, text: str, entities: list[EntityMatch]) -> float:
        """
        Calculate entity density as a ratio of entity characters to total.

        Args:
            text: The original text
            entities: List of extracted entities

        Returns:
            Ratio of entity coverage (0.0 to 1.0)
        """
        if not text or not entities:
            return 0.0

        total_chars = len(text)
        entity_chars = sum(e.length for e in entities)

        return entity_chars / total_chars

    def get_unique_entities(self, entities: list[EntityMatch]) -> list[EntityMatch]:
        """
        Deduplicate entities by normalized text.

        Args:
            entities: List of entities (may contain duplicates)

        Returns:
            List of unique entities (first occurrence kept)
        """
        seen: set[str] = set()
        unique: list[EntityMatch] = []

        for entity in entities:
            normalized = self._normalize(entity.text).lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(entity)

        return unique

    def group_by_type(
        self, entities: list[EntityMatch]
    ) -> dict[str, list[EntityMatch]]:
        """
        Group entities by their type.

        Args:
            entities: List of entities

        Returns:
            Dictionary mapping entity type to list of entities
        """
        groups: dict[str, list[EntityMatch]] = {}

        for entity in entities:
            if entity.entity_type not in groups:
                groups[entity.entity_type] = []
            groups[entity.entity_type].append(entity)

        return groups

    @staticmethod
    @lru_cache(maxsize=1024)
    def _normalize(text: str) -> str:
        """
        Normalize text for comparison.

        Removes extra whitespace and normalizes unicode.
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        return text

    @staticmethod
    def _fuzzy_match(s1: str, s2: str, threshold: float = 0.8) -> bool:
        """
        Check if two strings are a fuzzy match.

        Uses simple containment and token overlap for speed.

        Args:
            s1: First string
            s2: Second string
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            True if strings are a fuzzy match
        """
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Check containment
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return True

        # Check token overlap
        tokens1 = set(s1_lower.split())
        tokens2 = set(s2_lower.split())

        if not tokens1 or not tokens2:
            return False

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        jaccard = len(intersection) / len(union)
        return jaccard >= threshold


def extract_entities_simple(text: str) -> list[EntityMatch]:
    """
    Simple function to extract entities without class instantiation.

    Convenience function for one-off entity extraction.

    Args:
        text: Text to analyze

    Returns:
        List of EntityMatch objects
    """
    extractor = EntityExtractor()
    return extractor.extract_entities(text)

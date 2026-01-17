"""
Entity Enricher - Semantic Depth Enhancement

Responsibilities:
- Inject missing entities identified in analysis
- Add entity context (definitions, relationships)
- Build semantic triples (Subject-Predicate-Object)

Reference: docs/research/Content_Scoring_and_Quality_Framework.docx
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from seo_optimizer.ingestion.models import ContentNode, DocumentAST, NodeType

from .guardrails import SafetyGuardrails
from .models import ChangeType, OptimizationChange, OptimizationConfig

if TYPE_CHECKING:
    from seo_optimizer.analysis.models import SemanticScore


# =============================================================================
# Entity Types and Templates
# =============================================================================

ENTITY_CONTEXT_TEMPLATES = {
    "PERSON": [
        "{entity}, a recognized expert in the field,",
        "According to {entity},",
        "As {entity} notes,",
    ],
    "ORG": [
        "{entity}, a leading company in this space,",
        "Tools like {entity}",
        "{entity} offers",
    ],
    "PRODUCT": [
        "{entity} is a popular solution that",
        "Products like {entity}",
        "Using {entity},",
    ],
    "CONCEPT": [
        "{entity} refers to",
        "The concept of {entity}",
        "{entity} is an important factor in",
    ],
    "LOCATION": [
        "In {entity},",
        "Regions like {entity}",
        "{entity} has seen",
    ],
}

# Relationship predicates for semantic triples
SEMANTIC_PREDICATES = [
    "integrates with",
    "is used by",
    "complements",
    "works alongside",
    "enhances",
    "supports",
    "enables",
    "provides",
]


class EntityEnricher:
    """
    Enriches content with missing semantic entities.

    Adds entities naturally to improve topical coverage
    and semantic depth for both SEO and AI retrieval.
    """

    def __init__(
        self, config: OptimizationConfig, guardrails: SafetyGuardrails
    ) -> None:
        """Initialize the entity enricher."""
        self.config = config
        self.guardrails = guardrails

    def enrich(
        self,
        ast: DocumentAST,
        semantic_analysis: SemanticScore | None = None,
    ) -> list[OptimizationChange]:
        """
        Add missing entities to improve topical coverage.

        Args:
            ast: The document AST
            semantic_analysis: Pre-computed semantic analysis

        Returns:
            List of optimization changes
        """
        if not self.config.inject_entities:
            return []

        changes: list[OptimizationChange] = []

        # Get missing entities from analysis or config
        missing_entities = self._get_missing_entities(ast, semantic_analysis)

        if not missing_entities:
            return changes

        # Check entity density before adding
        all_entities = self.config.semantic_entities + list(missing_entities)
        density_check = self.guardrails.check_entity_density(ast.full_text, all_entities)

        if not density_check.is_safe:
            return changes  # Already too dense

        # Inject missing entities
        paragraphs = [
            node for node in ast.nodes if node.node_type == NodeType.PARAGRAPH
        ]

        for i, entity in enumerate(list(missing_entities)[:5]):  # Limit to 5
            if i >= len(paragraphs):
                break

            # Choose appropriate paragraph
            target_para = paragraphs[min(i * 2, len(paragraphs) - 1)]

            change = self._inject_entity_mention(target_para, entity)
            if change:
                changes.append(change)

                if len(changes) >= self.config.max_changes_per_section:
                    break

        # Add entity context for first mention
        context_changes = self._add_entity_context(ast, list(missing_entities)[:3])
        changes.extend(context_changes)

        return changes

    def _get_missing_entities(
        self,
        ast: DocumentAST,
        semantic_analysis: SemanticScore | None,
    ) -> set[str]:
        """
        Get entities that should be added to content.

        Args:
            ast: Document AST
            semantic_analysis: Pre-computed analysis

        Returns:
            Set of missing entity names
        """
        # Start with configured semantic entities
        expected = set(self.config.semantic_entities)

        # Add missing entities from analysis
        if semantic_analysis and semantic_analysis.missing_entities:
            expected.update(semantic_analysis.missing_entities)

        # Filter out already present entities
        full_text_lower = ast.full_text.lower()
        missing = {e for e in expected if e.lower() not in full_text_lower}

        return missing

    def _inject_entity_mention(
        self, node: ContentNode, entity: str
    ) -> OptimizationChange | None:
        """
        Add entity mention to a paragraph.

        Args:
            node: The paragraph node
            entity: Entity name to add

        Returns:
            Change if successful, None otherwise
        """
        original = node.text_content

        if entity.lower() in original.lower():
            return None  # Already present

        # Try to inject naturally
        modified = self._insert_entity_naturally(original, entity)

        if modified and modified != original:
            # Apply AI vocabulary filter
            check = self.guardrails.filter_ai_vocabulary(modified)
            modified = check.cleaned_text

            return OptimizationChange(
                change_type=ChangeType.ENTITY,
                location=f"Paragraph: {original[:30]}...",
                original=original[:80] + "..." if len(original) > 80 else original,
                optimized=modified[:80] + "..." if len(modified) > 80 else modified,
                reason=f"Added semantic entity '{entity}'",
                impact_score=2.5,
                section_id=node.node_id,
            )

        return None

    def _insert_entity_naturally(self, text: str, entity: str) -> str | None:
        """
        Insert entity mention naturally into text.

        Uses several strategies based on entity type and context.
        """
        # Strategy 1: Add as example in parentheses
        # "These tools are useful" → "These tools (like {entity}) are useful"
        tool_patterns = [
            (r"(tools|solutions|platforms|software|services)", r"\1 (like {entity})"),
            (r"(methods|approaches|techniques|strategies)", r"\1 (such as {entity})"),
            (r"(companies|organizations|businesses)", r"\1 (including {entity})"),
        ]

        for pattern, replacement in tool_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                modified = re.sub(
                    pattern,
                    replacement.format(entity=entity),
                    text,
                    count=1,
                    flags=re.IGNORECASE,
                )
                if modified != text:
                    return modified

        # Strategy 2: Add as introductory clause
        sentences = re.split(r"([.!?]+\s*)", text)
        if len(sentences) >= 2:
            # Find a suitable sentence to prepend context
            for i in range(0, len(sentences), 2):
                if sentences[i].strip():
                    # Add entity as context
                    prepend = f"With tools like {entity}, "
                    sentences[i] = prepend + sentences[i][0].lower() + sentences[i][1:]
                    return "".join(sentences)

        # Strategy 3: Add at end as example
        clean_text = text.rstrip(".")
        return f"{clean_text}, similar to {entity}."

    def _add_entity_context(
        self, ast: DocumentAST, entities: list[str]
    ) -> list[OptimizationChange]:
        """
        Add contextual information when entity is first mentioned.

        For example: "BERT (Google's language understanding model)..."
        """
        changes: list[OptimizationChange] = []

        # Get known entity contexts
        entity_definitions = self._get_entity_definitions()

        for entity in entities:
            if entity in entity_definitions:
                definition = entity_definitions[entity]

                # Find where to add context
                for node in ast.nodes:
                    if node.node_type != NodeType.PARAGRAPH:
                        continue

                    if entity in node.text_content:
                        # Already mentioned - add context
                        original = node.text_content
                        modified = self._add_inline_context(original, entity, definition)

                        if modified and modified != original:
                            changes.append(
                                OptimizationChange(
                                    change_type=ChangeType.ENTITY,
                                    location=f"Paragraph: {original[:30]}...",
                                    original=original[:80] + "...",
                                    optimized=modified[:80] + "...",
                                    reason=f"Added context for entity '{entity}'",
                                    impact_score=1.5,
                                    section_id=node.node_id,
                                )
                            )
                            break

        return changes

    def _add_inline_context(
        self, text: str, entity: str, definition: str
    ) -> str | None:
        """
        Add inline context after first entity mention.

        Example: "BERT" → "BERT (Google's language model)"
        """
        # Check if context already exists
        pattern = rf"{re.escape(entity)}\s*\([^)]+\)"
        if re.search(pattern, text, re.IGNORECASE):
            return None  # Context already present

        # Add context after first mention
        pattern = rf"\b{re.escape(entity)}\b"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            insert_pos = match.end()
            modified = f"{text[:insert_pos]} ({definition}){text[insert_pos:]}"
            return modified

        return None

    def _get_entity_definitions(self) -> dict[str, str]:
        """
        Get known entity definitions for context.

        In a full implementation, this would use a knowledge base.
        """
        # Common SEO/AI entities with definitions
        return {
            "BERT": "Google's language understanding model",
            "GPT": "OpenAI's generative pre-trained transformer",
            "E-E-A-T": "Experience, Expertise, Authoritativeness, Trustworthiness",
            "SERP": "Search Engine Results Page",
            "CTR": "Click-Through Rate",
            "SEO": "Search Engine Optimization",
            "NLP": "Natural Language Processing",
            "RAG": "Retrieval-Augmented Generation",
            "LLM": "Large Language Model",
            "API": "Application Programming Interface",
            "JSON-LD": "JavaScript Object Notation for Linked Data",
            "Schema.org": "structured data vocabulary",
            "Knowledge Graph": "Google's information database",
        }

    def build_semantic_triple(
        self, subject: str, predicate: str, obj: str
    ) -> str:
        """
        Generate sentence establishing entity relationship.

        Creates a semantic triple that helps search engines
        understand entity relationships.

        Args:
            subject: The subject entity
            predicate: The relationship verb
            obj: The object entity

        Returns:
            Complete sentence expressing the relationship
        """
        # Ensure proper capitalization
        subject = subject.strip()
        obj = obj.strip()
        predicate = predicate.lower().strip()

        # Build the sentence
        sentence = f"{subject} {predicate} {obj}"

        # Add context based on predicate type
        if predicate in ["integrates with", "works alongside"]:
            sentence += ", enabling seamless workflows"
        elif predicate in ["enhances", "supports"]:
            sentence += " to improve overall effectiveness"
        elif predicate == "provides":
            sentence += " for users seeking reliable solutions"

        return sentence + "."

    def generate_relationship_sentences(
        self, entities: list[str]
    ) -> list[str]:
        """
        Generate sentences that establish relationships between entities.

        Args:
            entities: List of entity names

        Returns:
            List of relationship sentences
        """
        if len(entities) < 2:
            return []

        sentences: list[str] = []

        # Generate pairwise relationships
        for i in range(len(entities) - 1):
            subject = entities[i]
            obj = entities[i + 1]

            # Choose an appropriate predicate
            predicate_index = i % len(SEMANTIC_PREDICATES)
            predicate = SEMANTIC_PREDICATES[predicate_index]

            sentence = self.build_semantic_triple(subject, predicate, obj)
            sentences.append(sentence)

        return sentences

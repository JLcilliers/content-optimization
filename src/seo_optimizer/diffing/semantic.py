"""
Semantic Similarity - Detects when text is reworded (not new).

Uses sentence-transformers to compute semantic similarity between
text segments. High similarity means the content is the same meaning,
just reworded - and should NOT be highlighted as new.

Key thresholds (research-validated):
- >= 0.85 cosine similarity: Same content, different words (DON'T highlight)
- > 0.90 similarity: Definitely redundant/duplicate (DON'T highlight)
- < 0.85 similarity: Genuinely different content (MAY highlight)

Per Content_Scoring research:
- "≥ 0.85 = Excellent coverage" / semantically equivalent
- "> 0.90 similarity between sections = Redundancy penalty"

Reference: docs/research/05-diffing-highlighting.md section 4
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Research-validated thresholds
SEMANTIC_EQUIVALENCE_THRESHOLD = 0.85
REDUNDANCY_THRESHOLD = 0.90
CONSERVATIVE_THRESHOLD = 0.80


# Global model cache for lazy loading
_model: Any | None = None
_model_name: str = "all-MiniLM-L6-v2"


def _get_model() -> Any:
    """
    Lazy load the sentence transformer model.

    Returns:
        Loaded SentenceTransformer model
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_model_name)
    return _model


class SemanticMatcher:
    """
    Embedding-based semantic similarity per research specs.

    Uses sentence-transformers to detect when content has the same
    meaning but different words - which should NOT be highlighted.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        equivalence_threshold: float = SEMANTIC_EQUIVALENCE_THRESHOLD,
        redundancy_threshold: float = REDUNDANCY_THRESHOLD,
    ) -> None:
        """
        Initialize semantic matcher.

        Args:
            model_name: Name of sentence-transformer model
            equivalence_threshold: Threshold for semantic equivalence (default 0.85)
            redundancy_threshold: Threshold for redundancy detection (default 0.90)
        """
        self._model: Any | None = None
        self._model_name = model_name
        self._embedding_cache: dict[str, NDArray[np.floating[Any]]] = {}
        self.equivalence_threshold = equivalence_threshold
        self.redundancy_threshold = redundancy_threshold

    @property
    def model(self) -> Any:
        """
        Lazy initialization to avoid loading on import.

        Returns:
            Loaded SentenceTransformer model
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def get_embedding(self, text: str) -> NDArray[np.floating[Any]]:
        """
        Get embedding with caching.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        if text not in self._embedding_cache:
            embedding = self.model.encode(text, convert_to_numpy=True)
            self._embedding_cache[text] = embedding
        return self._embedding_cache[text]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Handle empty strings
        if not text1.strip() or not text2.strip():
            return 0.0 if text1.strip() != text2.strip() else 1.0

        # Exact match optimization
        if text1.strip().lower() == text2.strip().lower():
            return 1.0

        # Very different lengths unlikely to be semantic matches
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        if len_ratio < 0.3:
            return 0.0

        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)

    def find_best_match(
        self,
        candidate: str,
        original_chunks: list[str],
    ) -> tuple[int, float]:
        """
        Find the original chunk most similar to candidate.

        Args:
            candidate: Text to match
            original_chunks: List of original text chunks

        Returns:
            Tuple of (index, similarity_score)
            Returns (-1, 0.0) if no chunks provided
        """
        if not original_chunks:
            return -1, 0.0

        best_idx = -1
        best_score = 0.0

        for idx, chunk in enumerate(original_chunks):
            similarity = self.compute_similarity(candidate, chunk)
            if similarity > best_score:
                best_score = similarity
                best_idx = idx

        return best_idx, best_score

    def is_semantically_equivalent(
        self,
        text1: str,
        text2: str,
        threshold: float | None = None,
    ) -> bool:
        """
        Check if two texts are semantically equivalent.

        Per Content_Scoring research:
        "≥ 0.85 = Excellent coverage" → semantically equivalent

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (default: equivalence_threshold)

        Returns:
            True if texts are semantically equivalent
        """
        threshold = threshold if threshold is not None else self.equivalence_threshold
        return self.compute_similarity(text1, text2) >= threshold

    def is_redundant(
        self,
        text1: str,
        text2: str,
        threshold: float | None = None,
    ) -> bool:
        """
        Check if two texts are redundant (near-duplicates).

        Per Content_Scoring research:
        "> 0.90 similarity between sections = Redundancy penalty"

        Args:
            text1: First text
            text2: Second text
            threshold: Redundancy threshold (default: redundancy_threshold)

        Returns:
            True if texts are redundant
        """
        threshold = threshold if threshold is not None else self.redundancy_threshold
        return self.compute_similarity(text1, text2) > threshold

    def compute_batch_similarity(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
    ) -> list[float]:
        """
        Compute similarity for multiple pairs efficiently.

        Batches embedding computation for better performance.

        Args:
            texts_a: List of first texts
            texts_b: List of second texts (same length as texts_a)

        Returns:
            List of similarity scores

        Raises:
            ValueError: If lists have different lengths
        """
        if len(texts_a) != len(texts_b):
            raise ValueError(
                f"Lists must have same length: {len(texts_a)} vs {len(texts_b)}"
            )

        if not texts_a:
            return []

        # Batch encode all texts
        all_texts = list(texts_a) + list(texts_b)
        embeddings = self.model.encode(all_texts, convert_to_numpy=True)

        # Split embeddings
        n = len(texts_a)
        embeddings_a = embeddings[:n]
        embeddings_b = embeddings[n:]

        # Compute cosine similarities
        similarities: list[float] = []
        for emb_a, emb_b in zip(embeddings_a, embeddings_b, strict=True):
            dot_product = np.dot(emb_a, emb_b)
            norm_a = np.linalg.norm(emb_a)
            norm_b = np.linalg.norm(emb_b)

            if norm_a == 0 or norm_b == 0:
                similarities.append(0.0)
            else:
                similarities.append(float(dot_product / (norm_a * norm_b)))

        return similarities

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()


# Global semantic matcher instance for convenience functions
_default_matcher: SemanticMatcher | None = None


def _get_default_matcher() -> SemanticMatcher:
    """Get or create the default semantic matcher."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = SemanticMatcher()
    return _default_matcher


def compute_semantic_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Compute semantic similarity between two text segments.

    Uses sentence-transformers embeddings and cosine similarity.

    Args:
        text_a: First text segment
        text_b: Second text segment

    Returns:
        Similarity score between 0.0 and 1.0
        - 1.0 = identical meaning
        - 0.0 = completely different meaning

    Example:
        >>> score = compute_semantic_similarity(
        ...     "The product helps users",
        ...     "The product assists customers"
        ... )
        >>> # score ≈ 0.92 (very similar meaning)
    """
    return _get_default_matcher().compute_similarity(text_a, text_b)


def compute_batch_similarity(
    texts_a: Sequence[str],
    texts_b: Sequence[str],
) -> list[float]:
    """
    Compute semantic similarity for multiple pairs efficiently.

    Batches embedding computation for better performance.

    Args:
        texts_a: List of first texts
        texts_b: List of second texts (same length as texts_a)

    Returns:
        List of similarity scores

    Raises:
        ValueError: If lists have different lengths
    """
    return _get_default_matcher().compute_batch_similarity(texts_a, texts_b)


def is_semantic_match(
    text_a: str,
    text_b: str,
    threshold: float = SEMANTIC_EQUIVALENCE_THRESHOLD,
) -> tuple[bool, float]:
    """
    Determine if two texts are semantically equivalent.

    This is the key function for preventing false positives.
    If text is reworded but means the same thing, we should
    NOT highlight it as new content.

    Args:
        text_a: First text
        text_b: Second text
        threshold: Similarity threshold (default 0.85)

    Returns:
        Tuple of (is_match, similarity_score)
        - is_match: True if similarity >= threshold
        - similarity_score: The actual similarity

    Example:
        >>> is_match, score = is_semantic_match(
        ...     "The product assists users",
        ...     "The product helps customers"
        ... )
        >>> # is_match = True, score = 0.92
        >>> # This means DON'T highlight - it's not new content
    """
    similarity = _get_default_matcher().compute_similarity(text_a, text_b)
    return similarity >= threshold, similarity


def get_embedding(text: str) -> list[float]:
    """
    Get the embedding vector for a text segment.

    Uses sentence-transformers to encode text into a dense vector.

    Args:
        text: Text to encode

    Returns:
        List of floats representing the embedding

    Note:
        Prefer compute_batch_similarity for multiple comparisons
        as it's more efficient.
    """
    embedding = _get_default_matcher().get_embedding(text)
    result: list[float] = embedding.tolist()
    return result


def preload_model() -> None:
    """
    Preload the sentence-transformers model.

    Call this during startup to avoid latency on first comparison.
    The model is loaded lazily by default.
    """
    _ = _get_default_matcher().model


def is_semantically_equivalent(
    text1: str,
    text2: str,
    threshold: float = SEMANTIC_EQUIVALENCE_THRESHOLD,
) -> bool:
    """
    Check if two texts are semantically equivalent.

    Convenience function for simple boolean check.

    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (default 0.85)

    Returns:
        True if texts are semantically equivalent
    """
    return _get_default_matcher().is_semantically_equivalent(text1, text2, threshold)


def is_redundant_content(
    text1: str,
    text2: str,
    threshold: float = REDUNDANCY_THRESHOLD,
) -> bool:
    """
    Check if two texts are redundant (near-duplicates).

    Per Content_Scoring research:
    "> 0.90 similarity between sections = Redundancy penalty"

    Args:
        text1: First text
        text2: Second text
        threshold: Redundancy threshold (default 0.90)

    Returns:
        True if texts are redundant
    """
    return _get_default_matcher().is_redundant(text1, text2, threshold)

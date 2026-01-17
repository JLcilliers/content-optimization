# Section 7: Safety Guardrails
## Technical Specification for SEO Content Optimization Tool

**Document Version:** 2.0
**Date:** January 16, 2026
**Status:** Active Specification

---

## 1. Executive Summary

Safety guardrails form the critical defensive layer that ensures content optimization improves SEO performance without introducing errors, manipulation signals, or brand inconsistencies. This specification defines four primary protection domains: over-optimization detection, factual grounding verification, brand voice validation, and highlight accuracy verification. Each domain includes concrete thresholds, scoring formulas, and decision logic suitable for production implementation.

The guardrail system operates on a principle of graduated intervention. Minor issues receive automated fixes, moderate concerns trigger warnings, significant changes require mandatory human review, and critical violations cause automatic rejection. This approach balances optimization efficiency with content safety, ensuring the tool enhances content without introducing harm. The system is designed for a document-in, document-out workflow where all new content additions must be highlighted in green while preserving original content unchanged.

Modern search engines have effectively closed loopholes for manipulative optimization tactics. Keyword stuffing, unnatural phrasing, and aggressive link manipulation now trigger algorithmic penalties rather than ranking improvements. Simultaneously, the proliferation of AI-generated content has heightened the importance of factual accuracy verification and hallucination detection. This specification addresses these challenges through a comprehensive validation pipeline that runs after content generation and before output creation.

---

## 2. Over-Optimization Checks

### 2.1 Density Limits

The system monitors three types of keyword density to detect manipulation while avoiding false positives on naturally keyword-rich content.

#### 2.1.1 Density Threshold Matrix

| Density Type | Calculation | Warning Threshold | Block Threshold | Revert Threshold |
|--------------|-------------|-------------------|-----------------|------------------|
| Exact Match | `(exact_keyword_count / total_words) * 100` | > 2.5% | > 4.0% | > 5.0% |
| Phrase Match | `(phrase_variants_count / total_words) * 100` | > 4.0% | > 6.0% | > 8.0% |
| Semantic Cluster | `(semantic_related_count / total_words) * 100` | > 8.0% | > 12.0% | > 15.0% |
| Combined Footprint | `(all_keyword_variants / total_words) * 100` | > 10.0% | > 15.0% | > 18.0% |

#### 2.1.2 Context-Aware Adjustments

Short-form content (< 300 words) tolerates higher densities due to limited space:

```
adjusted_threshold = base_threshold * min(1.5, 1 + (300 - word_count) / 1000)
```

Long-form content (> 2000 words) should demonstrate natural density decay:

```
expected_density_at_position = initial_density * (1 - (position / total_length) * 0.3)
violation = actual_density > expected_density_at_position * 1.2
```

### 2.2 Pattern Detection Rules

#### 2.2.1 Unnatural Repetition Detection

```
DETECTION RULES:
1. Repetition Distance Check:
   avg_distance = total_words_between_occurrences / (occurrence_count - 1)
   variance = standard_deviation(distances)

   IF avg_distance < 50 AND variance < 10 THEN
       flag = "SUSPICIOUSLY_REGULAR_SPACING"
       severity = HIGH

2. Chi-Square Uniformity Test:
   expected_distribution = uniform across document sections
   observed_distribution = actual keyword positions
   chi_square = sum((observed - expected)^2 / expected)

   IF chi_square_p_value > 0.95 THEN
       flag = "ARTIFICIALLY_UNIFORM_DISTRIBUTION"
       severity = MEDIUM

3. Section Concentration Check:
   section_densities = [density for each h2 section]
   max_density = max(section_densities)
   avg_density = mean(section_densities)

   IF max_density > avg_density * 2.5 THEN
       flag = "KEYWORD_CLUSTERING"
       severity = MEDIUM
```

#### 2.2.2 Awkward Insertion Detection

Perplexity-based detection identifies keyword insertions that disrupt natural sentence flow:

```python
def detect_awkward_insertions(content: str, keywords: List[str]) -> List[Alert]:
    alerts = []
    sentences = split_sentences(content)

    for sentence in sentences:
        base_perplexity = calculate_perplexity(sentence)

        for keyword in keywords:
            if keyword.lower() in sentence.lower():
                # Remove keyword and recalculate
                sentence_without = remove_keyword(sentence, keyword)
                reduced_perplexity = calculate_perplexity(sentence_without)

                perplexity_delta = base_perplexity - reduced_perplexity

                if perplexity_delta > 15:
                    alerts.append(Alert(
                        type="AWKWARD_INSERTION",
                        sentence=sentence,
                        keyword=keyword,
                        perplexity_delta=perplexity_delta,
                        severity="HIGH" if perplexity_delta > 25 else "MEDIUM"
                    ))

    return alerts
```

#### 2.2.3 Grammatical Pattern Anomalies

| Pattern | Detection Rule | Severity |
|---------|----------------|----------|
| Prepositional phrase stuffing | Count consecutive prepositional phrases > 2 | HIGH |
| Adjective stacking | Count consecutive adjectives > 3 | MEDIUM |
| Comma-separated keyword lists | Detect list patterns with > 50% keyword overlap | HIGH |
| Unnatural word order | Parse tree analysis for non-standard SVO structures | HIGH |
| Forced keyword as subject | Keyword appears as sentence subject > 40% of occurrences | MEDIUM |

### 2.3 Composite Keyword Stuffing Score

```
keyword_stuffing_score = (
    (density_score * 0.35) +
    (repetition_pattern_score * 0.25) +
    (perplexity_delta_score * 0.25) +
    (grammatical_anomaly_score * 0.15)
) * content_type_modifier

WHERE:
    density_score = min(100, (combined_footprint / block_threshold) * 100)
    repetition_pattern_score = chi_square_uniformity_score * 100
    perplexity_delta_score = min(100, avg(sentence_perplexity_deltas) * 5)
    grammatical_anomaly_score = (anomaly_count / sentence_count) * 100

    content_type_modifier:
        Product pages: 0.8 (higher tolerance expected)
        Blog posts: 1.0 (standard)
        Landing pages: 0.9
        YMYL content: 1.2 (stricter)
        Technical docs: 0.85
```

### 2.4 Action Thresholds

| Score Range | Action | System Behavior |
|-------------|--------|-----------------|
| 0-30 | PASS | No intervention required |
| 31-50 | WARN | Log warning, continue with flag |
| 51-70 | BLOCK | Halt optimization, require human review |
| 71-100 | REVERT | Reject changes, restore original content |

---

## 3. Factual Grounding Verification

### 3.1 Source Text Comparison

All generated content (especially FAQ answers) must be grounded in the source document or explicitly flagged for human verification.

#### 3.1.1 Grounding Score Calculation

```python
def calculate_grounding_score(
    generated_text: str,
    source_content: str,
    brand_context: Optional[str] = None
) -> GroundingResult:

    # Extract claims from generated text
    claims = extract_verifiable_claims(generated_text)

    grounding_scores = []
    for claim in claims:
        # Semantic search in source content
        source_matches = semantic_search(
            query=claim.text,
            corpus=source_content,
            top_k=3
        )

        # Calculate support score
        max_similarity = max(m.score for m in source_matches) if source_matches else 0

        # Check brand context if available
        brand_support = 0
        if brand_context:
            brand_matches = semantic_search(claim.text, brand_context, top_k=2)
            brand_support = max(m.score for m in brand_matches) if brand_matches else 0

        # Combined grounding score
        claim_grounding = max(max_similarity * 0.7 + brand_support * 0.3, max_similarity)

        grounding_scores.append(ClaimGrounding(
            claim=claim,
            source_support=max_similarity,
            brand_support=brand_support,
            combined_score=claim_grounding,
            supporting_passages=[m.text for m in source_matches[:2]]
        ))

    # Aggregate score
    overall_grounding = mean([c.combined_score for c in grounding_scores])

    return GroundingResult(
        overall_score=overall_grounding,
        claim_details=grounding_scores,
        ungrounded_claims=[c for c in grounding_scores if c.combined_score < 0.6]
    )
```

#### 3.1.2 Grounding Thresholds

| Grounding Score | Classification | Action |
|-----------------|----------------|--------|
| >= 0.85 | Fully Grounded | Auto-approve |
| 0.70 - 0.84 | Mostly Grounded | Approve with logging |
| 0.50 - 0.69 | Partially Grounded | Human review required |
| 0.30 - 0.49 | Weakly Grounded | Block, require source |
| < 0.30 | Ungrounded | Reject as hallucination |

### 3.2 Claim Extraction and Matching

#### 3.2.1 Claim Types to Extract

| Claim Type | Example Pattern | Extraction Method | Verification Priority |
|------------|-----------------|-------------------|----------------------|
| Statistical | "X% of users", "studies show" | Regex + NER | CRITICAL |
| Comparative | "faster than", "more effective" | Comparative adjective detection | HIGH |
| Absolute | "the best", "only solution" | Superlative detection | MEDIUM |
| Temporal | "as of 2026", "recently" | Date + temporal marker detection | HIGH |
| Attributed | "According to [source]" | Attribution phrase detection | CRITICAL |
| Causal | "causes", "results in" | Causal language patterns | HIGH |
| Factual | Specific names, numbers, dates | NER (PERSON, ORG, DATE, MONEY) | CRITICAL |

#### 3.2.2 Claim Extraction Algorithm

```python
CLAIM_INDICATORS = [
    r"studies show", r"research indicates", r"according to",
    r"data suggests", r"evidence shows", r"experts agree",
    r"proven to", r"results in", r"leads to", r"causes",
    r"\d+%\s+of", r"compared to", r"more than", r"less than",
    r"the (best|worst|only|first|largest|smallest)"
]

def extract_verifiable_claims(content: str) -> List[Claim]:
    claims = []
    sentences = split_sentences(content)

    for sentence in sentences:
        # Check for claim indicators
        for indicator in CLAIM_INDICATORS:
            if re.search(indicator, sentence, re.IGNORECASE):
                claims.append(Claim(
                    text=sentence,
                    type=classify_claim_type(sentence, indicator),
                    indicator_matched=indicator,
                    entities=extract_entities(sentence),
                    requires_source=needs_source_attribution(sentence)
                ))
                break

        # Also extract sentences with named entities
        entities = extract_entities(sentence)
        critical_entities = [e for e in entities if e.type in
                           ['PERSON', 'ORG', 'MONEY', 'PERCENT', 'DATE']]
        if critical_entities and sentence not in [c.text for c in claims]:
            claims.append(Claim(
                text=sentence,
                type="factual",
                entities=critical_entities,
                requires_source=len(critical_entities) > 1
            ))

    return claims
```

### 3.3 Hallucination Detection Signals

#### 3.3.1 Red Flag Patterns

```
HALLUCINATION_SIGNALS:

1. Specificity Without Source:
   - Exact statistics without citation ("73.2% of customers")
   - Named individuals without context
   - Specific dates for events not in source

2. Contradiction Detection:
   - Generated claim contradicts source content
   - Sentiment inversion (positive claim about negative topic)
   - Entity relationship mismatch

3. Knowledge Beyond Source:
   - Technical details not present in source
   - Industry benchmarks not provided
   - Competitor comparisons without basis

4. Temporal Impossibilities:
   - Future claims presented as fact
   - Historical claims inconsistent with dates
   - "Recent" claims about old information
```

#### 3.3.2 Hallucination Risk Score

```
hallucination_risk = (
    (1 - grounding_score) * 0.40 +
    specificity_without_source_penalty * 0.25 +
    contradiction_penalty * 0.20 +
    knowledge_extension_penalty * 0.15
)

WHERE:
    specificity_without_source_penalty:
        0.0 if no specific claims without source
        0.5 if 1-2 unattributed specific claims
        1.0 if 3+ unattributed specific claims

    contradiction_penalty:
        0.0 if no contradictions detected
        0.5 if minor contradiction (sentiment)
        1.0 if factual contradiction

    knowledge_extension_penalty:
        0.0 if all info traceable to source
        0.3 if reasonable inference from source
        0.7 if moderate extension beyond source
        1.0 if substantial unsourced content

THRESHOLDS:
    risk < 0.15: LOW (auto-approve)
    risk 0.15-0.35: MODERATE (approve with flag)
    risk 0.35-0.55: HIGH (human review required)
    risk > 0.55: CRITICAL (reject)
```

### 3.4 Confidence Scoring for Generated Content

```python
def score_generation_confidence(
    generated_text: str,
    source_content: str,
    generation_metadata: GenerationMetadata
) -> ConfidenceScore:

    # 1. Grounding confidence
    grounding = calculate_grounding_score(generated_text, source_content)

    # 2. Fluency confidence (perplexity-based)
    perplexity = calculate_perplexity(generated_text)
    fluency_score = normalize_perplexity(perplexity)  # Lower perplexity = higher score

    # 3. Consistency confidence (with brand/source style)
    style_consistency = calculate_style_similarity(generated_text, source_content)

    # 4. Claim verification confidence
    claims = extract_verifiable_claims(generated_text)
    verified_ratio = len([c for c in claims if c.verified]) / len(claims) if claims else 1.0

    # Composite confidence
    confidence = (
        grounding.overall_score * 0.40 +
        fluency_score * 0.20 +
        style_consistency * 0.15 +
        verified_ratio * 0.25
    )

    return ConfidenceScore(
        overall=confidence,
        grounding=grounding.overall_score,
        fluency=fluency_score,
        style=style_consistency,
        verification=verified_ratio,
        requires_review=confidence < 0.75
    )
```

---

## 4. Brand Voice Validation

### 4.1 Voice Profile Comparison

The system compares generated content against a brand voice profile derived from provided brand documents or extracted from source content.

#### 4.1.1 Voice Profile Construction

```python
class BrandVoiceProfile:
    def __init__(self, brand_documents: List[str]):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Build voice centroid from brand documents
        all_sentences = []
        for doc in brand_documents:
            all_sentences.extend(split_sentences(doc))

        embeddings = self.model.encode(all_sentences)
        self.voice_centroid = np.mean(embeddings, axis=0)
        self.voice_std = np.std(embeddings, axis=0)

        # Extract linguistic features
        self.formality_baseline = calculate_formality(brand_documents)
        self.avg_sentence_length = mean([len(s.split()) for s in all_sentences])
        self.vocabulary_profile = extract_vocabulary_profile(brand_documents)
        self.tone_markers = extract_tone_markers(brand_documents)

    def calculate_voice_similarity(self, text: str) -> float:
        text_embedding = self.model.encode(text)
        similarity = cosine_similarity(
            text_embedding.reshape(1, -1),
            self.voice_centroid.reshape(1, -1)
        )[0][0]
        return float(similarity)
```

#### 4.1.2 Voice Similarity Thresholds

| Similarity Score | Classification | Action |
|------------------|----------------|--------|
| >= 0.90 | Excellent Match | Pass |
| 0.85 - 0.89 | Good Match | Pass with logging |
| 0.75 - 0.84 | Acceptable | Warning |
| 0.65 - 0.74 | Drift Detected | Human review required |
| < 0.65 | Significant Drift | Reject |

### 4.2 Tone Consistency Metrics

#### 4.2.1 Formality Score

```
formality_score = (
    (formal_word_ratio * 0.3) +
    (avg_sentence_length_normalized * 0.2) +
    (passive_voice_ratio * 0.15) +
    (complex_sentence_ratio * 0.15) +
    (contraction_absence_score * 0.1) +
    (first_person_absence_score * 0.1)
)

WHERE:
    formal_word_ratio = formal_words / total_words
    avg_sentence_length_normalized = min(1, avg_words_per_sentence / 25)
    passive_voice_ratio = passive_sentences / total_sentences
    complex_sentence_ratio = sentences_with_subordinate_clauses / total_sentences
    contraction_absence_score = 1 - (contractions / potential_contractions)
    first_person_absence_score = 1 - (first_person_pronouns / total_pronouns)

FORMALITY SCALE:
    0.0 - 0.3: Casual/Conversational
    0.3 - 0.5: Informal/Friendly
    0.5 - 0.7: Neutral/Professional
    0.7 - 0.85: Formal/Business
    0.85 - 1.0: Very Formal/Academic
```

#### 4.2.2 Formality Drift Detection

```
formality_drift = abs(source_formality - optimized_formality)

THRESHOLDS:
    drift < 0.10: No concern
    drift 0.10-0.20: Minor drift (log only)
    drift 0.20-0.30: Moderate drift (warning)
    drift > 0.30: Significant drift (human review required)
```

#### 4.2.3 Sentiment Consistency

```python
def check_sentiment_consistency(original: str, optimized: str) -> SentimentResult:
    original_sentiment = analyze_sentiment(original)  # Range: -1 to 1
    optimized_sentiment = analyze_sentiment(optimized)

    shift = optimized_sentiment - original_sentiment

    return SentimentResult(
        original=original_sentiment,
        optimized=optimized_sentiment,
        shift=shift,
        polarity_flip=sign(original_sentiment) != sign(optimized_sentiment),
        action=determine_action(shift)
    )

def determine_action(shift: float) -> str:
    if abs(shift) > 0.4:
        return "REJECT"  # Major sentiment shift
    elif abs(shift) > 0.25:
        return "HUMAN_REVIEW"  # Significant shift
    elif abs(shift) > 0.15:
        return "WARNING"  # Moderate shift
    return "PASS"
```

### 4.3 Terminology Compliance Checks

#### 4.3.1 Vocabulary Compliance Rules

```yaml
terminology_compliance:
  required_terms:
    description: "Terms that must be preserved"
    examples:
      - brand_name: "Must appear exactly as specified"
      - product_names: "Never abbreviate or modify"
      - trademarked_terms: "Include TM/R symbols if present"
    action_on_violation: "BLOCK"

  preferred_terms:
    description: "Terms to use instead of alternatives"
    mappings:
      "customers": ["clients", "users", "buyers"]
      "solutions": ["products", "offerings"]
      "innovative": ["cutting-edge", "revolutionary"]
    action_on_violation: "WARNING"

  banned_terms:
    description: "Terms to never use"
    examples:
      - competitor_names
      - negative_terminology
      - outdated_product_names
      - informal_slang
    action_on_violation: "BLOCK"

  industry_jargon:
    description: "Technical terms requiring context"
    behavior: "flag_if_undefined"
    action: "WARNING"
```

#### 4.3.2 Terminology Violation Detection

```python
def check_terminology_compliance(
    optimized_text: str,
    terminology_rules: TerminologyConfig
) -> ComplianceResult:

    violations = []

    # Check required terms preserved
    for term in terminology_rules.required_terms:
        if term.original in source_text and term.original not in optimized_text:
            violations.append(Violation(
                type="REQUIRED_TERM_MISSING",
                term=term.original,
                severity="HIGH"
            ))

    # Check for banned terms
    for banned in terminology_rules.banned_terms:
        if banned.lower() in optimized_text.lower():
            violations.append(Violation(
                type="BANNED_TERM_USED",
                term=banned,
                severity="HIGH"
            ))

    # Check preferred term usage
    for preferred, alternatives in terminology_rules.preferred_mappings.items():
        for alt in alternatives:
            if alt.lower() in optimized_text.lower():
                violations.append(Violation(
                    type="NON_PREFERRED_TERM",
                    used=alt,
                    preferred=preferred,
                    severity="LOW"
                ))

    return ComplianceResult(
        violations=violations,
        score=1 - (len(violations) / expected_term_count),
        blocks=[v for v in violations if v.severity == "HIGH"]
    )
```

### 4.4 Brand Voice Consistency Score

```
voice_consistency_score = (
    (embedding_similarity * 0.35) +
    (formality_consistency * 0.20) +
    (terminology_compliance * 0.20) +
    (sentiment_consistency * 0.15) +
    (structural_consistency * 0.10)
)

WHERE:
    embedding_similarity = cosine_sim(optimized_embedding, brand_centroid)
    formality_consistency = 1 - abs(source_formality - optimized_formality)
    terminology_compliance = terms_compliant / total_terms_checked
    sentiment_consistency = 1 - abs(sentiment_shift)
    structural_consistency = sentence_length_ratio * paragraph_ratio

THRESHOLDS:
    score >= 0.85: PASS
    score 0.75-0.84: WARNING
    score 0.65-0.74: REVIEW_REQUIRED
    score < 0.65: REJECT
```

---

## 5. Highlight Accuracy Verification

This section defines verification checks that run after the diffing algorithm to ensure highlighting accuracy.

### 5.1 Post-Diff Sanity Checks

#### 5.1.1 Verification Checklist

```python
def verify_highlight_accuracy(
    original_snapshot: OriginalSnapshot,
    optimized_content: str,
    change_set: ChangeSet
) -> VerificationResult:

    issues = []

    # Check 1: No original content in highlights
    for addition in change_set.additions:
        highlighted_text = addition.content

        # Exact match check
        if highlighted_text in original_snapshot.full_text:
            issues.append(Issue(
                type="FALSE_POSITIVE",
                severity="CRITICAL",
                description="Highlighted text exists verbatim in original",
                location=addition.position,
                content=highlighted_text[:100]
            ))

        # Fuzzy match check (90% similarity)
        similar_passages = find_similar_passages(
            highlighted_text,
            original_snapshot.full_text,
            threshold=0.90
        )
        if similar_passages:
            issues.append(Issue(
                type="POTENTIAL_FALSE_POSITIVE",
                severity="HIGH",
                description="Highlighted text very similar to original",
                similarity=similar_passages[0].score,
                original_passage=similar_passages[0].text
            ))

    # Check 2: All new content is highlighted
    new_content_regions = identify_new_regions(original_snapshot, optimized_content)
    for region in new_content_regions:
        if not is_highlighted(region, change_set):
            issues.append(Issue(
                type="FALSE_NEGATIVE",
                severity="CRITICAL",
                description="New content not highlighted",
                content=region.text[:100]
            ))

    # Check 3: Highlight boundaries are clean
    for addition in change_set.additions:
        if has_partial_word_boundary(addition):
            issues.append(Issue(
                type="BOUNDARY_ERROR",
                severity="MEDIUM",
                description="Highlight starts/ends mid-word"
            ))

    return VerificationResult(
        passed=len([i for i in issues if i.severity == "CRITICAL"]) == 0,
        issues=issues,
        confidence=calculate_verification_confidence(issues)
    )
```

### 5.2 Existing Content Overlap Detection

#### 5.2.1 N-gram Overlap Analysis

```python
def detect_content_overlap(
    highlighted_text: str,
    original_content: str,
    n_gram_sizes: List[int] = [3, 4, 5, 6]
) -> OverlapResult:

    overlap_scores = {}

    for n in n_gram_sizes:
        original_ngrams = set(extract_ngrams(original_content, n))
        highlighted_ngrams = set(extract_ngrams(highlighted_text, n))

        if highlighted_ngrams:
            overlap = len(original_ngrams & highlighted_ngrams) / len(highlighted_ngrams)
            overlap_scores[f"{n}-gram"] = overlap

    # Weighted average (longer n-grams more significant)
    weights = {3: 0.1, 4: 0.2, 5: 0.3, 6: 0.4}
    weighted_overlap = sum(
        overlap_scores.get(f"{n}-gram", 0) * w
        for n, w in weights.items()
    )

    return OverlapResult(
        scores=overlap_scores,
        weighted_average=weighted_overlap,
        is_suspicious=weighted_overlap > 0.5,
        requires_review=weighted_overlap > 0.3
    )
```

#### 5.2.2 Semantic Similarity Check

```python
def check_semantic_overlap(
    highlighted_text: str,
    original_sentences: List[str],
    threshold: float = 0.85
) -> List[SemanticMatch]:

    matches = []
    highlighted_embedding = model.encode(highlighted_text)

    for sentence in original_sentences:
        original_embedding = model.encode(sentence)
        similarity = cosine_similarity(highlighted_embedding, original_embedding)

        if similarity >= threshold:
            matches.append(SemanticMatch(
                highlighted=highlighted_text,
                original=sentence,
                similarity=similarity,
                interpretation="rewording" if similarity < 0.95 else "duplicate"
            ))

    return matches
```

### 5.3 Confidence Threshold for Human Review

#### 5.3.1 Diff Confidence Calculation

```
diff_confidence = (
    (1 - false_positive_risk) * 0.40 +
    (1 - false_negative_risk) * 0.40 +
    boundary_precision * 0.20
)

WHERE:
    false_positive_risk = weighted_ngram_overlap + semantic_similarity_max * 0.5
    false_negative_risk = undetected_new_content_ratio
    boundary_precision = clean_boundaries / total_boundaries

CONFIDENCE LEVELS:
    >= 0.95: HIGH (auto-approve highlights)
    0.85-0.94: MEDIUM (approve with logging)
    0.70-0.84: LOW (human review recommended)
    < 0.70: VERY_LOW (human review required)
```

### 5.4 False Positive Prevention Strategies

```
PREVENTION RULES:

1. Conservative Highlighting Mode:
   - When in doubt, do NOT highlight
   - Prefer false negatives over false positives
   - User trust depends on green = genuinely new

2. Reworded Content Handling:
   - Semantic similarity > 0.85 to original = NOT new (don't highlight)
   - Semantic similarity 0.70-0.85 = partial rewrite (human review)
   - Semantic similarity < 0.70 = genuinely new (highlight)

3. Expanded Content Handling:
   - "The product helps users" -> "The product helps users save time"
   - Highlight ONLY "save time" (the actual addition)
   - Use character-level diffing for precise boundaries

4. Moved Content Handling:
   - Content that appears in different location = NOT new
   - Track content by semantic fingerprint, not position
   - Only highlight if content is truly added, not relocated

5. Multi-Pass Verification:
   - Pass 1: Character-level diff
   - Pass 2: Semantic similarity check
   - Pass 3: N-gram overlap analysis
   - Final: Human review for low-confidence regions
```

---

## 6. Human Review Triggers

### 6.1 Trigger Conditions Matrix

| Condition | Threshold | Trigger Level | Required Action |
|-----------|-----------|---------------|-----------------|
| Diff confidence < 0.70 | confidence_score | MANDATORY | Approve/reject highlight changes |
| Diff confidence 0.70-0.85 | confidence_score | RECOMMENDED | Review flagged sections |
| Keyword stuffing score > 50 | stuffing_score | MANDATORY | Confirm or reduce keyword usage |
| Keyword stuffing score 30-50 | stuffing_score | WARNING | Review keyword distribution |
| Grounding score < 0.50 | grounding_score | MANDATORY | Provide source or reject |
| Grounding score 0.50-0.70 | grounding_score | RECOMMENDED | Verify claims |
| Hallucination risk > 0.35 | hallucination_risk | MANDATORY | Verify all flagged claims |
| Voice similarity < 0.75 | voice_score | MANDATORY | Approve tone or revert |
| Voice similarity 0.75-0.85 | voice_score | RECOMMENDED | Review tone changes |
| Formality drift > 0.25 | formality_delta | MANDATORY | Confirm formality appropriate |
| Sentiment shift > 0.25 | sentiment_delta | MANDATORY | Verify sentiment acceptable |
| Polarity flip detected | polarity_change | MANDATORY | Investigate cause |
| Entity modified | entity_change | MANDATORY | Verify accuracy |
| Numerical value changed | number_change | MANDATORY | Always verify |
| New claim introduced | new_claim | RECOMMENDED | Verify or provide source |
| YMYL content detected | ymyl_flag | MANDATORY | Expert review required |
| Banned term introduced | term_violation | MANDATORY | Remove or justify |
| Legal/compliance term modified | legal_term | MANDATORY | Legal review required |
| Content length change > 25% | length_delta | WARNING | Review scope |
| Multiple sections modified | section_count > 3 | RECOMMENDED | Review comprehensiveness |

### 6.2 Review Interface Data Requirements

```json
{
  "review_item": {
    "id": "review_20260116_001",
    "document_id": "doc_12345",
    "created_at": "2026-01-16T10:30:00Z",

    "trigger_reasons": [
      {
        "type": "LOW_DIFF_CONFIDENCE",
        "value": 0.68,
        "threshold": 0.70,
        "severity": "MANDATORY"
      },
      {
        "type": "VOICE_DRIFT",
        "value": 0.73,
        "threshold": 0.75,
        "severity": "MANDATORY"
      }
    ],

    "content_comparison": {
      "original": {
        "word_count": 1250,
        "keyword_density": 1.8,
        "formality_score": 0.65,
        "sentiment_score": 0.3
      },
      "optimized": {
        "word_count": 1420,
        "keyword_density": 2.4,
        "formality_score": 0.58,
        "sentiment_score": 0.35
      },
      "deltas": {
        "word_count": "+170 (+13.6%)",
        "keyword_density": "+0.6 percentage points",
        "formality": "-0.07 (within tolerance)",
        "sentiment": "+0.05 (within tolerance)"
      }
    },

    "highlighted_sections": [
      {
        "id": "hl_001",
        "content": "Our innovative approach helps businesses...",
        "location": {"start": 456, "end": 523},
        "confidence": 0.92,
        "verification_status": "PASSED"
      },
      {
        "id": "hl_002",
        "content": "FAQ: What makes our service unique?...",
        "location": {"start": 1250, "end": 1420},
        "confidence": 0.65,
        "verification_status": "NEEDS_REVIEW",
        "concerns": ["Low grounding score (0.58)", "New claims introduced"]
      }
    ],

    "guardrail_scores": {
      "keyword_stuffing": 28,
      "factual_grounding": 0.72,
      "hallucination_risk": 0.22,
      "voice_consistency": 0.78,
      "highlight_confidence": 0.68
    },

    "recommendation": "REVIEW_REQUIRED",
    "auto_approve_eligible": false
  }
}
```

### 6.3 Approval/Rejection Flow

```
REVIEW WORKFLOW:

1. QUEUE ENTRY
   - Item enters review queue with priority based on trigger severity
   - Priority: MANDATORY > RECOMMENDED > WARNING
   - SLA: MANDATORY within 4 hours, RECOMMENDED within 24 hours

2. REVIEW PRESENTATION
   - Side-by-side diff view (original | optimized)
   - Highlighted sections clearly marked
   - Trigger reasons prominently displayed
   - Relevant metrics and scores shown

3. REVIEWER ACTIONS
   a. APPROVE_ALL
      - Accept all changes including highlights
      - Log approval with reviewer ID
      - Proceed to output generation

   b. APPROVE_PARTIAL
      - Select specific changes to accept
      - Reject or modify remaining changes
      - System regenerates with accepted changes only

   c. REJECT_ALL
      - Revert to original document
      - Log rejection reason
      - Optionally trigger re-optimization with adjusted parameters

   d. MODIFY
      - Reviewer makes manual edits
      - Modified content replaces optimization output
      - Manual changes logged for training data

   e. ESCALATE
      - Send to senior reviewer or subject matter expert
      - Add escalation notes
      - Reset SLA timer

4. POST-REVIEW
   - Decision logged with timestamp and reviewer
   - Metrics updated for guardrail calibration
   - Proceed to output or revision cycle
```

---

## 7. Changes Summary Specification

### 7.1 Summary Location

The changes summary appears in two locations:

1. **Document Header (First Page)**: Condensed summary box at the top of the optimized DOCX
2. **Separate Report File**: Detailed `changes_report.md` alongside the output DOCX

### 7.2 Metrics Included

#### 7.2.1 Core Metrics

| Metric | Description | Format |
|--------|-------------|--------|
| Total Words Added | Net new word count | "+170 words" |
| Total Words Removed | Words deleted (if any) | "-0 words" |
| Net Change | Word count delta | "+170 words (+13.6%)" |
| Sections Modified | Count of H2 sections with changes | "3 sections" |
| New Sections Added | Entirely new sections | "1 section (FAQ)" |
| Highlighted Regions | Count of green-highlighted areas | "5 regions" |

#### 7.2.2 SEO Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Primary Keyword Density | 1.8% | 2.4% | +0.6pp |
| Secondary Keyword Coverage | 2/5 | 4/5 | +2 keywords |
| Semantic Keyword Density | 5.2% | 6.8% | +1.6pp |
| Combined Keyword Footprint | 7.0% | 9.2% | +2.2pp |

#### 7.2.3 Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| FAQ Generated | Yes | New section added |
| Factual Grounding Score | 0.85 | PASS |
| Voice Consistency Score | 0.88 | PASS |
| Highlight Confidence | 0.92 | HIGH |
| Keyword Stuffing Score | 28/100 | PASS |
| Human Review Required | No | Auto-approved |

### 7.3 Summary Format Template

#### 7.3.1 In-Document Summary (DOCX Header)

```
+------------------------------------------------------------------+
|                    OPTIMIZATION SUMMARY                           |
+------------------------------------------------------------------+
| Document: [Original Filename]                                     |
| Processed: [Date/Time]                                            |
+------------------------------------------------------------------+
| CONTENT CHANGES                                                   |
|   Words Added: +170 (+13.6%)                                     |
|   Sections Modified: 3                                            |
|   FAQ Generated: Yes (5 Q&A pairs)                               |
|   Highlighted Regions: 5                                          |
+------------------------------------------------------------------+
| SEO IMPACT                                                        |
|   Keyword Density: 1.8% -> 2.4%                                  |
|   Keyword Coverage: 2/5 -> 4/5 secondary keywords                |
+------------------------------------------------------------------+
| QUALITY SCORES                                                    |
|   Grounding: 0.85 (PASS)  |  Voice: 0.88 (PASS)                 |
|   Highlight Confidence: 0.92 (HIGH)                              |
+------------------------------------------------------------------+
| All new content is highlighted in GREEN                          |
+------------------------------------------------------------------+
```

#### 7.3.2 Detailed Report (changes_report.md)

```markdown
# Content Optimization Report

**Document:** [Original Filename]
**Processed:** [Date/Time]
**Tool Version:** [Version]

---

## Executive Summary

This document was optimized for SEO targeting the following keywords:
- Primary: "[primary keyword]"
- Secondary: "[keyword 2]", "[keyword 3]", "[keyword 4]", "[keyword 5]"

**Key Changes:**
- Added 170 words (+13.6% content increase)
- Generated new FAQ section with 5 question-answer pairs
- Enhanced 3 existing sections with keyword-optimized content
- All additions highlighted in green for easy review

---

## Content Changes Detail

### New Sections Added

#### FAQ Section (Position: After main content)
- **Questions Added:** 5
- **Word Count:** 245 words
- **Grounding Score:** 0.87 (all answers derived from source content)

### Modified Sections

| Section | Change Type | Words Added | Highlight Regions |
|---------|-------------|-------------|-------------------|
| Introduction | Enhanced | +35 | 1 |
| Services Overview | Expanded | +62 | 2 |
| Benefits | Keyword integration | +28 | 2 |

---

## SEO Metrics Comparison

| Metric | Before | After | Change | Status |
|--------|--------|-------|--------|--------|
| Primary Keyword Density | 1.8% | 2.4% | +0.6pp | OPTIMAL |
| Secondary Keyword Coverage | 40% | 80% | +40pp | IMPROVED |
| Semantic Cluster Density | 5.2% | 6.8% | +1.6pp | GOOD |
| Combined Footprint | 7.0% | 9.2% | +2.2pp | WITHIN LIMITS |

---

## Quality Assurance Results

### Guardrail Checks

| Check | Score | Threshold | Status |
|-------|-------|-----------|--------|
| Keyword Stuffing | 28 | < 50 | PASS |
| Factual Grounding | 0.85 | >= 0.70 | PASS |
| Hallucination Risk | 0.12 | < 0.35 | PASS |
| Voice Consistency | 0.88 | >= 0.75 | PASS |
| Highlight Accuracy | 0.92 | >= 0.85 | PASS |

### Human Review Status

- **Required:** No
- **Reason:** All guardrail checks passed with high confidence

---

## Highlighted Content Inventory

| # | Location | Content Preview | Confidence |
|---|----------|-----------------|------------|
| 1 | Para 3 | "Our comprehensive approach to..." | 0.95 |
| 2 | Para 7 | "Businesses benefit from..." | 0.91 |
| 3 | Para 12 | "The solution provides..." | 0.93 |
| 4 | FAQ Q1 | "What services do you offer?..." | 0.88 |
| 5 | FAQ Q2-Q5 | "How does the process work?..." | 0.90 |

---

## Recommendations

1. Review the FAQ section to ensure answers align with current offerings
2. Consider adding internal links to the enhanced sections
3. Monitor keyword density in future updates to avoid over-optimization

---

*Report generated by SEO Content Optimizer v2.0*
```

---

## 8. Validation Pipeline

### 8.1 Pipeline Sequence

```
VALIDATION PIPELINE ORDER:

┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: PRE-GENERATION CHECKS                                   │
├─────────────────────────────────────────────────────────────────┤
│ 1.1 Source content baseline capture                              │
│     - Extract all entities (PERSON, ORG, DATE, MONEY, etc.)     │
│     - Capture all numerical values                               │
│     - Extract verifiable claims                                  │
│     - Calculate voice/tone baseline                              │
│                                                                  │
│ 1.2 Brand context loading                                        │
│     - Load terminology rules                                     │
│     - Build voice profile                                        │
│     - Set content-type-specific thresholds                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: GENERATION (External to guardrails)                     │
├─────────────────────────────────────────────────────────────────┤
│ Content generation and optimization occurs here                  │
│ Output: OptimizedDocumentAST                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: POST-GENERATION VALIDATION                              │
├─────────────────────────────────────────────────────────────────┤
│ 3.1 Over-optimization check                                      │
│     - Calculate density metrics                                  │
│     - Check repetition patterns                                  │
│     - Detect awkward insertions                                  │
│     - Compute keyword stuffing score                             │
│     └─ IF score > 70: REJECT immediately                        │
│     └─ IF score > 50: Flag for human review                     │
│                                                                  │
│ 3.2 Factual grounding verification                               │
│     - Compare generated claims to source                         │
│     - Calculate grounding scores                                 │
│     - Detect potential hallucinations                            │
│     - Flag unverifiable claims                                   │
│     └─ IF grounding < 0.30: REJECT as hallucination             │
│     └─ IF grounding < 0.50: Flag for human review               │
│                                                                  │
│ 3.3 Brand voice validation                                       │
│     - Calculate voice similarity                                 │
│     - Check formality drift                                      │
│     - Verify terminology compliance                              │
│     - Assess sentiment consistency                               │
│     └─ IF voice_score < 0.65: REJECT                            │
│     └─ IF voice_score < 0.75: Flag for human review             │
│                                                                  │
│ 3.4 Entity preservation check                                    │
│     - Verify all critical entities preserved                     │
│     - Check numerical accuracy                                   │
│     - Validate claim integrity                                   │
│     └─ IF any CRITICAL entity missing: REJECT                   │
│     └─ IF any number changed: REJECT                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: DIFFING (External to guardrails)                        │
├─────────────────────────────────────────────────────────────────┤
│ Character-level diff between original and optimized              │
│ Output: ChangeSet with highlight boundaries                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: HIGHLIGHT VERIFICATION                                  │
├─────────────────────────────────────────────────────────────────┤
│ 5.1 False positive detection                                     │
│     - Check highlighted text against original                    │
│     - N-gram overlap analysis                                    │
│     - Semantic similarity check                                  │
│     └─ IF verbatim match in original: CRITICAL ERROR            │
│     └─ IF similarity > 0.90: Flag for review                    │
│                                                                  │
│ 5.2 False negative detection                                     │
│     - Identify all new content regions                           │
│     - Verify each is highlighted                                 │
│     └─ IF new content unhighlighted: CRITICAL ERROR             │
│                                                                  │
│ 5.3 Boundary verification                                        │
│     - Check highlight boundaries are clean                       │
│     - Verify no partial-word highlights                          │
│     - Confirm sentence-level coherence                           │
│                                                                  │
│ 5.4 Confidence scoring                                           │
│     - Calculate diff confidence score                            │
│     └─ IF confidence < 0.70: MANDATORY human review             │
│     └─ IF confidence < 0.85: RECOMMENDED human review           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 6: AGGREGATE DECISION                                      │
├─────────────────────────────────────────────────────────────────┤
│ 6.1 Collect all check results                                    │
│ 6.2 Apply decision logic (see 8.3)                               │
│ 6.3 Route to appropriate path:                                   │
│     - AUTO_APPROVE: Proceed to output                            │
│     - HUMAN_REVIEW: Queue for review                             │
│     - REJECT: Return to generation with feedback                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 7: OUTPUT GENERATION (if approved)                         │
├─────────────────────────────────────────────────────────────────┤
│ - Generate DOCX with highlights                                  │
│ - Create changes summary                                         │
│ - Generate detailed report                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Pass/Fail Criteria Per Check

| Check | Pass Criteria | Fail Criteria | Review Criteria |
|-------|---------------|---------------|-----------------|
| Keyword Stuffing | score < 30 | score > 70 | score 30-70 |
| Factual Grounding | score >= 0.70 | score < 0.30 | score 0.30-0.70 |
| Hallucination Risk | risk < 0.35 | risk > 0.55 | risk 0.35-0.55 |
| Voice Consistency | score >= 0.85 | score < 0.65 | score 0.65-0.85 |
| Entity Preservation | rate >= 0.95 | rate < 0.70 | rate 0.70-0.95 |
| Numerical Accuracy | 100% preserved | any change | format-only changes |
| Highlight Confidence | conf >= 0.85 | conf < 0.70 | conf 0.70-0.85 |
| False Positive Check | 0 verbatim matches | any verbatim match | similarity > 0.85 |
| False Negative Check | 0 missed highlights | any missed | uncertain regions |
| Terminology Compliance | 0 violations | banned term used | non-preferred terms |

### 8.3 Aggregate Decision Logic

```python
def make_aggregate_decision(check_results: CheckResults) -> Decision:

    # REJECT conditions (any one triggers rejection)
    if check_results.keyword_stuffing_score > 70:
        return Decision.REJECT("Over-optimization: keyword stuffing detected")

    if check_results.factual_grounding < 0.30:
        return Decision.REJECT("Hallucination: content not grounded in source")

    if check_results.voice_consistency < 0.65:
        return Decision.REJECT("Brand voice: significant drift from brand profile")

    if check_results.entity_preservation < 0.70:
        return Decision.REJECT("Factual integrity: critical entities missing")

    if check_results.numerical_changes > 0:
        return Decision.REJECT("Factual integrity: numerical values modified")

    if check_results.highlight_false_positives > 0:
        return Decision.REJECT("Highlight error: existing content marked as new")

    if check_results.highlight_false_negatives > 0:
        return Decision.REJECT("Highlight error: new content not marked")

    if check_results.banned_terms_used:
        return Decision.REJECT("Terminology: banned terms introduced")

    # REVIEW conditions (accumulate triggers)
    review_triggers = []

    if check_results.keyword_stuffing_score > 50:
        review_triggers.append("High keyword density")

    if check_results.factual_grounding < 0.70:
        review_triggers.append("Moderate grounding concerns")

    if check_results.hallucination_risk > 0.35:
        review_triggers.append("Elevated hallucination risk")

    if check_results.voice_consistency < 0.85:
        review_triggers.append("Voice consistency below threshold")

    if check_results.highlight_confidence < 0.85:
        review_triggers.append("Low highlight confidence")

    if check_results.formality_drift > 0.25:
        review_triggers.append("Formality drift detected")

    if check_results.sentiment_shift > 0.25:
        review_triggers.append("Sentiment shift detected")

    if check_results.is_ymyl_content:
        review_triggers.append("YMYL content requires review")

    if check_results.new_claims_count > 0:
        review_triggers.append(f"{check_results.new_claims_count} new claims introduced")

    # Decision based on accumulated triggers
    if check_results.highlight_confidence < 0.70:
        return Decision.MANDATORY_REVIEW(review_triggers)

    if len(review_triggers) >= 3:
        return Decision.MANDATORY_REVIEW(review_triggers)

    if len(review_triggers) >= 1:
        return Decision.RECOMMENDED_REVIEW(review_triggers)

    # All checks passed
    return Decision.AUTO_APPROVE()
```

### 8.4 Bypass Options (Admin Override)

```yaml
admin_overrides:
  description: "Configuration for authorized bypass of guardrail checks"

  available_overrides:
    skip_keyword_density_check:
      requires_role: "admin"
      audit_logging: true
      max_bypass_score: 85  # Cannot bypass above this score

    skip_voice_validation:
      requires_role: "admin"
      audit_logging: true
      reason_required: true

    force_approve_highlights:
      requires_role: "senior_reviewer"
      audit_logging: true
      requires_manual_verification: true

    bypass_ymyl_review:
      requires_role: "compliance_officer"
      audit_logging: true
      legal_acknowledgment: true

    override_rejection:
      requires_role: "admin"
      audit_logging: true
      requires_two_approvers: true
      reason_required: true
      expires_after: "24h"

  audit_requirements:
    - All overrides logged with timestamp
    - Override reason documented
    - Approver identity recorded
    - Original check scores preserved
    - Override usage reviewed weekly
```

---

## 9. Testing Strategy

### 9.1 Test Cases by Guardrail

#### 9.1.1 Over-Optimization Tests

| Test ID | Scenario | Input | Expected Outcome |
|---------|----------|-------|------------------|
| OO-001 | Clean content | Natural text, 1.5% density | PASS, score < 20 |
| OO-002 | Moderate density | 3% exact match density | WARNING, score 30-50 |
| OO-003 | High density | 5% exact match density | BLOCK, score 50-70 |
| OO-004 | Keyword stuffing | 8% density, regular spacing | REJECT, score > 70 |
| OO-005 | Natural clustering | High density in relevant section | PASS with context adjustment |
| OO-006 | Awkward insertion | Keyword breaks sentence flow | Flag perplexity spike |
| OO-007 | Short content tolerance | 200 words, 4% density | PASS with adjusted threshold |
| OO-008 | Long content decay | 3000 words, consistent density | WARNING, no decay |

#### 9.1.2 Factual Grounding Tests

| Test ID | Scenario | Input | Expected Outcome |
|---------|----------|-------|------------------|
| FG-001 | Fully grounded | All claims from source | PASS, grounding > 0.85 |
| FG-002 | Mostly grounded | 80% claims from source | PASS, grounding 0.70-0.85 |
| FG-003 | Partial grounding | 50% claims from source | REVIEW, grounding 0.50-0.70 |
| FG-004 | Weak grounding | 30% claims from source | BLOCK, grounding 0.30-0.50 |
| FG-005 | Hallucination | Made-up statistics | REJECT, grounding < 0.30 |
| FG-006 | Reasonable inference | Derived from source logic | PASS with flag |
| FG-007 | Entity preservation | All entities maintained | PASS |
| FG-008 | Entity modification | Person name changed | REJECT |
| FG-009 | Number change | Price value altered | REJECT |
| FG-010 | Date accuracy | Date preserved correctly | PASS |

#### 9.1.3 Brand Voice Tests

| Test ID | Scenario | Input | Expected Outcome |
|---------|----------|-------|------------------|
| BV-001 | Perfect match | Same voice as source | PASS, similarity > 0.90 |
| BV-002 | Minor drift | Slight formality change | PASS, similarity 0.85-0.90 |
| BV-003 | Moderate drift | Noticeable tone shift | WARNING, similarity 0.75-0.85 |
| BV-004 | Significant drift | Major voice change | REVIEW, similarity 0.65-0.75 |
| BV-005 | Voice mismatch | Completely different voice | REJECT, similarity < 0.65 |
| BV-006 | Formality increase | Casual to formal | Flag formality drift |
| BV-007 | Formality decrease | Formal to casual | Flag formality drift |
| BV-008 | Sentiment flip | Positive to negative | REJECT, polarity flip |
| BV-009 | Banned term used | Competitor name inserted | BLOCK |
| BV-010 | Preferred term swap | Alternative term used | WARNING |

#### 9.1.4 Highlight Verification Tests

| Test ID | Scenario | Input | Expected Outcome |
|---------|----------|-------|------------------|
| HL-001 | Clean highlight | New content only | PASS, confidence > 0.95 |
| HL-002 | Verbatim match | Existing text highlighted | CRITICAL, false positive |
| HL-003 | Near-duplicate | 95% similar to original | FLAG, potential false positive |
| HL-004 | Reworded content | Semantic equivalent | NO highlight (correctly) |
| HL-005 | Expansion | "helps users" -> "helps users save" | Highlight "save" only |
| HL-006 | Moved content | Paragraph relocated | NO highlight (correctly) |
| HL-007 | New paragraph | Entirely new content | Full highlight |
| HL-008 | Boundary precision | Clean word boundaries | PASS |
| HL-009 | Partial word | Highlight mid-word | FLAG, boundary error |
| HL-010 | Missed content | New content not highlighted | CRITICAL, false negative |

### 9.2 Edge Case Scenarios

```
EDGE CASES TO TEST:

1. MIXED LANGUAGE CONTENT
   - Source in English, keywords in technical jargon
   - Non-ASCII characters in brand names
   - Mixed formal/informal sections

2. STRUCTURED CONTENT
   - Tables with data (preserve exactly)
   - Bulleted lists (keyword in list items)
   - Code snippets (never modify)
   - Block quotes (preserve attribution)

3. BOUNDARY CONDITIONS
   - Very short content (< 100 words)
   - Very long content (> 5000 words)
   - Single-sentence paragraphs
   - No headings structure

4. SPECIAL CONTENT TYPES
   - FAQ already exists (don't duplicate)
   - Legal disclaimers (never modify)
   - Testimonials with quotes (preserve exactly)
   - Technical specifications (preserve numbers)

5. MULTI-KEYWORD SCENARIOS
   - Conflicting keywords (synonyms)
   - Keyword in brand name
   - Keyword is common word
   - Long-tail keyword phrases

6. FORMAT PRESERVATION
   - Bold/italic within sentences
   - Hyperlinks in text
   - Footnotes and references
   - Image captions (preserve)

7. ERROR RECOVERY
   - Partial generation failure
   - Invalid source document
   - Missing brand context
   - Conflicting rules
```

### 9.3 False Positive/Negative Monitoring

```python
class GuardrailMetrics:
    """Track guardrail performance for continuous improvement."""

    def __init__(self):
        self.total_documents = 0
        self.auto_approved = 0
        self.sent_to_review = 0
        self.rejected = 0

        # False positive tracking (flagged but actually fine)
        self.review_approved = 0  # Sent to review, then approved
        self.rejection_overridden = 0  # Rejected, then manually approved

        # False negative tracking (passed but had issues)
        self.post_publish_issues = 0  # Issues found after publication
        self.user_reported_problems = 0  # User-reported content issues

    def calculate_false_positive_rate(self) -> float:
        """Rate at which guardrails flag acceptable content."""
        if self.sent_to_review == 0:
            return 0.0
        return self.review_approved / self.sent_to_review

    def calculate_false_negative_rate(self) -> float:
        """Rate at which guardrails miss problematic content."""
        if self.auto_approved == 0:
            return 0.0
        return self.post_publish_issues / self.auto_approved

    def generate_calibration_report(self) -> CalibrationReport:
        """Weekly report for threshold adjustment."""
        return CalibrationReport(
            false_positive_rate=self.calculate_false_positive_rate(),
            false_negative_rate=self.calculate_false_negative_rate(),
            review_burden=self.sent_to_review / self.total_documents,
            rejection_rate=self.rejected / self.total_documents,
            recommendations=self.generate_threshold_recommendations()
        )

    def generate_threshold_recommendations(self) -> List[str]:
        """Suggest threshold adjustments based on metrics."""
        recommendations = []

        fp_rate = self.calculate_false_positive_rate()
        fn_rate = self.calculate_false_negative_rate()

        if fp_rate > 0.30:
            recommendations.append(
                "High false positive rate (>30%). Consider relaxing thresholds."
            )

        if fn_rate > 0.05:
            recommendations.append(
                "Elevated false negative rate (>5%). Consider tightening thresholds."
            )

        if self.sent_to_review / self.total_documents > 0.50:
            recommendations.append(
                "High review burden (>50%). Consider adjusting review triggers."
            )

        return recommendations
```

### 9.4 Continuous Calibration Process

```
WEEKLY CALIBRATION CYCLE:

1. COLLECT METRICS
   - Gather all guardrail decisions from past week
   - Track human review outcomes
   - Record post-publication feedback
   - Note any override usage

2. ANALYZE PERFORMANCE
   - Calculate false positive/negative rates
   - Identify most frequent review triggers
   - Review rejected content patterns
   - Analyze override patterns

3. ADJUST THRESHOLDS
   - If false positive rate > 30%: Relax by 5%
   - If false negative rate > 5%: Tighten by 5%
   - Never adjust more than 10% per week
   - Document all changes with rationale

4. VALIDATE CHANGES
   - Run test suite against new thresholds
   - Verify no regression on edge cases
   - Confirm decision consistency

5. DEPLOY AND MONITOR
   - Roll out threshold changes
   - Monitor first 24 hours closely
   - Be prepared to rollback if issues
```

---

## 10. Configuration Reference

### 10.1 Default Configuration

```yaml
guardrails:
  version: "2.0"

  over_optimization:
    enabled: true
    exact_match_density:
      warning: 2.5
      block: 4.0
      reject: 5.0
    phrase_match_density:
      warning: 4.0
      block: 6.0
      reject: 8.0
    semantic_density:
      warning: 8.0
      block: 12.0
      reject: 15.0
    stuffing_score:
      warning: 30
      block: 50
      reject: 70
    perplexity_delta_threshold: 15

  factual_grounding:
    enabled: true
    grounding_threshold:
      pass: 0.70
      review: 0.50
      block: 0.30
    hallucination_risk_threshold:
      pass: 0.35
      review: 0.55
    entity_preservation_rate: 0.95
    numerical_change_tolerance: 0.0  # Zero tolerance

  brand_voice:
    enabled: true
    voice_similarity:
      pass: 0.85
      warning: 0.75
      reject: 0.65
    formality_drift_threshold: 0.25
    sentiment_shift_threshold: 0.25
    terminology_enforcement: strict

  highlight_verification:
    enabled: true
    confidence_threshold:
      auto_approve: 0.95
      recommend_review: 0.85
      require_review: 0.70
    semantic_similarity_threshold: 0.85
    ngram_overlap_threshold: 0.50

  human_review:
    enabled: true
    mandatory_triggers:
      - low_diff_confidence
      - high_keyword_stuffing
      - low_grounding
      - voice_drift
      - ymyl_content
      - entity_modification
      - numerical_change
    sla_hours:
      mandatory: 4
      recommended: 24

  reporting:
    include_document_summary: true
    generate_detailed_report: true
    report_format: markdown
```

### 10.2 Content Type Modifiers

```yaml
content_type_modifiers:
  blog_post:
    keyword_density_multiplier: 1.0
    voice_drift_tolerance: 1.1
    auto_approval_enabled: true

  product_page:
    keyword_density_multiplier: 0.8
    voice_drift_tolerance: 0.9
    auto_approval_enabled: false
    protected_sections:
      - pricing
      - specifications

  landing_page:
    keyword_density_multiplier: 0.9
    voice_drift_tolerance: 0.85
    auto_approval_enabled: false

  technical_documentation:
    keyword_density_multiplier: 0.85
    voice_drift_tolerance: 1.0
    factual_strictness: 1.2

  ymyl_content:
    keyword_density_multiplier: 1.2
    voice_drift_tolerance: 0.8
    factual_strictness: 1.5
    auto_approval_enabled: false
    mandatory_human_review: true
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Density** | Ratio of keyword occurrences to total word count |
| **Grounding Score** | Measure of how well generated content is supported by source |
| **Hallucination** | AI-generated content not supported by source material |
| **Voice Drift** | Deviation from established brand tone and style |
| **False Positive** | Existing content incorrectly flagged as new |
| **False Negative** | New content incorrectly not flagged/highlighted |
| **YMYL** | "Your Money Your Life" - high-stakes content categories |
| **Perplexity** | Measure of how predictable/natural text is to a language model |
| **ChangeSet** | Collection of identified new content regions with highlight boundaries |
| **Entity** | Named item (person, organization, date, etc.) extracted via NER |

---

## Appendix B: Decision Tree Summary

```
START: Content optimization complete
    │
    ├─► Over-optimization check
    │   ├─ Score > 70 ──────────────────────────► REJECT
    │   ├─ Score 50-70 ─────────────────────────► FLAG: REVIEW
    │   └─ Score < 50 ──────────────────────────► CONTINUE
    │
    ├─► Factual grounding check
    │   ├─ Grounding < 0.30 ────────────────────► REJECT
    │   ├─ Grounding 0.30-0.70 ─────────────────► FLAG: REVIEW
    │   ├─ Number changed ──────────────────────► REJECT
    │   ├─ Critical entity missing ─────────────► REJECT
    │   └─ Grounding >= 0.70 ───────────────────► CONTINUE
    │
    ├─► Brand voice check
    │   ├─ Similarity < 0.65 ───────────────────► REJECT
    │   ├─ Similarity 0.65-0.85 ────────────────► FLAG: REVIEW
    │   ├─ Banned term used ────────────────────► REJECT
    │   └─ Similarity >= 0.85 ──────────────────► CONTINUE
    │
    ├─► Highlight verification
    │   ├─ Verbatim match in original ──────────► REJECT (critical error)
    │   ├─ New content not highlighted ─────────► REJECT (critical error)
    │   ├─ Confidence < 0.70 ───────────────────► FLAG: MANDATORY REVIEW
    │   ├─ Confidence 0.70-0.85 ────────────────► FLAG: RECOMMENDED REVIEW
    │   └─ Confidence >= 0.85 ──────────────────► CONTINUE
    │
    ├─► Aggregate decision
    │   ├─ Any REJECT flag ─────────────────────► REJECT
    │   ├─ MANDATORY REVIEW flag ───────────────► HUMAN REVIEW (required)
    │   ├─ >= 3 REVIEW flags ───────────────────► HUMAN REVIEW (required)
    │   ├─ 1-2 REVIEW flags ────────────────────► HUMAN REVIEW (recommended)
    │   └─ No flags ────────────────────────────► AUTO-APPROVE
    │
    └─► Output
        ├─ REJECT ──────────────────────────────► Return error, suggest fixes
        ├─ HUMAN REVIEW ────────────────────────► Queue for reviewer
        └─ AUTO-APPROVE ────────────────────────► Generate output DOCX
```

---

*Document Version: 2.0*
*Last Updated: January 16, 2026*
*Status: Active Specification*

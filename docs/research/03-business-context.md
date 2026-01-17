# Section 3: Business Context Inference

## Executive Summary

The Business Context module serves as the intelligence layer that transforms raw documents into actionable understanding about the business, enabling contextually appropriate content generation. This module addresses a fundamental challenge in SEO content optimization: generating content that sounds authentic to the brand rather than generic AI-generated text.

The system implements a dual-path architecture for context acquisition. The primary path processes explicit brand documents (style guides, product catalogs, mission statements) when provided, extracting structured information about tone, terminology, products, services, and target audience. The secondary path activates when brand documents are unavailable, employing NLP techniques to infer business context directly from the source content through entity extraction, topic modeling, and industry classification.

The output is a unified `BusinessContext` object that downstream modules (FAQ generation, content enhancement) consume to ensure all generated content aligns with the inferred or explicit brand identity. A confidence scoring system accompanies all inferred values, allowing the generation module to adjust its approach based on certainty levels---using more conservative language when context confidence is low.

---

## 1. Brand Document Processing

### 1.1 Supported Document Types

The system accepts brand context through multiple document formats:

| Format | Extension | Parser | Use Case |
|--------|-----------|--------|----------|
| Microsoft Word | `.docx` | `python-docx` | Brand guides, style manuals |
| Plain Text | `.txt` | Native Python | Simple brand notes |
| Markdown | `.md` | `markdown` library | Technical brand docs |
| PDF | `.pdf` | `pdfplumber` | Marketing collateral |
| JSON/YAML | `.json`, `.yaml` | Native parsers | Structured brand configs |

### 1.2 Document Type Classification

Brand documents are classified by their content type to apply appropriate extraction strategies:

```python
from enum import Enum
from typing import List, Dict, Optional

class BrandDocumentType(Enum):
    STYLE_GUIDE = "style_guide"           # Tone, voice, formatting rules
    PRODUCT_CATALOG = "product_catalog"   # Products/services descriptions
    MISSION_STATEMENT = "mission_statement"  # Values, purpose, audience
    MARKETING_COPY = "marketing_copy"     # Example brand voice
    FAQ_REFERENCE = "faq_reference"       # Existing FAQ patterns
    TERMINOLOGY_GLOSSARY = "terminology"  # Preferred terms, jargon
    COMPETITOR_ANALYSIS = "competitor"    # Differentiation points
    UNKNOWN = "unknown"                   # General brand information
```

### 1.3 Extraction Pipeline

```
Brand Document Input
        |
        v
+-------------------+
| Document Parser   |  --> Raw text extraction
+-------------------+
        |
        v
+-------------------+
| Type Classifier   |  --> Determine document category
+-------------------+
        |
        v
+-------------------+
| Section Extractor |  --> Identify key sections (About, Products, etc.)
+-------------------+
        |
        v
+-------------------+
| Entity Extractor  |  --> spaCy NER for products, services, people
+-------------------+
        |
        v
+-------------------+
| Tone Analyzer     |  --> Formality, complexity, sentiment analysis
+-------------------+
        |
        v
+-------------------+
| Term Extractor    |  --> Identify preferred terminology
+-------------------+
        |
        v
BrandDocumentContext
```

### 1.4 Key Information Extraction

#### 1.4.1 Tone Extraction

Tone is extracted through multiple signals:

```python
class ToneProfile:
    """Represents the brand's communication tone"""
    formality: float          # 0.0 (casual) to 1.0 (formal)
    technicality: float       # 0.0 (simple) to 1.0 (technical)
    warmth: float             # 0.0 (professional/distant) to 1.0 (friendly)
    confidence: float         # 0.0 (hedging) to 1.0 (assertive)
    humor: float              # 0.0 (serious) to 1.0 (playful)

    # Explicit indicators (if found in style guide)
    explicit_descriptors: List[str]  # e.g., ["professional", "approachable"]
    avoid_words: List[str]           # Words the brand avoids
    prefer_words: List[str]          # Words the brand prefers
```

**Tone Detection Signals:**

| Signal | Detection Method | Weight |
|--------|------------------|--------|
| Explicit style guide rules | Pattern matching for "We are...", "Our tone is..." | High |
| Sentence complexity | Flesch-Kincaid readability score | Medium |
| First-person usage | Pronoun analysis (we/I vs. the company) | Medium |
| Contraction usage | Presence of "don't", "we're", etc. | Low |
| Exclamation frequency | Punctuation analysis | Low |
| Technical jargon density | Domain-specific term frequency | Medium |

#### 1.4.2 Terminology Extraction

```python
class TerminologyPreferences:
    """Preferred terms and phrases for the brand"""

    # Direct mappings: generic term -> brand-preferred term
    term_mappings: Dict[str, str]
    # Example: {"customers": "clients", "buy": "invest in", "cheap": "affordable"}

    # Product/service names (must use exact capitalization)
    proper_nouns: List[str]
    # Example: ["ServicePro Plus", "QuickStart Package"]

    # Industry jargon the brand uses
    jargon_terms: List[str]
    # Example: ["SaaS", "ARR", "churn rate"]

    # Terms to avoid
    blacklist_terms: List[str]
    # Example: ["cheap", "basic", "simple"]

    # Acronyms and their expansions
    acronyms: Dict[str, str]
    # Example: {"CRM": "Customer Relationship Management"}
```

**Terminology Extraction Logic:**

1. **Explicit glossaries**: Parse tables or lists with term definitions
2. **Repeated phrases**: Identify phrases used 3+ times (likely intentional)
3. **Capitalized terms**: Product names, service tiers
4. **Quoted terms**: Terms in quotes often indicate preferred usage
5. **Negation patterns**: "We don't say X, we say Y" patterns

#### 1.4.3 Products and Services Extraction

```python
class ProductServiceCatalog:
    """Extracted products and services"""

    products: List[ProductInfo]
    services: List[ServiceInfo]

class ProductInfo:
    name: str
    description: str
    key_features: List[str]
    target_audience: Optional[str]
    price_tier: Optional[str]  # "budget", "mid", "premium", "enterprise"

class ServiceInfo:
    name: str
    description: str
    deliverables: List[str]
    target_audience: Optional[str]
    service_type: str  # "consulting", "implementation", "support", "managed"
```

**Extraction Patterns:**

- Section headers: "Our Products", "Services", "What We Offer"
- Bullet lists under product/service sections
- Price/tier mentions with associated features
- spaCy NER for ORG and PRODUCT entities

#### 1.4.4 Value Propositions

```python
class ValuePropositions:
    """Core value propositions and differentiators"""

    primary_value: str                    # Main value proposition
    supporting_values: List[str]          # Secondary benefits
    differentiators: List[str]            # What sets them apart
    pain_points_addressed: List[str]      # Problems they solve
    proof_points: List[str]               # Evidence/stats supporting claims
```

**Detection Patterns:**

- "Why choose us" sections
- Comparative language ("unlike competitors", "the only solution that")
- Benefit statements following "so you can", "which means"
- Statistics and testimonial patterns

---

## 2. Content-Based Inference Pipeline

When brand documents are not provided, the system must infer business context from the source document alone.

### 2.1 Inference Architecture

```
Source Document Content
        |
        v
+------------------------+
| Text Preprocessing     |  --> Clean, normalize, segment
+------------------------+
        |
        v
+------------------------+
| Entity Extraction      |  --> spaCy NER pipeline
| (spaCy en_core_web_lg) |
+------------------------+
        |
        v
+------------------------+
| Topic Modeling         |  --> LDA/NMF for page subject
+------------------------+
        |
        v
+------------------------+
| Industry Classifier    |  --> Predict industry from content
+------------------------+
        |
        v
+------------------------+
| Confidence Aggregator  |  --> Score overall inference quality
+------------------------+
        |
        v
InferredBusinessContext
```

### 2.2 Entity Extraction (spaCy-Based)

#### 2.2.1 Entity Types and Business Signals

| spaCy Entity | Business Signal | Inference |
|--------------|-----------------|-----------|
| ORG | Company names, competitors | Business identity, market position |
| PRODUCT | Products mentioned | Product catalog |
| MONEY | Prices, revenue | Price tier, market segment |
| PERSON | Team members, founders | Company size, personalization |
| GPE (Location) | Service areas | Geographic scope |
| DATE | Founding date, milestones | Company maturity |
| CARDINAL | Stats, quantities | Scale of operations |

#### 2.2.2 Custom Entity Patterns

Beyond standard NER, add custom patterns for business-specific entities:

```python
from spacy.matcher import Matcher

# Custom patterns for business context
BUSINESS_PATTERNS = {
    "SERVICE_OFFERING": [
        [{"LOWER": {"IN": ["we", "our"]}}, {"LOWER": {"IN": ["provide", "offer", "deliver"]}}, {"POS": "NOUN", "OP": "+"}],
        [{"LOWER": "services"}, {"LOWER": "include"}, {"POS": "NOUN", "OP": "+"}],
    ],
    "TARGET_AUDIENCE": [
        [{"LOWER": "for"}, {"POS": "NOUN", "OP": "+"}, {"LOWER": {"IN": ["who", "that"]}}],
        [{"LOWER": {"IN": ["helping", "serving"]}}, {"POS": "NOUN", "OP": "+"}],
    ],
    "VALUE_PROP": [
        [{"LOWER": {"IN": ["save", "reduce", "increase", "improve"]}}, {"POS": "NOUN", "OP": "+"}],
        [{"LOWER": "without"}, {"POS": "VERB"}, {"POS": "NOUN", "OP": "+"}],
    ],
}
```

### 2.3 Topic Modeling for Page Subject

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class TopicExtractor:
    """Extract main topics from document content"""

    def extract_topics(self, text: str, n_topics: int = 3) -> List[TopicInfo]:
        """
        Returns primary topics with confidence scores
        """
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf = vectorizer.fit_transform([text])

        # NMF for topic extraction
        nmf = NMF(n_components=n_topics, random_state=42)
        nmf.fit(tfidf)

        # Extract top terms per topic
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_terms = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append(TopicInfo(
                terms=top_terms,
                weight=float(topic.max()),
                label=self._label_topic(top_terms)
            ))

        return topics

    def _label_topic(self, terms: List[str]) -> str:
        """Attempt to create human-readable topic label"""
        # Use first noun phrase or most distinctive term
        return terms[0].title()
```

### 2.4 Industry Classification

```python
class IndustryClassifier:
    """Classify content into industry categories"""

    # Industry indicators: keywords that strongly signal industry
    INDUSTRY_SIGNALS = {
        "technology": {
            "strong": ["software", "api", "cloud", "saas", "platform", "integration"],
            "moderate": ["digital", "automation", "data", "analytics", "ai"],
            "weak": ["solution", "system", "technology"]
        },
        "healthcare": {
            "strong": ["patient", "clinical", "hipaa", "medical", "healthcare"],
            "moderate": ["health", "care", "treatment", "diagnosis"],
            "weak": ["wellness", "provider", "practice"]
        },
        "finance": {
            "strong": ["banking", "investment", "portfolio", "fintech", "trading"],
            "moderate": ["financial", "capital", "asset", "fund"],
            "weak": ["money", "payment", "account"]
        },
        "ecommerce": {
            "strong": ["shopping cart", "checkout", "inventory", "sku"],
            "moderate": ["product", "shipping", "order", "catalog"],
            "weak": ["buy", "sell", "store"]
        },
        "professional_services": {
            "strong": ["consulting", "advisory", "engagement", "deliverable"],
            "moderate": ["expertise", "client", "project", "strategy"],
            "weak": ["service", "support", "help"]
        },
        "manufacturing": {
            "strong": ["production", "assembly", "cnc", "supply chain"],
            "moderate": ["manufacturing", "quality control", "warehouse"],
            "weak": ["product", "material", "equipment"]
        },
        "real_estate": {
            "strong": ["property", "listing", "mls", "mortgage"],
            "moderate": ["real estate", "home", "commercial", "lease"],
            "weak": ["space", "location", "building"]
        },
        "education": {
            "strong": ["curriculum", "enrollment", "lms", "student"],
            "moderate": ["learning", "course", "training", "certification"],
            "weak": ["education", "teach", "skill"]
        }
    }

    def classify(self, text: str) -> IndustryClassification:
        """
        Returns industry classification with confidence
        """
        text_lower = text.lower()
        scores = {}

        for industry, signals in self.INDUSTRY_SIGNALS.items():
            score = 0
            matches = []

            for term in signals["strong"]:
                if term in text_lower:
                    score += 3
                    matches.append((term, "strong"))

            for term in signals["moderate"]:
                if term in text_lower:
                    score += 2
                    matches.append((term, "moderate"))

            for term in signals["weak"]:
                if term in text_lower:
                    score += 1
                    matches.append((term, "weak"))

            scores[industry] = {"score": score, "matches": matches}

        # Determine primary industry
        sorted_industries = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

        if sorted_industries[0][1]["score"] == 0:
            return IndustryClassification(
                primary="general",
                confidence=0.3,
                reasoning="No strong industry signals detected"
            )

        primary = sorted_industries[0]
        secondary = sorted_industries[1] if len(sorted_industries) > 1 else None

        # Calculate confidence based on score differential
        confidence = min(0.9, primary[1]["score"] / 20)
        if secondary and secondary[1]["score"] > primary[1]["score"] * 0.7:
            confidence *= 0.8  # Reduce confidence if close competitor

        return IndustryClassification(
            primary=primary[0],
            secondary=secondary[0] if secondary else None,
            confidence=confidence,
            evidence=primary[1]["matches"],
            reasoning=f"Detected {len(primary[1]['matches'])} industry signals"
        )
```

### 2.5 Confidence Scoring

```python
class ContextConfidenceScorer:
    """Calculate confidence scores for inferred context"""

    CONFIDENCE_WEIGHTS = {
        "brand_doc_provided": 0.4,      # Brand docs dramatically increase confidence
        "entity_density": 0.15,          # More entities = more confident
        "industry_signal_strength": 0.2, # Clear industry indicators
        "content_length": 0.1,           # More content = more data to analyze
        "topic_coherence": 0.15          # Clear, focused topics
    }

    def calculate_confidence(
        self,
        brand_docs_provided: bool,
        entity_count: int,
        content_length: int,
        industry_confidence: float,
        topic_coherence: float
    ) -> ContextConfidence:
        """
        Returns overall confidence score with breakdown
        """
        scores = {}

        # Brand doc contribution
        scores["brand_doc"] = 1.0 if brand_docs_provided else 0.0

        # Entity density (normalized)
        entity_density = min(1.0, entity_count / 50)  # 50+ entities = max
        scores["entity_density"] = entity_density

        # Content length (normalized)
        length_score = min(1.0, content_length / 5000)  # 5000+ chars = max
        scores["content_length"] = length_score

        # Industry confidence (passed through)
        scores["industry"] = industry_confidence

        # Topic coherence (passed through)
        scores["topic_coherence"] = topic_coherence

        # Weighted average
        total = sum(
            scores[k.replace("_strength", "").replace("_provided", "").replace("_density", "_density")]
            * v for k, v in self.CONFIDENCE_WEIGHTS.items()
            if k.replace("_strength", "").replace("_provided", "").replace("_density", "_density") in scores
        )

        # Simplified calculation
        weighted_total = (
            scores["brand_doc"] * self.CONFIDENCE_WEIGHTS["brand_doc_provided"] +
            scores["entity_density"] * self.CONFIDENCE_WEIGHTS["entity_density"] +
            scores["industry"] * self.CONFIDENCE_WEIGHTS["industry_signal_strength"] +
            scores["content_length"] * self.CONFIDENCE_WEIGHTS["content_length"] +
            scores["topic_coherence"] * self.CONFIDENCE_WEIGHTS["topic_coherence"]
        )

        return ContextConfidence(
            overall=weighted_total,
            breakdown=scores,
            level=self._confidence_level(weighted_total)
        )

    def _confidence_level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.3:
            return "low"
        else:
            return "very_low"
```

---

## 3. BusinessContext Model Specification

### 3.1 Complete Model Definition

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"   # < 0.3: Use maximum caution, generic content
    LOW = "low"             # 0.3-0.5: Basic inference, conservative tone
    MEDIUM = "medium"       # 0.5-0.8: Reasonable inference, normal generation
    HIGH = "high"           # > 0.8: Strong confidence, full brand alignment

@dataclass
class ToneProfile:
    """Quantified representation of brand voice"""

    # Core dimensions (0.0 to 1.0 scale)
    formality: float = 0.5          # casual <-> formal
    technicality: float = 0.5       # simple <-> technical
    warmth: float = 0.5             # distant <-> friendly
    confidence: float = 0.5         # hedging <-> assertive
    humor: float = 0.2              # serious <-> playful

    # Explicit guidelines (from brand docs)
    explicit_descriptors: List[str] = field(default_factory=list)
    avoid_patterns: List[str] = field(default_factory=list)
    prefer_patterns: List[str] = field(default_factory=list)

    # Derived settings
    use_contractions: bool = True
    use_first_person: bool = True   # "we" vs "the company"
    use_second_person: bool = True  # "you" vs "customers"
    max_sentence_length: int = 25   # words

    def to_prompt_instructions(self) -> str:
        """Convert tone profile to LLM prompt instructions"""
        instructions = []

        if self.formality > 0.7:
            instructions.append("Use formal, professional language.")
        elif self.formality < 0.3:
            instructions.append("Use casual, conversational language.")

        if self.technicality > 0.7:
            instructions.append("Include technical terminology appropriate to the industry.")
        elif self.technicality < 0.3:
            instructions.append("Avoid jargon; use simple, accessible language.")

        if self.warmth > 0.7:
            instructions.append("Be warm and personable; connect with the reader.")
        elif self.warmth < 0.3:
            instructions.append("Maintain professional distance.")

        if self.confidence > 0.7:
            instructions.append("Be direct and assertive; avoid hedging language.")
        elif self.confidence < 0.3:
            instructions.append("Use measured language; acknowledge limitations.")

        if not self.use_contractions:
            instructions.append("Do not use contractions.")

        if self.explicit_descriptors:
            instructions.append(f"The brand voice is: {', '.join(self.explicit_descriptors)}.")

        if self.avoid_patterns:
            instructions.append(f"Avoid these patterns: {', '.join(self.avoid_patterns[:5])}.")

        return " ".join(instructions)


@dataclass
class TerminologyPreferences:
    """Brand-specific language preferences"""

    # Term mappings: generic -> preferred
    preferred_terms: Dict[str, str] = field(default_factory=dict)
    # Example: {"customers": "clients", "buy": "invest in"}

    # Must-use exact terms (product names, etc.)
    exact_terms: List[str] = field(default_factory=list)

    # Terms to never use
    forbidden_terms: List[str] = field(default_factory=list)

    # Acronyms with expansions
    acronyms: Dict[str, str] = field(default_factory=dict)

    # Industry jargon that's acceptable
    accepted_jargon: List[str] = field(default_factory=list)

    def apply_preferences(self, text: str) -> str:
        """Apply terminology preferences to text"""
        result = text
        for generic, preferred in self.preferred_terms.items():
            # Case-insensitive replacement preserving original case
            import re
            pattern = re.compile(re.escape(generic), re.IGNORECASE)
            result = pattern.sub(preferred, result)
        return result

    def validate_text(self, text: str) -> List[str]:
        """Return list of forbidden terms found in text"""
        violations = []
        text_lower = text.lower()
        for term in self.forbidden_terms:
            if term.lower() in text_lower:
                violations.append(term)
        return violations


@dataclass
class ProductService:
    """Represents a product or service offering"""
    name: str
    description: str = ""
    category: str = "general"           # product, service, feature, tier
    features: List[str] = field(default_factory=list)
    target_audience: str = ""
    price_tier: str = "unspecified"     # budget, mid, premium, enterprise
    is_primary: bool = False            # Core offering vs. add-on


@dataclass
class AudienceProfile:
    """Target audience characteristics"""
    primary_audience: str = ""          # Main target (e.g., "small business owners")
    secondary_audiences: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    expertise_level: str = "general"    # novice, intermediate, expert, mixed


@dataclass
class ContextConfidence:
    """Confidence metrics for context inference"""
    overall: float = 0.5
    breakdown: Dict[str, float] = field(default_factory=dict)
    level: str = "medium"


@dataclass
class BusinessContext:
    """
    Complete business context model.

    This is the primary output of the context module, consumed by
    downstream modules (FAQ generation, content enhancement) to
    ensure generated content aligns with brand identity.
    """

    # === IDENTIFICATION ===
    business_name: str = ""
    business_name_variations: List[str] = field(default_factory=list)
    # Example: ["Acme Corp", "Acme", "Acme Corporation"]

    # === CLASSIFICATION ===
    industry: str = "general"
    industry_secondary: Optional[str] = None
    business_type: str = "unknown"      # b2b, b2c, b2b2c, nonprofit, government
    company_size: str = "unknown"       # startup, smb, midmarket, enterprise

    # === OFFERINGS ===
    products_services: List[ProductService] = field(default_factory=list)
    primary_offering_summary: str = ""  # One-sentence description

    # === AUDIENCE ===
    target_audience: AudienceProfile = field(default_factory=AudienceProfile)

    # === VOICE & LANGUAGE ===
    tone: ToneProfile = field(default_factory=ToneProfile)
    terminology: TerminologyPreferences = field(default_factory=TerminologyPreferences)

    # === VALUE PROPS ===
    value_propositions: List[str] = field(default_factory=list)
    differentiators: List[str] = field(default_factory=list)

    # === METADATA ===
    confidence: ContextConfidence = field(default_factory=ContextConfidence)
    source: str = "inferred"            # "brand_docs", "inferred", "hybrid"
    geographic_scope: str = "unspecified"  # local, regional, national, global

    # === CONTEXT SOURCE TRACKING ===
    brand_docs_processed: List[str] = field(default_factory=list)
    inference_evidence: Dict[str, List[str]] = field(default_factory=dict)

    def get_safe_business_name(self) -> str:
        """Return business name or safe fallback"""
        if self.business_name:
            return self.business_name
        return "the business"

    def get_primary_products(self) -> List[ProductService]:
        """Return only primary product/service offerings"""
        return [ps for ps in self.products_services if ps.is_primary]

    def should_use_technical_language(self) -> bool:
        """Determine if technical language is appropriate"""
        return (
            self.tone.technicality > 0.5 or
            self.target_audience.expertise_level in ["intermediate", "expert"]
        )

    def to_generation_prompt(self) -> str:
        """
        Convert context to prompt instructions for content generation.
        Used by FAQ generator and content enhancer.
        """
        parts = []

        # Business identity
        if self.business_name:
            parts.append(f"You are writing content for {self.business_name}.")

        # Industry context
        if self.industry != "general":
            parts.append(f"This is a {self.industry} business.")

        # Audience
        if self.target_audience.primary_audience:
            parts.append(f"The target audience is {self.target_audience.primary_audience}.")

        # Offerings
        if self.primary_offering_summary:
            parts.append(f"The business offers: {self.primary_offering_summary}")

        # Tone instructions
        parts.append(self.tone.to_prompt_instructions())

        # Value props to emphasize
        if self.value_propositions:
            parts.append(f"Key value propositions to emphasize: {', '.join(self.value_propositions[:3])}.")

        # Confidence caveat
        if self.confidence.level == "low" or self.confidence.level == "very_low":
            parts.append("Note: Context confidence is low. Use conservative, generic language when uncertain.")

        return " ".join(parts)


# === FACTORY FUNCTIONS ===

def create_default_context() -> BusinessContext:
    """Create a default context with safe assumptions"""
    return BusinessContext(
        business_name="",
        industry="general",
        business_type="unknown",
        tone=ToneProfile(
            formality=0.5,
            technicality=0.5,
            warmth=0.5,
            confidence=0.5,
            humor=0.2
        ),
        confidence=ContextConfidence(
            overall=0.3,
            level="low"
        ),
        source="default"
    )


def create_context_from_brand_docs(
    brand_docs: List[BrandDocumentContext]
) -> BusinessContext:
    """
    Create context primarily from brand documents.
    High confidence path.
    """
    # Implementation merges all brand doc contexts
    # Handles conflicts via priority rules
    pass


def create_context_from_inference(
    entities: List[ExtractedEntity],
    topics: List[TopicInfo],
    industry: IndustryClassification,
    source_text: str
) -> BusinessContext:
    """
    Create context from content inference.
    Lower confidence path.
    """
    # Implementation builds context from NLP results
    pass
```

### 3.2 Field Requirements

| Field | Required | Default | Source Priority |
|-------|----------|---------|-----------------|
| `business_name` | No | `""` | Brand docs > H1 > Domain |
| `industry` | No | `"general"` | Brand docs > Classification > None |
| `business_type` | No | `"unknown"` | Brand docs > Inference |
| `products_services` | No | `[]` | Brand docs > Entity extraction |
| `target_audience` | No | Empty profile | Brand docs > Content inference |
| `tone` | No | Neutral defaults | Brand docs > Content analysis |
| `terminology` | No | Empty preferences | Brand docs only |
| `value_propositions` | No | `[]` | Brand docs > Content extraction |
| `confidence` | Yes | Calculated | Automatic |
| `source` | Yes | `"inferred"` | Automatic |

---

## 4. Fallback Priority Matrix

### 4.1 Complete Fallback Table

| Information | Priority 1 (Brand Docs) | Priority 2 (Content) | Priority 3 (Default) | Notes |
|-------------|-------------------------|----------------------|----------------------|-------|
| **Business name** | Explicit name from brand doc | First H1 heading / Most frequent ORG entity | `"the business"` | Never fabricate a name |
| **Industry** | Explicit industry statement | Industry classifier result | `"general"` | Use classifier confidence threshold (>0.5) |
| **Business type** | "B2B", "B2C" statements | Inference from audience language | `"unknown"` | Look for "enterprise", "consumer" signals |
| **Products/Services** | Product catalog document | PRODUCT entities + service patterns | `[]` | Only include if confidence > 0.6 |
| **Target audience** | Explicit audience statements | "for [audience]" patterns | Generic profile | Conservative when uncertain |
| **Formality** | Style guide rules | Sentence complexity analysis | `0.5` (neutral) | Flesch-Kincaid scoring |
| **Technicality** | Style guide rules | Technical term density | `0.5` (neutral) | Count jargon/acronyms |
| **Warmth** | Style guide rules | First-person usage, contractions | `0.5` (neutral) | Pronoun analysis |
| **Preferred terms** | Terminology glossary | Repeated exact phrases | `{}` | Only from brand docs |
| **Forbidden terms** | Style guide blacklist | N/A | `[]` | Only from brand docs |
| **Value propositions** | Mission/value statements | "We help you..." patterns | `[]` | Extract max 5 |
| **Geographic scope** | Explicit statements | GPE entity analysis | `"unspecified"` | Look for location patterns |

### 4.2 Conflict Resolution Rules

When multiple sources provide conflicting information:

```python
class ConflictResolver:
    """Resolve conflicts between context sources"""

    RESOLUTION_RULES = {
        # Rule: (conflict_type, resolution_strategy)
        "business_name": "brand_doc_wins",      # Brand doc always authoritative
        "industry": "highest_confidence",        # Use most confident source
        "tone_attributes": "brand_doc_wins",     # Brand doc always authoritative
        "products": "merge_unique",              # Combine from all sources
        "terminology": "brand_doc_wins",         # Brand doc always authoritative
        "audience": "brand_doc_primary",         # Brand doc primary, content supplements
    }

    def resolve(
        self,
        field: str,
        brand_value: Any,
        inferred_value: Any,
        brand_confidence: float,
        inferred_confidence: float
    ) -> Tuple[Any, str]:
        """
        Returns (resolved_value, resolution_reason)
        """
        rule = self.RESOLUTION_RULES.get(field, "highest_confidence")

        if rule == "brand_doc_wins":
            if brand_value is not None:
                return (brand_value, "brand_doc_authoritative")
            return (inferred_value, "brand_doc_missing_fallback_to_inference")

        elif rule == "highest_confidence":
            if brand_confidence >= inferred_confidence:
                return (brand_value, "brand_higher_confidence")
            return (inferred_value, "inference_higher_confidence")

        elif rule == "merge_unique":
            # Combine lists, remove duplicates
            merged = list(set(brand_value or []) | set(inferred_value or []))
            return (merged, "merged_unique_values")

        elif rule == "brand_doc_primary":
            # Use brand doc, supplement with inference
            if brand_value:
                return (brand_value, "brand_doc_primary")
            return (inferred_value, "inference_supplementary")

        return (inferred_value, "default_to_inference")
```

### 4.3 Confidence-Based Behavior

```python
def adjust_generation_behavior(context: BusinessContext) -> GenerationConfig:
    """
    Adjust generation parameters based on context confidence.
    """
    config = GenerationConfig()

    if context.confidence.level == "high":
        # Full brand alignment
        config.use_specific_terminology = True
        config.match_tone_precisely = True
        config.include_product_mentions = True
        config.assertiveness = "normal"

    elif context.confidence.level == "medium":
        # Moderate brand alignment
        config.use_specific_terminology = True
        config.match_tone_precisely = False  # Allow some variation
        config.include_product_mentions = True
        config.assertiveness = "normal"

    elif context.confidence.level == "low":
        # Conservative approach
        config.use_specific_terminology = False  # Generic terms only
        config.match_tone_precisely = False
        config.include_product_mentions = False  # Don't mention products by name
        config.assertiveness = "hedged"  # Use "may", "can", "often"

    else:  # very_low
        # Maximum caution
        config.use_specific_terminology = False
        config.match_tone_precisely = False
        config.include_product_mentions = False
        config.assertiveness = "very_hedged"
        config.add_disclaimer = True  # "This content may need review"

    return config
```

---

## 5. Integration with Downstream Modules

### 5.1 FAQ Generation Integration

The FAQ generator consumes BusinessContext to produce contextually appropriate questions and answers.

```python
class FAQGenerator:
    """Generate FAQ section using business context"""

    def generate_faqs(
        self,
        context: BusinessContext,
        page_content: str,
        keywords: List[str],
        num_faqs: int = 5
    ) -> List[FAQItem]:
        """
        Generate FAQs aligned with business context.
        """
        # Build generation prompt
        system_prompt = self._build_system_prompt(context)

        # Generate questions based on context
        questions = self._generate_questions(
            context=context,
            page_content=page_content,
            keywords=keywords
        )

        # Generate answers with appropriate tone
        faqs = []
        for question in questions[:num_faqs]:
            answer = self._generate_answer(
                question=question,
                context=context,
                page_content=page_content
            )
            faqs.append(FAQItem(question=question, answer=answer))

        # Validate against context
        validated_faqs = self._validate_faqs(faqs, context)

        return validated_faqs

    def _build_system_prompt(self, context: BusinessContext) -> str:
        """Build LLM system prompt from context"""
        prompt_parts = [
            "You are generating FAQ content for a website.",
            context.to_generation_prompt(),
        ]

        # Add terminology constraints
        if context.terminology.forbidden_terms:
            prompt_parts.append(
                f"Never use these terms: {', '.join(context.terminology.forbidden_terms)}"
            )

        # Add confidence-based instructions
        if context.confidence.level in ["low", "very_low"]:
            prompt_parts.append(
                "Use generic language. Do not make specific claims about "
                "the business that cannot be verified from the source content."
            )

        return "\n".join(prompt_parts)

    def _generate_questions(
        self,
        context: BusinessContext,
        page_content: str,
        keywords: List[str]
    ) -> List[str]:
        """Generate relevant questions based on context"""
        question_templates = []

        # Industry-specific question patterns
        if context.industry == "technology":
            question_templates.extend([
                f"How does {context.get_safe_business_name()} integrate with existing systems?",
                f"What security measures does {context.get_safe_business_name()} have?",
                "Is there an API available?",
            ])
        elif context.industry == "healthcare":
            question_templates.extend([
                f"Is {context.get_safe_business_name()} HIPAA compliant?",
                "How is patient data protected?",
            ])
        elif context.industry == "ecommerce":
            question_templates.extend([
                "What is the return policy?",
                "How long does shipping take?",
                "What payment methods are accepted?",
            ])

        # Audience-specific questions
        if context.target_audience.expertise_level == "novice":
            question_templates.append("How do I get started?")
            question_templates.append("Is training provided?")

        # Product-specific questions
        for product in context.get_primary_products()[:2]:
            question_templates.append(f"What is {product.name}?")
            question_templates.append(f"Who is {product.name} best suited for?")

        # Keyword-based questions
        for keyword in keywords[:3]:
            question_templates.append(f"What is {keyword}?")
            question_templates.append(f"How does {keyword} work?")

        return question_templates

    def _validate_faqs(
        self,
        faqs: List[FAQItem],
        context: BusinessContext
    ) -> List[FAQItem]:
        """Validate generated FAQs against context constraints"""
        validated = []

        for faq in faqs:
            # Check terminology violations
            violations = context.terminology.validate_text(faq.answer)
            if violations:
                # Rewrite answer to avoid forbidden terms
                faq.answer = context.terminology.apply_preferences(faq.answer)

            # Verify business name usage
            if context.business_name:
                # Ensure consistent naming
                for variation in context.business_name_variations:
                    if variation != context.business_name:
                        faq.answer = faq.answer.replace(
                            variation, context.business_name
                        )

            validated.append(faq)

        return validated
```

### 5.2 Content Enhancement Integration

```python
class ContentEnhancer:
    """Enhance existing content using business context"""

    def enhance_section(
        self,
        section: ContentNode,
        context: BusinessContext,
        keywords: List[str]
    ) -> EnhancedContent:
        """
        Enhance a content section while respecting brand context.
        """
        # Determine enhancement approach based on confidence
        approach = self._determine_approach(context)

        if approach == "full_enhancement":
            return self._full_enhance(section, context, keywords)
        elif approach == "conservative_enhancement":
            return self._conservative_enhance(section, context, keywords)
        else:
            return self._minimal_enhancement(section, keywords)

    def _determine_approach(self, context: BusinessContext) -> str:
        """Determine enhancement approach based on context confidence"""
        if context.confidence.overall >= 0.7:
            return "full_enhancement"
        elif context.confidence.overall >= 0.4:
            return "conservative_enhancement"
        else:
            return "minimal_enhancement"

    def _full_enhance(
        self,
        section: ContentNode,
        context: BusinessContext,
        keywords: List[str]
    ) -> EnhancedContent:
        """
        Full enhancement with brand alignment.
        - Add keyword-rich expansions
        - Use brand terminology
        - Match exact tone
        """
        enhancements = []

        # Generate expansions using context
        prompt = f"""
        {context.to_generation_prompt()}

        Enhance the following content by adding 1-2 sentences that:
        1. Naturally incorporate these keywords: {', '.join(keywords[:3])}
        2. Add value for the reader
        3. Match the existing tone and style

        Original content:
        {section.content}

        Provide only the additional sentences, not the original.
        """

        expansion = self._call_llm(prompt)

        # Apply terminology preferences
        expansion = context.terminology.apply_preferences(expansion)

        # Validate against forbidden terms
        if not context.terminology.validate_text(expansion):
            enhancements.append(Enhancement(
                type="expansion",
                content=expansion,
                position="after",
                confidence=0.9
            ))

        return EnhancedContent(
            original=section,
            enhancements=enhancements
        )

    def _conservative_enhance(
        self,
        section: ContentNode,
        context: BusinessContext,
        keywords: List[str]
    ) -> EnhancedContent:
        """
        Conservative enhancement.
        - Add brief keyword mentions
        - Use generic language
        - Avoid specific brand claims
        """
        # More cautious approach when confidence is medium
        prompt = f"""
        Add one brief, factual sentence to this content that naturally
        mentions: {keywords[0] if keywords else 'the topic'}.

        Keep the tone neutral and professional.
        Do not make specific claims about the business.

        Original: {section.content}
        """

        expansion = self._call_llm(prompt)

        return EnhancedContent(
            original=section,
            enhancements=[Enhancement(
                type="expansion",
                content=expansion,
                position="after",
                confidence=0.6
            )]
        )

    def _minimal_enhancement(
        self,
        section: ContentNode,
        keywords: List[str]
    ) -> EnhancedContent:
        """
        Minimal enhancement when context is unreliable.
        - Only add if absolutely necessary
        - Extremely generic language
        """
        # When confidence is very low, we may choose to not enhance at all
        return EnhancedContent(
            original=section,
            enhancements=[],
            skipped_reason="context_confidence_too_low"
        )
```

### 5.3 Validation Against Generated Content

```python
class ContentValidator:
    """Validate generated content against business context"""

    def validate(
        self,
        generated_content: str,
        context: BusinessContext
    ) -> ValidationResult:
        """
        Validate that generated content aligns with business context.
        """
        issues = []

        # 1. Terminology check
        forbidden_found = context.terminology.validate_text(generated_content)
        if forbidden_found:
            issues.append(ValidationIssue(
                type="forbidden_terminology",
                severity="high",
                details=f"Found forbidden terms: {forbidden_found}"
            ))

        # 2. Tone consistency check
        generated_tone = self._analyze_tone(generated_content)
        tone_drift = self._calculate_tone_drift(generated_tone, context.tone)
        if tone_drift > 0.3:  # 30% drift threshold
            issues.append(ValidationIssue(
                type="tone_inconsistency",
                severity="medium",
                details=f"Tone drift detected: {tone_drift:.2%}"
            ))

        # 3. Factual consistency check
        if context.confidence.level in ["medium", "high"]:
            # Only check facts when we have reliable context
            fact_issues = self._check_factual_consistency(
                generated_content, context
            )
            issues.extend(fact_issues)

        # 4. Business name consistency
        if context.business_name:
            name_issues = self._check_name_consistency(
                generated_content, context
            )
            issues.extend(name_issues)

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity == "high"]) == 0,
            issues=issues,
            suggested_fixes=self._generate_fixes(issues, context)
        )

    def _calculate_tone_drift(
        self,
        generated_tone: ToneProfile,
        target_tone: ToneProfile
    ) -> float:
        """Calculate how much generated tone differs from target"""
        dimensions = ['formality', 'technicality', 'warmth', 'confidence', 'humor']
        diffs = []

        for dim in dimensions:
            gen_val = getattr(generated_tone, dim)
            target_val = getattr(target_tone, dim)
            diffs.append(abs(gen_val - target_val))

        return sum(diffs) / len(diffs)

    def _check_factual_consistency(
        self,
        content: str,
        context: BusinessContext
    ) -> List[ValidationIssue]:
        """Check for factual inconsistencies with known context"""
        issues = []

        # Check industry-specific claims
        if context.industry == "healthcare":
            # Look for compliance claims that need verification
            if "hipaa" in content.lower() and "hipaa" not in context.inference_evidence.get("compliance", []):
                issues.append(ValidationIssue(
                    type="unverified_claim",
                    severity="high",
                    details="HIPAA compliance claim not verified in source"
                ))

        # Check product mentions match known products
        for product in context.products_services:
            # Verify product descriptions match
            pass  # Implementation depends on specific checks needed

        return issues
```

---

## 6. Testing Strategy

### 6.1 Test Categories

#### 6.1.1 Brand Document Processing Tests

```python
class TestBrandDocumentProcessing:
    """Tests for brand document parsing and extraction"""

    def test_style_guide_tone_extraction(self):
        """Test extraction of tone from explicit style guide"""
        style_guide = """
        Our Brand Voice
        ---------------
        We are professional yet approachable. We use clear,
        simple language and avoid jargon. Our tone is confident
        but never arrogant.

        Do: Use contractions (we're, you'll)
        Don't: Use corporate buzzwords
        """

        context = process_brand_document(style_guide, BrandDocumentType.STYLE_GUIDE)

        assert context.tone.formality < 0.5  # "approachable" = less formal
        assert context.tone.confidence > 0.6  # "confident"
        assert context.tone.use_contractions == True
        assert "buzzwords" in context.tone.avoid_patterns

    def test_product_catalog_extraction(self):
        """Test extraction of products from catalog document"""
        catalog = """
        Our Products

        ServicePro Basic - $29/month
        Perfect for small teams. Includes:
        - 5 user seats
        - Basic reporting
        - Email support

        ServicePro Plus - $99/month
        For growing businesses. Includes everything in Basic plus:
        - Unlimited users
        - Advanced analytics
        - Priority support
        """

        context = process_brand_document(catalog, BrandDocumentType.PRODUCT_CATALOG)

        assert len(context.products_services) == 2
        assert context.products_services[0].name == "ServicePro Basic"
        assert context.products_services[0].price_tier == "budget"
        assert context.products_services[1].name == "ServicePro Plus"
        assert "Advanced analytics" in context.products_services[1].features

    def test_terminology_glossary_extraction(self):
        """Test extraction of terminology preferences"""
        glossary = """
        Terminology Guide

        Always say "clients" not "customers"
        Always say "investment" not "cost" or "price"
        Never say "cheap" - use "affordable" instead

        Product names (use exact capitalization):
        - DataSync Pro
        - QuickStart Package
        """

        context = process_brand_document(glossary, BrandDocumentType.TERMINOLOGY_GLOSSARY)

        assert context.terminology.preferred_terms["customers"] == "clients"
        assert context.terminology.preferred_terms["cost"] == "investment"
        assert "cheap" in context.terminology.forbidden_terms
        assert "DataSync Pro" in context.terminology.exact_terms
```

#### 6.1.2 Content Inference Tests

```python
class TestContentInference:
    """Tests for context inference without brand documents"""

    def test_technology_industry_classification(self):
        """Test industry classification for tech content"""
        content = """
        Our cloud-based SaaS platform helps businesses automate their
        workflow through powerful API integrations. Deploy in minutes
        with our Kubernetes-native architecture.
        """

        context = infer_context_from_content(content)

        assert context.industry == "technology"
        assert context.confidence.overall >= 0.6
        assert context.tone.technicality > 0.5

    def test_healthcare_industry_classification(self):
        """Test industry classification for healthcare content"""
        content = """
        Our HIPAA-compliant patient portal enables secure communication
        between healthcare providers and patients. Clinical staff can
        review treatment histories and manage appointments efficiently.
        """

        context = infer_context_from_content(content)

        assert context.industry == "healthcare"
        assert "hipaa" in [e.lower() for e in context.inference_evidence.get("compliance", [])]

    def test_business_name_extraction_from_h1(self):
        """Test business name extraction when no brand docs"""
        content = """
        # Welcome to Acme Solutions

        We provide innovative solutions for modern businesses.
        Acme Solutions has been serving clients since 2015.
        """

        context = infer_context_from_content(content, h1="Welcome to Acme Solutions")

        assert context.business_name == "Acme Solutions"

    def test_low_confidence_generic_content(self):
        """Test that generic content yields low confidence"""
        content = """
        We offer great products and services. Our team is dedicated
        to customer satisfaction. Contact us today to learn more.
        """

        context = infer_context_from_content(content)

        assert context.confidence.level in ["low", "very_low"]
        assert context.industry == "general"

    def test_entity_extraction_accuracy(self):
        """Test NER entity extraction"""
        content = """
        Founded in San Francisco by John Smith in 2020, TechCorp
        has raised $50 million in Series B funding. Our product,
        DataFlow Pro, serves over 500 enterprise customers including
        Microsoft and Google.
        """

        context = infer_context_from_content(content)

        # Should extract:
        assert "San Francisco" in context.geographic_scope or context.inference_evidence.get("locations", [])
        assert any("TechCorp" in ps.name or context.business_name == "TechCorp" for ps in context.products_services + [context])
```

#### 6.1.3 Conflict Resolution Tests

```python
class TestConflictResolution:
    """Tests for handling conflicting information"""

    def test_brand_doc_wins_for_business_name(self):
        """Brand doc business name should override inferred name"""
        brand_doc_context = BusinessContext(
            business_name="Acme Corporation",
            source="brand_docs"
        )
        inferred_context = BusinessContext(
            business_name="Acme Corp",  # Abbreviated version
            source="inferred"
        )

        merged = merge_contexts(brand_doc_context, inferred_context)

        assert merged.business_name == "Acme Corporation"

    def test_terminology_from_brand_only(self):
        """Terminology preferences should only come from brand docs"""
        brand_doc_context = BusinessContext(
            terminology=TerminologyPreferences(
                preferred_terms={"customers": "clients"},
                forbidden_terms=["cheap"]
            ),
            source="brand_docs"
        )
        inferred_context = BusinessContext(
            terminology=TerminologyPreferences(
                preferred_terms={"customers": "users"},  # Different inference
            ),
            source="inferred"
        )

        merged = merge_contexts(brand_doc_context, inferred_context)

        assert merged.terminology.preferred_terms["customers"] == "clients"
        assert "cheap" in merged.terminology.forbidden_terms

    def test_products_merge_unique(self):
        """Products from both sources should be merged"""
        brand_doc_context = BusinessContext(
            products_services=[
                ProductService(name="Product A", is_primary=True)
            ],
            source="brand_docs"
        )
        inferred_context = BusinessContext(
            products_services=[
                ProductService(name="Product A", is_primary=True),
                ProductService(name="Product B", is_primary=False)  # Additional
            ],
            source="inferred"
        )

        merged = merge_contexts(brand_doc_context, inferred_context)

        assert len(merged.products_services) == 2
        product_names = [p.name for p in merged.products_services]
        assert "Product A" in product_names
        assert "Product B" in product_names
```

#### 6.1.4 Default Fallback Tests

```python
class TestDefaultFallbacks:
    """Tests for default value fallbacks"""

    def test_default_context_creation(self):
        """Test that default context has safe values"""
        context = create_default_context()

        assert context.business_name == ""
        assert context.industry == "general"
        assert context.tone.formality == 0.5
        assert context.confidence.level == "low"

    def test_safe_business_name_fallback(self):
        """Test business name fallback for empty name"""
        context = BusinessContext(business_name="")

        assert context.get_safe_business_name() == "the business"

    def test_generation_prompt_with_low_confidence(self):
        """Test that low confidence adds caution to prompt"""
        context = BusinessContext(
            confidence=ContextConfidence(overall=0.2, level="very_low")
        )

        prompt = context.to_generation_prompt()

        assert "conservative" in prompt.lower() or "generic" in prompt.lower()
```

### 6.2 Integration Tests

```python
class TestContextIntegration:
    """Integration tests for context with downstream modules"""

    def test_faq_generation_respects_terminology(self):
        """Test that FAQ generation uses preferred terminology"""
        context = BusinessContext(
            terminology=TerminologyPreferences(
                preferred_terms={"customers": "clients"},
                forbidden_terms=["cheap"]
            ),
            confidence=ContextConfidence(overall=0.8, level="high")
        )

        faqs = generate_faqs(context, "Sample content about helping customers save money")

        for faq in faqs:
            assert "customers" not in faq.answer.lower()
            assert "cheap" not in faq.answer.lower()

    def test_content_enhancement_adjusts_to_confidence(self):
        """Test that enhancement approach changes with confidence"""
        high_conf_context = BusinessContext(
            confidence=ContextConfidence(overall=0.9, level="high")
        )
        low_conf_context = BusinessContext(
            confidence=ContextConfidence(overall=0.2, level="very_low")
        )

        high_conf_result = enhance_content("Test content", high_conf_context)
        low_conf_result = enhance_content("Test content", low_conf_context)

        # High confidence should produce more enhancements
        assert len(high_conf_result.enhancements) >= len(low_conf_result.enhancements)

    def test_validation_catches_tone_drift(self):
        """Test that validation catches tone inconsistencies"""
        context = BusinessContext(
            tone=ToneProfile(formality=0.9, humor=0.1)  # Very formal, serious
        )

        casual_content = "Hey there! We've got some awesome stuff for ya! :)"

        result = validate_content(casual_content, context)

        assert not result.is_valid
        assert any(issue.type == "tone_inconsistency" for issue in result.issues)
```

### 6.3 Validation Criteria

| Test Category | Minimum Pass Rate | Critical Tests |
|---------------|-------------------|----------------|
| Brand doc parsing | 95% | Tone extraction, terminology extraction |
| Industry classification | 85% | Technology, healthcare, finance |
| Entity extraction | 90% | Business names, products |
| Conflict resolution | 100% | Brand doc priority tests |
| Default fallbacks | 100% | All fallback scenarios |
| Integration tests | 90% | Terminology in generation |

---

## 7. Implementation Roadmap

### Phase 1: Core Context Model (Week 1)
- Implement `BusinessContext` dataclass and related models
- Implement `ToneProfile` with scoring methods
- Implement `TerminologyPreferences` with validation
- Create factory functions for context creation
- Unit tests for all models

### Phase 2: Brand Document Processing (Week 2)
- Implement document type classification
- Build extraction pipeline for each document type
- Tone extraction from style guides
- Terminology extraction from glossaries
- Product extraction from catalogs
- Integration tests

### Phase 3: Content Inference (Week 3)
- Integrate spaCy for entity extraction
- Implement industry classifier
- Build topic extraction module
- Implement confidence scoring
- End-to-end inference tests

### Phase 4: Integration (Week 4)
- Connect to FAQ generator
- Connect to content enhancer
- Implement content validator
- Full integration testing
- Performance optimization

---

## Appendix A: Example Context Objects

### A.1 High-Confidence Context (Brand Docs Provided)

```python
BusinessContext(
    business_name="Acme Solutions",
    business_name_variations=["Acme", "Acme Solutions Inc."],
    industry="technology",
    industry_secondary="professional_services",
    business_type="b2b",
    company_size="midmarket",
    products_services=[
        ProductService(
            name="AcmeFlow",
            description="Enterprise workflow automation platform",
            category="product",
            features=["API integration", "Custom workflows", "Analytics dashboard"],
            target_audience="Enterprise IT teams",
            price_tier="enterprise",
            is_primary=True
        ),
        ProductService(
            name="AcmeFlow Starter",
            description="Small business workflow tool",
            category="product",
            features=["Basic workflows", "5 integrations"],
            target_audience="Small businesses",
            price_tier="mid",
            is_primary=False
        )
    ],
    primary_offering_summary="Enterprise workflow automation with powerful API integrations",
    target_audience=AudienceProfile(
        primary_audience="IT managers and operations teams at mid-to-large enterprises",
        secondary_audiences=["CTOs", "System administrators"],
        pain_points=["Manual processes", "Integration complexity", "Visibility gaps"],
        goals=["Automation", "Efficiency", "Real-time insights"],
        expertise_level="intermediate"
    ),
    tone=ToneProfile(
        formality=0.7,
        technicality=0.8,
        warmth=0.4,
        confidence=0.8,
        humor=0.1,
        explicit_descriptors=["professional", "knowledgeable", "direct"],
        avoid_patterns=["synergy", "leverage", "circle back"],
        prefer_patterns=["integrate", "automate", "streamline"],
        use_contractions=False,
        use_first_person=True,
        use_second_person=True,
        max_sentence_length=20
    ),
    terminology=TerminologyPreferences(
        preferred_terms={
            "customers": "clients",
            "buy": "implement",
            "tool": "platform",
            "features": "capabilities"
        },
        exact_terms=["AcmeFlow", "AcmeFlow Starter", "FlowEngine"],
        forbidden_terms=["simple", "basic", "easy", "cheap"],
        acronyms={"API": "Application Programming Interface"},
        accepted_jargon=["REST API", "webhook", "OAuth", "SSO"]
    ),
    value_propositions=[
        "Reduce manual work by 70%",
        "Connect all your systems in one platform",
        "Enterprise-grade security and compliance"
    ],
    differentiators=[
        "Only platform with native SAP and Salesforce connectors",
        "99.99% uptime SLA",
        "Dedicated customer success manager for all accounts"
    ],
    confidence=ContextConfidence(
        overall=0.92,
        breakdown={
            "brand_doc": 1.0,
            "entity_density": 0.85,
            "industry": 0.95,
            "content_length": 0.9,
            "topic_coherence": 0.88
        },
        level="high"
    ),
    source="brand_docs",
    geographic_scope="global",
    brand_docs_processed=["brand_style_guide.docx", "product_catalog.pdf"],
    inference_evidence={}
)
```

### A.2 Low-Confidence Context (Inferred Only)

```python
BusinessContext(
    business_name="",
    business_name_variations=[],
    industry="general",
    industry_secondary=None,
    business_type="unknown",
    company_size="unknown",
    products_services=[],
    primary_offering_summary="",
    target_audience=AudienceProfile(
        primary_audience="",
        secondary_audiences=[],
        pain_points=[],
        goals=[],
        expertise_level="general"
    ),
    tone=ToneProfile(
        formality=0.5,
        technicality=0.5,
        warmth=0.5,
        confidence=0.5,
        humor=0.2,
        explicit_descriptors=[],
        avoid_patterns=[],
        prefer_patterns=[],
        use_contractions=True,
        use_first_person=True,
        use_second_person=True,
        max_sentence_length=25
    ),
    terminology=TerminologyPreferences(
        preferred_terms={},
        exact_terms=[],
        forbidden_terms=[],
        acronyms={},
        accepted_jargon=[]
    ),
    value_propositions=[],
    differentiators=[],
    confidence=ContextConfidence(
        overall=0.25,
        breakdown={
            "brand_doc": 0.0,
            "entity_density": 0.3,
            "industry": 0.2,
            "content_length": 0.4,
            "topic_coherence": 0.35
        },
        level="very_low"
    ),
    source="inferred",
    geographic_scope="unspecified",
    brand_docs_processed=[],
    inference_evidence={
        "entities_found": ["some company", "product"],
        "topic_terms": ["service", "help", "solution"]
    }
)
```

---

## Appendix B: Industry Classification Reference

| Industry | Strong Signals (3pt) | Moderate Signals (2pt) | Weak Signals (1pt) |
|----------|---------------------|------------------------|-------------------|
| Technology | software, api, cloud, saas, platform | digital, automation, data, ai | solution, system |
| Healthcare | patient, clinical, hipaa, medical | health, treatment, diagnosis | wellness, provider |
| Finance | banking, investment, fintech | financial, capital, asset | money, payment |
| E-commerce | shopping cart, checkout, sku | product, shipping, order | buy, sell, store |
| Prof. Services | consulting, advisory, engagement | expertise, client, strategy | service, support |
| Manufacturing | production, assembly, supply chain | warehouse, quality control | equipment, material |
| Real Estate | property, listing, mls, mortgage | real estate, commercial, lease | building, location |
| Education | curriculum, enrollment, lms | learning, course, training | education, skill |
| Legal | attorney, litigation, compliance | legal, contract, regulation | law, counsel |
| Hospitality | reservation, booking, guest | hotel, restaurant, travel | service, experience |

---

*Document Version: 1.0*
*Created: 2026-01-16*
*Module: `src/context/`*
*Dependencies: spaCy en_core_web_lg, scikit-learn*

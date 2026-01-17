# Topic D: Entity-Based SEO & Semantic Depth

## Technical Specification Document for SEO + AI Content Optimization Tool

**Version:** 1.0
**Date:** January 2026
**Author:** AI Engineering Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Entity Types for SEO](#2-entity-types-for-seo)
3. [NER Pipeline Specification](#3-ner-pipeline-specification)
4. [Entity Authority Modeling](#4-entity-authority-modeling)
5. [Knowledge Graph Integration](#5-knowledge-graph-integration)
6. [Entity Coverage Scoring](#6-entity-coverage-scoring)
7. [Entity Density & Topical Authority](#7-entity-density--topical-authority)
8. [Semantic Relationship Extraction](#8-semantic-relationship-extraction)
9. [Implementation Specifications](#9-implementation-specifications)
10. [Success Metrics](#10-success-metrics)

---

## 1. Executive Summary

Entity-based SEO represents a fundamental shift in how search engines understand and rank content. Google's transition from keyword-matching to semantic understanding, powered by the Knowledge Graph (now containing over 800 billion facts about 8 billion entities), means that modern content optimization must focus on entities—distinct concepts, people, places, products, and ideas—rather than simple keyword density. A 2023 Ahrefs study found that 78% of SEO professionals consider entity recognition crucial for effective SEO strategies.

This document specifies the technical architecture for implementing entity-based content optimization within an SEO tool. The system will leverage Named Entity Recognition (NER) to identify entities within content, link them to authoritative knowledge bases (Wikipedia, Wikidata), score entity coverage against competitors, and provide actionable recommendations for improving topical authority. Research from Surfer SEO indicates that sites with comprehensive entity coverage are significantly more likely to rank in top positions.

The implementation prioritizes production reliability through a tiered NER approach: spaCy's `en_core_web_lg` for high-throughput processing with transformer-based models (BERT-NER) for precision-critical tasks. Entity linking leverages the spacy-entity-linker library for Wikidata integration, while authority scoring combines co-occurrence analysis, schema markup detection, and external knowledge graph presence. The scoring formulas presented balance coverage breadth with entity authority weighting.

Key deliverables include: an NER pipeline achieving >90% F1-score on standard benchmarks, entity coverage gap analysis against top-10 competitor pages, authority-weighted entity scoring, and integration with schema.org markup recommendations. The system will support E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) signal detection, which Google has emphasized as critical for YMYL (Your Money or Your Life) content categories.

---

## 2. Entity Types for SEO

### 2.1 Core Entity Categories

#### 2.1.1 People (PER)
**Description:** Authors, experts, historical figures, influencers, executives, thought leaders.

**SEO Relevance:**
- Direct connection to E-E-A-T signals (Expertise, Experience)
- Author entities linked to credentials boost content authority
- Expert citations strengthen topical authority claims
- Google's entity resolution connects professional profiles across platforms

**Examples:**
- "Dr. Jane Smith, PhD in Machine Learning at Stanford"
- "Elon Musk, CEO of Tesla"
- "Marie Curie, Nobel Prize-winning physicist"

**Priority Weight:** HIGH (0.9) for YMYL content, MEDIUM (0.6) for general content

#### 2.1.2 Organizations (ORG)
**Description:** Companies, institutions, government bodies, NGOs, brands, research organizations.

**SEO Relevance:**
- Brand entity establishment in Knowledge Graph
- Institutional citations add credibility
- Company mentions enable rich snippets and knowledge panels
- B2B content heavily relies on organizational entity density

**Examples:**
- "Google LLC"
- "World Health Organization"
- "Massachusetts Institute of Technology"

**Priority Weight:** HIGH (0.85) for B2B content, MEDIUM (0.5) for consumer content

#### 2.1.3 Places (LOC/GPE)
**Description:** Geographic locations, cities, countries, landmarks, addresses.

**SEO Relevance:**
- Critical for local SEO optimization
- Geographic relevance signals for location-based queries
- Travel and real estate content optimization
- Regional authority establishment

**Examples:**
- "San Francisco, California"
- "The Eiffel Tower"
- "Silicon Valley"

**Priority Weight:** CRITICAL (1.0) for local content, LOW (0.3) for non-geographic content

#### 2.1.4 Products/Services (PRODUCT)
**Description:** Commercial products, software, services, tools, platforms.

**SEO Relevance:**
- E-commerce and product review optimization
- Comparison content entity coverage
- Feature and specification entity extraction
- Enables Product schema markup

**Examples:**
- "iPhone 15 Pro Max"
- "Adobe Photoshop"
- "Amazon Web Services"

**Priority Weight:** CRITICAL (1.0) for commercial content, LOW (0.2) for informational content

#### 2.1.5 Concepts/Topics (CONCEPT)
**Description:** Abstract ideas, methodologies, scientific concepts, industry terms, processes.

**SEO Relevance:**
- Topical authority establishment
- Semantic depth indicators
- Related concept coverage for comprehensive content
- Enables topic clustering and pillar page strategy

**Examples:**
- "Machine Learning"
- "Search Engine Optimization"
- "Cognitive Behavioral Therapy"

**Priority Weight:** HIGH (0.8) for educational content, MEDIUM (0.5) for commercial content

#### 2.1.6 Events (EVENT)
**Description:** Conferences, historical events, product launches, recurring events.

**SEO Relevance:**
- Temporal relevance signals
- News and trending topic optimization
- Industry event coverage for B2B authority
- Enables Event schema markup

**Examples:**
- "Google I/O 2025"
- "World War II"
- "Black Friday 2025"

**Priority Weight:** HIGH (0.7) for news/timely content, LOW (0.2) for evergreen content

### 2.2 Priority Weighting Matrix by Content Type

| Entity Type | YMYL/Health | E-commerce | B2B/SaaS | Local Business | Educational | News |
|-------------|-------------|------------|----------|----------------|-------------|------|
| People | 0.95 | 0.40 | 0.70 | 0.60 | 0.85 | 0.90 |
| Organizations | 0.85 | 0.50 | 0.90 | 0.70 | 0.75 | 0.85 |
| Places | 0.30 | 0.40 | 0.30 | 1.00 | 0.40 | 0.70 |
| Products | 0.40 | 1.00 | 0.85 | 0.50 | 0.30 | 0.50 |
| Concepts | 0.80 | 0.50 | 0.75 | 0.30 | 0.95 | 0.60 |
| Events | 0.30 | 0.60 | 0.65 | 0.50 | 0.40 | 0.95 |

### 2.3 Extended Entity Types

Beyond core NER categories, SEO-relevant entity extraction should include:

| Extended Type | Description | Detection Method |
|---------------|-------------|------------------|
| MONEY | Prices, financial values | Regex + NER |
| DATE/TIME | Temporal references | NER + Rule-based |
| QUANTITY | Measurements, statistics | Regex + NER |
| URL | Web references | Regex |
| EMAIL | Contact information | Regex |
| WORK_OF_ART | Books, movies, songs | NER |
| LAW | Legal references, regulations | Custom NER |
| LANGUAGE | Programming/spoken languages | Custom NER |

---

## 3. NER Pipeline Specification

### 3.1 spaCy Models Comparison

#### 3.1.1 Model Overview

| Model | Size | Speed (words/sec) | NER F1 (OntoNotes) | GPU Required | Use Case |
|-------|------|-------------------|---------------------|--------------|----------|
| en_core_web_sm | 12 MB | 10,000+ | 85.5% | No | Development/Testing |
| en_core_web_md | 40 MB | 8,000+ | 86.4% | No | Light Production |
| en_core_web_lg | 560 MB | 6,000+ | 87.0% | No | **Recommended Production** |
| en_core_web_trf | 438 MB | 200-500 | 89.8% | Recommended | High-Accuracy Tasks |

#### 3.1.2 Performance Benchmarks

Based on spaCy documentation and community benchmarks:

**Speed Comparison (text length 200 tokens):**
- `en_core_web_lg`: 0.12 seconds
- `en_core_web_trf`: 0.72 seconds (6x slower)

**Speed Comparison (text length 1200 tokens):**
- `en_core_web_lg`: 0.12 seconds
- `en_core_web_trf`: 5.86 seconds (47x slower)

**Accuracy by Entity Type (en_core_web_lg on OntoNotes):**

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| PERSON | 0.906 | 0.937 | 0.921 |
| ORG | 0.880 | 0.857 | 0.868 |
| GPE | 0.962 | 0.941 | 0.951 |
| DATE | 0.875 | 0.892 | 0.883 |
| MONEY | 0.912 | 0.895 | 0.903 |

### 3.2 Transformer-Based Alternatives

#### 3.2.1 BERT-NER Models

**dslim/bert-base-NER (Hugging Face):**
- Base: bert-base-cased
- Training: CoNLL-2003
- F1 Score: ~91.3%
- Entity Types: PER, ORG, LOC, MISC

**Fine-tuned Domain Models:**
- BioBERT: Biomedical NER (F1: 89.7% on JNLPBA)
- SciBERT: Scientific text (F1: 82.5% on materials NER)
- FinBERT: Financial domain

#### 3.2.2 RoBERTa-Based Models

spaCy's `en_core_web_trf` uses RoBERTa-base internally:
- Architecture: roberta-base with byte-bpe encoding
- Window size: 144 tokens with stride 104
- Higher accuracy but significantly slower inference

### 3.3 Domain-Specific Fine-Tuning Considerations

**When to Fine-Tune:**
1. Domain vocabulary significantly differs from general text
2. Custom entity types required (e.g., INGREDIENT, SYMPTOM)
3. Baseline model achieves <85% F1 on domain test set
4. Sufficient annotated data available (minimum 2,000 sentences)

**Fine-Tuning Data Requirements:**

| Domain | Minimum Sentences | Recommended Sentences | Expected F1 Improvement |
|--------|-------------------|----------------------|------------------------|
| General | 2,000 | 5,000+ | +2-5% |
| Technical | 3,000 | 8,000+ | +5-10% |
| Specialized | 5,000 | 15,000+ | +10-15% |

**Fine-Tuning Process:**
```python
# Pseudo-code for spaCy fine-tuning
import spacy
from spacy.training import Example

# Load base model
nlp = spacy.load("en_core_web_lg")
ner = nlp.get_pipe("ner")

# Add custom labels
ner.add_label("PRODUCT")
ner.add_label("CONCEPT")

# Training loop with domain data
for epoch in range(30):
    losses = {}
    for text, annotations in training_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], losses=losses)
```

### 3.4 Entity Linking vs. Pure NER

#### 3.4.1 Comparison

| Aspect | Pure NER | NER + Entity Linking |
|--------|----------|---------------------|
| Output | Entity spans + types | Spans + types + KB IDs |
| Disambiguation | None | Resolves ambiguity |
| Use Case | Entity detection only | Full knowledge graph integration |
| Speed | Fast | 2-5x slower |
| Dependencies | NER model only | NER + KB + linking model |

#### 3.4.2 Entity Linking Benefits for SEO

1. **Disambiguation:** "Apple" resolved to company (Q312) vs. fruit (Q89)
2. **Authority Signals:** Linked entities have verifiable Wikipedia/Wikidata presence
3. **Relationship Extraction:** KB provides entity relationships
4. **Competitor Analysis:** Consistent entity IDs enable comparison

### 3.5 Recommended Approach

**Production Architecture: Tiered NER System**

```
Tier 1 (High Volume): en_core_web_lg
├── Use for: Initial content scan, real-time analysis
├── Throughput: 6,000+ words/second
├── Accuracy: ~87% F1
└── Cost: Low (CPU only)

Tier 2 (Precision): en_core_web_trf OR BERT-NER
├── Use for: Final analysis, YMYL content, gap analysis
├── Throughput: 200-500 words/second
├── Accuracy: ~90% F1
└── Cost: Medium (GPU recommended)

Tier 3 (Entity Linking): spacy-entity-linker
├── Use for: Knowledge graph integration
├── Links to: Wikidata (~1.3GB KB)
├── Provides: QIDs, Wikipedia URLs, confidence scores
└── Cost: High (initial KB download + memory)
```

**Rationale:**
- 90%+ of content analysis uses Tier 1 for speed
- Tier 2 reserved for detailed gap analysis and YMYL content
- Tier 3 activated for competitor comparison and authority scoring
- Reduces infrastructure costs while maintaining accuracy where needed

---

## 4. Entity Authority Modeling

### 4.1 Co-occurrence Signals

#### 4.1.1 Entity Proximity Scoring

Entities appearing near each other signal semantic relationships. Proximity is measured in tokens:

**Proximity Score Formula:**
```
proximity_score(e1, e2) = 1 / (1 + log(token_distance + 1))
```

**Distance Categories:**
| Distance (tokens) | Proximity Score | Relationship Strength |
|-------------------|-----------------|----------------------|
| 0-5 | 0.85-1.00 | Very Strong |
| 6-15 | 0.65-0.84 | Strong |
| 16-30 | 0.50-0.64 | Moderate |
| 31-50 | 0.35-0.49 | Weak |
| 50+ | 0.00-0.34 | Minimal |

**Example Calculation:**
```
Text: "Google's CEO Sundar Pichai announced..."
Entities: Google (position 0), Sundar Pichai (position 3)
Distance: 3 tokens
Proximity Score: 1 / (1 + log(3 + 1)) = 1 / (1 + 1.39) = 0.42

Wait - recalculating with natural log:
Proximity Score: 1 / (1 + ln(4)) = 1 / (1 + 1.386) = 0.42
```

#### 4.1.2 Section-Level Co-occurrence

Entities within the same section/heading have implicit relationship:

**Section Co-occurrence Score:**
```
section_cooccurrence(e1, e2) = base_weight * section_type_multiplier

Where:
- base_weight = 0.5
- section_type_multiplier:
  - Same paragraph: 1.0
  - Same section (H2/H3): 0.7
  - Same major section (H1): 0.4
  - Different sections: 0.1
```

#### 4.1.3 Cross-Document Patterns

For corpus-level analysis, aggregate co-occurrence across documents:

**Pointwise Mutual Information (PMI):**
```
PMI(e1, e2) = log2(P(e1, e2) / (P(e1) * P(e2)))

Where:
- P(e1, e2) = documents containing both e1 and e2 / total documents
- P(e1) = documents containing e1 / total documents
- P(e2) = documents containing e2 / total documents
```

**Normalized PMI (NPMI):**
```
NPMI(e1, e2) = PMI(e1, e2) / (-log2(P(e1, e2)))

Range: [-1, 1] where 1 = perfect co-occurrence
```

### 4.2 External Authority Signals

#### 4.2.1 Wikipedia Page Existence

**Authority Tiers:**

| Wikipedia Status | Authority Score | Description |
|------------------|-----------------|-------------|
| Featured Article | 1.0 | Highest quality Wikipedia content |
| Good Article | 0.9 | Reviewed and meets quality standards |
| Standard Article | 0.7 | Normal Wikipedia page exists |
| Stub Article | 0.4 | Minimal Wikipedia presence |
| Redirect Only | 0.2 | Entity recognized but no dedicated page |
| No Page | 0.0 | No Wikipedia presence |

**Detection via Wikipedia API:**
```python
import wikipedia

def get_wikipedia_authority(entity_name):
    try:
        page = wikipedia.page(entity_name, auto_suggest=False)
        # Check for quality indicators
        if "featured article" in page.categories:
            return 1.0
        elif "good article" in page.categories:
            return 0.9
        elif len(page.content) > 10000:
            return 0.7
        else:
            return 0.4
    except wikipedia.DisambiguationError:
        return 0.3  # Entity exists but ambiguous
    except wikipedia.PageError:
        return 0.0  # No page found
```

#### 4.2.2 Knowledge Graph Presence

**Wikidata Authority Indicators:**

| Indicator | Weight | Description |
|-----------|--------|-------------|
| QID Exists | 0.3 | Base presence in Wikidata |
| >10 Properties | +0.2 | Well-described entity |
| >50 Properties | +0.3 | Comprehensive entity |
| External IDs (VIAF, ISNI) | +0.1 each | Cross-referenced identity |
| Sitelinks >5 | +0.1 | Multi-language Wikipedia presence |
| Sitelinks >20 | +0.2 | Global recognition |

**Wikidata Authority Score:**
```
wikidata_authority = base_score + property_bonus + id_bonus + sitelink_bonus

Max Score: 1.0
```

#### 4.2.3 Backlink Profile (When Accessible)

For organizations and products with web presence:

| Metric | Weight | Calculation |
|--------|--------|-------------|
| Domain Authority | 0.4 | DA/100 |
| Referring Domains | 0.3 | min(log10(RD)/4, 1.0) |
| Brand Mentions | 0.3 | Unlinked mention frequency |

### 4.3 Schema Markup Signals

#### 4.3.1 sameAs Properties

The `sameAs` property in schema.org explicitly links entities to authoritative sources:

**sameAs Authority Sources (Ranked):**

| Source | Authority Weight | Example |
|--------|-----------------|---------|
| Wikipedia | 1.0 | https://en.wikipedia.org/wiki/Entity |
| Wikidata | 0.95 | https://www.wikidata.org/wiki/Q12345 |
| LinkedIn (Person) | 0.8 | https://linkedin.com/in/username |
| Crunchbase (Org) | 0.8 | https://crunchbase.com/organization/x |
| Official Website | 0.7 | https://entity-official-site.com |
| Twitter/X | 0.5 | https://twitter.com/handle |
| Facebook | 0.4 | https://facebook.com/page |

**sameAs Authority Score:**
```
sameas_authority = sum(source_weight * presence_indicator) / max_possible_score
```

#### 4.3.2 @type Definitions

Rich schema typing indicates entity clarity:

| @type Specificity | Authority Bonus |
|-------------------|-----------------|
| Generic (Thing) | 0.0 |
| Category (Person, Organization) | 0.2 |
| Specific (Physician, Corporation) | 0.4 |
| Highly Specific (Oncologist, PublicCompany) | 0.6 |

### 4.4 Composite Authority Scoring Formula

**Entity Authority Score (EAS):**

```
EAS(entity) = (
    w1 * cooccurrence_score +
    w2 * wikipedia_authority +
    w3 * wikidata_authority +
    w4 * schema_authority +
    w5 * external_authority
) / sum(weights)

Default Weights:
- w1 (cooccurrence): 0.15
- w2 (wikipedia): 0.30
- w3 (wikidata): 0.25
- w4 (schema): 0.20
- w5 (external): 0.10
```

**Worked Example:**

Entity: "OpenAI"
- cooccurrence_score: 0.75 (frequently co-occurs with "GPT", "AI", "ChatGPT")
- wikipedia_authority: 0.7 (standard Wikipedia article)
- wikidata_authority: 0.8 (Q21085437, 40+ properties, multiple sitelinks)
- schema_authority: 0.9 (sameAs to Wikipedia, Wikidata, Crunchbase)
- external_authority: 0.85 (high DA, many referring domains)

```
EAS(OpenAI) = (0.15*0.75 + 0.30*0.7 + 0.25*0.8 + 0.20*0.9 + 0.10*0.85) / 1.0
           = (0.1125 + 0.21 + 0.20 + 0.18 + 0.085)
           = 0.7875 (High Authority)
```

**Authority Tiers:**
| EAS Range | Authority Level | Interpretation |
|-----------|-----------------|----------------|
| 0.8 - 1.0 | Very High | Well-known, authoritative entity |
| 0.6 - 0.79 | High | Established entity with good signals |
| 0.4 - 0.59 | Medium | Recognized entity, room for improvement |
| 0.2 - 0.39 | Low | Limited external validation |
| 0.0 - 0.19 | Very Low | Unknown or poorly validated entity |

---

## 5. Knowledge Graph Integration

### 5.1 Wikipedia API Integration

#### 5.1.1 Entity Disambiguation

Wikipedia's disambiguation handling is critical for accurate entity linking:

```python
import wikipedia

class WikipediaEntityResolver:
    def __init__(self):
        self.cache = {}

    def resolve_entity(self, entity_text, context_entities=None):
        """
        Resolve entity text to Wikipedia page with disambiguation handling.

        Args:
            entity_text: Raw entity text from NER
            context_entities: Other entities in document for disambiguation

        Returns:
            dict with page_title, url, summary, categories, or None
        """
        cache_key = entity_text.lower()
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Attempt direct page lookup
            page = wikipedia.page(entity_text, auto_suggest=True)
            result = {
                'title': page.title,
                'url': page.url,
                'summary': page.summary[:500],
                'categories': page.categories[:10],
                'links': page.links[:50],
                'page_id': page.pageid
            }
            self.cache[cache_key] = result
            return result

        except wikipedia.DisambiguationError as e:
            # Handle disambiguation using context
            options = e.options
            best_match = self._select_best_option(options, context_entities)
            if best_match:
                return self.resolve_entity(best_match, context_entities)
            return {'disambiguation_options': options}

        except wikipedia.PageError:
            # Try search as fallback
            search_results = wikipedia.search(entity_text, results=5)
            if search_results:
                return self.resolve_entity(search_results[0], context_entities)
            return None

    def _select_best_option(self, options, context_entities):
        """Select best disambiguation option based on context."""
        if not context_entities:
            return options[0] if options else None

        # Score options by overlap with context entities
        best_score = 0
        best_option = options[0]

        for option in options:
            try:
                page = wikipedia.page(option)
                overlap = len(set(page.links) & set(context_entities))
                if overlap > best_score:
                    best_score = overlap
                    best_option = option
            except:
                continue

        return best_option
```

#### 5.1.2 Related Entities Extraction

Extract related entities from Wikipedia page structure:

```python
def extract_related_entities(page):
    """
    Extract related entities from Wikipedia page.

    Returns entities from:
    - Page links (internal Wikipedia links)
    - Categories (hierarchical classification)
    - See Also section
    - Infobox data (if parseable)
    """
    related = {
        'direct_links': [],
        'categories': [],
        'see_also': [],
        'infobox_entities': []
    }

    # Top linked entities (by frequency in text)
    link_counts = {}
    for link in page.links:
        link_counts[link] = page.content.lower().count(link.lower())

    related['direct_links'] = sorted(
        link_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # Category-based entities
    related['categories'] = [
        cat for cat in page.categories
        if not cat.startswith('Articles')
        and not cat.startswith('Wikipedia')
    ][:15]

    return related
```

#### 5.1.3 Category/Topic Mapping

Map Wikipedia categories to SEO topics:

```python
CATEGORY_TO_TOPIC = {
    'Technology companies': 'Technology',
    'Software': 'Software & Apps',
    'Machine learning': 'Artificial Intelligence',
    'Health': 'Health & Wellness',
    'Finance': 'Finance & Business',
    'Education': 'Education',
    # ... extended mapping
}

def map_categories_to_topics(categories):
    """Map Wikipedia categories to SEO topic clusters."""
    topics = set()
    for category in categories:
        for pattern, topic in CATEGORY_TO_TOPIC.items():
            if pattern.lower() in category.lower():
                topics.add(topic)
    return list(topics)
```

### 5.2 Wikidata SPARQL Queries

#### 5.2.1 Property Extraction

```sparql
# Get comprehensive entity properties
SELECT ?property ?propertyLabel ?value ?valueLabel
WHERE {
  wd:Q312 ?prop ?value .  # Q312 = Apple Inc.
  ?property wikibase:directClaim ?prop .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100
```

**Python Implementation:**

```python
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataClient:
    ENDPOINT = "https://query.wikidata.org/sparql"

    def __init__(self):
        self.sparql = SPARQLWrapper(self.ENDPOINT)
        self.sparql.setReturnFormat(JSON)
        self.cache = {}

    def get_entity_properties(self, qid):
        """Get all properties for a Wikidata entity."""
        query = f"""
        SELECT ?property ?propertyLabel ?value ?valueLabel
        WHERE {{
          wd:{qid} ?prop ?value .
          ?property wikibase:directClaim ?prop .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 100
        """

        self.sparql.setQuery(query)
        results = self.sparql.query().convert()

        properties = {}
        for result in results["results"]["bindings"]:
            prop_label = result["propertyLabel"]["value"]
            value_label = result.get("valueLabel", {}).get("value", "")
            if prop_label not in properties:
                properties[prop_label] = []
            properties[prop_label].append(value_label)

        return properties
```

#### 5.2.2 Relationship Mapping

```sparql
# Get entity relationships (parent companies, subsidiaries, etc.)
SELECT ?related ?relatedLabel ?relationLabel
WHERE {
  {
    wd:Q312 ?relation ?related .
    ?related wdt:P31/wdt:P279* wd:Q43229 .  # Instance of organization
  } UNION {
    ?related ?relation wd:Q312 .
    ?related wdt:P31/wdt:P279* wd:Q43229 .
  }
  ?prop wikibase:directClaim ?relation .
  ?prop rdfs:label ?relationLabel .
  FILTER(LANG(?relationLabel) = "en")
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 50
```

#### 5.2.3 Instance-of Hierarchies

```sparql
# Get entity type hierarchy
SELECT ?type ?typeLabel ?supertype ?supertypeLabel
WHERE {
  wd:Q312 wdt:P31 ?type .
  OPTIONAL { ?type wdt:P279 ?supertype . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
```

**Hierarchy Extraction:**

```python
def get_entity_hierarchy(self, qid):
    """Get full type hierarchy for entity."""
    query = f"""
    SELECT ?type ?typeLabel ?depth
    WHERE {{
      wd:{qid} wdt:P31/wdt:P279* ?type .
      {{
        SELECT ?type (COUNT(?mid) as ?depth)
        WHERE {{
          wd:{qid} wdt:P31/wdt:P279* ?mid .
          ?mid wdt:P279* ?type .
        }}
        GROUP BY ?type
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY ?depth
    LIMIT 20
    """

    self.sparql.setQuery(query)
    results = self.sparql.query().convert()

    hierarchy = []
    for result in results["results"]["bindings"]:
        hierarchy.append({
            'qid': result["type"]["value"].split("/")[-1],
            'label': result["typeLabel"]["value"],
            'depth': int(result["depth"]["value"])
        })

    return hierarchy
```

### 5.3 Domain Ontologies

#### 5.3.1 Schema.org Vocabulary

Core schema.org types for SEO entities:

| Entity Type | Schema.org Type | Key Properties |
|-------------|-----------------|----------------|
| Person | schema:Person | name, jobTitle, affiliation, sameAs |
| Organization | schema:Organization | name, url, logo, sameAs, founder |
| Product | schema:Product | name, brand, offers, review, aggregateRating |
| Place | schema:Place | name, address, geo, containedIn |
| Event | schema:Event | name, startDate, location, organizer |
| Article | schema:Article | headline, author, datePublished, publisher |
| HowTo | schema:HowTo | step, tool, supply, totalTime |
| FAQ | schema:FAQPage | mainEntity (Question/Answer pairs) |

**Schema Mapping Function:**

```python
SCHEMA_TYPE_MAPPING = {
    'PERSON': 'schema:Person',
    'ORG': 'schema:Organization',
    'GPE': 'schema:Place',
    'LOC': 'schema:Place',
    'PRODUCT': 'schema:Product',
    'EVENT': 'schema:Event',
    'WORK_OF_ART': 'schema:CreativeWork',
    'LAW': 'schema:Legislation',
    'MONEY': 'schema:MonetaryAmount',
    'DATE': 'schema:Date',
}

def ner_to_schema_type(ner_label):
    """Map NER label to schema.org type."""
    return SCHEMA_TYPE_MAPPING.get(ner_label, 'schema:Thing')
```

#### 5.3.2 Industry-Specific Ontologies

| Domain | Ontology | Use Case |
|--------|----------|----------|
| Medical | SNOMED CT, MeSH | Health content entity validation |
| Legal | LKIF, Legal-RDF | Legal content entity types |
| Finance | FIBO | Financial entity relationships |
| E-commerce | GoodRelations | Product/service entities |
| Academic | VIVO | Researcher/publication entities |

### 5.4 Implementation Trade-offs

| Approach | Latency | Accuracy | Cost | Freshness |
|----------|---------|----------|------|-----------|
| Local KB (spacy-entity-linker) | Low (10ms) | Medium | Low | Stale (monthly updates) |
| Wikipedia API | Medium (100-500ms) | High | Free (rate limited) | Fresh |
| Wikidata SPARQL | Medium (200-800ms) | Very High | Free (rate limited) | Fresh |
| Commercial APIs (Google KG) | Low (50-100ms) | Very High | High ($$$) | Fresh |
| Hybrid (Local + API fallback) | Low-Medium | High | Medium | Mostly Fresh |

**Recommended Hybrid Approach:**

1. **Primary:** Local Wikidata KB (spacy-entity-linker) for fast lookups
2. **Fallback:** Wikipedia API for disambiguation and missing entities
3. **Enrichment:** Wikidata SPARQL for relationship extraction (cached)
4. **Cache:** Redis with 24-hour TTL for API responses

---

## 6. Entity Coverage Scoring

### 6.1 Topic Modeling for Expected Entities

#### 6.1.1 Expected Entity Generation

For a given topic, generate expected entities using:

1. **Seed Query Expansion:** Use the target keyword to query knowledge bases
2. **Competitor Analysis:** Extract entities from top-ranking pages
3. **Topic Modeling:** Use LDA/BERTopic to identify topic-related entities

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

class ExpectedEntityGenerator:
    def __init__(self, ner_pipeline):
        self.ner = ner_pipeline
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def generate_expected_entities(self, topic, competitor_texts, n_entities=50):
        """
        Generate expected entities for a topic based on competitor analysis.

        Args:
            topic: Target topic/keyword
            competitor_texts: List of text from top-ranking pages
            n_entities: Number of expected entities to return

        Returns:
            List of expected entities with frequency and authority scores
        """
        # Extract entities from all competitor texts
        all_entities = {}

        for text in competitor_texts:
            doc = self.ner(text)
            for ent in doc.ents:
                key = (ent.text.lower(), ent.label_)
                if key not in all_entities:
                    all_entities[key] = {
                        'text': ent.text,
                        'label': ent.label_,
                        'frequency': 0,
                        'document_count': 0,
                        'positions': []
                    }
                all_entities[key]['frequency'] += 1
                all_entities[key]['document_count'] += 1

        # Score entities by importance
        total_docs = len(competitor_texts)
        scored_entities = []

        for key, entity in all_entities.items():
            # TF-IDF inspired scoring
            tf = entity['frequency'] / sum(e['frequency'] for e in all_entities.values())
            idf = np.log(total_docs / entity['document_count'])

            # Penalize entities appearing in all docs (too generic)
            coverage_ratio = entity['document_count'] / total_docs
            specificity_bonus = 1.0 if coverage_ratio < 0.8 else 0.7

            score = tf * idf * specificity_bonus

            scored_entities.append({
                **entity,
                'importance_score': score,
                'coverage_ratio': coverage_ratio
            })

        # Sort by importance and return top N
        scored_entities.sort(key=lambda x: x['importance_score'], reverse=True)
        return scored_entities[:n_entities]
```

### 6.2 Competitor Content Analysis

#### 6.2.1 Entity Extraction from Top-Ranking Pages

```python
import requests
from bs4 import BeautifulSoup

class CompetitorEntityAnalyzer:
    def __init__(self, ner_pipeline, entity_linker):
        self.ner = ner_pipeline
        self.linker = entity_linker

    def analyze_serp_competitors(self, urls, target_content):
        """
        Analyze entities in competitor pages and compare to target.

        Returns:
            - competitor_entities: Aggregated competitor entity profile
            - target_entities: Entities in target content
            - coverage_gaps: Missing entities in target
            - coverage_score: Overall coverage percentage
        """
        competitor_entities = {}

        for url in urls:
            text = self._extract_text(url)
            doc = self.ner(text)

            for ent in doc.ents:
                key = self._normalize_entity(ent.text, ent.label_)
                if key not in competitor_entities:
                    competitor_entities[key] = {
                        'text': ent.text,
                        'label': ent.label_,
                        'urls': [],
                        'total_mentions': 0,
                        'linked_qid': None
                    }
                competitor_entities[key]['urls'].append(url)
                competitor_entities[key]['total_mentions'] += 1

        # Link entities to Wikidata
        for key, entity in competitor_entities.items():
            linked = self.linker.link(entity['text'])
            if linked:
                competitor_entities[key]['linked_qid'] = linked.qid

        # Analyze target content
        target_doc = self.ner(target_content)
        target_entities = set()
        for ent in target_doc.ents:
            target_entities.add(self._normalize_entity(ent.text, ent.label_))

        # Identify gaps
        competitor_set = set(competitor_entities.keys())
        coverage_gaps = competitor_set - target_entities

        # Calculate coverage score
        coverage_score = len(target_entities & competitor_set) / len(competitor_set) if competitor_set else 0

        return {
            'competitor_entities': competitor_entities,
            'target_entities': target_entities,
            'coverage_gaps': [competitor_entities[key] for key in coverage_gaps],
            'coverage_score': coverage_score
        }

    def _normalize_entity(self, text, label):
        """Normalize entity for comparison."""
        return (text.lower().strip(), label)

    def _extract_text(self, url):
        """Extract main content text from URL."""
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script/style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text from main content area
        main = soup.find('main') or soup.find('article') or soup.find('body')
        return main.get_text(separator=' ', strip=True)
```

#### 6.2.2 Coverage Gap Identification

```python
def identify_priority_gaps(coverage_gaps, entity_authority_scores):
    """
    Prioritize coverage gaps by potential SEO impact.

    Priority = Gap Frequency * Authority Score * Type Weight
    """
    prioritized_gaps = []

    for gap in coverage_gaps:
        authority = entity_authority_scores.get(gap['linked_qid'], 0.5)
        type_weight = ENTITY_TYPE_WEIGHTS.get(gap['label'], 0.5)

        # Frequency across competitors (higher = more important)
        frequency_score = len(gap['urls']) / 10  # Normalize to max 10 competitors

        priority_score = frequency_score * authority * type_weight

        prioritized_gaps.append({
            **gap,
            'priority_score': priority_score,
            'authority': authority,
            'recommendation': _generate_recommendation(gap, priority_score)
        })

    prioritized_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
    return prioritized_gaps

def _generate_recommendation(gap, priority_score):
    """Generate actionable recommendation for gap."""
    if priority_score > 0.8:
        return f"CRITICAL: Add '{gap['text']}' ({gap['label']}) - appears in {len(gap['urls'])} competitor pages"
    elif priority_score > 0.5:
        return f"IMPORTANT: Consider mentioning '{gap['text']}' for better topic coverage"
    else:
        return f"OPTIONAL: '{gap['text']}' may enhance content depth"
```

### 6.3 Scoring Formula

#### 6.3.1 Coverage Percentage

**Basic Coverage Score:**
```
coverage_percentage = (matched_entities / expected_entities) * 100

Where:
- matched_entities = count of expected entities found in content
- expected_entities = count of entities from competitor analysis
```

#### 6.3.2 Authority-Weighted Coverage

**Weighted Coverage Score:**
```
weighted_coverage = sum(matched_entity_authority) / sum(expected_entity_authority) * 100

Where:
- matched_entity_authority = EAS(entity) for each matched entity
- expected_entity_authority = EAS(entity) for each expected entity
```

**Implementation:**

```python
def calculate_weighted_coverage(target_entities, expected_entities, authority_scores):
    """
    Calculate authority-weighted entity coverage score.

    Returns score from 0-100 where:
    - 0-40: Poor coverage
    - 41-60: Moderate coverage
    - 61-80: Good coverage
    - 81-100: Excellent coverage
    """
    matched_authority = 0
    total_authority = 0

    expected_set = {e['normalized_key'] for e in expected_entities}

    for entity in expected_entities:
        authority = authority_scores.get(entity['linked_qid'], 0.3)
        total_authority += authority

        if entity['normalized_key'] in target_entities:
            matched_authority += authority

    if total_authority == 0:
        return 0

    return (matched_authority / total_authority) * 100
```

#### 6.3.3 Diversity/Specificity Balance

Optimal entity coverage balances:
- **Breadth:** Covering multiple entity types
- **Depth:** Including specific, niche entities
- **Relevance:** Matching topic expectations

**Diversity Score:**
```
diversity_score = (unique_entity_types / max_entity_types) * type_distribution_entropy

Where:
- unique_entity_types = count of distinct entity labels (PERSON, ORG, etc.)
- max_entity_types = 6 (core types)
- type_distribution_entropy = -sum(p * log(p)) for each type proportion
```

**Specificity Score:**
```
specificity_score = specific_entities / total_entities

Where:
- specific_entities = entities with Wikipedia pages AND <50% competitor coverage
- total_entities = all entities in content
```

**Combined Entity Coverage Score (ECS):**
```
ECS = (0.5 * weighted_coverage) + (0.25 * diversity_score * 100) + (0.25 * specificity_score * 100)
```

### 6.4 Worked Example with Real Content

**Scenario:** Analyzing content about "Machine Learning for SEO"

**Step 1: Competitor Entity Extraction**

Top 5 ranking pages yield these entities (simplified):

| Entity | Type | Frequency | Authority |
|--------|------|-----------|-----------|
| Google | ORG | 5/5 pages | 0.95 |
| Machine Learning | CONCEPT | 5/5 pages | 0.85 |
| Neural Network | CONCEPT | 4/5 pages | 0.80 |
| TensorFlow | PRODUCT | 3/5 pages | 0.75 |
| Natural Language Processing | CONCEPT | 4/5 pages | 0.82 |
| RankBrain | PRODUCT | 4/5 pages | 0.70 |
| BERT | PRODUCT | 3/5 pages | 0.78 |
| Sundar Pichai | PERSON | 2/5 pages | 0.65 |
| Stanford University | ORG | 2/5 pages | 0.72 |
| Python | CONCEPT | 3/5 pages | 0.68 |

**Step 2: Target Content Entity Extraction**

Target content contains:
- Google (ORG) - matched
- Machine Learning (CONCEPT) - matched
- SEO (CONCEPT) - matched
- Neural Network (CONCEPT) - matched
- Python (CONCEPT) - matched

**Step 3: Calculate Scores**

**Basic Coverage:**
```
matched = 5
expected = 10
coverage_percentage = 5/10 * 100 = 50%
```

**Authority-Weighted Coverage:**
```
matched_authority = 0.95 + 0.85 + 0.80 + 0.68 = 3.28
total_authority = 0.95 + 0.85 + 0.80 + 0.75 + 0.82 + 0.70 + 0.78 + 0.65 + 0.72 + 0.68 = 7.70
weighted_coverage = 3.28 / 7.70 * 100 = 42.6%
```

**Diversity Score:**
```
unique_types = 2 (ORG, CONCEPT)
max_types = 6
type_proportions = {ORG: 0.2, CONCEPT: 0.8}
entropy = -(0.2*log(0.2) + 0.8*log(0.8)) = 0.50
diversity_score = (2/6) * 0.50 = 0.167
```

**Specificity Score:**
```
specific_entities = 2 (Neural Network, Python - both have Wikipedia but <50% in competitors)
total_entities = 5
specificity_score = 2/5 = 0.40
```

**Combined ECS:**
```
ECS = (0.5 * 42.6) + (0.25 * 16.7) + (0.25 * 40.0)
    = 21.3 + 4.2 + 10.0
    = 35.5 (Poor Coverage)
```

**Gap Recommendations:**
1. CRITICAL: Add "TensorFlow" (PRODUCT) - 3/5 competitors mention it
2. CRITICAL: Add "RankBrain" (PRODUCT) - 4/5 competitors, highly relevant
3. IMPORTANT: Add "BERT" (PRODUCT) - growing relevance to SEO
4. IMPORTANT: Add "NLP/Natural Language Processing" (CONCEPT)
5. OPTIONAL: Add "Stanford University" (ORG) - authority signal
6. OPTIONAL: Add "Sundar Pichai" (PERSON) - E-E-A-T signal

---

## 7. Entity Density & Topical Authority

### 7.1 Optimal Entity Mention Frequency

#### 7.1.1 Entity Density Formula

```
entity_density = (total_entity_tokens / total_content_tokens) * 100

Where:
- total_entity_tokens = sum of tokens in all entity mentions
- total_content_tokens = total word count of content
```

**Optimal Ranges by Content Type:**

| Content Type | Optimal Density | Warning Threshold | Stuffing Threshold |
|--------------|-----------------|-------------------|-------------------|
| General Blog | 3-6% | >8% | >12% |
| Technical Guide | 5-10% | >12% | >18% |
| Product Page | 4-8% | >10% | >15% |
| News Article | 6-10% | >12% | >16% |
| Academic Content | 8-15% | >18% | >25% |

#### 7.1.2 Entity Frequency Guidelines

**Unique Entity Mention Ratio:**
```
unique_entity_ratio = unique_entities / total_entity_mentions

Optimal: 0.4 - 0.7
- <0.4: Too repetitive (same entities mentioned repeatedly)
- >0.7: Too diverse (not enough reinforcement)
```

**Primary Entity Prominence:**
```
primary_prominence = primary_entity_mentions / total_entity_mentions

Optimal: 0.15 - 0.30
- Primary entity should be mentioned 3-5x more than secondary entities
- But shouldn't dominate (>40%) to maintain natural flow
```

### 7.2 Diminishing Returns Analysis

**Entity Coverage vs. Ranking Correlation:**

Based on SEO research patterns:

| Coverage Level | Marginal Ranking Benefit | Notes |
|----------------|-------------------------|-------|
| 0-30% | HIGH (+15-20 positions) | Critical baseline coverage |
| 31-50% | MEDIUM (+8-12 positions) | Competitive differentiation |
| 51-70% | LOW (+3-5 positions) | Incremental improvement |
| 71-85% | MINIMAL (+1-2 positions) | Diminishing returns |
| 86-100% | NEAR ZERO | May indicate over-optimization |

**Optimal Target:** 65-75% entity coverage for competitive topics

**Mathematical Model:**
```
ranking_benefit(coverage) = base_benefit * (1 - e^(-decay_rate * coverage))

Where:
- base_benefit = maximum possible ranking improvement
- decay_rate = 3.5 (empirically derived)
- coverage = entity coverage percentage (0-1)
```

### 7.3 Entity Stuffing Detection

#### 7.3.1 Detection Signals

| Signal | Threshold | Detection Method |
|--------|-----------|------------------|
| Unnatural density | >2x optimal | Compare to content type baseline |
| Repetition ratio | >5 mentions/entity | Count per unique entity |
| Proximity clustering | >3 entities/sentence avg | Sentence-level analysis |
| Context mismatch | Low coherence score | Semantic similarity check |
| Forced mentions | Entity without predicates | Dependency parsing |

#### 7.3.2 Stuffing Score Algorithm

```python
def calculate_stuffing_score(content, entities, content_type='general'):
    """
    Detect entity stuffing. Returns 0-100 where:
    - 0-20: Natural entity usage
    - 21-50: Slightly dense but acceptable
    - 51-75: Warning - may appear unnatural
    - 76-100: Likely stuffing - recommend reduction
    """

    # Factor 1: Density vs optimal
    density = calculate_entity_density(content, entities)
    optimal = OPTIMAL_DENSITY[content_type]
    density_score = max(0, (density - optimal) / optimal * 50)

    # Factor 2: Repetition
    entity_counts = count_entity_mentions(entities)
    avg_mentions = sum(entity_counts.values()) / len(entity_counts)
    repetition_score = max(0, (avg_mentions - 3) * 10)

    # Factor 3: Unnatural clustering
    sentences = split_sentences(content)
    entities_per_sentence = [count_entities_in_text(s, entities) for s in sentences]
    clustering_score = max(0, (np.std(entities_per_sentence) - 1.5) * 20)

    # Factor 4: Context coherence (using embeddings)
    coherence_score = measure_entity_context_coherence(content, entities)
    context_penalty = max(0, (0.7 - coherence_score) * 50)

    # Combined stuffing score
    stuffing_score = min(100, density_score + repetition_score + clustering_score + context_penalty)

    return {
        'score': stuffing_score,
        'density_factor': density_score,
        'repetition_factor': repetition_score,
        'clustering_factor': clustering_score,
        'context_factor': context_penalty,
        'recommendation': get_stuffing_recommendation(stuffing_score)
    }
```

### 7.4 Relationship to E-E-A-T Signals

#### 7.4.1 Entity Types and E-E-A-T Components

| E-E-A-T Component | Relevant Entity Types | Signal Examples |
|-------------------|----------------------|-----------------|
| Experience | PERSON, EVENT, DATE | Author mentions, first-hand accounts, dated experiences |
| Expertise | PERSON, ORG, CONCEPT | Credentials, certifications, technical concepts |
| Authoritativeness | ORG, PERSON, WORK_OF_ART | Citations, institutional affiliations, publications |
| Trustworthiness | ORG, PERSON, LOC | Verified organizations, real locations, identified authors |

#### 7.4.2 E-E-A-T Entity Score

```python
def calculate_eeat_entity_score(entities, schema_data=None):
    """
    Calculate E-E-A-T signals from entity profile.

    Returns scores for each component (0-1) and composite score.
    """
    scores = {
        'experience': 0,
        'expertise': 0,
        'authoritativeness': 0,
        'trustworthiness': 0
    }

    # Experience: Personal/temporal entities
    experience_entities = [e for e in entities if e['label'] in ['PERSON', 'DATE', 'EVENT']]
    has_first_person = detect_first_person_experience(content)
    scores['experience'] = min(1.0, len(experience_entities) * 0.1 + (0.3 if has_first_person else 0))

    # Expertise: Technical concepts and credentials
    concept_entities = [e for e in entities if e['label'] == 'CONCEPT']
    credential_mentions = detect_credentials(content)
    scores['expertise'] = min(1.0, len(concept_entities) * 0.05 + credential_mentions * 0.2)

    # Authoritativeness: Citations and recognized entities
    high_authority_entities = [e for e in entities if e.get('authority_score', 0) > 0.7]
    citation_count = count_citations(content)
    scores['authoritativeness'] = min(1.0, len(high_authority_entities) * 0.1 + citation_count * 0.1)

    # Trustworthiness: Schema markup, verified entities
    schema_bonus = 0.3 if schema_data and 'author' in schema_data else 0
    verified_entities = [e for e in entities if e.get('linked_qid')]
    scores['trustworthiness'] = min(1.0, len(verified_entities) * 0.05 + schema_bonus)

    # Composite score (Trust weighted highest per Google guidelines)
    composite = (
        scores['experience'] * 0.2 +
        scores['expertise'] * 0.25 +
        scores['authoritativeness'] * 0.25 +
        scores['trustworthiness'] * 0.3
    )

    return {**scores, 'composite': composite}
```

### 7.5 Content Depth Indicators

Entity-based depth indicators beyond simple coverage:

| Indicator | Measurement | Good Score |
|-----------|-------------|------------|
| Entity Hierarchy Depth | Levels of specificity (AI > ML > Deep Learning > CNN) | 3+ levels |
| Relationship Richness | Unique entity-entity relationships | >10 relationships |
| Cross-Type Coverage | Entity types represented | 4+ types |
| Authority Distribution | Mix of high/medium/low authority entities | 60/30/10 ratio |
| Temporal Span | Date range of mentioned entities/events | Appropriate to topic |

---

## 8. Semantic Relationship Extraction

### 8.1 Subject-Predicate-Object Triples

#### 8.1.1 Triple Extraction Architecture

```
Input Text
    │
    ▼
┌─────────────────────┐
│   NER Extraction    │ → Identifies entities (subjects/objects)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Dependency Parsing  │ → Identifies grammatical relationships
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Relation Extraction │ → Maps dependencies to predicates
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Triple Validation   │ → Filters invalid/incomplete triples
└─────────────────────┘
    │
    ▼
Output: [(subject, predicate, object), ...]
```

#### 8.1.2 spaCy-Based Triple Extraction

```python
import spacy

class TripleExtractor:
    def __init__(self, model="en_core_web_lg"):
        self.nlp = spacy.load(model)

        # Predicate patterns (dependency relations that indicate relationships)
        self.predicate_deps = {'nsubj', 'dobj', 'pobj', 'attr', 'prep'}

    def extract_triples(self, text):
        """
        Extract subject-predicate-object triples from text.

        Returns:
            List of (subject, predicate, object, confidence) tuples
        """
        doc = self.nlp(text)
        triples = []

        for sent in doc.sents:
            sent_triples = self._extract_from_sentence(sent)
            triples.extend(sent_triples)

        return self._deduplicate_triples(triples)

    def _extract_from_sentence(self, sent):
        """Extract triples from a single sentence."""
        triples = []

        # Find main verb (predicate)
        root = [token for token in sent if token.dep_ == 'ROOT']
        if not root:
            return triples
        root = root[0]

        # Find subject
        subjects = [child for child in root.children if child.dep_ in ('nsubj', 'nsubjpass')]

        # Find objects
        objects = [child for child in root.children if child.dep_ in ('dobj', 'pobj', 'attr')]

        # Also check prepositional phrases
        for child in root.children:
            if child.dep_ == 'prep':
                for grandchild in child.children:
                    if grandchild.dep_ == 'pobj':
                        objects.append(grandchild)

        # Generate triples
        for subj in subjects:
            subj_text = self._get_entity_span(subj)
            for obj in objects:
                obj_text = self._get_entity_span(obj)
                predicate = root.lemma_

                # Calculate confidence based on entity recognition
                confidence = self._calculate_confidence(subj, obj, root)

                if subj_text and obj_text:
                    triples.append((subj_text, predicate, obj_text, confidence))

        return triples

    def _get_entity_span(self, token):
        """Get full entity span for a token."""
        # Check if token is part of named entity
        if token.ent_type_:
            # Find full entity span
            for ent in token.doc.ents:
                if token in ent:
                    return ent.text

        # Otherwise, get noun chunk
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text

        return token.text if token.pos_ in ('NOUN', 'PROPN') else None

    def _calculate_confidence(self, subj, obj, predicate):
        """Calculate confidence score for triple."""
        confidence = 0.5  # Base confidence

        # Boost for named entities
        if subj.ent_type_:
            confidence += 0.2
        if obj.ent_type_:
            confidence += 0.2

        # Boost for clear predicate
        if predicate.pos_ == 'VERB':
            confidence += 0.1

        return min(1.0, confidence)

    def _deduplicate_triples(self, triples):
        """Remove duplicate triples, keeping highest confidence."""
        seen = {}
        for triple in triples:
            key = (triple[0].lower(), triple[1].lower(), triple[2].lower())
            if key not in seen or triple[3] > seen[key][3]:
                seen[key] = triple
        return list(seen.values())
```

### 8.2 Relationship Types

#### 8.2.1 Core Relationship Categories

| Relationship Type | Examples | Detection Pattern |
|-------------------|----------|-------------------|
| **is-a** (Instance) | "Python is a programming language" | subj + copula + obj(category) |
| **part-of** (Meronymy) | "GPU is part of a computer" | "part of", "component of", "belongs to" |
| **has-a** (Composition) | "Computer has a CPU" | "has", "contains", "includes" |
| **related-to** (Association) | "SEO relates to marketing" | "relates to", "associated with", co-occurrence |
| **created-by** (Authorship) | "TensorFlow was created by Google" | "created by", "developed by", "authored by" |
| **located-in** (Spatial) | "Google is based in California" | "located in", "based in", "headquarters in" |
| **works-for** (Affiliation) | "Sundar Pichai works for Google" | "works for", "CEO of", "employed by" |
| **causes** (Causation) | "Backlinks improve rankings" | "causes", "leads to", "results in" |

#### 8.2.2 Relationship Pattern Matching

```python
RELATIONSHIP_PATTERNS = {
    'is_a': [
        {'POS': 'NOUN', 'DEP': 'nsubj'},
        {'LEMMA': 'be'},
        {'POS': 'DET', 'OP': '?'},
        {'POS': 'NOUN', 'DEP': 'attr'}
    ],
    'part_of': [
        {'POS': 'NOUN'},
        {'LEMMA': {'IN': ['part', 'component', 'member']}},
        {'LOWER': 'of'},
        {'POS': 'NOUN'}
    ],
    'created_by': [
        {'POS': 'NOUN', 'DEP': 'nsubjpass'},
        {'LEMMA': {'IN': ['create', 'develop', 'build', 'found']}},
        {'LOWER': 'by'},
        {'POS': 'PROPN'}
    ]
}

class RelationshipClassifier:
    def __init__(self, nlp):
        self.nlp = nlp
        self.matcher = spacy.matcher.Matcher(nlp.vocab)

        for rel_type, pattern in RELATIONSHIP_PATTERNS.items():
            self.matcher.add(rel_type, [pattern])

    def classify_relationship(self, subject, predicate, obj, context):
        """Classify relationship type for a triple."""
        # Check predicate-based classification
        predicate_lower = predicate.lower()

        if predicate_lower in ('be', 'is', 'are', 'was', 'were'):
            return 'is_a'
        elif predicate_lower in ('have', 'has', 'contain', 'include'):
            return 'has_a'
        elif predicate_lower in ('create', 'develop', 'build', 'make'):
            return 'created_by'
        elif predicate_lower in ('work', 'employ'):
            return 'works_for'
        elif predicate_lower in ('locate', 'base', 'situate'):
            return 'located_in'
        elif predicate_lower in ('cause', 'lead', 'result'):
            return 'causes'
        else:
            return 'related_to'
```

### 8.3 Graph Construction from Content

```python
import networkx as nx

class ContentKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_nodes = {}
        self.relationship_edges = []

    def build_from_triples(self, triples, entity_metadata=None):
        """
        Build knowledge graph from extracted triples.

        Args:
            triples: List of (subject, predicate, object, confidence) tuples
            entity_metadata: Optional dict of entity -> {type, qid, authority}
        """
        for subj, pred, obj, conf in triples:
            # Add nodes
            self._add_entity_node(subj, entity_metadata)
            self._add_entity_node(obj, entity_metadata)

            # Add edge
            self.graph.add_edge(
                subj.lower(),
                obj.lower(),
                predicate=pred,
                confidence=conf,
                relationship_type=self._classify_relationship(pred)
            )

    def _add_entity_node(self, entity, metadata=None):
        """Add entity as node with metadata."""
        key = entity.lower()
        if key not in self.entity_nodes:
            node_data = {'label': entity, 'mentions': 1}
            if metadata and entity in metadata:
                node_data.update(metadata[entity])
            self.graph.add_node(key, **node_data)
            self.entity_nodes[key] = node_data
        else:
            self.graph.nodes[key]['mentions'] += 1

    def get_entity_centrality(self):
        """Calculate centrality metrics for entities."""
        return {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph)
        }

    def find_missing_relationships(self, reference_graph):
        """
        Compare content graph to reference (Wikidata) graph.
        Identify missing relationships.
        """
        missing = []

        for u, v, data in reference_graph.edges(data=True):
            if self.graph.has_node(u) and self.graph.has_node(v):
                if not self.graph.has_edge(u, v):
                    missing.append({
                        'subject': u,
                        'object': v,
                        'relationship': data.get('predicate', 'unknown'),
                        'importance': data.get('weight', 0.5)
                    })

        return sorted(missing, key=lambda x: x['importance'], reverse=True)

    def export_for_visualization(self):
        """Export graph for visualization (D3.js compatible)."""
        return {
            'nodes': [
                {'id': n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            'links': [
                {
                    'source': u,
                    'target': v,
                    'predicate': d['predicate'],
                    'confidence': d['confidence']
                }
                for u, v, d in self.graph.edges(data=True)
            ]
        }
```

### 8.4 Gap Identification vs. Knowledge Graphs

```python
class KnowledgeGapAnalyzer:
    def __init__(self, wikidata_client, triple_extractor):
        self.wikidata = wikidata_client
        self.extractor = triple_extractor

    def analyze_semantic_gaps(self, content, primary_entities):
        """
        Identify missing semantic relationships compared to Wikidata.

        Args:
            content: Text content to analyze
            primary_entities: List of main entities with QIDs

        Returns:
            Gaps with recommendations for content enhancement
        """
        # Extract triples from content
        content_triples = self.extractor.extract_triples(content)
        content_graph = self._build_graph(content_triples)

        # Get reference relationships from Wikidata
        reference_relationships = []
        for entity in primary_entities:
            if entity.get('qid'):
                wd_rels = self.wikidata.get_entity_relationships(entity['qid'])
                reference_relationships.extend(wd_rels)

        # Identify gaps
        gaps = []
        for ref_rel in reference_relationships:
            # Check if relationship exists in content
            if not self._relationship_exists(content_graph, ref_rel):
                gap = {
                    'subject': ref_rel['subject'],
                    'predicate': ref_rel['predicate'],
                    'object': ref_rel['object'],
                    'importance': self._calculate_importance(ref_rel),
                    'recommendation': self._generate_recommendation(ref_rel)
                }
                gaps.append(gap)

        return sorted(gaps, key=lambda x: x['importance'], reverse=True)

    def _calculate_importance(self, relationship):
        """Score relationship importance for SEO."""
        # High importance relationships
        high_importance = ['instance of', 'part of', 'creator', 'author', 'founder']
        medium_importance = ['located in', 'headquarters', 'industry', 'field of work']

        pred = relationship['predicate'].lower()

        if any(h in pred for h in high_importance):
            return 0.9
        elif any(m in pred for m in medium_importance):
            return 0.6
        else:
            return 0.3

    def _generate_recommendation(self, relationship):
        """Generate actionable recommendation for gap."""
        return (
            f"Consider mentioning that {relationship['subject']} "
            f"{relationship['predicate']} {relationship['object']}. "
            f"This relationship is documented in Wikidata and adds semantic depth."
        )
```

---

## 9. Implementation Specifications

### 9.1 NER Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Content Input                                │
│                    (HTML/Text/URL)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Preprocessing Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ HTML Strip  │→ │ Text Clean  │→ │ Sentence    │              │
│  │             │  │             │  │ Tokenize    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NER Extraction Layer                          │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Tier Selection Router                    │       │
│  │   • Content length < 1000 → Tier 1 (en_core_web_lg)  │       │
│  │   • YMYL content → Tier 2 (en_core_web_trf)          │       │
│  │   • Deep analysis → Tier 3 (+ entity linking)        │       │
│  └──────────────────────────────────────────────────────┘       │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │   Tier 1    │      │   Tier 2    │      │   Tier 3    │     │
│  │ spaCy lg    │      │ spaCy trf   │      │ + Wikidata  │     │
│  │ (CPU)       │      │ (GPU opt)   │      │ Linking     │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Entity Post-Processing                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Dedup &     │→ │ Merge       │→ │ Normalize   │              │
│  │ Coref       │  │ Overlapping │  │ Text        │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Enrichment                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Authority   │  │ Wikipedia   │  │ Wikidata    │              │
│  │ Scoring     │  │ Lookup      │  │ Properties  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Output Format                                │
│  {                                                               │
│    "entities": [...],                                           │
│    "triples": [...],                                            │
│    "coverage_score": 0.72,                                      │
│    "authority_scores": {...},                                   │
│    "gaps": [...],                                               │
│    "recommendations": [...]                                     │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Entity Resolution/Disambiguation Logic

```python
class EntityResolver:
    """
    Multi-stage entity resolution with caching and fallbacks.
    """

    def __init__(self, config):
        self.cache = RedisCache(config.redis_url, ttl=86400)
        self.local_kb = SpacyEntityLinker()
        self.wikipedia = WikipediaClient()
        self.wikidata = WikidataClient()

    def resolve(self, entity_text, entity_type, context=None):
        """
        Resolve entity to canonical form with KB linkage.

        Resolution Pipeline:
        1. Cache lookup
        2. Local KB (spacy-entity-linker)
        3. Wikipedia API
        4. Wikidata SPARQL
        5. Fuzzy matching fallback
        """
        cache_key = f"entity:{entity_text.lower()}:{entity_type}"

        # Stage 1: Cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = None

        # Stage 2: Local KB
        local_result = self.local_kb.link(entity_text)
        if local_result and local_result.confidence > 0.8:
            result = self._format_result(local_result)

        # Stage 3: Wikipedia (if local failed or low confidence)
        if not result or result['confidence'] < 0.8:
            wiki_result = self.wikipedia.resolve_entity(entity_text, context)
            if wiki_result and not wiki_result.get('disambiguation_options'):
                result = self._merge_results(result, wiki_result)

        # Stage 4: Wikidata enrichment
        if result and result.get('qid'):
            wd_props = self.wikidata.get_entity_properties(result['qid'])
            result['properties'] = wd_props
            result['authority'] = self._calculate_authority(result, wd_props)

        # Stage 5: Fuzzy fallback
        if not result:
            result = self._fuzzy_resolve(entity_text, entity_type)

        # Cache and return
        if result:
            self.cache.set(cache_key, result)

        return result

    def _format_result(self, local_result):
        return {
            'canonical_name': local_result.name,
            'qid': local_result.qid,
            'url': local_result.url,
            'confidence': local_result.confidence,
            'source': 'local_kb'
        }

    def _calculate_authority(self, result, properties):
        """Calculate entity authority score."""
        score = 0.3  # Base for having QID

        # Property count bonus
        prop_count = len(properties)
        if prop_count > 50:
            score += 0.3
        elif prop_count > 20:
            score += 0.2
        elif prop_count > 10:
            score += 0.1

        # External ID bonus
        external_ids = ['VIAF ID', 'ISNI', 'Library of Congress authority ID']
        for ext_id in external_ids:
            if ext_id in properties:
                score += 0.05

        # Wikipedia bonus
        if result.get('wikipedia_url'):
            score += 0.2

        return min(1.0, score)
```

### 9.3 Knowledge Base Query Caching

```python
import redis
import hashlib
import json
from functools import wraps

class KnowledgeBaseCache:
    """
    Multi-tier caching for knowledge base queries.

    Tier 1: In-memory LRU (hot data, <1ms)
    Tier 2: Redis (warm data, <10ms)
    Tier 3: SQLite (cold data, <50ms)
    """

    def __init__(self, redis_url, sqlite_path, memory_size=10000):
        self.memory_cache = LRUCache(maxsize=memory_size)
        self.redis = redis.from_url(redis_url)
        self.sqlite = sqlite3.connect(sqlite_path, check_same_thread=False)
        self._init_sqlite()

    def _init_sqlite(self):
        self.sqlite.execute("""
            CREATE TABLE IF NOT EXISTS kb_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 1
            )
        """)

    def get(self, key):
        """Get from cache with tier fallback."""
        # Tier 1: Memory
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Tier 2: Redis
        redis_value = self.redis.get(key)
        if redis_value:
            value = json.loads(redis_value)
            self.memory_cache[key] = value
            return value

        # Tier 3: SQLite
        cursor = self.sqlite.execute(
            "SELECT value FROM kb_cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row:
            value = json.loads(row[0])
            # Promote to higher tiers
            self.redis.setex(key, 3600, row[0])
            self.memory_cache[key] = value
            # Update access count
            self.sqlite.execute(
                "UPDATE kb_cache SET access_count = access_count + 1 WHERE key = ?",
                (key,)
            )
            return value

        return None

    def set(self, key, value, ttl=86400):
        """Set in all cache tiers."""
        json_value = json.dumps(value)

        # Tier 1: Memory (always)
        self.memory_cache[key] = value

        # Tier 2: Redis (with TTL)
        self.redis.setex(key, ttl, json_value)

        # Tier 3: SQLite (persistent)
        self.sqlite.execute(
            "INSERT OR REPLACE INTO kb_cache (key, value) VALUES (?, ?)",
            (key, json_value)
        )
        self.sqlite.commit()


def cached_kb_query(cache, ttl=86400):
    """Decorator for caching KB queries."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()

            # Try cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

### 9.4 Scoring Calculation Pseudocode

```python
class EntitySEOScorer:
    """
    Complete entity-based SEO scoring implementation.
    """

    def __init__(self, ner_pipeline, entity_resolver, competitor_analyzer):
        self.ner = ner_pipeline
        self.resolver = entity_resolver
        self.competitor = competitor_analyzer

        # Scoring weights (configurable)
        self.weights = {
            'coverage': 0.35,
            'authority': 0.25,
            'density': 0.15,
            'diversity': 0.10,
            'relationships': 0.15
        }

    def score_content(self, content, target_keyword, competitor_urls=None):
        """
        Calculate comprehensive entity SEO score.

        Returns:
            EntityScore object with component scores and recommendations
        """
        # Step 1: Extract entities from content
        doc = self.ner(content)
        entities = self._process_entities(doc)

        # Step 2: Resolve and enrich entities
        for entity in entities:
            resolution = self.resolver.resolve(
                entity['text'],
                entity['label'],
                context=[e['text'] for e in entities]
            )
            entity.update(resolution or {})

        # Step 3: Get competitor baseline (if URLs provided)
        if competitor_urls:
            competitor_profile = self.competitor.analyze_serp_competitors(
                competitor_urls, content
            )
            expected_entities = competitor_profile['competitor_entities']
        else:
            expected_entities = self._generate_expected_entities(target_keyword)

        # Step 4: Calculate component scores
        scores = {
            'coverage': self._calculate_coverage_score(entities, expected_entities),
            'authority': self._calculate_authority_score(entities),
            'density': self._calculate_density_score(content, entities),
            'diversity': self._calculate_diversity_score(entities),
            'relationships': self._calculate_relationship_score(content, entities)
        }

        # Step 5: Calculate weighted composite score
        composite = sum(
            scores[k] * self.weights[k]
            for k in scores
        )

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            entities, expected_entities, scores, competitor_profile
        )

        # Step 7: Identify coverage gaps
        gaps = self._identify_gaps(entities, expected_entities)

        return EntityScore(
            composite_score=composite,
            component_scores=scores,
            entities=entities,
            gaps=gaps,
            recommendations=recommendations,
            metadata={
                'entity_count': len(entities),
                'unique_types': len(set(e['label'] for e in entities)),
                'linked_entities': len([e for e in entities if e.get('qid')])
            }
        )

    def _calculate_coverage_score(self, entities, expected):
        """
        Calculate entity coverage against expected entities.

        Formula:
        coverage = sum(matched * authority) / sum(expected * authority)
        """
        if not expected:
            return 0.5  # No baseline, assume moderate coverage

        entity_keys = {(e['text'].lower(), e['label']) for e in entities}
        expected_keys = {(e['text'].lower(), e['label']) for e in expected}

        matched_authority = sum(
            expected[i].get('authority', 0.5)
            for i, e in enumerate(expected)
            if (e['text'].lower(), e['label']) in entity_keys
        )

        total_authority = sum(e.get('authority', 0.5) for e in expected)

        return matched_authority / total_authority if total_authority > 0 else 0

    def _calculate_authority_score(self, entities):
        """
        Calculate average authority of content entities.

        Score based on:
        - Wikipedia/Wikidata presence
        - Entity properties richness
        - External ID presence
        """
        if not entities:
            return 0

        authority_sum = sum(e.get('authority', 0.3) for e in entities)
        return authority_sum / len(entities)

    def _calculate_density_score(self, content, entities):
        """
        Score entity density (penalize over/under-optimization).

        Optimal density: 4-8%
        """
        word_count = len(content.split())
        entity_word_count = sum(len(e['text'].split()) for e in entities)

        density = entity_word_count / word_count if word_count > 0 else 0

        # Optimal range scoring
        if 0.04 <= density <= 0.08:
            return 1.0
        elif 0.02 <= density < 0.04 or 0.08 < density <= 0.12:
            return 0.7
        elif density < 0.02:
            return 0.4
        else:  # density > 0.12 (stuffing)
            return max(0.1, 1.0 - (density - 0.12) * 5)

    def _calculate_diversity_score(self, entities):
        """
        Score entity type diversity.

        Optimal: 4+ entity types represented
        """
        if not entities:
            return 0

        unique_types = set(e['label'] for e in entities)
        type_count = len(unique_types)

        if type_count >= 5:
            return 1.0
        elif type_count >= 4:
            return 0.85
        elif type_count >= 3:
            return 0.7
        elif type_count >= 2:
            return 0.5
        else:
            return 0.3

    def _calculate_relationship_score(self, content, entities):
        """
        Score semantic relationship richness.
        """
        triples = self.extractor.extract_triples(content)

        if not triples:
            return 0.3

        # Score based on triple count and quality
        triple_count = len(triples)
        avg_confidence = sum(t[3] for t in triples) / len(triples)

        # Relationship type diversity
        relationship_types = set(self._classify_relationship(t[1]) for t in triples)
        type_diversity = len(relationship_types) / 8  # 8 main relationship types

        return min(1.0, (triple_count / 20) * 0.4 + avg_confidence * 0.3 + type_diversity * 0.3)

    def _generate_recommendations(self, entities, expected, scores, competitor_profile):
        """Generate actionable recommendations based on scores."""
        recommendations = []

        # Coverage recommendations
        if scores['coverage'] < 0.6:
            gaps = self._identify_gaps(entities, expected)
            top_gaps = sorted(gaps, key=lambda x: x.get('importance', 0), reverse=True)[:5]
            recommendations.append({
                'priority': 'HIGH',
                'category': 'coverage',
                'message': f"Add missing key entities: {', '.join(g['text'] for g in top_gaps)}",
                'impact': 'Improve topical coverage by {:.0f}%'.format((0.6 - scores['coverage']) * 100)
            })

        # Authority recommendations
        if scores['authority'] < 0.5:
            low_authority = [e for e in entities if e.get('authority', 0) < 0.4]
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'authority',
                'message': "Strengthen entity authority by adding citations and linking to authoritative sources",
                'impact': 'Improve E-E-A-T signals'
            })

        # Density recommendations
        if scores['density'] < 0.7:
            if self._get_density(entities) < 0.04:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'density',
                    'message': "Increase entity mentions - content appears thin on specific entities",
                    'impact': 'Add 10-15 more entity mentions'
                })
            else:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'density',
                    'message': "Reduce entity stuffing - mentions appear unnatural",
                    'impact': 'Remove repetitive entity mentions'
                })

        # Diversity recommendations
        if scores['diversity'] < 0.7:
            missing_types = self._get_missing_entity_types(entities)
            recommendations.append({
                'priority': 'LOW',
                'category': 'diversity',
                'message': f"Add entities of types: {', '.join(missing_types)}",
                'impact': 'Improve semantic richness'
            })

        return recommendations


@dataclass
class EntityScore:
    """Entity SEO score result."""
    composite_score: float
    component_scores: Dict[str, float]
    entities: List[Dict]
    gaps: List[Dict]
    recommendations: List[Dict]
    metadata: Dict

    def to_dict(self):
        return asdict(self)

    def get_grade(self):
        """Convert score to letter grade."""
        if self.composite_score >= 0.9:
            return 'A+'
        elif self.composite_score >= 0.8:
            return 'A'
        elif self.composite_score >= 0.7:
            return 'B'
        elif self.composite_score >= 0.6:
            return 'C'
        elif self.composite_score >= 0.5:
            return 'D'
        else:
            return 'F'
```

---

## 10. Success Metrics

### 10.1 NER Precision/Recall Targets

#### 10.1.1 Baseline Performance Requirements

| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Precision | 85% | 90% | 95% |
| Recall | 80% | 88% | 92% |
| F1 Score | 82% | 89% | 93% |

#### 10.1.2 Per-Entity Type Targets

| Entity Type | Precision Target | Recall Target | Notes |
|-------------|------------------|---------------|-------|
| PERSON | 92% | 90% | Critical for E-E-A-T |
| ORG | 90% | 88% | Important for authority |
| GPE/LOC | 94% | 92% | Usually high accuracy |
| PRODUCT | 85% | 80% | May need domain tuning |
| CONCEPT | 80% | 75% | Hardest category |
| DATE | 95% | 93% | Pattern-based, high accuracy |

#### 10.1.3 Evaluation Protocol

```python
def evaluate_ner_performance(model, test_data):
    """
    Evaluate NER model on test dataset.

    test_data: List of (text, annotations) tuples
    annotations: {'entities': [(start, end, label), ...]}
    """
    from seqeval.metrics import precision_score, recall_score, f1_score

    y_true = []
    y_pred = []

    for text, annotations in test_data:
        doc = model(text)

        # Convert to BIO format
        true_labels = convert_to_bio(text, annotations['entities'])
        pred_labels = convert_to_bio(text, [(e.start_char, e.end_char, e.label_) for e in doc.ents])

        y_true.append(true_labels)
        y_pred.append(pred_labels)

    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'per_entity': classification_report(y_true, y_pred, output_dict=True)
    }
```

### 10.2 Entity Linking Accuracy

#### 10.2.1 Linking Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Linking Precision | Correct links / All predicted links | 85% |
| Linking Recall | Correct links / All gold links | 80% |
| NIL Accuracy | Correctly identified unlinkable entities | 90% |
| Disambiguation Accuracy | Correct choice among candidates | 82% |

#### 10.2.2 Evaluation Dataset

Build evaluation dataset from:
- Wikipedia anchor text (silver standard)
- Manually annotated SEO content (gold standard)
- AIDA-CoNLL benchmark (academic standard)

### 10.3 Coverage Score Correlation with Rankings

#### 10.3.1 Correlation Targets

| Correlation Type | Minimum r | Target r | Measurement |
|------------------|-----------|----------|-------------|
| Coverage vs. Ranking Position | -0.30 | -0.45 | Pearson correlation |
| Authority Score vs. Position | -0.25 | -0.40 | Pearson correlation |
| Composite Score vs. Position | -0.35 | -0.50 | Pearson correlation |

*Note: Negative correlation expected (higher score = lower/better position)*

#### 10.3.2 Validation Study Design

```python
def validate_score_ranking_correlation(keywords, scoring_pipeline):
    """
    Validate entity score correlation with actual rankings.

    Study Design:
    1. Select 100+ keywords across different niches
    2. For each keyword, analyze top 50 ranking URLs
    3. Calculate entity scores for each URL
    4. Compute correlation with ranking position
    """
    results = []

    for keyword in keywords:
        # Get SERP results
        serp = get_serp_results(keyword, top_n=50)

        for position, url in enumerate(serp, 1):
            content = extract_content(url)
            score = scoring_pipeline.score_content(content, keyword)

            results.append({
                'keyword': keyword,
                'url': url,
                'position': position,
                'composite_score': score.composite_score,
                'coverage_score': score.component_scores['coverage'],
                'authority_score': score.component_scores['authority']
            })

    df = pd.DataFrame(results)

    correlations = {
        'composite_vs_position': df['composite_score'].corr(df['position']),
        'coverage_vs_position': df['coverage_score'].corr(df['position']),
        'authority_vs_position': df['authority_score'].corr(df['position'])
    }

    return correlations
```

### 10.4 Processing Performance Requirements

#### 10.4.1 Latency Targets

| Operation | P50 Latency | P95 Latency | P99 Latency |
|-----------|-------------|-------------|-------------|
| NER (Tier 1, 1000 words) | 150ms | 300ms | 500ms |
| NER (Tier 2, 1000 words) | 800ms | 1500ms | 2500ms |
| Entity Linking (per entity) | 20ms | 50ms | 100ms |
| Full Analysis (2000 words) | 2s | 4s | 6s |
| Competitor Analysis (10 URLs) | 15s | 25s | 40s |

#### 10.4.2 Throughput Targets

| Deployment | Pages/minute | Concurrent Users |
|------------|--------------|------------------|
| Single Instance | 30 | 10 |
| Clustered (3 nodes) | 90 | 30 |
| With GPU | 120 | 50 |

#### 10.4.3 Resource Requirements

| Configuration | CPU | RAM | GPU | Storage |
|---------------|-----|-----|-----|---------|
| Minimum | 4 cores | 8GB | None | 10GB |
| Recommended | 8 cores | 16GB | Optional | 25GB |
| Production | 16 cores | 32GB | T4/A10 | 50GB |

### 10.5 Quality Assurance Checklist

#### 10.5.1 Pre-Release Validation

- [ ] NER F1 score >89% on held-out test set
- [ ] Entity linking accuracy >82% on evaluation dataset
- [ ] Coverage score shows negative correlation with rankings (r < -0.35)
- [ ] P95 latency within targets for all operations
- [ ] No memory leaks under sustained load
- [ ] Graceful degradation when external APIs unavailable
- [ ] Rate limiting implemented for external API calls
- [ ] Cache hit rate >80% for repeat queries

#### 10.5.2 Ongoing Monitoring

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| NER accuracy drift | >5% decline | Retrain/evaluate |
| API error rate | >5% | Check external services |
| Cache miss rate | >30% | Review cache strategy |
| P95 latency | >2x target | Scale/optimize |
| Memory usage | >80% | Investigate leaks |

---

## Appendix A: Decision Matrices

### A.1 NER Model Selection Matrix

| Criterion | Weight | en_core_web_sm | en_core_web_md | en_core_web_lg | en_core_web_trf |
|-----------|--------|----------------|----------------|----------------|-----------------|
| Accuracy | 0.30 | 2 | 3 | 4 | 5 |
| Speed | 0.25 | 5 | 4 | 4 | 1 |
| Memory | 0.15 | 5 | 4 | 3 | 2 |
| Cost (GPU) | 0.15 | 5 | 5 | 5 | 2 |
| Maintainability | 0.15 | 4 | 4 | 4 | 3 |
| **Weighted Score** | | **3.85** | **3.85** | **4.00** | **2.85** |

**Recommendation:** `en_core_web_lg` for primary use, `en_core_web_trf` for accuracy-critical tasks.

### A.2 Knowledge Base Integration Matrix

| Criterion | Weight | Local KB | Wikipedia API | Wikidata SPARQL | Commercial API |
|-----------|--------|----------|---------------|-----------------|----------------|
| Latency | 0.25 | 5 | 3 | 2 | 4 |
| Accuracy | 0.25 | 3 | 4 | 5 | 5 |
| Coverage | 0.20 | 3 | 4 | 5 | 4 |
| Cost | 0.15 | 5 | 4 | 4 | 1 |
| Freshness | 0.15 | 2 | 5 | 5 | 5 |
| **Weighted Score** | | **3.70** | **3.90** | **4.15** | **3.80** |

**Recommendation:** Hybrid approach with Local KB primary + Wikidata enrichment.

---

## Appendix B: References and Sources

### Research and Industry Sources

1. [Named Entity Recognition and SEO: The Ultimate Guide](https://marketbrew.ai/named-entity-recognition-and-seo) - MarketBrew
2. [Semantic SEO in 2025: A Complete Guide for Entity Based SEO](https://niumatrix.com/semantic-seo-guide/) - Niumatrix
3. [Entity-first SEO: How to align content with Google's Knowledge Graph](https://searchengineland.com/guide/entity-first-content-optimization) - Search Engine Land
4. [Entity-based SEO: An explainer for SEOs and content marketers](https://blog.hubspot.com/marketing/entities-seo) - HubSpot
5. [spaCy Facts & Figures](https://spacy.io/usage/facts-figures) - spaCy Documentation
6. [spaCy Trained Models & Pipelines](https://spacy.io/models) - spaCy Documentation
7. [spacy-entity-linker](https://github.com/egerber/spaCy-entity-linker) - GitHub
8. [spacyfishing: Entity-Fishing Wrapper](https://github.com/Lucaterre/spacyfishing) - GitHub
9. [Wikidata SPARQL Query Service](https://query.wikidata.org/) - Wikidata
10. [7 sameAs Schema Best Practices](https://aubreyyung.com/sameas-schema/) - Aubrey Yung
11. [Entity-based competitor analysis: An SEO's guide](https://searchengineland.com/entity-based-competitor-analysis-seo-guide-438259) - Search Engine Land
12. [Entity Co-occurrence: Meaning, Importance in NLP & SEO](https://www.immwit.com/wiki/entity-co-occurrence/) - Immwit
13. [Google Search Quality Rater Guidelines](https://guidelines.raterhub.com/searchqualityevaluatorguidelines.pdf) - Google
14. [Fine-Tuning BERT for Named Entity Recognition](https://medium.com/@whyamit101/fine-tuning-bert-for-named-entity-recognition-ner-b42bcf55b51d) - Medium
15. [Named Entity Recognition on OntoNotes v5](https://paperswithcode.com/sota/named-entity-recognition-ner-on-ontonotes-v5) - Papers With Code
16. [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) - Hugging Face
17. [Wikipedia-API Python Package](https://pypi.org/project/Wikipedia-API/) - PyPI
18. [Extracting Data from Wikidata Using SPARQL and Python](https://itnext.io/extracting-data-from-wikidata-using-sparql-and-python-59e0037996f) - ITNEXT
19. [Entity salience in SEO](https://www.szymonslowik.com/entity-salience-in-seo/) - Szymon Slowik
20. [Knowledge Graph Triplets](https://www.emergentmind.com/topics/knowledge-graph-triplets) - Emergent Mind

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Next Review: April 2026*

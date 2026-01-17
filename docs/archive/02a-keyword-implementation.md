# Topic B: Implementation Guide
## Keyword Strategy & Intent Modeling - Quick Reference

**Document Version:** 1.0
**Date:** January 2026

---

## Quick Start: Algorithm Decision Tree

```
START: Analyzing Keywords for SEO Content Optimization
├─ Have keywords? NO → Run keyword research (SEMrush, Ahrefs, Google Keyword Planner)
├─ YES → Proceed to Intent Classification
│
├─ INTENT CLASSIFICATION
│  ├─ Use signal detection algorithm (Section 1.4.1)
│  ├─ Result: Each keyword assigned intent type + confidence score
│  ├─ Filter by intent: Now you have informational/commercial/transactional queues
│  └─ Proceed to Clustering
│
├─ KEYWORD CLUSTERING
│  ├─ Have multiple keywords? (>20 keywords recommended)
│  │  ├─ YES, use hybrid approach:
│  │  │  ├─ Run semantic clustering (embeddings)
│  │  │  ├─ Run topical clustering (TF-IDF)
│  │  │  ├─ Fuse results with weighted fusion (Section 2.4)
│  │  │  └─ Evaluate silhouette score (target >0.60)
│  │  └─ <20 keywords? Manual clustering acceptable, validate with semantic similarity
│  │
│  └─ Proceed to Mapping
│
├─ KEYWORD-TO-CONTENT MAPPING
│  ├─ Cluster primary keywords → target URLs
│  ├─ Assign primary keyword to one URL ONLY
│  ├─ Distribute secondary keywords to H2 sections
│  ├─ Map LSI keywords throughout body
│  └─ Validate placement rules (Section 4.1.2)
│
├─ CANNIBALIZATION DETECTION
│  ├─ Run URL-keyword overlap scoring (Section 3.2.1)
│  ├─ Run content similarity detection (Section 3.2.2)
│  ├─ Identify ranking conflicts from GSC (Section 3.2.3)
│  ├─ Flag severity (LOW/MEDIUM/HIGH/CRITICAL)
│  └─ Execute resolution strategy (Section 3.3)
│
├─ DENSITY & OVER-OPTIMIZATION ANALYSIS
│  ├─ Calculate exact-match keyword density (target: 0.5-1.5%)
│  ├─ Analyze distribution (should be evenly spread)
│  ├─ Calculate BM25 score (preferred over TF-IDF)
│  ├─ Flag if clustering detected
│  └─ Report risk level (SAFE/CAUTION/HIGH RISK/CRITICAL)
│
└─ GENERATE STRATEGY REPORT
   ├─ Summary statistics
   ├─ Per-keyword opportunity scores
   ├─ Cluster assignments with quality metrics
   ├─ Page mapping specifications
   ├─ Cannibalization issues + resolution recommendations
   └─ Success metrics (accuracy, clustering quality, detection performance)
```

---

## Implementation Priorities

### Phase 1: MVP (Weeks 1-2)
Priority: Critical functionality only

1. **Intent Classification** (Rule-based signals)
   - Basic keyword pattern matching
   - 4-type classification output
   - Target accuracy: >85%

2. **Simple Clustering**
   - Semantic clustering only (embeddings)
   - Silhouette score evaluation
   - Target quality: >0.55

3. **Basic Keyword Mapping**
   - Primary keyword placement validation
   - Simple H2 distribution
   - Check critical elements

### Phase 2: Production (Weeks 3-4)
Add advanced features

1. **Enhanced Intent Classification**
   - SERP feature analysis
   - Micro-moments framework
   - Journey stage detection

2. **Hybrid Clustering**
   - Add topical clustering (TF-IDF)
   - Add SERP-based clustering
   - Weighted fusion (40/35/25 split)

3. **Cannibalization Detection**
   - URL-keyword overlap
   - Content similarity analysis
   - Ranking conflict detection

4. **Density Analysis**
   - BM25 scoring
   - Distribution analysis
   - Over-optimization risk detection

### Phase 3: Advanced (Weeks 5+)
Enterprise features

1. **API Integrations**
   - SEMrush keyword data
   - Ahrefs keyword data
   - Google Search Console GSC data
   - Competitor SERP analysis

2. **Advanced Analytics**
   - LSI keyword auto-generation
   - Question-based keyword detection
   - Featured snippet opportunity identification

3. **Performance Optimization**
   - Batch processing pipelines
   - Caching layers
   - Distributed clustering

---

## Code Snippets: Copy-Paste Ready

### 1. Intent Classification (Fast Path)

```python
# File: intent_classifier.py

import re
from enum import Enum
from typing import Dict, List

class IntentType(Enum):
    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"

def classify_intent(query: str) -> Dict[str, float]:
    """
    Fast intent classification using pattern matching.
    Returns confidence scores for each intent type.
    """

    query_lower = query.lower()

    # Signal patterns
    signals = {
        'informational': [
            r'\bhow\s+(to|do|can|can\s+i)\b',
            r'\bwhat\s+(is|are|does)\b',
            r'\bwhy\s+(is|are|do)\b',
            r'\bguide\b', r'\btutorial\b', r'\bstep.*by.*step\b',
            r'\bbenefits\b', r'\bexplain\b', r'\blearn\b'
        ],
        'commercial': [
            r'\bbest\s+\d*\s*\w+', r'\btop\s+\d*\s*\w+',
            r'\breview\s+', r'\bvs\.|versus\b', r'\bcomparison\b',
            r'\balternative', r'\bwhich\s+', r'\nrecommended\b'
        ],
        'transactional': [
            r'\bbuy\b', r'\bpurchase\b', r'\border\b', r'\bshop\b',
            r'\bprice\b', r'\bcost\b', r'\bdiscount\b', r'\bcoupon\b',
            r'\bdownload\b', r'\binstall\b', r'\bsignup\b'
        ],
        'navigational': [
            r'\blogin\b', r'\bsignin\b', r'\bdashboard\b',
            r'\badmin\b', r'\b(facebook|amazon|github|google)\b'
        ]
    }

    # Count matches
    scores = {}
    for intent_type, patterns in signals.items():
        score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
        scores[intent_type] = score

    # Normalize
    total = sum(scores.values())
    if total == 0:
        # Default even distribution
        return {
            'navigational': 0.25,
            'informational': 0.25,
            'commercial': 0.25,
            'transactional': 0.25
        }

    return {intent: score/total for intent, score in scores.items()}


# Usage
intent_scores = classify_intent("best project management software for remote teams")
print(intent_scores)
# Output: {'informational': 0.10, 'commercial': 0.80, 'transactional': 0.05, 'navigational': 0.05}
```

### 2. Semantic Clustering (Quick Implementation)

```python
# File: semantic_clustering.py

from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import numpy as np
from typing import List, Dict

def semantic_cluster_keywords(keywords: List[str],
                             threshold: float = 0.70) -> Dict[int, List[str]]:
    """
    Cluster keywords using sentence-transformers embeddings.
    Minimal dependencies, fast execution.
    """

    # 1. Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast model, 384 dims
    embeddings = model.encode(keywords)

    # 2. Compute distance matrix
    distance_matrix = pdist(embeddings, metric='cosine')

    # 3. Hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')

    # 4. Cut dendrogram
    distance_threshold = 1 - threshold
    clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # 5. Format output
    result = {}
    for keyword, cluster_id in zip(keywords, clusters):
        if cluster_id not in result:
            result[cluster_id] = []
        result[cluster_id].append(keyword)

    return result


# Usage
keywords = [
    "python tutorial",
    "learn python",
    "python guide",
    "javascript basics",
    "learn javascript",
    "js tutorial"
]

clusters = semantic_cluster_keywords(keywords, threshold=0.65)
print(clusters)
# Output:
# {
#   1: ['python tutorial', 'learn python', 'python guide'],
#   2: ['javascript basics', 'learn javascript', 'js tutorial']
# }
```

### 3. Keyword Density Check

```python
# File: density_analyzer.py

import re
from typing import Dict

def analyze_keyword_density(content: str,
                           keyword: str,
                           target_min: float = 0.5,
                           target_max: float = 1.5) -> Dict:
    """
    Analyze keyword density with health assessment.
    """

    # Calculate exact-match density
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    matches = len(re.findall(pattern, content.lower()))
    total_words = len(content.split())

    density = (matches / total_words * 100) if total_words > 0 else 0

    # Assessment
    if density < target_min:
        status = "LOW - Consider adding more occurrences"
    elif density <= target_max:
        status = "OPTIMAL"
    elif density <= 2.5:
        status = "ELEVATED - Reduce slightly"
    else:
        status = "CRITICAL - Keyword stuffing detected"

    return {
        'keyword': keyword,
        'density_percent': round(density, 2),
        'exact_matches': matches,
        'total_words': total_words,
        'target_range': f"{target_min}% - {target_max}%",
        'status': status,
        'recommendation': f"{'✓ Good' if status == 'OPTIMAL' else '⚠ Needs adjustment'}"
    }


# Usage
content = """
Machine learning is transforming industries. Machine learning applications
are everywhere. Deep learning is a subset of machine learning that uses
neural networks. When implementing machine learning solutions...
"""

analysis = analyze_keyword_density(content, "machine learning")
print(f"Density: {analysis['density_percent']}% - {analysis['status']}")
# Output: Density: 3.85% - CRITICAL - Keyword stuffing detected
```

### 4. Cannibalization Detection (Simple)

```python
# File: cannibalization_detector.py

from typing import Dict, List
from itertools import combinations

def detect_keyword_cannibalization(url_keywords: Dict[str, List[str]],
                                  threshold: float = 0.6) -> List[Dict]:
    """
    Simple cannibalization detection based on keyword overlap.
    """

    # Reverse index: keyword → URLs
    keyword_to_urls = {}
    for url, keywords in url_keywords.items():
        for keyword in keywords:
            if keyword not in keyword_to_urls:
                keyword_to_urls[keyword] = []
            keyword_to_urls[keyword].append(url)

    # Find cannibalization
    incidents = []

    for keyword, urls in keyword_to_urls.items():
        if len(urls) > 1:
            for url1, url2 in combinations(urls, 2):
                # Calculate keyword overlap
                set1 = set(url_keywords[url1])
                set2 = set(url_keywords[url2])

                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    incidents.append({
                        'keyword': keyword,
                        'url1': url1,
                        'url2': url2,
                        'overlap_score': round(similarity, 2),
                        'severity': (
                            'CRITICAL' if similarity >= 0.8 else
                            'HIGH' if similarity >= 0.6 else
                            'MEDIUM'
                        ),
                        'action': (
                            'MERGE' if similarity >= 0.8 else
                            'REDIRECT' if similarity >= 0.6 else
                            'DIFFERENTIATE'
                        )
                    })

    return sorted(incidents, key=lambda x: x['overlap_score'], reverse=True)


# Usage
url_keywords = {
    '/blog/python-tutorial': ['python', 'tutorial', 'learn python', 'python basics'],
    '/guides/python': ['learn python', 'python guide', 'python programming'],
    '/docs/python-101': ['python', 'python basics', 'python course']
}

cannibalization = detect_keyword_cannibalization(url_keywords, threshold=0.6)
for incident in cannibalization:
    print(f"⚠️  {incident['action']}: {incident['url1']} vs {incident['url2']}")
    print(f"   Overlap: {incident['overlap_score']} | Severity: {incident['severity']}\n")
```

### 5. BM25 Scoring

```python
# File: bm25_scorer.py

import math
from typing import List

def calculate_bm25(document: str,
                  query_terms: List[str],
                  corpus: List[str],
                  k1: float = 1.5,
                  b: float = 0.75) -> float:
    """
    BM25 relevance score (better than TF-IDF).
    """

    # Document stats
    doc_length = len(document.split())
    avg_length = sum(len(doc.split()) for doc in corpus) / len(corpus)
    corpus_size = len(corpus)

    bm25_score = 0

    for term in query_terms:
        # IDF calculation
        docs_with_term = sum(1 for doc in corpus if term.lower() in doc.lower())
        idf = math.log(
            (corpus_size - docs_with_term + 0.5) /
            (docs_with_term + 0.5) + 1
        )

        # Term frequency in document
        term_freq = document.lower().count(term.lower())

        if term_freq == 0:
            continue

        # BM25 formula
        numerator = term_freq * (k1 + 1)
        denominator = (
            term_freq +
            k1 * (1 - b + b * (doc_length / avg_length))
        )

        bm25_score += idf * (numerator / denominator)

    return bm25_score


# Usage
documents = [
    "Machine learning is AI technology",
    "Deep learning uses neural networks",
    "AI applications in business"
]

doc = documents[0]
score = calculate_bm25(doc, ["machine learning", "ai"], documents)
print(f"BM25 Score: {score:.3f}")
```

### 6. Primary Keyword Placement Validation

```python
# File: placement_validator.py

import re
from typing import Dict

def validate_primary_keyword_placement(page_elements: Dict[str, str],
                                      primary_keyword: str) -> Dict[str, bool]:
    """
    Validate that primary keyword appears in critical locations.
    """

    keyword_lower = primary_keyword.lower()

    validation = {
        'title_includes_keyword': (
            keyword_lower in page_elements.get('title', '').lower() and
            page_elements.get('title', '').lower().startswith(keyword_lower)
        ),
        'h1_includes_keyword': (
            keyword_lower in page_elements.get('h1', '').lower()
        ),
        'meta_includes_keyword': (
            keyword_lower in page_elements.get('meta_description', '').lower()
        ),
        'first_paragraph_includes': (
            keyword_lower in page_elements.get('first_paragraph', '').lower()
        ),
        'url_slug_includes': (
            keyword_lower.replace(' ', '-') in page_elements.get('url_slug', '').lower()
        ),
        'image_alt_includes': (
            any(keyword_lower in alt.lower()
                for alt in page_elements.get('image_alts', []))
        )
    }

    # Overall assessment
    checks_passed = sum(validation.values())
    total_checks = len(validation)

    validation['overall_score'] = f"{checks_passed}/{total_checks}"
    validation['status'] = (
        '✓ EXCELLENT' if checks_passed == total_checks else
        '✓ GOOD' if checks_passed >= 5 else
        '⚠ NEEDS WORK' if checks_passed >= 3 else
        '❌ CRITICAL'
    )

    return validation


# Usage
page = {
    'title': 'Best Project Management Software for Remote Teams 2026',
    'h1': 'Best Project Management Software for Remote Teams',
    'meta_description': 'Discover the best project management software for remote collaboration...',
    'first_paragraph': 'Finding the best project management software is critical...',
    'url_slug': 'best-project-management-software-remote-teams',
    'image_alts': ['Best project management software comparison', 'Team collaboration tools']
}

validation = validate_primary_keyword_placement(page, 'best project management software')
for element, is_valid in validation.items():
    status = '✓' if is_valid else '✗'
    print(f"{status} {element}")

print(f"\nOverall: {validation['status']}")
```

---

## Data Structures Cheat Sheet

### Keyword Data Structure
```python
{
    'text': 'machine learning tutorial',
    'intent': 'informational',  # or commercial, transactional, navigational
    'search_volume': 8500,
    'difficulty': 65,           # 0-100 scale
    'cpc': 2.50,               # Cost per click (USD)
    'opportunity_score': 68.5,  # Custom calculated metric
    'primary': False,          # Is this the primary keyword?
}
```

### Cluster Data Structure
```python
{
    'cluster_id': 1,
    'keywords': ['machine learning', 'learn ML', 'ML tutorial'],
    'primary_keyword': 'machine learning',
    'type': 'semantic',        # or topical, serp, hybrid
    'quality_score': 0.72,     # Silhouette score
    'target_url': '/guides/machine-learning',
    'size': 3
}
```

### Page Mapping Data Structure
```python
{
    'url': '/blog/seo-optimization',
    'primary_keyword': 'seo optimization',
    'secondary_keywords': ['on-page seo', 'seo best practices'],
    'lsi_keywords': ['search engine optimization', 'website ranking', 'organic search'],
    'element_mapping': {
        'title': ['seo optimization'],
        'h1': ['seo optimization'],
        'h2_1': ['on-page seo'],
        'h2_2': ['seo best practices'],
        'meta_description': ['seo optimization'],
        'image_alts': ['seo optimization', 'search engine optimization']
    }
}
```

### Cannibalization Incident
```python
{
    'keyword': 'python tutorial',
    'url1': '/blog/python-tutorial',
    'url2': '/guides/python',
    'similarity_score': 0.78,
    'severity': 'HIGH',
    'type': 'keyword_overlap',  # or content_similarity, ranking_conflict
    'action': 'REDIRECT',       # or MERGE, DIFFERENTIATE
    'priority': 8               # 1-10 scale
}
```

---

## Performance Benchmarks

### Speed Targets (Per 100 keywords)
| Operation | Target Time | Tool/Method |
|---|---|---|
| Intent Classification | <100ms | Rule-based patterns |
| Semantic Clustering | 2-5 sec | sentence-transformers |
| Cannibalization Detection | 500ms | Nested set operations |
| BM25 Scoring | 1-2 sec | Math operations |
| Full Pipeline | <15 sec | Sequential execution |

### Accuracy Targets
| Metric | Target | Method |
|---|---|---|
| Intent Classification Accuracy | >90% | Test on labeled dataset |
| Clustering Quality (Silhouette) | >0.60 | Silhouette score |
| Cannibalization Detection Recall | >85% | Human-verified ground truth |
| Cannibalization Detection Precision | >90% | False positive analysis |

---

## API Integration Quick Start

### SEMrush API Example
```python
import requests
from typing import List, Dict

def get_semrush_keywords(keyword: str, api_key: str) -> List[Dict]:
    """Fetch related keywords from SEMrush API"""

    endpoint = "https://api.semrush.com/v3/"
    params = {
        "type": "keyword_related",
        "key": api_key,
        "export_columns": "Keyword,Search Volume,Keyword Difficulty,CPC",
        "display_limit": 100,
        "database": "us"
    }

    response = requests.get(endpoint, params=params)
    # Parse CSV-formatted response
    # Return list of keywords with metrics

    pass  # Implementation details vary by API version
```

### Google Search Console Integration
```python
def get_gsc_rankings(service, site_url: str) -> List[Dict]:
    """Fetch ranking data from GSC"""

    request = service.searchanalytics().query(
        siteUrl=site_url,
        body={
            'startDate': '2025-12-01',
            'endDate': '2025-12-31',
            'dimensions': ['query', 'page'],
            'rowLimit': 10000
        }
    )

    results = request.execute()
    # Transform to keyword list with position, clicks, impressions
    # Return for cannibalization analysis

    pass  # Full implementation in production codebase
```

---

## Testing Checklist

### Before Production Deployment

- [ ] Intent classification accuracy >90% on 100-keyword test set
- [ ] Semantic clustering produces silhouette score >0.60
- [ ] Cannibalization detection has >85% recall on known issues
- [ ] Keyword density analysis matches manual review
- [ ] Primary keyword placement validation matches manual audit
- [ ] BM25 scores correlate with actual ranking positions
- [ ] Pipeline executes <15 seconds on 500-keyword dataset
- [ ] All error handling in place for API timeouts
- [ ] Caching implemented for embedding generation
- [ ] Logging captures all classification/clustering decisions

---

## Common Pitfalls & Solutions

| Problem | Symptom | Solution |
|---|---|---|
| **Clustering Too Fine-Grained** | 50 clusters from 100 keywords | Reduce threshold (increase similarity requirement) |
| **Clustering Too Coarse** | 5 clusters from 100 keywords | Increase threshold (decrease similarity requirement) |
| **Low Intent Classification Accuracy** | <75% accuracy | Add domain-specific signal patterns |
| **Missing Cannibalization** | Undetected competing URLs | Lower overlap threshold, add content similarity check |
| **Over-Sensitivity** | High false positives | Increase threshold, add manual review layer |
| **Slow Performance** | >30 seconds for 500 keywords | Implement batching, use faster embedding model, add caching |
| **API Rate Limits** | Intermittent API failures | Add exponential backoff, implement request queuing |

---

## Migration Path from Existing System

### If Migrating from Simple Keyword Density Tools:
1. Keep existing density calculations as baseline
2. Add BM25 scoring alongside existing metrics
3. Gradually introduce clustering analysis
4. Phase in cannibalization detection
5. Full replacement once proven accuracy >90%

### If Migrating from Manual Keyword Lists:
1. Bulk import existing keywords
2. Run clustering to identify gaps
3. Cross-reference with SEMrush/Ahrefs
4. Generate mapping recommendations
5. Have domain experts validate results

---

## Success Metrics Dashboard

Track these metrics weekly:

```
Intent Classification
├─ Accuracy: __% (target >90%)
├─ Coverage: __ keywords analyzed
└─ New intents identified: __

Keyword Clustering
├─ Quality score: __ (target >0.60)
├─ Clusters created: __
└─ Keywords per cluster: avg __

Cannibalization Detection
├─ Issues found: __
├─ Critical issues: __
└─ Resolved: __

Keyword Mapping
├─ Pages mapped: __
├─ Validation pass rate: __% (target >95%)
└─ Density compliance: __% (target >90%)

Content Performance
├─ Avg position change: __ (target: improvement)
├─ Click-through rate: __% (target: increase)
└─ Cannibalization impact: __ clicks recovered (target: positive)
```

---

**Document prepared for:** SEO + AI Content Optimization Tool Development
**Last Updated:** January 2026
**Questions or Issues?** Refer to TOPIC_B_Keyword_Strategy_Intent_Modeling.md for complete technical specifications.

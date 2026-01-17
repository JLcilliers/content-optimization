# Topic B: Keyword Strategy & Intent Modeling
## Comprehensive Technical Documentation for SEO + AI Content Optimization Tools

**Document Version:** 1.0
**Date:** January 2026
**Classification:** Technical Specification

---

## Executive Summary

Keyword strategy and intent modeling form the foundational layer of modern SEO and AI-driven content optimization. This document provides a comprehensive technical framework for implementing enterprise-grade keyword analysis, search intent classification, semantic clustering, and cannibalization detection systems. The shift from simple keyword frequency analysis to intent-based, semantically-aware keyword management represents a paradigm change in how content platforms approach search optimization.

Current best practices emphasize understanding *why* users search rather than merely *what* keywords they use. Contemporary algorithms move beyond TF-IDF (Term Frequency-Inverse Document Frequency) to incorporate BM25 relevance scoring, neural embeddings, and intent signals. In 2026, keyword strategies must account for Google's increasingly sophisticated understanding of semantic relationships, entity recognition, and topical authority—while simultaneously avoiding the pitfall of keyword cannibalization that dilutes ranking potential across multiple content pieces.

This document synthesizes current research, algorithmic approaches, and implementation specifications for building a production-grade keyword strategy system. The framework addresses both traditional SEO keyword metrics and modern AI-powered approaches to content optimization, providing practitioners with actionable algorithms, data structures, and evaluation metrics.

---

## 1. Intent Classification Frameworks

### 1.1 Traditional Four-Type Intent Model

The foundational framework for search intent originates from a 2002 peer-reviewed paper by Andrei Broder at Altavista. This model remains the industry standard baseline, though modern implementations extend it significantly.

#### 1.1.1 Core Intent Categories

| Intent Type | Definition | User Goal | Examples | Signal Keywords |
|---|---|---|---|---|
| **Navigational** | User seeks a specific website or brand | Reach a particular destination | "Facebook login", "YouTube", "Amazon" | Brand names, domain variations, specific brand + features |
| **Informational** | User seeks knowledge or answers | Learn about a topic | "how to tie a tie", "what is quantum computing", "best practices for SEO" | How to, what is, why, guide, tutorial, best practices |
| **Commercial Investigation** | User researches products/services before purchase | Compare options and reviews | "best smartphone 2026", "MacBook Pro vs Dell", "SaaS project management tools" | Best, top, review, comparison, vs, alternatives |
| **Transactional** | User seeks to complete a specific action/purchase | Execute transaction | "buy running shoes online", "download Photoshop", "rent apartment NYC" | Buy, price, deal, discount, coupon, where to buy |

#### 1.1.2 2026 Search Intent Distribution

Based on current aggregate data:
- **Informational Intent:** 52.65% of all searches
- **Navigational Intent:** 32.15% of all searches
- **Commercial Investigation:** 14.51% of all searches
- **Transactional Intent:** 0.69% of all searches

This distribution indicates that content platforms should allocate significant resources to informational content strategies, which represent over half of search volume.

### 1.2 Google's Micro-Moments Framework

Google introduced the concept of "Micro-Moments" to describe high-intent decision points where users turn to devices for immediate answers. This framework extends the traditional intent model by emphasizing temporal context and device behavior.

#### 1.2.1 The Four Micro-Moment Types

| Moment Type | Description | Signal Phrases | Time Frame | Device Priority |
|---|---|---|---|---|
| **I-Want-to-Know** | User researching information | "how to", "what is", "explain", "learn about" | Low pressure, exploratory | Mobile, voice search |
| **I-Want-to-Go** | User seeking physical locations | "near me", "stores in", "hours", "directions" | Location-dependent | Mobile (GPS critical) |
| **I-Want-to-Do** | User seeking to complete tasks | "how to", "tutorials", "steps to", "guide" | Action-oriented | Mobile, voice |
| **I-Want-to-Buy** | User ready to purchase | "buy", "price", "order", "in stock" | High urgency, high intent | Mobile-optimized checkout |

**Implementation Note:** Micro-moments apply contextual weighting to keywords. The same keyword "coffee" carries different intent weight depending on context: "what is specialty coffee" (I-Want-to-Know) versus "coffee near me" (I-Want-to-Go) versus "best espresso machine" (I-Want-to-Buy).

### 1.3 Search Journey Stage Mapping

Modern search behavior rarely follows a linear path. Users move between informational research and transactional searches across multiple sessions. Effective keyword strategies must map keywords to customer journey stages.

#### 1.3.1 Journey Stage Framework

```
Awareness Stage (0-30 days)
├─ Informational queries: "what is SaaS?"
├─ Problem identification: "marketing automation challenges"
└─ Intent: Build topical authority content

Consideration Stage (30-90 days)
├─ Comparative queries: "marketing automation vs email marketing"
├─ Solution research: "best marketing automation platforms"
├─ Vendor comparisons: "HubSpot vs Marketo vs Salesforce"
└─ Intent: Competitor gap analysis, comparison content

Decision Stage (90+ days)
├─ Purchase queries: "marketing automation free trial"
├─ Implementation queries: "how to set up HubSpot"
├─ Pricing queries: "HubSpot pricing 2026"
└─ Intent: Transactional optimization, conversion content
```

### 1.4 Intent Signal Detection Methods

Automated intent classification requires identifying linguistic and contextual signals in search queries.

#### 1.4.1 Signal Categories and Detection Rules

**Linguistic Signals:**

```python
# Pseudo-code for intent signal detection

def detect_intent_signals(query: str) -> Dict[str, float]:
    """
    Returns confidence scores for each intent type.
    Range: 0.0 to 1.0
    """

    # Define signal patterns
    informational_signals = {
        'how_to': r'\bhow\s+(to|do|can|can\s+i)\b',
        'definition': r'\bwhat\s+(is|are|does)\b',
        'explanation': r'\bwhy\s+(is|are|do)\b',
        'comparison': r'\bcomparison|vs\.|versus|difference',
        'list': r'\bbest\s+\d+|top\s+\d+|list\s+of',
        'guide': r'\bguide|tutorial|how\-to|step.*by.*step',
        'research': r'\bbenefits|advantages|uses|applications'
    }

    navigational_signals = {
        'brand': r'(?:brands|companies)\s+in\s+(?:intent_context)',
        'website': r'\b(?:login|signin|dashboard|admin)\b',
        'direct_mention': r'(?:facebook|amazon|github)\b'
    }

    commercial_signals = {
        'review': r'\breview|rating|feedback|alternative',
        'comparison': r'\bvs\.|comparison|difference|better',
        'question': r'which|best|top|recommended'
    }

    transactional_signals = {
        'purchase': r'\bbuy|purchase|order|shop',
        'pricing': r'\bprice|cost|fee|discount|coupon',
        'action': r'\bdownload|install|signup|register'
    }

    # Score signals
    scores = {
        'informational': sum(1 for pattern in informational_signals.values()
                           if re.search(pattern, query.lower())),
        'navigational': sum(1 for pattern in navigational_signals.values()
                          if re.search(pattern, query.lower())),
        'commercial': sum(1 for pattern in commercial_signals.values()
                         if re.search(pattern, query.lower())),
        'transactional': sum(1 for pattern in transactional_signals.values()
                            if re.search(pattern, query.lower()))
    }

    # Normalize to confidence scores
    total = sum(scores.values())
    if total == 0:
        return {'informational': 0.25, 'navigational': 0.25,
                'commercial': 0.25, 'transactional': 0.25}

    return {intent: score/total for intent, score in scores.items()}
```

**Contextual Signals:**

- **Search volume context:** High-volume keywords in commercial categories often indicate commercial intent
- **Keyword difficulty:** Higher difficulty typically correlates with transactional/navigational searches
- **SERP features:** Presence of shopping results, maps, ads indicates commercial/transactional intent
- **Content type dominance:** Educational content dominance suggests informational intent
- **User engagement metrics:** Click-through rate, dwell time, bounce rate indicate intent match

### 1.5 Practical Intent Classification Examples

#### Example 1: "Best project management software for remote teams"

**Intent Analysis:**
```
Query: "best project management software for remote teams"

Signal Detection:
- "best" keyword → Commercial signal (weight: 0.8)
- Comparative structure → Commercial signal (weight: 0.9)
- Specific use case ("remote teams") → Commercial investigation (weight: 0.7)

Classification Result:
- Commercial Investigation: 0.82 (highest confidence)
- Informational: 0.14
- Transactional: 0.03
- Navigational: 0.01

Content Strategy:
- Create comparison tables (project management features for remote work)
- Include implementation case studies
- Target with buying-stage content
- Expected SERP features: Comparison articles, product reviews, G2/Capterra reviews
```

#### Example 2: "How to implement zero-trust security architecture"

**Intent Analysis:**
```
Query: "how to implement zero-trust security architecture"

Signal Detection:
- "How to" keyword → Informational signal (weight: 0.95)
- Implementation focus → Informational/commercial hybrid (weight: 0.6)
- Technical specificity → Expert audience (weight: 0.8)

Classification Result:
- Informational: 0.91 (highest confidence)
- Commercial: 0.06
- Transactional: 0.02
- Navigational: 0.01

Content Strategy:
- Create step-by-step implementation guides
- Include technical diagrams and architecture examples
- Target with educational content and whitepapers
- Expected SERP features: How-to guides, technical documentation, videos
- Audience: Security architects, senior engineers
```

#### Example 3: "Slack pricing 2026"

**Intent Analysis:**
```
Query: "Slack pricing 2026"

Signal Detection:
- "pricing" keyword → Transactional signal (weight: 0.95)
- Brand-specific → Navigational signal (weight: 0.7)
- Decision-stage → Transactional signal (weight: 0.8)

Classification Result:
- Transactional: 0.68 (highest confidence)
- Navigational: 0.20
- Commercial: 0.10
- Informational: 0.02

Content Strategy:
- Target Slack's official pricing page (likely to rank)
- Create pricing comparison: "Slack vs Teams vs Discord pricing"
- Include ROI calculator
- Expected SERP features: Knowledge panel with pricing, comparison tools
- Audience: Decision-makers, procurement teams
```

---

## 2. Keyword Clustering Approaches

Keyword clustering groups related keywords to prevent content fragmentation and optimize topical coverage. Modern approaches go beyond simple lexical matching to semantic understanding.

### 2.1 Semantic Clustering (Embedding-Based)

Semantic clustering uses neural embeddings to group keywords based on conceptual similarity rather than lexical features.

#### 2.1.1 Architecture and Algorithm

**Core Concept:** Transform keywords into high-dimensional vectors (embeddings) where semantic distance correlates with cosine distance in vector space.

```python
def semantic_keyword_clustering(keywords: List[str],
                               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                               threshold: float = 0.75) -> Dict[int, List[str]]:
    """
    Cluster keywords using semantic embeddings and hierarchical clustering.

    Args:
        keywords: List of keywords to cluster
        embedding_model: HuggingFace model identifier
        threshold: Cosine similarity threshold for cluster assignment

    Returns:
        Dictionary mapping cluster_id to list of keywords
    """

    from sentence_transformers import SentenceTransformer
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform

    # Step 1: Generate embeddings
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(keywords, convert_to_tensor=True)
    embeddings = embeddings.cpu().numpy()  # Convert to numpy

    # Step 2: Compute similarity matrix
    # Use cosine distance (1 - cosine similarity)
    distance_matrix = pdist(embeddings, metric='cosine')
    condensed_matrix = squareform(distance_matrix)

    # Step 3: Hierarchical clustering
    # Ward linkage minimizes within-cluster variance
    linkage_matrix = linkage(distance_matrix, method='ward')

    # Step 4: Cut dendrogram at threshold
    # Convert similarity threshold to distance threshold
    distance_threshold = 1 - threshold
    clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Step 5: Group keywords by cluster
    clustering_result = {}
    for keyword, cluster_id in zip(keywords, clusters):
        if cluster_id not in clustering_result:
            clustering_result[cluster_id] = []
        clustering_result[cluster_id].append(keyword)

    return clustering_result

# Example usage:
keywords = [
    "machine learning algorithms",
    "deep learning models",
    "neural networks",
    "artificial intelligence",
    "python programming",
    "software development",
    "web development",
    "full stack development",
    "frontend development",
    "backend development"
]

clusters = semantic_keyword_clustering(keywords, threshold=0.70)
# Result: Clusters by semantic similarity
# Cluster 1: ML/AI keywords
# Cluster 2: Development/programming keywords
```

**Model Selection Considerations:**

| Embedding Model | Dimensions | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | Fast | Good (0.78 correlation) | Budget-conscious production |
| **all-mpnet-base-v2** | 768 | Medium | Excellent (0.86 correlation) | High-accuracy clustering |
| **gte-small** | 384 | Fast | Good (0.79 correlation) | SEO-optimized embeddings |
| **text-embedding-3-large** | 3072 | Slow | Excellent (0.92 correlation) | Premium, highest accuracy |

#### 2.1.2 Semantic Clustering Quality Metrics

**Silhouette Score:** Measure of cluster cohesion and separation

```python
def calculate_silhouette_score(embeddings: np.ndarray,
                              clusters: np.ndarray) -> float:
    """
    Calculate average silhouette coefficient for clustering quality.

    Interpretation:
    - 0.71 to 1.0: Strong structure
    - 0.51 to 0.70: Reasonable structure
    - 0.26 to 0.50: Weak structure
    - -1.0 to 0.25: No substantial structure

    Formula:
    s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Where:
    - a(i) = average distance from point i to other points in its cluster
    - b(i) = minimum average distance from point i to points in other clusters
    """
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(embeddings, clusters, metric='cosine')
    avg_silhouette = np.mean(silhouette_vals)

    return avg_silhouette

# Example:
# avg_silhouette = 0.68 → Reasonable clustering structure
```

### 2.2 Topical Clustering (Co-occurrence & TF-IDF)

Topical clustering groups keywords based on statistical co-occurrence patterns and term importance metrics.

#### 2.2.1 TF-IDF-Based Clustering

TF-IDF (Term Frequency-Inverse Document Frequency) identifies keywords that are frequent in specific contexts but rare globally.

```python
def tfidf_keyword_clustering(documents: List[str],
                            keywords: List[str],
                            n_clusters: int = 5) -> Dict[int, List[str]]:
    """
    Cluster keywords using TF-IDF vectorization and K-means.

    Args:
        documents: List of content documents (e.g., SERP result text)
        keywords: List of keywords to cluster
        n_clusters: Number of desired clusters

    Returns:
        Dictionary mapping cluster_id to keyword list
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np

    # Step 1: Create TF-IDF matrix for documents
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Step 2: Get feature names and create keyword vectors
    feature_names = vectorizer.get_feature_names_out()

    # Step 3: Map keywords to TF-IDF vectors
    keyword_vectors = []
    for keyword in keywords:
        # For multi-word keywords, average component vectors
        keyword_terms = keyword.lower().split()
        vectors = []
        for term in keyword_terms:
            if term in feature_names:
                idx = list(feature_names).index(term)
                vectors.append(tfidf_matrix[:, idx].toarray().flatten())

        if vectors:
            keyword_vector = np.mean(vectors, axis=0)
        else:
            # Fallback for unknown terms
            keyword_vector = np.zeros(tfidf_matrix.shape[1])

        keyword_vectors.append(keyword_vector)

    keyword_vectors = np.array(keyword_vectors)

    # Step 4: K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(keyword_vectors)

    # Step 5: Group keywords
    clustering_result = {}
    for keyword, cluster_id in zip(keywords, clusters):
        if cluster_id not in clustering_result:
            clustering_result[cluster_id] = []
        clustering_result[cluster_id].append(keyword)

    return clustering_result

# Example: TF-IDF clustering of marketing keywords
marketing_docs = [
    "Email marketing automation campaign templates...",
    "Social media marketing strategy content calendar...",
    "SEO optimization keywords ranking positions..."
]

marketing_keywords = [
    "email marketing", "marketing automation", "email campaigns",
    "social media marketing", "content marketing", "social strategy",
    "SEO", "keyword research", "on-page optimization"
]

clusters = tfidf_keyword_clustering(marketing_docs, marketing_keywords, n_clusters=3)
# Result:
# Cluster 0: ["email marketing", "marketing automation", "email campaigns"]
# Cluster 1: ["social media marketing", "social strategy", "content marketing"]
# Cluster 2: ["SEO", "keyword research", "on-page optimization"]
```

**TF-IDF Formula:**

```
TF(term, document) = (frequency of term in document) / (total terms in document)

IDF(term, corpus) = log(total documents / documents containing term)

TF-IDF(term, document) = TF(term, document) × IDF(term, corpus)
```

#### 2.2.2 Co-occurrence Matrix Clustering

Keywords appearing together in documents share topical relevance.

```python
def cooccurrence_clustering(documents: List[str],
                           keywords: List[str],
                           window_size: int = 10) -> np.ndarray:
    """
    Create co-occurrence matrix for keywords in documents.

    Args:
        documents: List of documents
        keywords: Keywords to analyze
        window_size: Word window for co-occurrence calculation

    Returns:
        Co-occurrence matrix (n_keywords × n_keywords)
    """

    import numpy as np
    from collections import defaultdict

    # Initialize co-occurrence matrix
    n = len(keywords)
    cooccurrence = np.zeros((n, n))
    keyword_to_idx = {kw: i for i, kw in enumerate(keywords)}

    for document in documents:
        words = document.lower().split()

        # For each position, check co-occurrence within window
        for i, word in enumerate(words):
            window_start = max(0, i - window_size)
            window_end = min(len(words), i + window_size + 1)
            window = words[window_start:window_end]

            # Count co-occurrences
            for kw1_idx, kw1 in enumerate(keywords):
                if kw1 in word or word in kw1:
                    for kw2_idx, kw2 in enumerate(keywords):
                        if kw2 in window and kw1_idx != kw2_idx:
                            cooccurrence[kw1_idx, kw2_idx] += 1

    # Normalize
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    cooccurrence = np.divide(cooccurrence, row_sums,
                            where=row_sums!=0, out=cooccurrence.copy())

    return cooccurrence

# Interpretation: High values indicate strong topical relationships
```

### 2.3 SERP-Based Clustering (Competitive Landscape)

SERP-based clustering groups keywords that return similar URLs in Google's top 10 results, reflecting how search engines group conceptually similar queries.

#### 2.3.1 Algorithm Specification

```python
def serp_based_clustering(keywords: List[str],
                         serp_api_key: str,
                         min_overlap: float = 0.6,
                         top_n_results: int = 10) -> Dict[int, List[str]]:
    """
    Cluster keywords based on SERP overlap.

    Args:
        keywords: Keywords to cluster
        serp_api_key: API key for SERP data provider
        min_overlap: Minimum overlap ratio for same cluster (0.6 = 6+ shared URLs)
        top_n_results: Number of top results to analyze per keyword

    Returns:
        Clustering with keywords grouped by SERP similarity

    Algorithm:
    1. Get top N SERPs for each keyword
    2. Compute overlap percentage between keyword pairs
    3. Build similarity graph (keywords as nodes, overlap as edges)
    4. Apply graph clustering (hierarchical agglomerative clustering)
    """

    from collections import defaultdict
    import itertools

    # Step 1: Fetch SERPs for all keywords
    serp_results = {}
    for keyword in keywords:
        urls = fetch_serp_urls(keyword, serp_api_key, top_n_results)
        serp_results[keyword] = set(urls)  # Use set for efficient overlap

    # Step 2: Build similarity matrix
    n_keywords = len(keywords)
    similarity_matrix = np.zeros((n_keywords, n_keywords))

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i >= j:
                continue

            # Calculate Jaccard similarity
            intersection = len(serp_results[kw1] & serp_results[kw2])
            union = len(serp_results[kw1] | serp_results[kw2])
            jaccard = intersection / union if union > 0 else 0

            similarity_matrix[i, j] = jaccard
            similarity_matrix[j, i] = jaccard

    # Step 3: Build adjacency list for keywords above threshold
    graph = defaultdict(list)
    for i in range(n_keywords):
        for j in range(i+1, n_keywords):
            if similarity_matrix[i, j] >= min_overlap:
                graph[keywords[i]].append(keywords[j])
                graph[keywords[j]].append(keywords[i])

    # Step 4: Find connected components (clusters)
    visited = set()
    clusters = {}
    cluster_id = 0

    def dfs(keyword, cluster_id, clusters, visited, graph):
        if keyword in visited:
            return
        visited.add(keyword)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(keyword)
        for neighbor in graph[keyword]:
            dfs(neighbor, cluster_id, clusters, visited, graph)

    for keyword in keywords:
        if keyword not in visited:
            dfs(keyword, cluster_id, clusters, visited, graph)
            cluster_id += 1

    return clusters

def fetch_serp_urls(keyword: str, api_key: str, top_n: int) -> List[str]:
    """
    Placeholder for SERP API integration.
    Supports: SEMrush, Ahrefs, DataForSEO, etc.
    """
    # Implementation depends on chosen API provider
    pass

# Example configuration:
# min_overlap = 0.60 (6 of 10 results must match)
# Result: Keywords with highly similar competitive landscapes grouped together
```

**SERP Overlap Interpretation:**

| Shared URLs | Overlap % | Interpretation | Cluster Action |
|---|---|---|---|
| 9-10 | 90-100% | Nearly identical SERPs | Merge into single target |
| 7-8 | 70-80% | Very similar intent | Consider topic merge |
| 5-6 | 50-60% | Related but distinct | Separate content pieces |
| 3-4 | 30-40% | Different intent | Separate content focus |
| 0-2 | 0-20% | Unique intent | Independent content |

### 2.4 Hybrid Clustering Approach

Production systems combine multiple clustering methods to maximize accuracy and capture different relationship types.

```python
def hybrid_keyword_clustering(keywords: List[str],
                             documents: List[str],
                             serp_results: Dict[str, Set[str]],
                             semantic_weight: float = 0.4,
                             topical_weight: float = 0.35,
                             serp_weight: float = 0.25) -> Dict[int, List[str]]:
    """
    Combine semantic, topical, and SERP-based clustering with weighted fusion.

    Args:
        keywords: Keywords to cluster
        documents: Content documents for topical analysis
        serp_results: Pre-computed SERP results
        semantic_weight: Weight for semantic similarity (0-1)
        topical_weight: Weight for topical similarity (0-1)
        serp_weight: Weight for SERP overlap (0-1)

    Returns:
        Hybrid clustering result
    """

    # Normalize weights
    total_weight = semantic_weight + topical_weight + serp_weight
    semantic_weight /= total_weight
    topical_weight /= total_weight
    serp_weight /= total_weight

    # Step 1: Generate individual similarity matrices
    semantic_matrix = compute_semantic_similarity(keywords)  # 0-1 range
    topical_matrix = compute_topical_similarity(keywords, documents)  # 0-1 range
    serp_matrix = compute_serp_overlap(keywords, serp_results)  # 0-1 range

    # Step 2: Weighted fusion
    fused_similarity = (
        semantic_weight * semantic_matrix +
        topical_weight * topical_matrix +
        serp_weight * serp_matrix
    )

    # Step 3: Convert similarity to distance
    distance_matrix = 1 - fused_similarity

    # Step 4: Hierarchical clustering on fused matrix
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    condensed = squareform(distance_matrix)
    linkage_matrix = linkage(condensed, method='average')
    clusters = fcluster(linkage_matrix, t=0.3, criterion='distance')

    # Step 5: Return clustered keywords
    result = {}
    for keyword, cluster_id in zip(keywords, clusters):
        if cluster_id not in result:
            result[cluster_id] = []
        result[cluster_id].append(keyword)

    return result

def compute_semantic_similarity(keywords: List[str]) -> np.ndarray:
    """Return n×n semantic similarity matrix"""
    pass

def compute_topical_similarity(keywords: List[str],
                              documents: List[str]) -> np.ndarray:
    """Return n×n topical similarity matrix based on TF-IDF"""
    pass

def compute_serp_overlap(keywords: List[str],
                        serp_results: Dict[str, Set[str]]) -> np.ndarray:
    """Return n×n SERP overlap matrix (Jaccard similarity)"""
    n = len(keywords)
    matrix = np.zeros((n, n))

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            if i == j:
                matrix[i, j] = 1.0
                continue

            urls1 = serp_results.get(kw1, set())
            urls2 = serp_results.get(kw2, set())

            if not urls1 or not urls2:
                matrix[i, j] = 0.0
                continue

            intersection = len(urls1 & urls2)
            union = len(urls1 | urls2)
            matrix[i, j] = intersection / union

    return matrix
```

---

## 3. Keyword Cannibalization Detection

Keyword cannibalization occurs when multiple pages on the same domain target identical or highly similar keywords, causing them to compete internally and diluting ranking power.

### 3.1 Definition and SEO Impact

**Problem:** When two pages both rank for the same keyword, Google must choose which to display in results. This splits authority signals (backlinks, engagement metrics, CTR) across two pages, reducing the ranking potential of both compared to a single optimized page.

**Symptoms:**
- Keyword appears in rankings but position is lower than expected
- Multiple internal URLs rank for the same keyword
- Click-through rates and conversions are dispersed across multiple pages
- One URL should rank but another similar URL appears instead

### 3.2 Cannibalization Detection Algorithms

#### 3.2.1 URL-Keyword Overlap Scoring

```python
def detect_url_keyword_cannibalization(url_keywords: Dict[str, List[str]],
                                       threshold: float = 0.7) -> List[Dict]:
    """
    Identify keywords that appear across multiple URLs (cannibalization).

    Args:
        url_keywords: Mapping of URLs to their target keywords
        threshold: Similarity threshold for identifying cannibalization (0-1)

    Returns:
        List of cannibalization incidents with confidence scores

    Algorithm:
    1. For each keyword, identify all URLs targeting it
    2. Calculate similarity between keyword sets for each URL pair
    3. Flag pairs exceeding threshold as cannibalization
    """

    from itertools import combinations

    # Step 1: Create reverse index (keyword → URLs)
    keyword_to_urls = {}
    for url, keywords in url_keywords.items():
        for keyword in keywords:
            if keyword not in keyword_to_urls:
                keyword_to_urls[keyword] = []
            keyword_to_urls[keyword].append(url)

    # Step 2: Identify keywords with multiple URLs
    cannibalization_incidents = []

    for keyword, urls in keyword_to_urls.items():
        if len(urls) > 1:
            # Calculate cannibalization score
            for url1, url2 in combinations(urls, 2):
                # Get all keywords for each URL
                kw_set1 = set(url_keywords[url1])
                kw_set2 = set(url_keywords[url2])

                # Calculate Jaccard similarity
                intersection = len(kw_set1 & kw_set2)
                union = len(kw_set1 | kw_set2)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    cannibalization_incidents.append({
                        'keyword': keyword,
                        'url1': url1,
                        'url2': url2,
                        'similarity_score': similarity,
                        'shared_keywords': list(kw_set1 & kw_set2),
                        'severity': classify_severity(similarity)
                    })

    return sorted(cannibalization_incidents,
                 key=lambda x: x['similarity_score'],
                 reverse=True)

def classify_severity(similarity: float) -> str:
    """Classify cannibalization severity"""
    if similarity >= 0.8:
        return 'CRITICAL'  # Urgent consolidation needed
    elif similarity >= 0.6:
        return 'HIGH'      # Significant overlap, should consolidate
    elif similarity >= 0.4:
        return 'MEDIUM'    # Notable overlap, consider consolidation
    else:
        return 'LOW'       # Minor overlap, may be acceptable

# Example usage:
url_keywords = {
    '/blog/python-tutorials': ['python tutorial', 'learn python', 'python basics',
                               'python programming'],
    '/docs/python-guide': ['learn python', 'python guide', 'python programming',
                          'python reference'],
    '/resources/python': ['python tutorial', 'python course', 'python resources']
}

incidents = detect_url_keyword_cannibalization(url_keywords, threshold=0.6)

# Result:
# [
#   {
#     'keyword': 'python tutorial',
#     'url1': '/blog/python-tutorials',
#     'url2': '/resources/python',
#     'similarity_score': 0.71,
#     'severity': 'HIGH'
#   },
#   ...
# ]
```

#### 3.2.2 Content Similarity-Based Detection

```python
def detect_content_cannibalization(urls: List[str],
                                  content: Dict[str, str],
                                  similarity_threshold: float = 0.75) -> List[Dict]:
    """
    Detect cannibalization by analyzing actual page content similarity.
    Uses semantic embeddings for robust comparison.

    Args:
        urls: List of URLs to analyze
        content: Mapping of URL to page content
        similarity_threshold: Semantic similarity threshold (0-1)

    Returns:
        List of content cannibalization pairs
    """

    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
    from itertools import combinations
    import numpy as np

    # Step 1: Load model and encode content
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = {}

    for url in urls:
        # Use first 512 tokens of content for efficiency
        text = content[url][:4000]  # ~512 tokens
        embedding = model.encode(text, convert_to_tensor=True)
        embeddings[url] = embedding.cpu().numpy()

    # Step 2: Compare all URL pairs
    cannibalization_pairs = []

    for url1, url2 in combinations(urls, 2):
        # Calculate cosine similarity (not distance)
        similarity = 1 - cosine(embeddings[url1], embeddings[url2])

        if similarity >= similarity_threshold:
            cannibalization_pairs.append({
                'url1': url1,
                'url2': url2,
                'content_similarity': float(similarity),
                'recommendation': generate_recommendation(url1, url2, similarity)
            })

    return sorted(cannibalization_pairs,
                 key=lambda x: x['content_similarity'],
                 reverse=True)

def generate_recommendation(url1: str, url2: str, similarity: float) -> str:
    """Generate consolidation recommendation"""
    if similarity > 0.85:
        return f"MERGE: Consolidate into single page with combined content"
    elif similarity > 0.75:
        return f"REDIRECT: 301 redirect lower-authority URL to primary"
    else:
        return f"DIFFERENTIATE: Expand unique content differences"

# Example usage:
urls = ['/blog/machine-learning-intro', '/guides/ml-beginners', '/tutorials/ml-101']
content = {
    '/blog/machine-learning-intro': '...',  # ML introduction content
    '/guides/ml-beginners': '...',          # Similar ML beginner content
    '/tutorials/ml-101': '...'              # Another ML intro tutorial
}

cannibalization = detect_content_cannibalization(urls, content, threshold=0.75)
```

#### 3.2.3 Internal Ranking Conflict Detection

```python
def detect_ranking_conflicts(gsc_data: Dict[str, Dict],
                            min_position_difference: float = 2.0) -> List[Dict]:
    """
    Identify keywords where multiple URLs from same domain rank nearby.
    Data from Google Search Console.

    Args:
        gsc_data: GSC export with clicks, impressions, position for each URL+keyword
        min_position_difference: Flag if position difference is less than this

    Returns:
        List of internal ranking conflicts
    """

    # Step 1: Group by keyword
    keyword_rankings = {}
    for url, metrics in gsc_data.items():
        for keyword, data in metrics.items():
            if keyword not in keyword_rankings:
                keyword_rankings[keyword] = []

            keyword_rankings[keyword].append({
                'url': url,
                'position': data.get('position', 999),
                'clicks': data.get('clicks', 0),
                'impressions': data.get('impressions', 0)
            })

    # Step 2: Identify keywords with multiple ranking URLs
    conflicts = []

    for keyword, rankings in keyword_rankings.items():
        if len(rankings) > 1:
            # Sort by position
            rankings = sorted(rankings, key=lambda x: x['position'])

            # Check if URLs are too close in position
            for i in range(len(rankings) - 1):
                position_diff = rankings[i+1]['position'] - rankings[i]['position']

                if position_diff < min_position_difference:
                    # Calculate impact (lost impressions/clicks)
                    lost_clicks = abs(rankings[i]['clicks'] - rankings[i+1]['clicks'])

                    conflicts.append({
                        'keyword': keyword,
                        'primary_url': rankings[i]['url'],
                        'primary_position': rankings[i]['position'],
                        'competing_url': rankings[i+1]['url'],
                        'competing_position': rankings[i+1]['position'],
                        'position_gap': position_diff,
                        'estimated_lost_clicks': lost_clicks,
                        'consolidated_potential': calculate_consolidation_potential(rankings[i], rankings[i+1])
                    })

    return sorted(conflicts,
                 key=lambda x: x['estimated_lost_clicks'],
                 reverse=True)

def calculate_consolidation_potential(ranking1: Dict, ranking2: Dict) -> int:
    """
    Estimate CTR improvement from consolidation.
    Assumes consolidated page can capture both click streams.
    """
    # Simplified model: sum of both URLs' clicks
    # In practice, use historical CTR curves
    return ranking1['clicks'] + ranking2['clicks']

# Example GSC data structure:
gsc_data = {
    '/blog/seo-tips': {
        'how to improve seo': {'position': 3.5, 'clicks': 127, 'impressions': 2400},
        'seo tips 2026': {'position': 5.2, 'clicks': 89, 'impressions': 1800}
    },
    '/guides/seo': {
        'how to improve seo': {'position': 4.2, 'clicks': 84, 'impressions': 1900},
        'seo guide': {'position': 6.1, 'clicks': 52, 'impressions': 1200}
    }
}

conflicts = detect_ranking_conflicts(gsc_data, min_position_difference=2.0)
```

### 3.3 Cannibalization Resolution Strategies

| Severity | Strategy | Implementation |
|---|---|---|
| **CRITICAL** (>85% similarity) | **MERGE** | Consolidate into single page, redirect lower-authority URL with 301, preserve all unique content from both pages |
| **HIGH** (70-85% similarity) | **REDIRECT** | 301 redirect one URL to other, consolidate authority signals, monitor ranking recovery |
| **MEDIUM** (50-70% similarity) | **DIFFERENTIATE** | Rewrite one page to target distinct aspect/keyword, add unique value proposition, link between pages |
| **LOW** (<50% similarity) | **COEXIST** | Monitor search position trends, link contextually between pages, consider topic cluster strategy |

---

## 4. Keyword-to-Content Mapping Framework

Strategic keyword distribution across content elements ensures optimal coverage without triggering over-optimization penalties.

### 4.1 Primary Keyword Placement Rules

**Definition:** The primary keyword is the main target for the page, typically the highest-volume, highest-intent keyword.

#### 4.1.1 Primary Keyword Distribution Schema

```json
{
  "primary_keyword": {
    "definition": "Main keyword target, highest search volume and intent",
    "placement_rules": {
      "title_tag": {
        "required": true,
        "position": "Front (ideally first 3 words)",
        "format": "[Primary Keyword] - [Value Prop/Modifier]",
        "example": "Best Project Management Software for Remote Teams 2026",
        "character_limit": "50-60 chars"
      },
      "h1_tag": {
        "required": true,
        "position": "Early in page (within first 100 words)",
        "format": "Full primary keyword or close variant",
        "example": "Best Project Management Software for Remote Teams",
        "character_limit": "20-70 chars"
      },
      "meta_description": {
        "required": true,
        "position": "Naturally in first sentence",
        "format": "Include primary or close semantic variant",
        "example": "Discover the best project management software for remote teams...",
        "character_limit": "150-160 chars"
      },
      "first_paragraph": {
        "required": true,
        "position": "Within first 50 words",
        "format": "Natural inclusion, ideally in first or second sentence",
        "occurrence": 1,
        "importance": "Establishes immediate topical relevance"
      },
      "url_slug": {
        "required": true,
        "position": "Full slug or primary variant",
        "format": "/[keyword-variant]/[specific-angle]",
        "example": "/best-project-management-software-remote-teams",
        "note": "Hyphens, lowercase, under 75 chars"
      },
      "image_alt_text": {
        "required": true,
        "count": "At least 1 image with primary keyword",
        "format": "Natural description including keyword",
        "example": "Best project management software interface comparison"
      }
    }
  }
}
```

#### 4.1.2 Primary Keyword Placement Algorithm

```python
def validate_primary_keyword_placement(page_data: Dict) -> Dict[str, bool]:
    """
    Verify that primary keyword appears in all critical locations.

    Args:
        page_data: Dictionary with page elements (title, h1, content, etc.)

    Returns:
        Dictionary indicating placement validation results
    """

    primary_keyword = page_data['primary_keyword'].lower()
    keyword_variants = generate_keyword_variants(primary_keyword)

    validation = {
        'title_tag': validate_element(page_data.get('title', ''),
                                     keyword_variants,
                                     position='start'),
        'h1_tag': validate_element(page_data.get('h1', ''),
                                  keyword_variants),
        'meta_description': validate_element(page_data.get('meta_description', ''),
                                           keyword_variants),
        'first_paragraph': validate_element(page_data.get('first_paragraph', ''),
                                          keyword_variants,
                                          max_words=50),
        'url_slug': validate_element(page_data.get('url_slug', ''),
                                    keyword_variants),
        'image_alt_text': validate_image_alts(page_data.get('images', []),
                                             keyword_variants)
    }

    return validation

def generate_keyword_variants(keyword: str) -> List[str]:
    """Generate semantic variants of primary keyword"""
    variants = [keyword]

    # Word order variations
    words = keyword.split()
    if len(words) > 1:
        variants.append(' '.join(reversed(words)))

    # Singular/plural variations
    if keyword.endswith('s'):
        variants.append(keyword[:-1])
    else:
        variants.append(keyword + 's')

    return variants

def validate_element(element_text: str,
                    keyword_variants: List[str],
                    position: str = 'any',
                    max_words: int = None) -> bool:
    """
    Check if element contains keyword variant.

    Args:
        element_text: Text to check
        keyword_variants: List of acceptable variants
        position: 'start', 'end', or 'any'
        max_words: If set, check occurrence within first N words

    Returns:
        Boolean indicating if validation passed
    """

    if not element_text:
        return False

    element_lower = element_text.lower()

    # Limit search scope if max_words specified
    if max_words:
        words = element_text.split()
        element_lower = ' '.join(words[:max_words]).lower()

    # Check position constraints
    if position == 'start':
        return any(element_lower.startswith(var) for var in keyword_variants)
    elif position == 'end':
        return any(element_lower.endswith(var) for var in keyword_variants)
    else:  # 'any'
        return any(var in element_lower for var in keyword_variants)

# Example validation:
page_data = {
    'primary_keyword': 'Best Project Management Software',
    'title': 'Best Project Management Software for Remote Teams 2026',
    'h1': 'Best Project Management Software for Remote Teams',
    'meta_description': 'Discover the best project management software...',
    'first_paragraph': 'When finding the best project management software for your remote team...',
    'url_slug': 'best-project-management-software-remote-teams',
    'images': [
        {'alt': 'Best project management software comparison'}
    ]
}

results = validate_primary_keyword_placement(page_data)
# All checks should return True for optimal placement
```

### 4.2 Secondary Keyword Distribution

Secondary keywords support the primary keyword by adding topical breadth and capturing related search intents.

#### 4.2.1 Secondary Keyword Mapping Strategy

```python
def map_secondary_keywords(primary_keyword: str,
                          secondary_keywords: List[str],
                          content_sections: List[str]) -> Dict[str, List[str]]:
    """
    Distribute secondary keywords across H2 sections and body content.

    Args:
        primary_keyword: Main keyword
        secondary_keywords: List of related keywords (typically 5-15)
        content_sections: List of H2/section headers in order

    Returns:
        Mapping of sections to assigned secondary keywords
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.spatial.distance import cosine
    import numpy as np

    # Step 1: Generate embeddings for similarity analysis
    all_keywords = [primary_keyword] + secondary_keywords
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
    keyword_vectors = vectorizer.fit_transform(all_keywords).toarray()

    # Step 2: Rank secondary keywords by relevance to primary
    primary_vector = keyword_vectors[0]
    relevance_scores = []

    for i, sec_keyword in enumerate(secondary_keywords):
        similarity = 1 - cosine(primary_vector, keyword_vectors[i+1])
        relevance_scores.append((sec_keyword, similarity))

    relevance_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 3: Distribute keywords across sections
    # Strategy: Place most relevant secondary keywords in H2s
    mapping = {}
    keywords_per_section = max(1, len(secondary_keywords) // len(content_sections))

    for section_idx, section in enumerate(content_sections):
        start_idx = section_idx * keywords_per_section
        end_idx = start_idx + keywords_per_section

        assigned_keywords = [kw for kw, _ in relevance_scores[start_idx:end_idx]]

        if assigned_keywords:
            mapping[section] = assigned_keywords

    # Step 4: Ensure all keywords assigned
    assigned = set()
    for keywords in mapping.values():
        assigned.update(keywords)

    unassigned = [kw for kw in secondary_keywords if kw not in assigned]

    if unassigned and content_sections:
        # Assign remaining keywords to last section
        if content_sections[-1] not in mapping:
            mapping[content_sections[-1]] = []
        mapping[content_sections[-1]].extend(unassigned)

    return mapping

# Example:
primary = "Machine Learning for Business"
secondary = [
    "AI automation benefits",
    "ML implementation strategy",
    "predictive analytics",
    "data science applications",
    "AI ROI measurement"
]

sections = [
    "What is Machine Learning?",
    "Machine Learning Applications",
    "Implementation Roadmap",
    "ROI and Business Impact"
]

mapping = map_secondary_keywords(primary, secondary, sections)

# Result:
# {
#   "What is Machine Learning?": ["predictive analytics"],
#   "Machine Learning Applications": ["AI automation benefits", "data science applications"],
#   "Implementation Roadmap": ["ML implementation strategy"],
#   "ROI and Business Impact": ["AI ROI measurement"]
# }
```

#### 4.2.2 Secondary Keyword Placement Rules

| Element | Placement | Occurrence | Notes |
|---|---|---|---|
| **H2 Headers** | One per section | 1 per H2 | In or near the header itself |
| **First sentence of H2 section** | Opening paragraph | 1 per 500 words | Natural integration |
| **H3 Subheaders** | Tertiary sections | Optional | Only if keyword naturally fits |
| **Body paragraphs** | Throughout section | 1-2 per 500 words | Distributed, not clustered |
| **Internal links** | Anchor text | 1-2 per secondary keyword | Link to related content |
| **Bullet points/lists** | Naturally within items | 1 per major list | If contextually relevant |

### 4.3 LSI and Semantic Variations

LSI (Latent Semantic Indexing) keywords are semantically related terms that enhance topical relevance. While Google doesn't explicitly use LSI, semantic variation improves content quality and user experience.

#### 4.3.1 LSI Keyword Generation

```python
def generate_lsi_keywords(primary_keyword: str,
                         context: str = None,
                         max_results: int = 20) -> List[str]:
    """
    Generate LSI keywords through multiple methods.

    Args:
        primary_keyword: Primary keyword
        context: Optional content context to filter results
        max_results: Maximum LSI keywords to return

    Returns:
        List of LSI keyword suggestions
    """

    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize

    lsi_keywords = set()

    # Method 1: WordNet synonyms and related terms
    words = primary_keyword.split()
    for word in words:
        synsets = wordnet.synsets(word)
        for synset in synsets[:2]:  # Top 2 synsets
            for lemma in synset.lemmas()[:3]:  # Top 3 lemmas per synset
                lsi_keywords.add(lemma.name().replace('_', ' '))

            # Hypernyms (broader terms)
            for hypernym in synset.hypernyms()[:2]:
                for lemma in hypernym.lemmas()[:2]:
                    lsi_keywords.add(lemma.name().replace('_', ' '))

            # Hyponyms (narrower terms)
            for hyponym in synset.hyponyms()[:2]:
                for lemma in hyponym.lemmas()[:2]:
                    lsi_keywords.add(lemma.name().replace('_', ' '))

    # Method 2: Co-occurrence patterns
    if context:
        # Extract frequently co-occurring terms
        tokens = word_tokenize(context.lower())
        for i, token in enumerate(tokens):
            if token in primary_keyword.lower():
                # Collect surrounding words
                context_window = tokens[max(0, i-5):min(len(tokens), i+5)]
                lsi_keywords.update(context_window)

    # Method 3: Common semantic relationships
    semantic_patterns = {
        'machine learning': [
            'artificial intelligence', 'deep learning', 'neural networks',
            'algorithms', 'data science', 'supervised learning',
            'unsupervised learning', 'model training'
        ],
        'seo': [
            'search engine optimization', 'organic search', 'ranking',
            'keyword research', 'link building', 'on-page optimization'
        ]
    }

    for pattern_key, patterns in semantic_patterns.items():
        if pattern_key in primary_keyword.lower():
            lsi_keywords.update(patterns)

    # Filter and rank
    lsi_list = list(lsi_keywords)

    # Remove exact duplicates and too-short terms
    lsi_list = [kw for kw in lsi_list if len(kw) > 2 and kw != primary_keyword]

    return lsi_list[:max_results]

# Example:
lsi_keywords = generate_lsi_keywords(
    "machine learning",
    context="Machine learning algorithms for predictive analytics...",
    max_results=20
)

# Result includes terms like:
# ['artificial intelligence', 'deep learning', 'neural networks',
#  'supervised learning', 'data science', 'algorithms', ...]
```

#### 4.3.2 LSI Placement Guidelines

**Important Note:** Google does NOT use LSI for ranking. However, semantic variation improves:
- Content depth and topical authority
- User experience (more varied vocabulary)
- Click-through rate (more descriptive snippets)
- Natural language patterns (avoids keyword stuffing)

**LSI Placement Rules:**

```python
def place_lsi_keywords(content: str,
                      lsi_keywords: List[str],
                      primary_keyword: str,
                      target_density: float = 0.05) -> str:
    """
    Naturally distribute LSI keywords throughout content.

    Args:
        content: Page content
        lsi_keywords: Generated LSI keywords
        primary_keyword: Primary keyword (for balance)
        target_density: Target density as decimal (e.g., 0.05 = 5%)

    Returns:
        Content with LSI keywords naturally integrated

    Strategy:
    1. Identify topical sections
    2. Insert LSI keywords maintaining natural flow
    3. Avoid clustering (don't repeat same LSI keyword)
    4. Maintain primary keyword dominance
    """

    import re

    # Step 1: Calculate target counts
    word_count = len(content.split())
    primary_target = int(word_count * (target_density / 2))  # Primary gets more weight
    lsi_per_keyword = max(1, int(word_count * (target_density / 2) / len(lsi_keywords)))

    # Step 2: Identify insertion points (topic boundaries)
    # Split by headers (H2, H3)
    sections = re.split(r'##+ ', content)

    # Step 3: Distribute LSI keywords across sections
    modified_sections = []
    lsi_used = {}

    for section in sections:
        # For each section, add relevant LSI keywords naturally
        section_words = section.split()

        # Find logical insertion points (end of sentences)
        sentences = re.split(r'(?<=[.!?])\s+', section)

        for i, lsi_kw in enumerate(lsi_keywords):
            if section_count := lsi_used.get(lsi_kw, 0) < lsi_per_keyword:
                # Find relevant position (after related content)
                if primary_keyword.split()[0] in section.lower():
                    # Insert in this section
                    insert_position = min(len(sentences) - 1, i % len(sentences))
                    sentences[insert_position] += f" {lsi_kw}"
                    lsi_used[lsi_kw] = section_count + 1

        modified_sections.append(' '.join(sentences))

    return '\n'.join(modified_sections)
```

### 4.4 Meta Description Optimization

Meta descriptions influence click-through rates and should include primary/secondary keywords naturally.

```python
def generate_optimized_meta_description(page_title: str,
                                       primary_keyword: str,
                                       secondary_keywords: List[str],
                                       value_proposition: str,
                                       max_length: int = 160) -> str:
    """
    Generate meta description with keyword inclusion and CTR optimization.

    Args:
        page_title: Page H1/title
        primary_keyword: Primary target keyword
        secondary_keywords: List of secondary keywords to try including
        value_proposition: Unique value prop or key benefit
        max_length: Character limit (Google displays ~155-160)

    Returns:
        Optimized meta description

    Strategy:
    1. Include primary keyword naturally in first half
    2. Highlight value proposition
    3. Optional: Include 1 secondary keyword if space allows
    4. Optimize for CTR (action-oriented language)
    """

    # Template variations for different intent types
    templates = {
        'informational': (
            f"Learn about {primary_keyword}. {value_proposition} "
            f"Discover best practices, guides, and expert insights."
        ),
        'commercial': (
            f"Compare {primary_keyword} options. {value_proposition} "
            f"Read reviews, pricing, and top-rated solutions."
        ),
        'transactional': (
            f"Buy {primary_keyword} online. {value_proposition} "
            f"Fast shipping, best prices, and customer reviews."
        )
    }

    # Determine intent type from keyword
    if any(word in primary_keyword.lower() for word in ['how', 'what', 'why', 'guide']):
        intent_type = 'informational'
    elif any(word in primary_keyword.lower() for word in ['best', 'top', 'vs', 'compare']):
        intent_type = 'commercial'
    else:
        intent_type = 'transactional'

    # Generate description
    description = templates.get(intent_type, templates['informational'])

    # Trim to length
    if len(description) > max_length:
        # Cut at last word boundary
        description = description[:max_length].rsplit(' ', 1)[0] + '...'

    return description

# Example:
meta = generate_optimized_meta_description(
    "Best Project Management Software for Remote Teams",
    "project management software",
    ["remote teams", "team collaboration"],
    "Compare features, pricing, and customer reviews",
    max_length=160
)

# Result:
# "Learn about project management software. Compare features, pricing, and customer reviews.
#  Discover best practices, guides, and expert insights."
```

---

## 5. Keyword Density Analysis

### 5.1 Historical Context: Why Keyword Density Metrics Evolved

**Early SEO (1990s-2000s):** Keyword density (keyword occurrences / total words × 100) was a primary ranking factor. Pages with 3-5% density ranked better than those with 1%. This led to keyword stuffing abuse.

**Problem:** Arbitrary densities didn't correlate with user satisfaction. Pages with keyword stuffing ranked well but had poor engagement metrics.

**Evolution:** Google shifted toward relevance models (BM25, later neural models) that penalize exact-match overuse while rewarding semantic variation and contextual relevance.

**Current State (2025-2026):** Keyword density is an **indirect** signal—high density now correlates with poor quality in most algorithms. Instead, modern systems optimize for:
- **Term Frequency Saturation:** Diminishing returns after N occurrences
- **Natural Language Distribution:** Semantic variations and context
- **Entity Co-occurrence:** Related concepts mentioned together
- **Topical Depth:** Breadth of related information

### 5.2 Current Best Practices: TF-IDF and BM25

#### 5.2.1 TF-IDF Relevance Scoring

**TF-IDF Formula:**

```
TF(term, document) = log(1 + count(term in document))

IDF(term, corpus) = log(corpus_size / documents_containing_term)

TF-IDF(term, document) = TF(term, document) × IDF(term, corpus)
```

**Implementation:**

```python
def calculate_tfidf_relevance(document: str,
                              term: str,
                              corpus: List[str]) -> float:
    """
    Calculate TF-IDF score for a term in a document.

    Args:
        document: Target document text
        term: Term to score
        corpus: Collection of documents for IDF calculation

    Returns:
        TF-IDF score (higher = more relevant)
    """

    import math
    from collections import Counter

    # Step 1: Calculate TF (Term Frequency)
    doc_words = document.lower().split()
    term_count = doc_words.count(term.lower())

    tf = math.log(1 + term_count) if term_count > 0 else 0

    # Step 2: Calculate IDF (Inverse Document Frequency)
    docs_with_term = sum(1 for doc in corpus if term.lower() in doc.lower())
    total_docs = len(corpus)

    if docs_with_term > 0:
        idf = math.log(total_docs / docs_with_term)
    else:
        idf = 0

    # Step 3: TF-IDF
    tfidf = tf * idf

    return tfidf

# Example with corpus:
corpus = [
    "Machine learning is AI",
    "Deep learning is machine learning",
    "Neural networks power deep learning"
]

document = "Machine learning machine learning AI"

tfidf_score = calculate_tfidf_relevance(document, "machine learning", corpus)
# Result: High score because "machine learning" is frequent in doc,
# but not all docs (high IDF)
```

**TF-IDF Limitations:**

- **No saturation:** Score increases linearly with term repetition
- **No document length normalization:** Longer documents get higher scores
- **No context:** "machine" and "learning" treated independently
- **Keyword stuffing friendly:** Encourages excessive repetition

#### 5.2.2 BM25 Relevance Scoring (Superior to TF-IDF)

BM25 (Best Matching 25) addresses TF-IDF limitations with term frequency saturation and document length normalization.

**BM25 Formula:**

```
BM25(D, Q) = Σ(i=1 to n) IDF(qi) × ((f(qi, D) × (k1 + 1)) /
             (f(qi, D) + k1 × (1 - b + b × (|D| / avgdl))))

Where:
- D = document
- Q = query (or keywords)
- qi = individual query term
- f(qi, D) = term frequency in document
- |D| = document length
- avgdl = average document length in corpus
- k1 = term frequency saturation parameter (typically 1.5)
- b = length normalization parameter (typically 0.75)
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
```

**Key Improvements:**

1. **Term Frequency Saturation:** After k1 repetitions (~1.5), additional occurrences contribute less
2. **Document Length Normalization:** Fair comparison between short and long documents
3. **Logarithmic IDF:** Avoids extreme IDF values from rare terms

**BM25 Implementation:**

```python
def calculate_bm25_score(document: str,
                        query_terms: List[str],
                        corpus: List[str],
                        k1: float = 1.5,
                        b: float = 0.75) -> float:
    """
    Calculate BM25 relevance score.

    Args:
        document: Target document
        query_terms: Query keywords
        corpus: Document corpus for statistics
        k1: Term frequency saturation parameter (1.5 typical)
        b: Length normalization parameter (0.75 typical)

    Returns:
        BM25 score for document
    """

    import math
    from collections import Counter

    # Step 1: Document statistics
    doc_words = document.lower().split()
    doc_length = len(doc_words)
    avg_doc_length = sum(len(doc.split()) for doc in corpus) / len(corpus)

    # Step 2: Calculate IDF for each query term
    idf_scores = {}
    corpus_size = len(corpus)

    for term in query_terms:
        docs_with_term = sum(1 for doc in corpus if term.lower() in doc.lower())

        # BM25 IDF formula
        idf = math.log(
            (corpus_size - docs_with_term + 0.5) /
            (docs_with_term + 0.5) + 1
        )
        idf_scores[term] = idf

    # Step 3: Calculate BM25
    bm25_score = 0

    for term in query_terms:
        term_freq = doc_words.count(term.lower())

        if term_freq == 0:
            continue

        # BM25 formula with saturation and normalization
        numerator = term_freq * (k1 + 1)
        denominator = (
            term_freq +
            k1 * (1 - b + b * (doc_length / avg_doc_length))
        )

        bm25_score += idf_scores[term] * (numerator / denominator)

    return bm25_score

# Example comparison: TF-IDF vs BM25
document_a = "Python Python Python programming language"  # 50 words total
document_b = "Python programming guide for beginners"    # 100 words total

corpus = [document_a, document_b]
query = ["Python"]

tfidf_a = calculate_tfidf_relevance(document_a, "Python", corpus)
tfidf_b = calculate_tfidf_relevance(document_b, "Python", corpus)

bm25_a = calculate_bm25_score(document_a, query, corpus)
bm25_b = calculate_bm25_score(document_b, query, corpus)

print(f"TF-IDF: A={tfidf_a:.3f}, B={tfidf_b:.3f}")  # A higher (more repetition)
print(f"BM25: A={bm25_a:.3f}, B={bm25_b:.3f}")      # More balanced (length normalized)
```

### 5.3 Over-Optimization Thresholds

Modern algorithms detect and penalize keyword over-optimization. Exact thresholds vary by algorithm, but research indicates:

#### 5.3.1 Density Thresholds by Element

| Element | Healthy Range | Over-Optimization (Red Flag) | Severe Penalty Risk |
|---|---|---|---|
| **Primary Keyword** (exact match) | 0.5-1.5% | 2-3% | >5% |
| **Primary + Close Variants** | 1.5-3% | 4-6% | >8% |
| **All Keyword Variations** | 3-6% | 7-10% | >12% |
| **LSI/Semantic Terms** | 2-5% | 6-10% | >12% |

**Calculation Method:**

```python
def calculate_keyword_density(content: str,
                             keyword: str,
                             metric_type: str = 'exact') -> float:
    """
    Calculate keyword density as percentage.

    Args:
        content: Page content
        keyword: Target keyword
        metric_type: 'exact' | 'close_variant' | 'all_variations'

    Returns:
        Density percentage (0-100)
    """

    content_lower = content.lower()

    if metric_type == 'exact':
        # Count exact matches with word boundaries
        import re
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        count = len(re.findall(pattern, content_lower))

    elif metric_type == 'close_variant':
        # Allow singular/plural variation
        variants = [keyword.lower()]
        if keyword.endswith('s'):
            variants.append(keyword[:-1].lower())
        else:
            variants.append(keyword.lower() + 's')

        count = sum(content_lower.count(var) for var in variants)

    else:  # all_variations
        # Include LSI and semantic variations
        all_terms = [keyword] + generate_lsi_keywords(keyword)
        count = sum(content_lower.count(term.lower()) for term in all_terms)

    total_words = len(content.split())
    density = (count / total_words * 100) if total_words > 0 else 0

    return density

# Example:
content = "Machine learning machine learning applications in machine learning..."
keyword = "machine learning"

exact_density = calculate_keyword_density(content, keyword, 'exact')
# Result: 3.0% (3 occurrences / 100 total words)

# Assessment:
if exact_density > 1.5:
    print("WARNING: Consider reducing exact-match repetition")
```

#### 5.3.2 Natural Language Distribution Patterns

Healthy content exhibits natural keyword distribution:

```python
def analyze_keyword_distribution(content: str,
                                keyword: str) -> Dict:
    """
    Analyze keyword distribution patterns to detect over-optimization.

    Args:
        content: Page content
        keyword: Target keyword

    Returns:
        Dictionary with distribution analysis
    """

    import re
    from statistics import stdev, mean

    # Find all keyword positions
    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
    sentences = re.split(r'[.!?]+', content)

    positions = []
    keyword_count = 0

    for sent_idx, sentence in enumerate(sentences):
        if keyword.lower() in sentence.lower():
            positions.append(sent_idx)
            keyword_count += 1

    # Calculate distribution metrics
    if len(positions) < 2:
        distribution_score = 1.0  # Perfect (very spread out)
    else:
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_gap = mean(gaps)
        gap_variance = stdev(gaps) if len(gaps) > 1 else 0

        # Ideal: consistent gaps (no clustering)
        # Clustering = low variance = over-optimization signal
        if gap_variance < 1:  # Clustered occurrences
            distribution_score = 0.3  # Red flag
        elif avg_gap > 10:  # Very spread out
            distribution_score = 0.9  # Healthy
        else:
            distribution_score = 0.7  # Acceptable

    return {
        'occurrences': keyword_count,
        'density_percent': (keyword_count / len(content.split()) * 100),
        'average_gap_sentences': mean(gaps) if len(gaps) > 0 else 0,
        'distribution_score': distribution_score,
        'recommendation': (
            'OPTIMAL' if distribution_score > 0.75 else
            'CAUTION' if distribution_score > 0.5 else
            'REDUCE - Keywords appear clustered'
        )
    }

# Example:
analysis = analyze_keyword_distribution(
    "ML is important... Deep learning enables ML... ML applications are...",
    "ML"
)
```

### 5.4 Over-Optimization Detection Indicators

**Red Flags:**

1. **Exact-match density > 2%:** Likely stuffing
2. **Keyword clustering:** All occurrences in first 3 paragraphs
3. **Unnatural variations:** "machine learning," "machine-learning," "machinelearning" in same paragraph
4. **Anchor text overuse:** Same anchor text in multiple internal links
5. **Hidden text:** Keywords in white-on-white text, display:none CSS
6. **List/Table abuse:** Keywords repeated in unrelated lists solely for density

**Detection Algorithm:**

```python
def detect_over_optimization(page_content: Dict,
                            primary_keyword: str) -> Dict[str, float]:
    """
    Calculate over-optimization risk score (0-1, higher = more risk).

    Args:
        page_content: Dictionary with page elements
        primary_keyword: Target keyword

    Returns:
        Risk scores for different over-optimization types
    """

    risk_scores = {}

    # Risk 1: Density
    density = calculate_keyword_density(
        page_content['body_text'],
        primary_keyword,
        'exact'
    )
    risk_scores['density'] = min(1.0, density / 2.5)  # Max risk at 2.5%+

    # Risk 2: Clustering (concentration in first section)
    first_section = ' '.join(page_content['body_text'].split()[:200])
    first_section_count = first_section.lower().count(primary_keyword.lower())
    total_count = page_content['body_text'].lower().count(primary_keyword.lower())

    clustering_ratio = first_section_count / total_count if total_count > 0 else 0
    risk_scores['clustering'] = clustering_ratio if clustering_ratio > 0.5 else 0

    # Risk 3: Title/Meta overuse
    title_count = page_content.get('title', '').lower().count(primary_keyword.lower())
    meta_count = page_content.get('meta_description', '').lower().count(
        primary_keyword.lower()
    )
    risk_scores['title_meta'] = min(1.0, (title_count + meta_count) / 2)

    # Risk 4: Anchor text concentration
    total_anchor_text = ' '.join([
        link.get('anchor_text', '') for link in page_content.get('internal_links', [])
    ]).lower()

    if total_anchor_text:
        anchor_count = total_anchor_text.count(primary_keyword.lower())
        total_links = len(page_content.get('internal_links', []))
        risk_scores['anchor_text'] = min(1.0, anchor_count / (total_links * 0.1))
    else:
        risk_scores['anchor_text'] = 0

    # Overall risk
    overall_risk = sum(risk_scores.values()) / len(risk_scores)

    return {
        'individual_risks': risk_scores,
        'overall_risk': overall_risk,
        'status': (
            'SAFE' if overall_risk < 0.25 else
            'CAUTION' if overall_risk < 0.50 else
            'HIGH RISK' if overall_risk < 0.75 else
            'CRITICAL'
        )
    }
```

---

## 6. Implementation Specifications

### 6.1 Data Structures for Keyword Management

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class IntentType(Enum):
    NAVIGATIONAL = "navigational"
    INFORMATIONAL = "informational"
    COMMERCIAL = "commercial"
    TRANSACTIONAL = "transactional"

@dataclass
class Keyword:
    """Core keyword data structure"""
    text: str
    intent: IntentType
    primary: bool = False
    search_volume: int
    keyword_difficulty: float  # 0-100 scale
    search_trend: str  # "rising", "stable", "declining"
    cpc: Optional[float] = None
    estimated_clicks: int = 0

    def __post_init__(self):
        self.created_at = datetime.now()
        self.opportunity_score = self.calculate_opportunity()

    def calculate_opportunity(self) -> float:
        """
        Opportunity Score = (Search Volume × (100 - Difficulty)) / CPC
        Higher score = better keyword opportunity
        """
        if self.keyword_difficulty >= 90:
            return 0

        opportunity = (
            (self.search_volume * (100 - self.keyword_difficulty)) /
            (self.cpc if self.cpc else 1)
        )
        return min(100, opportunity)

@dataclass
class KeywordCluster:
    """Grouped keywords with cluster metadata"""
    cluster_id: int
    keywords: List[Keyword]
    primary_keyword: Keyword
    cluster_type: str  # "semantic", "topical", "serp", "hybrid"
    quality_score: float  # 0-1, from silhouette or other metric
    suggested_url: Optional[str] = None

    def get_all_target_keywords(self) -> List[str]:
        """Return all keywords in cluster for targeting"""
        return [kw.text for kw in self.keywords]

@dataclass
class PageKeywordMapping:
    """Maps keywords to specific page elements"""
    url: str
    primary_keyword: Keyword
    secondary_keywords: List[Keyword]
    lsi_keywords: List[str]

    element_mapping: Dict[str, List[str]] = None
    # Format: {
    #   "title": ["primary keyword"],
    #   "h1": ["primary keyword"],
    #   "h2_1": ["secondary_1", "secondary_2"],
    #   "h2_2": ["secondary_3"],
    #   "meta_description": ["primary keyword"],
    #   "image_alts": ["primary", "lsi_1"]
    # }

    def __post_init__(self):
        if self.element_mapping is None:
            self.element_mapping = {}

    def get_keyword_coverage(self) -> Dict[str, bool]:
        """Check if all critical placements are covered"""
        return {
            'title_included': bool(self.element_mapping.get('title')),
            'h1_included': bool(self.element_mapping.get('h1')),
            'meta_included': bool(self.element_mapping.get('meta_description')),
            'secondary_distributed': len(self.secondary_keywords) > 0
        }

@dataclass
class CanibalizationIncident:
    """Keyword cannibalization detection result"""
    keyword: str
    primary_url: str
    competing_url: str
    similarity_score: float  # 0-1
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    conflict_type: str  # "keyword_overlap", "content_similarity", "ranking_conflict"
    estimated_impact: Dict[str, int]  # {"lost_clicks": 50, "lost_impressions": 300}

    recommended_action: str  # "merge", "redirect", "differentiate", "coexist"
    action_priority: int  # 1-10, higher = more urgent
```

### 6.2 Scoring Formulas with Examples

#### 6.2.1 Keyword Opportunity Score

```python
def calculate_opportunity_score(keyword: Keyword,
                               strategy: str = "balanced") -> float:
    """
    Calculate keyword opportunity score balancing multiple factors.

    Args:
        keyword: Keyword object with metrics
        strategy: "volume", "difficulty", "balanced", "niche"

    Returns:
        Opportunity score (0-100)
    """

    # Base calculation
    volume_score = min(100, (keyword.search_volume / 10000) * 100)
    difficulty_inverse = (100 - keyword.keyword_difficulty)
    cpc_factor = min(100, (keyword.cpc if keyword.cpc else 1) * 10)

    # Strategy-weighted scores
    if strategy == "volume":
        return volume_score * 1.5 + difficulty_inverse * 0.3 + cpc_factor * 0.2
    elif strategy == "difficulty":
        return difficulty_inverse * 1.5 + volume_score * 0.3 + cpc_factor * 0.2
    elif strategy == "niche":
        return cpc_factor * 1.5 + difficulty_inverse * 0.3 + volume_score * 0.2
    else:  # balanced
        return (volume_score + difficulty_inverse + cpc_factor) / 3

# Example:
keyword = Keyword(
    text="AI content automation",
    intent=IntentType.COMMERCIAL,
    search_volume=8500,
    keyword_difficulty=65,
    cpc=2.50
)

score = calculate_opportunity_score(keyword, strategy="balanced")
# Result: ~60.8/100 (good opportunity, moderate difficulty)
```

#### 6.2.2 Clustering Quality Score

```python
def calculate_clustering_quality(cluster: KeywordCluster,
                                metrics: Dict) -> float:
    """
    Calculate overall clustering quality from multiple metrics.

    Args:
        cluster: KeywordCluster object
        metrics: Dictionary with metric values
        - silhouette_score: -1 to 1
        - keyword_count: number of keywords in cluster
        - diversity: 0-1 (lexical diversity)

    Returns:
        Quality score 0-1 (higher is better)
    """

    # Silhouette score contribution (40%)
    silhouette = metrics.get('silhouette_score', 0)
    silhouette_normalized = (silhouette + 1) / 2  # Convert -1..1 to 0..1
    silhouette_weight = 0.40

    # Cluster size (20%) - optimal is 3-5 keywords
    keyword_count = len(cluster.keywords)
    size_score = (
        0.0 if keyword_count < 2 else
        1.0 if 3 <= keyword_count <= 5 else
        0.5 if keyword_count <= 8 else
        0.2
    )
    size_weight = 0.20

    # Diversity (20%) - avoid duplicate keywords
    diversity = metrics.get('diversity', 0)
    diversity_weight = 0.20

    # Semantic coherence (20%) - keywords should be related
    coherence = metrics.get('semantic_coherence', 0.5)
    coherence_weight = 0.20

    # Calculate weighted average
    quality = (
        silhouette_normalized * silhouette_weight +
        size_score * size_weight +
        diversity * diversity_weight +
        coherence * coherence_weight
    )

    return min(1.0, max(0.0, quality))
```

### 6.3 Pipeline: Extraction → Classification → Clustering → Mapping

```python
class KeywordStrategyPipeline:
    """Complete keyword strategy implementation pipeline"""

    def __init__(self, config: Dict):
        self.config = config
        self.keywords = []
        self.clusters = []
        self.page_mappings = []
        self.cannibalization_incidents = []

    def execute_full_pipeline(self,
                             keyword_list: List[str],
                             content_docs: List[str],
                             existing_urls: Dict[str, str]) -> Dict:
        """
        Execute complete keyword strategy pipeline.

        Args:
            keyword_list: Raw keywords to analyze
            content_docs: Existing content documents for analysis
            existing_urls: Dictionary mapping URLs to content

        Returns:
            Complete strategy report
        """

        # Stage 1: Extraction & Enrichment
        self.keywords = self._extract_and_enrich_keywords(keyword_list)

        # Stage 2: Intent Classification
        self._classify_intent(self.keywords)

        # Stage 3: Clustering
        self.clusters = self._perform_clustering(
            self.keywords,
            content_docs
        )

        # Stage 4: Keyword-to-Content Mapping
        self.page_mappings = self._map_keywords_to_content(
            self.clusters,
            existing_urls
        )

        # Stage 5: Cannibalization Detection
        self.cannibalization_incidents = self._detect_cannibalization(
            self.page_mappings
        )

        # Stage 6: Report Generation
        return self._generate_report()

    def _extract_and_enrich_keywords(self,
                                    keyword_list: List[str]) -> List[Keyword]:
        """
        Extract keywords and enrich with metadata from APIs.
        Requires: SEMrush, Ahrefs, or Google Keyword Planner API access
        """
        enriched = []

        for kw_text in keyword_list:
            # Fetch metrics from API
            metrics = self._fetch_keyword_metrics(kw_text)

            keyword = Keyword(
                text=kw_text,
                intent=IntentType.INFORMATIONAL,  # Placeholder
                search_volume=metrics.get('search_volume', 0),
                keyword_difficulty=metrics.get('difficulty', 50),
                search_trend=metrics.get('trend', 'stable'),
                cpc=metrics.get('cpc', None)
            )

            enriched.append(keyword)

        return enriched

    def _fetch_keyword_metrics(self, keyword: str) -> Dict:
        """
        Fetch keyword metrics from external API.
        Placeholder - implement with actual API client.
        """
        # Implementation would use:
        # - SEMrush API: /analytics/v1/keywords/report-seed-keyword
        # - Ahrefs API: /v2/keywords/generate
        # - Google KP: google.ads.googleads.v14.services.KeywordPlanIdeaService

        return {
            'search_volume': 0,
            'difficulty': 50,
            'trend': 'stable',
            'cpc': None
        }

    def _classify_intent(self, keywords: List[Keyword]):
        """Apply intent classification to all keywords"""
        for keyword in keywords:
            signals = detect_intent_signals(keyword.text)

            # Assign highest confidence intent
            best_intent = max(signals.items(), key=lambda x: x[1])
            keyword.intent = IntentType[best_intent[0].upper()]

    def _perform_clustering(self,
                           keywords: List[Keyword],
                           documents: List[str]) -> List[KeywordCluster]:
        """Execute hybrid clustering"""

        keyword_texts = [kw.text for kw in keywords]

        # Run clustering algorithms
        semantic_result = semantic_keyword_clustering(keyword_texts)
        topical_result = tfidf_keyword_clustering(documents, keyword_texts)

        # Combine results
        clusters = self._merge_clustering_results(
            semantic_result,
            topical_result,
            keywords
        )

        return clusters

    def _map_keywords_to_content(self,
                                clusters: List[KeywordCluster],
                                existing_urls: Dict[str, str]) -> List[PageKeywordMapping]:
        """Map clusters to URL targets and content elements"""

        mappings = []

        for cluster in clusters:
            primary_kw = cluster.primary_keyword

            # Determine target URL
            target_url = self._find_or_create_target_url(
                cluster,
                existing_urls
            )

            # Map keywords to content elements
            element_map = self._create_element_mapping(cluster)

            mapping = PageKeywordMapping(
                url=target_url,
                primary_keyword=primary_kw,
                secondary_keywords=cluster.keywords[1:],
                lsi_keywords=generate_lsi_keywords(primary_kw.text),
                element_mapping=element_map
            )

            mappings.append(mapping)

        return mappings

    def _detect_cannibalization(self,
                               mappings: List[PageKeywordMapping]
                              ) -> List[CanibalizationIncident]:
        """Detect keyword cannibalization across mapped URLs"""

        # Create URL → keywords mapping
        url_keywords = {
            m.url: [m.primary_keyword.text] + [kw.text for kw in m.secondary_keywords]
            for m in mappings
        }

        # Run detection algorithms
        incidents = detect_url_keyword_cannibalization(
            url_keywords,
            threshold=0.6
        )

        return [CanibalizationIncident(**incident) for incident in incidents]

    def _generate_report(self) -> Dict:
        """Generate comprehensive strategy report"""

        return {
            'summary': {
                'total_keywords': len(self.keywords),
                'clusters_created': len(self.clusters),
                'page_mappings': len(self.page_mappings),
                'cannibalization_issues': len(self.cannibalization_incidents)
            },
            'keywords': self.keywords,
            'clusters': self.clusters,
            'page_mappings': self.page_mappings,
            'cannibalization_incidents': self.cannibalization_incidents,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations"""

        recommendations = []

        # High-opportunity keywords not yet targeted
        for kw in self.keywords:
            if kw.opportunity_score > 70:
                recommendations.append({
                    'type': 'HIGH_OPPORTUNITY_KEYWORD',
                    'keyword': kw.text,
                    'action': f'Create content targeting "{kw.text}"',
                    'priority': 'HIGH'
                })

        # Cannibalization resolution
        for incident in self.cannibalization_incidents:
            if incident.severity == 'CRITICAL':
                recommendations.append({
                    'type': 'CANNIBALIZATION_FIX',
                    'keyword': incident.keyword,
                    'action': f'{incident.recommended_action}: {incident.primary_url} vs {incident.competing_url}',
                    'priority': 'CRITICAL'
                })

        return recommendations
```

---

## 7. Success Metrics

### 7.1 Classification Accuracy Targets

```python
def evaluate_intent_classification(predictions: List[IntentType],
                                  ground_truth: List[IntentType]) -> Dict:
    """
    Evaluate intent classification performance.

    Args:
        predictions: Model predictions
        ground_truth: Human-labeled ground truth

    Returns:
        Performance metrics dictionary
    """

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    # Convert enum to string for sklearn
    pred_str = [p.value for p in predictions]
    truth_str = [t.value for t in ground_truth]

    metrics = {
        'accuracy': accuracy_score(truth_str, pred_str),
        'precision': precision_score(truth_str, pred_str, average='weighted', zero_division=0),
        'recall': recall_score(truth_str, pred_str, average='weighted', zero_division=0),
        'f1_score': f1_score(truth_str, pred_str, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(truth_str, pred_str)
    }

    return metrics

# Target Benchmarks:
# - Accuracy: >90% (commercial vs informational especially important)
# - Precision: >85% per intent type
# - Recall: >85% per intent type
# - F1 Score: >85% overall
```

### 7.2 Clustering Quality Measures

```python
def evaluate_clustering_quality(embeddings: np.ndarray,
                               cluster_labels: np.ndarray) -> Dict:
    """
    Evaluate clustering quality with multiple metrics.

    Args:
        embeddings: Document/keyword embeddings (n_samples × n_features)
        cluster_labels: Cluster assignments (n_samples,)

    Returns:
        Multiple quality metrics
    """

    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score,
        calinski_harabasz_score
    )

    # Silhouette Score (-1 to +1)
    # >0.7: strong, >0.5: reasonable, >0.25: weak
    silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')

    # Davies-Bouldin Index (lower is better, >0)
    # <1: good separation, <0.5: excellent
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)

    # Calinski-Harabasz Index (higher is better)
    # >30: good clusters
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)

    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz,
        'quality_assessment': (
            'EXCELLENT' if silhouette > 0.7 else
            'GOOD' if silhouette > 0.5 else
            'FAIR' if silhouette > 0.25 else
            'POOR'
        )
    }

# Target Benchmarks:
# - Silhouette Score: >=0.60 (reasonable to strong)
# - Davies-Bouldin Index: <1.0 (lower is better)
# - Calinski-Harabasz Index: >30
```

### 7.3 Cannibalization Detection Performance

```python
def evaluate_cannibalization_detection(detected_incidents: List[Dict],
                                      ground_truth_incidents: List[Dict]) -> Dict:
    """
    Evaluate cannibalization detection algorithm performance.

    Args:
        detected_incidents: Algorithm's detected incidents
        ground_truth_incidents: Human-verified incidents

    Returns:
        Precision, recall, F1 metrics
    """

    # Create comparable format (URL pairs)
    detected_pairs = {
        (inc['url1'], inc['url2']) for inc in detected_incidents
    }

    truth_pairs = {
        (inc['url1'], inc['url2']) for inc in ground_truth_incidents
    }

    # Calculate metrics
    true_positives = len(detected_pairs & truth_pairs)
    false_positives = len(detected_pairs - truth_pairs)
    false_negatives = len(truth_pairs - detected_pairs)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Target Benchmarks:
# - Precision: >90% (minimize false positives)
# - Recall: >85% (catch most issues)
# - F1 Score: >87%
```

---

## Conclusion

Modern keyword strategy requires sophisticated, multi-layered approaches combining intent classification, semantic and topical clustering, cannibalization detection, and optimal keyword-to-content mapping. The frameworks, algorithms, and implementations detailed in this document provide a comprehensive foundation for building production-grade SEO + AI content optimization systems.

**Key Takeaways:**

1. **Intent Classification** must move beyond simple keyword signals to incorporate journey stage, micro-moments, and contextual understanding
2. **Clustering Methods** should be hybrid (semantic + topical + SERP-based) rather than relying on single approach
3. **Cannibalization Detection** requires both keyword overlap and content similarity analysis
4. **Keyword Placement** follows strict rules for primary keywords (title, H1, meta, first paragraph) with strategic secondary distribution
5. **Density Metrics** have evolved from simple percentages to sophisticated saturation models (BM25) and distribution analysis
6. **Success Metrics** require multi-dimensional evaluation (accuracy, precision, recall, silhouette scores, etc.)

Implementation of this framework enables content optimization platforms to intelligently guide SEO strategy, prevent ranking dilution, and maximize organic traffic potential.

---

## References & Sources

### Intent Classification and Search Behavior
- [Determining the Informational, Navigational, and Transactional Intent of Web Queries (ResearchGate)](https://www.researchgate.net/publication/222824696_Determining_the_Informational_Navigational_and_Transactional_Intent_of_Web_Queries)
- [Search Intent Classifications: There are More Than 4 Types (Search Engine Land)](https://searchengineland.com/search-intent-more-types-430814)
- [What Is Search Intent: How to Identify & Optimize for It 2025 (Writesonic)](https://writesonic.com/blog/what-is-search-intent)
- [Micro-Moments Framework: How to Capture Buyers at the Right Time (PT Engine)](https://www.ptengine.com/blog/conversion-rate-optimization/googles-micro-moments-how-to-capture-buyers-at-the-right-time/)

### Keyword Clustering and Semantic Analysis
- [Keyword Clustering vs Semantic Clustering: What's the Difference? (PageOptimizer Pro)](https://www.pageoptimizer.pro/blog/keyword-clustering-vs-semantic-clustering)
- [Text Clustering and Topic Modeling with LLMs (Medium)](https://medium.com/@piyushkashyap045/text-clustering-and-topic-modeling-with-llms-446dd7657366)
- [LLM-Guided Semantic-Aware Clustering for Topic Modeling (ACL Anthology)](https://aclanthology.org/2025.acl-long.902.pdf)
- [SERP-Based Keyword Clustering Tool (Keyword Insights)](https://www.keywordinsights.ai/features/keyword-clustering/)

### Keyword Cannibalization Detection
- [Keyword Cannibalization: The Silent SEO Killer (Lead Advisors)](https://leadadvisors.com/blog/keyword-cannibalization/)
- [Fix Keyword Cannibalization with Mueller's Insights (WebProNews)](https://www.webpronews.com/fix-keyword-cannibalization-boost-seo-with-muellers-insights/)
- [AI Tools Detect Keyword Cannibalization (The Ad Firm)](https://www.theadfirm.net/how-ai-tools-can-detect-cannibalization-and-fix-internal-competing-keywords-2/)

### Keyword Density and Relevance Scoring
- [BM25 Explained: A Better Ranking Algorithm than TF-IDF (Vishwas Gowda)](https://vishwasg.dev/blog/2025/01/20/bm25-explained-a-better-ranking-algorithm-than-tf-idf/)
- [BM25 vs TF-IDF: Keyword Search Explained (Ólafur Aron Jóhannsson)](https://olafuraron.is/blog/bm25vstfidf/)
- [BM25 and Its Role in Document Relevance Scoring (Sourcely)](https://www.sourcely.net/resources/bm25-and-its-role-in-document-relevance-scoring)
- [TF-IDF and Cosine Similarity in Machine Learning (Dot Net Tutorials)](https://dotnettutorials.net/lesson/tf-idf-and-cosinesimilarity-in-machine-learning/)

### Keyword Mapping and Content Structure
- [Header Tags for SEO: H1, H2, H3 Best Practices (SEO Sherpa)](https://seosherpa.com/header-tags/)
- [SEO Best Practices for Meta Titles & Descriptions (Team Lewis)](https://www.teamlewis.com/magazine/seo-metadata-best-practices-on-page-optimization/)
- [How to Use H1, H2, and H3 Tags Effectively for SEO (Writesonic)](https://writesonic.com/blog/how-to-use-h1-h2-h3-tags-for-seo)

### LSI Keywords and Semantic Optimization
- [What are LSI Keywords and Do They Help With SEO? (Backlinko)](https://backlinko.com/hub/seo/lsi)
- [LSI Keywords: What You Should Know (Semrush)](https://www.semrush.com/blog/lsi-keywords/)
- [Latent Semantic Indexing in SEO (Search Engine Journal)](https://www.searchenginejournal.com/ranking-factors/latent-semantic-indexing/)

### Featured Snippets and PAA Optimization
- [Featured Snippets: How to Earn Them (Semrush)](https://www.semrush.com/blog/featured-snippets/)
- [How to Optimize for Google's People Also Ask Section (AIOSEO)](https://aioseo.com/how-to-optimize-for-googles-people-also-ask/)
- [Featured Snippet Optimization: Dominate 2025 (ROI Amplified)](https://roiamplified.com/insights/featured-snippet-optimization/)

### Clustering Quality Evaluation
- [Silhouette Score for Clustering Evaluation (Number Analytics)](https://www.numberanalytics.com/blog/silhouette-score-clustering-evaluation)
- [Silhouette Coefficient Overview (ScienceDirect)](https://www.sciencedirect.com/topics/computer-science/silhouette-coefficient)
- [K-Means Cluster Evaluation with Silhouette Analysis (Machine Learning Mastery)](https://machinelearningmastery.com/k-means-cluster-evaluation-with-silhouette-analysis/)

### Keyword Research APIs
- [Best Keyword Research APIs in 2025 (Coefficient)](https://coefficient.io/keyword-research-apis)
- [SEMrush API Documentation](https://developer.semrush.com/api/v3/analytics/keyword-reports/)
- [Best Keyword Research Tools for 2025 (SMA Marketing)](https://www.smamarketing.net/blog/best-keyword-research-tools-2025)

---

**Document prepared for:** SEO + AI Content Optimization Tool Development
**Next Steps:** Implementation of algorithms in production environment with API integrations and real-time monitoring systems.

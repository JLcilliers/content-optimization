# Topic B: Quick Reference Guide
## Keyword Strategy & Intent Modeling - Lookup Tables & Decision Matrices

**Document Version:** 1.0
**Date:** January 2026

---

## 1. Intent Classification Quick Lookup

### Signal Keywords by Intent Type

#### Informational Intent (52.65% of searches)
```
Primary Signals: how, what, why, where, when
Related: guide, tutorial, tips, best practices, learn, steps, FAQ, explain

Example: "how to start a blog", "what is digital marketing", "best SEO practices"

Content Strategy: Educational articles, how-to guides, video tutorials, webinars
Target Keywords Per Page: 1 primary + 3-5 secondary keywords
Expected SERP Features: Featured snippet, knowledge panel, related questions
```

#### Navigational Intent (32.15% of searches)
```
Primary Signals: brand name, [brand] login, [brand] app, [brand] + feature

Related: official website, dashboard, portal, platform specific

Example: "Facebook login", "Gmail inbox", "Amazon account", "Slack download"

Content Strategy: Improve site navigation, mobile UX, branded content
Target Keywords Per Page: Brand/product specific
Expected SERP Features: Knowledge panel, site links, official page highlighted
```

#### Commercial Investigation (14.51% of searches)
```
Primary Signals: best, top, review, compare, vs, alternatives, which

Related: [product category] for [use case], top rated, highest rated

Example: "best CRM software", "Salesforce vs HubSpot", "top email marketing tools"

Content Strategy: Comparison articles, review roundups, feature matrices
Target Keywords Per Page: 1 primary + 5-8 secondary keywords
Expected SERP Features: Product reviews, comparison tables, ratings/reviews
```

#### Transactional Intent (0.69% of searches)
```
Primary Signals: buy, purchase, price, discount, coupon, free trial, download

Related: where to buy, order online, in stock, fast shipping, deal

Example: "buy MacBook Pro", "PHP hosting discount", "download Photoshop free trial"

Content Strategy: Product pages, pricing pages, checkout optimization
Target Keywords Per Page: Product/price specific
Expected SERP Features: Shopping results, reviews, pricing info, stock status
```

### Quick Intent Determination

**Use This Decision Tree:**
```
Does the query include: "how to", "what is", "guide", "tutorial", "tips"?
└─ YES → INFORMATIONAL
└─ NO → Continue

Does the query include: brand name or specific product?
└─ YES → NAVIGATIONAL
└─ NO → Continue

Does the query include: "best", "review", "compare", "vs", "alternatives"?
└─ YES → COMMERCIAL
└─ NO → Continue

Does the query include: "buy", "price", "discount", "coupon", "download"?
└─ YES → TRANSACTIONAL
└─ NO → Likely INFORMATIONAL (default)
```

---

## 2. Keyword Clustering Decision Matrix

### Which Clustering Method to Use?

| Your Situation | Recommended Method | Why | Trade-offs |
|---|---|---|---|
| <50 keywords, low budget | **Semantic Only** | Fast, cheap, covers 70% of needs | May miss topical relationships |
| 50-500 keywords, standard budget | **Hybrid (Semantic + Topical)** | Best accuracy, captures all relationships | 2-3x slower, requires TF-IDF corpus |
| 500+ keywords, high volume | **Hybrid + SERP-based** | Most accurate, reflects real SERP | Expensive API calls, longest runtime |
| Keyword clusters already exist | **Validate Only** | Don't reinvent the wheel | Audit for missing keywords |
| Seasonal/trending content | **SERP-based (update monthly)** | Catches changing intent | API costs, requires regular updates |

### Clustering Quality Assessment

**Silhouette Score Interpretation:**

| Score Range | Quality | Action |
|---|---|---|
| **0.71 - 1.0** | STRONG | ✓ Accept clustering, proceed to mapping |
| **0.51 - 0.70** | REASONABLE | ✓ Acceptable, monitor for improvements |
| **0.26 - 0.50** | WEAK | ⚠ Consider re-clustering with different method |
| **-1.0 - 0.25** | POOR | ❌ Re-cluster, try different threshold/algorithm |

### Algorithm Comparison

| Factor | Semantic | Topical (TF-IDF) | SERP-Based | Hybrid |
|---|---|---|---|---|
| **Speed** | Fast | Medium | Slow | Medium |
| **Cost** | $0 | $0 | $50-500/month | $50-500/month |
| **Accuracy** | Good (85%) | Good (80%) | Excellent (95%) | Excellent (92%) |
| **Intent Capture** | OK | Good | Excellent | Excellent |
| **Implementation** | Easy | Medium | Hard | Medium |
| **Best For** | MVP, small sites | Content audits | Competitor analysis | Production systems |

---

## 3. Keyword Placement Rules by Element

### Complete Placement Specification

#### Title Tag
```
Position: Front (first 3 words optimal)
Format: [Primary Keyword] - [Value Prop]
Character Limit: 50-60 characters
Include: Exact primary keyword or close variant
Example: "Best Project Management Software for Remote Teams 2026"

SEO Impact: HIGH (directly affects CTR and ranking)
Quality Score: CRITICAL
```

#### H1 Heading (Page Title)
```
Position: Early (within first 100 words)
Format: Full primary keyword or close variant
Character Limit: 20-70 characters
Count Per Page: EXACTLY 1 (only one H1)
Include: Primary keyword naturally
Example: "Best Project Management Software for Remote Teams"

SEO Impact: HIGH
Quality Score: CRITICAL
```

#### Meta Description
```
Position: Naturally, ideally first sentence
Format: Include primary or semantic variant
Character Limit: 150-160 characters
Inclusion: Primary keyword optional but recommended
Include CTR Elements: Action words, benefit statement, emotional trigger
Example: "Discover the best project management software for remote teams. Compare features, pricing, and customer reviews for 2026."

SEO Impact: MEDIUM (mostly affects CTR)
Quality Score: IMPORTANT
```

#### First Paragraph
```
Position: Within first 50 words
Occurrence Count: 1 per paragraph
Format: Natural inclusion, don't force
Include: Primary keyword or close variant
Purpose: Establish immediate topical relevance

SEO Impact: HIGH (Google weights early content heavily)
Quality Score: CRITICAL
```

#### H2 Subheadings (3-8 per page typical)
```
Position: Distribute throughout content
Occurrence Per H2: 1 per 500 words of section content
Format: Include secondary keyword naturally
Count: 3-8 H2s per page is standard
Secondary Keywords: Assign one per H2 (not all H2s need keywords)
Example H2s: "Machine Learning Applications", "Implementation Strategy", "ROI Measurement"

SEO Impact: MEDIUM-HIGH
Quality Score: IMPORTANT
```

#### URL Slug
```
Format: /[keyword-variant]/[specific-angle]
Length: Under 75 characters
Included: Primary keyword variant in slug
Example: /best-project-management-software-remote-teams
Separator: Hyphens only, lowercase

SEO Impact: MEDIUM (ranking + UX)
Quality Score: IMPORTANT
```

#### Image Alt Text
```
Count: At least 1 image with primary keyword
Format: Descriptive, natural sentence
Include: Primary keyword in at least one image
Example: "Best project management software interface showing team collaboration features"
Purpose: Accessibility + image search ranking

SEO Impact: LOW-MEDIUM
Quality Score: NICE TO HAVE
```

#### Anchor Text (Internal Links)
```
Occurrence: 1-2 per secondary keyword
Format: Keyword as anchor text or URL as anchor
Spread: Avoid clustering all links in one section
Include: Mix of keyword anchors + branded anchors
Best Practice: "Best project management software for remote teams" → /best-pm-software
Avoid: Generic "click here", but also avoid over-optimization

SEO Impact: MEDIUM
Quality Score: IMPORTANT
```

#### Body Content
```
Density (Primary): 0.5-1.5% (exact match)
Density (All variants): 2-4%
Distribution: Evenly spread, not clustered
Variation: Use semantic variations and synonyms
First 100 words: Include primary keyword once
Every 500 words: Refresh primary keyword once
Avoid: Exact same phrase repeated verbatim

SEO Impact: HIGH
Quality Score: CRITICAL
```

---

## 4. Keyword Density Thresholds

### Exact-Match Density Table

| Density % | Exact Phrase Count | Assessment | Action |
|---|---|---|---|
| **<0.3%** | 1-2 in 500 words | UNDER-OPTIMIZED | Add more occurrences |
| **0.5-1.5%** | 2-3 in 500 words | OPTIMAL | ✓ Good, no change needed |
| **1.5-2.5%** | 3-5 in 500 words | ELEVATED | Monitor, may reduce slightly |
| **2.5-3.5%** | 5-7 in 500 words | HIGH | ⚠ Reduce, add variations |
| **3.5-5%** | 7-10 in 500 words | VERY HIGH | ⚠ Significant reduction needed |
| **>5%** | 10+ in 500 words | CRITICAL | ❌ Likely keyword stuffing penalty |

### Quick Calculation
```
Density % = (Exact Keyword Matches / Total Words) × 100

Example:
Content: 2,000 words
Primary keyword occurrences: 18
Density: (18 / 2000) × 100 = 0.9%
Assessment: OPTIMAL ✓
```

### Variant Density Table (Primary + Similar Forms)

| Variant Type | Density Range | Notes |
|---|---|---|
| **Exact phrase** | 0.5-1.5% | "machine learning" |
| **With articles** | +0.3-0.5% | "the machine learning", "machine learning model" |
| **Singular/plural** | +0.2-0.4% | "machines", "learnings" (if natural) |
| **Word order variation** | +0.2-0.3% | "learning machine" (if natural) |
| **Total acceptable** | 1.5-3.0% | All variations combined |

---

## 5. Cannibalization Severity & Action Matrix

### Severity Assessment

| Similarity | Severity | GSC Impact | Recommendation |
|---|---|---|---|
| **>0.85** | CRITICAL | Both URLs losing traffic | **MERGE** content + 301 redirect |
| **0.70-0.85** | HIGH | One URL suppressed | **REDIRECT** lower authority to higher |
| **0.50-0.70** | MEDIUM | Partial suppression | **DIFFERENTIATE** add unique angles |
| **<0.50** | LOW | Possible coexistence | **MONITOR** track rankings closely |

### Resolution Workflow

#### MERGE (Similarity >0.85)
```
1. Identify which URL has more authority (links, traffic, age)
2. Copy all unique content from secondary URL to primary
3. Update internal links to point to primary only
4. Set up 301 redirect: secondary → primary
5. Wait 4-6 weeks for consolidation
6. Monitor: combined URL should rank higher than both originals
7. Expected improvement: +20-40% combined traffic
```

#### REDIRECT (Similarity 0.70-0.85)
```
1. Choose primary URL (higher authority or better content)
2. Set 301 redirect: secondary → primary
3. Keep redirect in place permanently
4. Update all internal links within 2 weeks
5. Notify Search Console of change (if available)
6. Monitor rankings in GSC
7. Expected: Secondary's rankings transfer to primary
```

#### DIFFERENTIATE (Similarity 0.50-0.70)
```
1. Analyze: What makes them different currently?
2. Expand one URL to cover angle the other doesn't
3. Add unique: Examples, data, case studies, different use case
4. Internal link: Secondary → Primary with context
5. Update both H1s to reflect unique angles
6. Resubmit to Search Console
7. Expected: Both URLs can rank for different aspects
```

#### MONITOR (Similarity <0.50)
```
1. Track both URLs in GSC weekly
2. Set alert: if either URL drops >30% traffic
3. Check: Do they rank for same keywords? (GSC)
4. If no overlap: keep as is
5. If overlap: implement DIFFERENTIATE strategy
6. Expected: Natural coexistence, minimal conflict
```

### Red Flags for Cannibalization

```
❌ CRITICAL SIGNS:
├─ Same URL ranks #2 and #4 for different queries
├─ Similar content on multiple URLs
├─ All occurrences of keyword in first 3 paragraphs
├─ 90%+ keyword overlap between two pages
├─ Duplicate meta descriptions across URLs
├─ Same internal linking strategy
└─ Both URLs have same authority signals

⚠️ WARNING SIGNS:
├─ Keyword appears in multiple H1s
├─ Similar H2 sections on different URLs
├─ 70%+ keyword overlap
├─ Both URLs in top 20 for same keyword
├─ Very close publication dates
└─ Slight variations only ("guide" vs "tutorial")

✓ ACCEPTABLE COEXISTENCE:
├─ Different keywords targeted (keyword clusters)
├─ Different intent levels (informational vs transactional)
├─ Clear content differentiation
├─ 50%+ keyword overlap acceptable if intentional
├─ One URL older/more authoritative
└─ Clear internal linking hierarchy
```

---

## 6. SEO Success Metrics by Keyword Type

### Informational Keywords (52% of searches)
**Goal:** Build topical authority, capture organic awareness

| Metric | Target | Rationale |
|---|---|---|
| **Position** | Top 3 | High intent, no commercial pressure |
| **Click-through Rate** | 25-35% | Educational content CTR |
| **Time on Page** | 2+ minutes | Deep engagement expected |
| **Pages Per Session** | 2.5+ | Cluster content linking |
| **Bounce Rate** | <45% | Users find answers |
| **Ranking Timeline** | 2-4 months | Lower competition |

### Commercial Keywords (14.5% of searches)
**Goal:** Become trusted authority, prepare for conversion

| Metric | Target | Rationale |
|---|---|---|
| **Position** | Top 5 | Competitive, needs review coverage |
| **Click-through Rate** | 15-25% | Less intent than transactional |
| **Time on Page** | 3+ minutes | Detailed comparisons needed |
| **Scroll Depth** | 70%+ | Readers deep-dive comparisons |
| **Bounce Rate** | <40% | Users comparing options |
| **Ranking Timeline** | 3-6 months | Higher competition |
| **Secondary Metric** | Backlinks | Editorial links indicate authority |

### Transactional Keywords (0.69% of searches)
**Goal:** Drive immediate conversions

| Metric | Target | Rationale |
|---|---|---|
| **Position** | Top 3 | Highest intent, users ready to buy |
| **Click-through Rate** | 20-40% | Strong purchase intent |
| **Conversion Rate** | 2-5% | Depends on product |
| **Time on Page** | 1-2 minutes | Users know what they want |
| **Form Completion** | 5-10% | Qualified leads |
| **Cart Abandonment** | <70% | Standard benchmark |
| **Ranking Timeline** | 1-3 months | Easiest to rank |

### Navigational Keywords (32% of searches)
**Goal:** Brand defense, user experience

| Metric | Target | Rationale |
|---|---|---|
| **Position** | #1 (mandatory) | Your own brand/site |
| **Click-through Rate** | >60% | Direct intent |
| **Bounce Rate** | Varies | Users going exactly where intended |
| **Time on Page** | <1 minute | Direct navigation expected |
| **Ranking Timeline** | Immediate | Should rank immediately |

---

## 7. Content Element Checklist by Intent Type

### Informational Content Checklist
```
❑ Headline includes "guide", "tips", "how to", "best practices"
❑ Intro paragraph explains what reader will learn
❑ Table of contents for >2000 word pieces
❑ Multiple H2 sections (5-8 sections minimum)
❑ Examples throughout (1 per section minimum)
❑ Visuals: 1-2 per 500 words
❑ Related readings/further resources section
❑ FAQ section (mirrors PAA questions)
❑ Video embedded if available
❑ Workable checklists/templates
```

### Commercial Content Checklist
```
❑ Headline includes "best", "top", "review", "compare"
❑ Intro: Clear positioning statement
❑ Comparison table (if multiple options)
❑ Feature breakdowns per option
❑ Pros/cons for each alternative
❑ Pricing comparison (if available)
❑ Customer review aggregation
❑ Photos/screenshots of products
❑ "Verdict" or recommendation section
❑ Links to official product pages
❑ Clear pros/cons formatting
```

### Transactional Content Checklist
```
❑ Headline: Clear product/price focus
❑ Immediate trust signals (reviews, ratings)
❑ Clear pricing display
❑ "Add to Cart" button above fold
❑ Product images (5+ angles minimum)
❑ Detailed product specifications
❑ Customer reviews embedded
❑ Security badges/trust signals
❑ Return policy clearly stated
❑ Fast shipping/delivery info
❑ Stock status indicator
```

### Navigational Content Checklist
```
❑ Page title matches search query exactly
❑ H1 exactly matches page purpose
❑ Clear navigation to requested resource
❑ Mobile-optimized interface
❑ Fast loading (GTmetrics >90)
❑ Intuitive information hierarchy
❑ Search function (if internal navigation)
❑ Breadcrumbs present
❑ Related navigation links
❑ Account/login forms prominent
```

---

## 8. Over-Optimization Risk Assessment Matrix

### Quick Risk Check

Answer these questions (1 = Yes, 0 = No):

| Indicator | Yes=1 | No=0 |
|---|---|---|
| Exact-match density >2% | 1 | 0 |
| All keyword occurrences in first 25% of content | 1 | 0 |
| Keywords appear in more than 50% of H2 headers | 1 | 0 |
| 5+ internal links with identical anchor text | 1 | 0 |
| Multiple title tags with same keyword | 1 | 0 |
| Meta description repeated across multiple pages | 1 | 0 |
| Same keyword in alt text of multiple images | 1 | 0 |
| Keyword appears in meta keywords tag | 1 | 0 |
| Hidden text detected (CSS display:none, white text) | 1 | 0 |
| Unrelated lists/tables with keyword inserted | 1 | 0 |

### Risk Score Interpretation

```
Score Calculation: Sum all 1s

0-2: SAFE ✓
└─ No action needed, continue optimization

3-4: CAUTION ⚠️
└─ Review and reduce density slightly
└─ Spread keywords more naturally
└─ Add more semantic variations

5-7: HIGH RISK ⚠️⚠️
└─ Significant reduction needed
└─ Remove keyword from some H2s
└─ Rewrite for natural flow
└─ Add LSI keywords to balance

8-10: CRITICAL ❌
└─ Likely penalties in effect
└─ Major content rewrite needed
└─ Reduce density to 0.5-1% max
└─ Check for manual penalties in Search Console
└─ Consider 301 redirect if severe
```

---

## 9. Implementation Timeline

### Week 1-2: MVP Setup
```
Monday-Tuesday:
└─ Set up intent classification (rule-based)
└─ Configure semantic clustering
└─ Deploy on 100 test keywords

Wednesday-Thursday:
└─ Validate accuracy >85%
└─ Test clustering quality >0.55
└─ Fix signal patterns

Friday:
└─ Internal testing complete
└─ Documentation ready
└─ Ready for beta testers
```

### Week 3-4: Production Features
```
Monday-Tuesday:
└─ Implement cannibalization detection
└─ Add keyword density analysis
└─ Deploy validation checks

Wednesday-Thursday:
└─ Add topical clustering (TF-IDF)
└─ Integrate SERP-based clustering
└─ Implement weighted fusion

Friday:
└─ Performance optimization
└─ Load testing (500+ keywords)
└─ Error handling complete
```

### Week 5-6: Advanced Features
```
Monday-Tuesday:
└─ SEMrush API integration
└─ Ahrefs API integration
└─ GSC data import

Wednesday-Thursday:
└─ LSI keyword generation
└─ Featured snippet detection
└─ Question-based keyword identification

Friday:
└─ Full pipeline testing
└─ Performance benchmarks
└─ Production readiness
```

---

## 10. Common Questions Quick Answers

### Q: How many keywords should I target per page?
**A:**
- Primary: 1 keyword ONLY
- Secondary: 3-8 keywords (depending on page length)
- LSI/semantic: 10-20 variations
- Total "keyword focus": 1 primary + 5 secondary average

### Q: Can two pages rank for the same keyword?
**A:**
- Not ideally. Google will choose one as primary.
- If intentional: make sure clear differentiation (intent levels)
- If unintentional: implement cannibalization resolution

### Q: What's a good keyword density?
**A:**
- Exact-match: 0.5-1.5%
- Primary + variants: 1.5-3.0%
- Higher density = higher penalty risk
- Natural content usually falls in this range naturally

### Q: How long does ranking take?
**A:**
- Informational keywords: 2-4 months
- Commercial keywords: 3-6 months
- Transactional keywords: 1-3 months
- Navigational: Immediate (should be your own brand)

### Q: Should I use LSI keywords?
**A:**
- Google doesn't use LSI for ranking
- BUT: Semantic variations help content quality
- Use them naturally, not forced
- Improves CTR and user experience

### Q: How do I fix cannibalization?
**A:**
- Similarity >85%: MERGE + 301 redirect
- Similarity 70-85%: 301 redirect
- Similarity 50-70%: Add differentiation
- Similarity <50%: Monitor

### Q: What if I have 1000+ keywords?
**A:**
- Use clustering first (reduce to 100-200 clusters)
- Prioritize by opportunity score
- Use batch processing for speed
- Focus on top 20% of keywords first (80/20 rule)

---

## 11. Integration with Other SEO Activities

### Coordinates With:

**Technical SEO**
- Ensure fast page load for keyword landing pages
- Mobile responsiveness for all target keywords
- Structured data markup for rich snippets
- Internal linking based on keyword clusters

**Content Creation**
- Brief writers using keyword mapping
- Assign specific H2s by secondary keywords
- Specify LSI keyword requirements
- Set density targets in style guide

**Backlink Strategy**
- Prioritize links for commercial/high-opportunity keywords
- Anchor text from keyword cluster strategy
- Guest posting on related topical clusters
- Internal link anchors from secondary keywords

**User Experience**
- Keyword clustering informs site architecture
- Intent classification informs content hierarchy
- Primary keywords inform navigation labels
- Cannibalization prevention improves findability

**Analytics & Reporting**
- Track ranking positions by keyword cluster
- Monitor density compliance
- Measure intent classification accuracy
- Calculate cluster ROI (traffic per cluster)

---

**Quick Reference prepared for:** SEO + AI Content Optimization Tool Development
**Print Friendly:** Consider printing decision matrices for team reference
**Last Updated:** January 2026

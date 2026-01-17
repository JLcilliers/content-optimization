# Topic C: On-Page SEO Optimization Framework
## Technical Specification for AI Content Optimization Tool

**Document Version:** 1.0
**Date:** January 16, 2026
**Author:** SEO Technical Research Team

---

## Executive Summary

This document defines a comprehensive on-page SEO optimization framework for implementation in an AI-powered content optimization tool. The framework provides machine-readable rules, validation algorithms, and scoring formulas for all critical on-page SEO elements.

On-page SEO in 2026 has evolved beyond simple keyword placement to encompass user experience signals, accessibility compliance (WCAG 2.1 AA), mobile-first design principles, and structured data implementation. Search engines now use AI to evaluate content quality, making it essential to balance technical optimization with authentic user value. The rise of AI Overviews and Answer Engine Optimization (AEO) demands that content demonstrates clear topical relationships through internal linking, descriptive anchor text, and predictable navigation patterns.

This framework addresses nine core optimization domains: title tags, meta descriptions, heading hierarchy, URL structure, internal linking, image optimization, schema markup, content structure, and implementation specifications. Each domain includes specific thresholds, validation rules, and weighted scoring formulas to produce actionable recommendations. The system uses a 0-100 point scale with weighted priority categories: Critical (40%), High (30%), Medium (20%), and Low (10%).

Research shows that pages with optimized on-page elements experience 36% higher click-through rates, while proper schema markup and content structure significantly improve visibility in AI-powered search results. This framework enables systematic evaluation and improvement of all on-page elements while maintaining compliance with accessibility standards and emerging search engine requirements.

---

## 1. Title Tag Optimization Rules

### 1.1 Character and Pixel Length Constraints

**Primary Constraint:** Pixel width, not character count
**Optimal Range:** 50-60 characters (~580-600 pixels)
**Absolute Maximum:** 600 pixels to prevent truncation

| Display Context | Pixel Limit | Approximate Characters | Truncation Risk |
|----------------|-------------|----------------------|-----------------|
| Desktop SERP | 600px | 50-60 chars | Low if ≤600px |
| Mobile SERP | 600px | 50-60 chars | Medium if >55 chars |
| Social Shares | Variable | 40-50 chars | High if >50 chars |

**Validation Rule:**
```
IF pixel_width(title) > 600 THEN
  score = 0
  warning = "Title exceeds 600px and will be truncated"
ELSE IF pixel_width(title) > 580 AND pixel_width(title) <= 600 THEN
  score = 85
  warning = "Title approaching truncation threshold"
ELSE IF pixel_width(title) >= 480 AND pixel_width(title) <= 580 THEN
  score = 100
  warning = null
ELSE IF pixel_width(title) < 480 THEN
  score = 70
  warning = "Title may be too short to maximize SERP real estate"
END IF
```

**Note:** Different characters consume different pixel widths (W=13px, i=3px average). Calculate using Google's Roboto font or equivalent SERP rendering font.

### 1.2 Keyword Placement Hierarchy

**Front-Loading Priority:** Target keywords in first 3 words rank 1.5 positions higher on average.

**Placement Scoring:**
```
keyword_position_score = 100 - (word_position * 5)

IF primary_keyword_position <= 3 THEN
  position_score = 100
ELSE IF primary_keyword_position <= 6 THEN
  position_score = 85
ELSE IF primary_keyword_position <= 10 THEN
  position_score = 70
ELSE
  position_score = 50
END IF
```

**Natural Integration Requirements:**
- Primary keyword: Exactly 1 occurrence
- Secondary keywords: 0-1 occurrence
- LSI/semantic variations: Encouraged
- Keyword stuffing detection: Flag if keyword density >15% of title

### 1.3 CTR Optimization Factors

**Power Words** (increase perceived value):
- Ultimate, Complete, Essential, Proven, Definitive, Comprehensive
- Expert, Professional, Advanced, Beginner, Simple, Easy
- Free, New, Secret, Hack, Cheat Sheet

**Power Word Scoring:**
```
IF contains_power_word(title) THEN
  ctr_boost = +15 points
ELSE
  ctr_boost = 0
END IF
```

**Numbers and Lists:**
- Numbers outperform generic alternatives by 23% CTR
- Odd numbers perform slightly better than even (7 > 8)
- Ranges work well (5-10 Tips)

**Number Scoring:**
```
IF contains_number(title) THEN
  ctr_boost = +20 points
ELSE
  ctr_boost = 0
END IF
```

**Bracketed Additions:**
- Increase CTR by 15-20%
- Examples: [2026 Guide], [Free Template], [Updated], [Step-by-Step]
- Position: End of title preferred

**Bracket Scoring:**
```
IF contains_bracket_modifier(title) THEN
  ctr_boost = +15 points
ELSE
  ctr_boost = 0
END IF
```

**Year References:**
- Include current year (2026) for time-sensitive content
- Signals freshness and relevance
- Update annually for evergreen content

### 1.4 Brand Name Positioning

**Options:**
1. **Brand at End:** "Complete SEO Guide | BrandName" (recommended for content pages)
2. **Brand at Start:** "BrandName: Complete SEO Guide" (recommended for homepage)
3. **No Brand:** Use space for keywords (for highly competitive queries)

**Brand Separator Characters:**
- Pipe (|): Most common, 53% usage
- Hyphen (-): 28% usage
- Colon (:): 19% usage

**Scoring Logic:**
```
IF is_homepage THEN
  IF brand_at_start THEN score = 100
  ELSE score = 80
ELSE IF is_content_page THEN
  IF brand_at_end THEN score = 100
  ELSE IF no_brand THEN score = 95
  ELSE score = 70
END IF
```

### 1.5 Duplicate Title Detection

**Validation:**
```
duplicates = SELECT COUNT(*) FROM pages
             WHERE title_tag = current_title
             GROUP BY title_tag

IF duplicates > 1 THEN
  score = 0
  error = "Duplicate title tag detected on {duplicates} pages"
ELSE
  score = 100
END IF
```

### 1.6 Complete Title Tag Scoring Formula

```
total_title_score = (
  (length_score * 0.25) +
  (keyword_position_score * 0.30) +
  (ctr_elements_score * 0.25) +
  (brand_placement_score * 0.10) +
  (uniqueness_score * 0.10)
) * 100

WHERE:
  length_score = pixel_width validation score (0-1)
  keyword_position_score = position scoring (0-1)
  ctr_elements_score = (power_words + numbers + brackets) / 300 (capped at 1)
  brand_placement_score = brand positioning score (0-1)
  uniqueness_score = duplicate detection score (0-1)
```

### 1.7 Concrete Examples

| Title Example | Length | Keywords | CTR Elements | Score | Analysis |
|--------------|--------|----------|-------------|-------|----------|
| **GOOD:** "7 Proven SEO Strategies to Rank #1 [2026 Guide]" | 54 chars | Front-loaded | Number, Power word, Bracket, Year | 95/100 | Excellent structure with all CTR elements |
| **GOOD:** "Complete Python Tutorial for Beginners \| CodeAcademy" | 58 chars | Early position | Power word, Brand separator | 88/100 | Strong brand integration |
| **BAD:** "Learn About SEO and Digital Marketing Strategies Online" | 60 chars | Weak positioning | Generic, no CTR elements | 45/100 | Missing numbers, power words, and specific value |
| **BAD:** "SEO SEO Tips Marketing Guide SEO 2026 SEO Tutorial" | 52 chars | Over-stuffed | Keyword stuffing detected | 15/100 | Excessive keyword repetition (keyword density 32%) |
| **BAD:** "Welcome to Our Website - Home Page - SEO Services" | 52 chars | Weak keywords | No CTR elements | 30/100 | Generic, unfocused, weak value proposition |

---

## 2. Meta Description Framework

### 2.1 Length Guidelines

**Optimal Ranges:**
| Device | Pixel Limit | Character Range | Recommendation |
|--------|------------|-----------------|----------------|
| Desktop | 930px | 155-160 chars | Aim for 155-158 |
| Mobile | 680px | ~120 chars | Front-load key info in first 120 |
| Universal Safe | 680px | 120 chars | Critical info here |

**Validation Logic:**
```
IF char_length(description) < 120 THEN
  score = 70
  warning = "Description too short; missing opportunity for keywords/CTA"
ELSE IF char_length(description) >= 120 AND char_length(description) <= 158 THEN
  score = 100
  warning = null
ELSE IF char_length(description) > 158 AND char_length(description) <= 160 THEN
  score = 90
  warning = "May truncate on some mobile devices"
ELSE IF char_length(description) > 160 THEN
  score = 60
  warning = "Will be truncated on most devices"
END IF
```

**Front-Loading Rule:**
- Place most important keywords and value proposition in first 120 characters
- This ensures visibility on all devices (mobile truncates earlier)

### 2.2 Call-to-Action Patterns

**Effective CTA Phrases:**
- Action-oriented: "Learn how to...", "Discover...", "Get started with..."
- Urgency: "Start today", "Don't miss", "Limited time"
- Value: "Free guide", "Step-by-step tutorial", "Proven strategies"
- Discovery: "Find out", "See how", "Unlock"

**CTA Detection and Scoring:**
```
action_verbs = ["learn", "discover", "get", "find", "see", "unlock", "start", "explore", "master"]
value_indicators = ["free", "guide", "tutorial", "template", "proven", "complete", "ultimate"]

IF contains_any(description, action_verbs) THEN
  cta_score += 50
END IF

IF contains_any(description, value_indicators) THEN
  cta_score += 50
END IF

final_cta_score = MIN(cta_score, 100)
```

### 2.3 Keyword Inclusion Rules

**Best Practices:**
- Include primary keyword: 1 occurrence (required)
- Include secondary keywords: 1-2 occurrences (optional)
- Include LSI/semantic variations: Encouraged
- Natural language required: No forced keyword insertion

**Keyword Stuffing Detection:**
```
primary_keyword_count = count_occurrences(description, primary_keyword)
total_words = word_count(description)
keyword_density = (primary_keyword_count * len(primary_keyword)) / total_words

IF primary_keyword_count > 2 THEN
  score = 40
  warning = "Keyword appears too frequently (stuffing detected)"
ELSE IF keyword_density > 0.10 THEN
  score = 60
  warning = "Keyword density too high"
ELSE IF primary_keyword_count == 1 THEN
  score = 100
ELSE IF primary_keyword_count == 0 THEN
  score = 50
  warning = "Primary keyword not found in description"
END IF
```

### 2.4 Structured Data Interaction

**Rich Snippet Enhancement:**
- Review stars: Reserve ~40 characters for star display
- Date stamps: Reserve ~25 characters
- Pricing: Reserve ~30 characters
- Author bylines: Reserve ~35 characters

**Adjusted Length When Rich Snippets Present:**
```
IF has_review_schema THEN
  recommended_length = 120 chars
ELSE IF has_product_schema THEN
  recommended_length = 130 chars
ELSE
  recommended_length = 155 chars
END IF
```

### 2.5 Mobile Truncation Handling

**Strategy:**
1. First 120 characters: Primary value proposition + primary keyword + CTA
2. Characters 121-158: Supporting details, secondary keywords, brand

**Mobile-First Validation:**
```
first_120_chars = description[0:120]

IF contains(first_120_chars, primary_keyword) AND
   contains_cta(first_120_chars) THEN
  mobile_score = 100
ELSE IF contains(first_120_chars, primary_keyword) THEN
  mobile_score = 80
ELSE
  mobile_score = 50
  warning = "Primary keyword/CTA not in first 120 chars (mobile-visible portion)"
END IF
```

### 2.6 Template Patterns for Content Types

**Blog Post Template:**
```
"[Action Verb] [Primary Keyword] with [Unique Value Prop]. [Supporting Detail]. [CTA with urgency/value]."

Example: "Learn advanced Python techniques with our comprehensive 2026 guide. Step-by-step tutorials, code examples, and expert tips. Start coding today."
(152 chars)
```

**Product Page Template:**
```
"[Product Name]: [Key Benefit] for [Target Audience]. [Unique Feature/Differentiator]. [Price/Offer if competitive]. [CTA]."

Example: "ProSEO Tool: Automated content optimization for digital marketers. AI-powered analysis, real-time suggestions. Try free for 14 days."
(143 chars)
```

**Service Page Template:**
```
"[Service] in [Location/Niche]. [Key Benefit 1] and [Key Benefit 2]. [Social Proof/Authority]. [CTA]."

Example: "SEO Consulting in Chicago. Increase organic traffic and revenue with proven strategies. 500+ clients served. Get your free audit."
(139 chars)
```

**How-To/Tutorial Template:**
```
"How to [Achieve Result]: [Number] [Power Word] [Method/Steps]. [Difficulty Level]. [Time/Resource Requirement]. [CTA]."

Example: "How to Rank on Google: 7 proven SEO strategies for beginners. Easy to implement in 30 days. Download our free checklist."
(128 chars)
```

### 2.7 Complete Meta Description Scoring Formula

```
total_description_score = (
  (length_score * 0.30) +
  (keyword_score * 0.25) +
  (cta_score * 0.20) +
  (mobile_optimization_score * 0.15) +
  (uniqueness_score * 0.10)
) * 100

WHERE:
  length_score = length validation (0-1)
  keyword_score = keyword inclusion without stuffing (0-1)
  cta_score = CTA detection score (0-1)
  mobile_optimization_score = first 120 chars quality (0-1)
  uniqueness_score = duplicate detection (0-1)
```

### 2.8 Examples

| Meta Description | Length | Score | Analysis |
|-----------------|--------|-------|----------|
| **GOOD:** "Discover 10 proven SEO strategies to rank higher in 2026. Step-by-step guide with real examples. Start optimizing your content today for better results." | 156 | 95/100 | Perfect length, CTA, keyword front-loaded, action verb |
| **GOOD:** "Learn Python programming from scratch with our beginner-friendly tutorial. Free exercises, video lessons, and downloadable code samples included." | 148 | 92/100 | Strong value prop, keywords natural, clear CTA |
| **BAD:** "SEO tips and tricks for your website. SEO services. Learn SEO today. SEO marketing strategies for SEO optimization and better SEO results." | 138 | 25/100 | Severe keyword stuffing, no clear value, robotic |
| **BAD:** "Welcome to our site." | 21 | 15/100 | Far too short, no keywords, no value proposition |
| **BAD:** "In this comprehensive guide we will explore various techniques and methodologies for search engine optimization including on-page factors, technical considerations, content strategy development, and link building approaches that have proven effective." | 267 | 40/100 | Far too long (will truncate), overly formal, no CTA |

---

## 3. Heading Hierarchy Validation

### 3.1 H1 Requirements

**Rules:**
- **Quantity:** Exactly 1 H1 per page (critical requirement)
- **Length:** 30-60 characters optimal (max 70 characters)
- **Keyword:** Must contain primary keyword
- **Position:** Should appear before main content
- **Uniqueness:** Different from title tag (but related)

**H1 Validation Logic:**
```
h1_count = count_h1_tags(page)
h1_length = char_length(h1_text)
h1_contains_keyword = contains(h1_text, primary_keyword)

IF h1_count == 0 THEN
  score = 0
  error = "No H1 tag found"
ELSE IF h1_count > 1 THEN
  score = 30
  error = "Multiple H1 tags detected ({h1_count} found)"
ELSE IF h1_count == 1 THEN
  IF h1_length < 20 THEN
    length_score = 60
    warning = "H1 too short"
  ELSE IF h1_length >= 20 AND h1_length <= 70 THEN
    length_score = 100
  ELSE
    length_score = 70
    warning = "H1 too long (over 70 chars)"
  END IF

  IF h1_contains_keyword THEN
    keyword_score = 100
  ELSE
    keyword_score = 40
    warning = "H1 missing primary keyword"
  END IF

  score = (length_score * 0.4) + (keyword_score * 0.6)
END IF
```

**H1 vs Title Relationship:**
```
IF h1_text == title_text THEN
  warning = "H1 identical to title tag; consider differentiation for variety"
  score_penalty = -10
ELSE IF similarity(h1_text, title_text) > 0.8 THEN
  warning = "H1 very similar to title tag"
  score_penalty = -5
ELSE
  score_penalty = 0
END IF
```

### 3.2 H2-H6 Nesting Rules

**Critical Rule:** Do not skip heading levels when descending the hierarchy.

**Valid Sequences:**
- H1 → H2 → H3 → H4 (correct)
- H1 → H2 → H2 → H3 (correct)
- H1 → H2 → H4 (INVALID - skipped H3)

**Heading Level Validation Algorithm:**
```
function validate_heading_hierarchy(headings[]):
  current_level = 0
  errors = []

  for each heading in headings:
    heading_level = extract_level(heading)  # Returns 1-6

    IF heading_level == 1 AND current_level > 0 THEN
      errors.append("Multiple H1 tags detected")
    END IF

    IF heading_level > current_level + 1 THEN
      errors.append("Skipped heading level: H{current_level} to H{heading_level}")
    END IF

    current_level = heading_level
  end for

  IF len(errors) == 0 THEN
    return score = 100
  ELSE
    return score = MAX(0, 100 - (len(errors) * 20))
  END IF
end function
```

**Note:** When closing subsections, it IS acceptable to skip levels upward (H4 → H2 is valid when starting new section).

### 3.3 Keyword Distribution Across Headings

**Best Practices:**
- H1: Primary keyword (required)
- H2s: Primary keyword variations, secondary keywords
- H3-H6: Long-tail keywords, LSI terms, question-based phrases

**Keyword Distribution Scoring:**
```
h1_has_primary = contains(h1, primary_keyword)
h2_keyword_count = count_h2s_with_keywords(h2_list, [primary_variations, secondary_keywords])
total_h2_count = count(h2_list)

IF h1_has_primary THEN
  h1_score = 100
ELSE
  h1_score = 0
END IF

IF total_h2_count > 0 THEN
  h2_keyword_ratio = h2_keyword_count / total_h2_count

  IF h2_keyword_ratio >= 0.5 AND h2_keyword_ratio <= 0.8 THEN
    h2_score = 100
  ELSE IF h2_keyword_ratio > 0.8 THEN
    h2_score = 70
    warning = "Possible keyword stuffing in H2 tags"
  ELSE
    h2_score = 60
    warning = "Low keyword presence in H2 tags"
  END IF
ELSE
  h2_score = 50
  warning = "No H2 tags found"
END IF

keyword_distribution_score = (h1_score * 0.6) + (h2_score * 0.4)
```

### 3.4 Heading Density Recommendations

**Ideal Density:**
- One heading per 200-300 words of content
- Minimum: One heading per 500 words
- H2 tags: 3-6 per article (for 1500-2000 word content)

**Density Validation:**
```
word_count = count_words(page_content)
heading_count = count_all_headings(page)  # H2-H6
heading_density = word_count / heading_count

IF heading_density < 150 THEN
  score = 60
  warning = "Too many headings; content may feel fragmented"
ELSE IF heading_density >= 150 AND heading_density <= 400 THEN
  score = 100
ELSE IF heading_density > 400 AND heading_density <= 600 THEN
  score = 80
  warning = "Consider adding more headings for readability"
ELSE
  score = 50
  warning = "Too few headings; add structure to improve scannability"
END IF
```

### 3.5 Accessibility Compliance (WCAG)

**WCAG 2.1 AA Requirements:**
- **1.3.1 Info and Relationships (Level A):** Headings must not be in reverse order
- **2.4.6 Headings and Labels (Level AA):** Headings must describe topic or purpose
- **2.4.10 Section Headings (Level AAA):** Use section headings to organize content

**Accessibility Validation:**
```
function validate_wcag_compliance(headings[]):
  issues = []

  # Check for reverse order (WCAG 1.3.1)
  for i in range(1, len(headings)):
    if heading_level(headings[i-1]) > heading_level(headings[i]) + 1:
      issues.append("Reverse heading order violates WCAG 1.3.1")
    end if
  end for

  # Check for descriptive headings (WCAG 2.4.6)
  for heading in headings:
    if char_length(heading) < 10 OR is_generic(heading):
      issues.append("Non-descriptive heading: '{heading}' violates WCAG 2.4.6")
    end if
  end for

  # Check for adequate section headings (WCAG 2.4.10 - AAA)
  if word_count / heading_count > 500:
    issues.append("Insufficient section headings for content length (WCAG 2.4.10)")
  end if

  return issues
end function

accessibility_score = MAX(0, 100 - (len(issues) * 15))
```

**Generic Heading Detection:**
```
generic_headings = ["Introduction", "Conclusion", "Overview", "Summary", "Click here", "Read more"]

IF heading_text IN generic_headings THEN
  is_generic = true
  recommendation = "Make heading more descriptive and keyword-specific"
END IF
```

### 3.6 Complete Heading Hierarchy Scoring Formula

```
total_heading_score = (
  (h1_requirements_score * 0.30) +
  (nesting_validation_score * 0.25) +
  (keyword_distribution_score * 0.20) +
  (heading_density_score * 0.15) +
  (accessibility_score * 0.10)
) * 100

WHERE:
  h1_requirements_score = H1 validation (0-1)
  nesting_validation_score = hierarchy validation (0-1)
  keyword_distribution_score = keyword usage (0-1)
  heading_density_score = density check (0-1)
  accessibility_score = WCAG compliance (0-1)
```

### 3.7 Examples

**GOOD Example:**
```html
<h1>Complete Guide to Python Programming for Beginners</h1>

<h2>What is Python?</h2>
<h3>History of Python</h3>
<h3>Why Learn Python in 2026?</h3>

<h2>Python Installation and Setup</h2>
<h3>Installing Python on Windows</h3>
<h3>Installing Python on Mac</h3>
<h3>Installing Python on Linux</h3>

<h2>Python Basics for Beginners</h2>
<h3>Variables and Data Types</h3>
<h4>Strings</h4>
<h4>Integers</h4>
<h4>Lists</h4>
<h3>Control Flow</h3>
<h4>If Statements</h4>
<h4>For Loops</h4>
```
**Score:** 98/100 - Perfect hierarchy, descriptive headings, keyword distribution, no skipped levels

**BAD Example:**
```html
<h1>Python</h1>
<h1>Welcome to Python Tutorial</h1>  <!-- Multiple H1s -->

<h4>Getting Started</h4>  <!-- Skipped H2 and H3 -->
<h2>Installation</h2>  <!-- After H4 - confusing order -->
<h3>Click Here</h3>  <!-- Generic, non-descriptive -->
<h2>Python Python Python Tutorial</h2>  <!-- Keyword stuffing -->
```
**Score:** 25/100 - Multiple H1s, skipped levels, generic headings, keyword stuffing

---

## 4. URL Structure Optimization

### 4.1 Slug Generation Rules

**Optimal Slug Length:**
- **Ideal:** 3-5 words (25-30 characters)
- **Maximum:** 60 characters
- **Word Count:** 2-6 words

**Slug Scoring:**
```
word_count = count_words(slug)
char_length = len(slug)

IF word_count < 2 THEN
  score = 60
  warning = "Slug too short; add descriptive keywords"
ELSE IF word_count >= 2 AND word_count <= 6 THEN
  IF char_length <= 30 THEN
    score = 100
  ELSE IF char_length <= 60 THEN
    score = 90
  ELSE
    score = 70
    warning = "Slug exceeds recommended 60 character limit"
  END IF
ELSE
  score = 50
  warning = "Slug too long; remove unnecessary words"
END IF
```

### 4.2 Keyword Inclusion

**Rules:**
- Include primary keyword in slug (required)
- Include secondary keyword if natural (optional)
- Avoid keyword stuffing

**Keyword Validation:**
```
IF contains(slug, primary_keyword) THEN
  keyword_score = 100
ELSE IF contains_variation(slug, primary_keyword) THEN
  keyword_score = 85
  info = "Slug contains keyword variation"
ELSE
  keyword_score = 40
  warning = "Primary keyword not found in URL slug"
END IF
```

### 4.3 Stop Words Removal

**Common Stop Words to Remove:**
- Articles: a, an, the
- Conjunctions: and, or, but
- Prepositions: of, in, on, at, to, for, with, from

**Exception:** Keep stop words if they're part of a brand name or significantly impact meaning.

**Stop Word Scoring:**
```
stop_words = ["a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for"]
slug_words = split(slug, "-")
stop_word_count = count_matches(slug_words, stop_words)
total_words = len(slug_words)

IF stop_word_count == 0 THEN
  score = 100
ELSE IF stop_word_count / total_words < 0.3 THEN
  score = 85
  info = "Consider removing stop words: {stop_words_found}"
ELSE
  score = 60
  warning = "Remove excessive stop words for cleaner URL"
END IF
```

### 4.4 Special Character Handling

**Allowed Characters:**
- Lowercase letters (a-z)
- Numbers (0-9)
- Hyphens (-) as word separators

**Forbidden Characters:**
- Underscores (_): Use hyphens instead
- Spaces: Convert to hyphens
- Special chars: &, %, $, @, !, etc.
- Uppercase letters: Convert to lowercase

**Character Validation:**
```
valid_pattern = /^[a-z0-9-]+$/

IF regex_match(slug, valid_pattern) THEN
  score = 100
ELSE
  score = 0
  error = "Slug contains invalid characters"

  # Provide auto-fix suggestion
  cleaned_slug = slug.lower()
  cleaned_slug = replace(cleaned_slug, "_", "-")
  cleaned_slug = replace(cleaned_slug, " ", "-")
  cleaned_slug = remove_special_chars(cleaned_slug)

  suggestion = "Suggested slug: {cleaned_slug}"
END IF
```

### 4.5 Hierarchy Reflection

**Category/Subcategory Structure:**

**Examples:**
```
GOOD: /blog/seo/on-page-optimization
GOOD: /products/software/analytics-tools
GOOD: /services/web-design/ecommerce

BAD: /page123
BAD: /blog/seo/on-page-optimization/advanced/techniques/2026  (too deep)
```

**Depth Recommendations:**
- **Homepage:** /
- **Top-level pages:** /about, /services
- **Category pages:** /blog, /products
- **Content pages:** /blog/seo-guide (2 levels)
- **Deep content:** /blog/seo/technical-guide (3 levels max)

**Hierarchy Scoring:**
```
depth = count_slashes(url_path)

IF depth <= 1 THEN
  hierarchy_score = 100  # Top-level or category
ELSE IF depth == 2 THEN
  hierarchy_score = 100  # Optimal depth
ELSE IF depth == 3 THEN
  hierarchy_score = 85  # Acceptable
  info = "URL depth acceptable but consider flattening structure"
ELSE
  hierarchy_score = 60
  warning = "URL too deep ({depth} levels); flatten hierarchy"
END IF
```

### 4.6 Date Handling in URLs

**Rule:** Avoid dates for evergreen content; include for time-sensitive content.

**Examples:**
```
EVERGREEN (no date):
  /seo-guide
  /python-tutorial

TIME-SENSITIVE (include date):
  /2026-seo-trends
  /january-2026-algorithm-update

AVOID (date in path):
  /2026/01/seo-guide  (will feel outdated next year)
```

**Date Detection:**
```
IF contains_date_pattern(slug) THEN
  IF is_evergreen_content THEN
    score = 60
    warning = "Evergreen content includes date; will appear outdated"
  ELSE
    score = 100
    info = "Date appropriate for time-sensitive content"
  END IF
ELSE
  score = 100
END IF
```

### 4.7 Complete URL Structure Scoring Formula

```
total_url_score = (
  (length_score * 0.20) +
  (keyword_score * 0.30) +
  (character_validity_score * 0.20) +
  (hierarchy_score * 0.15) +
  (stop_word_score * 0.10) +
  (date_handling_score * 0.05)
) * 100

WHERE:
  length_score = slug length validation (0-1)
  keyword_score = keyword inclusion (0-1)
  character_validity_score = valid characters only (0-1)
  hierarchy_score = URL depth validation (0-1)
  stop_word_score = stop word removal (0-1)
  date_handling_score = date appropriateness (0-1)
```

### 4.8 Examples

| URL | Score | Analysis |
|-----|-------|----------|
| `/blog/seo-guide-2026` | 98/100 | Perfect: 3 words, keyword-rich, no stop words, appropriate date |
| `/products/analytics-tools` | 100/100 | Excellent: clean, hierarchical, descriptive |
| `/blog/learn-how-to-do-seo-optimization-for-your-website` | 55/100 | Too long (9 words), excessive stop words |
| `/page?id=123&category=seo` | 20/100 | Parameters instead of clean slug; non-descriptive |
| `/Blog/SEO_Guide` | 40/100 | Uppercase, underscores instead of hyphens |

---

## 5. Internal Linking Strategy

### 5.1 Contextual Linking Rules

**Definition:** Links embedded within the body content (not navigation/footer).

**Placement Best Practices:**
- **Top 30% of page:** Links here receive more weight from Google
- **Within relevant context:** Surrounded by related content
- **Natural reading flow:** Don't disrupt user experience

**Contextual Link Scoring:**
```
link_position = calculate_position_percentage(link, page_content)

IF link_position <= 0.30 THEN
  position_score = 100
  info = "Link in high-value position (top 30% of page)"
ELSE IF link_position <= 0.70 THEN
  position_score = 85
ELSE
  position_score = 70
  info = "Consider moving important links higher on page"
END IF
```

**Contextual Relevance:**
```
surrounding_text = extract_surrounding_text(link, words_before=50, words_after=50)
link_target_content = extract_content(link.href)
relevance_score = calculate_semantic_similarity(surrounding_text, link_target_content)

IF relevance_score > 0.7 THEN
  context_score = 100
ELSE IF relevance_score > 0.4 THEN
  context_score = 75
  info = "Link contextually relevant but could be stronger"
ELSE
  context_score = 50
  warning = "Link not strongly related to surrounding content"
END IF
```

### 5.2 Anchor Text Optimization

**Anchor Text Types:**
1. **Exact Match:** "SEO guide" linking to SEO guide page
2. **Partial Match:** "complete SEO resource" linking to SEO guide
3. **Branded:** "BrandName SEO tool"
4. **Generic:** "click here", "read more" (avoid)
5. **Naked URL:** "https://example.com/seo-guide" (avoid in content)

**Optimal Distribution:**
| Anchor Type | Percentage | Purpose |
|------------|-----------|----------|
| Exact Match | 10-20% | Direct keyword relevance |
| Partial Match | 40-50% | Natural variation |
| Branded | 20-30% | Brand authority |
| Generic | <10% | Minimize usage |
| Naked URL | <5% | Avoid in body content |

**Anchor Text Validation:**
```
function validate_anchor_text(anchor_text, target_url):
  target_keyword = extract_primary_keyword(target_url)

  # Classify anchor type
  IF anchor_text == target_keyword THEN
    type = "exact_match"
  ELSE IF contains(anchor_text, target_keyword) THEN
    type = "partial_match"
  ELSE IF is_brand_name(anchor_text) THEN
    type = "branded"
  ELSE IF anchor_text IN ["click here", "read more", "this", "here"] THEN
    type = "generic"
    score = 30
    warning = "Avoid generic anchor text; use descriptive keywords"
  ELSE IF is_url(anchor_text) THEN
    type = "naked_url"
    score = 40
    warning = "Use descriptive anchor text instead of naked URL"
  ELSE
    type = "branded_or_misc"
  END IF

  # Score based on type and context
  IF type IN ["exact_match", "partial_match"] THEN
    score = 100
  ELSE IF type == "branded" THEN
    score = 85
  END IF

  return score, type
end function
```

**Over-Optimization Detection:**
```
exact_match_percentage = (exact_match_count / total_internal_links) * 100

IF exact_match_percentage > 30 THEN
  score = 50
  warning = "Over-optimization: {exact_match_percentage}% exact match anchors; vary anchor text"
ELSE IF exact_match_percentage >= 10 AND exact_match_percentage <= 20 THEN
  score = 100
ELSE IF exact_match_percentage < 10 THEN
  score = 80
  info = "Consider adding more exact match anchors for keyword relevance"
END IF
```

### 5.3 Link Density Recommendations

**Definition:** Ratio of linked text to total text on page.

**Optimal Ratios:**
- **Ideal:** 2-5 internal links per 500 words
- **Minimum:** 1 link per 500 words
- **Maximum:** 10 links per 500 words

**Link Density Scoring:**
```
word_count = count_words(page_content)
internal_link_count = count_internal_links(page)
links_per_500_words = (internal_link_count / word_count) * 500

IF links_per_500_words < 1 THEN
  score = 60
  warning = "Add more internal links to distribute link equity"
ELSE IF links_per_500_words >= 2 AND links_per_500_words <= 5 THEN
  score = 100
ELSE IF links_per_500_words > 5 AND links_per_500_words <= 10 THEN
  score = 80
  info = "Link density acceptable but monitor for over-linking"
ELSE
  score = 40
  warning = "Excessive internal linking; reduce to avoid spam appearance"
END IF
```

### 5.4 Hub/Spoke Content Architecture

**Definition:** Hub pages (pillar content) link to related spoke pages (supporting content).

**Architecture Rules:**
1. **Hub page:** Comprehensive guide (2000+ words)
2. **Spoke pages:** Deep-dive subtopics (1000-1500 words)
3. **Hub links to all spokes:** Bidirectional linking required
4. **Spokes link back to hub:** Establish topical authority

**Hub/Spoke Validation:**
```
IF is_hub_page THEN
  spoke_pages = identify_related_spokes(page)
  outbound_links_to_spokes = count_links_to_spokes(page, spoke_pages)

  IF outbound_links_to_spokes == len(spoke_pages) THEN
    hub_score = 100
  ELSE
    hub_score = (outbound_links_to_spokes / len(spoke_pages)) * 100
    warning = "Hub missing links to {len(spoke_pages) - outbound_links_to_spokes} spoke pages"
  END IF
ELSE IF is_spoke_page THEN
  hub_page = identify_hub(page)

  IF links_to(page, hub_page) THEN
    spoke_score = 100
  ELSE
    spoke_score = 40
    warning = "Spoke page should link back to hub: {hub_page.title}"
  END IF
END IF
```

### 5.5 Orphan Page Detection

**Definition:** Pages with no internal links pointing to them.

**Validation:**
```
function detect_orphan_pages(site):
  all_pages = get_all_pages(site)
  orphan_pages = []

  for page in all_pages:
    inbound_links = count_internal_inbound_links(page)

    IF inbound_links == 0 THEN
      orphan_pages.append(page)
    END IF
  end for

  IF len(orphan_pages) > 0 THEN
    score = MAX(0, 100 - (len(orphan_pages) * 5))
    error = "Found {len(orphan_pages)} orphan pages with no internal links"
  ELSE
    score = 100
  END IF

  return score, orphan_pages
end function
```

**Fix Recommendations:**
- Add links from related content
- Include in navigation or sidebar
- Link from hub pages
- Add to sitemap if not already present

### 5.6 Link Equity Distribution

**Concept:** Distribute link authority from high-authority pages to pages needing a boost.

**Priority Linking Strategy:**
```
function prioritize_internal_links(pages):
  # Calculate page authority (simplified)
  for page in pages:
    page.authority = calculate_page_authority(page)  # Based on backlinks, traffic, etc.
  end for

  # Identify high-authority pages
  high_authority_pages = filter(pages, authority > 70)

  # Identify pages needing boost
  low_performing_pages = filter(pages, authority < 40 AND has_business_value == true)

  # Recommend links
  recommendations = []
  for low_page in low_performing_pages:
    relevant_high_pages = find_topically_relevant(low_page, high_authority_pages)

    for high_page in relevant_high_pages:
      IF not links_to(high_page, low_page) THEN
        recommendations.append({
          "from": high_page.url,
          "to": low_page.url,
          "anchor_suggestion": generate_anchor_text(low_page),
          "context_suggestion": suggest_context(high_page, low_page)
        })
      END IF
    end for
  end for

  return recommendations
end function
```

### 5.7 Complete Internal Linking Scoring Formula

```
total_internal_linking_score = (
  (contextual_placement_score * 0.20) +
  (anchor_text_score * 0.25) +
  (link_density_score * 0.20) +
  (architecture_score * 0.20) +
  (orphan_detection_score * 0.10) +
  (equity_distribution_score * 0.05)
) * 100

WHERE:
  contextual_placement_score = position and relevance (0-1)
  anchor_text_score = anchor text optimization (0-1)
  link_density_score = links per content ratio (0-1)
  architecture_score = hub/spoke implementation (0-1)
  orphan_detection_score = no orphan pages (0-1)
  equity_distribution_score = strategic linking (0-1)
```

### 5.8 Examples

**GOOD Internal Linking:**
```html
<p>When optimizing your content, understanding <a href="/seo/on-page-optimization">on-page SEO factors</a>
is crucial for ranking success. Start by mastering <a href="/seo/title-tag-optimization">title tag optimization</a>,
which remains one of the most important ranking factors in 2026.</p>

<p>For a complete overview, see our <a href="/seo/complete-guide">comprehensive SEO guide</a>.</p>
```
**Score:** 95/100 - Descriptive anchors, contextually relevant, varied anchor text

**BAD Internal Linking:**
```html
<p>SEO is important. <a href="/page1">Click here</a> to learn more about SEO.
We also have <a href="/seo-guide">SEO guide</a>, <a href="/seo-tips">SEO tips</a>,
and <a href="/seo-checklist">SEO checklist</a> for SEO optimization.</p>

<p>For more info, go to <a href="https://example.com/seo">https://example.com/seo</a>.</p>
```
**Score:** 35/100 - Generic "click here", keyword stuffing, naked URL, poor context

---

## 6. Image Optimization

### 6.1 Alt Text Generation Rules

**Purpose:**
1. Accessibility: Describe image for screen readers
2. SEO: Provide context for search engines
3. Fallback: Display when image fails to load

**Core Requirements:**

| Requirement | Specification | Validation |
|------------|--------------|------------|
| **Length** | 80-125 characters max | CRITICAL |
| **Word Count** | 3-15 words | Recommended |
| **Keyword Inclusion** | 1 occurrence max | Optional but beneficial |
| **Descriptive Quality** | Specific, actionable description | Required |
| **Grammar** | Complete sentences or phrases | Recommended |

**Alt Text Validation Logic:**
```
char_length = len(alt_text)
word_count = count_words(alt_text)
keyword_count = count_occurrences(alt_text, primary_keyword)

# Length validation
IF char_length == 0 THEN
  score = 0
  error = "Missing alt text (accessibility violation)"
ELSE IF char_length > 125 THEN
  score = 60
  warning = "Alt text exceeds 125 character limit; may be truncated by screen readers"
ELSE IF char_length >= 80 AND char_length <= 125 THEN
  length_score = 100
ELSE IF char_length < 80 THEN
  length_score = 85
  info = "Alt text acceptable but could be more descriptive"
END IF

# Word count validation
IF word_count < 3 THEN
  word_score = 70
  info = "Alt text very brief; add more context"
ELSE IF word_count >= 3 AND word_count <= 15 THEN
  word_score = 100
ELSE
  word_score = 80
  info = "Alt text wordy; consider condensing"
END IF

# Keyword validation
IF keyword_count == 1 THEN
  keyword_score = 100
ELSE IF keyword_count == 0 THEN
  keyword_score = 80
  info = "Consider including relevant keyword naturally"
ELSE IF keyword_count > 1 THEN
  keyword_score = 40
  warning = "Keyword stuffing detected in alt text"
END IF

final_score = (length_score * 0.4) + (word_score * 0.3) + (keyword_score * 0.3)
```

### 6.2 Descriptive Requirements

**Good Alt Text Characteristics:**
- Describes image content specifically
- Includes relevant context (e.g., "Graph showing...", "Screenshot of...")
- Mentions key details (colors, numbers, actions, people)
- Avoids redundant phrases like "image of" or "picture of"

**Quality Scoring:**
```
redundant_phrases = ["image of", "picture of", "photo of", "graphic of"]
generic_terms = ["image", "photo", "graphic", "icon"]

IF starts_with_any(alt_text, redundant_phrases) THEN
  quality_score = 70
  warning = "Remove redundant phrase: '{detected_phrase}'"
END IF

IF alt_text IN generic_terms THEN
  quality_score = 20
  error = "Alt text too generic; describe what the image shows"
END IF

# Check for specificity
IF contains_numbers_or_specific_details(alt_text) THEN
  quality_score += 15
  info = "Good: includes specific details"
END IF

IF describes_action_or_context(alt_text) THEN
  quality_score += 15
  info = "Good: describes context or action"
END IF
```

### 6.3 Decorative Image Handling

**Rule:** Decorative images (purely aesthetic, no informational value) should have empty alt attributes.

**Examples:**
- Background patterns
- Spacer images
- Purely decorative icons
- Separator lines

**Validation:**
```
IF image_is_decorative THEN
  IF alt_text == "" OR alt_text == null THEN
    score = 100
    info = "Correct: decorative image has empty alt attribute"
  ELSE
    score = 70
    warning = "Decorative image should have empty alt attribute: alt=''"
  END IF
ELSE
  # Content image must have alt text
  IF alt_text == "" OR alt_text == null THEN
    score = 0
    error = "Content image missing alt text (accessibility violation)"
  END IF
END IF
```

### 6.4 File Naming Conventions

**Rules:**
- Use descriptive names with keywords
- Lowercase letters only
- Hyphens as word separators (not underscores or spaces)
- No special characters
- Keep under 60 characters

**File Name Validation:**
```
filename = extract_filename(image_src)  # Without extension
valid_pattern = /^[a-z0-9-]+$/

IF regex_match(filename, valid_pattern) THEN
  IF len(filename) <= 60 THEN
    IF contains_keyword(filename) THEN
      score = 100
    ELSE
      score = 85
      info = "Consider including relevant keyword in filename"
    END IF
  ELSE
    score = 70
    warning = "Filename too long; shorten to under 60 characters"
  END IF
ELSE
  score = 40
  error = "Invalid filename format; use lowercase letters, numbers, and hyphens only"

  # Suggest correction
  cleaned_filename = filename.lower()
  cleaned_filename = replace(cleaned_filename, " ", "-")
  cleaned_filename = replace(cleaned_filename, "_", "-")
  cleaned_filename = remove_special_chars(cleaned_filename)

  suggestion = "Suggested filename: {cleaned_filename}.{extension}"
END IF
```

**Examples:**
```
GOOD:
  seo-optimization-checklist.png
  python-data-structures-diagram.jpg
  mobile-responsive-design-example.webp

BAD:
  IMG_1234.jpg (non-descriptive)
  SEO_Optimization.PNG (uppercase, underscores)
  my image file (2026).png (spaces, special chars)
```

### 6.5 Lazy Loading Considerations

**Rule:** Implement lazy loading for below-the-fold images to improve page speed.

**Exceptions:** Do NOT lazy load:
- Above-the-fold images
- Logo images
- Critical first-screen content

**Implementation:**
```html
<!-- Above-the-fold: NO lazy loading -->
<img src="hero-image.jpg" alt="SEO optimization dashboard screenshot" />

<!-- Below-the-fold: USE lazy loading -->
<img src="section-image.jpg" alt="Graph showing traffic increase" loading="lazy" />
```

**Validation:**
```
IF image_position <= 0.6 THEN  # Top 60% of page
  IF has_lazy_loading(image) THEN
    score = 70
    warning = "Remove lazy loading from above-the-fold image for faster LCP"
  ELSE
    score = 100
  END IF
ELSE
  IF has_lazy_loading(image) THEN
    score = 100
    info = "Good: lazy loading improves page speed"
  ELSE
    score = 80
    info = "Consider adding lazy loading to below-fold images"
  END IF
END IF
```

### 6.6 Caption Optimization

**When to Use Captions:**
- Data visualizations (graphs, charts)
- Screenshots requiring context
- Process diagrams
- Infographics

**Caption Best Practices:**
- Complement alt text (don't duplicate)
- Include additional context or insights
- Can be longer than alt text
- May include links or calls-to-action

**Example:**
```html
<figure>
  <img src="seo-traffic-growth.png"
       alt="Line graph showing 150% organic traffic increase from January to December 2026">
  <figcaption>
    Our clients saw an average 150% increase in organic traffic after implementing
    our <a href="/seo-strategy">comprehensive SEO strategy</a>. Data from 500+ campaigns.
  </figcaption>
</figure>
```

### 6.7 Complete Image Optimization Scoring Formula

```
total_image_score = (
  (alt_text_score * 0.40) +
  (filename_score * 0.20) +
  (decorative_handling_score * 0.15) +
  (lazy_loading_score * 0.15) +
  (caption_usage_score * 0.10)
) * 100

WHERE:
  alt_text_score = alt text validation (0-1)
  filename_score = filename conventions (0-1)
  decorative_handling_score = correct empty alt for decorative (0-1)
  lazy_loading_score = appropriate lazy loading (0-1)
  caption_usage_score = caption optimization (0-1)
```

### 6.8 Examples

| Image Context | Alt Text | Filename | Score | Analysis |
|--------------|---------|----------|-------|----------|
| Screenshot of analytics dashboard | "Google Analytics dashboard showing 45,000 monthly visitors and 3.2% conversion rate" | `google-analytics-dashboard-metrics.png` | 98/100 | Excellent: specific, descriptive, keyword-rich filename |
| Decorative background pattern | `` (empty) | `background-pattern.png` | 100/100 | Perfect: correctly identified as decorative |
| Product photo | "Red wireless headphones" | `red-wireless-headphones-product.jpg` | 90/100 | Good: descriptive but could add brand/model |
| Team photo | "Photo of our team" | `IMG_5678.jpg` | 35/100 | Poor: generic alt text, non-descriptive filename |
| Complex infographic | "Infographic about SEO SEO SEO tips and SEO strategies for SEO optimization" | `seo-infographic-2026.png` | 25/100 | Bad: keyword stuffing in alt text |

---

## 7. Schema Markup Recommendations

### 7.1 Article Schema Requirements

**Use Cases:**
- Blog posts
- News articles
- Editorial content
- Tutorials and guides

**Required Properties:**
- `@type`: "Article", "BlogPosting", or "NewsArticle"
- `headline`: Article title (max 110 characters)
- `image`: Representative image URL
- `datePublished`: ISO 8601 format
- `author`: Author information

**Recommended Properties:**
- `dateModified`: Last update date
- `description`: Article summary
- `publisher`: Organization information with logo
- `mainEntityOfPage`: Canonical URL

**Implementation Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Complete On-Page SEO Guide for 2026",
  "image": "https://example.com/images/seo-guide-featured.jpg",
  "author": {
    "@type": "Person",
    "name": "Jane Smith",
    "url": "https://example.com/author/jane-smith"
  },
  "publisher": {
    "@type": "Organization",
    "name": "SEO Experts Inc",
    "logo": {
      "@type": "ImageObject",
      "url": "https://example.com/logo.png"
    }
  },
  "datePublished": "2026-01-15T08:00:00+00:00",
  "dateModified": "2026-01-16T10:30:00+00:00",
  "description": "Learn proven on-page SEO strategies to improve your search rankings in 2026.",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://example.com/seo/on-page-guide"
  }
}
```

**Validation:**
```
required_fields = ["@type", "headline", "image", "datePublished", "author"]
missing_fields = []

FOR field IN required_fields:
  IF NOT exists(schema, field) THEN
    missing_fields.append(field)
  END IF
END FOR

IF len(missing_fields) > 0 THEN
  score = MAX(0, 100 - (len(missing_fields) * 20))
  error = "Missing required Article schema fields: {missing_fields}"
ELSE
  score = 100
END IF

# Validate headline length
IF len(schema.headline) > 110 THEN
  score -= 10
  warning = "Headline exceeds 110 character limit for rich results"
END IF
```

### 7.2 FAQ Schema Opportunities

**Use Cases:**
- FAQ pages
- Content with Q&A format
- Customer support pages
- Product pages with common questions

**Benefits:**
- Expanded SERP presence (FAQ accordion)
- Answers featured directly in search results
- Increased click-through rates
- Voice search optimization

**Implementation Requirements:**
- Minimum 2 questions (recommended 5-10)
- Questions must be actual user questions
- Answers must directly answer the question
- Each Q&A pair must be relevant to page topic

**Implementation Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is on-page SEO?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "On-page SEO refers to optimization techniques applied directly to web pages to improve search rankings. This includes optimizing title tags, meta descriptions, headings, content quality, internal links, and images."
      }
    },
    {
      "@type": "Question",
      "name": "How long should a meta description be in 2026?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "The optimal meta description length is 155-160 characters for desktop and 120 characters for mobile. Front-load important information in the first 120 characters to ensure visibility on all devices."
      }
    }
  ]
}
```

**Validation:**
```
question_count = len(schema.mainEntity)

IF question_count < 2 THEN
  score = 50
  warning = "Add at least 2 questions for FAQ schema (currently {question_count})"
ELSE IF question_count >= 2 AND question_count <= 4 THEN
  score = 85
  info = "Consider adding more questions for better coverage"
ELSE IF question_count >= 5 AND question_count <= 15 THEN
  score = 100
ELSE
  score = 90
  info = "Excessive questions; ensure all are high-value"
END IF

# Validate answer quality
FOR qa_pair IN schema.mainEntity:
  answer_length = len(qa_pair.acceptedAnswer.text)

  IF answer_length < 40 THEN
    score -= 10
    warning = "Answer too short: '{qa_pair.name}'"
  ELSE IF answer_length > 500 THEN
    score -= 5
    info = "Consider condensing long answer: '{qa_pair.name}'"
  END IF
END FOR
```

**Important 2026 Note:** Google deprecated FAQ rich results for non-authoritative sites. Prioritize for:
- Government/educational sites
- Well-established brands
- Medical/health sites

### 7.3 HowTo Schema Patterns

**Use Cases:**
- Step-by-step tutorials
- Recipe instructions
- DIY guides
- Process documentation

**Benefits:**
- Step-by-step rich results in search
- Featured snippet opportunities
- Visual step indicators
- Mobile-friendly presentation

**Implementation Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "name": "How to Optimize Title Tags for SEO",
  "description": "Learn how to create effective title tags that improve click-through rates and search rankings.",
  "image": "https://example.com/images/title-tag-optimization.jpg",
  "totalTime": "PT15M",
  "estimatedCost": {
    "@type": "MonetaryAmount",
    "currency": "USD",
    "value": "0"
  },
  "tool": [
    {
      "@type": "HowToTool",
      "name": "SEO audit tool"
    }
  ],
  "step": [
    {
      "@type": "HowToStep",
      "name": "Identify target keyword",
      "text": "Research and select the primary keyword for your page using keyword research tools.",
      "image": "https://example.com/images/step1.jpg",
      "url": "https://example.com/how-to-optimize-title-tags#step1"
    },
    {
      "@type": "HowToStep",
      "name": "Write compelling title",
      "text": "Create a title that includes your keyword in the first 3 words and stays under 60 characters.",
      "image": "https://example.com/images/step2.jpg",
      "url": "https://example.com/how-to-optimize-title-tags#step2"
    }
  ]
}
```

**Validation:**
```
required_fields = ["name", "step"]
step_count = len(schema.step)

IF step_count < 2 THEN
  score = 40
  error = "HowTo schema requires at least 2 steps (found {step_count})"
ELSE IF step_count >= 2 AND step_count <= 20 THEN
  score = 100
ELSE
  score = 80
  warning = "Many steps; ensure content is truly step-by-step"
END IF

# Validate each step
FOR step IN schema.step:
  IF NOT exists(step.name) OR NOT exists(step.text) THEN
    score -= 15
    error = "Step missing required 'name' or 'text' field"
  END IF

  IF exists(step.image) THEN
    score += 5
    info = "Good: step includes image for rich results"
  END IF
END FOR
```

### 7.4 Breadcrumb Markup

**Use Cases:**
- All pages with hierarchical navigation
- E-commerce category/product pages
- Blog posts with categories
- Multi-level website structures

**Benefits:**
- Breadcrumb trail in search results
- Improved site navigation understanding
- Better crawling and indexing
- Enhanced mobile SERP presentation

**Implementation Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://example.com/"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "SEO Guides",
      "item": "https://example.com/seo"
    },
    {
      "@type": "ListItem",
      "position": 3,
      "name": "On-Page SEO",
      "item": "https://example.com/seo/on-page-optimization"
    }
  ]
}
```

**Validation:**
```
breadcrumb_depth = len(schema.itemListElement)

IF breadcrumb_depth < 2 THEN
  score = 70
  info = "Shallow breadcrumb (only {breadcrumb_depth} levels)"
ELSE IF breadcrumb_depth >= 2 AND breadcrumb_depth <= 5 THEN
  score = 100
ELSE
  score = 80
  warning = "Deep breadcrumb ({breadcrumb_depth} levels); consider flattening"
END IF

# Validate position sequence
FOR i IN range(len(schema.itemListElement)):
  expected_position = i + 1
  actual_position = schema.itemListElement[i].position

  IF actual_position != expected_position THEN
    score = 60
    error = "Breadcrumb positions not sequential (expected {expected_position}, got {actual_position})"
  END IF
END FOR
```

### 7.5 Organization/Author Schema

**Use Cases:**
- Homepage (Organization)
- Author bio pages (Person)
- About us pages (Organization)
- Contact pages (Organization)

**Organization Schema Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "SEO Experts Inc",
  "url": "https://example.com",
  "logo": "https://example.com/logo.png",
  "description": "Leading SEO consulting and content optimization services.",
  "sameAs": [
    "https://www.facebook.com/seoexpertsinc",
    "https://twitter.com/seoexpertsinc",
    "https://www.linkedin.com/company/seoexpertsinc"
  ],
  "contactPoint": {
    "@type": "ContactPoint",
    "telephone": "+1-555-123-4567",
    "contactType": "Customer Service",
    "areaServed": "US",
    "availableLanguage": "English"
  }
}
```

**Person (Author) Schema Example:**
```json
{
  "@context": "https://schema.org",
  "@type": "Person",
  "name": "Jane Smith",
  "url": "https://example.com/author/jane-smith",
  "image": "https://example.com/authors/jane-smith.jpg",
  "jobTitle": "Senior SEO Consultant",
  "worksFor": {
    "@type": "Organization",
    "name": "SEO Experts Inc"
  },
  "sameAs": [
    "https://twitter.com/janesmith",
    "https://www.linkedin.com/in/janesmith"
  ]
}
```

### 7.6 Schema Markup Priority Scoring

**Weighted Priority by Page Type:**

| Page Type | Required Schema | Priority Score Weight |
|-----------|----------------|---------------------|
| Blog Post | Article | 100% |
| Homepage | Organization | 100% |
| FAQ Page | FAQPage | 100% |
| Tutorial | HowTo | 100% |
| Product | Product | 100% |
| All Pages | Breadcrumb | 80% |
| Author Page | Person | 90% |

**Overall Schema Score:**
```
total_schema_score = (
  (schema_presence_score * 0.40) +
  (schema_validity_score * 0.35) +
  (schema_completeness_score * 0.25)
) * 100

WHERE:
  schema_presence_score = has appropriate schema for page type (0-1)
  schema_validity_score = passes Google structured data validator (0-1)
  schema_completeness_score = includes recommended properties (0-1)
```

---

## 8. Content Structure Rules

### 8.1 Paragraph Length Guidelines

**Optimal Length:**
- **Ideal:** 1-3 sentences per paragraph
- **Maximum:** 5 sentences per paragraph
- **Character Count:** 50-150 characters per paragraph

**Why Short Paragraphs:**
- Mobile readability (wall-of-text effect)
- Improved scannability
- Better engagement and comprehension
- Lower bounce rates

**Validation Logic:**
```
paragraphs = extract_paragraphs(content)
long_paragraph_count = 0

FOR paragraph IN paragraphs:
  sentence_count = count_sentences(paragraph)
  char_count = len(paragraph)

  IF sentence_count > 5 OR char_count > 400 THEN
    long_paragraph_count += 1
  END IF
END FOR

long_paragraph_ratio = long_paragraph_count / len(paragraphs)

IF long_paragraph_ratio == 0 THEN
  score = 100
ELSE IF long_paragraph_ratio <= 0.2 THEN
  score = 85
  info = "{long_paragraph_count} paragraphs could be shorter"
ELSE IF long_paragraph_ratio <= 0.4 THEN
  score = 70
  warning = "Many long paragraphs ({long_paragraph_count}); break into smaller chunks"
ELSE
  score = 50
  warning = "Excessive long paragraphs; significantly impairs readability"
END IF
```

### 8.2 Reading Level Targets

**Optimal Reading Level:**
- **General Audience:** 7th-9th grade (Flesch-Kincaid)
- **Technical Content:** 10th-12th grade
- **Academic/Medical:** 12th+ grade

**Flesch Reading Ease Scale:**
| Score | Difficulty | Grade Level | Recommendation |
|-------|-----------|-------------|----------------|
| 90-100 | Very Easy | 5th grade | May be too simple |
| 80-89 | Easy | 6th grade | Good for broad audience |
| 70-79 | Fairly Easy | 7th grade | **IDEAL for most content** |
| 60-69 | Standard | 8-9th grade | **IDEAL for most content** |
| 50-59 | Fairly Difficult | 10-12th grade | Technical content |
| 30-49 | Difficult | College | Specialized content |
| 0-29 | Very Difficult | College grad | Academic/research |

**Flesch-Kincaid Calculation:**
```
function calculate_flesch_kincaid(text):
  total_words = count_words(text)
  total_sentences = count_sentences(text)
  total_syllables = count_syllables(text)

  avg_sentence_length = total_words / total_sentences
  avg_syllables_per_word = total_syllables / total_words

  reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
  grade_level = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59

  return reading_ease, grade_level
end function
```

**Scoring Logic:**
```
reading_ease, grade_level = calculate_flesch_kincaid(content)

IF content_type == "general" THEN
  IF reading_ease >= 60 AND reading_ease <= 80 THEN
    score = 100
  ELSE IF reading_ease >= 50 AND reading_ease < 60 THEN
    score = 85
    info = "Consider simplifying language for broader audience"
  ELSE IF reading_ease < 50 THEN
    score = 60
    warning = "Content difficult to read; simplify sentence structure and vocabulary"
  ELSE
    score = 90
    info = "Content very easy; may lack depth"
  END IF
ELSE IF content_type == "technical" THEN
  IF reading_ease >= 50 AND reading_ease <= 70 THEN
    score = 100
  ELSE
    score = 85
  END IF
END IF
```

### 8.3 Whitespace and Formatting

**Best Practices:**
- Blank line between paragraphs
- Use of bulleted/numbered lists for 3+ items
- Bold/italic for emphasis (sparingly)
- Blockquotes for citations
- Code blocks for technical content

**List Usage Validation:**
```
list_opportunities = detect_list_patterns(content)
# Patterns: "1.", "first,", "second,", etc.

IF len(list_opportunities) > 0 AND not has_lists(content) THEN
  score = 70
  warning = "Consider converting enumerations to proper lists for better scannability"
ELSE IF has_lists(content) THEN
  score = 100
  info = "Good use of lists for structured information"
ELSE
  score = 85
END IF
```

**Emphasis Usage:**
```
bold_count = count_bold_elements(content)
italic_count = count_italic_elements(content)
word_count = count_words(content)

emphasis_ratio = (bold_count + italic_count) / word_count

IF emphasis_ratio > 0.05 THEN
  score = 70
  warning = "Overuse of bold/italic formatting; reduces impact"
ELSE IF emphasis_ratio >= 0.01 AND emphasis_ratio <= 0.05 THEN
  score = 100
  info = "Appropriate use of emphasis"
ELSE
  score = 90
  info = "Consider adding emphasis to key points"
END IF
```

### 8.4 Mobile-First Considerations

**Viewport Optimization:**
- Content width: Max 600px on mobile
- Touch targets: Minimum 48x48 pixels
- Font size: Minimum 16px (body text)
- Line height: 1.5-1.8 for readability

**Mobile Readability Validation:**
```
IF font_size < 16 THEN
  score = 60
  warning = "Font size below 16px; difficult to read on mobile"
ELSE
  score = 100
END IF

IF line_height < 1.5 OR line_height > 2.0 THEN
  score = 80
  info = "Optimal line-height is 1.5-1.8 for mobile readability"
ELSE
  score = 100
END IF
```

**Content Above the Fold:**
- Primary keyword in first 100 words
- Clear value proposition visible immediately
- No intrusive interstitials (Google penalty)

**Above-Fold Validation:**
```
first_100_words = extract_first_words(content, 100)

IF contains(first_100_words, primary_keyword) THEN
  score = 100
  info = "Primary keyword appears early in content"
ELSE
  score = 70
  warning = "Include primary keyword in first 100 words for mobile users"
END IF
```

### 8.5 Content Length Recommendations

**Length by Content Type:**
| Content Type | Minimum | Optimal | Maximum |
|-------------|---------|---------|---------|
| Blog Post | 800 | 1500-2500 | 4000 |
| Pillar Content | 2000 | 3000-5000 | 10000 |
| Product Page | 300 | 500-1000 | 2000 |
| Category Page | 150 | 300-500 | 1000 |
| FAQ Page | 500 | 1000-1500 | 3000 |
| Tutorial | 1200 | 2000-3500 | 6000 |

**2026 Principle:** Ideal content length fully satisfies search intent—no more, no less.

**Length Validation:**
```
word_count = count_words(content)

IF content_type == "blog_post" THEN
  IF word_count >= 1500 AND word_count <= 2500 THEN
    score = 100
  ELSE IF word_count >= 800 AND word_count < 1500 THEN
    score = 80
    info = "Consider expanding content for better depth"
  ELSE IF word_count < 800 THEN
    score = 60
    warning = "Content too short; may not fully cover topic"
  ELSE IF word_count > 4000 THEN
    score = 85
    info = "Very long content; ensure all information adds value"
  END IF
END IF
```

### 8.6 Complete Content Structure Scoring Formula

```
total_content_structure_score = (
  (paragraph_length_score * 0.25) +
  (reading_level_score * 0.20) +
  (formatting_score * 0.20) +
  (mobile_readability_score * 0.20) +
  (content_length_score * 0.15)
) * 100

WHERE:
  paragraph_length_score = short paragraph validation (0-1)
  reading_level_score = Flesch-Kincaid appropriateness (0-1)
  formatting_score = whitespace, lists, emphasis (0-1)
  mobile_readability_score = mobile-first optimization (0-1)
  content_length_score = appropriate depth (0-1)
```

---

## 9. Implementation Specifications

### 9.1 Rule Definitions (Machine-Readable Format)

**JSON Rule Schema:**
```json
{
  "rule_id": "TITLE_001",
  "category": "title_tag",
  "name": "Title Tag Length Validation",
  "description": "Validates title tag length in pixels",
  "severity": "critical",
  "priority_weight": 0.40,
  "validation": {
    "type": "length",
    "metric": "pixel_width",
    "thresholds": {
      "min": 480,
      "optimal_min": 480,
      "optimal_max": 580,
      "max": 600
    }
  },
  "scoring": {
    "formula": "IF pixel_width > 600 THEN 0 ELSE IF pixel_width > 580 THEN 85 ELSE IF pixel_width >= 480 THEN 100 ELSE 70",
    "max_score": 100
  },
  "remediation": {
    "action": "Shorten title to 50-60 characters",
    "priority": "high",
    "estimated_impact": "+15% CTR improvement"
  }
}
```

**Rule Priority Categories:**
| Priority | Weight | Severity | Examples |
|---------|--------|----------|----------|
| Critical | 0.40 | Errors blocking SEO | Missing H1, duplicate titles, no alt text |
| High | 0.30 | Major ranking factors | Title length, keyword placement, schema |
| Medium | 0.20 | Important optimizations | Meta descriptions, internal links, headings |
| Low | 0.10 | Nice-to-have improvements | Image filenames, caption usage |

### 9.2 Scoring Formulas for Each Element

**Master On-Page SEO Score:**
```
total_on_page_score = (
  (title_tag_score * 0.15) +
  (meta_description_score * 0.10) +
  (heading_hierarchy_score * 0.15) +
  (url_structure_score * 0.10) +
  (internal_linking_score * 0.15) +
  (image_optimization_score * 0.10) +
  (schema_markup_score * 0.10) +
  (content_structure_score * 0.15)
) * 100

WHERE each component_score is 0-100
```

**Component Weight Rationale:**
- **Title Tag (15%):** Direct ranking factor, high CTR impact
- **Headings (15%):** Content structure, accessibility, keyword distribution
- **Internal Linking (15%):** Link equity, site architecture, crawlability
- **Content Structure (15%):** User experience, mobile optimization, readability
- **Meta Description (10%):** CTR factor (not direct ranking)
- **URL Structure (10%):** Minor ranking factor, UX impact
- **Images (10%):** Accessibility, page speed, image search
- **Schema (10%):** Rich results, AI Overviews, featured snippets

### 9.3 Priority Weighting System

**Issue Classification:**

**Critical Issues (Must Fix):**
- Missing or duplicate H1
- Missing alt text on content images
- Duplicate title tags
- Title/meta exceeds maximum length
- Broken internal links
- Missing required schema fields

**High Priority (Should Fix):**
- Keyword not in title/H1
- Heading hierarchy violations
- Low internal link density
- No schema markup
- Reading level too difficult

**Medium Priority (Recommended):**
- Suboptimal meta description length
- Generic anchor text
- Missing image captions
- Long paragraphs

**Low Priority (Optional):**
- Image filename optimization
- Additional schema properties
- Minor formatting improvements

**Priority Score Adjustment:**
```
function calculate_priority_weighted_score(issues):
  critical_count = count_issues_by_priority(issues, "critical")
  high_count = count_issues_by_priority(issues, "high")
  medium_count = count_issues_by_priority(issues, "medium")
  low_count = count_issues_by_priority(issues, "low")

  priority_score = 100
  priority_score -= (critical_count * 20)  # -20 points per critical
  priority_score -= (high_count * 10)      # -10 points per high
  priority_score -= (medium_count * 5)     # -5 points per medium
  priority_score -= (low_count * 2)        # -2 points per low

  return MAX(0, priority_score)
end function
```

### 9.4 Validation Pipeline Architecture

**Processing Flow:**
```
1. INPUT: Page URL or HTML content
   ↓
2. EXTRACTION: Parse HTML, extract elements
   ↓
3. ANALYSIS: Run validation rules on each element
   ↓
4. SCORING: Calculate component scores
   ↓
5. AGGREGATION: Compute total on-page score
   ↓
6. PRIORITIZATION: Classify and rank issues
   ↓
7. RECOMMENDATIONS: Generate actionable fixes
   ↓
8. OUTPUT: Report with scores, issues, suggestions
```

**Validation Pipeline Code Structure:**
```javascript
class OnPageSEOValidator {

  validate(pageUrl) {
    // 1. Extract page elements
    const pageData = this.extractPageElements(pageUrl);

    // 2. Run all validation rules
    const titleScore = this.validateTitleTag(pageData.title);
    const metaScore = this.validateMetaDescription(pageData.meta);
    const headingScore = this.validateHeadings(pageData.headings);
    const urlScore = this.validateURL(pageData.url);
    const linkScore = this.validateInternalLinks(pageData.links);
    const imageScore = this.validateImages(pageData.images);
    const schemaScore = this.validateSchema(pageData.schema);
    const contentScore = this.validateContentStructure(pageData.content);

    // 3. Calculate weighted total
    const totalScore = this.calculateTotalScore({
      titleScore,
      metaScore,
      headingScore,
      urlScore,
      linkScore,
      imageScore,
      schemaScore,
      contentScore
    });

    // 4. Generate recommendations
    const recommendations = this.generateRecommendations(pageData);

    // 5. Return report
    return {
      score: totalScore,
      breakdown: { titleScore, metaScore, /* ... */ },
      issues: this.classifyIssues(recommendations),
      actionable_items: this.prioritizeRecommendations(recommendations)
    };
  }

  calculateTotalScore(scores) {
    return (
      scores.titleScore * 0.15 +
      scores.metaScore * 0.10 +
      scores.headingScore * 0.15 +
      scores.urlScore * 0.10 +
      scores.linkScore * 0.15 +
      scores.imageScore * 0.10 +
      scores.schemaScore * 0.10 +
      scores.contentScore * 0.15
    );
  }
}
```

**Parallel Processing for Speed:**
```
async function validatePageParallel(pageData) {
  const validations = await Promise.all([
    validateTitleTag(pageData.title),
    validateMetaDescription(pageData.meta),
    validateHeadings(pageData.headings),
    validateURL(pageData.url),
    validateInternalLinks(pageData.links),
    validateImages(pageData.images),
    validateSchema(pageData.schema),
    validateContentStructure(pageData.content)
  ]);

  return aggregateResults(validations);
}
```

### 9.5 Output Format Specification

**JSON Report Structure:**
```json
{
  "page_url": "https://example.com/seo-guide",
  "analysis_date": "2026-01-16T10:30:00Z",
  "overall_score": 82,
  "grade": "B",
  "breakdown": {
    "title_tag": {
      "score": 95,
      "details": {
        "length": 58,
        "pixel_width": 570,
        "keyword_position": 1,
        "has_power_words": true,
        "has_number": true
      },
      "issues": [],
      "recommendations": [
        "Consider adding bracketed modifier for +15% CTR boost"
      ]
    },
    "meta_description": {
      "score": 88,
      "details": {
        "length": 156,
        "has_cta": true,
        "keyword_count": 1
      },
      "issues": [],
      "recommendations": []
    },
    "headings": {
      "score": 75,
      "details": {
        "h1_count": 1,
        "h2_count": 6,
        "hierarchy_valid": true,
        "keyword_distribution": 0.66
      },
      "issues": [
        {
          "severity": "medium",
          "message": "H1 missing primary keyword",
          "element": "<h1>Welcome to Our SEO Guide</h1>",
          "recommendation": "Include 'on-page SEO' in H1"
        }
      ],
      "recommendations": [
        "Add primary keyword to H1 tag"
      ]
    }
  },
  "prioritized_issues": {
    "critical": [],
    "high": [
      {
        "category": "headings",
        "message": "H1 missing primary keyword",
        "impact": "Medium ranking impact",
        "fix": "Update H1 to include 'on-page SEO'"
      }
    ],
    "medium": [],
    "low": []
  },
  "estimated_improvements": {
    "potential_score_increase": "+8 points",
    "estimated_traffic_impact": "+12% organic traffic",
    "estimated_ctr_improvement": "+5%"
  }
}
```

---

## 10. Success Metrics

### 10.1 Rule Coverage Percentage

**Definition:** Percentage of on-page SEO rules successfully implemented and validated.

**Calculation:**
```
rule_coverage = (rules_passing / total_rules) * 100

WHERE:
  rules_passing = number of rules with score >= 80
  total_rules = total number of applicable rules
```

**Benchmarks:**
| Coverage | Grade | Interpretation |
|---------|-------|----------------|
| 95-100% | A+ | Exceptional optimization |
| 90-94% | A | Excellent optimization |
| 80-89% | B | Good optimization |
| 70-79% | C | Fair optimization |
| 60-69% | D | Poor optimization |
| <60% | F | Critical issues present |

### 10.2 Validation Accuracy

**Definition:** Accuracy of rule validation compared to manual expert audit.

**Measurement:**
```
validation_accuracy = (correct_assessments / total_assessments) * 100

WHERE:
  correct_assessments = rules where automated score matches expert score (±5 points)
  total_assessments = total rules evaluated
```

**Target:** ≥95% accuracy compared to expert audits

**Validation Methodology:**
1. Run automated validation on 100 sample pages
2. Have SEO experts manually audit same pages
3. Compare scores for each rule
4. Calculate accuracy percentage
5. Identify and fix discrepancies

### 10.3 Before/After SEO Score Improvements

**Tracking Methodology:**
```
1. BASELINE: Run initial audit, record score
2. IMPLEMENTATION: Apply recommended fixes
3. POST-OPTIMIZATION: Re-run audit
4. CALCULATE: Improvement delta
```

**Improvement Metrics:**
```
score_improvement = post_optimization_score - baseline_score
improvement_percentage = (score_improvement / baseline_score) * 100

Example:
  Baseline: 65/100
  Post-Optimization: 88/100
  Improvement: +23 points (+35.4%)
```

**Expected Improvement Ranges:**
| Baseline Score | Expected Improvement | Typical Final Score |
|---------------|---------------------|-------------------|
| 0-40 (Poor) | +30-50 points | 60-75 |
| 41-60 (Fair) | +20-30 points | 70-85 |
| 61-80 (Good) | +10-20 points | 80-92 |
| 81-95 (Excellent) | +5-10 points | 90-98 |
| 96-100 (Perfect) | 0-4 points | 96-100 |

**Real-World Impact Correlation:**
```
# Based on industry averages
IF score_improvement >= 20 THEN
  estimated_traffic_increase = "15-25%"
  estimated_ctr_improvement = "10-18%"
ELSE IF score_improvement >= 10 THEN
  estimated_traffic_increase = "8-15%"
  estimated_ctr_improvement = "5-10%"
ELSE IF score_improvement >= 5 THEN
  estimated_traffic_increase = "3-8%"
  estimated_ctr_improvement = "2-5%"
ELSE
  estimated_traffic_increase = "0-3%"
  estimated_ctr_improvement = "0-2%"
END IF
```

### 10.4 Performance Tracking Dashboard

**Key Metrics to Display:**
1. **Overall On-Page Score:** 0-100 scale
2. **Component Breakdown:** Radar chart showing 8 categories
3. **Issue Count by Priority:** Critical, High, Medium, Low
4. **Top 5 Quick Wins:** Highest impact, lowest effort fixes
5. **Historical Trend:** Score over time (line graph)
6. **Competitive Benchmark:** Compare to competitor averages

**Dashboard Example:**
```
┌─────────────────────────────────────────────┐
│  On-Page SEO Score: 82/100 (Grade: B)       │
├─────────────────────────────────────────────┤
│                                             │
│  Component Scores:                          │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░ Title Tag      95/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░ Meta Desc     88/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░ Headings      75/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░ URL           92/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ Links         70/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ Images        85/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░ Schema        78/100 │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░ Content       90/100 │
│                                             │
├─────────────────────────────────────────────┤
│  Issues Found:                              │
│  🔴 Critical: 0                             │
│  🟠 High:     2                             │
│  🟡 Medium:   5                             │
│  🟢 Low:      8                             │
│                                             │
├─────────────────────────────────────────────┤
│  Top Quick Wins:                            │
│  1. Add primary keyword to H1  [+5 pts]    │
│  2. Increase internal link density [+4 pts]│
│  3. Add FAQ schema markup      [+3 pts]    │
│  4. Optimize 3 image filenames [+2 pts]    │
│  5. Shorten 8 long paragraphs  [+2 pts]    │
└─────────────────────────────────────────────┘
```

---

## 11. Summary Rule Table

| Category | Rule | Optimal Range | Weight | Priority |
|---------|------|--------------|--------|---------|
| **Title Tag** | Length | 50-60 chars (580px) | 15% | Critical |
| | Keyword Position | First 3 words | 15% | High |
| | CTR Elements | Power word + number + bracket | 15% | Medium |
| | Uniqueness | No duplicates | 15% | Critical |
| **Meta Description** | Length | 155-158 chars | 10% | High |
| | Mobile Front-load | Key info in first 120 chars | 10% | High |
| | CTA Presence | Action verb + value | 10% | Medium |
| | Keyword Inclusion | 1 occurrence | 10% | Medium |
| **Headings** | H1 Count | Exactly 1 | 15% | Critical |
| | H1 Keyword | Contains primary keyword | 15% | High |
| | Hierarchy | No skipped levels | 15% | Critical |
| | Density | 1 heading per 200-300 words | 15% | Medium |
| **URL Structure** | Length | 3-5 words (25-30 chars) | 10% | Medium |
| | Keyword Inclusion | Primary keyword present | 10% | High |
| | Character Validity | Lowercase, hyphens only | 10% | High |
| | Depth | ≤3 levels | 10% | Low |
| **Internal Links** | Density | 2-5 links per 500 words | 15% | Medium |
| | Anchor Text | 10-20% exact match | 15% | High |
| | Contextual Placement | Top 30% of page | 15% | Medium |
| | No Orphan Pages | All pages linked | 15% | High |
| **Images** | Alt Text Length | 80-125 chars | 10% | Critical |
| | Alt Descriptiveness | Specific, detailed | 10% | Critical |
| | Filename | Keyword-rich, hyphens | 10% | Low |
| | Lazy Loading | Below-fold images | 10% | Medium |
| **Schema Markup** | Presence | Appropriate type for page | 10% | High |
| | Validity | Passes Google validator | 10% | High |
| | Completeness | All recommended fields | 10% | Medium |
| **Content Structure** | Paragraph Length | 1-3 sentences | 15% | Medium |
| | Reading Level | 7th-9th grade (60-80 Flesch) | 15% | Medium |
| | Content Length | 1500-2500 words (blog) | 15% | Medium |
| | Mobile Optimization | 16px font, keyword in first 100 words | 15% | High |

---

## 12. Appendix: Tools and Resources

### 12.1 Recommended Validation Tools

- **Title/Meta Length:** [SERP Preview Tool](https://mrs.digital/tools/meta-length-checker/)
- **Schema Validation:** [Google Rich Results Test](https://search.google.com/test/rich-results)
- **Accessibility:** [WAVE Web Accessibility Tool](https://wave.webaim.org/)
- **Reading Level:** Flesch-Kincaid calculators
- **Mobile Testing:** Google Mobile-Friendly Test

### 12.2 Implementation Checklist

**Pre-Optimization:**
- [ ] Audit current on-page score
- [ ] Identify critical issues
- [ ] Prioritize fixes by impact/effort ratio

**Optimization Phase:**
- [ ] Update title tags (all pages)
- [ ] Optimize meta descriptions
- [ ] Fix heading hierarchy
- [ ] Implement schema markup
- [ ] Optimize images (alt text, filenames)
- [ ] Improve internal linking
- [ ] Enhance content structure

**Post-Optimization:**
- [ ] Re-run validation
- [ ] Verify improvements
- [ ] Monitor rankings and traffic
- [ ] Iterate on low-performing pages

---

## Sources

Research for this document was compiled from the following sources:

**Title Tag Optimization:**
- [What should the title tag length be in 2025?](https://searchengineland.com/title-tag-length-388468)
- [How to Optimize Title Tags & Meta Descriptions in 2026 | Straight North](https://www.straightnorth.com/blog/title-tags-and-meta-descriptions-how-to-write-and-optimize-them-in-2026/)
- [Meta Title/Description Guide: 2026 Best Practices](https://www.stanventures.com/blog/meta-title-length-meta-description-length/)
- [How to Optimize Website Title Tags for SEO in 2026](https://zoer.ai/posts/zoer/optimize-website-title-tags-seo)
- [The definitive guide to title tag SEO best practices post Google leak](https://www.hobo-web.co.uk/title-tags/)

**Meta Description Optimization:**
- [How to Write Compelling Meta Descriptions for SEO and CTR (2025) - Analytify](https://analytify.io/how-to-write-meta-descriptions-for-seo-and-ctr/)
- [Meta Title and Description Character Limit (2026 Guidelines)](https://www.wscubetech.com/blog/meta-title-description-length/)
- [Meta Description Example Length and Best Practices](https://www.vazoola.com/resources/meta-description)

**Heading Hierarchy:**
- [Headings | Web Accessibility Initiative (WAI) | W3C](https://www.w3.org/WAI/tutorials/page-structure/headings/)
- [How-to: Accessible heading structure - The A11Y Project](https://www.a11yproject.com/posts/how-to-accessible-heading-structure/)
- [Using H1, H2, H3 Heading Tags for SEO and UX • LockedownSEO](https://lockedownseo.com/using-html-heading-tags/)
- [Header Tags (H1–H6): Structure and SEO Best Practices](https://www.llmvlab.com/guides/header-tags)

**Internal Linking:**
- [SEO Link Best Practices for Google | Google Search Central](https://developers.google.com/search/docs/crawling-indexing/links-crawlable)
- [Top 6 Internal Linking Best Practices for SEO in 2025 - LinkStorm](https://linkstorm.io/resources/internal-linking-best-practices)
- [Internal Linking Strategy: Complete SEO Guide for 2026](https://www.ideamagix.com/blog/internal-linking-strategy-seo-guide-2026/)
- [Internal Link Anchor Text Optimization | SEO Tips & Tools](https://www.ibeamconsulting.com/blog/seo-internal-link-anchor-text-optimization/)

**Image Optimization:**
- [Image SEO Best Practices | Google Search Central](https://developers.google.com/search/docs/appearance/google-images)
- [The Definitive Guide to Image SEO & Alt Text Best Practices](https://www.hobo-web.co.uk/images-and-alt-text-seo-checklist/)
- [Alt Text to Supercharge Discoverability: SEO Guidelines](https://www.amsive.com/insights/seo/alt-text-to-supercharge-discoverability-seo-guidelines-for-smarter-image-optimization/)
- [How Long Should Image Alt Text Be?](https://www.airops.com/blog/how-long-should-image-alt-text-be)

**URL Structure:**
- [URL Slug Optimisation Guide for SEO Safe Decisions 2026](https://seoservicecare.com/url-slug-guide/)
- [SEO URL Structure: 7 Best Practices For Creating SEO URLs (2025) - Shopify](https://www.shopify.com/blog/seo-url)
- [SEO-Friendly URLs: Keyword Tips & Best Practices for 2026](https://www.stanventures.com/blog/url-structure/)
- [URL Slugs: How to Create SEO-Friendly URLs (10 Easy Steps)](https://seosherpa.com/url-slugs/)

**Schema Markup:**
- [Schema Markup in 2026: Why It's Now Critical for SERP Visibility](https://almcorp.com/blog/schema-markup-detailed-guide-2026-serp-visibility/)
- [Schema Markup Guide: Step-by-Step SEO Strategy for 2026](https://www.clickrank.ai/schema-markup/)
- [FAQ (FAQPage, Question, Answer) structured data](https://developers.google.com/search/docs/appearance/structured-data/faqpage)
- [Learn About Article Schema Markup | Google Search Central](https://developers.google.com/search/docs/appearance/structured-data/article)

**Content Structure:**
- [What Is the Ideal Content Length for SEO in 2026?](https://www.clickrank.ai/ideal-content-length-for-seo/)
- [SEO Writing: The 13 Rules For Creating SEO Optimized Content (2026 A-Z Guide)](https://elementor.com/blog/seo-writing/)
- [Content Writing 2026: How to Create SEO-Friendly Content That Ranks](https://www.clickrank.ai/content-writing/)
- [On-Page SEO: The Definitive Guide + FREE Template (2026)](https://backlinko.com/on-page-seo)

**SEO Scoring:**
- [How is optimization score calculated](https://seo.ai/faq/how-is-optimization-score-calculated)
- [Google's 200 Ranking Factors: The Complete List (2026)](https://backlinko.com/google-ranking-factors)
- [What is On-Page SEO score and how is it calculated?](https://dataforseo.com/help-center/how-on-page-seo-score-is-calculated)

---

**End of Document**

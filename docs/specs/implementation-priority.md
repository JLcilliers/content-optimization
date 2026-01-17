# Implementation Priority & Risk Assessment

## Executive Summary

This document defines the recommended implementation order for the SEO + AI Content Optimizer. The **critical path** is the diffing/highlighting system—if this fails, the tool is worthless. The build order prioritizes getting the critical path working first, with iterative addition of features.

---

## Implementation Phases

### Phase 0: Project Setup (Day 1)

| Task | Priority | Effort | Deliverable |
|------|----------|--------|-------------|
| Initialize Python project with `uv` | P0 | 1h | `pyproject.toml` |
| Configure pre-commit hooks (ruff, mypy) | P0 | 1h | `.pre-commit-config.yaml` |
| Set up pytest structure | P0 | 1h | `tests/` directory |
| Create Pydantic base models | P0 | 2h | `src/models/` |
| Install core dependencies | P0 | 1h | Lock file |

**Exit Criteria**: Project runs, tests execute, type checking works.

---

### Phase 1: Critical Path - Diffing System (Week 1-2)

**Rationale**: Build the hardest, most critical component first. If diffing can't achieve zero false positives, we need to know early.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement text normalization | P0 | 4h | Phase 0 | `src/diffing/normalizer.py` |
| Implement basic diff algorithm | P0 | 8h | Normalizer | `src/diffing/differ.py` |
| Add semantic similarity detection | P0 | 8h | Basic diff | `src/diffing/semantic_matcher.py` |
| Add move detection | P1 | 6h | Basic diff | `src/diffing/move_detector.py` |
| Add boundary refinement | P0 | 8h | Semantic + Move | `src/diffing/boundary_finder.py` |
| Implement ChangeSet generation | P0 | 4h | Boundary finder | `src/diffing/changeset.py` |
| Create 35+ edge case tests | P0 | 12h | All above | `tests/diffing/` |
| Achieve 100% test pass rate | P0 | - | Tests | Validation |

**Exit Criteria**:
- All 35 edge cases pass
- Zero false positives in test suite
- Zero false negatives in test suite
- Manual review of 10 real document pairs

**Risk Mitigation**: If semantic similarity proves unreliable, fall back to conservative character-level diffing only (higher false negative rate acceptable, zero false positive rate mandatory).

---

### Phase 2: DOCX I/O Pipeline (Week 2-3)

**Rationale**: Connect the diffing system to real DOCX files.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement DOCX parser | P0 | 8h | Phase 0 | `src/ingestion/docx_parser.py` |
| Create DocumentAST models | P0 | 4h | Parser | `src/ingestion/models.py` |
| Implement OriginalSnapshot | P0 | 4h | Models | `src/ingestion/snapshot.py` |
| Implement run-level highlighting | P0 | 8h | Phase 1 | `src/output/highlighter.py` |
| Implement DOCX reconstruction | P0 | 8h | Highlighter | `src/output/docx_builder.py` |
| End-to-end test: DOCX → DOCX | P0 | 4h | All above | Integration test |

**Exit Criteria**:
- Parse any valid DOCX without errors
- Preserve 100% of original structure
- Apply green highlights only to ChangeSet items
- Cross-platform compatibility (Word, Google Docs, LibreOffice)

---

### Phase 3: Basic Analysis (Week 3-4)

**Rationale**: Add keyword analysis to identify optimization opportunities.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement keyword mapping | P1 | 6h | Phase 2 | `src/analysis/keyword_mapper.py` |
| Implement density analysis | P1 | 4h | Keyword mapper | `src/analysis/density_analyzer.py` |
| Implement gap detection | P1 | 6h | Density analyzer | `src/analysis/gap_detector.py` |
| Create OptimizationPlan model | P1 | 2h | Gap detector | `src/analysis/models.py` |
| Tests for analysis module | P1 | 4h | All above | `tests/analysis/` |

**Exit Criteria**:
- Correctly map keywords to document sections
- Identify under-optimized areas
- Generate actionable OptimizationPlan

---

### Phase 4: FAQ Generation (Week 4-5)

**Rationale**: First content generation feature—high user value.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement FAQ detection | P1 | 4h | Phase 2 | `src/analysis/faq_detector.py` |
| Implement question generation | P1 | 8h | Phase 3 | `src/generation/faq_generator.py` |
| Implement answer grounding | P1 | 6h | Question gen | Grounding checks |
| Integrate LLM (optional) | P2 | 8h | Question/answer | LLM integration |
| Add FAQ to DocumentAST | P1 | 4h | Generator | AST update |
| Diff integration (FAQ = new) | P1 | 2h | Phase 1 | Auto-highlight FAQ |
| Tests for FAQ generation | P1 | 4h | All above | `tests/generation/` |

**Exit Criteria**:
- Correctly detect missing FAQ sections
- Generate relevant, grounded Q&As
- All generated FAQ content highlighted green
- No hallucinated content

---

### Phase 5: Business Context (Week 5-6)

**Rationale**: Improves FAQ quality but not required for MVP.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement brand doc parser | P2 | 6h | Phase 2 | `src/context/brand_parser.py` |
| Implement entity extraction | P2 | 6h | spaCy setup | `src/context/entity_extractor.py` |
| Implement context inference | P2 | 8h | Entity extraction | `src/context/context_builder.py` |
| Create BusinessContext model | P2 | 2h | Inference | `src/context/models.py` |
| Integrate context into FAQ gen | P2 | 4h | Phase 4 + Context | Integration |
| Tests for context module | P2 | 4h | All above | `tests/context/` |

**Exit Criteria**:
- Parse brand documents when provided
- Infer business context from content alone
- FAQ generation uses context appropriately

---

### Phase 6: Safety Guardrails (Week 6-7)

**Rationale**: Production safety—can ship MVP without all guardrails, but must have basics.

| Task | Priority | Effort | Depends On | Deliverable |
|------|----------|--------|------------|-------------|
| Implement over-optimization detection | P1 | 6h | Phase 3 | `src/guardrails/over_optimization.py` |
| Implement highlight verification | P1 | 4h | Phase 1 | `src/guardrails/highlight_check.py` |
| Implement factual grounding check | P2 | 8h | Phase 4 | `src/guardrails/factual_check.py` |
| Implement brand voice check | P3 | 6h | Phase 5 | `src/guardrails/voice_check.py` |
| Create changes summary report | P2 | 4h | Phase 2 | `src/output/report_generator.py` |
| Tests for guardrails | P1 | 4h | All above | `tests/guardrails/` |

**Exit Criteria**:
- Block outputs with over-optimization
- Verify highlight accuracy post-diff
- Generate useful changes summary

---

## MVP Definition

### MVP Features (Must Have)

| Feature | Phase | Status |
|---------|-------|--------|
| DOCX parsing with structure preservation | 2 | Required |
| Keyword-to-section mapping | 3 | Required |
| Optimization gap detection | 3 | Required |
| FAQ generation (if missing) | 4 | Required |
| Precise diffing (zero false positives) | 1 | **CRITICAL** |
| Green highlighting on new content only | 2 | **CRITICAL** |
| DOCX output with preserved formatting | 2 | Required |
| Over-optimization detection | 6 | Required |

### Post-MVP Features (Nice to Have)

| Feature | Phase | Priority |
|---------|-------|----------|
| Brand document parsing | 5 | P2 |
| Context-aware FAQ generation | 5 | P2 |
| Factual grounding verification | 6 | P2 |
| Brand voice validation | 6 | P3 |
| Content enhancement beyond FAQ | Future | P3 |
| Detailed changes summary report | 6 | P2 |

---

## Dependency Graph

```
Phase 0: Setup
    │
    ▼
Phase 1: Diffing (CRITICAL)
    │
    ├──────────────────────┐
    ▼                      │
Phase 2: DOCX I/O          │
    │                      │
    ▼                      │
Phase 3: Analysis ◄────────┘
    │
    ▼
Phase 4: FAQ Generation
    │
    ├──────────────────────┐
    ▼                      │
Phase 5: Context           │
    │                      │
    ▼                      │
Phase 6: Guardrails ◄──────┘
```

---

## Risk Assessment

### High Risk (Probability × Impact)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Diffing false positives | Medium | Critical | Conservative thresholds, extensive testing |
| Semantic similarity unreliable | Medium | High | Fall back to character-level only |
| DOCX formatting corruption | Low | High | Preserve original, only add highlights |
| LLM hallucination in FAQ | Medium | High | Source grounding required, no LLM fallback |

### Medium Risk

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Run splitting edge cases | Medium | Medium | Comprehensive test fixtures |
| Cross-platform compatibility | Low | Medium | Test on Word, Docs, LibreOffice |
| Context inference low quality | Medium | Low | Graceful degradation to no context |

### Low Risk

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| spaCy model loading time | Low | Low | Lazy loading, caching |
| Keyword density calculation errors | Low | Low | Unit tests |

---

## Risk Mitigation Matrix

```
                    │ Low Probability │ High Probability
────────────────────┼─────────────────┼──────────────────
High Impact         │ Formatting      │ Diffing FPs
                    │ corruption      │ Semantic unreliable
                    │ (watch closely) │ (MITIGATE NOW)
────────────────────┼─────────────────┼──────────────────
Low Impact          │ Model loading   │ Context quality
                    │ (accept)        │ (accept/degrade)
```

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0: Setup | 1 day | Day 1 |
| Phase 1: Diffing | 2 weeks | Week 2 |
| Phase 2: DOCX I/O | 1 week | Week 3 |
| Phase 3: Analysis | 1 week | Week 4 |
| Phase 4: FAQ Gen | 1.5 weeks | Week 5.5 |
| Phase 5: Context | 1 week | Week 6.5 |
| Phase 6: Guardrails | 1 week | Week 7.5 |

**MVP Ready**: End of Week 5 (Phases 0-4 + basic guardrails)
**Full Feature**: End of Week 8 (all phases + polish)

---

## Recommended Starting Point

1. **Today**: Initialize project (Phase 0)
2. **Tomorrow**: Start diffing system (Phase 1)
3. **Focus**: Spend 60% of Phase 1 time on edge case testing

The diffing system is the foundation. If it works perfectly, everything else builds on solid ground. If it has bugs, the entire tool is unreliable.

---

## Success Metrics

### MVP Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| Highlight false positive rate | 0% | Automated + manual testing |
| Highlight false negative rate | <5% | Manual testing |
| DOCX structure preservation | 100% | Automated testing |
| FAQ relevance score | >0.7 | Manual review |
| Processing time (avg doc) | <5s | Benchmark |

### Production Success

| Metric | Target | Measurement |
|--------|--------|-------------|
| User-reported highlight errors | 0/month | Support tickets |
| User satisfaction with FAQ quality | >80% | Survey |
| Cross-platform compatibility issues | 0 | Testing |

---

*Document Version: 2.0 (Revised Scope)*
*Created: 2026-01-16*

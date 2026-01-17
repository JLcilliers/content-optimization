"""
Tests for the diffing module - CRITICAL TEST SUITE

This is the most important test suite in the project.
The diffing system must achieve:
- Zero false positives (never highlight existing content)
- Near-zero false negatives (rarely miss new content)

Test categories:
- test_differ.py: Core diffing algorithm tests
- test_semantic.py: Semantic similarity detection
- test_edge_cases.py: Comprehensive edge case coverage (35+ cases)
"""

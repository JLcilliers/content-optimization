"""
Tests for content analysis endpoints.
"""

import io


class TestDocumentAnalysis:
    """Tests for document analysis endpoint."""

    def test_analyze_document_success(self, client, sample_docx_content):
        """Test successful document analysis."""
        response = client.post(
            "/api/v1/analyze/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert "geo_score" in data
        assert "seo_score" in data
        assert "issues" in data
        assert "recommendations" in data

    def test_analyze_document_with_keywords(self, client, sample_docx_content):
        """Test document analysis with primary and secondary keywords."""
        response = client.post(
            "/api/v1/analyze/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={
                "primary_keyword": "content optimization",
                "secondary_keywords": "SEO, rankings, visibility",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "keyword_density" in data

    def test_analyze_document_invalid_file(self, client):
        """Test analysis with invalid file type."""
        response = client.post(
            "/api/v1/analyze/document",
            files={"file": ("test.txt", b"This is not a docx file")},
        )
        assert response.status_code == 400
        assert "docx" in response.json()["detail"].lower()


class TestTextAnalysis:
    """Tests for text analysis endpoint."""

    def test_analyze_text_success(self, client):
        """Test successful text analysis."""
        response = client.post(
            "/api/v1/analyze/text",
            json={
                "text": "This is a test paragraph about SEO content optimization. "
                        "Good content helps with search engine rankings.",
                "primary_keyword": "SEO",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "geo_score" in data
        assert "word_count" in data
        assert data["word_count"] > 0

    def test_analyze_text_without_keyword(self, client):
        """Test text analysis without keyword."""
        response = client.post(
            "/api/v1/analyze/text",
            json={
                "text": "This is a simple test paragraph.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "geo_score" in data

    def test_analyze_text_empty(self, client):
        """Test analysis with empty text."""
        response = client.post(
            "/api/v1/analyze/text",
            json={"text": ""},
        )
        # Should either succeed with zeros or return error
        assert response.status_code in [200, 400, 422]


class TestAnalysisScores:
    """Tests for analysis score calculation."""

    def test_scores_are_valid_range(self, client, sample_docx_content):
        """Test that all scores are in valid range."""
        response = client.post(
            "/api/v1/analyze/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO"},
        )
        assert response.status_code == 200
        data = response.json()

        # All scores should be between 0 and 100
        assert 0 <= data["geo_score"] <= 100
        assert 0 <= data["seo_score"] <= 100
        assert 0 <= data["semantic_score"] <= 100
        assert 0 <= data["ai_readiness_score"] <= 100
        assert 0 <= data["readability_score"] <= 100

    def test_geo_score_is_weighted_average(self, client, sample_docx_content):
        """Test that GEO score is a weighted average of component scores."""
        response = client.post(
            "/api/v1/analyze/document",
            files={"file": ("test.docx", sample_docx_content)},
        )
        assert response.status_code == 200
        data = response.json()

        # GEO score should be roughly in range of component scores
        component_min = min(
            data["seo_score"],
            data["semantic_score"],
            data["ai_readiness_score"],
            data["readability_score"],
        )
        component_max = max(
            data["seo_score"],
            data["semantic_score"],
            data["ai_readiness_score"],
            data["readability_score"],
        )

        # Allow some tolerance for weighting
        assert component_min - 10 <= data["geo_score"] <= component_max + 10

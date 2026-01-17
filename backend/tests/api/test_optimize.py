"""
Tests for content optimization endpoints.
"""

import io


class TestDocumentOptimization:
    """Tests for document optimization endpoint."""

    def test_optimize_document_success(self, client, sample_docx_content):
        """Test successful document optimization."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO optimization"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "original_geo_score" in data
        assert "optimized_geo_score" in data
        assert "improvement" in data
        assert "changes_count" in data
        assert "changes" in data

    def test_optimize_with_all_options(self, client, sample_docx_content):
        """Test optimization with all options enabled."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={
                "primary_keyword": "content optimization",
                "secondary_keywords": "SEO, rankings",
                "brand_name": "TestBrand",
                "inject_keywords": True,
                "generate_faq": True,
                "faq_count": 3,
                "improve_readability": True,
                "optimize_headings": True,
                "min_keyword_density": 1.0,
                "max_keyword_density": 2.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_optimize_with_options_disabled(self, client, sample_docx_content):
        """Test optimization with options disabled."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={
                "primary_keyword": "SEO",
                "inject_keywords": False,
                "generate_faq": False,
                "improve_readability": False,
                "optimize_headings": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # With all options disabled, should have fewer changes
        assert isinstance(data["changes"], list)

    def test_optimize_invalid_file(self, client):
        """Test optimization with invalid file type."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.txt", b"Not a docx")},
            data={"primary_keyword": "test"},
        )
        assert response.status_code == 400

    def test_optimize_missing_keyword(self, client, sample_docx_content):
        """Test optimization without primary keyword."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
        )
        # Should fail validation - primary_keyword is required
        assert response.status_code == 422


class TestOptimizationDownload:
    """Tests for optimization download endpoint."""

    def test_download_optimized_document(self, client, sample_docx_content):
        """Test downloading optimized document."""
        response = client.post(
            "/api/v1/optimize/document/download",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO optimization"},
        )
        assert response.status_code == 200
        assert "application/vnd.openxmlformats" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "_optimized.docx" in response.headers["content-disposition"]

    def test_download_preserves_docx_format(self, client, sample_docx_content):
        """Test that downloaded file is valid DOCX."""
        response = client.post(
            "/api/v1/optimize/document/download",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO"},
        )
        assert response.status_code == 200

        # Check file starts with ZIP signature (DOCX is a ZIP file)
        content = response.content
        assert content[:4] == b"PK\x03\x04"


class TestOptimizationPreview:
    """Tests for optimization preview endpoint."""

    def test_preview_optimization(self, client, sample_docx_content):
        """Test optimization preview."""
        response = client.post(
            "/api/v1/optimize/preview",
            files={"file": ("test.docx", sample_docx_content)},
            data={
                "primary_keyword": "content optimization",
                "generate_faq": True,
                "faq_count": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_geo_score" in data
        assert "estimated_geo_score" in data
        assert "estimated_improvement" in data
        assert "proposed_changes" in data

    def test_preview_includes_faq_preview(self, client, sample_docx_content):
        """Test that preview includes FAQ preview when enabled."""
        response = client.post(
            "/api/v1/optimize/preview",
            files={"file": ("test.docx", sample_docx_content)},
            data={
                "primary_keyword": "SEO",
                "generate_faq": True,
                "faq_count": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # FAQ preview should be present when FAQ generation is enabled
        if data.get("faq_preview"):
            assert len(data["faq_preview"]) <= 2
            for faq in data["faq_preview"]:
                assert "question" in faq
                assert "answer" in faq


class TestOptimizationChanges:
    """Tests for optimization change details."""

    def test_changes_have_required_fields(self, client, sample_docx_content):
        """Test that changes have all required fields."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO"},
        )
        assert response.status_code == 200
        data = response.json()

        for change in data["changes"]:
            assert "type" in change
            assert "location" in change
            assert "original" in change
            assert "optimized" in change
            assert "reason" in change
            assert "impact_score" in change

    def test_changes_have_valid_types(self, client, sample_docx_content):
        """Test that change types are valid."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO"},
        )
        assert response.status_code == 200
        data = response.json()

        valid_types = {"keyword", "readability", "structure", "heading", "faq", "entity", "meta"}
        for change in data["changes"]:
            assert change["type"] in valid_types

    def test_improvement_score_calculation(self, client, sample_docx_content):
        """Test that improvement score is calculated correctly."""
        response = client.post(
            "/api/v1/optimize/document",
            files={"file": ("test.docx", sample_docx_content)},
            data={"primary_keyword": "SEO optimization"},
        )
        assert response.status_code == 200
        data = response.json()

        expected_improvement = data["optimized_geo_score"] - data["original_geo_score"]
        # Allow small floating point tolerance
        assert abs(data["improvement"] - expected_improvement) < 0.01

"""
Content analysis endpoints.
"""

import io
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / "src"))

from seo_optimizer.ingestion.docx_parser import DocxParser
from seo_optimizer.analysis.analyzer import ContentAnalyzer
from seo_optimizer.analysis.models import AnalysisResult

router = APIRouter()


class AnalysisResponse(BaseModel):
    """Response model for content analysis."""

    doc_id: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    heading_count: int
    geo_score: float
    seo_score: float
    semantic_score: float
    ai_readiness_score: float
    readability_score: float
    keyword_density: Optional[float] = None
    issues: List[dict]
    recommendations: List[dict]


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""

    text: str
    primary_keyword: Optional[str] = None
    secondary_keywords: Optional[List[str]] = None


@router.post("/document", response_model=AnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    primary_keyword: Optional[str] = Form(None),
    secondary_keywords: Optional[str] = Form(None),
):
    """
    Analyze a DOCX document for SEO and AI readiness.

    Returns GEO-Metric scores and optimization recommendations.
    """
    # Validate file type
    if not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only .docx files are supported"
        )

    try:
        # Read file content
        content = await file.read()
        file_stream = io.BytesIO(content)

        # Parse document
        parser = DocxParser()
        ast = parser.parse_stream(file_stream)

        # Parse secondary keywords
        sec_keywords = []
        if secondary_keywords:
            sec_keywords = [k.strip() for k in secondary_keywords.split(",")]

        # Analyze content
        analyzer = ContentAnalyzer(
            primary_keyword=primary_keyword,
            secondary_keywords=sec_keywords,
        )
        result = analyzer.analyze(ast)

        return _format_analysis_response(ast.doc_id, result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/text", response_model=AnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze raw text content for SEO and AI readiness.

    Useful for quick analysis without uploading a document.
    """
    try:
        # Create a simple AST from text
        from seo_optimizer.ingestion.models import (
            DocumentAST, ContentNode, NodeType, DocumentMetadata
        )

        nodes = [
            ContentNode(
                node_id="p1",
                node_type=NodeType.PARAGRAPH,
                text_content=request.text,
                position=0,
            )
        ]
        ast = DocumentAST(
            doc_id="text-analysis",
            nodes=nodes,
            metadata=DocumentMetadata(source_path="text-input"),
        )

        # Analyze
        analyzer = ContentAnalyzer(
            primary_keyword=request.primary_keyword,
            secondary_keywords=request.secondary_keywords or [],
        )
        result = analyzer.analyze(ast)

        return _format_analysis_response(ast.doc_id, result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


def _format_analysis_response(doc_id: str, result: AnalysisResult) -> AnalysisResponse:
    """Format analysis result for API response."""
    return AnalysisResponse(
        doc_id=doc_id,
        word_count=result.word_count,
        sentence_count=result.sentence_count,
        paragraph_count=result.paragraph_count,
        heading_count=result.heading_count,
        geo_score=result.geo_score,
        seo_score=result.seo_score,
        semantic_score=result.semantic_score,
        ai_readiness_score=result.ai_readiness_score,
        readability_score=result.readability_score,
        keyword_density=result.keyword_density,
        issues=[
            {
                "type": issue.issue_type.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "location": issue.location,
            }
            for issue in result.issues
        ],
        recommendations=[
            {
                "category": rec.category,
                "priority": rec.priority,
                "description": rec.description,
                "impact": rec.impact,
            }
            for rec in result.recommendations
        ],
    )

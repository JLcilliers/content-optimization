"""
Content analysis endpoints.
"""

import io
import sys
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Add src to path for imports - must happen before any seo_optimizer imports
_src_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

if TYPE_CHECKING:
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
        # Lazy imports to allow app startup even if modules have issues
        from seo_optimizer.ingestion.docx_parser import DocxParser
        from seo_optimizer.analysis.analyzer import ContentAnalyzer, AnalyzerConfig
        from seo_optimizer.analysis.models import KeywordConfig

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

        # Create keyword config (use default if no primary keyword provided)
        keyword_config = KeywordConfig(
            primary_keyword=primary_keyword or "content",
            secondary_keywords=sec_keywords,
        )

        # Create analyzer config - disable semantic analysis (requires spacy)
        # and entity extraction until NLP deps are fully installed
        analyzer_config = AnalyzerConfig(
            enable_semantic_analysis=False,
            enable_entity_extraction=False,
        )

        # Analyze content
        analyzer = ContentAnalyzer(config=analyzer_config)
        result = analyzer.analyze(ast, keywords=keyword_config)

        return _format_analysis_response(ast.doc_id, result)

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not fully configured: {str(e)}"
        )
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
        # Lazy imports to allow app startup even if modules have issues
        from seo_optimizer.ingestion.models import (
            DocumentAST, ContentNode, NodeType, DocumentMetadata, PositionInfo
        )
        from seo_optimizer.analysis.analyzer import ContentAnalyzer, AnalyzerConfig
        from seo_optimizer.analysis.models import KeywordConfig

        nodes = [
            ContentNode(
                node_id="p1",
                node_type=NodeType.PARAGRAPH,
                text_content=request.text,
                position=PositionInfo(position_id="p1", start_char=0, end_char=len(request.text)),
            )
        ]
        ast = DocumentAST(
            doc_id="text-analysis",
            nodes=nodes,
            metadata=DocumentMetadata(source_path="text-input"),
        )

        # Create keyword config (use default if no primary keyword provided)
        keyword_config = KeywordConfig(
            primary_keyword=request.primary_keyword or "content",
            secondary_keywords=request.secondary_keywords or [],
        )

        # Create analyzer config - disable semantic analysis (requires spacy)
        analyzer_config = AnalyzerConfig(
            enable_semantic_analysis=False,
            enable_entity_extraction=False,
        )

        # Analyze
        analyzer = ContentAnalyzer(config=analyzer_config)
        result = analyzer.analyze(ast, keywords=keyword_config)

        return _format_analysis_response(ast.doc_id, result)

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not fully configured: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


def _format_analysis_response(doc_id: str, result: "AnalysisResult") -> AnalysisResponse:
    """Format analysis result for API response."""
    # Access nested attributes from AnalysisResult structure
    stats = result.document_stats
    geo = result.geo_score

    # Build issues list from all_issues in geo_score
    issues_list = []
    for issue in geo.all_issues:
        issues_list.append({
            "type": issue.category.value if hasattr(issue, 'category') else "unknown",
            "severity": issue.severity.value if hasattr(issue, 'severity') else "info",
            "message": issue.message if hasattr(issue, 'message') else str(issue),
            "location": getattr(issue, 'location', None),
        })

    # Build recommendations list (currently stored as strings)
    recs_list = []
    for rec in result.recommendations:
        if isinstance(rec, str):
            recs_list.append({
                "category": "general",
                "priority": "medium",
                "description": rec,
                "impact": "Improves content quality",
            })
        else:
            # Handle Recommendation objects if present
            recs_list.append({
                "category": rec.category.value if hasattr(rec.category, 'value') else str(rec.category),
                "priority": rec.priority.value if hasattr(rec.priority, 'value') else str(rec.priority),
                "description": rec.description,
                "impact": rec.impact,
            })

    return AnalysisResponse(
        doc_id=doc_id,
        word_count=stats.word_count,
        sentence_count=stats.sentence_count,
        paragraph_count=stats.paragraph_count,
        heading_count=stats.heading_count,
        geo_score=geo.total,
        seo_score=geo.seo_score.total,
        semantic_score=geo.semantic_score.total,
        ai_readiness_score=geo.ai_score.total,
        readability_score=geo.readability_score.total,
        keyword_density=geo.seo_score.keyword_density if hasattr(geo.seo_score, 'keyword_density') else None,
        issues=issues_list,
        recommendations=recs_list,
    )

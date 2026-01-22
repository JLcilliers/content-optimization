"""
URL-based content analysis and optimization endpoints.

These endpoints allow users to analyze and optimize content directly from URLs
using the Firecrawl API for content extraction.
"""

import io
import sys
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, validator

# Add src to path for imports - must happen before any seo_optimizer imports
_src_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

router = APIRouter()


# Request/Response Models

class UrlScrapeRequest(BaseModel):
    """Request model for URL scraping."""

    url: HttpUrl

    @validator("url")
    def validate_url(cls, v):
        """Validate URL format."""
        parsed = urlparse(str(v))
        if parsed.scheme not in ("http", "https"):
            raise ValueError("URL must use http or https protocol")
        return v


class UrlScrapeResponse(BaseModel):
    """Response model for URL scrape info."""

    success: bool
    url: str
    title: Optional[str] = None
    word_count: int
    paragraph_count: int
    heading_count: int
    has_faq: bool
    description: Optional[str] = None


class UrlAnalyzeRequest(BaseModel):
    """Request model for URL analysis."""

    url: HttpUrl
    primary_keyword: Optional[str] = None
    secondary_keywords: Optional[List[str]] = None


class UrlAnalyzeResponse(BaseModel):
    """Response model for URL analysis."""

    success: bool
    url: str
    title: Optional[str] = None
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


class UrlOptimizeRequest(BaseModel):
    """Request model for URL optimization."""

    url: HttpUrl
    primary_keyword: str
    secondary_keywords: Optional[List[str]] = None
    brand_name: Optional[str] = None
    inject_keywords: bool = True
    generate_faq: bool = True
    faq_count: int = 5
    improve_readability: bool = True
    optimize_headings: bool = True
    min_keyword_density: float = 1.0
    max_keyword_density: float = 2.5
    output_format: Literal["markdown", "docx"] = "markdown"


class UrlOptimizeResponse(BaseModel):
    """Response model for URL optimization."""

    success: bool
    original_url: str
    original_geo_score: float
    optimized_geo_score: float
    improvement: float
    optimized_content: str
    changes_count: int
    changes: List[dict]
    warnings: List[str]


# Helper functions

async def _get_firecrawl_client():
    """Get configured Firecrawl client."""
    from backend.app.core.config import settings
    from seo_optimizer.ingestion.firecrawl_client import FirecrawlClient

    if not settings.FIRECRAWL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Firecrawl API key not configured. Please set FIRECRAWL_API_KEY in .env"
        )

    return FirecrawlClient(
        api_key=settings.FIRECRAWL_API_KEY,
        api_url=settings.FIRECRAWL_API_URL,
        timeout=settings.FIRECRAWL_TIMEOUT,
    )


async def _scrape_url(url: str):
    """Scrape URL and return parsed AST."""
    from seo_optimizer.ingestion.firecrawl_client import (
        FirecrawlError,
        FirecrawlRateLimitError,
        FirecrawlBlockedError,
        FirecrawlTimeoutError,
    )
    from seo_optimizer.ingestion.markdown_parser import parse_markdown

    client = await _get_firecrawl_client()

    try:
        response = await client.scrape(url)

        if not response.success or not response.markdown:
            raise HTTPException(
                status_code=422,
                detail="Failed to extract content from URL"
            )

        # Parse markdown into AST
        ast = parse_markdown(response.markdown, source_url=url)

        return ast, response

    except FirecrawlRateLimitError:
        raise HTTPException(status_code=429, detail="Too many requests, please wait")
    except FirecrawlBlockedError:
        raise HTTPException(status_code=403, detail="This site cannot be analyzed")
    except FirecrawlTimeoutError:
        raise HTTPException(status_code=504, detail="Page took too long to load")
    except FirecrawlError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))


def _count_document_elements(ast) -> dict:
    """Count various elements in a DocumentAST."""
    from seo_optimizer.ingestion.models import NodeType

    word_count = 0
    paragraph_count = 0
    heading_count = 0
    has_faq = False

    for node in ast.nodes:
        if node.node_type == NodeType.HEADING:
            heading_count += 1
            text_lower = node.text_content.lower()
            if "faq" in text_lower or "frequently asked" in text_lower:
                has_faq = True
        elif node.node_type == NodeType.PARAGRAPH:
            paragraph_count += 1

        # Count words in all content
        word_count += len(node.text_content.split())

    return {
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "heading_count": heading_count,
        "has_faq": has_faq,
    }


# Endpoints

@router.post("/scrape", response_model=UrlScrapeResponse)
async def scrape_url(request: UrlScrapeRequest):
    """
    Scrape a URL and return basic content information.

    This is a lightweight endpoint for previewing URL content before analysis.
    """
    url = str(request.url)

    try:
        ast, response = await _scrape_url(url)
        counts = _count_document_elements(ast)

        return UrlScrapeResponse(
            success=True,
            url=url,
            title=response.metadata.title,
            word_count=counts["word_count"],
            paragraph_count=counts["paragraph_count"],
            heading_count=counts["heading_count"],
            has_faq=counts["has_faq"],
            description=response.metadata.description,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scrape failed: {str(e)}")


@router.post("/analyze", response_model=UrlAnalyzeResponse)
async def analyze_url(request: UrlAnalyzeRequest):
    """
    Analyze a URL for SEO and AI readiness.

    Returns GEO-Metric scores and optimization recommendations.
    """
    url = str(request.url)

    try:
        from seo_optimizer.analysis.analyzer import ContentAnalyzer, AnalyzerConfig
        from seo_optimizer.analysis.models import KeywordConfig

        ast, response = await _scrape_url(url)

        # Create keyword config
        keyword_config = KeywordConfig(
            primary_keyword=request.primary_keyword or "content",
            secondary_keywords=request.secondary_keywords or [],
        )

        # Create analyzer config
        analyzer_config = AnalyzerConfig(
            enable_semantic_analysis=False,
            enable_entity_extraction=False,
        )

        # Analyze content
        analyzer = ContentAnalyzer(config=analyzer_config)
        result = analyzer.analyze(ast, keywords=keyword_config)

        # Format response
        stats = result.document_stats
        geo = result.geo_score

        # Build issues list
        issues_list = []
        for issue in geo.all_issues:
            issues_list.append({
                "type": issue.category.value if hasattr(issue, 'category') else "unknown",
                "severity": issue.severity.value if hasattr(issue, 'severity') else "info",
                "message": issue.message if hasattr(issue, 'message') else str(issue),
                "location": getattr(issue, 'location', None),
            })

        # Build recommendations list
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
                recs_list.append({
                    "category": rec.category.value if hasattr(rec.category, 'value') else str(rec.category),
                    "priority": rec.priority.value if hasattr(rec.priority, 'value') else str(rec.priority),
                    "description": rec.description,
                    "impact": rec.impact,
                })

        return UrlAnalyzeResponse(
            success=True,
            url=url,
            title=response.metadata.title,
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

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Service not fully configured: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/optimize", response_model=UrlOptimizeResponse)
async def optimize_url(request: UrlOptimizeRequest):
    """
    Optimize URL content for SEO and AI readiness.

    Returns optimized content as markdown with change indicators.
    """
    url = str(request.url)

    try:
        from seo_optimizer.optimization.pipeline import OptimizationPipeline
        from seo_optimizer.optimization.models import OptimizationConfig
        from seo_optimizer.output.markdown_writer import write_markdown, MarkdownWriterConfig

        ast, response = await _scrape_url(url)

        # Create optimization config
        config = OptimizationConfig(
            primary_keyword=request.primary_keyword,
            secondary_keywords=request.secondary_keywords or [],
            brand_name=request.brand_name,
            inject_keywords=request.inject_keywords,
            generate_faq=request.generate_faq,
            faq_count=request.faq_count,
            improve_readability=request.improve_readability,
            optimize_headings=request.optimize_headings,
            min_keyword_density=request.min_keyword_density,
            max_keyword_density=request.max_keyword_density,
        )

        # Run optimization pipeline
        pipeline = OptimizationPipeline(config)
        result = pipeline.optimize(ast)

        # Generate output
        if request.output_format == "markdown":
            # Build change map for markdown output
            change_map = {}
            if result.optimization_result:
                for change in result.optimization_result.changes:
                    if hasattr(change, 'node_id') and change.node_id:
                        change_map[change.node_id] = "new"

            md_config = MarkdownWriterConfig(new_content_indicator="comments")
            optimized_content = write_markdown(
                result.optimized_ast if result.optimized_ast else ast,
                change_map,
                md_config,
            )
        else:
            # For DOCX, we'll handle it in the download endpoint
            optimized_content = "Use /optimize/download endpoint for DOCX output"

        return UrlOptimizeResponse(
            success=result.success,
            original_url=url,
            original_geo_score=result.optimization_result.original_geo_score if result.optimization_result else 0.0,
            optimized_geo_score=result.optimization_result.optimized_geo_score if result.optimization_result else 0.0,
            improvement=result.optimization_result.geo_improvement if result.optimization_result else 0.0,
            optimized_content=optimized_content,
            changes_count=len(result.optimization_result.changes) if result.optimization_result else 0,
            changes=[
                {
                    "type": change.change_type.value,
                    "location": change.location,
                    "original": change.original[:100] + "..." if len(change.original) > 100 else change.original,
                    "optimized": change.optimized[:100] + "..." if len(change.optimized) > 100 else change.optimized,
                    "reason": change.reason,
                    "impact_score": change.impact_score,
                }
                for change in (result.optimization_result.changes if result.optimization_result else [])
            ],
            warnings=result.warnings,
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Service not fully configured: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/optimize/download")
async def optimize_url_download(request: UrlOptimizeRequest):
    """
    Optimize URL content and return as a downloadable DOCX file.

    New content is highlighted in green.
    """
    url = str(request.url)

    try:
        from seo_optimizer.optimization.pipeline import OptimizationPipeline
        from seo_optimizer.optimization.models import OptimizationConfig
        from seo_optimizer.output.docx_writer import DocxWriter

        ast, response = await _scrape_url(url)

        # Create optimization config
        config = OptimizationConfig(
            primary_keyword=request.primary_keyword,
            secondary_keywords=request.secondary_keywords or [],
            brand_name=request.brand_name,
            inject_keywords=request.inject_keywords,
            generate_faq=request.generate_faq,
            faq_count=request.faq_count,
            improve_readability=request.improve_readability,
            optimize_headings=request.optimize_headings,
            min_keyword_density=request.min_keyword_density,
            max_keyword_density=request.max_keyword_density,
        )

        # Run optimization pipeline
        pipeline = OptimizationPipeline(config)
        result = pipeline.optimize(ast)

        if not result.success:
            raise HTTPException(status_code=500, detail="Optimization failed")

        # Generate DOCX from optimized AST
        writer = DocxWriter()
        output_stream = io.BytesIO()

        optimized_ast = result.optimized_ast if result.optimized_ast else ast
        writer.write_to_stream(
            optimized_ast,
            output_stream,
            highlight_changes=True,
            change_map=result.change_map or {},
        )
        output_stream.seek(0)

        # Generate filename from URL
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or "page"
        path_part = parsed_url.path.strip("/").replace("/", "_")[:30] or "content"
        output_filename = f"{hostname}_{path_part}_optimized.docx"

        return StreamingResponse(
            output_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"'
            }
        )

    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"Service not fully configured: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

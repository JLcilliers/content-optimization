"""
Content optimization endpoints.
"""

import io
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add src to path for imports - must happen before any seo_optimizer imports
_src_path = str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

router = APIRouter()


class OptimizationRequest(BaseModel):
    """Request model for optimization settings."""

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


class OptimizationResponse(BaseModel):
    """Response model for optimization results."""

    success: bool
    original_geo_score: float
    optimized_geo_score: float
    improvement: float
    changes_count: int
    changes: List[dict]
    warnings: List[str]


class OptimizationPreviewResponse(BaseModel):
    """Response model for optimization preview."""

    original_geo_score: float
    estimated_geo_score: float
    estimated_improvement: float
    proposed_changes: List[dict]
    faq_preview: Optional[List[dict]] = None


@router.post("/document", response_model=OptimizationResponse)
async def optimize_document(
    file: UploadFile = File(...),
    primary_keyword: str = Form(...),
    secondary_keywords: Optional[str] = Form(None),
    brand_name: Optional[str] = Form(None),
    inject_keywords: bool = Form(True),
    generate_faq: bool = Form(True),
    faq_count: int = Form(5),
    improve_readability: bool = Form(True),
    optimize_headings: bool = Form(True),
    min_keyword_density: float = Form(1.0),
    max_keyword_density: float = Form(2.5),
):
    """
    Optimize a DOCX document for SEO and AI readiness.

    Returns optimization results with detailed change information.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only .docx files are supported"
        )

    try:
        # Lazy imports to allow app startup even if modules have issues
        from seo_optimizer.ingestion.docx_parser import DocxParser
        from seo_optimizer.optimization.pipeline import OptimizationPipeline
        from seo_optimizer.optimization.models import OptimizationConfig

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

        # Create optimization config
        config = OptimizationConfig(
            primary_keyword=primary_keyword,
            secondary_keywords=sec_keywords,
            brand_name=brand_name,
            inject_keywords=inject_keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
            improve_readability=improve_readability,
            optimize_headings=optimize_headings,
            min_keyword_density=min_keyword_density,
            max_keyword_density=max_keyword_density,
        )

        # Run optimization pipeline
        pipeline = OptimizationPipeline(config)
        result = pipeline.optimize(ast)

        return OptimizationResponse(
            success=result.success,
            original_geo_score=result.optimization_result.original_geo_score if result.optimization_result else 0.0,
            optimized_geo_score=result.optimization_result.optimized_geo_score if result.optimization_result else 0.0,
            improvement=result.optimization_result.geo_improvement if result.optimization_result else 0.0,
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

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not fully configured: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/document/download")
async def optimize_and_download(
    file: UploadFile = File(...),
    primary_keyword: str = Form(...),
    secondary_keywords: Optional[str] = Form(None),
    brand_name: Optional[str] = Form(None),
    inject_keywords: bool = Form(True),
    generate_faq: bool = Form(True),
    faq_count: int = Form(5),
    improve_readability: bool = Form(True),
    optimize_headings: bool = Form(True),
    min_keyword_density: float = Form(1.0),
    max_keyword_density: float = Form(2.5),
):
    """
    Optimize a DOCX document and return the optimized file.

    New content is highlighted in green.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only .docx files are supported"
        )

    try:
        # Lazy imports to allow app startup even if modules have issues
        from seo_optimizer.ingestion.docx_parser import DocxParser
        from seo_optimizer.optimization.pipeline import OptimizationPipeline
        from seo_optimizer.optimization.models import OptimizationConfig
        from seo_optimizer.output.docx_writer import DocxWriter

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

        # Create optimization config
        config = OptimizationConfig(
            primary_keyword=primary_keyword,
            secondary_keywords=sec_keywords,
            brand_name=brand_name,
            inject_keywords=inject_keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
            improve_readability=improve_readability,
            optimize_headings=optimize_headings,
            min_keyword_density=min_keyword_density,
            max_keyword_density=max_keyword_density,
        )

        # Run optimization pipeline
        pipeline = OptimizationPipeline(config)
        result = pipeline.optimize(ast)

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail="Optimization failed"
            )

        # Generate output document
        writer = DocxWriter()
        output_stream = io.BytesIO()
        writer.write_to_stream(
            result.optimized_ast,
            output_stream,
            result.change_map,
            highlight_new=True,
        )
        output_stream.seek(0)

        # Generate filename
        original_name = Path(file.filename).stem
        output_filename = f"{original_name}_optimized.docx"

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
        raise HTTPException(
            status_code=503,
            detail=f"Service not fully configured: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/preview", response_model=OptimizationPreviewResponse)
async def preview_optimization(
    file: UploadFile = File(...),
    primary_keyword: str = Form(...),
    secondary_keywords: Optional[str] = Form(None),
    brand_name: Optional[str] = Form(None),
    inject_keywords: bool = Form(True),
    generate_faq: bool = Form(True),
    faq_count: int = Form(5),
):
    """
    Preview optimization changes without applying them.

    Useful for reviewing proposed changes before committing.
    """
    if not file.filename.endswith(".docx"):
        raise HTTPException(
            status_code=400,
            detail="Only .docx files are supported"
        )

    try:
        # Lazy imports to allow app startup even if modules have issues
        from seo_optimizer.ingestion.docx_parser import DocxParser
        from seo_optimizer.optimization.pipeline import OptimizationPipeline
        from seo_optimizer.optimization.models import OptimizationConfig

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

        # Create optimization config
        config = OptimizationConfig(
            primary_keyword=primary_keyword,
            secondary_keywords=sec_keywords,
            brand_name=brand_name,
            inject_keywords=inject_keywords,
            generate_faq=generate_faq,
            faq_count=faq_count,
        )

        # Run optimization pipeline in preview mode
        pipeline = OptimizationPipeline(config)
        result = pipeline.optimize(ast)

        # Format FAQ preview if generated
        faq_preview = None
        if result.optimization_result and result.optimization_result.generated_faq:
            faq_preview = [
                {
                    "question": faq.question,
                    "answer": faq.answer[:200] + "..." if len(faq.answer) > 200 else faq.answer,
                }
                for faq in result.optimization_result.generated_faq
            ]

        return OptimizationPreviewResponse(
            original_geo_score=result.optimization_result.original_geo_score if result.optimization_result else 0.0,
            estimated_geo_score=result.optimization_result.optimized_geo_score if result.optimization_result else 0.0,
            estimated_improvement=result.optimization_result.geo_improvement if result.optimization_result else 0.0,
            proposed_changes=[
                {
                    "type": change.change_type.value,
                    "location": change.location,
                    "description": change.reason,
                    "impact_score": change.impact_score,
                }
                for change in (result.optimization_result.changes if result.optimization_result else [])
            ],
            faq_preview=faq_preview,
        )

    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service not fully configured: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Preview failed: {str(e)}"
        )

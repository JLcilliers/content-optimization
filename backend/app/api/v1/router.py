"""
API v1 router.
"""

from fastapi import APIRouter

from backend.app.api.v1.endpoints import analyze, optimize, health, url

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
api_router.include_router(optimize.router, prefix="/optimize", tags=["Optimization"])
api_router.include_router(url.router, prefix="/url", tags=["URL"])

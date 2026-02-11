"""API routes"""
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Optional
import logging
from utils.metrics import track_query_metrics
from api.models import QueryRequest, QueryResponse, HealthResponse, Source
from services.rag_service import RAGService

from slowapi.errors import RateLimitExceeded
from api.rate_limiter import limiter 

#Logger Initialization
logger = logging.getLogger(__name__)

#Router Initialization
router = APIRouter()

# Dependency to get RAG service
def get_rag_service():
    from api.main import rag_service
    return rag_service

@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute") 
@track_query_metrics
async def query_documentation(
    request: Request,
    req: QueryRequest,
    rag: RAGService = Depends(get_rag_service)
):
    """Query the documentation"""
    try:
        logger.info(f"Query: {req.query}")
        
        result = rag.query(
            query=req.query,
            top_k=req.top_k,
            library_filter=req.library
        )
        
        return QueryResponse(
            answer=result['answer'],
            sources=[Source(**s) for s in result['sources']],
            response_time=result['response_time'],
            cached=False
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check(rag: RAGService = Depends(get_rag_service)):
    """Health check"""
    try:
        collection_info = rag.vector_store.get_collection_info()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            collection_info=collection_info
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")

@router.get("/libraries")
async def list_libraries(rag: RAGService = Depends(get_rag_service)):
    """Get indexed libraries (simplified for now)"""
    info = rag.vector_store.get_collection_info()
    return {
        "total_documents": info['points_count'],
        "collection": info['name']
    }

@router.get("/cache/stats")
async def get_cache_stats(rag: RAGService = Depends(get_rag_service)):
    """Get cache statistics"""
    if not rag.cache:
        raise HTTPException(status_code=404, detail="Cache not enabled")
    
    return rag.cache.get_stats()

@router.post("/cache/invalidate")
async def invalidate_cache(
    pattern: str = "*",
    rag: RAGService = Depends(get_rag_service)
):
    """Invalidate cache (admin only in production)"""
    if not rag.cache:
        raise HTTPException(status_code=404, detail="Cache not enabled")
    
    rag.cache.invalidate(pattern)
    return {"status": "success", "pattern": pattern}


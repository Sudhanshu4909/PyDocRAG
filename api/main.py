from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routes import router
from api.rate_limiter import limiter, rate_limit_exceeded_handler  # Import here
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from services.rag_service import RAGService
from config.settings import get_settings
from utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service: RAGService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    global rag_service
    
    logger.info("Starting up...")
    rag_service = RAGService()
    logger.info("Ready to serve requests")
    
    yield
    
    logger.info("Shutting down...")

# Create app
settings = get_settings()
app = FastAPI(
    title="PyDocRAG API",
    description="Production RAG for Python Documentation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Add rate limiter to app state (HERE in main.py)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["rag"])

@app.get("/")
async def root():
    return {
        "name": "PyDocRAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }
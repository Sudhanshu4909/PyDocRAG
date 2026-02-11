"""Pydantic models for API"""
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    library: Optional[str] = None

class Source(BaseModel):
    library: str
    title: str
    url: str
    score: float
    doc_type: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    response_time: float
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    version: str
    collection_info: dict
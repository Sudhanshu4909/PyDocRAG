"""Main RAG service - refactored from Rag_pipeline.py"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import logging

from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel
from core.llm_providers import BaseLLM, OllamaLLM
from config.settings import get_settings
from utils.cache import RedisCache
from config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    content: str
    library: str
    url: str
    title: str
    score: float
    doc_type: str
    code_blocks: List[str]

class RAGService:
    def __init__(self):
        settings = get_settings()
        
        # Existing initialization
        self.vector_store = VectorStore()
        self.embeddings = EmbeddingModel()
        self.llm = self._init_llm()
        
        # Add cache
        if settings.ENABLE_CACHE:
            self.cache = RedisCache(
                redis_url=settings.REDIS_URL,
                ttl=settings.CACHE_TTL
            )
            logger.info("Cache enabled")
        else:
            self.cache = None
            logger.info("Cache disabled")
    
    def query(self, query: str, top_k: int = None, library_filter: Optional[str] = None) -> Dict:
        """Query with caching"""
        
        # Check cache if enabled
        if self.cache:
            cache_key = self.cache._make_key("query", query, top_k or get_settings().DEFAULT_TOP_K, library_filter or "")
            cached = self.cache.get(cache_key)
            if cached:
                cached['cached'] = True
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached
        
        # Process query (existing logic)
        start_time = time.time()
        contexts = self.retrieve(query, top_k, library_filter)
        answer = self.generate_answer(query, contexts)
        
        result = {
            'answer': answer,
            'sources': [...],  # existing code
            'response_time': time.time() - start_time,
            'cached': False
        }
        
        # Cache result
        if self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def _init_llm(self) -> BaseLLM:
        """Initialize LLM based on config"""
        settings = get_settings()
        
        if settings.LLM_PROVIDER == "ollama":
            return OllamaLLM(
                model=settings.LLM_MODEL,
                base_url=settings.LLM_BASE_URL
            )
        # Add other providers as needed
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        library_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents"""
        
        settings = get_settings()
        if top_k is None:
            top_k = settings.DEFAULT_TOP_K
        
        # Generate query embedding
        query_vector = self.embeddings.encode(query)[0]
        
        # Search vector store
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            library_filter=library_filter
        )
        
        # Convert to RetrievalResult
        return [
            RetrievalResult(
                content=r['content'],
                library=r['library'],
                url=r['url'],
                title=r['title'],
                score=r['score'],
                doc_type=r['doc_type'],
                code_blocks=r['code_blocks']
            )
            for r in results
        ]
    
    def generate_answer(
        self,
        query: str,
        contexts: List[RetrievalResult]
    ) -> str:
        """Generate answer from contexts"""
        
        # Build context
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(f"""
[{i}] {ctx.library} - {ctx.title}
{ctx.content}
URL: {ctx.url}
""")
        
        context_text = "\n\n".join(context_parts)
        
        # System prompt
        system_prompt = """You are a helpful Python documentation assistant.

Rules:
1. Answer based ONLY on the provided context
2. Cite sources using [N] notation
3. Provide working code examples when relevant
4. If you're unsure, say so
5. Be concise but complete"""
        
        # User prompt
        user_prompt = f"""Context from documentation:

{context_text}

Question: {query}

Answer:"""
        
        # Generate
        answer = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        return answer
    
    def query(
        self,
        query: str,
        top_k: int = None,
        library_filter: Optional[str] = None
    ) -> Dict:
        """End-to-end query processing"""
        
        start_time = time.time()
        
        # Retrieve
        contexts = self.retrieve(query, top_k, library_filter)
        
        logger.info(f"Retrieved {len(contexts)} documents for query: {query[:50]}...")
        
        # Generate
        answer = self.generate_answer(query, contexts)
        
        response_time = time.time() - start_time
        
        return {
            'answer': answer,
            'sources': [
                {
                    'library': ctx.library,
                    'title': ctx.title,
                    'url': ctx.url,
                    'score': ctx.score,
                    'doc_type': ctx.doc_type
                }
                for ctx in contexts
            ],
            'response_time': response_time
        }
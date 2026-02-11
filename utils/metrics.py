"""Prometheus metrics for monitoring"""
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import time
import functools

# Query metrics
query_total = Counter(
    'rag_queries_total',
    'Total number of queries',
    ['status', 'cached']
)

query_duration = Histogram(
    'rag_query_duration_seconds',
    'Query processing time',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Retrieval metrics
retrieval_scores = Histogram(
    'rag_retrieval_scores',
    'Document retrieval scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

documents_retrieved = Histogram(
    'rag_documents_retrieved',
    'Number of documents retrieved',
    buckets=[1, 3, 5, 10, 20, 50]
)

# Cache metrics
cache_hits = Counter('rag_cache_hits_total', 'Cache hits')
cache_misses = Counter('rag_cache_misses_total', 'Cache misses')

# LLM metrics
llm_calls = Counter('rag_llm_calls_total', 'LLM API calls', ['provider'])
llm_tokens = Counter('rag_llm_tokens_total', 'LLM tokens used', ['type'])  # type: prompt/completion

# System metrics
active_queries = Gauge('rag_active_queries', 'Currently processing queries')
collection_size = Gauge('rag_collection_size', 'Documents in vector store')

# Application info
app_info = Info('rag_application', 'Application information')
app_info.info({
    'version': '1.0.0',
    'environment': 'production'
})

def track_query_metrics(func):
    """Decorator to track query metrics"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        active_queries.inc()
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            cached = getattr(result, "cached", False)
            query_total.labels(status='success', cached=cached).inc()
            
            # Record duration
            duration = time.time() - start_time
            query_duration.observe(duration)
            
            # Record retrieval metrics
            if 'sources' in result:
                documents_retrieved.observe(len(result['sources']))
                for source in result['sources']:
                    retrieval_scores.observe(source['score'])
            
            # Record cache metrics
            if cached:
                cache_hits.inc()
            else:
                cache_misses.inc()
            
            return result
            
        except Exception as e:
            query_total.labels(status='error', cached=False).inc()
            raise
        finally:
            active_queries.dec()
    
    return wrapper

def track_llm_call(provider: str, prompt_tokens: int, completion_tokens: int):
    """Track LLM usage"""
    llm_calls.labels(provider=provider).inc()
    llm_tokens.labels(type='prompt').inc(prompt_tokens)
    llm_tokens.labels(type='completion').inc(completion_tokens)

def update_collection_size(size: int):
    """Update collection size gauge"""
    collection_size.set(size)
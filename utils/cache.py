"""Redis caching with advanced features"""
import redis
import json
import hashlib
from typing import Optional, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    """Advanced Redis caching for RAG"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        logger.info(f"Connected to Redis at {redis_url}")
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key"""
        content = f"{prefix}:" + ":".join(str(arg) for arg in args)
        if kwargs:
            content += ":" + json.dumps(kwargs, sort_keys=True)
        return f"rag:{hashlib.sha256(content.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        try:
            data = self.client.get(key)
            if data:
                logger.debug(f"Cache HIT: {key[:16]}...")
                return json.loads(data)
            logger.debug(f"Cache MISS: {key[:16]}...")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set in cache"""
        try:
            ttl = ttl or self.ttl
            self.client.setex(key, ttl, json.dumps(value))
            logger.debug(f"Cache SET: {key[:16]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def invalidate(self, pattern: str):
        """Invalidate cache by pattern"""
        try:
            keys = self.client.keys(f"rag:{pattern}*")
            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def cache_query(self, ttl: Optional[int] = None):
        """Decorator for caching query results"""
        def decorator(func):
            @wraps(func)
            def wrapper(query: str, *args, **kwargs):
                # Generate cache key
                key = self._make_key("query", query, *args, **kwargs)
                
                # Try cache
                cached = self.get(key)
                if cached:
                    cached['cached'] = True
                    return cached
                
                # Execute function
                result = func(query, *args, **kwargs)
                
                # Cache result
                if result and 'answer' in result:
                    self.set(key, result, ttl)
                    result['cached'] = False
                
                return result
            
            return wrapper
        return decorator
    
    def cache_embedding(self, ttl: int = 86400):
        """Decorator for caching embeddings (24h default)"""
        def decorator(func):
            @wraps(func)
            def wrapper(text: str, *args, **kwargs):
                key = self._make_key("embedding", text)
                
                cached = self.get(key)
                if cached:
                    return cached
                
                result = func(text, *args, **kwargs)
                self.set(key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        try:
            info = self.client.info()
            return {
                'total_keys': self.client.dbsize(),
                'memory_used': info['used_memory_human'],
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, info: dict) -> float:
        """Calculate cache hit rate"""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
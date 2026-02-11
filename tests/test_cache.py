#!/usr/bin/env python3
"""Test cache implementation"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache import RedisCache
import time

def test_cache():
    """Test cache functionality"""
    
    print("Testing Cache Implementation")
    print("=" * 60)
    
    try:
        # Initialize cache
        print("\n1. Initializing cache...")
        cache = RedisCache(redis_url="redis://localhost:6379", ttl=60)
        print("   ✓ Cache initialized")
        
        # Test 2: Basic set/get
        print("\n2. Testing basic set/get...")
        test_key = "test:query:123"
        test_value = {"answer": "Hello World", "score": 0.95}
        
        cache.set(test_key, test_value)
        retrieved = cache.get(test_key)
        
        assert retrieved == test_value, "Retrieved value doesn't match!"
        print("   ✓ Set/Get working")
        
        # Test 3: Cache miss
        print("\n3. Testing cache miss...")
        missing = cache.get("nonexistent:key:999")
        assert missing is None, "Should return None for missing key"
        print("   ✓ Cache miss handled correctly")
        
        # Test 4: Cache key generation
        print("\n4. Testing cache key generation...")
        key1 = cache._make_key("query", "test query", top_k=5)
        key2 = cache._make_key("query", "test query", top_k=5)
        key3 = cache._make_key("query", "different query", top_k=5)
        
        assert key1 == key2, "Same inputs should generate same key"
        assert key1 != key3, "Different inputs should generate different keys"
        print("   ✓ Key generation working")
        
        # Test 5: TTL expiration (short test)
        print("\n5. Testing TTL expiration...")
        short_ttl_cache = RedisCache(redis_url="redis://localhost:6379", ttl=2)
        short_ttl_cache.set("ttl:test", {"data": "expires soon"}, ttl=2)
        
        # Should exist immediately
        assert short_ttl_cache.get("ttl:test") is not None
        print("   ✓ Value exists immediately")
        
        # Wait and check expiration
        print("   Waiting 3 seconds for expiration...")
        time.sleep(3)
        assert short_ttl_cache.get("ttl:test") is None
        print("   ✓ Value expired correctly")
        
        # Test 6: Cache stats
        print("\n6. Testing cache statistics...")
        stats = cache.get_stats()
        print(f"   Total keys: {stats.get('total_keys', 'N/A')}")
        print(f"   Memory used: {stats.get('memory_used', 'N/A')}")
        print(f"   Hit rate: {stats.get('hit_rate', 0):.2f}%")
        print("   ✓ Stats retrieved")
        
        # Test 7: Cache invalidation
        print("\n7. Testing cache invalidation...")
        cache.set("test:item:1", {"data": "item1"})
        cache.set("test:item:2", {"data": "item2"})
        cache.set("other:item:3", {"data": "item3"})
        
        cache.invalidate("test:item")
        
        assert cache.get("test:item:1") is None
        assert cache.get("test:item:2") is None
        assert cache.get("other:item:3") is not None  # Should still exist
        print("   ✓ Invalidation working")
        
        print("\n" + "=" * 60)
        print("✓ All cache tests passed!")
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cache()
    sys.exit(0 if success else 1)
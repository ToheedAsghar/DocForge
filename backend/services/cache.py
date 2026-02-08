"""
CACHE SERVICE - REDIS BASED CACHING FOR QUERY RESULTS

JOB: Cache RAG query results to avoid re-processing identical query.

STRATEGY:
1. key: SHA256 hash of the query text (case-insensitive)
2. Value: Complete reuslt dictionary (answer, metadata, etc.)
3. TTL: Configurable (default 1 hour)
4. namespace: 'query': prefix for organization
"""

import json
import redis
import hashlib
from backend.config import settings
from typing import Optional, Dict, Any

class CacheService:
    """
    Redis-based caching service for RAG query results.

    PROCESS:
    When user enters 'What's this document about?' we:
    1. Hash the query -> 'abc123...
    2. Checks redis for key 'query:abc123...'
    3. If found -> return cached answer (instant)
    4. If not found -> run the full pipeline -> cache the result

    """

    def __init__(self) -> None:
        self.enabled = False
        self.redis_client = None

        if not settings.CACHE_ENABLED:
            print(f"[INFO]\tCache is disabled in config. Disabling cache service.")
            return
        
        try:
            # create redis client
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses = True,
                socket_connect_timeout = 2, # don't wait too long
                socket_timeout = 2
            )

            # test connection
            self.redis_client.ping()

            # if we got here, Redis is working
            self.enabled = True
            print(f"[INFO]\tCache service initialized and connected to Redis at {settings.REDIS_URL}")

        except redis.ConnectionError as e:
            print(f"[ERROR]\tFailed to connect to Redis at {settings.REDIS_URL}: {str(e)}")
            print(f"[WARNING]\tRunning with cache (queries will be show)")
            self.enabled = False
            self.redis_client = None

        except Exception as e:
            print(f"[ERROR]\tUnexpected error initializing cache service: {str(e)}")
            print(f"[WARNING]\tRunning with cache (queries will be shown)")
            self.enabled = False
            self.redis_client = None

    def _generate_key(self, query: str) -> str:
        """
        Generate Cache key from query text.
        EXAMPLE:
        - query: "What is The Document About?"
        - normalized: 'what is the document about?'
        - hash: 'abc123...'
        - key: 'query:abc123...'
        """

        normalized = query.lower().strip()

        hash_hex = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
        key = f"query:{hash_hex}"
        return key
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        RETRIEVE CACHED RESULT FOR A QUERY
        """
        
        # if cache is disabled, return None
        if not self.enabled:
            return None

        try:
            key = self._generate_key(query)
            cached_data = self.redis_client.get(key)

            if cached_data:
                print(f"[CACHE HIT]\tQuery: {query[:50]}...")
                result = json.loads(cached_data)
                result['_from_cache'] = True # add metadata
                return result

            else:
                print(f"[CACHE MISS]\tQuery: {query[:50]}...")
                return None
        except json.JSONDecodeError as e:
            print(f"[CACHE ERROR]\tFailed to parse cached data for query: {query[:50]}: {str(e)}")
            
            # delete the corrupted entry
            try:
                key = self._generate_key(query)
                self.redis_client.delete(key)
            except:
                pass

            return None
        
        except Exception as e:
            print(f"[CACHE ERROR]\tUnexpected error retrieving cached data for query: {query[:50]}: {str(e)}")
            return None

    def set(self, query: str, result: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store Query Result in Cache"""

        if not self.enabled:
            return False

        try:
            key = self._generate_key(query)
            ttl = ttl or settings.CACHE_TTL_SECONDS
            result_cpy = {k: v for k,v in result.items() if not k.startswith('_')}

            # serialize to JSON
            json_data = json.dumps(result_cpy)

            # store in Redis with expiration
            self.redis_client.setex(
                name=key,
                value=json_data,
                time=ttl
            )

            print(f"[CACHE SET]\tQuery: {query[:50]}...")
            return True

        except Exception as e:
            print(f"[CACHE ERROR]\tUnexpected error storing query result in cache: {query[:50]}: {str(e)}")
            return False
    def delete(self, query: str) -> bool:
        """
        Delete the cached results for a specific query.

        WHY: Maybe the answer changed or user wants fresh results
        """

        if not self.enabled:
            return False
        
        try:
            key = self._generate_key(query)
            deleted = self.redis_client.delete(key)

            if deleted:
                print(f"[CACHE DELETED]\tQuery: {query[:50]}...")
                return True
            else:
                print(f"[CACHE MISS]\tQuery: {query[:50]}... Not found in cache")
                return False
        except Exception as e:
            print(f"[CACHE ERROR]\tUnexpected error deleting query result from cache: {query[:50]}: {str(e)}")
            return False
        
    def clear_all(self) -> int:
        """
        Clear ALL cached queries
        WHY: Fresh start or documents were updated
        """

        if not self.enabled: 
            return 0

        try:
            keys = self.redis_client.keys('query:*')
            if keys:
                deleted = self.redis_client.delete(*keys)
                print(f"[CACHE CLEARED]\tCleared {len(keys)} cached queries")
                return deleted
            else:
                print(f"[CACHE EMPTY]\tNo cached queries found")
                return 0

        except Exception as e:
            print(f"[CACHE ERROR]\tUnexpected error clearing all cached queries: {str(e)}")
            return 0

# --- GLOBAL SINGLETON INSTANCE --- #
_cache_service = None

def get_cache_service() -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service

# --- CONVENIENCE FUNCTION --- #
def cache_query_result(query: str, result: Dict[str, Any]) -> bool:
    cache = get_cache_service()
    return cache.set(query, result)

def get_cached_result(query: str) -> Optional[Dict[str: Any]]:
    cache = get_cache_service()
    return cache.get(query)

def clear_cache() -> int:
    cache = get_cache_service()
    return cache.clear_all()

        

    
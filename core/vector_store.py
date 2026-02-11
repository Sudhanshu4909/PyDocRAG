"""Vector store wrapper for Qdrant - FIXED VERSION"""
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from config.settings import get_settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations"""
    
    def __init__(self):
        settings = get_settings()
        
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.EMBEDDING_DIMENSION
        
        logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")
    
    def create_collection(self, recreate: bool = False):
        """Create or recreate collection"""
        try:
            if recreate:
                # Delete if exists
                try:
                    self.client.delete_collection(collection_name=self.collection_name)
                    logger.info(f"Deleted existing collection: {self.collection_name}")
                except Exception:
                    pass  # Collection doesn't exist, that's fine
                
                # Create new
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                # Check if collection exists
                collections = self.client.get_collections()
                collection_names = [c.name for c in collections.collections]
                
                if self.collection_name not in collection_names:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection: {self.collection_name}")
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def upsert_documents(self, documents: List[Dict]):
        """Bulk insert documents"""
        try:
            points = [
                PointStruct(
                    id=doc['id'],
                    vector=doc['vector'],
                    payload={
                        'content': doc['content'],
                        'library': doc['library'],
                        'url': doc['url'],
                        'title': doc['title'],
                        'doc_type': doc['doc_type'],
                        'code_blocks': doc.get('code_blocks', []),
                        'metadata': doc.get('metadata', {})
                    }
                )
                for doc in documents
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} documents")
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        library_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search for similar documents"""
        try:
            # Build filter
            search_filter = None
            if library_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="library",
                            match=MatchValue(value=library_filter)
                        )
                    ]
                )
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter
            )
            
            return [
                {
                    'id': hit.id,
                    'score': hit.score,
                    'content': hit.payload['content'],
                    'library': hit.payload['library'],
                    'url': hit.payload['url'],
                    'title': hit.payload['title'],
                    'doc_type': hit.payload['doc_type'],
                    'code_blocks': hit.payload.get('code_blocks', []),
                    'metadata': hit.payload.get('metadata', {})
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics - with fallback for version compatibility"""
        try:
            # Try the normal way first
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count if hasattr(info, 'vectors_count') else info.points_count,
                'status': info.status
            }
        except Exception as e:
            logger.warning(f"Standard get_collection failed (likely version mismatch): {e}")
            logger.info("Using fallback method to get collection info...")
            
            # Fallback: use count instead
            try:
                # Get count using scroll
                count_result = self.client.count(
                    collection_name=self.collection_name
                )
                
                return {
                    'name': self.collection_name,
                    'points_count': count_result.count,
                    'vectors_count': count_result.count,
                    'status': 'active'  # Assume active if we can count
                }
            except Exception as e2:
                logger.error(f"Fallback method also failed: {e2}")
                # Last resort - return minimal info
                return {
                    'name': self.collection_name,
                    'points_count': -1,
                    'vectors_count': -1,
                    'status': 'unknown'
                }
    
    def count_documents(self) -> int:
        """Get total document count - more reliable than get_collection_info"""
        try:
            result = self.client.count(collection_name=self.collection_name)
            return result.count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return -1
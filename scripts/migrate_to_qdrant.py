#!/usr/bin/env python3
"""Migrate existing documents to Qdrant - FIXED VERSION"""
import json
import sys
from pathlib import Path
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    """Migrate documents to Qdrant"""
    settings = get_settings()
    
    # Initialize
    logger.info("Initializing components...")
    vector_store = VectorStore()
    embeddings = EmbeddingModel()
    
    # Create collection
    logger.info("Creating collection...")
    vector_store.create_collection(recreate=True)
    
    # Load documents
    docs_dir = Path(settings.DOCS_OUTPUT_DIR)
    if not docs_dir.exists():
        logger.error(f"Docs directory not found: {docs_dir}")
        logger.info("Please run the data collection script first or check DOCS_OUTPUT_DIR in .env")
        return
    
    all_documents = []
    doc_id = 0
    total_docs = 0
    
    # Count total documents first
    logger.info("Counting documents...")
    for library_dir in docs_dir.iterdir():
        if not library_dir.is_dir():
            continue
        chunks_file = library_dir / "chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file) as f:
                total_docs += sum(1 for _ in f)
    
    logger.info(f"Found {total_docs} documents to migrate")
    
    if total_docs == 0:
        logger.warning("No documents found! Please run data collection first.")
        return
    
    # Process each library
    with tqdm(total=total_docs, desc="Migrating documents") as pbar:
        for library_dir in docs_dir.iterdir():
            if not library_dir.is_dir():
                continue
            
            chunks_file = library_dir / "chunks.jsonl"
            if not chunks_file.exists():
                continue
            
            logger.info(f"Processing {library_dir.name}...")
            
            # Load chunks
            chunks = []
            with open(chunks_file) as f:
                for line in f:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
                        continue
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                
                # Get embeddings
                contents = [c['content'] for c in batch]
                vectors = embeddings.encode(contents, batch_size=batch_size)
                
                # Prepare documents
                for chunk, vector in zip(batch, vectors):
                    all_documents.append({
                        'id': doc_id,
                        'vector': vector,
                        'content': chunk['content'],
                        'library': chunk['library'],
                        'url': chunk['url'],
                        'title': chunk['title'],
                        'doc_type': chunk['doc_type'],
                        'code_blocks': chunk.get('code_blocks', []),
                        'metadata': chunk.get('metadata', {})
                    })
                    doc_id += 1
                    pbar.update(1)
                
                # Upload in batches
                if len(all_documents) >= 100:
                    vector_store.upsert_documents(all_documents)
                    all_documents = []
    
    # Upload remaining
    if all_documents:
        vector_store.upsert_documents(all_documents)
    
    # Verify - use simplified check to avoid compatibility issue
    logger.info("Verifying migration...")
    try:
        # Try to get collection info
        info = vector_store.get_collection_info()
        logger.info(f"✓ Migration complete! Total documents: {info['points_count']}")
    except Exception as e:
        # If get_collection_info fails due to version mismatch, do a simple search test
        logger.warning(f"Could not get collection info due to version mismatch: {e}")
        logger.info("Verifying with test search instead...")
        
        # Do a test search to verify
        test_vector = embeddings.encode("test")[0]
        results = vector_store.search(test_vector, top_k=1)
        
        if results:
            logger.info(f"✓ Migration complete! Verified with test search - found {len(results)} result(s)")
            logger.info(f"✓ Total documents migrated: {doc_id}")
        else:
            logger.error("Migration may have failed - no results found in test search")

if __name__ == "__main__":
    migrate()
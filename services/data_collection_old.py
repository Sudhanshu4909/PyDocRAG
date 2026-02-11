"""
PyDocAI - Python Documentation Data Collection System

This module handles scraping and processing documentation from multiple sources:
1. Official Python documentation (docs.python.org)
2. Popular library documentation (Read the Docs, GitHub Pages)
3. PyPI package metadata
4. Code examples from official repos
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class DocumentChunk:
    """Represents a processed documentation chunk"""
    content: str
    library: str
    version: str
    doc_type: str  # 'tutorial', 'api', 'guide', 'example'
    url: str
    title: str
    code_blocks: List[str]
    metadata: Dict
    chunk_id: str
    
    def to_dict(self):
        return asdict(self)


class PythonDocsCollector:
    """Collects documentation from various Python sources"""
    
    # Top Python libraries to index
    CORE_LIBRARIES = [
    # 🔹 Python Standard Library
    'python', 'asyncio', 'multiprocessing', 'threading', 'concurrent',
    'subprocess', 'pathlib', 'typing', 'dataclasses', 'functools',
    'itertools', 'collections', 'logging', 'argparse', 'json', 're',
    'hashlib', 'sqlite3', 'unittest', 'doctest',

    # 🔹 Data Science & Numerical Computing
    'numpy', 'pandas', 'scipy', 'statsmodels', 'sympy',
    'numba', 'cupy', 'polars', 'dask',

    # 🔹 Machine Learning
    'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
    'imbalanced-learn', 'mlxtend', 'optuna', 'hyperopt',

    # 🔹 Deep Learning Frameworks
    'tensorflow', 'keras', 'pytorch', 'torchvision', 'torchaudio',
    'jax', 'flax', 'paddlepaddle',

    # 🔹 NLP & Transformers
    'transformers', 'sentence-transformers', 'spacy', 'nltk',
    'gensim', 'textblob', 'fasttext', 'stanza',

    # 🔹 Generative AI & LLM Ecosystem
    'langchain', 'llama-index', 'haystack', 'autogen',
    'crew-ai', 'semantic-kernel',
    'openai', 'anthropic', 'cohere', 'tiktoken',
    'vllm', 'ctransformers', 'ollama',

    # 🔹 Vector Databases & Retrieval
    'faiss', 'chromadb', 'qdrant-client', 'weaviate-client',
    'milvus', 'pinecone-client', 'pgvector', 'redisvl',

    # 🔹 Computer Vision
    'opencv-python', 'pillow', 'scikit-image', 'albumentations',
    'detectron2', 'mmcv', 'ultralytics', 'mediapipe',

    # 🔹 Speech / Audio AI
    'whisper', 'torchaudio', 'librosa', 'speechrecognition',
    'pyttsx3', 'pyaudio', 'coqui-tts',

    # 🔹 Backend Frameworks & APIs
    'fastapi', 'flask', 'django', 'starlette', 'sanic',
    'falcon', 'litestar', 'bottle',

    # 🔹 Async & Networking
    'aiohttp', 'httpx', 'requests', 'websockets',
    'trio', 'uvloop', 'anyio', 'gevent',

    # 🔹 Scraping & Automation
    'beautifulsoup4', 'lxml', 'scrapy', 'selenium',
    'playwright', 'pyppeteer',

    # 🔹 Databases & ORMs
    'sqlalchemy', 'peewee', 'tortoise-orm',
    'psycopg2', 'asyncpg', 'pymysql', 'motor',
    'redis', 'pymongo', 'cassandra-driver',

    # 🔹 Big Data & Distributed Computing
    'pyspark', 'ray', 'modin', 'rapids',
    'vaex', 'apache-beam',

    # 🔹 MLOps & Experiment Tracking
    'mlflow', 'wandb', 'neptune', 'clearml',
    'kedro', 'dvc',

    # 🔹 Deployment & Serving
    'bentoml', 'tritonclient', 'seldon-core',
    'kserve', 'onnxruntime', 'torchserve',

    # 🔹 Visualization & Dashboards
    'matplotlib', 'seaborn', 'plotly',
    'bokeh', 'altair', 'dash', 'streamlit', 'gradio',

    # 🔹 Cloud SDKs
    'boto3', 'google-cloud-storage', 'google-cloud-aiplatform',
    'azure-storage-blob', 'azure-ai-ml',

    # 🔹 Testing & Dev Tools
    'pytest', 'hypothesis', 'black', 'ruff',
    'mypy', 'isort', 'pre-commit'
]

    
    def __init__(self, output_dir: str = "data/raw_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyDocAI Documentation Collector (Educational Project)'
        })
    
    def collect_all(self, libraries: Optional[List[str]] = None):
        """Collect documentation for all specified libraries"""
        libraries = libraries or self.CORE_LIBRARIES
        
        results = {
            'libraries_processed': [],
            'total_pages': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        for library in libraries:
            print(f"\n{'='*60}")
            print(f"Processing: {library}")
            print(f"{'='*60}")
            
            try:
                lib_result = self.collect_library_docs(library)
                results['libraries_processed'].append(library)
                results['total_pages'] += lib_result['pages']
                results['total_chunks'] += lib_result['chunks']
                print(f"✓ Collected {lib_result['pages']} pages, {lib_result['chunks']} chunks")
            except Exception as e:
                error_msg = f"Error processing {library}: {str(e)}"
                print(f"✗ {error_msg}")
                results['errors'].append(error_msg)
            
            # Rate limiting
            time.sleep(2)
        
        # Save collection summary
        with open(self.output_dir / 'collection_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def collect_library_docs(self, library: str) -> Dict:
        """Collect documentation for a specific library"""
        
        # Get documentation URL
        doc_url = self._get_documentation_url(library)
        if not doc_url:
            raise ValueError(f"Could not find documentation URL for {library}")
        
        print(f"Documentation URL: {doc_url}")
        
        # Create library output directory
        lib_dir = self.output_dir / library
        chunks_file = lib_dir / "chunks.jsonl"

        # Skip if already processed
        if chunks_file.exists():
            print(f"✓ Skipping {library}, already collected")
            return {
                "library": library,
                "pages": 0,
                "chunks": 0,
                "output_dir": str(lib_dir)
            }

        lib_dir.mkdir(exist_ok=True)
        
        # Scrape documentation
        pages = self._scrape_documentation(doc_url, library, lib_dir)
        
        # Process and chunk pages
        chunks = self._process_pages(pages, library)
        
        # Save chunks
        chunks_file = lib_dir / 'chunks.jsonl'
        with open(chunks_file, 'w') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict()) + '\n')
        
        return {
            'library': library,
            'pages': len(pages),
            'chunks': len(chunks),
            'output_dir': str(lib_dir)
        }
    
    def _get_documentation_url(self, library: str) -> Optional[str]:
        """Get the documentation URL for a library"""
        
        # Special cases
        doc_urls = {
            'python': 'https://docs.python.org/3/',
            'numpy': 'https://numpy.org/doc/stable/',
            'pandas': 'https://pandas.pydata.org/docs/',
            'matplotlib': 'https://matplotlib.org/stable/',
            'scikit-learn': 'https://scikit-learn.org/stable/',
            'tensorflow': 'https://www.tensorflow.org/api_docs/python',
            'pytorch': 'https://pytorch.org/docs/stable/',
            'requests': 'https://requests.readthedocs.io/en/latest/',
            'flask': 'https://flask.palletsprojects.com/',
            'django': 'https://docs.djangoproject.com/en/stable/',
            'fastapi': 'https://fastapi.tiangolo.com/',
            'sqlalchemy': 'https://docs.sqlalchemy.org/',
            'pytest': 'https://docs.pytest.org/en/stable/',
            'beautifulsoup4': 'https://www.crummy.com/software/BeautifulSoup/bs4/doc/',
        }
        
        if library in doc_urls:
            return doc_urls[library]
        
        # Try to get from PyPI
        try:
            response = self.session.get(f'https://pypi.org/pypi/{library}/json')
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                
                # Try project_urls first
                project_urls = info.get('project_urls', {})
                for key in ['Documentation', 'Docs', 'documentation', 'docs']:
                    if key in project_urls:
                        return project_urls[key]
                
                # Try home_page
                if info.get('home_page'):
                    return info['home_page']
        except Exception as e:
            print(f"Error fetching PyPI data: {e}")
        
        return None
    
    def _scrape_documentation(self, base_url: str, library: str, 
                             output_dir: Path, max_pages: int = 500) -> List[Dict]:
        """Scrape documentation pages from a base URL"""
        
        visited = set()
        to_visit = [base_url]
        pages = []
        
        domain = urlparse(base_url).netloc
        
        while to_visit and len(pages) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content
                content = self._extract_content(soup, url)
                if not content:
                    continue
                
                pages.append({
                    'url': url,
                    'title': content['title'],
                    'html': content['html'],
                    'text': content['text'],
                    'code_blocks': content['code_blocks']
                })
                
                print(f"  [{len(pages)}/{max_pages}] Scraped: {content['title'][:50]}...")
                
                # Find more links
                for link in soup.find_all('a', href=True):
                    absolute_url = urljoin(url, link['href'])
                    
                    # Only follow links within the same domain
                    if urlparse(absolute_url).netloc == domain:
                        # Remove fragments
                        absolute_url = absolute_url.split('#')[0]
                        if absolute_url not in visited and absolute_url not in to_visit:
                            to_visit.append(absolute_url)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error scraping {url}: {e}")
                continue
        
        return pages
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """Extract meaningful content from a documentation page"""
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Try to find main content area
        main_content = (
            soup.find('main') or 
            soup.find('article') or 
            soup.find('div', class_=re.compile(r'content|main|body|document')) or
            soup.find('body')
        )
        
        if not main_content:
            return None
        
        # Extract title
        title = (
            soup.find('h1').get_text(strip=True) if soup.find('h1') 
            else soup.title.string if soup.title else 'Untitled'
        )
        
        # Extract code blocks
        code_blocks = []
        for code in main_content.find_all(['code', 'pre']):
            code_text = code.get_text(strip=True)
            if len(code_text) > 10:  # Only meaningful code blocks
                code_blocks.append(code_text)
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up text
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return {
            'title': title,
            'html': str(main_content),
            'text': text,
            'code_blocks': code_blocks
        }
    
    def _process_pages(self, pages: List[Dict], library: str) -> List[DocumentChunk]:
        """Process pages into chunks suitable for RAG"""
        
        chunks = []
        
        for page in pages:
            # Determine document type
            doc_type = self._classify_doc_type(page['title'], page['text'])
            
            # Split into chunks
            page_chunks = self._smart_chunk_text(
                page['text'], 
                page['code_blocks'],
                max_chunk_size=800,
                overlap=100
            )
            
            for i, chunk_text in enumerate(page_chunks):
                # Extract code blocks in this chunk
                chunk_code = [
                    code for code in page['code_blocks']
                    if code in chunk_text
                ]
                
                chunk_id = hashlib.md5(
                    f"{library}:{page['url']}:{i}".encode()
                ).hexdigest()
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    library=library,
                    version='latest',  # Could extract from URL
                    doc_type=doc_type,
                    url=page['url'],
                    title=page['title'],
                    code_blocks=chunk_code,
                    metadata={
                        'chunk_index': i,
                        'total_chunks': len(page_chunks),
                        'has_code': len(chunk_code) > 0
                    },
                    chunk_id=chunk_id
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _classify_doc_type(self, title: str, text: str) -> str:
        """Classify the type of documentation"""
        
        title_lower = title.lower()
        text_lower = text[:500].lower()
        
        if any(word in title_lower for word in ['tutorial', 'guide', 'getting started', 'introduction']):
            return 'tutorial'
        elif any(word in title_lower for word in ['api', 'reference', 'class', 'function', 'method']):
            return 'api'
        elif any(word in title_lower for word in ['example', 'sample', 'demo']):
            return 'example'
        elif 'import' in text_lower or 'def ' in text_lower or 'class ' in text_lower:
            return 'example'
        else:
            return 'guide'
    
    def _smart_chunk_text(self, text: str, code_blocks: List[str], 
                         max_chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Intelligently chunk text, preserving code blocks and context"""
        
        chunks = []
        current_chunk = ""
        
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = ' '.join(current_chunk.split()[-overlap:])
                    current_chunk = overlap_text + '\n\n' + paragraph
                else:
                    # Paragraph itself is too long, split it
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += ('. ' if current_chunk else '') + sentence
            else:
                current_chunk += ('\n\n' if current_chunk else '') + paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


# Example usage
if __name__ == "__main__":
    collector = PythonDocsCollector(output_dir="data/python_docs")
    
    # Start with a smaller set for testing
    test_libraries = ['requests', 'fastapi', 'pandas']
    
    print("Starting Python Documentation Collection")
    print("=" * 60)
    
    results = collector.collect_all(libraries= PythonDocsCollector.CORE_LIBRARIES)
    
    print("\n" + "=" * 60)
    print("Collection Complete!")
    print(f"Libraries processed: {len(results['libraries_processed'])}")
    print(f"Total pages: {results['total_pages']}")
    print(f"Total chunks: {results['total_chunks']}")
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
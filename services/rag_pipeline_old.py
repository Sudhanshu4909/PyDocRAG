"""
PyDocAI - Advanced RAG Pipeline for Python Documentation

Features:
- Code-aware semantic search
- Hybrid retrieval (vector + keyword + code similarity)
- Query understanding and expansion
- Re-ranking with cross-encoders
- Multi-turn conversation support
"""

import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import ast

# Vector store and embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# LLM integration - Support for multiple providers
import os
from abc import ABC, abstractmethod


@dataclass
class RetrievedChunk:
    """Represents a retrieved documentation chunk with scores"""
    content: str
    library: str
    doc_type: str
    url: str
    title: str
    code_blocks: List[str]
    score: float
    retrieval_method: str
    metadata: Dict


class CodeSimilarityCalculator:
    """Calculates similarity between code snippets"""
    
    def __init__(self):
        # Code-specific keywords and patterns
        self.python_keywords = {
            'def', 'class', 'import', 'from', 'return', 'yield',
            'async', 'await', 'lambda', 'with', 'as', 'try', 'except'
        }
    
    def extract_code_features(self, code: str) -> Dict:
        """Extract features from code snippet"""
        features = {
            'keywords': set(),
            'imports': [],
            'functions': [],
            'classes': [],
            'decorators': []
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        features['imports'].append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    features['imports'].append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    features['functions'].append(node.name)
                    # Extract decorators
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            features['decorators'].append(dec.id)
                elif isinstance(node, ast.ClassDef):
                    features['classes'].append(node.name)
        except:
            # If parsing fails, use regex
            features['imports'] = re.findall(r'import\s+(\w+)', code)
            features['functions'] = re.findall(r'def\s+(\w+)', code)
            features['classes'] = re.findall(r'class\s+(\w+)', code)
        
        # Extract keywords
        tokens = code.lower().split()
        features['keywords'] = set(tokens) & self.python_keywords
        
        return features
    
    def calculate_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code snippets"""
        
        features1 = self.extract_code_features(code1)
        features2 = self.extract_code_features(code2)
        
        # Calculate Jaccard similarity for different feature types
        scores = []
        
        # Import similarity
        if features1['imports'] or features2['imports']:
            import_sim = self._jaccard_similarity(
                set(features1['imports']), 
                set(features2['imports'])
            )
            scores.append(import_sim * 2)  # Imports are important
        
        # Function name similarity
        if features1['functions'] or features2['functions']:
            func_sim = self._jaccard_similarity(
                set(features1['functions']), 
                set(features2['functions'])
            )
            scores.append(func_sim * 1.5)
        
        # Keyword similarity
        keyword_sim = self._jaccard_similarity(
            features1['keywords'], 
            features2['keywords']
        )
        scores.append(keyword_sim)
        
        return np.mean(scores) if scores else 0.0
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        """Generate response with conversation history"""
        pass


class OllamaLLM(BaseLLM):
    """Ollama for local open-source models (Llama, Mistral, etc.)"""
    
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
        
        Popular models:
        - llama3.2:3b (lightweight, fast)
        - llama3.1:8b (balanced)
        - mistral:7b (excellent for code)
        - codellama:13b (specialized for code)
        - qwen2.5:7b (great for technical tasks)
        """
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            print(f"✓ Initialized Ollama with model: {model}")
        except ImportError:
            raise ImportError("Please install ollama: pip install ollama")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            )
            return response['message']['content']
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            response = self.client.chat(
                model=self.model,
                messages=full_messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"


class HuggingFaceLLM(BaseLLM):
    """HuggingFace Transformers for local models"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: str = "auto"):
        """
        Initialize HuggingFace model
        
        Popular models:
        - meta-llama/Llama-3.2-3B-Instruct (needs authentication)
        - mistralai/Mistral-7B-Instruct-v0.2
        - microsoft/phi-2 (lightweight, 2.7B)
        - HuggingFaceH4/zephyr-7b-beta (excellent instruction following)
        - codellama/CodeLlama-7b-Instruct-hf
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            print(f"Loading model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device
            )
            
            print(f"✓ Loaded {model_name}")
            
        except ImportError:
            raise ImportError("Please install: pip install transformers torch accelerate")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            # Format with system prompt
            if system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
            else:
                full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                return_full_text=False
            )
            
            return outputs[0]['generated_text']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        # Convert messages to prompt format
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"<|system|>\n{system_prompt}")
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            prompt_parts.append(f"<|{role}|>\n{content}")
        
        prompt_parts.append("<|assistant|>\n")
        full_prompt = "\n".join(prompt_parts)
        
        return self.generate(full_prompt, "", max_tokens)


class VLLMLLMServer(BaseLLM):
    """vLLM for high-performance inference server"""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize vLLM client (requires vLLM server running)
        
        Start vLLM server:
        python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="dummy"  # vLLM doesn't require real API key
            )
            self.model = model
            print(f"✓ Connected to vLLM server at {base_url}")
        except ImportError:
            raise ImportError("Please install: pip install openai")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class LlamaCppLLM(BaseLLM):
    """llama.cpp for efficient CPU inference"""
    
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4):
        """
        Initialize llama.cpp
        
        Download GGUF models from HuggingFace:
        - TheBloke/Mistral-7B-Instruct-v0.2-GGUF
        - TheBloke/CodeLlama-7B-Instruct-GGUF
        - TheBloke/Llama-2-7B-Chat-GGUF
        """
        try:
            from llama_cpp import Llama
            
            print(f"Loading model from: {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=0,  # Set to -1 for GPU
                verbose=False
            )
            print(f"✓ Loaded model with {n_ctx} context window")
        except ImportError:
            raise ImportError("Please install: pip install llama-cpp-python")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\n\nAssistant:"
            
            output = self.llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=["User:", "\n\n\n"]
            )
            
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        # Convert to prompt format
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}")
        
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)
        
        return self.generate(full_prompt, "", max_tokens)


class TogetherAILLM(BaseLLM):
    """Together AI for hosted open-source models"""
    
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2", api_key: str = None):
        """
        Initialize Together AI client
        
        Available models:
        - mistralai/Mistral-7B-Instruct-v0.2
        - meta-llama/Llama-3-8b-chat-hf
        - codellama/CodeLlama-34b-Instruct-hf
        - Qwen/Qwen2.5-7B-Instruct
        """
        try:
            import together
            
            self.client = together.Together(api_key=api_key or os.environ.get("TOGETHER_API_KEY"))
            self.model = model
            print(f"✓ Initialized Together AI with model: {model}")
        except ImportError:
            raise ImportError("Please install: pip install together")
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_with_messages(self, messages: List[Dict], system_prompt: str = "", max_tokens: int = 2000) -> str:
        try:
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"


class LLMFactory:
    """Factory to create LLM instances"""
    
    @staticmethod
    def create_llm(provider: str = "ollama", **kwargs) -> BaseLLM:
        """
        Create LLM instance based on provider
        
        Args:
            provider: One of 'ollama', 'huggingface', 'vllm', 'llamacpp', 'together'
            **kwargs: Provider-specific arguments
        
        Examples:
            # Ollama (easiest to setup)
            llm = LLMFactory.create_llm('ollama', model='llama3.2:3b')
            
            # HuggingFace (for local inference)
            llm = LLMFactory.create_llm('huggingface', model_name='microsoft/phi-2')
            
            # vLLM (for production)
            llm = LLMFactory.create_llm('vllm', base_url='http://localhost:8000')
            
            # llama.cpp (for CPU inference)
            llm = LLMFactory.create_llm('llamacpp', model_path='./models/mistral-7b-instruct.gguf')
            
            # Together AI (hosted)
            llm = LLMFactory.create_llm('together', model='mistralai/Mistral-7B-Instruct-v0.2')
        """
        
        providers = {
            'ollama': OllamaLLM,
            'huggingface': HuggingFaceLLM,
            'vllm': VLLMLLMServer,
            'llamacpp': LlamaCppLLM,
            'together': TogetherAILLM,
        }
        
        if provider not in providers:
            raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
        
        try:
            return providers[provider](**kwargs)
        except Exception as e:
            print(f"Error initializing {provider}: {e}")
            raise


class QueryUnderstanding:
    """Understands and enhances user queries"""
    
    def __init__(self, llm_client: BaseLLM = None):
        self.llm = llm_client
        self.intent_patterns = {
            'how_to': r'how (do|can|to|should)',
            'what_is': r'what (is|are|does)',
            'error': r'(error|exception|bug|issue|problem|fail)',
            'example': r'(example|sample|demo|show me)',
            'best_practice': r'(best|better|should|recommend)',
            'comparison': r'(vs|versus|difference|compare)',
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine intent and extract metadata"""
        
        query_lower = query.lower()
        
        # Detect intent
        intent = 'general'
        for intent_type, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                intent = intent_type
                break
        
        # Extract mentioned libraries
        libraries = self._extract_libraries(query)
        
        # Detect if query contains code
        has_code = bool(re.search(r'`[^`]+`|```', query))
        
        # Extract code snippets from query
        code_snippets = re.findall(r'```python\n(.*?)```', query, re.DOTALL)
        code_snippets.extend(re.findall(r'`([^`]+)`', query))
        
        return {
            'intent': intent,
            'libraries': libraries,
            'has_code': has_code,
            'code_snippets': code_snippets,
            'is_comparative': 'comparison' in intent,
            'needs_example': intent in ['how_to', 'example']
        }
    
    def _extract_libraries(self, query: str) -> List[str]:
        """Extract Python library names from query"""
        
        common_libraries = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
            'tensorflow', 'pytorch', 'keras', 'flask', 'django',
            'fastapi', 'requests', 'beautifulsoup', 'selenium',
            'pytest', 'unittest', 'asyncio', 'multiprocessing',
            'sqlalchemy', 'pydantic', 'click', 'argparse'
        }
        
        query_lower = query.lower()
        found_libraries = [lib for lib in common_libraries if lib in query_lower]
        
        return found_libraries
    
    def expand_query(self, query: str, context: Dict) -> List[str]:
        """Generate query variations for better retrieval"""
        
        variations = [query]
        
        # Add library-specific variation
        if context['libraries']:
            for lib in context['libraries']:
                variations.append(f"{lib} {query}")
        
        # Add intent-specific variations
        if context['intent'] == 'how_to':
            variations.append(f"tutorial {query}")
            variations.append(f"example {query}")
        elif context['intent'] == 'what_is':
            variations.append(f"documentation {query}")
        elif context['intent'] == 'error':
            variations.append(f"troubleshooting {query}")
            variations.append(f"fix {query}")
        
        return variations[:3]  # Limit to 3 variations


class PyDocRAGPipeline:
    """Main RAG pipeline for Python documentation"""
    
    def __init__(self, 
                 docs_dir: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 llm_provider: str = "ollama",
                 llm_config: Optional[Dict] = None):
        """
        Initialize RAG Pipeline
        
        Args:
            docs_dir: Directory containing documentation chunks
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross-encoder model for re-ranking
            llm_provider: LLM provider ('ollama', 'huggingface', 'vllm', 'llamacpp', 'together')
            llm_config: Configuration dict for LLM provider
        
        Example:
            # Using Ollama (default, easiest)
            pipeline = PyDocRAGPipeline(
                docs_dir="data/python_docs",
                llm_provider="ollama",
                llm_config={"model": "llama3.2:3b"}
            )
            
            # Using local HuggingFace model
            pipeline = PyDocRAGPipeline(
                docs_dir="data/python_docs",
                llm_provider="huggingface",
                llm_config={"model_name": "microsoft/phi-2"}
            )
            
            # Using vLLM server
            pipeline = PyDocRAGPipeline(
                docs_dir="data/python_docs",
                llm_provider="vllm",
                llm_config={"base_url": "http://localhost:8000", "model": "mistralai/Mistral-7B-Instruct-v0.2"}
            )
        """
        
        self.docs_dir = Path(docs_dir)
        
        # Initialize models
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        
        print("Loading re-ranker model...")
        self.reranker = CrossEncoder(reranker_model)
        
        self.code_similarity = CodeSimilarityCalculator()
        
        # Initialize LLM
        llm_config = llm_config or {}
        print(f"\nInitializing LLM provider: {llm_provider}")
        self.llm = LLMFactory.create_llm(llm_provider, **llm_config)
        
        self.query_understanding = QueryUnderstanding(self.llm)
        
        # Storage
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None
        
        print("RAG Pipeline initialized")
    
    def index_documents(self):
        """Load and index all documentation chunks"""
        
        print("\nIndexing documents...")
        
        # Load all chunks from all libraries
        for lib_dir in self.docs_dir.iterdir():
            if not lib_dir.is_dir():
                continue
            
            chunks_file = lib_dir / 'chunks.jsonl'
            if not chunks_file.exists():
                continue
            
            print(f"Loading {lib_dir.name}...")
            
            with open(chunks_file, 'r') as f:
                for line in f:
                    chunk_data = json.loads(line)
                    self.chunks.append(chunk_data)
        
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        texts = [chunk['content'] for chunk in self.chunks]
        self.embeddings = self.embedder.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        print("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.faiss_index.add(self.embeddings)
        
        # Create BM25 index
        print("Building BM25 index...")
        tokenized_docs = [chunk['content'].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print("Indexing complete!")
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 10,
                 rerank: bool = True,
                 filters: Optional[Dict] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant documentation chunks
        
        Args:
            query: User query
            top_k: Number of results to return
            rerank: Whether to use re-ranking
            filters: Optional filters (e.g., {'library': 'pandas', 'doc_type': 'tutorial'})
        """
        
        # Understand query
        query_context = self.query_understanding.analyze_query(query)
        print(f"\nQuery Intent: {query_context['intent']}")
        print(f"Detected Libraries: {query_context['libraries']}")
        
        # Generate query variations
        query_variations = self.query_understanding.expand_query(query, query_context)
        
        # Retrieve using multiple methods
        all_results = []
        
        # 1. Vector search
        vector_results = self._vector_search(query_variations, top_k * 2)
        all_results.extend(vector_results)
        
        # 2. Keyword search
        bm25_results = self._bm25_search(query, top_k * 2)
        all_results.extend(bm25_results)
        
        # 3. Code search if query contains code
        if query_context['code_snippets']:
            code_results = self._code_search(
                query_context['code_snippets'][0], 
                top_k
            )
            all_results.extend(code_results)
        
        # Deduplicate and apply filters
        unique_results = self._deduplicate_results(all_results)
        
        if filters:
            unique_results = [
                r for r in unique_results 
                if all(r.metadata.get(k) == v for k, v in filters.items())
            ]
        
        # Re-rank if requested
        if rerank and len(unique_results) > top_k:
            unique_results = self._rerank_results(query, unique_results, top_k * 2)
        
        # Boost results based on query context
        unique_results = self._boost_results(unique_results, query_context)
        
        # Sort by score and return top_k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:top_k]
    
    def _vector_search(self, queries: List[str], top_k: int) -> List[RetrievedChunk]:
        """Semantic search using embeddings"""
        
        results = []
        
        for query in queries:
            # Embed query
            query_embedding = self.embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            for score, idx in zip(scores[0], indices[0]):
                chunk = self.chunks[idx]
                results.append(RetrievedChunk(
                    content=chunk['content'],
                    library=chunk['library'],
                    doc_type=chunk['doc_type'],
                    url=chunk['url'],
                    title=chunk['title'],
                    code_blocks=chunk['code_blocks'],
                    score=float(score),
                    retrieval_method='vector',
                    metadata=chunk['metadata']
                ))
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Keyword-based search using BM25"""
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(RetrievedChunk(
                content=chunk['content'],
                library=chunk['library'],
                doc_type=chunk['doc_type'],
                url=chunk['url'],
                title=chunk['title'],
                code_blocks=chunk['code_blocks'],
                score=float(scores[idx]),
                retrieval_method='bm25',
                metadata=chunk['metadata']
            ))
        
        return results
    
    def _code_search(self, code_snippet: str, top_k: int) -> List[RetrievedChunk]:
        """Search based on code similarity"""
        
        results = []
        
        for i, chunk in enumerate(self.chunks):
            if not chunk['code_blocks']:
                continue
            
            # Calculate max similarity with any code block in chunk
            max_similarity = 0
            for code_block in chunk['code_blocks']:
                similarity = self.code_similarity.calculate_similarity(
                    code_snippet, 
                    code_block
                )
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0.1:  # Threshold
                results.append(RetrievedChunk(
                    content=chunk['content'],
                    library=chunk['library'],
                    doc_type=chunk['doc_type'],
                    url=chunk['url'],
                    title=chunk['title'],
                    code_blocks=chunk['code_blocks'],
                    score=max_similarity,
                    retrieval_method='code',
                    metadata=chunk['metadata']
                ))
        
        # Sort by similarity
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _deduplicate_results(self, results: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Remove duplicate chunks, keeping highest score"""
        
        seen = {}
        for result in results:
            key = (result.library, result.url, result.title)
            if key not in seen or result.score > seen[key].score:
                seen[key] = result
        
        return list(seen.values())
    
    def _rerank_results(self, query: str, results: List[RetrievedChunk], 
                       top_k: int) -> List[RetrievedChunk]:
        """Re-rank results using cross-encoder"""
        
        # Prepare pairs for re-ranking
        pairs = [[query, result.content] for result in results]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update scores
        for result, score in zip(results, rerank_scores):
            # Combine original and rerank score
            result.score = 0.5 * result.score + 0.5 * float(score)
        
        return results
    
    def _boost_results(self, results: List[RetrievedChunk], 
                      query_context: Dict) -> List[RetrievedChunk]:
        """Boost scores based on query context"""
        
        for result in results:
            # Boost if library matches
            if result.library in query_context['libraries']:
                result.score *= 1.3
            
            # Boost examples if needed
            if query_context['needs_example'] and result.doc_type == 'example':
                result.score *= 1.2
            
            # Boost if has code when code is in query
            if query_context['has_code'] and result.code_blocks:
                result.score *= 1.15
        
        return results
    
    def generate_answer(self, 
                       query: str, 
                       retrieved_chunks: List[RetrievedChunk],
                       conversation_history: Optional[List[Dict]] = None) -> Dict:
        """Generate answer using LLM with retrieved context"""
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"""
### Source {i}: {chunk.library} - {chunk.title}
URL: {chunk.url}
Type: {chunk.doc_type}

{chunk.content}

{'Code Examples:' if chunk.code_blocks else ''}
{chr(10).join(f'```python\n{code}\n```' for code in chunk.code_blocks[:2])}
""")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # System prompt
        system_prompt = """You are PyDocAI, an expert Python programming assistant with access to comprehensive Python and library documentation.

Your role:
1. Provide accurate, helpful answers based on the documentation context
2. Include working code examples when appropriate
3. Cite sources using the provided URLs
4. Explain concepts clearly for different skill levels
5. Suggest best practices and warn about common pitfalls

Guidelines:
- Always reference the documentation sources
- Prefer official documentation over third-party sources
- Include code examples that are complete and runnable
- Explain WHY something works, not just HOW
- If information is not in the context, say so clearly"""
        
        # User message with context
        user_message = f"""Documentation Context:
{context}

---

User Question: {query}

Please provide a comprehensive answer based on the documentation above. Include code examples where helpful and cite the sources."""
        
        # Generate response using the configured LLM
        if conversation_history:
            messages = conversation_history + [{"role": "user", "content": user_message}]
            answer = self.llm.generate_with_messages(messages, system_prompt, max_tokens=2000)
        else:
            answer = self.llm.generate(user_message, system_prompt, max_tokens=2000)
        
        return {
            'answer': answer,
            'sources': [{
                'library': chunk.library,
                'title': chunk.title,
                'url': chunk.url,
                'doc_type': chunk.doc_type,
                'score': chunk.score,
                'retrieval_method': chunk.retrieval_method
            } for chunk in retrieved_chunks],
            'query_context': self.query_understanding.analyze_query(query)
        }
    
    def query(self, 
             query: str, 
             top_k: int = 5,
             conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        End-to-end query processing
        
        Args:
            query: User question
            top_k: Number of sources to use
            conversation_history: Previous conversation for context
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        print(f"\nProcessing query: {query}")
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k=top_k, rerank=True)
        
        print(f"Retrieved {len(retrieved)} chunks")
        for i, chunk in enumerate(retrieved[:3], 1):
            print(f"  {i}. [{chunk.library}] {chunk.title[:50]}... (score: {chunk.score:.3f})")
        
        # Generate answer
        result = self.generate_answer(query, retrieved, conversation_history)
        
        return result


# Example usage
if __name__ == "__main__":
    # Example 1: Using Ollama (easiest setup - just install Ollama and pull a model)
    print("Example 1: Using Ollama")
    print("Setup: Install Ollama from https://ollama.ai")
    print("       Run: ollama pull llama3.2:3b")
    print("-" * 80)
    
    pipeline_ollama = PyDocRAGPipeline(
        docs_dir="data/python_docs",
        llm_provider="ollama",
        llm_config={"model": "llama3.2:3b"}
    )
    
    # Example 2: Using HuggingFace local model
    # print("\nExample 2: Using HuggingFace")
    # pipeline_hf = PyDocRAGPipeline(
    #     docs_dir="data/python_docs",
    #     llm_provider="huggingface",
    #     llm_config={"model_name": "microsoft/phi-2"}
    # )
    
    # Example 3: Using llama.cpp for CPU inference
    # print("\nExample 3: Using llama.cpp")
    # pipeline_cpp = PyDocRAGPipeline(
    #     docs_dir="data/python_docs",
    #     llm_provider="llamacpp",
    #     llm_config={"model_path": "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"}
    # )
    
    # Example 4: Using vLLM server (for production)
    # print("\nExample 4: Using vLLM")
    # print("Setup: python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2")
    # pipeline_vllm = PyDocRAGPipeline(
    #     docs_dir="data/python_docs",
    #     llm_provider="vllm",
    #     llm_config={
    #         "base_url": "http://localhost:8000",
    #         "model": "mistralai/Mistral-7B-Instruct-v0.2"
    #     }
    # )
    
    # Index documents
    pipeline_ollama.index_documents()
    
    # Query examples
    queries = [
        "How do I make async HTTP requests with aiohttp?",
        "What's the difference between DataFrame.apply() and DataFrame.map() in pandas?",
        "Show me an example of using FastAPI with Pydantic models",
        "How to handle exceptions in asyncio tasks?"
    ]
    
    for query in queries:
        print("\n" + "="*80)
        result = pipeline_ollama.query(query, top_k=5)
        print(f"\nQuery: {query}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used: {len(result['sources'])}")
        for i, src in enumerate(result['sources'][:3], 1):
            print(f"  {i}. [{src['library']}] {src['title'][:60]}...")
"""LLM provider abstractions"""
# Copy the BaseLLM, OllamaLLM, HuggingFaceLLM, etc. from your Rag_pipeline.py
# Keep them as-is, they're already good!

from abc import ABC, abstractmethod
from typing import List, Dict

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

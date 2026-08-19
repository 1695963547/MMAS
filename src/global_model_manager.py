#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global model manager - unified management of all model instances to avoid duplicate loading
"""

import threading
import time
import gc
import torch
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# Import quantization config
from quantization_config import QuantizationConfigManager, log_memory_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelCacheEntry:
    """Model cache entry"""
    model: Any
    tokenizer: Any
    model_path: str
    load_time: float
    last_access_time: float
    memory_usage: int  # bytes
    device: str
    reference_count: int = 0  # reference count
    is_persistent: bool = False  # whether persistent model; persistent models are not auto-cleaned

class GlobalModelManager:
    """Global unified model manager - singleton pattern
    
    functionality:
    1. unified management of all model instances to avoid duplicate loading
    2. Model preloading and cache management
    3. Memory optimization and auto cleanup
    4. Thread-safe model sharing
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    # Path normalization utility function
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize the given path to an absolute path with forward slashes, 
        and map common HuggingFace aliases to local paths to avoid duplicate loading"""
        import os
        from pathlib import Path

        # 1) 去除首尾空格
        raw = str(path).strip()
        
        # 2) Alias mapping
        lower_key = raw.lower().replace('\\', '/').rstrip('/')
        project_root = Path(__file__).resolve().parent
        alias_map = {
            # 70B GPTQ模型映射
            'meta-llama-3-70b-instruct': str(project_root / 'llama' / 'meta-llama-3-70b-instruct-gptq-int4'),
            'meta-llama/meta-llama-3-70b-instruct': str(project_root / 'llama' / 'meta-llama-3-70b-instruct-gptq-int4'),
            'meta-llama/Meta-Llama-3-70B-Instruct': str(project_root / 'llama' / 'meta-llama-3-70b-instruct-gptq-int4'),
            'llama3-70b': str(project_root / 'llama' / 'meta-llama-3-70b-instruct-gptq-int4'),
            'llama3-70b-instruct': str(project_root / 'llama' / 'meta-llama-3-70b-instruct-gptq-int4'),
            # 8B模型映射（保留作as备选）
            'meta-llama-3-8b-instruct': str(project_root / 'llama' / 'Meta-Llama-3-8B-Instruct'),
            'meta-llama/meta-llama-3-8b-instruct': str(project_root / 'llama' / 'Meta-Llama-3-8B-Instruct'),
            'meta-llama/Meta-Llama-3-8B-Instruct': str(project_root / 'llama' / 'Meta-Llama-3-8B-Instruct'),
            'llama3-8b': str(project_root / 'llama' / 'Meta-Llama-3-8B-Instruct'),
            'llama3-8b-instruct': str(project_root / 'llama' / 'Meta-Llama-3-8B-Instruct'),
            # 语义模型映射（使用 HuggingFace name，this地cacheor自动下载）
            'paraphrase-multilingual-minilm-l12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/paraphrase-multilingual-minilm-l12-v2': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        }
        
        path_to_check = alias_map.get(lower_key, raw)

        # 3) Checking pathis否asvaliddirectory
        if os.path.isdir(path_to_check):
            # If valid directory, force return absolute path with forward slashes
            abs_path = Path(path_to_check).resolve().as_posix()
            # Ensure path uses forward slashes to prevent transformers library from misidentifying as repo ID
            return abs_path.replace('\\', '/')
        
        # 4) If not an existing directory, assume it is a HuggingFace Hub ID
        # In this case, return the original unmodified identifier
        return raw

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._model_cache: Dict[str, ModelCacheEntry] = {}
            self._cache_lock = threading.RLock()
            
            # quantization config
            self._quantization_config = None
            self._quantization_type = "none"
            
            # Configuration parameters
            self._max_cache_size = 10  # max cached models
            self._memory_threshold = 0.60  # memory usage threshold
            self._cleanup_interval = 60  # cleanup interval (seconds)
            self._enable_persistent_cache = True  # enable persistent cache
            self._persistent_models = set()  # persistent model set
            
            # Preloaded model config - initially empty, must be set via configure_preload_models
            self._preload_configs = {
                'main_llm': {
                    'model_name': None,  # No default value, must be configured externally
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'priority': 1
                },
                'semantic_model': {
                    'model_name': 'paraphrase-multilingual-MiniLM-L12-v2',  # Semantic model keeps default value
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'priority': 2
                }
            }
            
            self.initialized = True
            print("[GMM] Global model manager initialized", flush=True)
            
            # Starting 后台清理线程
            self._start_cleanup_thread()
    
    def configure_quantization(self, quantization_type: str = "none"):
        """Configure quantization settings
        
        Args:
            quantization_type (str): quantization type ("none", "8bit", "4bit")
        """
        self._quantization_type = quantization_type
        self._quantization_config = QuantizationConfigManager.get_quantization_config(quantization_type)
        
        if quantization_type != "none":
            print(f"[GMM] configured{quantization_type}量化", flush=True)
            log_memory_usage("Before quantization config")
        else:
            print("[GMM] Quantization not enabled", flush=True)

    def configure_preload_models(self, main_llm_path: Optional[str] = None, 
                            semantic_model_path: Optional[str] = None, 
                            disable_main_llm: bool = False) -> None:
        """Configure preload model paths
        
        Args:
            main_llm_path: Main LLM path, uses default if None
            semantic_model_path: Semantic model path, uses default if None
            disable_main_llm: Whether to disable main LLM preload (for API mode)
        """
        # If main LLM disabled, remove from config
        if disable_main_llm:
            if 'main_llm' in self._preload_configs:
                del self._preload_configs['main_llm']
                print("[GMM] Main LLM preload disabled (API mode)", flush=True)
        elif main_llm_path is not None:
            self._preload_configs['main_llm']['model_name'] = main_llm_path
            print(f"[GMM] Configured main LLM path: {main_llm_path}", flush=True)
        else:
            # If main_llm_path not provided and not disabled, check existing config
            if self._preload_configs['main_llm']['model_name'] is None:
                print("[GMM] WARNING: Main LLM path not configured", flush=True)
        
        if semantic_model_path is not None:
            self._preload_configs['semantic_model']['model_name'] = semantic_model_path
            print(f"[GMM] Configured semantic model path: {semantic_model_path}", flush=True)
    
    def preload_models(self) -> None:
        """Preload core models (auto-set as persistent)"""
        print("[GMM] Starting core model preloading...", flush=True)
        
        for model_key, config in self._preload_configs.items():
            try:
                # Check if model_name is configured
                if config['model_name'] is None:
                    print(f"[GMM] Skipped preloading {model_key}：model path not configured", flush=True)
                    continue
                    
                # Normalize paths to avoid duplicate caching due to slash direction differences
                model_name = self._normalize_path(config['model_name'])
                device = config['device']
                
                print(f"[GMM] Preloading model: {model_name}", flush=True)
                model, tokenizer = self._load_model_internal(model_name, device)
                
                # Cache model (model_name already normalized), preloaded models auto-set as persistent
                cache_key = f"{model_name}_{device}"
                self._cache_model(cache_key, model, tokenizer, model_name, device, is_persistent=True)
                
                print(f"[GMM] 模型 {model_name} preloading complete", flush=True)
                
            except Exception as e:
                print(f"[GMM] Preloading model {model_key} Failed : {e}", flush=True)
        
        print("[GMM] Core model preloading complete", flush=True)
    
    def get_model(self, model_name: str, device: str = 'cuda', is_persistent: bool = False) -> Tuple[Any, Any]:
        """Get model and tokenizer - smart cache matching
        
        Args:
            model_name: Model name or preload config key
            device: Device type
            is_persistent: Whether to set as persistent model
            
        Returns:
            Tuple[model, tokenizer]
        """
        # First check if it is a preload config key
        actual_model_name = model_name
        if model_name in self._preload_configs:
            actual_model_name = self._preload_configs[model_name]['model_name']
            print(f"[GMM] Identified preload config key '{model_name}' -> '{actual_model_name}'", flush=True)
        
        normalized_model_name = self._normalize_path(actual_model_name)
        
        # For single-GPU environment, unify cache key generation strategy
        if device == 'cuda' and torch.cuda.device_count() == 1:
            # In single-GPU environment, use cuda as device identifier
            cache_key = f"{normalized_model_name}_cuda"
            pass  # debug: 单GPU统一cache键
        else:
            cache_key = f"{normalized_model_name}_{device}"
        
        with self._cache_lock:
            # Check cache
            if cache_key in self._model_cache:
                entry = self._model_cache[cache_key]
                entry.last_access_time = time.time()
                entry.reference_count += 1
                logger.info(f"Got model from cache: {model_name} -> {actual_model_name} (reference count: {entry.reference_count})")
                return entry.model, entry.tokenizer
            
            # If not found in cache, check if other models need cleanup (non-persistent mode only)
            if len(self._model_cache) > 0 and not self._enable_persistent_cache:
                logger.warning(f"Multiple model loading detected, force cleanup to ensure singleton pattern")
                self._force_cleanup_all_models()
                # Force garbage collection and VRAM cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Check VRAM before loading new model
            if torch.cuda.is_available():
                self._check_and_cleanup_memory()
            
            # Loading new model
            logger.info(f"Loading new model: {model_name} -> {actual_model_name}")
            model, tokenizer = self._load_model_internal(actual_model_name, device)
            
            # Cache model - using unified device identifier
            final_device = 'cuda' if device == 'cuda' and torch.cuda.device_count() == 1 else device
            self._cache_model(cache_key, model, tokenizer, normalized_model_name, final_device, is_persistent)
            
            return model, tokenizer
    
    def release_model(self, model_name: str, device: str = 'cuda') -> None:
        """Release model reference
        
        Args:
            model_name: Model name
            device: Device type
        """
        normalized_model_name = self._normalize_path(model_name)
        cache_key = f"{normalized_model_name}_{device}"
        
        with self._cache_lock:
            if cache_key in self._model_cache:
                entry = self._model_cache[cache_key]
                entry.reference_count = max(0, entry.reference_count - 1)
                logger.info(f"Release model reference: {model_name} (reference count: {entry.reference_count})")
    
    def _load_model_internal(self, model_name: str, device: str) -> Tuple[Any, Any]:
        """Internal model loading method"""
        try:
            # Check if it is a SentenceTransformer model
            if self._is_sentence_transformer_model(model_name):
                return self._load_sentence_transformer(model_name, device)
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import os
            import json
            
            # Force use of local model path
            local_model_path_str = self._normalize_path(model_name)
            logger.info(f"Using normalized model path: {local_model_path_str}")
            
            # Verify local path exists
            if not os.path.exists(local_model_path_str):
                raise FileNotFoundError(f"Local model path does not exist: {local_model_path_str}")
            
            # Check if it is a GPTQ model
            is_gptq_model = self._is_gptq_model(local_model_path_str)
            
            if is_gptq_model:
                logger.info(f"GPTQ model detected, loading with AutoGPTQForCausalLM: {local_model_path_str}")
                return self._load_gptq_model(local_model_path_str, device)
            
            # Disable torch compilation to avoid triton dependency
            import os
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            
            # Load tokenizer - force local mode
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path_str,
                trust_remote_code=True,
                padding_side='left',
                local_files_only=True,  # force local files only
                use_fast=True  # Use fast tokenizer to avoid slow tokenizer returning bool issue
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model - force local mode, optimize memory usage
            model_kwargs = {
                'trust_remote_code': True,
                'local_files_only': True,  # force local files only
                'ignore_mismatched_sizes': True,  # ignore mismatched sizes
            }
            
            # Set parameters based on quantization config
            if self._quantization_config is not None:
                model_kwargs['quantization_config'] = self._quantization_config
                model_kwargs['device_map'] = "auto"  # auto device mapping needed for quantization
                logger.info(f"使用{self._quantization_type}quantization configLoading 模型")
            else:
                model_kwargs['torch_dtype'] = torch.bfloat16  # use bfloat16 when not quantized
                model_kwargs['device_map'] = None  # disable auto device mapping, manual control
            
            model = AutoModelForCausalLM.from_pretrained(local_model_path_str, **model_kwargs)

            # Decide whether to tie weights based on model config
            if model.config.tie_word_embeddings:
                model.tie_weights()
            else:
                logger.info("Model config `tie_word_embeddings` is False. Skipping `tie_weights()`.")

            # For quantized models, no need to manually move to CUDA device
            if device == 'cuda' and self._quantization_config is None:
                model.to('cuda')

            model.eval()
            
            # Add model loading verification logs
            if hasattr(model, 'dtype'):
                logger.info(f"Model dtype: {model.dtype}")
            if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
                logger.info(f"Attention implementation: {model.config._attn_implementation}")
            
            # VRAM usage logging
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"VRAM usage after model loading: {allocated:.2f}GB / {reserved:.2f}GB")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _is_gptq_model(self, model_path: str) -> bool:
        """Check if it is a GPTQ model"""
        import os
        import json
        
        # Check if quantize_config.json exists
        quantize_config_path = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(quantize_config_path):
            try:
                with open(quantize_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("quant_method") == "gptq"
            except Exception:
                pass
        
        # Checking config.jsonquantization_config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    quantization_config = config.get("quantization_config", {})
                    return quantization_config.get("quant_method") == "gptq"
            except Exception:
                pass
        
        return False

    def _load_gptq_model(self, model_path: str, device: str) -> Tuple[Any, Any]:
        """Loading GPTQ模型 - Prefer transformers library, GPTQModel as fallback"""
        # First try loading GPTQ model with transformers library
        try:
            logger.info(f"尝试使用transformers库Loading GPTQ模型: {model_path}")
            return self._load_gptq_with_transformers(model_path, device)
        except Exception as transformers_error:
            logger.warning(f"transformers库Loading GPTQ模型Failed : {transformers_error}")
            logger.info("Fall back to GPTQModel库...")
            
            # Fall back to GPTQModel
            try:
                return self._load_gptq_with_gptqmodel(model_path, device)
            except Exception as gptq_error:
                logger.error(f"GPTQModel库也Loading Failed : {gptq_error}")
                raise Exception(f"All GPTQ loading methods failed。transformersincorrect: {transformers_error}; GPTQModelincorrect: {gptq_error}")

    def _load_gptq_with_transformers(self, model_path: str, device: str) -> Tuple[Any, Any]:
        """使用transformers库Loading GPTQ模型"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os
        
        logger.info(f"使用transformers库Loading GPTQ模型: {model_path}")
        
        # Disable 可能导致questionOptimization
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        
        # RecordLoading Starting Timeand显存status
        load_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"模型Loading 前显存使用: {allocated_before:.2f}GB / {reserved_before:.2f}GB")
        
        # Loading tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Setting pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("setpad_tokenaseos_token")
        
        # Setting chat_template
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            logger.info("setchat_template")
        
        # Loading 模型 - 使用保守Setting 
        logger.info("Loading GPTQ模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        # RecordLoading Timeand显存使用
        load_time = time.time() - load_start_time
        logger.info(f"transformers库GPTQ模型Loading 耗时: {load_time:.2f}seconds")
        
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM usage after model loading: {allocated_after:.2f}GB / {reserved_after:.2f}GB")
        
        model.eval()
        logger.info(f"transformers库GPTQ模型Loading Successfully : {type(model)}")
        
        return model, tokenizer

    def _load_gptq_with_gptqmodel(self, model_path: str, device: str) -> Tuple[Any, Any]:
        """使用GPTQModel库Loading GPTQ模型（备选plan）"""
        try:
            from gptqmodel import GPTQModel
            from transformers import AutoTokenizer
            import os
            import json
            
            logger.info(f"使用GPTQModelLoading GPTQ模型: {model_path}")
            
            # Loading tokenizer
            logger.info("Starting Loading tokenizer")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    use_fast=True,
                    local_files_only=True,
                    trust_remote_code=True,
                    legacy=False
                )
                logger.info(f"使用fast tokenizerLoading Successfully : {type(tokenizer)}")
            except Exception as e:
                logger.error(f"Fast tokenizerLoading Failed : {e}")
                # 如果fast tokenizerFailed ，尝试不使用local_files_only
                try:
                    logger.warning("尝试不使用local_files_only重新Loading tokenizer")
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path, 
                        use_fast=True,
                        trust_remote_code=True,
                        legacy=False
                    )
                    logger.info("使用线modeLoading tokenizerSuccessfully ")
                except Exception as e2:
                    logger.error(f"alltokenizerLoading Method都Failed : {e2}")
                    raise e2
            
            # validationtokenizerObject
            if not hasattr(tokenizer, 'encode') or not hasattr(tokenizer, 'decode'):
                raise ValueError(f"Loading tokenizerObjectinvalid: {type(tokenizer)}")
            
            # Setting pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("setpad_tokenaseos_token")
            
            # CodeSetting chat_template，避免修改模型Configuring file
            if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
                tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
                logger.info("已CodeSetting chat_template")
            
            # 使用GPTQModelLoading 已量化模型
            logger.info("使用GPTQModelLoading 已量化GPTQ模型")
            # RecordLoading Starting Timeand显存status
            load_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 清理显存cache
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"模型Loading 前显存使用: {allocated_before:.2f}GB / {reserved_before:.2f}GB")
            
            # 针for48GB显存EnvironmentOptimizationGPTQ模型Loading strategy
            try:
                if device == "cuda":
                    logger.info("Detected CUDA设备，使用Optimization大显存Loading strategy")
                    
                    # Set environment variablesOptimization内存使用
                    import os
                    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True"
                    
                    # 使用低内存modeLoading ，避免虚拟内存不足
                    model = GPTQModel.from_quantized(
                        model_path,
                        device="cuda:0",  # 直接指定设备
                        use_safetensors=True,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,  # Enable 低CPU内存使用mode
                        torch_dtype=torch.float16  # 使用float16减少内存占用
                    )
                else:
                    model = GPTQModel.from_quantized(
                        model_path,
                        device="cpu",
                        use_safetensors=True,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    
                logger.info(f"GPTQ模型Loading Successfully ，设备: {device}")
                
            except Exception as model_load_error:
                # 模型Loading Failed 时清理显存andprovide详细incorrectInformation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                error_msg = str(model_load_error)
                if "os error 1455" in error_msg or "页面file太小" in error_msg:
                    logger.error("Detected Windows虚拟内存不足question。suggestion解决plan：")
                    logger.error("1. 增加Windows页面File sizeto至少100GB")
                    logger.error("2. or者systemSetting Enable '自动管理all驱动器分页File size'")
                    logger.error("3. 重启system后retry")
                
                logger.error(f"GPTQModelLoading Failed : {model_load_error}")
                raise
            
            # RecordLoading Timeand显存使用
            load_time = time.time() - load_start_time
            logger.info(f"GPTQ模型Loading 耗时: {load_time:.2f}seconds")
            
            if torch.cuda.is_available():
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"VRAM usage after model loading: {allocated_after:.2f}GB / {reserved_after:.2f}GB")
            
            model.eval()
            
            logger.info(f"GPTQ模型Loading Successfully : {type(model)}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"使用GPTQModelLoading GPTQ模型Failed  {model_path}: {e}")
            import traceback
            logger.error(f"详细incorrectInformation: {traceback.format_exc()}")
            raise
    
    def _is_sentence_transformer_model(self, model_name: str) -> bool:
        """Check if it is a SentenceTransformer model"""
        # Checking 模型pathis否包含sentence-transformer相关关键词
        model_name_lower = model_name.lower()
        return any(keyword in model_name_lower for keyword in [
            'sentence', 'minilm', 'paraphrase', 'multilingual'
        ])
    
    def _load_sentence_transformer(self, model_name: str, device: str) -> Tuple[Any, Any]:
        """Loading SentenceTransformer模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Loading SentenceTransformer模型 - Force this地mode
            model = SentenceTransformer(
                model_name, 
                device=device,
                local_files_only=True  # force local files only
            )
            
            # SentenceTransformer没has独立tokenizer，returnNone
            return model, None
            
        except Exception as e:
            logger.error(f"Loading SentenceTransformer模型Failed  {model_name}: {e}")
            raise
    
    def _cache_model(self, cache_key: str, model: Any, tokenizer: Any, 
                    model_path: str, device: str, is_persistent: bool = False) -> None:
        """cache模型"""
        # Check cacheSizeLimit（持久化模型不受thisLimit）
        if not is_persistent and len(self._model_cache) >= self._max_cache_size:
            self._cleanup_old_models()
        
        # Computing 内存使用量
        memory_usage = self._estimate_model_memory(model)
        
        # Creating cacheEntry
        entry = ModelCacheEntry(
            model=model,
            tokenizer=tokenizer,
            model_path=self._normalize_path(model_path),
            load_time=time.time(),
            last_access_time=time.time(),
            memory_usage=memory_usage,
            device=device,
            reference_count=1,
            is_persistent=is_persistent
        )
        
        self._model_cache[cache_key] = entry
        
        # 如果is持久化模型，添加to持久化Set
        if is_persistent:
            self._persistent_models.add(cache_key)
            logger.info(f"持久化模型已cache: {cache_key} (内存: {memory_usage / 1024**3:.2f}GB)")
        else:
            logger.info(f"模型已cache: {cache_key} (内存: {memory_usage / 1024**3:.2f}GB)")
    
    def _estimate_model_memory(self, model: Any) -> int:
        """估算模型内存使用量"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint()
            else:
                # 简单估算
                param_count = sum(p.numel() for p in model.parameters())
                return param_count * 4  # HypothesiseachParameter4bytes
        except:
            return 0
    
    def _cleanup_old_models(self) -> None:
        """清理旧模型（保护持久化模型）"""
        if not self._model_cache:
            return
        
        # per最后访问Timeranking，移除最旧模型
        sorted_entries = sorted(
            self._model_cache.items(),
            key=lambda x: (x[1].reference_count, x[1].last_access_time)
        )
        
        # 移除reference countas0且最旧非持久化模型
        for cache_key, entry in sorted_entries:
            if entry.reference_count == 0 and not entry.is_persistent:
                # 清理模型KVcache
                if hasattr(entry.model, 'past_key_values'):
                    entry.model.past_key_values = None
                if hasattr(entry.model, '_past_key_values'):
                    entry.model._past_key_values = None
                
                logger.info(f"清理旧模型: {cache_key}")
                del self._model_cache[cache_key]
                
                # 清理GPU内存
                if entry.device == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                gc.collect()
                break
    
    def _force_cleanup_all_models(self) -> None:
        """Force 清理all非持久化模型"""
        logger.info("Force 清理all非持久化cache模型")
        for cache_key, entry in list(self._model_cache.items()):
            # Skipping 持久化模型
            if entry.is_persistent:
                logger.info(f"Skipping 持久化模型: {cache_key}")
                continue
                
            # 清理模型KVcache（如果存）
            if hasattr(entry.model, 'past_key_values'):
                entry.model.past_key_values = None
            if hasattr(entry.model, '_past_key_values'):
                entry.model._past_key_values = None
            
            logger.info(f"Force 清理模型: {cache_key}")
            del self._model_cache[cache_key]
        
        # Force 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # 再times清理
        
        gc.collect()
    
    def _check_and_cleanup_memory(self) -> None:
        """Checking 并清理显存（保护持久化模型）"""
        if torch.cuda.is_available():
            # Getting 显存使用情况
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            logger.info(f"Current显存使用: {allocated / 1024**3:.2f}GB / {reserved / 1024**3:.2f}GB")
            
            # 如果显存使用过高，优先清理非持久化模型
            if allocated > 40 * 1024**3:  # 超过40GB才清理，as70B模型provide足够Space
                logger.warning("显存使用过高，Executing Force 清理（保护持久化模型）")
                self._force_cleanup_all_models()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def _start_cleanup_thread(self) -> None:
        """Starting 后台清理线程"""
        def cleanup_worker():
            while True:
                time.sleep(self._cleanup_interval)
                with self._cache_lock:
                    if self._enable_persistent_cache:
                        # 持久化cachemode下，只清理非持久化模型
                        self._cleanup_old_models()
                    else:
                        # 传统mode，清理all旧模型
                        self._cleanup_old_models()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.info("后台清理线程started")
    
    def set_model_persistent(self, model_path: str, persistent: bool = True, device: str = None) -> bool:
        """Setting 模型as持久化cache"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        normalized_model_path = self._normalize_path(model_path)
        cache_key = f"{normalized_model_path}_{device}"
        
        with self._cache_lock:
            if cache_key in self._model_cache:
                entry = self._model_cache[cache_key]
                entry.is_persistent = persistent
                
                if persistent:
                    self._persistent_models.add(cache_key)
                    logger.info(f"模型Setting as持久化: {cache_key}")
                else:
                    self._persistent_models.discard(cache_key)
                    logger.info(f"模型取消持久化: {cache_key}")
                return True
            else:
                logger.warning(f"模型Not found，no法Setting 持久化: {cache_key}")
                return False
    
    def get_persistent_models(self) -> list:
        """Getting all持久化模型List"""
        with self._cache_lock:
            return list(self._persistent_models)
    
    def clear_all_persistent_flags(self) -> None:
        """清除all持久化标记"""
        with self._cache_lock:
            for cache_key in list(self._persistent_models):
                if cache_key in self._model_cache:
                    self._model_cache[cache_key].is_persistent = False
            self._persistent_models.clear()
            logger.info("已清除all持久化标记")
    
    def enable_persistent_cache(self, enable: bool = True) -> None:
        """Enable orDisable 持久化cachefunctionality"""
        self._enable_persistent_cache = enable
        if enable:
            logger.info("持久化cachefunctionalityenabled")
        else:
            logger.info("持久化cachefunctionalitydisabled")
            # Disable 时清除all持久化标记
            self.clear_all_persistent_flags()
    
    def get_semantic_model(self, device: str = None) -> Tuple[Any, Any]:
        """Getting 语义模型（SentenceTransformer）
        
        Args:
            device: Device type，如果asNone则自动选择
            
        Returns:
            Tuple[model, tokenizer] - for于SentenceTransformer，tokenizer可能asNone
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Getting 语义模型Configuring 
        semantic_config = self._preload_configs.get('semantic_model')
        if not semantic_config:
            raise ValueError("语义模型not configured，请先调用configure_preload_models")
        
        model_name = semantic_config['model_name']
        
        try:
            # 尝试fromcacheGetting orLoading 模型
            model, tokenizer = self.get_model(model_name, device, is_persistent=True)
            logger.info(f"Successfully Getting 语义模型: {model_name}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"Getting 语义模型Failed : {e}")
            raise

    def get_cache_info(self) -> Dict[str, Any]:
        """Getting cacheInformation"""
        with self._cache_lock:
            total_memory = sum(entry.memory_usage for entry in self._model_cache.values())
            
            return {
                'cached_models': len(self._model_cache),
                'total_memory_gb': total_memory / 1024**3,
                'models': {
                    key: {
                        'model_path': entry.model_path,
                        'device': entry.device,
                        'memory_gb': entry.memory_usage / 1024**3,
                        'reference_count': entry.reference_count,
                        'last_access': entry.last_access_time
                    }
                    for key, entry in self._model_cache.items()
                }
            }
    
    def clear_cache(self, preserve_persistent: bool = True) -> None:
        """Clear allcache
        
        Args:
            preserve_persistent: is否保留持久化模型（DefaultTrue）
        """
        with self._cache_lock:
            if preserve_persistent:
                # Checking is否has持久化模型
                persistent_count = sum(1 for entry in self._model_cache.values() if entry.is_persistent)
                if persistent_count > 0:
                    logger.info(f"保留 {persistent_count} 持久化模型，Skipping 清理")
                    return
            
            logger.info("Clear 模型cache")
            
            # 清理each模型KVcache
            for cache_key, entry in self._model_cache.items():
                if hasattr(entry.model, 'past_key_values'):
                    entry.model.past_key_values = None
                if hasattr(entry.model, '_past_key_values'):
                    entry.model._past_key_values = None
            
            self._model_cache.clear()
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # 再times清理
            
            gc.collect()

# GlobalInstance
_global_model_manager = None

def get_global_model_manager() -> GlobalModelManager:
    """Getting Global model manager instance"""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = GlobalModelManager()
    return _global_model_manager

def preload_core_models(main_llm_path: Optional[str] = None, 
                        semantic_model_path: Optional[str] = None) -> None:
    """预Loading 核心模型便捷Function
    
    Args:
        main_llm_path: Main LLM path, uses default if None
        semantic_model_path: Semantic model path, uses default if None
    """
    manager = get_global_model_manager()
    manager.configure_preload_models(main_llm_path, semantic_model_path)
    manager.preload_models()

if __name__ == "__main__":
    # testCode
    manager = get_global_model_manager()
    print("Global model managertest")
    print(manager.get_cache_info())
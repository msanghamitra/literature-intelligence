# src/embeddings/model_registry.py
# Model registry for managing different embedding models.

from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional
from enum import Enum
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


# Default model (fast + good enough for MVP)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ModelType(Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    SENTENCE_TRANSFORMER = "sentence_transformer"


# Registry of available models with their metadata
MODEL_REGISTRY = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": ModelType.SENTENCE_TRANSFORMER,
        "dimension": 384,
        "description": "Fast, general-purpose sentence transformer",
        "max_length": 256,
        "language": "en",
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "type": ModelType.SENTENCE_TRANSFORMER,
        "dimension": 768,
        "description": "High-quality sentence transformer (slower but more accurate)",
        "max_length": 384,
        "language": "en",
    },
    "bert-base-uncased": {
        "name": "bert-base-uncased",
        "type": ModelType.TRANSFORMER,
        "dimension": 768,
        "description": "Original BERT base model",
        "max_length": 512,
        "language": "en",
    },
    "roberta-base": {
        "name": "roberta-base",
        "type": ModelType.TRANSFORMER,
        "dimension": 768,
        "description": "RoBERTa base model",
        "max_length": 512,
        "language": "en",
    },
    "distilbert-base-uncased": {
        "name": "distilbert-base-uncased",
        "type": ModelType.TRANSFORMER,
        "dimension": 768,
        "description": "Distilled BERT model (faster)",
        "max_length": 512,
        "language": "en",
    },
    "scibert-scivocab-uncased": {
        "name": "allenai/scibert_scivocab_uncased",
        "type": ModelType.TRANSFORMER,
        "dimension": 768,
        "description": "BERT trained on scientific text",
        "max_length": 512,
        "language": "en",
    },
    "biobert-base": {
        "name": "dmis-lab/biobert-base-cased-v1.2",
        "type": ModelType.TRANSFORMER,
        "dimension": 768,
        "description": "BERT trained on biomedical literature",
        "max_length": 512,
        "language": "en",
    },
}


def get_model_info(model_key: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Get metadata for a model from the registry.
    
    Args:
        model_key: Key of the model in the registry
        
    Returns:
        Dictionary with model metadata
    """
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    else:
        # If not in registry, assume it's a direct HuggingFace model name
        try:
            config = AutoConfig.from_pretrained(model_key)
            return {
                "name": model_key,
                "type": ModelType.TRANSFORMER,
                "dimension": config.hidden_size,
                "description": f"Custom model: {model_key}",
                "max_length": getattr(config, "max_position_embeddings", 512),
                "language": "multi",
            }
        except:
            # Fallback to default
            return MODEL_REGISTRY["all-MiniLM-L6-v2"]


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models in the registry.
    
    Returns:
        Dictionary of model keys to their metadata
    """
    return MODEL_REGISTRY.copy()


@lru_cache(maxsize=8)
def get_model(model_key: str = "all-MiniLM-L6-v2") -> Any:
    """
    Load a model from the registry with caching.
    
    Args:
        model_key: Key of the model in the registry
        
    Returns:
        Loaded model with tokenizer attached as an attribute
    """
    model_info = get_model_info(model_key)
    model_name = model_info["name"]
    model_type = model_info["type"]
    
    try:
        if model_type == ModelType.SENTENCE_TRANSFORMER:
            # For sentence-transformers, we need to use a different approach
            # since they typically come with pooling layers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            # Attach tokenizer for consistency
            model.tokenizer = model._first_module().tokenizer
        else:
            # For regular transformers
            model = AutoModel.from_pretrained(model_name)
            model.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add model info as attributes
        model.model_info = model_info
        model.model_key = model_key
        
        return model
    except Exception as e:
        print(f"Error loading model {model_key}: {e}")
        # Fallback to default model
        if model_key != "all-MiniLM-L6-v2":
            print("Falling back to default model: all-MiniLM-L6-v2")
            return get_model("all-MiniLM-L6-v2")
        else:
            raise


def get_model_dimension(model_key: str = "all-MiniLM-L6-v2") -> int:
    """
    Get the embedding dimension for a model.
    
    Args:
        model_key: Key of the model in the registry
        
    Returns:
        Embedding dimension
    """
    model_info = get_model_info(model_key)
    return model_info["dimension"]


def get_model_max_length(model_key: str = "all-MiniLM-L6-v2") -> int:
    """
    Get the maximum sequence length for a model.
    
    Args:
        model_key: Key of the model in the registry
        
    Returns:
        Maximum sequence length
    """
    model_info = get_model_info(model_key)
    return model_info["max_length"]


def clear_model_cache():
    """
    Clear the model cache.
    Useful when switching devices or freeing memory.
    """
    get_model.cache_clear()


if __name__ == "__main__":
    # Test the model registry
    print("Available models:")
    for key, info in list_available_models().items():
        print(f"  {key}: {info['description']} (dim: {info['dimension']})")
    
    # Test loading a model
    print("\nLoading default model...")
    model = get_model()
    print(f"Model loaded: {model.model_info['name']}")
    print(f"Dimension: {get_model_dimension()}")
    print(f"Max length: {get_model_max_length()}")
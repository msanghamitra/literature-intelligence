# src/retrieval/base_retriever.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict, Any
import pandas as pd

class RetrievalMode(Enum):
    """Available retrieval strategies"""
    KEYWORD = "keyword"      # arXiv default (recent first)
    SEMANTIC = "semantic"    # Embedding similarity
    HYBRID = "hybrid"        # Combined approach

@dataclass
class RetrievalConfig:
    """Configuration for retrieval behavior"""
    mode: RetrievalMode = RetrievalMode.SEMANTIC
    candidate_cap: int = 200
    semantic_min_similarity: float = 0.20
    use_fallback: bool = True
    
    # Hybrid weights
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    
    # Scientific feature weights
    citation_weight: float = 0.0  # Not available yet
    venue_weight: float = 0.3
    recency_weight: float = 0.2

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Retrieve papers for a query"""
        pass
    
    def get_name(self) -> str:
        """Get retriever name"""
        return self.__class__.__name__
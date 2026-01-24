from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import datetime

class SearchMode(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

@dataclass
class SearchQuery:
    """Search parameters from user."""
    text: str
    mode: SearchMode = SearchMode.HYBRID
    max_results: int = 20
    categories: List[str] = field(default_factory=list)
    candidate_cap: int = 200
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Weights for hybrid mode
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    venue_weight: float = 0.3
    recency_weight: float = 0.2
    semantic_min_similarity: float = 0.20
    
    @classmethod
    def from_legacy_args(cls, topic: str, mode: str, top_k: int, 
                         category: Optional[str] = None, 
                         candidate_cap: int = 200) -> "SearchQuery":
        """Backward compatibility with old search_arxiv_live()"""
        mode_enum = {
            "keyword": SearchMode.KEYWORD,
            "semantic": SearchMode.SEMANTIC,
            "hybrid": SearchMode.HYBRID
        }.get(mode.lower(), SearchMode.HYBRID)
        
        categories = []
        if category:
            categories = [c.strip() for c in category.split(",") if c.strip()]
        
        return cls(
            text=topic,
            mode=mode_enum,
            max_results=top_k,
            categories=categories,
            candidate_cap=candidate_cap
        )
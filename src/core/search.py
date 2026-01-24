"""
Main search entry point - uses domain models.
"""
import pandas as pd
import time
from typing import Tuple, Optional

from src.core.query import SearchQuery, SearchMode
from src.retrieval import (
    KeywordRetriever, SemanticRetriever, HybridRetriever,
    RetrievalConfig, RetrievalMode
)

# PUBLIC API - Same signature as before
def search_arxiv_live(
    topic: str,
    mode: str,
    top_k: int,
    category: Optional[str] = None,
    candidate_cap: int = 200,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[pd.DataFrame, str]:
    """
    EXACT SAME FUNCTION SIGNATURE as before.
    Your Streamlit app calls this unchanged.
    """
    # Convert to domain object
    query = SearchQuery.from_legacy_args(
        topic=topic,
        mode=mode,
        top_k=top_k,
        category=category,
        candidate_cap=candidate_cap
    )
    
    # Map to retriever
    config_kwargs = {
        "mode": RetrievalMode(mode.upper()),
        "candidate_cap": candidate_cap,
        "semantic_min_similarity": query.semantic_min_similarity,
        "keyword_weight": query.keyword_weight,
        "semantic_weight": query.semantic_weight,
        "venue_weight": query.venue_weight,
        "recency_weight": query.recency_weight
    }
    
    if query.categories:
        config_kwargs["categories"] = query.categories
    
    config = RetrievalConfig(**config_kwargs)
    
    # Choose retriever
    if mode == "keyword":
        retriever = KeywordRetriever(config)
    elif mode == "semantic":
        retriever = SemanticRetriever(config)
    else:  # hybrid or default
        retriever = HybridRetriever(config)
    
    # Execute search (same as before)
    results_df, message = retriever.retrieve(
        query.text, 
        top_k=query.max_results
    )
    
    return results_df, message

# NEW: Better API for internal use
def search_with_query(query: SearchQuery) -> Tuple[pd.DataFrame, str]:
    """Use domain query object directly."""
    return search_arxiv_live(
        topic=query.text,
        mode=query.mode.value,
        top_k=query.max_results,
        category=",".join(query.categories) if query.categories else None,
        candidate_cap=query.candidate_cap,
        model_name=query.model_name
    )

# Helper functions from old search.py
def create_retriever(mode: str = "semantic", **config_kwargs):
    """Factory function to create retrievers."""
    config = RetrievalConfig(**config_kwargs)
    
    if mode == "keyword":
        return KeywordRetriever(config)
    elif mode == "semantic":
        return SemanticRetriever(config)
    elif mode == "hybrid":
        return HybridRetriever(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def advanced_search(query: str, mode: str = "hybrid", top_k: int = 10, 
                    category: Optional[str] = None, include_scores: bool = True):
    """Advanced search with metadata."""
    retriever = create_retriever(mode=mode)
    results, message = retriever.retrieve(query, top_k=top_k, category=category)
    
    metadata = {
        "mode": mode,
        "results_count": len(results),
        "has_scores": "similarity" in results.columns or "hybrid_score" in results.columns
    }
    
    if not include_scores and not results.empty:
        cols_to_drop = ["_source", "similarity_norm", "source_score", 
                       "venue_score", "recency_score"]
        cols_to_drop = [c for c in cols_to_drop if c in results.columns]
        results = results.drop(columns=cols_to_drop)
    
    return results, message, metadata

# Quick test - same as before
if __name__ == "__main__":
    print("Testing backward compatibility...")
    df, msg = search_arxiv_live("machine learning", "semantic", top_k=3)
    print(f"Results: {len(df)} papers, Message: {msg}")
    
    if not df.empty:
        print(f"Columns: {list(df.columns)}")
    
    print("\nTesting domain query...")
    query = SearchQuery.from_legacy_args("transformer", "hybrid", 5)
    df2, msg2 = search_with_query(query)
    print(f"Domain query results: {len(df2)} papers")
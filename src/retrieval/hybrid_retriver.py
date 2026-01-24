# src/retrieval/hybrid_retriever.py
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
from .base_retriever import BaseRetriever, RetrievalConfig, RetrievalMode
from .keyword_retriever import KeywordRetriever
from .semantic_retriever import SemanticRetriever

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining keyword and semantic approaches"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        config = config or RetrievalConfig(mode=RetrievalMode.HYBRID)
        super().__init__(config)
        self.keyword_retriever = KeywordRetriever(config)
        self.semantic_retriever = SemanticRetriever(config)
    
    def retrieve(self, query: str, top_k: int = 10, category: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Hybrid retrieval combining both strategies"""
        if not query or not query.strip():
            return pd.DataFrame(), "Please enter a query."
        
        # Get results from both retrievers
        keyword_df, kw_msg = self.keyword_retriever.retrieve(
            query, top_k=self.config.candidate_cap, category=category
        )
        semantic_df, sem_msg = self.semantic_retriever.retrieve(
            query, top_k=self.config.candidate_cap, category=category
        )
        
        if keyword_df.empty and semantic_df.empty:
            return pd.DataFrame(), "No results found from either retriever."
        
        # Combine and deduplicate
        combined = self._combine_results(keyword_df, semantic_df)
        
        # Apply hybrid scoring
        scored = self._apply_hybrid_scoring(combined)
        
        return scored.head(top_k), f"Hybrid search: {len(combined)} unique papers"
    
    def _combine_results(self, keyword_df: pd.DataFrame, semantic_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and deduplicate results"""
        # Tag source
        keyword_df = keyword_df.copy()
        semantic_df = semantic_df.copy()
        keyword_df["_source"] = "keyword"
        semantic_df["_source"] = "semantic"
        
        # Ensure similarity column exists
        if "similarity" not in keyword_df.columns:
            keyword_df["similarity"] = np.nan
        
        # Combine
        combined = pd.concat([keyword_df, semantic_df], ignore_index=True)
        
        # Deduplicate by arxiv_id, keep highest similarity
        combined = combined.sort_values("similarity", ascending=False, na_position='last')
        combined = combined.drop_duplicates(subset=["arxiv_id"], keep="first")
        
        return combined.reset_index(drop=True)
    
    def _apply_hybrid_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hybrid scoring algorithm"""
        df = df.copy()
        
        # Normalize similarity scores to [0, 1]
        if df["similarity"].notna().any():
            sim_min = df["similarity"].min()
            sim_max = df["similarity"].max()
            if sim_max > sim_min:
                df["similarity_norm"] = (df["similarity"] - sim_min) / (sim_max - sim_min)
            else:
                df["similarity_norm"] = 0.5
        else:
            df["similarity_norm"] = 0.5
        
        # Source score (keyword papers get base score)
        df["source_score"] = df["_source"].apply(lambda x: 1.0 if x == "keyword" else 0.0)
        
        # Extract simple features
        df["venue_score"] = self._extract_venue_score(df)
        df["recency_score"] = self._extract_recency_score(df)
        
        # Hybrid score calculation
        df["hybrid_score"] = (
            self.config.keyword_weight * df["source_score"] +
            self.config.semantic_weight * df["similarity_norm"] +
            self.config.venue_weight * df["venue_score"] +
            self.config.recency_weight * df["recency_score"]
        )
        
        # Sort by hybrid score
        return df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    
    def _extract_venue_score(self, df: pd.DataFrame) -> pd.Series:
        """Simple venue scoring based on arXiv categories"""
        def score_category(cat: str) -> float:
            if not isinstance(cat, str):
                return 0.5
            
            high_prestige = {"cs.CV", "cs.LG", "cs.CL", "stat.ML"}
            med_prestige = {"cs.AI", "cs.NE", "q-bio.QM"}
            
            if cat in high_prestige:
                return 0.9
            elif cat in med_prestige:
                return 0.7
            else:
                return 0.5
        
        if "primary_category" in df.columns:
            return df["primary_category"].apply(score_category)
        return pd.Series([0.5] * len(df))
    
    def _extract_recency_score(self, df: pd.DataFrame) -> pd.Series:
        """Score based on publication recency"""
        def score_date(date_str: str) -> float:
            try:
                year = int(str(date_str)[:4])
                years_old = 2024 - year
                return max(0.1, np.exp(-years_old / 5))
            except:
                return 0.5
        
        if "published" in df.columns:
            return df["published"].apply(score_date)
        return pd.Series([0.5] * len(df))
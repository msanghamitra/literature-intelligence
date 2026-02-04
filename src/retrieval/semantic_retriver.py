# src/retrieval/semantic_retriever.py
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from .base_retriever import BaseRetriever, RetrievalConfig, RetrievalMode
from arxiv_loader import fetch_arxiv_papers_df
from src.embeddings.embedder import embed_text, embed_batch

class SemanticRetriever(BaseRetriever):
    """Retriever using semantic similarity (embeddings)"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config or RetrievalConfig(mode=RetrievalMode.SEMANTIC))
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    def retrieve(self, query: str, top_k: int = 10, category: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Semantic retrieval with reranking"""
        if not query or not query.strip():
            return pd.DataFrame(), "Please enter a query."
        
        # Fetch candidates
        candidate_n = min(self.config.candidate_cap, max(top_k * 3, 50))
        df_raw, msg = fetch_arxiv_papers_df(
            topic=query,
            max_results=candidate_n,
            category=category
        )
        
        if df_raw.empty:
            return df_raw, msg
        
        # Normalize
        df = self._normalize_df(df_raw)
        
        # Semantic reranking
        df_reranked = self._semantic_rerank(query, df)
        
        # Fallback if similarity is low
        best_sim = float(df_reranked["similarity"].max()) if "similarity" in df_reranked.columns else 0.0
        if self.config.use_fallback and best_sim < self.config.semantic_min_similarity:
            df_fallback = self._normalize_df(df_raw).head(top_k).copy()
            df_fallback["similarity"] = np.nan
            return df_fallback, f"No strong semantic matches. Showing latest {top_k} papers."
        
        return df_reranked.head(top_k), msg
    
    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic dataframe normalization"""
        df = df.copy()
        if df.empty:
            return df
        
        df["title"] = df.get("title", "Untitled").fillna("Untitled")
        df["summary"] = df.get("summary", "").fillna("")
        if "text_unit" not in df.columns:
            df["text_unit"] = df["title"] + ". " + df["summary"]
        
        return df
    
    def _semantic_rerank(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """Rerank candidates by semantic similarity"""
        if df.empty:
            return df
        
        texts = df["text_unit"].astype(str).fillna("").tolist()
        
        # Embed query and candidates
        q_vec = embed_text(query, model_name=self.model_name)
        cand_vecs = embed_batch(texts, model_name=self.model_name, show_progress_bar=False)
        
        # Cosine similarity
        sims = cand_vecs @ q_vec
        
        # Add similarity scores
        df_ranked = df.copy()
        df_ranked["similarity"] = sims
        return df_ranked.sort_values("similarity", ascending=False).reset_index(drop=True)
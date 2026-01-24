# src/retrieval/keyword_retriever.py
from typing import Tuple, Optional
import pandas as pd
from .base_retriever import BaseRetriever, RetrievalConfig, RetrievalMode
from arxiv_loader import fetch_arxiv_papers_df

class KeywordRetriever(BaseRetriever):
    """Retriever using arXiv's keyword search (most recent first)"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config or RetrievalConfig(mode=RetrievalMode.KEYWORD))
    
    def retrieve(self, query: str, top_k: int = 10, category: Optional[str] = None, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Keyword retrieval - uses arXiv's default ranking"""
        if not query or not query.strip():
            return pd.DataFrame(), "Please enter a query."
        
        df_raw, msg = fetch_arxiv_papers_df(
            topic=query, 
            max_results=top_k, 
            category=category
        )
        
        if df_raw.empty:
            return df_raw, msg
        
        # Normalize dataframe
        df = self._normalize_df(df_raw)
        return df.head(top_k), msg
    
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
# src/services/search_service.py
"""
BUSINESS LOGIC: Search functionality
Moved from streamlit_app.py
"""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from src.data.arxiv_loader import *


@dataclass
class Paper:
    """Domain model for papers"""
    id: str
    title: str
    abstract: str
    authors: str
    published: str
    pdf_url: str
    entry_id: str
    similarity: Optional[float] = None
    text_unit: Optional[str] = None
    
    @property
    def authors_list(self):
        """Parse authors string into list"""
        if not self.authors:
            return []
        return [a.strip() for a in self.authors.split(",")]


class SearchService:
    """Service for search operations"""
    
    def search_arxiv(self, topic: str, mode: str, top_k: int, 
                     category: Optional[str] = None) -> List[Paper]:
        """Search arXiv and return Paper objects"""
        # Call existing search function - FIXED FUNCTION NAME
        df, msg = fetch_arxiv_papers_df(  # CHANGED FROM search_arxiv_live
            topic=topic,
            max_results=top_k,  # CHANGED PARAMETER NAME
            category=category
        )
        
        if df.empty:
            return []
        
        # Convert DataFrame to Paper objects
        return self._df_to_papers(df)
    
    def _df_to_papers(self, df: pd.DataFrame) -> List[Paper]:
        """Convert DataFrame to list of Paper objects"""
        papers = []
        for _, row in df.iterrows():
            paper = Paper(
                id=row.get("arxiv_id", f"row_{_}"),
                title=row.get("title", "Untitled"),
                abstract=row.get("summary", ""),
                authors=row.get("authors", "Unknown authors"),
                published=row.get("published", ""),
                pdf_url=row.get("pdf_url", ""),
                entry_id=row.get("entry_id", ""),
                similarity=row.get("similarity"),
                text_unit=row.get("text_unit", "")
            )
            papers.append(paper)
        return papers
# src/services/summary_service.py
"""
BUSINESS LOGIC: Summarization service
USES: src.preprocessing.summarizer
"""
from typing import List
from src.services.search_service import Paper

# Import existing  functions
from src.preprocessing.summarizer import summarize_text, summarize_batch


class SummaryService:
    """Service that USES the existing summarizer model"""
    
    def summarize_paper(self, paper, max_length: int = 128, min_length: int = 32) -> str:
        """Summarize a single paper"""
        text = getattr(paper, 'text_unit', None) or getattr(paper, 'abstract', '')
        
        if not text:
            return "No text available for summarization."
        
        # Use EXISTING function from summarizer.py
        return summarize_text(
            text=text,
            max_length=max_length,
            min_length=min_length
        )
    
    def summarize_papers_batch(self, papers: List[Paper], **kwargs) -> List[str]:
        """Summarize multiple papers"""
        texts = []
        for paper in papers:
            text = getattr(paper, 'text_unit', None) or getattr(paper, 'abstract', '')
            if text:
                texts.append(text)
        
        if not texts:
            return []
        
        # Use EXISTING batch function from summarizer.py
        return summarize_batch(texts, **kwargs)
    
    def generate_corpus_summaries(self):
        """
        Generate summaries for entire corpus
        Uses existing run_full_corpus_summarisation() from summarizer.py
        """
        from src.preprocessing.summarizer import run_full_corpus_summarisation
        
        try:
            df = run_full_corpus_summarisation()
            return {
                "success": True,
                "message": f"Generated summaries for {len(df)} papers",
                "output_path": str(getattr(df, 'output_path', 'Unknown'))
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
# src/services/qa_service.py
"""
BUSINESS LOGIC: Q&A functionality
Moved from streamlit_app.py
"""
from typing import Dict, Any
from src.services.search_service import Paper
from src.data.pdf_text import get_pdf_pages_from_url
from src.rag.qa_engine import QAEngine


class QAService:
    """Service for paper Q&A"""
    
    def answer_question(self, paper: Paper, question: str) -> Dict[str, Any]:
        """Answer question about a paper"""
        if not question or not question.strip():
            return {"error": "Please enter a question."}
        
        if not paper.pdf_url or not str(paper.pdf_url).strip():
            return {"error": "No PDF URL available for this paper."}
        
        try:
            # Get PDF pages
            pages, msg = get_pdf_pages_from_url(str(paper.pdf_url), cache_key=str(paper.id))
            
            if not pages:
                return {"error": f"Could not extract PDF text: {msg}"}
            
            # Get answer
            answer = answer_question_from_pdf(
                question=question, 
                pages=pages, 
                top_chunks=5
            )
            
            return answer
            
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}
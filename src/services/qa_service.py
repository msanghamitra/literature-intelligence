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
    
    def __init__(self):
        """Initialize QA service with QA engine"""
        self.qa_engine = QAEngine()
    
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
            
            # Create temporary chunks from pages for context building
            from src.rag.context_builder import DocumentChunk
            chunks = []
            for pdf_page in pages:
                # PdfPageText object has .page and .text attributes
                chunk = DocumentChunk(
                    text=pdf_page.text,  # Extract text from PdfPageText object
                    metadata={
                        'source': str(paper.pdf_url),
                        'page': pdf_page.page,
                        'title': paper.title
                    },
                    embedding=None
                )
                chunks.append(chunk)
            
            # Generate embeddings for chunks
            chunks = self.qa_engine.context_builder.embed_chunks(chunks)
            
            # Store chunks temporarily
            paper_id = str(paper.id)
            self.qa_engine.loaded_documents[paper_id] = chunks
            
            # Query using the QA engine
            result = self.qa_engine.query(
                query=question,
                pdf_paths=[paper_id],
                top_k=5
            )
            
            # Clean up
            self.qa_engine.unload_document(paper_id)
            
            # Format response
            if result.get('answer'):
                return {
                    "answer": result['answer'],
                    "context_snippet": result.get('context', '')[:500] + "..." if len(result.get('context', '')) > 500 else result.get('context', ''),
                    "from_cache": result.get('from_cache', False)
                }
            else:
                return {"error": "Could not generate answer"}
            
        except Exception as e:
            return {"error": f"Error processing question: {str(e)}"}
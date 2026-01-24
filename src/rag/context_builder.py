"""
Context Builder for RAG System
Handles document chunking, embedding, and context preparation
"""
from typing import List, Dict, Any, Optional
import hashlib
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import PyPDF2
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None

class ContextBuilder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the context builder
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
        return text
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Split document into chunks
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of document chunks
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        text_chunks = self.text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(text_chunks)
            })
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))
        
        return chunks
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def build_context_from_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Complete pipeline: extract, chunk, and embed PDF
        
        Args:
            pdf_path: Path to PDF file
            metadata: Document metadata
            
        Returns:
            List of embedded document chunks
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Chunk document
        chunks = self.chunk_document(text, metadata)
        
        # Embed chunks
        embedded_chunks = self.embed_chunks(chunks)
        
        return embedded_chunks
    
    def find_relevant_chunks(self, 
                            query: str, 
                            chunks: List[DocumentChunk], 
                            top_k: int = 5) -> List[DocumentChunk]:
        """
        Find most relevant chunks for a query
        
        Args:
            query: User query
            chunks: List of document chunks
            top_k: Number of top chunks to return
            
        Returns:
            List of most relevant chunks
        """
        if not chunks:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            if chunk.embedding is not None:
                similarity = np.dot(query_embedding, chunk.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                )
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]
    
    def build_prompt_context(self, relevant_chunks: List[DocumentChunk]) -> str:
        """
        Build context string from relevant chunks
        
        Args:
            relevant_chunks: List of relevant document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            source_info = chunk.metadata.get('source', 'Unknown')
            context_parts.append(f"[Context {i+1} from {source_info}]:\n{chunk.text}\n")
        
        return "\n".join(context_parts)
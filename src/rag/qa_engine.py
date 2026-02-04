"""
QA Engine for RAG System
Lightweight extractive QA approach (no heavy LLM required)
"""
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.context_builder import ContextBuilder, DocumentChunk
from src.rag.cache_manager import CacheManager

class QAEngine:
    def __init__(self, 
                 cache_enabled: bool = True,
                 cache_dir: str = "./cache"):
        """
        Initialize QA Engine with lightweight extractive QA
        
        Args:
            cache_enabled: Whether to enable caching
            cache_dir: Directory for cache storage
        """
        # Initialize context builder
        self.context_builder = ContextBuilder()
        
        # Initialize cache manager
        self.cache_enabled = cache_enabled
        self.cache_manager = CacheManager(cache_dir) if cache_enabled else None
        
        # Store loaded documents
        self.loaded_documents: Dict[str, List[DocumentChunk]] = {}
    
    def load_pdf_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Load and process PDF document
        
        Args:
            pdf_path: Path to PDF file
            metadata: Document metadata
            
        Returns:
            List of processed document chunks
        """
        if metadata is None:
            metadata = {'source': os.path.basename(pdf_path)}
        
        # Check cache first
        if self.cache_enabled:
            cached_chunks = self.cache_manager.get_cached_document_chunks(pdf_path)
            if cached_chunks:
                print(f"Loaded {len(cached_chunks)} chunks from cache for {pdf_path}")
                self.loaded_documents[pdf_path] = cached_chunks
                return cached_chunks
        
        # Process document
        print(f"Processing PDF: {pdf_path}")
        chunks = self.context_builder.build_context_from_pdf(pdf_path, metadata)
        
        # Cache the results
        if self.cache_enabled:
            self.cache_manager.cache_document_chunks(pdf_path, chunks)
        
        # Store in memory
        self.loaded_documents[pdf_path] = chunks
        
        print(f"Processed {len(chunks)} chunks from {pdf_path}")
        return chunks
    
    def unload_document(self, pdf_path: str) -> None:
        """
        Unload a document from memory
        
        Args:
            pdf_path: Path to PDF file to unload
        """
        if pdf_path in self.loaded_documents:
            del self.loaded_documents[pdf_path]
            print(f"Unloaded document: {pdf_path}")
    
    def get_loaded_documents(self) -> List[str]:
        """
        Get list of loaded document paths
        
        Returns:
            List of loaded document paths
        """
        return list(self.loaded_documents.keys())
    
    def query(self, 
              query: str, 
              pdf_paths: Optional[List[str]] = None,
              top_k: int = 5) -> Dict[str, Any]:
        """
        Query the loaded documents using extractive QA
        
        Args:
            query: User query
            pdf_paths: Specific documents to query (None for all loaded)
            top_k: Number of relevant chunks to use
            
        Returns:
            Dictionary with answer and metadata
        """
        # Use all loaded documents if none specified
        if pdf_paths is None:
            pdf_paths = list(self.loaded_documents.keys())
        
        if not pdf_paths:
            return {
                'answer': 'No documents loaded. Please load a PDF first.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Check query cache
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_query_result(query)
            if cached_result:
                context, answer = cached_result
                return {
                    'answer': answer,
                    'sources': pdf_paths,
                    'context': context,
                    'from_cache': True
                }
        
        # Collect all chunks from specified documents
        all_chunks = []
        for pdf_path in pdf_paths:
            if pdf_path in self.loaded_documents:
                all_chunks.extend(self.loaded_documents[pdf_path])
            else:
                print(f"Warning: Document {pdf_path} not loaded")
        
        if not all_chunks:
            return {
                'answer': 'No relevant document chunks found.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Find relevant chunks
        relevant_chunks = self.context_builder.find_relevant_chunks(query, all_chunks, top_k)
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant information found in the documents.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Build context
        context = self.context_builder.build_prompt_context(relevant_chunks)
        
        # Generate answer using extractive approach (no heavy LLM)
        answer = self._generate_extractive_answer(query, relevant_chunks)
        
        # Cache the result
        if self.cache_enabled:
            self.cache_manager.cache_query_result(query, context, answer)
        
        # Extract source information
        sources = []
        for chunk in relevant_chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'from_cache': False
        }
    
    def _generate_extractive_answer(self, query: str, relevant_chunks: List[DocumentChunk]) -> str:
        """
        Generate answer using lightweight extractive approach
        Returns the most relevant text passages
        
        Args:
            query: User query
            relevant_chunks: Most relevant document chunks
            
        Returns:
            Generated answer (extracted text)
        """
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Build answer from most relevant chunks
        answer_parts = []
        
        for i, chunk in enumerate(relevant_chunks[:3], 1):  # Use top 3 chunks
            page = chunk.metadata.get('page', 'Unknown')
            text = chunk.text.strip()
            
            # Truncate long chunks
            if len(text) > 300:
                text = text[:300] + "..."
            
            answer_parts.append(f"[Page {page}]: {text}")
        
        answer = "\n\n".join(answer_parts)
        
        # Add a helpful note
        answer = f"Based on the paper, here are the most relevant passages:\n\n{answer}\n\n(This is an extractive answer from the document. For specific details, please refer to the full paper.)"
        
        return answer
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache
        
        Args:
            cache_type: Specific cache type to clear
        """
        if self.cache_enabled:
            self.cache_manager.clear_cache(cache_type)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache_enabled:
            return self.cache_manager.get_cache_stats()
        return {}
    
    def live_qa_session(self, pdf_path: str):
        """
        Start a live Q&A session for a PDF
        
        Args:
            pdf_path: Path to PDF file
        """
        # Load the document
        self.load_pdf_document(pdf_path)
        
        print(f"\n=== Live Q&A Session for {os.path.basename(pdf_path)} ===")
        print("Type 'exit' to end the session")
        print("Type 'clear' to clear the cache")
        print("Type 'stats' to see cache statistics")
        print("=" * 50)
        
        while True:
            try:
                query = input("\nQuestion: ").strip()
                
                if query.lower() == 'exit':
                    print("Ending Q&A session.")
                    break
                
                elif query.lower() == 'clear':
                    self.clear_cache()
                    print("Cache cleared.")
                    continue
                
                elif query.lower() == 'stats':
                    stats = self.get_cache_stats()
                    if stats:
                        for cache_type, stat in stats.items():
                            print(f"{cache_type}: {stat['valid_entries']}/{stat['total_entries']} entries")
                    else:
                        print("Cache disabled or no statistics available.")
                    continue
                
                elif not query:
                    continue
                
                # Process query
                result = self.query(query, [pdf_path])
                
                print(f"\nAnswer: {result['answer']}")
                
                if result['sources']:
                    print(f"\nSources: {', '.join(result['sources'])}")
                
                if result['from_cache']:
                    print("(Retrieved from cache)")
            
            except KeyboardInterrupt:
                print("\n\nSession interrupted.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


# Main function for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG QA Engine (Lightweight)")
    parser.add_argument("--pdf", type=str, help="Path to PDF file for Q&A")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Initialize QA Engine
    qa_engine = QAEngine(
        cache_enabled=not args.no_cache
    )
    
    if args.pdf:
        if os.path.exists(args.pdf):
            # Start live Q&A session
            qa_engine.live_qa_session(args.pdf)
        else:
            print(f"Error: PDF file not found: {args.pdf}")
    else:
        print("Please provide a PDF file with --pdf argument")
    
    def load_pdf_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Load and process PDF document
        
        Args:
            pdf_path: Path to PDF file
            metadata: Document metadata
            
        Returns:
            List of processed document chunks
        """
        if metadata is None:
            metadata = {'source': os.path.basename(pdf_path)}
        
        # Check cache first
        if self.cache_enabled:
            cached_chunks = self.cache_manager.get_cached_document_chunks(pdf_path)
            if cached_chunks:
                print(f"Loaded {len(cached_chunks)} chunks from cache for {pdf_path}")
                self.loaded_documents[pdf_path] = cached_chunks
                return cached_chunks
        
        # Process document
        print(f"Processing PDF: {pdf_path}")
        chunks = self.context_builder.build_context_from_pdf(pdf_path, metadata)
        
        # Cache the results
        if self.cache_enabled:
            self.cache_manager.cache_document_chunks(pdf_path, chunks)
        
        # Store in memory
        self.loaded_documents[pdf_path] = chunks
        
        print(f"Processed {len(chunks)} chunks from {pdf_path}")
        return chunks
    
    def unload_document(self, pdf_path: str) -> None:
        """
        Unload a document from memory
        
        Args:
            pdf_path: Path to PDF file to unload
        """
        if pdf_path in self.loaded_documents:
            del self.loaded_documents[pdf_path]
            print(f"Unloaded document: {pdf_path}")
    
    def get_loaded_documents(self) -> List[str]:
        """
        Get list of loaded document paths
        
        Returns:
            List of loaded document paths
        """
        return list(self.loaded_documents.keys())
    
    def query(self, 
              query: str, 
              pdf_paths: Optional[List[str]] = None,
              top_k: int = 5) -> Dict[str, Any]:
        """
        Query the loaded documents
        
        Args:
            query: User query
            pdf_paths: Specific documents to query (None for all loaded)
            top_k: Number of relevant chunks to use
            
        Returns:
            Dictionary with answer and metadata
        """
        # Use all loaded documents if none specified
        if pdf_paths is None:
            pdf_paths = list(self.loaded_documents.keys())
        
        if not pdf_paths:
            return {
                'answer': 'No documents loaded. Please load a PDF first.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Check query cache
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_query_result(query)
            if cached_result:
                context, answer = cached_result
                return {
                    'answer': answer,
                    'sources': pdf_paths,
                    'context': context,
                    'from_cache': True
                }
        
        # Collect all chunks from specified documents
        all_chunks = []
        for pdf_path in pdf_paths:
            if pdf_path in self.loaded_documents:
                all_chunks.extend(self.loaded_documents[pdf_path])
            else:
                print(f"Warning: Document {pdf_path} not loaded")
        
        if not all_chunks:
            return {
                'answer': 'No relevant document chunks found.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Find relevant chunks
        relevant_chunks = self.context_builder.find_relevant_chunks(query, all_chunks, top_k)
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant information found in the documents.',
                'sources': [],
                'context': '',
                'from_cache': False
            }
        
        # Build context
        context = self.context_builder.build_prompt_context(relevant_chunks)
        
        # Generate prompt
        prompt = self._build_qa_prompt(query, context)
        
        # Generate answer
        answer = self._generate_answer(prompt)
        
        # Cache the result
        if self.cache_enabled:
            self.cache_manager.cache_query_result(query, context, answer)
        
        # Extract source information
        sources = []
        for chunk in relevant_chunks:
            source = chunk.metadata.get('source', 'Unknown')
            if source not in sources:
                sources.append(source)
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'from_cache': False
        }
    
    def _build_qa_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for QA
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        # Limit context to prevent token overflow (approx 2000 chars = ~500 tokens)
        max_context_chars = 2000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "...\n[Context truncated]"
        
        prompt_template = """Context: {context}

Question: {question}

Answer:"""
        
        return prompt_template.format(context=context, question=query)
    
    def _generate_answer(self, prompt: str) -> str:
        """
        Generate answer using LLM
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated answer
        """
        try:
            # Use invoke() method for LangChain models
            response = self.llm.invoke(prompt)
            
            # Response is typically a string with the full generation
            answer = response
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            return answer.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache
        
        Args:
            cache_type: Specific cache type to clear
        """
        if self.cache_enabled:
            self.cache_manager.clear_cache(cache_type)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if self.cache_enabled:
            return self.cache_manager.get_cache_stats()
        return {}
    
    def live_qa_session(self, pdf_path: str):
        """
        Start a live Q&A session for a PDF
        
        Args:
            pdf_path: Path to PDF file
        """
        # Load the document
        self.load_pdf_document(pdf_path)
        
        print(f"\n=== Live Q&A Session for {os.path.basename(pdf_path)} ===")
        print("Type 'exit' to end the session")
        print("Type 'clear' to clear the cache")
        print("Type 'stats' to see cache statistics")
        print("=" * 50)
        
        while True:
            try:
                query = input("\nQuestion: ").strip()
                
                if query.lower() == 'exit':
                    print("Ending Q&A session.")
                    break
                
                elif query.lower() == 'clear':
                    self.clear_cache()
                    print("Cache cleared.")
                    continue
                
                elif query.lower() == 'stats':
                    stats = self.get_cache_stats()
                    if stats:
                        for cache_type, stat in stats.items():
                            print(f"{cache_type}: {stat['valid_entries']}/{stat['total_entries']} entries")
                    else:
                        print("Cache disabled or no statistics available.")
                    continue
                
                elif not query:
                    continue
                
                # Process query
                result = self.query(query, [pdf_path])
                
                print(f"\nAnswer: {result['answer']}")
                
                if result['sources']:
                    print(f"\nSources: {', '.join(result['sources'])}")
                
                if result['from_cache']:
                    print("(Retrieved from cache)")
            
            except KeyboardInterrupt:
                print("\n\nSession interrupted.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


# Main function for standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG QA Engine")
    parser.add_argument("--pdf", type=str, help="Path to PDF file for Q&A")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Initialize QA Engine
    qa_engine = QAEngine(
        model_name=args.model,
        cache_enabled=not args.no_cache
    )
    
    if args.pdf:
        if os.path.exists(args.pdf):
            # Start live Q&A session
            qa_engine.live_qa_session(args.pdf)
        else:
            print(f"Error: PDF file not found: {args.pdf}")
    else:
        print("Please provide a PDF file with --pdf argument")
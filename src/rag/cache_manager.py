"""
Cache Manager for RAG System
Handles caching of embeddings and query results
"""
import json
import pickle
import hashlib
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different cache types
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.query_cache_dir = self.cache_dir / "queries"
        self.chunk_cache_dir = self.cache_dir / "chunks"
        
        # Create subdirectories
        for dir_path in [self.embedding_cache_dir, self.query_cache_dir, self.chunk_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Cache configuration
        self.max_cache_size = 1000  # Maximum entries per cache type
        self.cache_ttl_hours = 24  # Time to live in hours
    
    def _generate_cache_key(self, data: Any) -> str:
        """
        Generate cache key from data
        
        Args:
            data: Data to generate key from
            
        Returns:
            MD5 hash string
        """
        if isinstance(data, str):
            data_str = data
        elif isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, list):
            data_str = str(data)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """
        Get cache file path
        
        Args:
            cache_type: Type of cache ('embeddings', 'queries', 'chunks')
            key: Cache key
            
        Returns:
            Path to cache file
        """
        if cache_type == "embeddings":
            return self.embedding_cache_dir / f"{key}.pkl"
        elif cache_type == "queries":
            return self.query_cache_dir / f"{key}.json"
        elif cache_type == "chunks":
            return self.chunk_cache_dir / f"{key}.pkl"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cache is still valid based on TTL
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check file age
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        cache_age = datetime.now() - file_mtime
        
        return cache_age < timedelta(hours=self.cache_ttl_hours)
    
    def cache_embeddings(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache text embeddings
        
        Args:
            text: Original text
            embedding: Generated embedding
        """
        key = self._generate_cache_key(text)
        cache_path = self._get_cache_path("embeddings", key)
        
        # Clean old cache if needed
        self._clean_old_cache("embeddings")
        
        # Save embedding
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'text': text,
                'embedding': embedding,
                'timestamp': datetime.now().isoformat()
            }, f)
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._generate_cache_key(text)
        cache_path = self._get_cache_path("embeddings", key)
        
        if cache_path.exists() and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    return data['embedding']
            except:
                pass
        
        return None
    
    def cache_query_result(self, query: str, context: str, answer: str) -> None:
        """
        Cache query results
        
        Args:
            query: User query
            context: Used context
            answer: Generated answer
        """
        key = self._generate_cache_key(query)
        cache_path = self._get_cache_path("queries", key)
        
        # Clean old cache if needed
        self._clean_old_cache("queries")
        
        # Save query result
        with open(cache_path, 'w') as f:
            json.dump({
                'query': query,
                'context': context,
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def get_cached_query_result(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Get cached query result
        
        Args:
            query: User query
            
        Returns:
            Tuple of (context, answer) or None if not found
        """
        key = self._generate_cache_key(query)
        cache_path = self._get_cache_path("queries", key)
        
        if cache_path.exists() and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data['context'], data['answer']
            except:
                pass
        
        return None
    
    def cache_document_chunks(self, pdf_path: str, chunks: List[Any]) -> None:
        """
        Cache document chunks
        
        Args:
            pdf_path: Path to PDF file
            chunks: List of document chunks
        """
        key = self._generate_cache_key(pdf_path)
        cache_path = self._get_cache_path("chunks", key)
        
        # Clean old cache if needed
        self._clean_old_cache("chunks")
        
        # Save chunks
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'pdf_path': pdf_path,
                'chunks': chunks,
                'timestamp': datetime.now().isoformat()
            }, f)
    
    def get_cached_document_chunks(self, pdf_path: str) -> Optional[List[Any]]:
        """
        Get cached document chunks
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of document chunks or None if not found
        """
        key = self._generate_cache_key(pdf_path)
        cache_path = self._get_cache_path("chunks", key)
        
        if cache_path.exists() and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    return data['chunks']
            except:
                pass
        
        return None
    
    def _clean_old_cache(self, cache_type: str) -> None:
        """
        Clean old cache entries
        
        Args:
            cache_type: Type of cache to clean
        """
        cache_dir = self._get_cache_path(cache_type, "").parent
        
        if not cache_dir.exists():
            return
        
        # Get all cache files
        cache_files = list(cache_dir.glob("*"))
        
        # If we have more than max_cache_size, remove oldest
        if len(cache_files) > self.max_cache_size:
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            files_to_remove = cache_files[:-self.max_cache_size]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except:
                    pass
        
        # Remove expired cache entries
        for file_path in cache_dir.glob("*"):
            if not self._is_cache_valid(file_path):
                try:
                    file_path.unlink()
                except:
                    pass
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache
        
        Args:
            cache_type: Specific cache type to clear, or None for all
        """
        if cache_type is None:
            # Clear all caches
            dirs_to_clear = [self.embedding_cache_dir, self.query_cache_dir, self.chunk_cache_dir]
        else:
            dirs_to_clear = [self._get_cache_path(cache_type, "").parent]
        
        for cache_dir in dirs_to_clear:
            if cache_dir.exists():
                for file_path in cache_dir.glob("*"):
                    try:
                        file_path.unlink()
                    except:
                        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {}
        
        for cache_type in ["embeddings", "queries", "chunks"]:
            cache_dir = self._get_cache_path(cache_type, "").parent
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*"))
                valid_files = [f for f in cache_files if self._is_cache_valid(f)]
                
                stats[cache_type] = {
                    'total_entries': len(cache_files),
                    'valid_entries': len(valid_files),
                    'size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
                }
        
        return stats
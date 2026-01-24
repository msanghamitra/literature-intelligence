# data/vector_store.py
# Vector store for semantic search and RAG retrieval.

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import json
from datetime import datetime
import pickle

# Import from embeddings module - FIXED PATH
try:
    # Try relative import first (since vector_store.py is in data/)
    from ..embeddings.embedder import (
        embed_text, embed_batch, retrieve_top_k, 
        load_embeddings, load_embedding_index, load_embedding_metadata,
        validate_embeddings, embeddings_exist, get_corpus_stats,
        run_full_corpus_embedding, choose_text_column
    )
except ImportError:
    # Fallback to direct import
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from embeddings.embedder import (
        embed_text, embed_batch, retrieve_top_k, 
        load_embeddings, load_embedding_index, load_embedding_metadata,
        validate_embeddings, embeddings_exist, get_corpus_stats,
        run_full_corpus_embedding, choose_text_column
    )

# Import model registry and pooling strategies
try:
    from ..embeddings.model_registry import (
        get_model_info, get_model_dimension, list_available_models,
        DEFAULT_MODEL
    )
    from ..embeddings.pooling_strategies import PoolingStrategy, apply_pooling_strategy
except ImportError:
    # Fallback
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from embeddings.model_registry import (
        get_model_info, get_model_dimension, list_available_models,
        DEFAULT_MODEL
    )
    from embeddings.pooling_strategies import PoolingStrategy, apply_pooling_strategy

# Optional FAISS for faster similarity search (if available)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Using numpy for similarity search.")

# Optional scikit-learn for alternative indexing
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class VectorStore:
    """
    Vector store for semantic search and RAG retrieval.
    Supports multiple backends (numpy, FAISS, scikit-learn).
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
        index_backend: str = "numpy",  # "numpy", "faiss", "sklearn"
        device: str = "cpu",
        normalize_embeddings: bool = True,
        index_path: Optional[Path] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the embedding model
            pooling_strategy: Pooling strategy for embeddings
            index_backend: Backend for similarity search
            device: Device for embedding computation
            normalize_embeddings: Whether to normalize embeddings
            index_path: Path to saved index (if loading from disk)
        """
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.index_backend = index_backend
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        
        # Model info
        self.model_info = get_model_info(model_name)
        self.embedding_dim = self.model_info["dimension"]
        
        # State
        self.embeddings = None
        self.index_df = None
        self.metadata = None
        self.index = None  # FAISS/sklearn index if using those backends
        self.is_trained = False
        
        # Load from disk if index_path is provided
        if index_path:
            self.load_from_disk(index_path)
    
    def build_index(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64,
        show_progress: bool = True
    ) -> "VectorStore":
        """
        Build the vector index from texts.
        
        Args:
            texts: List of text documents to index
            metadata_list: Optional metadata for each document
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
            
        Returns:
            Self for method chaining
        """
        if not texts:
            raise ValueError("No texts provided for indexing")
        
        print(f"Building index with {len(texts)} documents...")
        print(f"Model: {self.model_name}")
        print(f"Pooling: {self.pooling_strategy.value}")
        print(f"Backend: {self.index_backend}")
        
        # Generate embeddings
        self.embeddings = embed_batch(
            texts,
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            batch_size=batch_size,
            normalize=self.normalize_embeddings,
            show_progress_bar=show_progress,
            device=self.device
        )
        
        # Create metadata dataframe
        if metadata_list:
            if len(metadata_list) != len(texts):
                raise ValueError(f"Metadata list length ({len(metadata_list)}) must match texts length ({len(texts)})")
            self.index_df = pd.DataFrame(metadata_list)
        else:
            # Create simple metadata
            self.index_df = pd.DataFrame({
                "id": list(range(len(texts))),
                "text_preview": [t[:100] + "..." if len(t) > 100 else t for t in texts]
            })
        
        self.index_df["embedding_index"] = list(range(len(texts)))
        
        # Build search index
        self._build_search_index()
        
        # Update metadata
        self.metadata = {
            "model": self.model_name,
            "pooling_strategy": self.pooling_strategy.value,
            "index_backend": self.index_backend,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(texts),
            "created_at": datetime.now().isoformat(),
            "normalized": self.normalize_embeddings,
            "device": self.device,
        }
        
        self.is_trained = True
        print(f"Index built with {len(texts)} documents")
        
        return self
    
    def _build_search_index(self):
        """Build the search index based on the selected backend."""
        if self.index_backend == "faiss" and FAISS_AVAILABLE:
            self._build_faiss_index()
        elif self.index_backend == "sklearn" and SKLEARN_AVAILABLE:
            self._build_sklearn_index()
        elif self.index_backend == "numpy":
            # numpy backend doesn't need a separate index
            self.index = None
        else:
            print(f"Backend '{self.index_backend}' not available. Falling back to numpy.")
            self.index_backend = "numpy"
            self.index = None
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")
        
        n, d = self.embeddings.shape
        
        # Create and train index
        if n < 1000:
            # Small dataset - use exact search
            self.index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity (vectors normalized)
        else:
            # Larger dataset - use approximate search for speed
            nlist = min(100, int(np.sqrt(n)))  # Number of Voronoi cells
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings.astype(np.float32))
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
    
    def _build_sklearn_index(self):
        """Build scikit-learn NearestNeighbors index."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available.")
        
        # For cosine similarity, use algorithm='brute' with cosine metric
        # Since embeddings are normalized, cosine = dot product
        self.index = NearestNeighbors(
            n_neighbors=min(50, len(self.embeddings)),
            algorithm='brute',
            metric='cosine'
        )
        self.index.fit(self.embeddings)
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: Optional[float] = None,
        return_embeddings: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            return_embeddings: Whether to return query embedding
            
        Returns:
            DataFrame of results and array of similarity scores
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Generate query embedding
        query_embedding = embed_text(
            query,
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            normalize=self.normalize_embeddings,
            device=self.device
        )
        
        if np.all(query_embedding == 0):
            print("Warning: Query embedding is zero vector")
            return pd.DataFrame(), np.array([])
        
        # Search based on backend
        if self.index_backend == "faiss" and self.index is not None:
            indices, scores = self._search_faiss(query_embedding, k)
        elif self.index_backend == "sklearn" and self.index is not None:
            indices, scores = self._search_sklearn(query_embedding, k)
        else:
            indices, scores = self._search_numpy(query_embedding, k)
        
        # Apply threshold if specified
        if threshold is not None:
            mask = scores >= threshold
            indices = indices[mask]
            scores = scores[mask]
        
        # Get results
        results = self.index_df.iloc[indices].copy() if len(indices) > 0 else pd.DataFrame()
        results["similarity"] = scores
        
        # Sort by similarity
        if not results.empty:
            results = results.sort_values("similarity", ascending=False)
            results = results.reset_index(drop=True)
        
        if return_embeddings:
            return results, scores, query_embedding
        else:
            return results, scores
    
    def _search_numpy(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using numpy dot product."""
        # Cosine similarity with normalized vectors = dot product
        similarities = self.embeddings @ query_embedding
        
        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(-similarities)[:k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        return indices[0], scores[0]
    
    def _search_sklearn(self, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using scikit-learn."""
        # Reshape for sklearn
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search (sklearn returns distances, not similarities)
        k = min(k, len(self.embeddings))
        distances, indices = self.index.kneighbors(query_embedding, n_neighbors=k)
        
        # Convert cosine distance to cosine similarity
        # cosine_distance = 1 - cosine_similarity
        scores = 1 - distances[0]
        
        return indices[0], scores
    
    def add_documents(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 64
    ) -> "VectorStore":
        """
        Add documents to existing index.
        
        Args:
            texts: List of text documents to add
            metadata_list: Optional metadata for each document
            batch_size: Batch size for embedding
            
        Returns:
            Self for method chaining
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if not texts:
            return self
        
        print(f"Adding {len(texts)} documents to index...")
        
        # Generate embeddings for new documents
        new_embeddings = embed_batch(
            texts,
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            batch_size=batch_size,
            normalize=self.normalize_embeddings,
            show_progress_bar=False,
            device=self.device
        )
        
        # Create metadata for new documents
        start_idx = len(self.embeddings)
        if metadata_list:
            if len(metadata_list) != len(texts):
                raise ValueError(f"Metadata list length ({len(metadata_list)}) must match texts length ({len(texts)})")
            new_metadata_df = pd.DataFrame(metadata_list)
        else:
            new_metadata_df = pd.DataFrame({
                "id": list(range(start_idx, start_idx + len(texts))),
                "text_preview": [t[:100] + "..." if len(t) > 100 else t for t in texts]
            })
        
        new_metadata_df["embedding_index"] = list(range(start_idx, start_idx + len(texts)))
        
        # Append to existing data
        self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.index_df = pd.concat([self.index_df, new_metadata_df], ignore_index=True)
        
        # Rebuild index
        self._rebuild_index()
        
        # Update metadata
        self.metadata["num_documents"] = len(self.embeddings)
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        print(f"Added {len(texts)} documents. Total: {len(self.embeddings)}")
        
        return self
    
    def _rebuild_index(self):
        """Rebuild the search index after adding documents."""
        if self.index_backend == "faiss" and self.index is not None:
            # Clear and rebuild FAISS index
            self.index.reset()
            self.index.add(self.embeddings.astype(np.float32))
        elif self.index_backend == "sklearn" and self.index is not None:
            # Rebuild sklearn index
            self.index.fit(self.embeddings)
        # numpy backend doesn't need rebuilding
    
    def remove_documents(self, indices: List[int]) -> "VectorStore":
        """
        Remove documents from index.
        
        Args:
            indices: List of indices to remove
            
        Returns:
            Self for method chaining
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if not indices:
            return self
        
        # Filter out indices
        mask = np.ones(len(self.embeddings), dtype=bool)
        mask[indices] = False
        
        # Apply mask
        self.embeddings = self.embeddings[mask]
        self.index_df = self.index_df.iloc[mask].reset_index(drop=True)
        
        # Update embedding indices
        self.index_df["embedding_index"] = list(range(len(self.embeddings)))
        
        # Rebuild index
        self._rebuild_index()
        
        # Update metadata
        self.metadata["num_documents"] = len(self.embeddings)
        self.metadata["updated_at"] = datetime.now().isoformat()
        
        print(f"Removed {len(indices)} documents. Remaining: {len(self.embeddings)}")
        
        return self
    
    def save_to_disk(self, save_dir: Path):
        """
        Save vector store to disk.
        
        Args:
            save_dir: Directory to save to
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained vector store")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(save_dir / "embeddings.npy", self.embeddings)
        
        # Save index dataframe
        self.index_df.to_csv(save_dir / "index_df.csv", index=False)
        
        # Save metadata
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save FAISS/sklearn index if applicable
        if self.index_backend == "faiss" and self.index is not None:
            faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))
        elif self.index_backend == "sklearn" and self.index is not None:
            with open(save_dir / "sklearn_index.pkl", "wb") as f:
                pickle.dump(self.index, f)
        
        print(f"Vector store saved to {save_dir}")
    
    def load_from_disk(self, load_dir: Path) -> "VectorStore":
        """
        Load vector store from disk.
        
        Args:
            load_dir: Directory to load from
            
        Returns:
            Self for method chaining
        """
        # Check if files exist
        required_files = ["embeddings.npy", "index_df.csv", "metadata.json"]
        for file in required_files:
            if not (load_dir / file).exists():
                raise FileNotFoundError(f"Required file not found: {load_dir / file}")
        
        # Load metadata
        with open(load_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Update instance attributes from metadata
        self.model_name = self.metadata.get("model", DEFAULT_MODEL)
        pooling_str = self.metadata.get("pooling_strategy", "mean")
        self.pooling_strategy = PoolingStrategy(pooling_str)
        self.index_backend = self.metadata.get("index_backend", "numpy")
        self.normalize_embeddings = self.metadata.get("normalized", True)
        self.device = self.metadata.get("device", "cpu")
        
        # Update model info
        self.model_info = get_model_info(self.model_name)
        self.embedding_dim = self.model_info["dimension"]
        
        # Load embeddings
        self.embeddings = np.load(load_dir / "embeddings.npy")
        
        # Load index dataframe
        self.index_df = pd.read_csv(load_dir / "index_df.csv")
        
        # Load index if applicable
        if self.index_backend == "faiss" and (load_dir / "faiss_index.bin").exists():
            self.index = faiss.read_index(str(load_dir / "faiss_index.bin"))
        elif self.index_backend == "sklearn" and (load_dir / "sklearn_index.pkl").exists():
            with open(load_dir / "sklearn_index.pkl", "rb") as f:
                self.index = pickle.load(f)
        else:
            self.index = None
        
        self.is_trained = True
        
        print(f"Vector store loaded from {load_dir}")
        print(f"  Documents: {len(self.embeddings)}")
        print(f"  Model: {self.model_name}")
        print(f"  Backend: {self.index_backend}")
        
        return self
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        stats = self.metadata.copy()
        
        # Add additional stats
        if self.embeddings is not None and len(self.embeddings) > 0:
            stats["embedding_mean_norm"] = float(np.mean(np.linalg.norm(self.embeddings, axis=1)))
            stats["embedding_std_norm"] = float(np.std(np.linalg.norm(self.embeddings, axis=1)))
            stats["embedding_min"] = float(np.min(self.embeddings))
            stats["embedding_max"] = float(np.max(self.embeddings))
        
        return stats
    
    def similarity(
        self,
        text1: str,
        text2: str,
        return_embeddings: bool = False
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            return_embeddings: Whether to return embeddings
            
        Returns:
            Similarity score (and optionally embeddings)
        """
        emb1 = embed_text(
            text1,
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            normalize=self.normalize_embeddings,
            device=self.device
        )
        
        emb2 = embed_text(
            text2,
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            normalize=self.normalize_embeddings,
            device=self.device
        )
        
        similarity = float(emb1 @ emb2)  # Dot product for normalized vectors
        
        if return_embeddings:
            return similarity, emb1, emb2
        else:
            return similarity


class ArxivVectorStore(VectorStore):
    """
    Specialized vector store for arXiv papers.
    Integrates with the embeddings module's corpus loading.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
        index_backend: str = "numpy",
        device: str = "cpu",
        use_summaries: bool = True
    ):
        """
        Initialize arXiv vector store.
        
        Args:
            model_name: Name of the embedding model
            pooling_strategy: Pooling strategy for embeddings
            index_backend: Backend for similarity search
            device: Device for embedding computation
            use_summaries: Whether to use paper summaries (if available)
        """
        super().__init__(model_name, pooling_strategy, index_backend, device)
        self.use_summaries = use_summaries
        
        # Try to load precomputed embeddings
        if embeddings_exist():
            self._load_precomputed()
    
    def _load_precomputed(self):
        """Load precomputed embeddings from the embeddings module."""
        try:
            # Validate embeddings
            if not validate_embeddings():
                print("Warning: Precomputed embeddings validation failed")
                return
            
            # Load data from embeddings module
            self.embeddings = load_embeddings()
            self.index_df = load_embedding_index()
            self.metadata = load_embedding_metadata()
            
            # Update instance attributes from metadata
            self.model_name = self.metadata.get("model", DEFAULT_MODEL)
            pooling_str = self.metadata.get("pooling_strategy", "mean")
            self.pooling_strategy = PoolingStrategy(pooling_str)
            self.normalize_embeddings = True  # Embeddings are normalized
            
            # Update model info
            self.model_info = get_model_info(self.model_name)
            self.embedding_dim = self.model_info["dimension"]
            
            # Build search index
            self._build_search_index()
            
            self.is_trained = True
            
            print(f"Loaded precomputed embeddings: {len(self.embeddings)} papers")
            print(f"  Model: {self.model_name}")
            print(f"  Pooling: {self.pooling_strategy.value}")
            
        except Exception as e:
            print(f"Error loading precomputed embeddings: {e}")
            self.is_trained = False
    
    def build_from_corpus(
        self,
        batch_size: int = 64,
        force_rebuild: bool = False
    ) -> "ArxivVectorStore":
        """
        Build vector store from arXiv corpus.
        
        Args:
            batch_size: Batch size for embedding
            force_rebuild: Force rebuild even if precomputed exists
            
        Returns:
            Self for method chaining
        """
        # Use precomputed if available and not forcing rebuild
        if self.is_trained and not force_rebuild:
            print("Using precomputed embeddings")
            return self
        
        # Run full corpus embedding
        print("Building embeddings from arXiv corpus...")
        df_use, embeddings = run_full_corpus_embedding(
            model_name=self.model_name,
            pooling_strategy=self.pooling_strategy,
            batch_size=batch_size,
            device=self.device
        )
        
        # Prepare metadata
        metadata_list = []
        for _, row in df_use.iterrows():
            metadata = {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "published": row.get("published", ""),
                "updated": row.get("updated", ""),
                "pdf_url": row.get("pdf_url", ""),
                "text_preview": row.get("text_unit", "")[:200] + "..." if len(row.get("text_unit", "")) > 200 else row.get("text_unit", ""),
            }
            if "summary" in row:
                metadata["summary"] = row["summary"]
            metadata_list.append(metadata)
        
        # Build index
        self.build_index(
            texts=df_use[choose_text_column(df_use)].tolist(),
            metadata_list=metadata_list,
            batch_size=batch_size,
            show_progress=True
        )
        
        print(f"Built arXiv vector store with {len(self.embeddings)} papers")
        
        return self
    
    def search_papers(
        self,
        query: str,
        k: int = 10,
        year_filter: Optional[int] = None,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Search for arXiv papers.
        
        Args:
            query: Search query
            k: Number of results
            year_filter: Filter by year (e.g., 2023)
            threshold: Minimum similarity threshold
            
        Returns:
            DataFrame of search results
        """
        # Search for similar papers
        results, scores = self.search(query, k=k, threshold=threshold)
        
        if results.empty:
            return results
        
        # Apply year filter if specified
        if year_filter is not None and "published" in results.columns:
            # Extract year from published date
            results["year"] = pd.to_datetime(results["published"]).dt.year
            results = results[results["year"] == year_filter]
        
        return results


# Factory function for easy creation
def create_vector_store(
    store_type: str = "arxiv",  # "arxiv" or "generic"
    **kwargs
) -> Union[ArxivVectorStore, VectorStore]:
    """
    Factory function to create vector stores.
    
    Args:
        store_type: Type of vector store ("arxiv" or "generic")
        **kwargs: Arguments passed to vector store constructor
        
    Returns:
        Vector store instance
    """
    if store_type.lower() == "arxiv":
        return ArxivVectorStore(**kwargs)
    else:
        return VectorStore(**kwargs)


if __name__ == "__main__":
    # Test the vector store
    print("Testing VectorStore...")
    
    # Create arXiv vector store
    vs = ArxivVectorStore(index_backend="numpy")
    
    # Check if precomputed embeddings exist
    if vs.is_trained:
        print(f"Loaded {len(vs.embeddings)} precomputed embeddings")
        
        # Test search
        results, scores = vs.search_papers("machine learning", k=3)
        print(f"\nSearch results for 'machine learning':")
        if not results.empty:
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"  {i+1}. {row.get('title', 'No title')} (similarity: {row.get('similarity', 0):.4f})")
        
        # Get stats
        stats = vs.get_stats()
        print(f"\nVector store stats:")
        for key, value in stats.items():
            if key not in ["created_at", "updated_at"]:
                print(f"  {key}: {value}")
    else:
        print("No precomputed embeddings found.")
        print("To build embeddings, run: python -m embeddings.embedder")
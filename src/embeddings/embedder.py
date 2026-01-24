# src/embeddings/embedder.py
# Core embedding functionality for semantic search + RAG retrieval.

from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any, Union
import json
import hashlib
import numpy as np
import pandas as pd

from .model_registry import get_model, get_model_dimension, DEFAULT_MODEL
from .pooling_strategies import mean_pooling, cls_pooling, PoolingStrategy


# Base dir = project root (two levels up: src/embeddings -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

CORPUS_CLEAN_PATH = DATA_DIR / "corpus_clean.csv"
CORPUS_SUMMARY_PATH = DATA_DIR / "corpus_with_summaries.csv"

# Where embeddings are stored
EMB_DIR = DATA_DIR / "embeddings"
EMB_PATH = EMB_DIR / "paper_embeddings.npy"
EMB_INDEX_PATH = EMB_DIR / "paper_embeddings_index.csv"
EMB_METADATA_PATH = EMB_DIR / "embeddings_metadata.json"


# 1) Cached embedder
@lru_cache(maxsize=1)
def get_embedder(
    model_name: str = DEFAULT_MODEL,
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    device: str = "cpu"
) -> Any:
    """
    Loads the embedding model once per Python process.
    Streamlit will call retrieval many times; caching prevents repeated model load.
    """
    model = get_model(model_name)
    model.to(device)
    return model


# 2) Embed text (single) and batch
def embed_text(
    text: Optional[str],
    model_name: str = DEFAULT_MODEL,
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    normalize: bool = True,
    device: str = "cpu"
) -> np.ndarray:
    """
    Embed a single text string.
    Returns a vector (float32) suitable for similarity search.
    """
    if not text or not text.strip():
        # Return zero vector matching expected dimension for consistency
        dim = get_model_dimension(model_name)
        return np.zeros(dim, dtype=np.float32)

    model = get_embedder(model_name, pooling_strategy, device)
    try:
        # Tokenize and get embeddings
        inputs = model.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply pooling strategy
        if pooling_strategy == PoolingStrategy.MEAN:
            embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        elif pooling_strategy == PoolingStrategy.CLS:
            embedding = cls_pooling(outputs.last_hidden_state)
        else:
            embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        
        embedding = embedding.cpu().numpy().flatten()
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Error embedding text: {e}")
        # Return zero vector on error for consistency
        dim = get_model_dimension(model_name)
        return np.zeros(dim, dtype=np.float32)


def embed_batch(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = False,
    device: str = "cpu"
) -> np.ndarray:
    """
    Embed a batch of texts.
    Returns vectors (float32).
    """
    if not texts:
        return np.array([], dtype=np.float32)
    
    model = get_embedder(model_name, pooling_strategy, device)
    embeddings = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = model.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Apply pooling strategy
            if pooling_strategy == PoolingStrategy.MEAN:
                batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            elif pooling_strategy == PoolingStrategy.CLS:
                batch_embeddings = cls_pooling(outputs.last_hidden_state)
            else:
                batch_embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize if requested
            if normalize:
                norms = torch.norm(batch_embeddings, dim=1, keepdim=True)
                batch_embeddings = batch_embeddings / norms
            
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if show_progress_bar:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        if embeddings:
            return np.vstack(embeddings).astype(np.float32)
        else:
            return np.array([], dtype=np.float32)
            
    except Exception as e:
        print(f"Error in embed_batch: {e}")
        return np.array([], dtype=np.float32)


# 3) Load corpus (same "best available" logic as summarizer.py)
def load_clean_corpus() -> pd.DataFrame:
    """Load the cleaned corpus without summaries."""
    return pd.read_csv(CORPUS_CLEAN_PATH)


def load_summary_corpus() -> pd.DataFrame:
    """Load corpus with summaries if available, otherwise fall back to clean corpus."""
    if CORPUS_SUMMARY_PATH.exists():
        return pd.read_csv(CORPUS_SUMMARY_PATH)
    return load_clean_corpus()


# 4) Choose what field to embed
def choose_text_column(df: pd.DataFrame) -> str:
    """
    Prefer embedding the summary (fast, consistent) if it exists,
    otherwise fall back to text_unit.
    """
    if "summary" in df.columns and df["summary"].notna().any():
        return "summary"
    return "text_unit"


def get_texts_hash(texts: List[str]) -> str:
    """Generate hash of texts to detect changes."""
    text_string = "||".join(texts)
    return hashlib.md5(text_string.encode()).hexdigest()


# 5) Build + save embeddings for the full corpus
def run_full_corpus_embedding(
    model_name: str = DEFAULT_MODEL,
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generates embeddings for each paper (one vector per row),
    saves .npy (vectors) + .csv (row alignment/index), returns df + embeddings.
    """
    df = load_summary_corpus()

    text_col = choose_text_column(df)
    texts = df[text_col].astype(str).fillna("").tolist()

    # Keep track of which rows are actually embeddable (non-empty text)
    mask = pd.Series(texts).str.strip().ne("")
    df_use = df.loc[mask].reset_index(drop=True)
    texts_use = [t for t, ok in zip(texts, mask.tolist()) if ok]

    if not texts_use:
        print("No valid texts to embed")
        # Save empty structures for consistency
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMB_PATH, np.array([], dtype=np.float32))
        
        # Create minimal index
        index_df = pd.DataFrame(columns=["id", "row_idx", "embedded_text_col", "embedding_model", "pooling_strategy"])
        index_df.to_csv(EMB_INDEX_PATH, index=False)
        
        # Save metadata
        metadata = {
            "model": model_name,
            "pooling_strategy": pooling_strategy.value,
            "texts_hash": get_texts_hash(texts_use),
            "timestamp": pd.Timestamp.now().isoformat(),
            "num_embeddings": 0,
            "dimension": get_model_dimension(model_name),
            "text_column": text_col,
            "total_rows": len(df),
            "embedded_rows": 0,
            "device": device,
        }
        with open(EMB_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
            
        return df_use, np.array([], dtype=np.float32)

    # Process in batches for memory efficiency with large corpora
    print(f"Embedding {len(texts_use)} texts in batches of {batch_size}...")
    embeddings = embed_batch(
        texts_use,
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        batch_size=batch_size,
        normalize=True,
        show_progress_bar=True,
        device=device
    )

    # Create directory if it doesn't exist
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(EMB_PATH, embeddings)

    # Save a small "alignment file" so you can map embeddings[i] -> df_use row i
    keep_cols = [c for c in ["id", "title", "published", "updated", "pdf_url"] if c in df_use.columns]
    index_df = df_use[keep_cols].copy() if keep_cols else df_use[[text_col]].copy()
    index_df["row_idx"] = np.arange(len(df_use), dtype=int)
    index_df["embedded_text_col"] = text_col
    index_df["embedding_model"] = model_name
    index_df["pooling_strategy"] = pooling_strategy.value
    index_df.to_csv(EMB_INDEX_PATH, index=False)

    # Save metadata for validation and versioning
    metadata = {
        "model": model_name,
        "pooling_strategy": pooling_strategy.value,
        "texts_hash": get_texts_hash(texts_use),
        "timestamp": pd.Timestamp.now().isoformat(),
        "num_embeddings": len(embeddings),
        "dimension": embeddings.shape[1] if len(embeddings) > 0 else get_model_dimension(model_name),
        "text_column": text_col,
        "total_rows": len(df),
        "embedded_rows": len(df_use),
        "batch_size": batch_size,
        "device": device,
    }
    with open(EMB_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings to: {EMB_PATH}")
    print(f"Saved embeddings index to: {EMB_INDEX_PATH}")
    print(f"Saved embeddings metadata to: {EMB_METADATA_PATH}")
    print(f"Embedded column: {text_col} | Rows embedded: {len(df_use)}")
    print(f"Embedding dimension: {metadata['dimension']}")
    print(f"Pooling strategy: {pooling_strategy.value}")

    return df_use, embeddings


# 6) Load embeddings from disk
def load_embeddings() -> np.ndarray:
    """Load embeddings from disk with validation."""
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMB_PATH}. Run run_full_corpus_embedding() first.")
    
    embeddings = np.load(EMB_PATH)
    
    # Basic validation
    if np.isnan(embeddings).any():
        print("Warning: Embeddings contain NaN values")
    
    return embeddings


def load_embedding_index() -> pd.DataFrame:
    """Load embedding index from disk."""
    if not EMB_INDEX_PATH.exists():
        raise FileNotFoundError(f"Embeddings index not found at {EMB_INDEX_PATH}. Run run_full_corpus_embedding() first.")
    return pd.read_csv(EMB_INDEX_PATH)


def load_embedding_metadata() -> Dict[str, Any]:
    """Load embeddings metadata."""
    if not EMB_METADATA_PATH.exists():
        raise FileNotFoundError(f"Embeddings metadata not found at {EMB_METADATA_PATH}.")
    
    with open(EMB_METADATA_PATH, "r") as f:
        return json.load(f)


# 7) Validation functions
def validate_embeddings() -> bool:
    """
    Validate that embeddings, index, and metadata are aligned and consistent.
    Returns True if valid, False otherwise.
    """
    # Check files exist
    if not EMB_PATH.exists() or not EMB_INDEX_PATH.exists() or not EMB_METADATA_PATH.exists():
        print("One or more embedding files are missing")
        return False
    
    try:
        # Load data
        embeddings = load_embeddings()
        index_df = load_embedding_index()
        metadata = load_embedding_metadata()
        
        # Check lengths match
        if len(embeddings) != len(index_df):
            print(f"Mismatch: embeddings have {len(embeddings)} rows, index has {len(index_df)} rows")
            return False
        
        # Check metadata consistency
        if metadata.get("num_embeddings", 0) != len(embeddings):
            print(f"Mismatch: metadata says {metadata.get('num_embeddings')} embeddings, but loaded {len(embeddings)}")
            return False
        
        # Check embedding dimension
        if len(embeddings) > 0:
            expected_dim = metadata.get("dimension", get_model_dimension(metadata.get("model", DEFAULT_MODEL)))
            if embeddings.shape[1] != expected_dim:
                print(f"Mismatch: embeddings have dimension {embeddings.shape[1]}, expected {expected_dim}")
                return False
        
        # Check for NaN in embeddings
        if len(embeddings) > 0 and np.isnan(embeddings).any():
            print("Warning: Embeddings contain NaN values")
            # Return True with warning for now, but you might want to return False
        
        print("Embeddings validation passed")
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def embeddings_exist() -> bool:
    """Check if embeddings files exist."""
    return EMB_PATH.exists() and EMB_INDEX_PATH.exists() and EMB_METADATA_PATH.exists()


# 8) Semantic retrieval (top-k papers for a query) over the LOCAL CORPUS embeddings
def retrieve_top_k(
    query: str,
    k: int = 5,
    model_name: str = DEFAULT_MODEL,
    pooling_strategy: PoolingStrategy = PoolingStrategy.MEAN,
    device: str = "cpu"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns top-k most similar papers to the query (semantic search) over the precomputed local corpus.
    Assumes embeddings are normalized => similarity = dot product.
    Returns DataFrame of results and array of similarity scores.
    """
    if not query or not query.strip():
        return pd.DataFrame(), np.array([], dtype=np.float32)
    
    try:
        emb = load_embeddings()
        idx = load_embedding_index()
        
        if len(emb) == 0 or len(idx) == 0:
            print("No embeddings available")
            return pd.DataFrame(), np.array([], dtype=np.float32)
        
        q_vec = embed_text(query, model_name=model_name, pooling_strategy=pooling_strategy, device=device)
        
        # Check for zero vector (empty query or embedding error)
        if np.all(q_vec == 0):
            print("Query embedding resulted in zero vector")
            return pd.DataFrame(), np.array([], dtype=np.float32)
        
        # Ensure shape compatibility
        if q_vec.shape[0] != emb.shape[1]:
            print(f"Shape mismatch: query vector dim {q_vec.shape[0]}, embedding dim {emb.shape[1]}")
            return pd.DataFrame(), np.array([], dtype=np.float32)
        
        # Cosine similarity with normalized vectors = dot product
        sims = emb @ q_vec
        
        # Handle case where k > available embeddings
        k = min(k, len(sims))
        top_idx = np.argsort(-sims)[:k]
        
        results = idx.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        
        # Sort by similarity (descending) and return
        return results.sort_values("similarity", ascending=False).reset_index(drop=True), sims[top_idx]
        
    except FileNotFoundError as e:
        print(f"Embeddings not found: {e}")
        return pd.DataFrame(), np.array([], dtype=np.float32)
    except Exception as e:
        print(f"Error in retrieve_top_k: {e}")
        return pd.DataFrame(), np.array([], dtype=np.float32)


def get_corpus_stats() -> Dict[str, Any]:
    """
    Get statistics about the embedded corpus.
    """
    stats = {}
    
    if embeddings_exist():
        try:
            metadata = load_embedding_metadata()
            stats.update(metadata)
            
            # Add additional calculated stats
            emb = load_embeddings()
            if len(emb) > 0:
                stats["embedding_norm_mean"] = float(np.mean(np.linalg.norm(emb, axis=1)))
                stats["embedding_norm_std"] = float(np.std(np.linalg.norm(emb, axis=1)))
                stats["embedding_min"] = float(np.min(emb))
                stats["embedding_max"] = float(np.max(emb))
                stats["embedding_mean"] = float(np.mean(emb))
                stats["embedding_std"] = float(np.std(emb))
        except Exception as e:
            stats["error"] = str(e)
    
    return stats


# Add torch import at the top of the main execution block
if __name__ == "__main__":
    import torch
    
    # Run full corpus embedding when executed directly
    print("Starting full corpus embedding...")
    df, embeddings = run_full_corpus_embedding()
    
    # Validate the embeddings
    is_valid = validate_embeddings()
    print(f"Embeddings validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Print statistics
    stats = get_corpus_stats()
    print("\nEmbedding Statistics:")
    for key, value in stats.items():
        if key not in ["texts_hash", "timestamp"]:  # Skip hash and timestamp for brevity
            print(f"  {key}: {value}")
    
    # Test retrieval
    if is_valid and len(embeddings) > 0:
        print("\nTesting retrieval with sample query 'machine learning':")
        results, scores = retrieve_top_k("machine learning", k=3)
        if not results.empty:
            print(f"Found {len(results)} results:")
            for i, (_, row) in enumerate(results.iterrows()):
                print(f"  {i+1}. {row.get('title', 'No title')} (similarity: {row['similarity']:.4f})")
# src/models/embeddings.py
# Build and reuse vector embeddings for each paper so we can do semantic search + RAG retrieval.
# Embeddings = numeric vectors that capture meaning (so "heart attack" ~ "myocardial infarction").

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Base dir = project root (two levels up: src/models -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

CORPUS_CLEAN_PATH = DATA_DIR / "corpus_clean.csv"
CORPUS_SUMMARY_PATH = DATA_DIR / "corpus_with_summaries.csv"

# Where embeddings are stored
EMB_DIR = DATA_DIR / "embeddings"
EMB_PATH = EMB_DIR / "paper_embeddings.npy"
EMB_INDEX_PATH = EMB_DIR / "paper_embeddings_index.csv"

# Default embedding model (fast + good enough for MVP)
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# 1) Cached embedder (same idea as your cached summarizer)
@lru_cache(maxsize=1)
def get_embedder(model_name: str = DEFAULT_EMB_MODEL) -> SentenceTransformer:
    """
    Loads the SentenceTransformer once per Python process.
    Streamlit will call retrieval many times; caching prevents repeated model load.
    """
    return SentenceTransformer(model_name, device="cpu")


# 2) Embed text (single) and batch
def embed_text(text: str, model_name: str = DEFAULT_EMB_MODEL) -> np.ndarray:
    if not text or not text.strip():
        # return a zero vector only if needed, but better to avoid embedding empty text
        return np.array([], dtype=np.float32)

    model = get_embedder(model_name)
    vec = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,  # makes cosine similarity = dot product
        show_progress_bar=False,
    )
    return vec.astype(np.float32)


def embed_batch(texts: list[str], model_name: str = DEFAULT_EMB_MODEL, batch_size: int = 64) -> np.ndarray:
    model = get_embedder(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.astype(np.float32)


# 3) Load corpus (same "best available" logic as summarizer.py)
def load_clean_corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS_CLEAN_PATH)


def load_summary_corpus() -> pd.DataFrame:
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


# 5) Build + save embeddings for the full corpus (like run_full_corpus_summarisation)
def run_full_corpus_embedding(model_name: str = DEFAULT_EMB_MODEL) -> tuple[pd.DataFrame, np.ndarray]:
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

    embeddings = embed_batch(texts_use, model_name=model_name)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, embeddings)

    # Save a small "alignment file" so you can map embeddings[i] -> df_use row i
    keep_cols = [c for c in ["id", "title", "published", "updated", "pdf_url"] if c in df_use.columns]
    index_df = df_use[keep_cols].copy() if keep_cols else df_use[[text_col]].copy()
    index_df["row_idx"] = np.arange(len(df_use), dtype=int)
    index_df["embedded_text_col"] = text_col
    index_df["embedding_model"] = model_name
    index_df.to_csv(EMB_INDEX_PATH, index=False)

    print(f"Saved embeddings to: {EMB_PATH}")
    print(f"Saved embeddings index to: {EMB_INDEX_PATH}")
    print(f"Embedded column: {text_col} | Rows embedded: {len(df_use)}")

    return df_use, embeddings


# 6) Load embeddings from disk
def load_embeddings() -> np.ndarray:
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Embeddings not found at {EMB_PATH}. Run run_full_corpus_embedding() first.")
    return np.load(EMB_PATH)


def load_embedding_index() -> pd.DataFrame:
    if not EMB_INDEX_PATH.exists():
        raise FileNotFoundError(f"Embeddings index not found at {EMB_INDEX_PATH}. Run run_full_corpus_embedding() first.")
    return pd.read_csv(EMB_INDEX_PATH)


# 7) Semantic retrieval (top-k papers for a query)
def retrieve_top_k(
    query: str,
    k: int = 5,
    model_name: str = DEFAULT_EMB_MODEL,
) -> pd.DataFrame:
    """
    Returns top-k most similar papers to the query (semantic search).
    Assumes embeddings are normalized => similarity = dot product.
    """
    if not query or not query.strip():
        return pd.DataFrame()

    emb = load_embeddings()
    idx = load_embedding_index()

    q_vec = embed_text(query, model_name=model_name)
    if q_vec.size == 0:
        return pd.DataFrame()

    # cosine similarity with normalized vectors = dot product
    sims = emb @ q_vec
    top_idx = np.argsort(-sims)[:k]

    results = idx.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results.sort_values("similarity", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    run_full_corpus_embedding()

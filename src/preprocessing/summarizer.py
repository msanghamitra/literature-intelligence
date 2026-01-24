# For each paper, take text_unit and generate a short, consistent, human-readable
# summary using a transformer model, and store it as `summary`.

from pathlib import Path
import pandas as pd
from transformers import pipeline
from functools import lru_cache

# Base dir = project root (three levels up: src/processing -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

CORPUS_CLEAN_PATH = DATA_DIR / "corpus_clean.csv"
CORPUS_SUMMARY_PATH = DATA_DIR / "corpus_with_summaries.csv"

# 1. Set up summarisation pipeline
@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # CPU
    )

# 2. Summarize text
def summarize_text(text: str,
                   max_length: int = 128,
                   min_length: int = 32) -> str:
    if not text or not text.strip():
        return ""
    
    summarizer = get_summarizer()
    out = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        truncation=True,
    )[0]["summary_text"]
    return out

# 3. Summarize Batch
def summarize_batch(texts: list[str],
                   max_length: int = 128,
                   min_length: int = 32) -> list[str]:
    summarizer = get_summarizer()
    results = summarizer(
        texts,
        max_length=max_length,
        min_length=min_length,
        truncation=True,
    )
    return [r["summary_text"] for r in results]

# 4. Loading data from corpus
def load_clean_corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS_CLEAN_PATH)

def load_summary_corpus() -> pd.DataFrame:
    if CORPUS_SUMMARY_PATH.exists():
        return pd.read_csv(CORPUS_SUMMARY_PATH)
    return load_clean_corpus()

# 5. Full Corpus Summarization
def run_full_corpus_summarisation() -> pd.DataFrame:
    df = load_clean_corpus()
    texts = df["text_unit"].astype(str).tolist()
    summaries = summarize_batch(texts)
    df["summary"] = summaries
    df.to_csv(CORPUS_SUMMARY_PATH, index=False)
    print(f"Saved summaries to {CORPUS_SUMMARY_PATH}")
    return df

if __name__ == "__main__":
    run_full_corpus_summarisation()
# For each paper, take text_unit and generate a short, consistent, human-readable
# summary using a transformer model, and store it as `summary`.

from pathlib import Path
import pandas as pd
from transformers import pipeline
from functools import lru_cache


# Base dir = project root (two levels up: src/models -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

CORPUS_CLEAN_PATH = DATA_DIR / "corpus_clean.csv"
CORPUS_SUMMARY_PATH = DATA_DIR / "corpus_with_summaries.csv"


# 1. Set up summarisation pipeline
#lru_cache(maxsize=1) means: the first time you call this, it loads the HF pipeline.
#Subsequent calls return the same pipeline object.
#This is crucial for Streamlit: the app can call summarize_text many times (different papers, different clicks) without re-loading the model each time → faster + less memory.
#one model instance lives as long as the app / Python process.

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Small summarisation model # Alternate: model="google/flan-t5-small"
        device=-1  # CPU
    )

# 2. Summerize text
#This is the function the Streamlit/Gradio app can call every time the user clicks “Summarise”.
#You can easily expose max_length / min_length as sliders in the UI.
#It uses get_summarizer() so it doesn’t reload the model.

def summarize_text(text: str,
                    max_length: int=128,
                    min_length: int=32) -> str:
    if not text or not text.strip():
        return ""
    
    summarizer = get_summarizer()
    out = summarizer(
        text,
        max_length = max_length,
        min_length = min_length,
        truncation = True,
    )[0]["summary_text"]
    return out


# 3.Summarize Batch
# Its a function you can from one of a script, a CLI, a background job or even from another app

def summarize_batch(texts: list[str],
                   max_length: int = 128,
                   min_length: int = 32) -> list[str]:
    summarizer = get_summarizer()
    results = summarizer(
        texts,
        max_length = max_length,
        min_length = min_length,
        truncation = True,
    )
    return [r["summary_text"]for r in results]


# 4.Loading data from corpus
#The app doesn’t need to know which CSV is there; it just asks for “the best available corpus”.
def load_clean_corpus()-> pd.DataFrame:
    return pd.read_csv(CORPUS_CLEAN_PATH)

def load_summary_corpus()-> pd.DataFrame:
    if CORPUS_SUMMARY_PATH.exists():
        return pd.read_csv(CORPUS_SUMMARY_PATH)
    return load_clean_corpus()


# 5.Full Corpus Summarization
# In the Streamlit sidebar we hook this to a button "Run full corpus summarisation". 
# So you can still regenerate corpus_with_summaries.csv directly from the UI if you want.
def run_full_corpus_summarisation() -> pd.DataFrame:
    df = load_clean_corpus()

    texts = df["text_unit"].astype(str).tolist()
    summaries = summarize_batch(texts)

    df["summary"] = summaries
    df.to_csv(CORPUS_SUMMARY_PATH, index=False)
    print(f"Saved summaries to the path{CORPUS_SUMMARY_PATH}")
    return df

if __name__ == "__main__":
    run_full_corpus_summarisation()
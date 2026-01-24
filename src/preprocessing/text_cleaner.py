import re
from pathlib import Path
import pandas as pd

# Updated paths for new structure
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"
RAW_METADATA_PATH = DATA_DIR / "metadata.csv"
CLEAN_CORPUS_PATH = DATA_DIR / "corpus_clean.csv"

def basic_clean(text: str) -> str:
    """Text normalisation for titles/abstracts."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_corpus() -> pd.DataFrame:
    """Load metadata, clean text, and build the main corpus."""
    # Load metadata produced by arxiv_loader.py
    df = pd.read_csv(RAW_METADATA_PATH)

    # Parse dates so we can filter later by time
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df["updated"] = pd.to_datetime(df["updated"], errors="coerce")

    # Clean title + abstract
    df['title_clean'] = df['title'].apply(basic_clean)
    df['abstract_clean'] = df['summary'].apply(basic_clean)

    # Main Text Unit = title + abstract
    df["text_unit"] = df["title_clean"] + ". " + df["abstract_clean"]

    # Columns needed downstream
    cols = [
        "arxiv_id",
        "title_clean",
        "abstract_clean",
        "text_unit",
        "authors",
        "published",
        "updated",
        "primary_category",
        "pdf_url",
    ]

    cols = [c for c in cols if c in df.columns]

    cleaned = df[cols]
    cleaned.to_csv(CLEAN_CORPUS_PATH, index=False)
    print(f"Saved clean corpus to {CLEAN_CORPUS_PATH} (n={len(cleaned)})")
    return cleaned

if __name__ == "__main__":
    build_corpus()
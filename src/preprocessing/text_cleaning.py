import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/arxiv_papers")
RAW_METADATA_PATH = DATA_DIR/ "metadata.csv"                          #Created by arvix_loader.py
CLEAN_CORPUS_PATH = DATA_DIR/ "corpus_clean.csv"                      #Would be created by text_cleaning.py

def basic_clean(text: str) -> str:
    """ Text normalisation for titles/abstract."""
    if not isinstance(text,str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"s\+", " ", text)
    return text.strip()

def build_corpus():
    #Load meta produced by arvix_loader.py
    df = pd.read_csv(RAW_METADATA_PATH)

    #Clean title + Summary
    df['title_clean'] = df['title'].apply(basic_clean)
    df['abstract_clean'] = df['summary'].apply(basic_clean)

    #Main Text Unit = title + abstract
    df["text_unit"] = df["title_clean"]+". "+ df["abstract_clean"]

    #Columns needed downstream
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

if __name__== "__main__":
    df_clean = build_corpus()
    print(df_clean.head(10))
# src/models/topics.py
# Lightweight topic modelling: TF-IDF + KMeans (fast, no GPU, good demo)
# Output:
# - topics.csv (topic_id -> top keywords + counts)
# - corpus_with_topics.csv (paper rows with assigned topic_id)

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Base dir = project root (two levels up: src/models -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

CORPUS_CLEAN_PATH = DATA_DIR / "corpus_clean.csv"
CORPUS_SUMMARY_PATH = DATA_DIR / "corpus_with_summaries.csv"

TOPICS_DIR = DATA_DIR / "topics"
TOPICS_PATH = TOPICS_DIR / "topics.csv"
CORPUS_WITH_TOPICS_PATH = TOPICS_DIR / "corpus_with_topics.csv"


def load_best_corpus() -> pd.DataFrame:
    if CORPUS_SUMMARY_PATH.exists():
        return pd.read_csv(CORPUS_SUMMARY_PATH)
    return pd.read_csv(CORPUS_CLEAN_PATH)


def choose_text_column(df: pd.DataFrame) -> str:
    # Prefer model summary if it exists, otherwise use text_unit
    if "summary" in df.columns and df["summary"].notna().any():
        return "summary"
    return "text_unit"


def build_topics(
    n_topics: int = 8,
    top_terms: int = 10,
    max_features: int = 20000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds topics for the current corpus and saves:
      - topics.csv
      - corpus_with_topics.csv
    Returns: (topics_df, corpus_with_topics_df)
    """
    df = load_best_corpus()
    text_col = choose_text_column(df)

    texts = df[text_col].astype(str).fillna("")
    mask = texts.str.strip().ne("")
    df_use = df.loc[mask].reset_index(drop=True)
    texts_use = texts.loc[mask].tolist()

    if len(df_use) < max(5, n_topics):
        raise ValueError(f"Not enough documents to build {n_topics} topics. Have {len(df_use)}.")

    vec = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vec.fit_transform(texts_use)

    km = KMeans(n_clusters=n_topics, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    df_use["topic_id"] = labels

    # Extract top terms per cluster
    terms = vec.get_feature_names_out()
    centroids = km.cluster_centers_
    topics = []
    for topic_id in range(n_topics):
        top_idx = centroids[topic_id].argsort()[::-1][:top_terms]
        keywords = [terms[i] for i in top_idx]
        count = int((labels == topic_id).sum())
        topics.append(
            {
                "topic_id": topic_id,
                "doc_count": count,
                "keywords": ", ".join(keywords),
                "text_col": text_col,
            }
        )

    topics_df = pd.DataFrame(topics).sort_values("doc_count", ascending=False).reset_index(drop=True)

    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(TOPICS_PATH, index=False)
    df_use.to_csv(CORPUS_WITH_TOPICS_PATH, index=False)

    print(f"Saved topics to: {TOPICS_PATH}")
    print(f"Saved corpus with topics to: {CORPUS_WITH_TOPICS_PATH}")
    return topics_df, df_use


def load_topics() -> pd.DataFrame:
    if not TOPICS_PATH.exists():
        raise FileNotFoundError(f"{TOPICS_PATH} not found. Run build_topics() first.")
    return pd.read_csv(TOPICS_PATH)


def load_corpus_with_topics() -> pd.DataFrame:
    if not CORPUS_WITH_TOPICS_PATH.exists():
        raise FileNotFoundError(f"{CORPUS_WITH_TOPICS_PATH} not found. Run build_topics() first.")
    return pd.read_csv(CORPUS_WITH_TOPICS_PATH)


if __name__ == "__main__":
    build_topics()

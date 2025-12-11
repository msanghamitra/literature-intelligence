from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# --- Make sure the project root is on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.summarizer import (
    load_summary_corpus,          # loads the best corpus
    summarize_text,               # summarises a single text (for "Summarise now")
    run_full_corpus_summarisation # runs the batch job over the entire corpus
)

# -------------------------------------------------
# 1. Load and normalise the data for the UI
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def get_corpus() -> pd.DataFrame:
    df = load_summary_corpus()

    # Normalise title column
    if "title" in df.columns:
        df["__title"] = df["title"]
    elif "title_clean" in df.columns:
        df["__title"] = df["title_clean"]
    else:
        df["__title"] = "Untitled"

    # Normalise abstract/summary column
    if "summary" in df.columns:
        df["__abstract"] = df["summary"]
    elif "abstract_clean" in df.columns:
        df["__abstract"] = df["abstract_clean"]
    else:
        # ❗ You had "__abstract_" here; UI expects "__abstract"
        df["__abstract"] = ""

    # Ensure text_unit exists
    if "text_unit" not in df.columns:
        df["text_unit"] = (
            df["__title"].fillna("") + ". " + df["__abstract"].fillna("")
        )

    return df


# -------------------------------------------------
# 2. Simple text search
# -------------------------------------------------
def search_corpus(df: pd.DataFrame, query: str, top_k: int = 10) -> pd.DataFrame:
    # If the user hasn’t typed anything, just return the first top_k rows.
    if not query:
        return df.head(top_k)

    q = query.lower()
    mask = (
        df["__title"].astype(str).str.lower().str.contains(q)
        | df["__abstract"].astype(str).str.lower().str.contains(q)
    )
    # __title contains the query OR __abstract contains the query (case-insensitive).
    return df[mask].head(top_k)


# -------------------------------------------------
# 3. App layout & sidebar
# -------------------------------------------------
st.set_page_config(
    page_title="Research Librarian",
    layout="wide",
)

st.sidebar.title("Customise here")
st.sidebar.markdown("Prototype UI - summarisation + future features")

top_k = st.sidebar.slider("Number of results", 5, 50, 10, step=5)
max_len = st.sidebar.slider("Max summary length", 64, 256, 128, step=16)
min_len = st.sidebar.slider("Min summary length", 16, 64, 32, step=8)

# Batch summarisation button
st.sidebar.markdown("---")
if st.sidebar.button("🏃 Run full corpus summarisation (offline)"):
    with st.spinner("Summarising entire corpus…"):
        run_full_corpus_summarisation()
    st.sidebar.success("Done. Reload the app to see updated summaries.")

# Tabs for different feature areas
tab_summaries, tab_topics, tab_qa = st.tabs(
    ["Summaries", "Topics (coming soon)", "Q&A (coming soon)"]
)

df = get_corpus()

# -------------------------------------------------
# 4. Summaries tab
# -------------------------------------------------
with tab_summaries:
    st.header("Search papers & view summaries")

    query = st.text_input(
        "Search by keyword, title, or abstract",
        placeholder="e.g. diffusion models, federated learning, GNNs…",
    )

    if query:
        results = search_corpus(df, query, top_k=top_k)

        if results.empty:
            st.warning("No papers found for this query.")
        else:
            st.write(f"Showing up to **{len(results)}** papers.")
            for idx, row in results.iterrows():
                title = row["__title"]
                authors = row.get("authors", "Unknown authors")
                published = row.get("published", "")

                with st.expander(title):
                    meta = f"**Authors:** {authors}"
                    if isinstance(published, str) and published:
                        meta += f"  |  **Published:** {published}"
                    st.markdown(meta)

                    st.markdown("**Abstract / existing summary**")
                    st.write(row["__abstract"])

                    # Precomputed CSV summary if present
                    if "summary" in row and isinstance(row["summary"], str):
                        st.markdown("**Precomputed model summary (CSV)**")
                        st.write(row["summary"])

                    # On-the-fly summarisation with the model
                    if st.button("Summarise now", key=f"sum_{idx}"):
                        with st.spinner("Generating summary…"):
                            text = row.get("text_unit", row["__abstract"])
                            new_summary = summarize_text(
                                text,
                                max_length=max_len,
                                min_length=min_len,
                            )
                            st.markdown("**On-the-fly summary**")
                            st.write(new_summary)
    else:
        st.info("🔍 Enter a search query above to explore the corpus.")


# -------------------------------------------------
# 5. Topics + Q&A placeholders
# -------------------------------------------------
with tab_topics:
    st.header("Topics (coming soon)")
    st.write(
        "- Plug BERTopic/other topics here later\n"
        "- Topic list, keywords, and representative papers\n"
        "- Topic filter for the summaries tab"
    )

with tab_qa:
    st.header("Paper Q&A (coming soon)")
    st.write(
        "- Later: select a paper and ask questions\n"
        "- Implement RAG over `text_unit` + an LLM"
    )

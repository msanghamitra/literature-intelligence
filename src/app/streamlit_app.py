from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# --- Make sure the project root is on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.summarizer import (
    load_summary_corpus,
    summarize_text,
    run_full_corpus_summarisation
)

from src.models.embeddings import (
    run_full_corpus_embedding,
    retrieve_top_k
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

    # ✅ Use ORIGINAL abstract for display (not the model summary)
    if "abstract_clean" in df.columns:
        df["__abstract"] = df["abstract_clean"]
    elif "summary" in df.columns:
        df["__abstract"] = df["summary"]
    else:
        df["__abstract"] = ""

    # ✅ Keep model-generated summary separately
    if "summary" in df.columns:
        df["__summary"] = df["summary"]
    else:
        df["__summary"] = ""

    # Ensure text_unit exists: title + abstract (original)
    if "text_unit" not in df.columns:
        df["text_unit"] = df["__title"].fillna("") + ". " + df["__abstract"].fillna("")

    return df


def get_semantic_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align with embeddings.py:
    - embeddings built over 'summary' if present, else 'text_unit'
    - empty rows filtered, reset_index(drop=True)
    """
    if "summary" in df.columns and df["summary"].notna().any():
        text_col = "summary"
    else:
        text_col = "text_unit"

    texts = df[text_col].astype(str).fillna("")
    mask = texts.str.strip().ne("")
    df_use = df.loc[mask].reset_index(drop=True)
    df_use["__embedded_text_col"] = text_col
    return df_use


# -------------------------------------------------
# 2. Simple keyword search
# -------------------------------------------------
def search_corpus(df: pd.DataFrame, query: str, top_k: int = 10) -> pd.DataFrame:
    if not query:
        return df.head(top_k)

    q = query.lower()
    mask = (
        df["__title"].astype(str).str.lower().str.contains(q)
        | df["__abstract"].astype(str).str.lower().str.contains(q)
        | df["__summary"].astype(str).str.lower().str.contains(q)
    )
    return df[mask].head(top_k)


# -------------------------------------------------
# 3. App layout & sidebar
# -------------------------------------------------
st.set_page_config(page_title="Research Librarian", layout="wide")

st.sidebar.title("Customise here")
st.sidebar.markdown("Prototype UI - summarisation + embeddings + topics + Q&A")

top_k = st.sidebar.slider("Number of results", 5, 50, 10, step=5)
max_len = st.sidebar.slider("Max summary length", 64, 256, 128, step=16)
min_len = st.sidebar.slider("Min summary length", 16, 64, 32, step=8)

st.sidebar.markdown("---")
if st.sidebar.button("🏃 Run full corpus summarisation (offline)"):
    with st.spinner("Summarising entire corpus…"):
        run_full_corpus_summarisation()
    st.cache_data.clear()
    st.sidebar.success("Done. Reloaded cached data.")

st.sidebar.markdown("---")
if st.sidebar.button("🧠 Build embeddings (semantic search)"):
    with st.spinner("Building embeddings for corpus…"):
        run_full_corpus_embedding()
    st.cache_data.clear()
    st.sidebar.success("Embeddings saved. Semantic search is ready.")

# ✅ Topics build button (lazy import to avoid sklearn import at startup)
st.sidebar.markdown("---")
if st.sidebar.button("🧩 Build topics (offline)"):
    try:
        from src.models.topics import build_topics
        with st.spinner("Building topics…"):
            build_topics()
        st.sidebar.success("Topics built. Open the Topics tab.")
    except Exception as e:
        st.sidebar.error(f"Topics failed: {e}")

df = get_corpus()

# -------------------------------------------------
# Search mode moved to TOP (above tabs)
# -------------------------------------------------
st.markdown("### Search mode")
search_mode = st.radio(
    "Choose how to search the library",
    ["Keyword (fast)", "Semantic (embeddings)"],
    index=0,
    horizontal=True
)

# Tabs
tab_summaries, tab_topics, tab_qa = st.tabs(["Summaries", "Topics", "Q&A"])


# -------------------------------------------------
# 4. Summaries tab
# -------------------------------------------------
with tab_summaries:
    st.header("Search papers & view summaries")

    query = st.text_input(
        "Search by keyword or meaning (semantic)",
        placeholder="e.g. diffusion models, federated learning, GNNs…",
    )

    if not query:
        st.info("🔍 Enter a search query above to explore the corpus.")
    else:
        if search_mode == "Keyword (fast)":
            results = search_corpus(df, query, top_k=top_k)

            if results.empty:
                st.warning("No papers found for this query.")
            else:
                st.write(f"Showing up to **{len(results)}** papers.")
                for idx, row in results.iterrows():
                    title = row["__title"]
                    authors = row.get("authors", "Unknown authors")
                    published = row.get("published", "")
                    pdf_url = row.get("pdf_url", "")

                    with st.expander(title):
                        meta = f"**Authors:** {authors}"
                        if isinstance(published, str) and published:
                            meta += f"  |  **Published:** {published}"
                        st.markdown(meta)

                        if isinstance(pdf_url, str) and pdf_url.strip():
                            st.markdown(f"[Read paper (PDF)]({pdf_url})")

                        st.markdown("**Abstract (original)**")
                        st.write(row["__abstract"])

                        precomputed = row.get("__summary", "")
                        if isinstance(precomputed, str) and precomputed.strip():
                            st.markdown("**Precomputed model summary (CSV)**")
                            st.write(precomputed)

                        if st.button("Summarise now", key=f"sum_kw_{idx}"):
                            with st.spinner("Generating summary…"):
                                text = row.get("text_unit", row["__abstract"])
                                new_summary = summarize_text(
                                    text,
                                    max_length=max_len,
                                    min_length=min_len,
                                )

                            if isinstance(precomputed, str) and precomputed.strip() and new_summary.strip() == precomputed.strip():
                                st.info("This matches the precomputed summary (same content + similar settings).")
                            else:
                                st.markdown("**On-the-fly summary**")
                                st.write(new_summary)

        else:
            try:
                sem_results = retrieve_top_k(query, k=top_k)
            except FileNotFoundError:
                st.warning("Embeddings not found yet. Use the sidebar button **Build embeddings** first.")
                sem_results = pd.DataFrame()

            if sem_results.empty:
                st.warning("No semantic results (or embeddings not built).")
            else:
                df_sem = get_semantic_df(df)
                st.write(f"Showing top **{len(sem_results)}** semantic matches.")

                for i, r in sem_results.iterrows():
                    row_idx = int(r.get("row_idx", -1))
                    sim = float(r.get("similarity", 0.0))

                    if row_idx < 0 or row_idx >= len(df_sem):
                        continue

                    paper = df_sem.iloc[row_idx]
                    title = paper["__title"]
                    authors = paper.get("authors", "Unknown authors")
                    published = paper.get("published", "")
                    pdf_url = paper.get("pdf_url", "")

                    with st.expander(f"{title}  (similarity: {sim:.3f})"):
                        meta = f"**Authors:** {authors}"
                        if isinstance(published, str) and published:
                            meta += f"  |  **Published:** {published}"
                        st.markdown(meta)

                        if isinstance(pdf_url, str) and pdf_url.strip():
                            st.markdown(f"[Read paper (PDF)]({pdf_url})")

                        st.markdown("**Abstract (original)**")
                        st.write(paper["__abstract"])

                        precomputed = paper.get("__summary", "")
                        if isinstance(precomputed, str) and precomputed.strip():
                            st.markdown("**Precomputed model summary (CSV)**")
                            st.write(precomputed)

                        if st.button("Summarise now", key=f"sum_sem_{row_idx}_{i}"):
                            with st.spinner("Generating summary…"):
                                text = paper.get("text_unit", paper["__abstract"])
                                new_summary = summarize_text(
                                    text,
                                    max_length=max_len,
                                    min_length=min_len,
                                )

                            if isinstance(precomputed, str) and precomputed.strip() and new_summary.strip() == precomputed.strip():
                                st.info("This matches the precomputed summary (same content + similar settings).")
                            else:
                                st.markdown("**On-the-fly summary**")
                                st.write(new_summary)


# -------------------------------------------------
# 5. Topics tab (lazy import)
# -------------------------------------------------
with tab_topics:
    st.header("Topics")

    try:
        from src.models.topics import build_topics, load_topics, load_corpus_with_topics
    except Exception as e:
        st.error(f"Could not load Topics module/dependencies: {e}")
        st.stop()

    st.write("Build topics once, then explore them here.")

    if st.button("🧩 Build / Rebuild topics"):
        with st.spinner("Building topics…"):
            build_topics()
        st.success("Topics built.")

    try:
        topics_df = load_topics()
        corpus_topics_df = load_corpus_with_topics()
    except FileNotFoundError:
        st.warning("Topics not built yet. Click **Build / Rebuild topics**.")
        topics_df = pd.DataFrame()
        corpus_topics_df = pd.DataFrame()

    if not topics_df.empty:
        st.subheader("Topic list")
        st.dataframe(topics_df, width="stretch")

        st.subheader("Browse papers by topic")
        topic_ids = topics_df["topic_id"].tolist()
        selected_topic = st.selectbox("Select a topic_id", topic_ids)

        subset = corpus_topics_df[corpus_topics_df["topic_id"] == selected_topic].copy()
        st.write(f"Papers in topic **{selected_topic}**: {len(subset)}")

        show_cols = [c for c in ["title", "title_clean", "published", "pdf_url"] if c in subset.columns]
        if show_cols:
            st.dataframe(subset[show_cols].head(30), width="stretch")
        else:
            st.dataframe(subset.head(30), width="stretch")


# -------------------------------------------------
# 6. Q&A tab (lazy import)
# -------------------------------------------------
with tab_qa:
    st.header("Paper Q&A (RAG)")
    st.write("Retrieval (embeddings) + extractive QA over retrieved context.")

    question = st.text_input("Ask a question", placeholder="e.g. What is graphene used for in these papers?")
    k_ctx = st.slider("Number of papers to retrieve", 2, 10, 5)

    if st.button("🤖 Answer"):
        try:
            from src.models.rag import answer_question
        except Exception as e:
            st.error(f"Could not load RAG module/dependencies: {e}")
            st.stop()

        try:
            with st.spinner("Retrieving relevant papers + answering…"):
                res = answer_question(question, k=k_ctx)
        except FileNotFoundError:
            st.warning("Embeddings not found. Build embeddings first in the sidebar.")
            res = None

        if res:
            if res.get("error"):
                st.warning(res["error"])
            else:
                st.subheader("Answer")
                st.write(res["answer"])
                st.caption(f"Confidence score: {res.get('score', 0.0):.3f}")

                st.subheader("Sources")
                sources = res.get("sources")
                if isinstance(sources, pd.DataFrame) and not sources.empty:
                    for _, srow in sources.iterrows():
                        title = srow.get("title", "Untitled")
                        sim = float(srow.get("similarity", 0.0))
                        pdf_url = srow.get("pdf_url", "")

                        st.markdown(f"- **{title}** (similarity: {sim:.3f})")
                        if isinstance(pdf_url, str) and pdf_url.strip():
                            st.markdown(f"  - [Read paper (PDF)]({pdf_url})")

                with st.expander("Context preview (debug)"):
                    st.write(res.get("context_preview", ""))

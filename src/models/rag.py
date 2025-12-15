# src/models/rag.py
# Minimal RAG: embeddings retrieval + extractive QA over retrieved context.

from pathlib import Path
from functools import lru_cache
import pandas as pd
from transformers import pipeline

from src.models.embeddings import retrieve_top_k
from src.models.summarizer import load_summary_corpus

# Base dir = project root (two levels up: src/models -> src -> project)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"


def _choose_embedding_text_col(df: pd.DataFrame) -> str:
    # Must match embeddings.py build logic
    if "summary" in df.columns and df["summary"].notna().any():
        return "summary"
    return "text_unit"


def _get_semantic_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Must align with embeddings.py index mapping (mask non-empty, reset_index).
    This ensures row_idx from embeddings_index.csv maps correctly.
    """
    text_col = _choose_embedding_text_col(df)
    texts = df[text_col].astype(str).fillna("")
    mask = texts.str.strip().ne("")
    return df.loc[mask].reset_index(drop=True)


@lru_cache(maxsize=1)
def get_qa_pipeline():
    # Small extractive QA model (CPU OK)
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=-1
    )


def retrieve_context(
    query: str,
    k: int = 5,
    max_chars: int = 6000,
) -> tuple[str, pd.DataFrame]:
    """
    Returns:
      - context string (concatenated from top-k docs)
      - sources dataframe (titles, similarity, pdf_url if present)
    """
    df = load_summary_corpus()
    df_sem = _get_semantic_df(df)

    sem = retrieve_top_k(query, k=k)

    if sem.empty:
        return "", pd.DataFrame()

    chunks = []
    sources = []

    for _, r in sem.iterrows():
        row_idx = int(r.get("row_idx", -1))
        sim = float(r.get("similarity", 0.0))
        if row_idx < 0 or row_idx >= len(df_sem):
            continue

        paper = df_sem.iloc[row_idx]
        title = str(paper.get("title", paper.get("title_clean", "Untitled")))
        pdf_url = paper.get("pdf_url", "")
        # Prefer summary if present, else abstract/text_unit
        body = paper.get("summary", "") or paper.get("abstract_clean", "") or paper.get("text_unit", "")

        chunk = f"TITLE: {title}\nCONTENT: {str(body)}\n"
        chunks.append(chunk)

        sources.append(
            {
                "title": title,
                "similarity": sim,
                "pdf_url": pdf_url if isinstance(pdf_url, str) else "",
            }
        )

        if sum(len(c) for c in chunks) > max_chars:
            break

    context = "\n---\n".join(chunks)[:max_chars]
    sources_df = pd.DataFrame(sources).sort_values("similarity", ascending=False).reset_index(drop=True)
    return context, sources_df


def answer_question(
    question: str,
    query: str | None = None,
    k: int = 5,
) -> dict:
    """
    If query is not provided, we use the question as the retrieval query.
    Returns dict with:
      - answer
      - score
      - sources (df)
      - context_preview
    """
    retrieval_query = query or question
    context, sources = retrieve_context(retrieval_query, k=k)

    if not context.strip():
        return {
            "answer": "",
            "score": 0.0,
            "sources": sources,
            "context_preview": "",
            "error": "No context retrieved. Build embeddings and try a different query.",
        }

    qa = get_qa_pipeline()
    out = qa(question=question, context=context)

    return {
        "answer": out.get("answer", ""),
        "score": float(out.get("score", 0.0)),
        "sources": sources,
        "context_preview": context[:800] + ("..." if len(context) > 800 else ""),
    }


if __name__ == "__main__":
    res = answer_question("What is the main idea?", query="graphene", k=3)
    print(res["answer"], res["score"])
    print(res["sources"].head(3))

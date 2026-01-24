# arxiv_loader.py
"""
arxiv_loader.py (simple, live-only)

Goal:
Type a topic → fetch the latest arXiv papers → show N results → done.

What this file provides:
- fetch_arxiv_papers_df(topic, max_results, category) -> (df, message)
- (optional) CLI: python arxiv_loader.py --query "machine learning" --max_results 10

Notes:
- No local corpus logic here.
- No PDF downloading by default (keeps it fast + economical).
- Query is built as all:"<topic>" so any topic works.
- Sorts by SubmittedDate (latest first).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import arxiv
import pandas as pd


# -------------------------------
# Data model
# -------------------------------
@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    summary: str          # abstract
    authors: str
    published: str
    updated: str
    primary_category: str
    categories: str
    entry_id: str         # arXiv page URL
    pdf_url: str
    text_unit: str        # title + abstract


# -------------------------------
# Helpers
# -------------------------------
def _clean(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").split()).strip()


def build_search_query(topic: str, category: Optional[str] = None) -> str:
    """
    Convert a topic into a robust arXiv query.
    Example:
      topic="machine learning" -> all:"machine learning"
      + category="cs.LG" -> cat:cs.LG AND (all:"machine learning")
    """
    topic = (topic or "").strip()
    if not topic:
        return ""

    base = f'all:"{topic}"'
    if category:
        category = category.strip()
        if category:
            return f"cat:{category} AND ({base})"
    return base


def build_result_message(topic: str, requested: int, found: int, category: Optional[str]) -> str:
    cat = f", category={category}" if category else ""
    if found == 0:
        return f'No relevant papers found for "{topic}"{cat}.'
    if found < requested:
        return f'Only {found} relevant papers found for "{topic}"{cat} (requested {requested}). Showing all {found}.'
    return f'Found {found} latest papers for "{topic}"{cat}.'


# -------------------------------
# Core: live fetch from arXiv
# -------------------------------
def fetch_arxiv_papers(
    topic: str,
    max_results: int = 10,
    category: Optional[str] = None,
) -> List[PaperMetadata]:
    """
    Fetch latest arXiv papers for a topic (and optional category).
    Latest-first sorting.
    """
    search_query = build_search_query(topic, category)
    if not search_query:
        return []

    search = arxiv.Search(
        query=search_query,
        max_results=int(max_results),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[PaperMetadata] = []
    for r in search.results():
        paper_id = r.get_short_id()

        title = _clean(r.title)
        summary = _clean(r.summary)
        authors = ", ".join(a.name for a in (r.authors or []))

        published = r.published.strftime("%Y-%m-%d") if getattr(r, "published", None) else ""
        updated = r.updated.strftime("%Y-%m-%d") if getattr(r, "updated", None) else ""

        primary_category = getattr(r, "primary_category", "") or ""
        categories_list = getattr(r, "categories", []) or []
        categories = ", ".join(categories_list)

        entry_id = getattr(r, "entry_id", "") or ""
        pdf_url = getattr(r, "pdf_url", "") or ""

        text_unit = _clean(f"{title}. {summary}")

        papers.append(
            PaperMetadata(
                arxiv_id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                published=published,
                updated=updated,
                primary_category=primary_category,
                categories=categories,
                entry_id=entry_id,
                pdf_url=pdf_url,
                text_unit=text_unit,
            )
        )

    return papers


def fetch_arxiv_papers_df(
    topic: str,
    max_results: int = 10,
    category: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    UI-friendly helper:
    Returns:
      df: DataFrame of up to N latest papers
      msg: message that you can show in Streamlit (e.g. fewer than N papers)
    """
    papers = fetch_arxiv_papers(topic=topic, max_results=max_results, category=category)
    msg = build_result_message(topic=topic, requested=int(max_results), found=len(papers), category=category)

    if not papers:
        return pd.DataFrame(), msg

    df = pd.DataFrame([asdict(p) for p in papers])
    return df, msg


# -------------------------------
# Optional: CLI runner
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch latest arXiv papers by topic.")
    parser.add_argument("--query", type=str, required=True, help='Topic, e.g. "machine learning"')
    parser.add_argument("--max_results", type=int, default=10, help="How many papers (default: 10)")
    parser.add_argument("--category", type=str, default=None, help='Optional category, e.g. "cs.LG"')
    args = parser.parse_args()

    df, msg = fetch_arxiv_papers_df(topic=args.query, max_results=args.max_results, category=args.category)
    print(msg)
    if not df.empty:
        print(df[["title", "published", "primary_category", "pdf_url"]].head(min(len(df), 10)).to_string(index=False))


if __name__ == "__main__":
    main()

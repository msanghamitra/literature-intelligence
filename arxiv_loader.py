import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import arxiv


# ---------------------------------------------------------------------
# Data model: store metadata for each arXiv paper
# ---------------------------------------------------------------------

@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    summary: str
    authors: str
    published: str
    updated: str
    primary_category: str
    pdf_url: str


# ---------------------------------------------------------------------
# Core function: query arXiv and (optionally) download PDFs
# ---------------------------------------------------------------------

def fetch_arxiv_papers(
    query: str,
    max_results: int = 20,
    category: str | None = None,
    download_pdfs: bool = False,
    output_dir: str = "data/arxiv_papers",
) -> List[PaperMetadata]:
    """
    Query arXiv for papers matching the given query/topic and optionally
    download their PDFs.

    Args:
        query: Free-text search query, e.g. "mRNA vaccine stability"
        max_results: Max number of papers to retrieve
        category: Optional arXiv category filter, e.g. "cs.CL", "q-bio.BM"
                  NOTE: arxiv.Search does NOT take a 'category' argument
                  directly in many versions, so we inject it into the query
                  string as:  cat:cs.CL AND (your query)
        download_pdfs: If True, download PDFs to output_dir/pdfs/
        output_dir: Base folder to store metadata (and PDFs if enabled)

    Returns:
        List of PaperMetadata objects.
    """

    output_path = Path(output_dir)
    pdf_path = output_path / "pdfs"
    output_path.mkdir(parents=True, exist_ok=True)
    if download_pdfs:
        pdf_path.mkdir(parents=True, exist_ok=True)

    # Build search query string
    # If category is provided, prepend cat:category
    # Example: cat:q-bio.BM AND (AI in medicine)
    if category:
        search_query = f"cat:{category} AND ({query})"
    else:
        search_query = query

    # IMPORTANT: arxiv.Search in many versions does NOT support "category="
    # so we only pass query, max_results, sort_by, sort_order.
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[PaperMetadata] = []

    for result in search.results():
        paper_id = result.get_short_id()
        title = result.title.strip().replace("\n", " ")
        summary = result.summary.strip().replace("\n", " ")
        authors = ", ".join(a.name for a in result.authors)
        published = result.published.strftime("%Y-%m-%d") if result.published else ""
        updated = result.updated.strftime("%Y-%m-%d") if result.updated else ""
        primary_category = result.primary_category
        pdf_url = result.pdf_url

        meta = PaperMetadata(
            arxiv_id=paper_id,
            title=title,
            summary=summary,
            authors=authors,
            published=published,
            updated=updated,
            primary_category=primary_category,
            pdf_url=pdf_url,
        )
        papers.append(meta)

        # Optionally download PDF
        if download_pdfs:
            pdf_file = pdf_path / f"{paper_id}.pdf"
            try:
                print(f"Downloading PDF for {paper_id} -> {pdf_file}")
                result.download_pdf(filename=str(pdf_file))
            except Exception as e:
                print(f"⚠️  Failed to download PDF for {paper_id}: {e}")

    # Save metadata to CSV for later NLP processing
    csv_path = output_path / "metadata.csv"
    print(f"\nSaving metadata for {len(papers)} papers to {csv_path}")
    if papers:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(asdict(papers[0]).keys()),
            )
            writer.writeheader()
            for p in papers:
                writer.writerow(asdict(p))

    return papers


# ---------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query arXiv by topic and optionally download PDFs."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help='Topic or query string, e.g. "mRNA vaccine stability"',
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=20,
        help="Maximum number of papers to fetch (default: 20)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help='Optional arXiv category, e.g. "cs.CL", "q-bio.BM", "stat.ML"',
    )
    parser.add_argument(
        "--download_pdfs",
        action="store_true",
        help="If set, also download PDFs for each paper",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/arxiv_papers",
        help="Directory to save metadata (and PDFs if enabled)",
    )

    args = parser.parse_args()

    print(
        f"\n🔍 Querying arXiv for topic: '{args.query}' "
        f"(max_results={args.max_results}, category={args.category})"
    )

    papers = fetch_arxiv_papers(
        query=args.query,
        max_results=args.max_results,
        category=args.category,
        download_pdfs=args.download_pdfs,
        output_dir=args.output_dir,
    )

    print(f"\n✅ Done. Retrieved {len(papers)} papers.")


if __name__ == "__main__":
    main()

# src/models/pdf_text.py
from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader


# Cache folder inside project data
BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "arxiv_papers" / "pdf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PdfPageText:
    page: int
    text: str


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:200] if len(s) > 200 else s


def _hash_key(pdf_url: str, cache_key: Optional[str] = None) -> str:
    base = (cache_key or "") + "|" + (pdf_url or "")
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def download_pdf(pdf_url: str, cache_key: Optional[str] = None, timeout: int = 30) -> Path:
    """
    Download PDF (if not already cached) and return local path.
    """
    if not pdf_url or not pdf_url.strip():
        raise ValueError("Missing pdf_url")

    h = _hash_key(pdf_url, cache_key=cache_key)
    filename = _safe_filename(f"{h}.pdf")
    pdf_path = CACHE_DIR / filename

    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    # Download
    req = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "Mozilla/5.0"}  # avoids occasional 403
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()

    pdf_path.write_bytes(data)
    return pdf_path


def extract_pdf_pages_text(pdf_path: Path) -> List[PdfPageText]:
    """
    Extract text page-by-page using pypdf.
    """
    reader = PdfReader(str(pdf_path))
    pages: List[PdfPageText] = []

    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = " ".join(txt.replace("\n", " ").split()).strip()
        pages.append(PdfPageText(page=i + 1, text=txt))

    return pages


def get_pdf_pages_from_url(
    pdf_url: str,
    cache_key: Optional[str] = None,
) -> Tuple[List[PdfPageText], str]:
    """
    Returns (pages, msg). Uses disk caching for the PDF.
    """
    try:
        pdf_path = download_pdf(pdf_url, cache_key=cache_key)
    except Exception as e:
        return [], f"Could not download PDF: {e}"

    try:
        pages = extract_pdf_pages_text(pdf_path)
    except Exception as e:
        return [], f"Could not extract text from PDF: {e}"

    non_empty = sum(1 for p in pages if p.text.strip())
    if non_empty == 0:
        return pages, "PDF text extraction returned empty text (paper may be scanned/figure-heavy)."
    return pages, f"Extracted text from {non_empty}/{len(pages)} pages."
##comand streamlit run src/app/streamlit_app.py
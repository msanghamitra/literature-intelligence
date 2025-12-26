# Paperminer - Literature Intelligence Solution

AI-powered literature intelligence solution that automatically retrieves, summarizes, and organizes scientific papers (starting with arXiv), enabling research scientists to explore topics faster and ask natural-language questions over the corpus.

Under the hood, the system uses:

- Paper Retreival - Finding relavant paper bases on a search
- Transformer-based summarisation ( NLP / deep learning).
- Sentence-transformer embeddings with FAISS / cosine similarity for semantic vector search.
- BERTopic for topic modelling (unsupervised ML / NLP).
- A planned RAG-style Q&A layer with an LLM for grounded, natural-language answers over the literature.

Together, this forms an end-to-end AI stack for intelligent literature exploration.


---

## 1. Problem

Research scientists often:

- Spend hours manually searching and skimming dozens of PDFs.
- Struggle to get a **high-level map** of themes in a new research area.
- Repeat ad-hoc literature scans that are hard to reproduce or share.

This project aims to turn a raw collection of papers into a **navigable, queryable knowledge layer**.

---

## 2. Solution Overview

The Literature Intelligence Engine:

1. **Ingests papers from arXiv** for a given query.
2. **Extracts text** (title, abstract, and selected sections).
3. **Generates concise summaries** for each paper using transformer-based summarisation.
4. **Clusters papers into topics** using BERTopic to reveal research themes.
5. **Builds an embedding index** over summaries for semantic search.
6. Exposes a **Streamlit UI** where users can:
   - Browse papers by topic.
   - Search by keyword.
   - Ask questions (RAG-style) over the selected corpus.

---

## 3. Value for Research Scientists

- **Faster literature triage:** Quickly identify which 5–30 papers are worth reading in depth instead of manually skimming dozens of articles.
- **Structured overview of a field:** Topic modelling clusters papers into coherent themes (e.g. biomarkers, trial design, model architectures), providing a “map” of the literature.
- **Question-driven exploration:** A retrieval-augmented Q&A interface lets users ask natural-language questions and get answers grounded in specific papers, with links for deeper reading.
- **Repeatable scans:** The pipeline can be re-run for the same or updated query, making literature scans reproducible and comparable over time.

---

## 4. Architecture (MVP)

**Pipeline**

1. **Ingestion**
   - `arxiv_loader.py` calls the arXiv API.
   - Saves metadata and PDFs / abstracts to `data/arxiv_papers/`.

2. **Preprocessing**
   - Clean and normalise text.
   - Use title + abstract (and optionally intro/conclusion) as the main text unit.

3. **Summarisation**
   - Hugging Face `transformers` pipeline with:
     - `sshleifer/distilbart-cnn-12-6` (DistilBART, CPU-friendly).
   - Outputs short, readable summaries per paper.

4. **Embeddings & Topics**
   - Embeddings via `sentence-transformers`:
     - `all-MiniLM-L6-v2` for fast, good-quality sentence embeddings.
   - BERTopic for topic modelling over summaries.

5. **RAG-style Q&A (MVP)**
   - Retrieve top-k relevant summaries via embeddings.
   - (Phase 1) Return retrieved summaries + metadata.
   - (Phase 2) Optionally call a QA/LLM model to synthesise an answer.

6. **UI**
   - Streamlit app:
     - Topic distribution overview.
     - Paper list with summaries and topics.
     - Search box + basic Q&A interface.

---

## 5. Tech Stack

- **Language:** Python 3.10 / 3.11
- **Data:** pandas, numpy
- **NLP & Models:** transformers, sentence-transformers, torch
- **Topic Modelling:** BERTopic, scikit-learn
- **Vector Search:** FAISS (optional) or cosine similarity
- **UI:** Streamlit
- **Data Source:** arXiv API (initially)

---

## 6. Project Structure (planned)

```bash
literature_intelligence/
├── data/
│   └── arxiv_papers/
│       ├── metadata.csv          # metadata for ingested papers
│       ├── main.py               # example script / entrypoint
│       └── ...                   # pdfs or cached text (optional)
├── src/                          # (to be created)
│   ├── ingestion/
│   │   └── arxiv_client.py       # wrapper around arXiv API
│   ├── preprocessing/
│   │   └── text_cleaning.py
│   ├── models/
│   │   ├── summarizer.py         # summarize_text()
│   │   ├── embeddings.py         # embed_texts()
│   │   ├── topics.py             # topic modelling helpers
│   │   └── rag.py                # retrieval + Q&A logic
│   └── app/
│       └── ui_streamlit.py       # Streamlit interface
├── arxiv_loader.py               # initial ingestion script
├── requirements.txt
└── README.md

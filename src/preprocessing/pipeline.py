"""
Unified processing pipeline for arXiv papers.
Runs all processing steps in sequence.
"""

from pathlib import Path
import pandas as pd
from .text_cleaner import build_corpus
from .summarizer import run_full_corpus_summarisation
from .topic_modeler import build_topics
from .chunking import chunk_full_text_papers

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

def run_full_pipeline(
    clean_corpus: bool = True,
    summarize: bool = True,
    build_topics_flag: bool = True,
    chunk_papers: bool = False,
    n_topics: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete processing pipeline.
    
    Args:
        clean_corpus: Run text cleaning
        summarize: Generate summaries
        build_topics_flag: Build topic models
        chunk_papers: Chunk papers (requires full text)
        n_topics: Number of topics for modeling
        
    Returns:
        Dictionary of DataFrames from each step
    """
    results = {}
    
    # Step 1: Clean corpus
    if clean_corpus:
        print("Step 1: Cleaning corpus...")
        clean_df = build_corpus()
        results['clean_corpus'] = clean_df
        print(f"  → Cleaned {len(clean_df)} papers")
    
    # Step 2: Generate summaries
    if summarize:
        print("Step 2: Generating summaries...")
        summary_df = run_full_corpus_summarisation()
        results['summaries'] = summary_df
        print(f"  → Generated {summary_df['summary'].notna().sum()} summaries")
    
    # Step 3: Build topics
    if build_topics_flag:
        print("Step 3: Building topics...")
        topics_df, corpus_with_topics = build_topics(n_topics=n_topics)
        results['topics'] = topics_df
        results['corpus_with_topics'] = corpus_with_topics
        print(f"  → Built {len(topics_df)} topics")
    
    # Step 4: Chunk papers (optional, requires full text)
    if chunk_papers:
        print("Step 4: Chunking papers...")
        chunks_df = chunk_full_text_papers()
        results['chunks'] = chunks_df
        print(f"  → Created {len(chunks_df)} chunks")
    
    print("Pipeline completed successfully!")
    return results

if __name__ == "__main__":
    # Run the full pipeline with default settings
    results = run_full_pipeline()
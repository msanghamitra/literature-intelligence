"""
Batch script to build a corpus of scientific papers from arXiv.
Can be run periodically to update the corpus.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.client.arxiv_client import ArxivClient
from src.data.store.document_store import DocumentStore
from src.data.store.vector_store import VectorStore
from src.embeddings.embedder import Embedder
import yaml
import logging
from datetime import datetime
from tqdm import tqdm


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'build_corpus_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path="config/retriever.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build a corpus of scientific papers from arXiv")
    
    parser.add_argument("--query", type=str, default="machine learning",
                       help="Search query for arXiv")
    parser.add_argument("--max-papers", type=int, default=100,
                       help="Maximum number of papers to fetch")
    parser.add_argument("--start-date", type=str,
                       help="Start date for papers (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                       help="End date for papers (YYYY-MM-DD)")
    parser.add_argument("--categories", type=str, nargs="+",
                       default=["cs.AI", "cs.LG", "cs.CL", "cs.CV"],
                       help="arXiv categories to search")
    parser.add_argument("--output-dir", type=str, default="data/corpus",
                       help="Output directory for corpus")
    parser.add_argument("--config", type=str, default="config/retriever.yaml",
                       help="Path to configuration file")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Rebuild vector index even if it exists")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip PDF download (metadata only)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def main():
    """Main function to build corpus."""
    args = parse_arguments()
    logger = setup_logging(getattr(logging, args.log_level))
    
    logger.info(f"Starting corpus build with query: {args.query}")
    logger.info(f"Max papers: {args.max_papers}")
    logger.info(f"Categories: {args.categories}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize clients and stores
        logger.info("Initializing arXiv client...")
        arxiv_client = ArxivClient()
        
        logger.info("Initializing document store...")
        doc_store = DocumentStore(args.output_dir)
        
        logger.info("Initializing embedder...")
        embedder = Embedder(config_path="config/embedding.yaml")
        
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            index_path=f"{args.output_dir}/vector_index",
            embedder=embedder
        )
        
        # Search for papers
        logger.info(f"Searching arXiv for: {args.query}")
        search_results = arxiv_client.search(
            query=args.query,
            max_results=args.max_papers,
            categories=args.categories,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        logger.info(f"Found {len(search_results)} papers")
        
        # Process papers
        successful = 0
        failed = 0
        
        for i, paper in enumerate(tqdm(search_results, desc="Processing papers")):
            try:
                paper_id = paper.arxiv_id or f"paper_{i}"
                
                # Download PDF if not skipping
                if not args.skip_download and hasattr(paper, 'pdf_url') and paper.pdf_url:
                    logger.debug(f"Downloading PDF for {paper_id}")
                    pdf_path = arxiv_client.download_pdf(
                        paper.pdf_url, 
                        f"{args.output_dir}/pdfs/{paper_id}.pdf"
                    )
                    paper.local_pdf_path = pdf_path
                
                # Store paper metadata
                doc_store.save_paper(paper)
                
                # Extract and embed text if PDF was downloaded
                if hasattr(paper, 'local_pdf_path') and paper.local_pdf_path:
                    logger.debug(f"Extracting text from {paper_id}")
                    text = arxiv_client.extract_text_from_pdf(paper.local_pdf_path)
                    
                    # Chunk text
                    chunks = embedder.chunk_text(text)
                    
                    # Embed chunks
                    embeddings = embedder.embed_batch(chunks)
                    
                    # Add to vector store
                    for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_id = f"{paper_id}_chunk_{chunk_idx}"
                        metadata = {
                            "paper_id": paper_id,
                            "chunk_idx": chunk_idx,
                            "title": paper.title,
                            "authors": ", ".join(paper.authors) if paper.authors else "",
                            "year": paper.published.year if paper.published else None,
                            "abstract": paper.abstract[:500] if paper.abstract else ""
                        }
                        vector_store.add(chunk_id, embedding, chunk, metadata)
                
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process paper {i}: {e}")
                failed += 1
        
        # Save vector index
        if not args.skip_download:
            logger.info("Saving vector index...")
            vector_store.save()
        
        # Generate statistics
        stats = {
            "total_papers": len(search_results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(search_results) if search_results else 0,
            "query": args.query,
            "categories": args.categories,
            "max_papers": args.max_papers,
            "build_date": datetime.now().isoformat(),
            "output_dir": args.output_dir
        }
        
        # Save statistics
        import json
        stats_file = f"{args.output_dir}/build_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Corpus build complete!")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("CORPUS BUILD SUMMARY")
        print(f"{'='*60}")
        print(f"Total papers processed: {len(search_results)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Output directory: {args.output_dir}")
        print(f"Statistics saved to: {stats_file}")
        
        if failed > 0:
            print(f"\nWarning: {failed} papers failed to process.")
            print("Check the log file for details.")
        
    except Exception as e:
        logger.error(f"Fatal error during corpus build: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
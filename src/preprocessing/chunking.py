"""
Scientific paper chunking module for processing full-text PDFs.
Implements semantic chunking strategies for scientific papers.
"""

from pathlib import Path
import re
from typing import List, Dict, Any
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "arxiv_papers"

class ScientificPaperChunker:
    """
    Chunk scientific papers based on semantic structure.
    Handles sections like abstract, introduction, methodology, results, etc.
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize chunker with size and overlap parameters.
        
        Args:
            chunk_size: Maximum character count per chunk
            overlap: Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.section_headers = [
            r'abstract',
            r'introduction',
            r'background',
            r'related work',
            r'method(?:ology)?',
            r'experiments?',
            r'results?',
            r'discussion',
            r'conclusions?',
            r'appendix',
            r'references',
        ]
    
    def find_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify sections in scientific paper text.
        
        Args:
            text: Full paper text
            
        Returns:
            List of dictionaries with section info
        """
        sections = []
        text_lower = text.lower()
        
        # Look for section headers
        for header_pattern in self.section_headers:
            pattern = rf'\n\s*\d*\.?\s*{header_pattern}\s*\n'
            matches = list(re.finditer(pattern, text_lower))
            
            for match in matches:
                sections.append({
                    'header': header_pattern,
                    'start': match.start(),
                    'end': match.end(),
                    'original_match': text[match.start():match.end()]
                })
        
        # Sort sections by position
        sections.sort(key=lambda x: x['start'])
        
        # Add the full text as final section
        if sections:
            sections.append({
                'header': 'remaining',
                'start': sections[-1]['end'],
                'end': len(text),
                'original_match': ''
            })
        
        return sections
    
    def chunk_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text by identified sections.
        
        Args:
            text: Full paper text
            
        Returns:
            List of chunks with metadata
        """
        sections = self.find_sections(text)
        chunks = []
        
        if not sections:
            # If no sections found, use sliding window approach
            return self.sliding_window_chunk(text)
        
        for i, section in enumerate(sections):
            start = section['start']
            end = section['end'] if i == len(sections) - 1 else sections[i + 1]['start']
            
            section_text = text[start:end].strip()
            if not section_text:
                continue
            
            # Further chunk large sections
            if len(section_text) > self.chunk_size:
                sub_chunks = self.sliding_window_chunk(section_text)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'chunk_id': f"{section['header']}_{j}",
                        'section': section['header'],
                        'chunk_index': j,
                        'text': sub_chunk['text'],
                        'char_start': start + sub_chunk['char_start'],
                        'char_end': start + sub_chunk['char_end']
                    })
            else:
                chunks.append({
                    'chunk_id': section['header'],
                    'section': section['header'],
                    'chunk_index': 0,
                    'text': section_text,
                    'char_start': start,
                    'char_end': end
                })
        
        return chunks
    
    def sliding_window_chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Sliding window chunking for text without clear sections.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at sentence boundary
            if end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'chunk_id': f"chunk_{chunk_index}",
                    'section': 'continuous',
                    'chunk_index': chunk_index,
                    'text': chunk_text,
                    'char_start': start,
                    'char_end': end
                })
                chunk_index += 1
            
            start = end - self.overlap if end - self.overlap > start else end
        
        return chunks
    
    def process_paper(self, paper_id: str, text: str) -> pd.DataFrame:
        """
        Process a single paper and return chunks as DataFrame.
        
        Args:
            paper_id: arXiv ID
            text: Full text of the paper
            
        Returns:
            DataFrame with chunks
        """
        chunks = self.chunk_by_sections(text)
        
        # Add paper metadata to each chunk
        for chunk in chunks:
            chunk['arxiv_id'] = paper_id
            chunk['chunk_size'] = len(chunk['text'])
        
        return pd.DataFrame(chunks)


def chunk_full_text_papers(metadata_path: Path = None) -> pd.DataFrame:
    """
    Batch process papers for chunking.
    
    Args:
        metadata_path: Path to metadata CSV
        
    Returns:
        DataFrame with all chunks
    """
    if metadata_path is None:
        metadata_path = DATA_DIR / "metadata.csv"
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Initialize chunker
    chunker = ScientificPaperChunker()
    
    all_chunks = []
    
    # This is a placeholder - in practice, you'd load full text from PDFs
    # For now, we'll chunk the abstracts
    for _, row in df.iterrows():
        chunks_df = chunker.process_paper(
            row['arxiv_id'],
            row.get('summary', '')  # Using abstract as placeholder
        )
        all_chunks.append(chunks_df)
    
    # Combine all chunks
    if all_chunks:
        combined_df = pd.concat(all_chunks, ignore_index=True)
        
        # Save chunks
        chunks_path = DATA_DIR / "paper_chunks.csv"
        combined_df.to_csv(chunks_path, index=False)
        print(f"Saved {len(combined_df)} chunks to {chunks_path}")
        
        return combined_df
    
    return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    chunks_df = chunk_full_text_papers()
    if not chunks_df.empty:
        print(f"Generated {len(chunks_df)} chunks from papers")
        print(chunks_df.head())
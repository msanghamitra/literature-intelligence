from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
sys.path.append(str(BASE_DIR))

from src.models.embeddings import retrieve_top_k
print(retrieve_top_k("transformer summarization for scientific papers", k=5))

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import arxiv 

## Data model. This part of code is to store the information about each paper you retrieve from the arXiv API in a clean, structured way.
class PaperMetadata:
    def __init__(self, arxiv_id, title, summary, authors, published, updated, primary_category, pdf_url):
        self.arxiv_id = arxiv_id
        self.title = title
        self.summary = summary
        self.authors = authors
        self.published = published
        self.updated = updated
        self.primary_category = primary_category
        self.pdf_url = pdf_url
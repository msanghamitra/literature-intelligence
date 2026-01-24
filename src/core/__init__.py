"""
Core domain models for scientific papers.
"""

from .paper import Paper
from .query import SearchQuery, SortBy
from .search_result import SearchResult

__all__ = ['Paper', 'SearchQuery', 'SortBy', 'SearchResult']
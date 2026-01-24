cat > src/ranking/feature_extractor.py << 'EOF'
"""
Feature extraction for scientific paper ranking.
Extracts features like citations, venue prestige, author reputation, recency, etc.
"""
import re
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class VenueType(Enum):
    CONFERENCE = "conference"
    JOURNAL = "journal"
    WORKSHOP = "workshop"
    PREPRINT = "preprint"
    UNKNOWN = "unknown"


@dataclass
class RankingFeatures:
    """Features for ranking scientific papers."""
    # Citation metrics
    citation_count: int = 0
    citation_velocity: float = 0.0  # Citations per year
    h_index_first_author: int = 0
    h_index_last_author: int = 0
    
    # Venue prestige
    venue_impact_factor: float = 0.0
    venue_core_rating: str = "C"  # A*, A, B, C
    venue_acceptance_rate: float = 0.3  # Lower is more prestigious
    
    # Author reputation
    author_count: int = 1
    has_famous_author: bool = False
    author_reputation_score: float = 0.0
    
    # Content quality signals
    has_abstract: bool = True
    has_github: bool = False
    has_dataset: bool = False
    has_arxiv_version: bool = True
    page_count: int = 0
    
    # Temporal features
    days_since_publication: int = 0
    is_recent: bool = False  # Published in last 2 years
    is_very_recent: bool = False  # Published in last 6 months
    
    # Text-based features
    abstract_length: int = 0
    title_length: int = 0
    has_question_in_title: bool = False
    has_colon_in_title: bool = False
    
    # Community engagement
    arxiv_comments: int = 0
    arxiv_version_count: int = 1
    has_been_cited_in_patent: bool = False


class FeatureExtractor:
    """Extract ranking features from scientific papers."""
    
    # Prestigious venues with their impact factors
    PRESTIGIOUS_VENUES = {
        # Conferences
        "neurips": {"type": VenueType.CONFERENCE, "core": "A*", "impact": 4.5},
        "icml": {"type": VenueType.CONFERENCE, "core": "A*", "impact": 4.2},
        "iclr": {"type": VenueType.CONFERENCE, "core": "A*", "impact": 4.0},
        "cvpr": {"type": VenueType.CONFERENCE, "core": "A*", "impact": 4.1},
        "acl": {"type": VenueType.CONFERENCE, "core": "A", "impact": 3.8},
        "emnlp": {"type": VenueType.CONFERENCE, "core": "A", "impact": 3.7},
        "naacl": {"type": VenueType.CONFERENCE, "core": "A", "impact": 3.6},
        "aaai": {"type": VenueType.CONFERENCE, "core": "A", "impact": 3.5},
        "ijcai": {"type": VenueType.CONFERENCE, "core": "A", "impact": 3.4},
        "kdd": {"type": VenueType.CONFERENCE, "core": "A*", "impact": 4.0},
        
        # Journals
        "nature": {"type": VenueType.JOURNAL, "core": "A*", "impact": 42.8},
        "science": {"type": VenueType.JOURNAL, "core": "A*", "impact": 41.8},
        "cell": {"type": VenueType.JOURNAL, "core": "A*", "impact": 36.2},
        "pnas": {"type": VenueType.JOURNAL, "core": "A*", "impact": 9.6},
        "jmlr": {"type": VenueType.JOURNAL, "core": "A*", "impact": 3.5},
        "ieee": {"type": VenueType.JOURNAL, "core": "A", "impact": 2.5},
        "springer": {"type": VenueType.JOURNAL, "core": "B", "impact": 1.8},
    }
    
    # Famous authors in ML/AI (partial list)
    FAMOUS_AUTHORS = {
        "yoshua bengio", "yann lecun", "andrew ng", "geoffrey hinton",
        "ilya sutskever", "demis hassabis", "fei-fei li", "jürgen schmidhuber",
        "stuart russell", "peter norvig", "michael jordan", "christopher manning",
        "joelle pineau", "daphne koller", "pieter abbeel", "sergey levine"
    }
    
    def __init__(self, citation_db=None, author_db=None):
        """
        Initialize feature extractor.
        
        Args:
            citation_db: Optional database for citation counts
            author_db: Optional database for author metrics
        """
        self.citation_db = citation_db
        self.author_db = author_db
    
    def extract_from_paper(self, paper) -> RankingFeatures:
        """
        Extract ranking features from a paper object.
        
        Args:
            paper: Paper object with metadata
            
        Returns:
            RankingFeatures object
        """
        features = RankingFeatures()
        
        # Extract basic features
        features = self._extract_basic_features(features, paper)
        features = self._extract_citation_features(features, paper)
        features = self._extract_venue_features(features, paper)
        features = self._extract_author_features(features, paper)
        features = self._extract_content_features(features, paper)
        features = self._extract_temporal_features(features, paper)
        
        return features
    
    def _extract_basic_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract basic paper features."""
        features.page_count = self._estimate_page_count(paper)
        features.has_abstract = bool(paper.abstract and len(paper.abstract) > 50)
        features.has_arxiv_version = hasattr(paper, 'arxiv_id') and paper.arxiv_id is not None
        
        # Check for GitHub/dataset links
        if hasattr(paper, 'links') and paper.links:
            features.has_github = any('github' in str(link).lower() for link in paper.links)
            features.has_dataset = any(
                'kaggle' in str(link).lower() or 
                'dataset' in str(link).lower() or
                'huggingface' in str(link).lower() 
                for link in paper.links
            )
        
        return features
    
    def _extract_citation_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract citation-related features."""
        if hasattr(paper, 'citation_count'):
            features.citation_count = paper.citation_count or 0
        
        # Calculate citation velocity if we have publication date
        if features.citation_count > 0 and hasattr(paper, 'published'):
            try:
                pub_date = paper.published
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                
                days_since_pub = (datetime.now(pub_date.tzinfo) - pub_date).days
                if days_since_pub > 365:  # At least 1 year old
                    years = days_since_pub / 365.25
                    features.citation_velocity = features.citation_count / years
            except:
                pass
        
        # Check if cited in patent (simple heuristic)
        if hasattr(paper, 'references'):
            features.has_been_cited_in_patent = any(
                'patent' in str(ref).lower() for ref in paper.references
            )
        
        return features
    
    def _extract_venue_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract venue-related features."""
        venue_info = self._get_venue_info(paper)
        
        features.venue_core_rating = venue_info.get('core', 'C')
        features.venue_impact_factor = venue_info.get('impact', 1.0)
        features.venue_acceptance_rate = venue_info.get('acceptance_rate', 0.3)
        
        return features
    
    def _extract_author_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract author-related features."""
        if hasattr(paper, 'authors'):
            features.author_count = len(paper.authors)
            
            # Check for famous authors
            features.has_famous_author = any(
                any(famous in author_name.lower() for famous in self.FAMOUS_AUTHORS)
                for author_name in paper.authors
            )
            
            # Simple author reputation score
            if features.author_count > 0:
                features.author_reputation_score = min(
                    1.0, 
                    features.author_count * 0.1 + (1.0 if features.has_famous_author else 0.0)
                )
        
        return features
    
    def _extract_content_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract content-based features."""
        if hasattr(paper, 'abstract'):
            features.abstract_length = len(paper.abstract) if paper.abstract else 0
        
        if hasattr(paper, 'title'):
            features.title_length = len(paper.title) if paper.title else 0
            title_lower = paper.title.lower() if paper.title else ""
            features.has_question_in_title = '?' in paper.title
            features.has_colon_in_title = ':' in paper.title
        
        return features
    
    def _extract_temporal_features(self, features: RankingFeatures, paper) -> RankingFeatures:
        """Extract temporal features."""
        if hasattr(paper, 'published'):
            try:
                pub_date = paper.published
                if isinstance(pub_date, str):
                    pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                
                days_since = (datetime.now(pub_date.tzinfo) - pub_date).days
                features.days_since_publication = days_since
                features.is_recent = days_since < 730  # 2 years
                features.is_very_recent = days_since < 180  # 6 months
            except:
                pass
        
        return features
    
    def _get_venue_info(self, paper) -> Dict:
        """Get venue information for a paper."""
        venue = ""
        if hasattr(paper, 'venue'):
            venue = paper.venue or ""
        elif hasattr(paper, 'journal_ref'):
            venue = paper.journal_ref or ""
        
        venue_lower = venue.lower()
        
        # Check against known venues
        for venue_key, venue_info in self.PRESTIGIOUS_VENUES.items():
            if venue_key in venue_lower:
                return venue_info
        
        # Default venue info
        return {
            "type": VenueType.UNKNOWN,
            "core": "C",
            "impact": 1.0,
            "acceptance_rate": 0.5
        }
    
    def _estimate_page_count(self, paper) -> int:
        """Estimate page count from PDF or metadata."""
        if hasattr(paper, 'page_count') and paper.page_count:
            return paper.page_count
        
        # Estimate from PDF URL if available
        if hasattr(paper, 'pdf_url') and paper.pdf_url:
            # Simple heuristic: arXiv papers often have specific format
            if 'arxiv' in paper.pdf_url:
                # arXiv papers are typically 8-15 pages
                return 10
        
        return 0
    
    def get_feature_vector(self, features: RankingFeatures) -> np.ndarray:
        """
        Convert features to a normalized vector for ranking.
        
        Returns:
            Normalized feature vector
        """
        # Normalize each feature to [0, 1] range
        vector = [
            # Citation features (log scale for citation count)
            np.log1p(features.citation_count) / 10,  # Assuming max ~22k citations
            min(features.citation_velocity / 100, 1.0),  # Cap at 100 citations/year
            
            # Venue features
            {
                "A*": 1.0, "A": 0.8, "B": 0.6, "C": 0.4
            }.get(features.venue_core_rating, 0.4),
            min(features.venue_impact_factor / 50, 1.0),  # Cap at impact factor 50
            1.0 - features.venue_acceptance_rate,  # Lower acceptance = better
            
            # Author features
            min(features.author_count / 10, 1.0),  # Cap at 10 authors
            1.0 if features.has_famous_author else 0.0,
            features.author_reputation_score,
            
            # Content quality
            1.0 if features.has_abstract else 0.0,
            1.0 if features.has_github else 0.0,
            1.0 if features.has_dataset else 0.0,
            1.0 if features.has_arxiv_version else 0.0,
            min(features.page_count / 50, 1.0),  # Cap at 50 pages
            
            # Temporal features
            1.0 - min(features.days_since_publication / 3650, 1.0),  # Decay over 10 years
            1.0 if features.is_recent else 0.0,
            1.0 if features.is_very_recent else 0.0,
            
            # Text features
            min(features.abstract_length / 2000, 1.0),  # Cap at 2000 chars
            min(features.title_length / 200, 1.0),  # Cap at 200 chars
            1.0 if features.has_question_in_title else 0.0,
            1.0 if features.has_colon_in_title else 0.0,
            
            # Community engagement
            min(features.arxiv_comments / 100, 1.0),  # Cap at 100 comments
            min(features.arxiv_version_count / 5, 1.0),  # Cap at 5 versions
            1.0 if features.has_been_cited_in_patent else 0.0,
        ]
        
        return np.array(vector)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the vector."""
        return [
            "log_citation_count",
            "citation_velocity",
            "venue_core_rating",
            "venue_impact_factor",
            "venue_selectivity",
            "author_count",
            "has_famous_author",
            "author_reputation",
            "has_abstract",
            "has_github",
            "has_dataset",
            "has_arxiv_version",
            "page_count",
            "recency",
            "is_recent",
            "is_very_recent",
            "abstract_length",
            "title_length",
            "has_question_in_title",
            "has_colon_in_title",
            "arxiv_comments",
            "arxiv_versions",
            "cited_in_patent",
        ]
EOF
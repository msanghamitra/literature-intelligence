# src/ranking/feature_extractor.py
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

# src/ranking/citation_ranker.py
cat > src/ranking/citation_ranker.py << 'EOF'
"""
Citation-based ranking algorithms for scientific papers.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import networkx as nx


@dataclass
class CitationRank:
    """Citation ranking result."""
    paper_id: str
    citation_count: int
    pagerank_score: float
    hub_score: float
    authority_score: float
    freshness_weighted_citations: float
    normalized_score: float


class CitationRanker:
    """
    Advanced citation-based ranking using network analysis.
    
    Implements:
    1. PageRank algorithm on citation network
    2. HITS algorithm (Hub and Authority scores)
    3. Time-weighted citation counts
    4. Citation velocity analysis
    """
    
    def __init__(self, alpha: float = 0.85, max_iterations: int = 100):
        """
        Initialize citation ranker.
        
        Args:
            alpha: Damping factor for PageRank (probability of following links)
            max_iterations: Maximum iterations for convergence
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.citation_graph = nx.DiGraph()
    
    def build_citation_network(self, papers: List) -> nx.DiGraph:
        """
        Build citation network from papers.
        
        Args:
            papers: List of paper objects with citation/reference information
            
        Returns:
            Directed citation graph
        """
        graph = nx.DiGraph()
        
        for paper in papers:
            paper_id = self._get_paper_id(paper)
            if not paper_id:
                continue
            
            # Add paper as node with metadata
            graph.add_node(paper_id, paper=paper)
            
            # Add citation edges if references are available
            if hasattr(paper, 'references') and paper.references:
                for ref_id in paper.references:
                    # Add reference node if it doesn't exist
                    if not graph.has_node(ref_id):
                        graph.add_node(ref_id)
                    
                    # Add citation edge (paper cites ref_id)
                    graph.add_edge(paper_id, ref_id)
        
        self.citation_graph = graph
        return graph
    
    def calculate_pagerank(self, papers: List = None) -> Dict[str, float]:
        """
        Calculate PageRank scores for papers.
        
        Args:
            papers: Optional list of papers, uses cached graph if None
            
        Returns:
            Dictionary mapping paper_id to PageRank score
        """
        if papers is not None:
            self.build_citation_network(papers)
        
        if len(self.citation_graph) == 0:
            return {}
        
        # Calculate PageRank
        pagerank_scores = nx.pagerank(
            self.citation_graph, 
            alpha=self.alpha, 
            max_iter=self.max_iterations
        )
        
        return pagerank_scores
    
    def calculate_hits_scores(self, papers: List = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate HITS (Hub and Authority) scores.
        
        Args:
            papers: Optional list of papers, uses cached graph if None
            
        Returns:
            Tuple of (hub_scores, authority_scores)
        """
        if papers is not None:
            self.build_citation_network(papers)
        
        if len(self.citation_graph) == 0:
            return {}, {}
        
        # Calculate HITS scores
        hubs, authorities = nx.hits(self.citation_graph, max_iter=self.max_iterations)
        
        return hubs, authorities
    
    def calculate_freshness_weighted_citations(self, papers: List) -> Dict[str, float]:
        """
        Calculate time-weighted citation counts.
        
        Recent citations are weighted more heavily than old citations.
        
        Args:
            papers: List of paper objects
            
        Returns:
            Dictionary mapping paper_id to freshness-weighted citation score
        """
        scores = {}
        
        for paper in papers:
            paper_id = self._get_paper_id(paper)
            if not paper_id:
                continue
            
            if hasattr(paper, 'citation_dates') and paper.citation_dates:
                # Calculate weighted sum based on citation recency
                total_weight = 0.0
                current_year = datetime.now().year
                
                for citation_date in paper.citation_dates:
                    try:
                        if isinstance(citation_date, str):
                            citation_year = int(citation_date[:4])
                        else:
                            citation_year = citation_date.year
                        
                        # Exponential decay: citations decay by 10% per year
                        years_ago = current_year - citation_year
                        weight = 0.9 ** years_ago
                        total_weight += weight
                    except:
                        total_weight += 1.0  # Default weight if date parsing fails
                
                scores[paper_id] = total_weight
            else:
                # Fallback to simple citation count
                citation_count = getattr(paper, 'citation_count', 0)
                scores[paper_id] = float(citation_count)
        
        return scores
    
    def calculate_citation_velocity(self, papers: List) -> Dict[str, float]:
        """
        Calculate citation velocity (citations per year).
        
        Args:
            papers: List of paper objects
            
        Returns:
            Dictionary mapping paper_id to citation velocity
        """
        velocities = {}
        
        for paper in papers:
            paper_id = self._get_paper_id(paper)
            if not paper_id:
                continue
            
            citation_count = getattr(paper, 'citation_count', 0)
            
            if hasattr(paper, 'published') and paper.published:
                try:
                    pub_date = paper.published
                    if isinstance(pub_date, str):
                        pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    
                    days_since_pub = (datetime.now(pub_date.tzinfo) - pub_date).days
                    
                    if days_since_pub > 0:
                        years = days_since_pub / 365.25
                        velocity = citation_count / years
                        velocities[paper_id] = velocity
                    else:
                        velocities[paper_id] = 0.0
                except:
                    velocities[paper_id] = 0.0
            else:
                velocities[paper_id] = 0.0
        
        return velocities
    
    def rank_papers(self, papers: List, weights: Dict[str, float] = None) -> List[CitationRank]:
        """
        Rank papers using multiple citation metrics.
        
        Args:
            papers: List of paper objects
            weights: Dictionary of metric weights. Default weights:
                {
                    "pagerank": 0.3,
                    "authority": 0.2,
                    "freshness": 0.25,
                    "velocity": 0.15,
                    "count": 0.1
                }
            
        Returns:
            List of CitationRank objects sorted by normalized score
        """
        if weights is None:
            weights = {
                "pagerank": 0.3,
                "authority": 0.2,
                "freshness": 0.25,
                "velocity": 0.15,
                "count": 0.1
            }
        
        # Calculate all metrics
        pagerank_scores = self.calculate_pagerank(papers)
        hub_scores, authority_scores = self.calculate_hits_scores(papers)
        freshness_scores = self.calculate_freshness_weighted_citations(papers)
        velocity_scores = self.calculate_citation_velocity(papers)
        
        # Normalize each metric to [0, 1] range
        def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            max_score = max(scores.values())
            if max_score > 0:
                return {k: v / max_score for k, v in scores.items()}
            return {k: 0.0 for k in scores.keys()}
        
        norm_pagerank = normalize_scores(pagerank_scores)
        norm_authority = normalize_scores(authority_scores)
        norm_freshness = normalize_scores(freshness_scores)
        norm_velocity = normalize_scores(velocity_scores)
        
        # Calculate combined scores
        results = []
        for paper in papers:
            paper_id = self._get_paper_id(paper)
            if not paper_id:
                continue
            
            citation_count = getattr(paper, 'citation_count', 0)
            
            # Get normalized scores (default to 0 if not available)
            pagerank = norm_pagerank.get(paper_id, 0.0)
            authority = norm_authority.get(paper_id, 0.0)
            hub = hub_scores.get(paper_id, 0.0)
            freshness = norm_freshness.get(paper_id, 0.0)
            velocity = norm_velocity.get(paper_id, 0.0)
            
            # Calculate weighted combination
            normalized_score = (
                weights["pagerank"] * pagerank +
                weights["authority"] * authority +
                weights["freshness"] * freshness +
                weights["velocity"] * velocity +
                weights["count"] * (min(citation_count, 1000) / 1000)  # Cap at 1000 citations
            )
            
            result = CitationRank(
                paper_id=paper_id,
                citation_count=citation_count,
                pagerank_score=pagerank,
                hub_score=hub,
                authority_score=authority,
                freshness_weighted_citations=freshness_scores.get(paper_id, 0.0),
                normalized_score=normalized_score
            )
            results.append(result)
        
        # Sort by normalized score (descending)
        results.sort(key=lambda x: x.normalized_score, reverse=True)
        
        return results
    
    def get_citation_network_metrics(self) -> Dict:
        """
        Calculate network-level metrics for the citation graph.
        
        Returns:
            Dictionary of network metrics
        """
        if len(self.citation_graph) == 0:
            return {}
        
        metrics = {
            "num_nodes": self.citation_graph.number_of_nodes(),
            "num_edges": self.citation_graph.number_of_edges(),
            "density": nx.density(self.citation_graph),
            "average_clustering": nx.average_clustering(self.citation_graph.to_undirected()),
            "is_weakly_connected": nx.is_weakly_connected(self.citation_graph),
            "is_strongly_connected": nx.is_strongly_connected(self.citation_graph),
        }
        
        # Try to calculate more metrics if graph is not empty
        try:
            if metrics["num_nodes"] > 0:
                metrics["average_shortest_path_length"] = nx.average_shortest_path_length(
                    self.citation_graph.to_undirected()
                )
        except:
            metrics["average_shortest_path_length"] = float('inf')
        
        # Calculate degree statistics
        if metrics["num_nodes"] > 0:
            in_degrees = [d for n, d in self.citation_graph.in_degree()]
            out_degrees = [d for n, d in self.citation_graph.out_degree()]
            
            metrics.update({
                "avg_in_degree": np.mean(in_degrees),
                "avg_out_degree": np.mean(out_degrees),
                "max_in_degree": max(in_degrees),
                "max_out_degree": max(out_degrees),
            })
        
        return metrics
    
    def _get_paper_id(self, paper) -> Optional[str]:
        """Extract unique paper ID from paper object."""
        if hasattr(paper, 'arxiv_id') and paper.arxiv_id:
            return f"arxiv:{paper.arxiv_id}"
        elif hasattr(paper, 'doi') and paper.doi:
            return f"doi:{paper.doi}"
        elif hasattr(paper, 'id') and paper.id:
            return str(paper.id)
        elif hasattr(paper, 'title') and paper.title:
            return f"title:{hash(paper.title)}"
        return None


class SimpleCitationRanker:
    """
    Simplified citation ranker for when full network analysis isn't needed.
    """
    
    @staticmethod
    def rank_by_citation_count(papers: List, normalize: bool = True) -> List[Tuple]:
        """
        Rank papers by citation count.
        
        Args:
            papers: List of paper objects
            normalize: Whether to normalize scores to [0, 1]
            
        Returns:
            List of (paper, score) tuples sorted by score
        """
        scored_papers = []
        
        for paper in papers:
            citation_count = getattr(paper, 'citation_count', 0)
            scored_papers.append((paper, citation_count))
        
        # Sort by citation count (descending)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize if requested
        if normalize and scored_papers:
            max_score = max(score for _, score in scored_papers)
            if max_score > 0:
                scored_papers = [(paper, score / max_score) 
                                for paper, score in scored_papers]
        
        return scored_papers
    
    @staticmethod
    def rank_by_citation_velocity(papers: List) -> List[Tuple]:
        """
        Rank papers by citation velocity (citations per year).
        
        Args:
            papers: List of paper objects
            
        Returns:
            List of (paper, velocity) tuples sorted by velocity
        """
        velocities = []
        current_year = datetime.now().year
        
        for paper in papers:
            citation_count = getattr(paper, 'citation_count', 0)
            
            if hasattr(paper, 'published') and paper.published:
                try:
                    pub_date = paper.published
                    if isinstance(pub_date, str):
                        pub_year = int(pub_date[:4])
                    else:
                        pub_year = pub_date.year
                    
                    years_since_pub = max(1, current_year - pub_year)
                    velocity = citation_count / years_since_pub
                    velocities.append((paper, velocity))
                except:
                    velocities.append((paper, 0.0))
            else:
                velocities.append((paper, 0.0))
        
        # Sort by velocity (descending)
        velocities.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize
        max_velocity = max(score for _, score in velocities) if velocities else 1
        if max_velocity > 0:
            velocities = [(paper, score / max_velocity) 
                         for paper, score in velocities]
        
        return velocities
EOF

# src/ranking/venue_ranker.py
cat > src/ranking/venue_ranker.py << 'EOF'
"""
Venue-based ranking for scientific papers.
Rank papers based on conference/journal prestige.
"""
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml


class VenueTier(Enum):
    """Tier system for academic venues."""
    TIER_1 = "tier_1"      # Top-tier (e.g., NeurIPS, ICML, Nature, Science)
    TIER_2 = "tier_2"      # Strong conferences/journals
    TIER_3 = "tier_3"      # Good conferences/journals
    TIER_4 = "tier_4"      # Average conferences/journals
    WORKSHOP = "workshop"  # Workshops
    PREPRINT = "preprint"  # arXiv, bioRxiv, etc.
    UNKNOWN = "unknown"


@dataclass
class VenueScore:
    """Venue scoring result."""
    venue_name: str
    normalized_name: str
    tier: VenueTier
    score: float
    impact_factor: float
    acceptance_rate: float
    h5_index: float
    core_rating: str  # A*, A, B, C
    is_top_conference: bool
    is_top_journal: bool


class VenueRanker:
    """
    Rank papers based on venue prestige.
    
    Uses multiple sources of venue reputation:
    1. CORE Conference Ranking
    2. Google Scholar Metrics
    3. Journal Impact Factors
    4. Community perception
    """
    
    def __init__(self, venue_db_path: Optional[str] = None):
        """
        Initialize venue ranker.
        
        Args:
            venue_db_path: Path to YAML file with venue database
        """
        self.venue_db = self._load_venue_database(venue_db_path)
        self._build_venue_patterns()
    
    def _load_venue_database(self, db_path: Optional[str]) -> Dict:
        """Load venue database from YAML file or use built-in."""
        if db_path:
            try:
                with open(db_path, 'r') as f:
                    return yaml.safe_load(f)
            except:
                pass
        
        # Built-in venue database
        return {
            "conferences": {
                # Tier 1: Top-tier ML/AI conferences
                "neurips": {
                    "full_name": "Conference on Neural Information Processing Systems",
                    "tier": "tier_1",
                    "core": "A*",
                    "h5_index": 222,
                    "acceptance_rate": 0.21,
                    "is_top": True
                },
                "icml": {
                    "full_name": "International Conference on Machine Learning",
                    "tier": "tier_1",
                    "core": "A*",
                    "h5_index": 189,
                    "acceptance_rate": 0.22,
                    "is_top": True
                },
                "iclr": {
                    "full_name": "International Conference on Learning Representations",
                    "tier": "tier_1",
                    "core": "A*",
                    "h5_index": 156,
                    "acceptance_rate": 0.25,
                    "is_top": True
                },
                "cvpr": {
                    "full_name": "Conference on Computer Vision and Pattern Recognition",
                    "tier": "tier_1",
                    "core": "A*",
                    "h5_index": 218,
                    "acceptance_rate": 0.22,
                    "is_top": True
                },
                "acl": {
                    "full_name": "Annual Meeting of the Association for Computational Linguistics",
                    "tier": "tier_1",
                    "core": "A*",
                    "h5_index": 121,
                    "acceptance_rate": 0.21,
                    "is_top": True
                },
                
                # Tier 2: Strong conferences
                "aaai": {
                    "full_name": "AAAI Conference on Artificial Intelligence",
                    "tier": "tier_2",
                    "core": "A",
                    "h5_index": 98,
                    "acceptance_rate": 0.15,
                    "is_top": False
                },
                "ijcai": {
                    "full_name": "International Joint Conference on Artificial Intelligence",
                    "tier": "tier_2",
                    "core": "A",
                    "h5_index": 76,
                    "acceptance_rate": 0.14,
                    "is_top": False
                },
                "emnlp": {
                    "full_name": "Conference on Empirical Methods in Natural Language Processing",
                    "tier": "tier_2",
                    "core": "A",
                    "h5_index": 115,
                    "acceptance_rate": 0.23,
                    "is_top": False
                },
                "kdd": {
                    "full_name": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
                    "tier": "tier_2",
                    "core": "A*",
                    "h5_index": 95,
                    "acceptance_rate": 0.16,
                    "is_top": True
                },
                
                # Tier 3: Good conferences
                "iccv": {
                    "full_name": "International Conference on Computer Vision",
                    "tier": "tier_3",
                    "core": "A",
                    "h5_index": 102,
                    "acceptance_rate": 0.25,
                    "is_top": False
                },
                "eccv": {
                    "full_name": "European Conference on Computer Vision",
                    "tier": "tier_3",
                    "core": "A",
                    "h5_index": 89,
                    "acceptance_rate": 0.26,
                    "is_top": False
                },
                "naacl": {
                    "full_name": "North American Chapter of the Association for Computational Linguistics",
                    "tier": "tier_3",
                    "core": "A",
                    "h5_index": 45,
                    "acceptance_rate": 0.28,
                    "is_top": False
                },
            },
            
            "journals": {
                # Tier 1: Top journals
                "nature": {
                    "full_name": "Nature",
                    "tier": "tier_1",
                    "impact_factor": 42.78,
                    "core": "A*",
                    "is_top": True
                },
                "science": {
                    "full_name": "Science",
                    "tier": "tier_1",
                    "impact_factor": 41.84,
                    "core": "A*",
                    "is_top": True
                },
                "cell": {
                    "full_name": "Cell",
                    "tier": "tier_1",
                    "impact_factor": 36.22,
                    "core": "A*",
                    "is_top": True
                },
                "pnas": {
                    "full_name": "Proceedings of the National Academy of Sciences",
                    "tier": "tier_1",
                    "impact_factor": 9.58,
                    "core": "A*",
                    "is_top": True
                },
                
                # ML/AI specific journals
                "jmlr": {
                    "full_name": "Journal of Machine Learning Research",
                    "tier": "tier_1",
                    "impact_factor": 3.5,
                    "core": "A*",
                    "is_top": True
                },
                "tplp": {
                    "full_name": "Transactions on Pattern Analysis and Machine Intelligence",
                    "tier": "tier_1",
                    "impact_factor": 17.86,
                    "core": "A*",
                    "is_top": True
                },
                "ijcv": {
                    "full_name": "International Journal of Computer Vision",
                    "tier": "tier_2",
                    "impact_factor": 7.41,
                    "core": "A",
                    "is_top": False
                },
                "cl": {
                    "full_name": "Computational Linguistics",
                    "tier": "tier_2",
                    "impact_factor": 2.65,
                    "core": "A",
                    "is_top": False
                },
            },
            
            "workshops": {
                # Common workshop prefixes/suffixes
                "_workshop": {"tier": "workshop", "score": 0.3},
                "_w": {"tier": "workshop", "score": 0.3},
                "workshop_": {"tier": "workshop", "score": 0.3},
                "wacv": {"tier": "workshop", "score": 0.4},
            },
            
            "preprints": {
                "arxiv": {"tier": "preprint", "score": 0.5},
                "biorxiv": {"tier": "preprint", "score": 0.5},
                "medrxiv": {"tier": "preprint", "score": 0.5},
                "ssrn": {"tier": "preprint", "score": 0.4},
                "chemrxiv": {"tier": "preprint", "score": 0.4},
            }
        }
    
    def _build_venue_patterns(self):
        """Build regex patterns for venue matching."""
        self.conference_patterns = {}
        self.journal_patterns = {}
        
        # Build patterns for conferences
        for conf_key, conf_info in self.venue_db.get("conferences", {}).items():
            patterns = [
                re.compile(rf'\b{conf_key}\b', re.IGNORECASE),
                re.compile(rf'\b{conf_info.get("full_name", "")}\b', re.IGNORECASE)
            ]
            self.conference_patterns[conf_key] = {
                "patterns": patterns,
                "info": conf_info
            }
        
        # Build patterns for journals
        for journal_key, journal_info in self.venue_db.get("journals", {}).items():
            patterns = [
                re.compile(rf'\b{journal_key}\b', re.IGNORECASE),
                re.compile(rf'\b{journal_info.get("full_name", "")}\b', re.IGNORECASE)
            ]
            self.journal_patterns[journal_key] = {
                "patterns": patterns,
                "info": journal_info
            }
    
    def extract_venue_from_text(self, text: str) -> Optional[str]:
        """
        Extract venue name from text.
        
        Args:
            text: Text containing venue information
            
        Returns:
            Extracted venue name or None
        """
        if not text:
            return None
        
        # Clean text
        text_lower = text.lower().strip()
        
        # Check for arXiv
        if 'arxiv' in text_lower:
            return 'arxiv'
        
        # Check for known conferences
        for conf_key, conf_data in self.conference_patterns.items():
            for pattern in conf_data["patterns"]:
                if pattern.search(text_lower):
                    return conf_key
        
        # Check for known journals
        for journal_key, journal_data in self.journal_patterns.items():
            for pattern in journal_data["patterns"]:
                if pattern.search(text_lower):
                    return journal_key
        
        # Try to extract using common patterns
        patterns = [
            r'proceedings of (?:the )?(.+)',
            r'(.+) conference',
            r'(.+) symposium',
            r'in (?:the )?(.+):',
            r'journal of (.+)',
            r'(.+) journal',
            r'(.+) transactions',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                venue = match.group(1).strip()
                if len(venue) > 3:  # Avoid very short matches
                    return venue
        
        return None
    
    def score_venue(self, venue_name: str) -> VenueScore:
        """
        Score a venue based on its name.
        
        Args:
            venue_name: Name of the venue
            
        Returns:
            VenueScore object
        """
        if not venue_name:
            return self._create_unknown_venue_score()
        
        venue_lower = venue_name.lower().strip()
        
        # Check for preprints
        for preprint_key, preprint_info in self.venue_db.get("preprints", {}).items():
            if preprint_key in venue_lower:
                return VenueScore(
                    venue_name=venue_name,
                    normalized_name=preprint_key,
                    tier=VenueTier.PREPRINT,
                    score=preprint_info.get("score", 0.5),
                    impact_factor=0.0,
                    acceptance_rate=1.0,
                    h5_index=0.0,
                    core_rating="C",
                    is_top_conference=False,
                    is_top_journal=False
                )
        
        # Check for workshops
        for workshop_key, workshop_info in self.venue_db.get("workshops", {}).items():
            if workshop_key in venue_lower:
                return VenueScore(
                    venue_name=venue_name,
                    normalized_name=workshop_key,
                    tier=VenueTier.WORKSHOP,
                    score=workshop_info.get("score", 0.3),
                    impact_factor=0.0,
                    acceptance_rate=0.5,
                    h5_index=0.0,
                    core_rating="C",
                    is_top_conference=False,
                    is_top_journal=False
                )
        
        # Check conferences
        for conf_key, conf_data in self.conference_patterns.items():
            for pattern in conf_data["patterns"]:
                if pattern.search(venue_lower):
                    info = conf_data["info"]
                    return VenueScore(
                        venue_name=venue_name,
                        normalized_name=conf_key,
                        tier=VenueTier(info["tier"]),
                        score=self._tier_to_score(info["tier"]),
                        impact_factor=0.0,
                        acceptance_rate=info.get("acceptance_rate", 0.25),
                        h5_index=info.get("h5_index", 0.0),
                        core_rating=info.get("core", "C"),
                        is_top_conference=info.get("is_top", False),
                        is_top_journal=False
                    )
        
        # Check journals
        for journal_key, journal_data in self.journal_patterns.items():
            for pattern in journal_data["patterns"]:
                if pattern.search(venue_lower):
                    info = journal_data["info"]
                    return VenueScore(
                        venue_name=venue_name,
                        normalized_name=journal_key,
                        tier=VenueTier(info["tier"]),
                        score=self._tier_to_score(info["tier"]),
                        impact_factor=info.get("impact_factor", 0.0),
                        acceptance_rate=0.2,  # Typical journal acceptance rate
                        h5_index=0.0,
                        core_rating=info.get("core", "C"),
                        is_top_conference=False,
                        is_top_journal=info.get("is_top", False)
                    )
        
        # Unknown venue
        return self._create_unknown_venue_score(venue_name)
    
    def _tier_to_score(self, tier: str) -> float:
        """Convert tier to numerical score."""
        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.8,
            "tier_3": 0.6,
            "tier_4": 0.4,
            "workshop": 0.3,
            "preprint": 0.5,
            "unknown": 0.2
        }
        return tier_scores.get(tier, 0.2)
    
    def _create_unknown_venue_score(self, venue_name: str = "Unknown") -> VenueScore:
        """Create score for unknown venue."""
        return VenueScore(
            venue_name=venue_name,
            normalized_name="unknown",
            tier=VenueTier.UNKNOWN,
            score=0.2,
            impact_factor=0.0,
            acceptance_rate=0.5,
            h5_index=0.0,
            core_rating="C",
            is_top_conference=False,
            is_top_journal=False
        )
    
    def rank_papers_by_venue(self, papers: List) -> List[Tuple]:
        """
        Rank papers by venue prestige.
        
        Args:
            papers: List of paper objects
            
        Returns:
            List of (paper, venue_score) tuples sorted by venue score
        """
        scored_papers = []
        
        for paper in papers:
            # Extract venue from paper
            venue = self._get_paper_venue(paper)
            venue_score = self.score_venue(venue)
            
            scored_papers.append((paper, venue_score))
        
        # Sort by venue score (descending)
        scored_papers.sort(key=lambda x: x[1].score, reverse=True)
        
        return scored_papers
    
    def get_venue_statistics(self, papers: List) -> Dict:
        """
        Get statistics about venues in paper list.
        
        Args:
            papers: List of paper objects
            
        Returns:
            Dictionary of venue statistics
        """
        venue_counts = {}
        tier_counts = {tier.value: 0 for tier in VenueTier}
        scores = []
        
        for paper in papers:
            venue = self._get_paper_venue(paper)
            venue_score = self.score_venue(venue)
            
            # Count venues
            norm_name = venue_score.normalized_name
            venue_counts[norm_name] = venue_counts.get(norm_name, 0) + 1
            
            # Count tiers
            tier_counts[venue_score.tier.value] += 1
            
            # Collect scores
            scores.append(venue_score.score)
        
        # Calculate statistics
        total_papers = len(papers)
        if total_papers == 0:
            return {}
        
        stats = {
            "total_papers": total_papers,
            "unique_venues": len(venue_counts),
            "venue_distribution": dict(sorted(venue_counts.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10]),  # Top 10
            "tier_distribution": tier_counts,
            "avg_venue_score": sum(scores) / total_papers if scores else 0,
            "top_venue": max(venue_counts.items(), key=lambda x: x[1])[0] if venue_counts else "None",
            "papers_in_top_tier": tier_counts.get("tier_1", 0),
            "papers_in_preprint": tier_counts.get("preprint", 0),
        }
        
        return stats
    
    def _get_paper_venue(self, paper) -> str:
        """Extract venue from paper object."""
        # Try different possible venue fields
        venue_fields = ['venue', 'journal_ref', 'conference', 'journal', 
                       'booktitle', 'series', 'publisher']
        
        for field in venue_fields:
            if hasattr(paper, field):
                venue = getattr(paper, field)
                if venue and str(venue).strip():
                    return str(venue).strip()
        
        # Check arXiv ID
        if hasattr(paper, 'arxiv_id') and paper.arxiv_id:
            return 'arxiv'
        
        return ""


class HybridVenueRanker:
    """
    Hybrid venue ranker that combines venue prestige with paper-specific factors.
    """
    
    def __init__(self, venue_ranker: VenueRanker):
        """
        Initialize hybrid ranker.
        
        Args:
            venue_ranker: Base venue ranker
        """
        self.venue_ranker = venue_ranker
    
    def rank_papers(self, papers: List, 
                    venue_weight: float = 0.6,
                    citation_weight: float = 0.3,
                    recency_weight: float = 0.1) -> List[Tuple]:
        """
        Rank papers using hybrid scoring.
        
        Args:
            papers: List of paper objects
            venue_weight: Weight for venue score (0-1)
            citation_weight: Weight for citation score (0-1)
            recency_weight: Weight for recency score (0-1)
            
        Returns:
            List of (paper, hybrid_score) tuples sorted by score
        """
        # Normalize weights
        total_weight = venue_weight + citation_weight + recency_weight
        venue_weight /= total_weight
        citation_weight /= total_weight
        recency_weight /= total_weight
        
        scored_papers = []
        
        for paper in papers:
            # Get venue score
            venue = self.venue_ranker._get_paper_venue(paper)
            venue_score_obj = self.venue_ranker.score_venue(venue)
            venue_score = venue_score_obj.score
            
            # Get citation score (normalized)
            citation_count = getattr(paper, 'citation_count', 0)
            citation_score = min(citation_count / 1000, 1.0)  # Cap at 1000 citations
            
            # Get recency score
            recency_score = self._calculate_recency_score(paper)
            
            # Calculate hybrid score
            hybrid_score = (
                venue_weight * venue_score +
                citation_weight * citation_score +
                recency_weight * recency_score
            )
            
            scored_papers.append((paper, hybrid_score, {
                "venue_score": venue_score,
                "citation_score": citation_score,
                "recency_score": recency_score,
                "venue_tier": venue_score_obj.tier.value
            }))
        
        # Sort by hybrid score (descending)
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        return scored_papers
    
    def _calculate_recency_score(self, paper) -> float:
        """Calculate recency score for paper."""
        if not hasattr(paper, 'published') or not paper.published:
            return 0.5
        
        try:
            from datetime import datetime
            pub_date = paper.published
            if isinstance(pub_date, str):
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            
            days_since = (datetime.now(pub_date.tzinfo) - pub_date).days
            
            # Exponential decay: papers lose half their recency value every 2 years
            years = days_since / 365.25
            recency = 0.5 ** (years / 2)  # Half-life of 2 years
            
            return min(max(recency, 0.0), 1.0)
        except:
            return 0.5


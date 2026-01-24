
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

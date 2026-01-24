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
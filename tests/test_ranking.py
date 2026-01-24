"""
Tests for ranking module.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ranking.feature_extractor import FeatureExtractor, RankingFeatures
from src.ranking.citation_ranker import CitationRanker, SimpleCitationRanker
from src.ranking.venue_ranker import VenueRanker, HybridVenueRanker, VenueTier


class TestRankingFeatures:
    """Test RankingFeatures dataclass."""
    
    def test_default_values(self):
        """Test default values of RankingFeatures."""
        features = RankingFeatures()
        
        assert features.citation_count == 0
        assert features.venue_core_rating == "C"
        assert features.author_count == 1
        assert features.has_abstract == True
        assert features.is_recent == False
    
    def test_custom_values(self):
        """Test custom values for RankingFeatures."""
        features = RankingFeatures(
            citation_count=100,
            venue_core_rating="A*",
            author_count=3,
            has_github=True,
            is_recent=True
        )
        
        assert features.citation_count == 100
        assert features.venue_core_rating == "A*"
        assert features.author_count == 3
        assert features.has_github == True
        assert features.is_recent == True


class TestFeatureExtractor:
    """Test FeatureExtractor class."""
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        assert extractor.citation_db is None
        assert extractor.author_db is None
    
    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        
        # Create a mock paper
        paper = Mock()
        paper.abstract = "This is a test abstract for a scientific paper."
        paper.arxiv_id = "1234.56789"
        paper.links = ["https://github.com/test/repo", "https://arxiv.org/abs/1234.56789"]
        
        features = RankingFeatures()
        features = extractor._extract_basic_features(features, paper)
        
        assert features.has_abstract == True
        assert features.has_arxiv_version == True
        assert features.has_github == True
        assert features.has_dataset == False
    
    def test_extract_citation_features(self):
        """Test citation feature extraction."""
        extractor = FeatureExtractor()
        
        # Create a mock paper with citations
        paper = Mock()
        paper.citation_count = 50
        paper.published = datetime(2020, 1, 1)
        paper.references = []
        
        features = RankingFeatures()
        features = extractor._extract_citation_features(features, paper)
        
        assert features.citation_count == 50
        assert features.citation_velocity > 0
        assert features.has_been_cited_in_patent == False
    
    def test_extract_venue_features(self):
        """Test venue feature extraction."""
        extractor = FeatureExtractor()
        
        # Create mock papers for different venues
        test_cases = [
            ("NeurIPS 2023", "A*", 4.5),
            ("ICML 2023", "A*", 4.2),
            ("arXiv preprint", "C", 1.0),  # Unknown venue
        ]
        
        for venue_name, expected_core, expected_impact in test_cases:
            paper = Mock()
            paper.venue = venue_name
            
            features = RankingFeatures()
            features = extractor._extract_venue_features(features, paper)
            
            assert features.venue_core_rating == expected_core
            assert features.venue_impact_factor == expected_impact
    
    def test_extract_author_features(self):
        """Test author feature extraction."""
        extractor = FeatureExtractor()
        
        # Test with famous author
        paper = Mock()
        paper.authors = ["Yoshua Bengio", "John Doe", "Jane Smith"]
        
        features = RankingFeatures()
        features = extractor._extract_author_features(features, paper)
        
        assert features.author_count == 3
        assert features.has_famous_author == True
        assert features.author_reputation_score > 0
        
        # Test without famous author
        paper.authors = ["John Doe", "Jane Smith"]
        
        features = RankingFeatures()
        features = extractor._extract_author_features(features, paper)
        
        assert features.has_famous_author == False
    
    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        extractor = FeatureExtractor()
        
        # Recent paper
        recent_paper = Mock()
        recent_paper.published = datetime.now() - timedelta(days=100)
        
        features = RankingFeatures()
        features = extractor._extract_temporal_features(features, recent_paper)
        
        assert features.is_recent == True
        assert features.is_very_recent == True
        
        # Old paper
        old_paper = Mock()
        old_paper.published = datetime.now() - timedelta(days=1000)
        
        features = RankingFeatures()
        features = extractor._extract_temporal_features(features, old_paper)
        
        assert features.is_recent == False
        assert features.is_very_recent == False
    
    def test_get_feature_vector(self):
        """Test feature vector generation."""
        extractor = FeatureExtractor()
        
        # Create features
        features = RankingFeatures(
            citation_count=100,
            citation_velocity=20.5,
            venue_core_rating="A*",
            venue_impact_factor=4.5,
            venue_acceptance_rate=0.2,
            author_count=3,
            has_famous_author=True,
            author_reputation_score=0.8,
            has_abstract=True,
            has_github=True,
            has_dataset=False,
            has_arxiv_version=True,
            page_count=15,
            days_since_publication=365,
            is_recent=True,
            is_very_recent=False,
            abstract_length=500,
            title_length=100,
            has_question_in_title=False,
            has_colon_in_title=True,
            arxiv_comments=10,
            arxiv_version_count=2,
            has_been_cited_in_patent=False
        )
        
        vector = extractor.get_feature_vector(features)
        
        # Check vector properties
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(extractor.get_feature_names())
        assert all(0 <= x <= 1 for x in vector)
    
    def test_get_feature_names(self):
        """Test feature names generation."""
        extractor = FeatureExtractor()
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert "log_citation_count" in names
        assert "venue_core_rating" in names
        assert "has_famous_author" in names


class TestCitationRanker:
    """Test CitationRanker class."""
    
    def test_initialization(self):
        """Test CitationRanker initialization."""
        ranker = CitationRanker()
        assert ranker.alpha == 0.85
        assert ranker.max_iterations == 100
    
    def test_build_citation_network(self):
        """Test citation network building."""
        ranker = CitationRanker()
        
        # Create mock papers with citations
        paper1 = Mock()
        paper1.arxiv_id = "1234.56789"
        paper1.references = []
        
        paper2 = Mock()
        paper2.arxiv_id = "9876.54321"
        paper2.references = ["1234.56789"]  # Cites paper1
        
        papers = [paper1, paper2]
        
        graph = ranker.build_citation_network(papers)
        
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert graph.has_edge("arxiv:9876.54321", "arxiv:1234.56789")
    
    def test_calculate_pagerank(self):
        """Test PageRank calculation."""
        ranker = CitationRanker()
        
        # Build a simple citation network
        paper1 = Mock()
        paper1.arxiv_id = "paper1"
        paper1.references = []
        
        paper2 = Mock()
        paper2.arxiv_id = "paper2"
        paper2.references = ["paper1"]
        
        paper3 = Mock()
        paper3.arxiv_id = "paper3"
        paper3.references = ["paper1", "paper2"]
        
        papers = [paper1, paper2, paper3]
        
        pagerank_scores = ranker.calculate_pagerank(papers)
        
        assert len(pagerank_scores) == 3
        assert all(0 <= score <= 1 for score in pagerank_scores.values())
        
        # Paper1 should have highest PageRank (cited by both paper2 and paper3)
        assert pagerank_scores["arxiv:paper1"] > pagerank_scores["arxiv:paper2"]
        assert pagerank_scores["arxiv:paper1"] > pagerank_scores["arxiv:paper3"]
    
    def test_calculate_freshness_weighted_citations(self):
        """Test freshness-weighted citation calculation."""
        ranker = CitationRanker()
        
        # Create mock papers
        paper1 = Mock()
        paper1.arxiv_id = "paper1"
        paper1.citation_count = 100
        paper1.citation_dates = ["2020-01-01", "2021-01-01", "2023-01-01"]
        
        paper2 = Mock()
        paper2.arxiv_id = "paper2"
        paper2.citation_count = 50
        paper2.citation_dates = []
        
        papers = [paper1, paper2]
        
        scores = ranker.calculate_freshness_weighted_citations(papers)
        
        assert len(scores) == 2
        assert scores["arxiv:paper1"] > scores["arxiv:paper2"]
    
    def test_rank_papers(self):
        """Test paper ranking with multiple metrics."""
        ranker = CitationRanker()
        
        # Create mock papers
        papers = []
        for i in range(5):
            paper = Mock()
            paper.arxiv_id = f"paper{i}"
            paper.citation_count = 10 * i
            paper.published = datetime(2020, 1, 1)
            paper.citation_dates = [f"{2020 + j}-01-01" for j in range(i)]
            papers.append(paper)
        
        # Add citation relationships
        papers[1].references = ["paper0"]
        papers[2].references = ["paper0", "paper1"]
        papers[3].references = ["paper0", "paper1", "paper2"]
        papers[4].references = ["paper0", "paper1", "paper2", "paper3"]
        
        results = ranker.rank_papers(papers)
        
        assert len(results) == 5
        assert all(isinstance(result, ranker.__annotations__['CitationRank']) for result in results)
        
        # Results should be sorted by normalized score
        for i in range(len(results) - 1):
            assert results[i].normalized_score >= results[i + 1].normalized_score


class TestSimpleCitationRanker:
    """Test SimpleCitationRanker class."""
    
    def test_rank_by_citation_count(self):
        """Test ranking by citation count."""
        # Create mock papers with different citation counts
        papers = []
        for i in range(5):
            paper = Mock()
            paper.citation_count = i * 10
            papers.append(paper)
        
        ranked = SimpleCitationRanker.rank_by_citation_count(papers)
        
        assert len(ranked) == 5
        
        # Should be sorted by citation count (descending)
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]
    
    def test_rank_by_citation_velocity(self):
        """Test ranking by citation velocity."""
        papers = []
        current_year = datetime.now().year
        
        for i in range(3):
            paper = Mock()
            paper.citation_count = 100
            paper.published = datetime(current_year - i - 1, 1, 1)
            papers.append(paper)
        
        ranked = SimpleCitationRanker.rank_by_citation_velocity(papers)
        
        assert len(ranked) == 3
        
        # Newer papers should have higher velocity
        assert ranked[0][1] > ranked[1][1] > ranked[2][1]


class TestVenueRanker:
    """Test VenueRanker class."""
    
    def test_initialization(self):
        """Test VenueRanker initialization."""
        ranker = VenueRanker()
        assert "neurips" in ranker.venue_db["conferences"]
        assert "nature" in ranker.venue_db["journals"]
    
    def test_extract_venue_from_text(self):
        """Test venue extraction from text."""
        ranker = VenueRanker()
        
        test_cases = [
            ("Published in NeurIPS 2023", "neurips"),
            ("ICML conference proceedings", "icml"),
            ("arXiv preprint arXiv:1234.56789", "arxiv"),
            ("Journal of Machine Learning Research", "jmlr"),
            ("Some unknown conference", None),
        ]
        
        for text, expected in test_cases:
            result = ranker.extract_venue_from_text(text)
            if expected:
                assert result == expected
            else:
                assert result is None or result != expected
    
    def test_score_venue(self):
        """Test venue scoring."""
        ranker = VenueRanker()
        
        test_cases = [
            ("NeurIPS", VenueTier.TIER_1, 1.0),
            ("ICML", VenueTier.TIER_1, 1.0),
            ("AAAI", VenueTier.TIER_2, 0.8),
            ("arXiv", VenueTier.PREPRINT, 0.5),
            ("Some Workshop", VenueTier.WORKSHOP, 0.3),
            ("Unknown Conference", VenueTier.UNKNOWN, 0.2),
        ]
        
        for venue_name, expected_tier, expected_score in test_cases:
            score = ranker.score_venue(venue_name)
            
            assert score.tier == expected_tier
            assert abs(score.score - expected_score) < 0.01
    
    def test_rank_papers_by_venue(self):
        """Test paper ranking by venue."""
        ranker = VenueRanker()
        
        # Create mock papers with different venues
        papers = []
        venues = ["NeurIPS", "ICML", "arXiv", "Unknown Conference"]
        
        for venue in venues:
            paper = Mock()
            paper.venue = venue
            papers.append(paper)
        
        ranked = ranker.rank_papers_by_venue(papers)
        
        assert len(ranked) == 4
        
        # Should be sorted by venue score (descending)
        for i in range(len(ranked) - 1):
            assert ranked[i][1].score >= ranked[i + 1][1].score
    
    def test_get_venue_statistics(self):
        """Test venue statistics calculation."""
        ranker = VenueRanker()
        
        # Create mock papers
        papers = []
        for venue in ["NeurIPS", "ICML", "NeurIPS", "arXiv", "Unknown"]:
            paper = Mock()
            paper.venue = venue
            papers.append(paper)
        
        stats = ranker.get_venue_statistics(papers)
        
        assert stats["total_papers"] == 5
        assert stats["unique_venues"] == 4
        assert stats["venue_distribution"]["neurips"] == 2
        assert stats["papers_in_top_tier"] == 3  # 2 NeurIPS + 1 ICML


class TestHybridVenueRanker:
    """Test HybridVenueRanker class."""
    
    def test_rank_papers(self):
        """Test hybrid paper ranking."""
        venue_ranker = VenueRanker()
        hybrid_ranker = HybridVenueRanker(venue_ranker)
        
        # Create mock papers
        papers = []
        for i in range(3):
            paper = Mock()
            paper.venue = "NeurIPS" if i == 0 else "arXiv"
            paper.citation_count = 100 * i
            paper.published = datetime(2023 - i, 1, 1)
            papers.append(paper)
        
        ranked = hybrid_ranker.rank_papers(papers)
        
        assert len(ranked) == 3
        
        # Should be sorted by hybrid score
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1]
        
        # Check that each result has breakdown
        for _, _, breakdown in ranked:
            assert "venue_score" in breakdown
            assert "citation_score" in breakdown
            assert "recency_score" in breakdown


def test_ranking_pipeline():
    """Test end-to-end ranking pipeline."""
    # This test simulates a complete ranking pipeline
    from src.ranking import FeatureExtractor, CitationRanker, VenueRanker
    
    # Create mock papers
    papers = []
    for i in range(10):
        paper = Mock()
        paper.arxiv_id = f"paper{i}"
        paper.title = f"Test Paper {i}"
        paper.abstract = f"This is test paper {i} about machine learning."
        paper.authors = [f"Author {i}"]
        paper.venue = "NeurIPS" if i < 5 else "arXiv"
        paper.citation_count = i * 20
        paper.published = datetime(2023, 1, 1)
        paper.references = []
        papers.append(paper)
    
    # Add some citation relationships
    papers[1].references = ["paper0"]
    papers[2].references = ["paper0", "paper1"]
    
    # Extract features
    extractor = FeatureExtractor()
    feature_vectors = []
    
    for paper in papers:
        features = extractor.extract_from_paper(paper)
        vector = extractor.get_feature_vector(features)
        feature_vectors.append((paper, vector))
    
    assert len(feature_vectors) == 10
    
    # Rank by citations
    citation_ranker = CitationRanker()
    citation_results = citation_ranker.rank_papers(papers)
    
    assert len(citation_results) == 10
    
    # Rank by venue
    venue_ranker = VenueRanker()
    venue_results = venue_ranker.rank_papers_by_venue(papers)
    
    assert len(venue_results) == 10
    
    # Combine rankings (simple approach)
    combined_scores = {}
    
    for i, paper in enumerate(papers):
        # Get scores from different rankers
        citation_score = next((r.normalized_score for r in citation_results 
                              if r.paper_id == f"arxiv:paper{i}"), 0)
        venue_score = next((s[1].score for s in venue_results 
                           if s[0] == paper), 0)
        
        # Combine with weights
        combined_score = 0.6 * citation_score + 0.4 * venue_score
        combined_scores[paper] = combined_score
    
    # Sort by combined score
    ranked_papers = sorted(combined_scores.items(), 
                          key=lambda x: x[1], 
                          reverse=True)
    
    assert len(ranked_papers) == 10
    assert ranked_papers[0][1] >= ranked_papers[-1][1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
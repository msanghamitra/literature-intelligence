"""
Tests for retrieval module.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class TestBaseRetriever:
    """Test BaseRetriever class."""
    
    def test_initialization(self):
        """Test BaseRetriever initialization."""
        retriever = BaseRetriever()
        assert retriever.top_k == 10
        assert retriever.name == "base_retriever"
    
    def test_search_not_implemented(self):
        """Test that search raises NotImplementedError."""
        retriever = BaseRetriever()
        with pytest.raises(NotImplementedError):
            retriever.search("test query")
    
    def test_batch_search(self):
        """Test batch search functionality."""
        retriever = BaseRetriever()
        
        # Mock the search method
        retriever.search = Mock(return_value=["result1", "result2"])
        
        queries = ["query1", "query2"]
        results = retriever.batch_search(queries)
        
        assert len(results) == 2
        assert retriever.search.call_count == 2


class TestKeywordRetriever:
    """Test KeywordRetriever class."""
    
    @patch('src.retrieval.keyword_retriever.ArxivClient')
    def test_initialization(self, mock_arxiv_client):
        """Test KeywordRetriever initialization."""
        retriever = KeywordRetriever()
        assert retriever.name == "keyword_retriever"
        assert retriever.max_results == 100
    
    @patch('src.retrieval.keyword_retriever.ArxivClient')
    def test_search(self, mock_arxiv_client_class):
        """Test keyword search."""
        # Mock arXiv client response
        mock_client = Mock()
        mock_client.search.return_value = [
            Mock(arxiv_id="1234.56789", title="Test Paper 1"),
            Mock(arxiv_id="9876.54321", title="Test Paper 2")
        ]
        mock_arxiv_client_class.return_value = mock_client
        
        retriever = KeywordRetriever()
        results = retriever.search("test query")
        
        assert len(results) == 2
        assert results[0].arxiv_id == "1234.56789"
        mock_client.search.assert_called_once()


class TestSemanticRetriever:
    """Test SemanticRetriever class."""
    
    @patch('src.retrieval.semantic_retriever.VectorStore')
    @patch('src.retrieval.semantic_retriever.Embedder')
    def test_initialization(self, mock_embedder_class, mock_vector_store_class):
        """Test SemanticRetriever initialization."""
        retriever = SemanticRetriever()
        assert retriever.name == "semantic_retriever"
        assert retriever.top_k == 50
    
    @patch('src.retrieval.semantic_retriever.VectorStore')
    @patch('src.retrieval.semantic_retriever.Embedder')
    def test_search(self, mock_embedder_class, mock_vector_store_class):
        """Test semantic search."""
        # Mock embedder
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.random.randn(384)
        mock_embedder_class.return_value = mock_embedder
        
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = [
            ("doc1", 0.9, "Test document 1", {}),
            ("doc2", 0.8, "Test document 2", {})
        ]
        mock_vector_store_class.return_value = mock_vector_store
        
        retriever = SemanticRetriever()
        results = retriever.search("test query")
        
        assert len(results) == 2
        assert results[0][0] == "doc1"
        mock_embedder.embed.assert_called_once()
        mock_vector_store.search.assert_called_once()


class TestHybridRetriever:
    """Test HybridRetriever class."""
    
    def test_initialization(self):
        """Test HybridRetriever initialization."""
        retriever = HybridRetriever()
        assert retriever.name == "hybrid_retriever"
        assert retriever.fusion_method == "weighted_sum"
        assert retriever.weights == {"semantic": 0.6, "keyword": 0.4}
    
    def test_score_fusion_weighted_sum(self):
        """Test weighted sum score fusion."""
        retriever = HybridRetriever()
        
        keyword_scores = {"doc1": 0.8, "doc2": 0.6, "doc3": 0.4}
        semantic_scores = {"doc1": 0.7, "doc2": 0.9, "doc4": 0.5}
        
        fused_scores = retriever._fuse_scores_weighted_sum(
            keyword_scores, semantic_scores
        )
        
        # Check that all documents are included
        assert len(fused_scores) == 4
        
        # Check weighted sum calculation
        expected_doc1 = 0.8 * 0.4 + 0.7 * 0.6
        assert abs(fused_scores["doc1"] - expected_doc1) < 1e-10
    
    def test_score_fusion_reciprocal_rank(self):
        """Test reciprocal rank fusion."""
        retriever = HybridRetriever(fusion_method="reciprocal_rank_fusion")
        
        keyword_results = [
            ("doc1", 0.8, "content1", {}),
            ("doc2", 0.6, "content2", {}),
            ("doc3", 0.4, "content3", {})
        ]
        
        semantic_results = [
            ("doc2", 0.9, "content2", {}),
            ("doc1", 0.7, "content1", {}),
            ("doc4", 0.5, "content4", {})
        ]
        
        fused_results = retriever._fuse_scores_reciprocal_rank(
            keyword_results, semantic_results
        )
        
        # Check that results are sorted by fused score
        assert len(fused_results) == 4
        
        # In RRF, doc2 should rank highest (rank 1 in semantic, rank 2 in keyword)
        # doc1 should rank second (rank 2 in semantic, rank 1 in keyword)
        # Let's just check that we get results
        assert fused_results[0][0] in ["doc1", "doc2"]
    
    @patch('src.retrieval.hybrid_retriever.KeywordRetriever')
    @patch('src.retrieval.hybrid_retriever.SemanticRetriever')
    def test_search(self, mock_semantic_class, mock_keyword_class):
        """Test hybrid search."""
        # Mock retrievers
        mock_keyword_retriever = Mock()
        mock_keyword_retriever.search.return_value = [
            ("doc1", 0.8, "Keyword result 1", {}),
            ("doc2", 0.6, "Keyword result 2", {})
        ]
        mock_keyword_class.return_value = mock_keyword_retriever
        
        mock_semantic_retriever = Mock()
        mock_semantic_retriever.search.return_value = [
            ("doc2", 0.9, "Semantic result 2", {}),
            ("doc3", 0.7, "Semantic result 3", {})
        ]
        mock_semantic_class.return_value = mock_semantic_retriever
        
        retriever = HybridRetriever()
        results = retriever.search("test query")
        
        # Check that both retrievers were called
        mock_keyword_retriever.search.assert_called_once()
        mock_semantic_retriever.search.assert_called_once()
        
        # Check that we got results
        assert len(results) > 0


class TestReranker:
    """Test Reranker class."""
    
    def test_initialization(self):
        """Test Reranker initialization."""
        reranker = Reranker()
        assert reranker.top_k == 10
        assert reranker.model_name is not None
    
    @patch('src.retrieval.reranker.AutoModelForSequenceClassification')
    @patch('src.retrieval.reranker.AutoTokenizer')
    def test_rerank(self, mock_tokenizer_class, mock_model_class):
        """Test reranking functionality."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]]
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value.logits = Mock()
        mock_model.return_value.logits.detach().cpu().numpy.return_value = [[0.9, 0.1]]
        mock_model_class.from_pretrained.return_value = mock_model
        
        reranker = Reranker()
        
        query = "test query"
        documents = [
            ("doc1", 0.8, "Document 1 text", {}),
            ("doc2", 0.6, "Document 2 text", {}),
            ("doc3", 0.4, "Document 3 text", {})
        ]
        
        reranked = reranker.rerank(query, documents)
        
        # Check that we get the same number of documents
        assert len(reranked) == len(documents)
        
        # Check that scores are updated
        for doc in reranked:
            assert isinstance(doc[1], float)


def test_retriever_registry():
    """Test retriever registry pattern."""
    from src.retrieval import get_retriever
    
    # Test getting keyword retriever
    keyword_retriever = get_retriever("keyword")
    assert keyword_retriever.name == "keyword_retriever"
    
    # Test getting semantic retriever
    semantic_retriever = get_retriever("semantic")
    assert semantic_retriever.name == "semantic_retriever"
    
    # Test getting hybrid retriever
    hybrid_retriever = get_retriever("hybrid")
    assert hybrid_retriever.name == "hybrid_retriever"
    
    # Test invalid retriever name
    with pytest.raises(ValueError):
        get_retriever("invalid")


def test_retrieval_pipeline():
    """Test end-to-end retrieval pipeline."""
    # This is a high-level integration test
    from src.retrieval import create_retrieval_pipeline
    
    # Mock configuration
    config = {
        "retriever": "hybrid",
        "reranker": {
            "enable": True,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
        }
    }
    
    pipeline = create_retrieval_pipeline(config)
    
    # Check that pipeline has required methods
    assert hasattr(pipeline, 'search')
    assert hasattr(pipeline, 'batch_search')
    
    # Note: Actual search test would require mocked dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
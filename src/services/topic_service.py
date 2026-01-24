# src/services/topic_service.py
"""
BUSINESS LOGIC: Topic analysis service
USES: src/models/topics.py (existing model)
"""
import pandas as pd
from typing import List, Dict, Any, Optional

# Import existing model
from src.preprocessing.topic_modeller import build_topics, load_topics, load_corpus_with_topics
from src.services.search_service import Paper  # Your Paper class


class TopicService:
    """Service that USES the existing topics model"""
    
    def __init__(self):
        pass
    
    def generate_topics_from_corpus(self, n_topics: int = 8) -> Dict[str, Any]:
        """
        Use existing topics.py to generate topics for stored corpus
        """
        try:
            # Use the EXISTING function from topics.py
            topics_df, corpus_df = build_topics(n_topics=n_topics)
            
            return {
                "success": True,
                "topics_df": topics_df,
                "corpus_with_topics": corpus_df,
                "n_topics": n_topics
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_existing_topics(self) -> Dict[str, Any]:
        """
        Load pre-generated topics from disk
        """
        try:
            topics_df = load_topics()
            corpus_df = load_corpus_with_topics()
            
            return {
                "success": True,
                "topics_df": topics_df,
                "corpus_with_topics": corpus_df
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "No topics found. Generate topics first."
            }
    
    def analyze_live_results(self, papers: List[Paper]) -> Dict[str, Any]:
        """
        Quick topic analysis for live search results (lightweight)
        This is DIFFERENT from the corpus-based topics in topics.py
        """
        if not papers or len(papers) < 5:
            return {
                "success": False,
                "error": f"Need at least 5 papers for topic analysis. Found {len(papers) if papers else 0}.",
                "topics_df": pd.DataFrame(),
                "paper_labels": [],
                "paper_titles": []
            }
        
        try:
            # Convert papers to DataFrame format
            paper_data = []
            for paper in papers:
                paper_data.append({
                    "arxiv_id": getattr(paper, 'id', ''),
                    "title": getattr(paper, 'title', ''),
                    "abstract": getattr(paper, 'abstract', ''),
                    "text_unit": f"{getattr(paper, 'title', '')}. {getattr(paper, 'abstract', '')}",
                    "authors": getattr(paper, 'authors', ''),
                    "published": getattr(paper, 'published', ''),
                    "pdf_url": getattr(paper, 'pdf_url', ''),
                })
            
            df = pd.DataFrame(paper_data)
            
            # Run lightweight topic modeling
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            
            texts = df["text_unit"].fillna("").tolist()
            
            # Determine number of topics (1 topic per 3-5 papers)
            n_topics = min(8, max(2, len(texts) // 3))
            
            # Vectorize
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            
            X = vectorizer.fit_transform(texts)
            
            # Cluster
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)
            
            # Extract topic keywords
            terms = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_id in range(n_topics):
                top_idx = kmeans.cluster_centers_[topic_id].argsort()[::-1][:10]
                keywords = [terms[i] for i in top_idx]
                count = int((labels == topic_id).sum())
                
                topics.append({
                    "topic_id": topic_id,
                    "doc_count": count,
                    "keywords": ", ".join(keywords),
                })
            
            topics_df = pd.DataFrame(topics).sort_values("doc_count", ascending=False)
            
            return {
                "success": True,
                "topics_df": topics_df,
                "paper_labels": labels.tolist(),
                "paper_titles": df["title"].tolist(),
                "paper_ids": df["arxiv_id"].tolist(),
                "num_topics": n_topics
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Topic analysis failed: {str(e)}",
                "topics_df": pd.DataFrame(),
                "paper_labels": [],
                "paper_titles": []
            }
    
    def get_papers_in_topic(
        self, 
        topic_id: int, 
        papers: List[Paper], 
        topic_data: Optional[Dict[str, Any]] = None
    ) -> List[Paper]:
        """
        Get papers belonging to a specific topic from live results
        """
        if not topic_data or "paper_labels" not in topic_data:
            return []
        
        labels = topic_data["paper_labels"]
        
        # Find papers with this topic label
        papers_in_topic = []
        for i, paper in enumerate(papers):
            if i < len(labels) and labels[i] == topic_id:
                papers_in_topic.append(paper)
        
        return papers_in_topic


# For backward compatibility
def analyze_live_results(papers: List[Paper]) -> Dict[str, Any]:
    """Standalone function for backward compatibility"""
    service = TopicService()
    return service.analyze_live_results(papers)


if __name__ == "__main__":
    # Test the service
    print("Testing TopicService...")
    
    # Create mock papers for testing
    class MockPaper:
        def __init__(self, title, abstract, authors="Author"):
            self.id = f"test_{hash(title)}"
            self.title = title
            self.abstract = abstract
            self.authors = authors
            self.published = "2023-01-01"
            self.pdf_url = "http://example.com"
    
    # Create test papers
    test_papers = [
        MockPaper(
            "Machine Learning for Image Classification",
            "This paper presents a new machine learning approach for image classification using convolutional neural networks."
        ),
        MockPaper(
            "Deep Learning in Natural Language Processing",
            "We explore deep learning techniques for natural language processing tasks including sentiment analysis and machine translation."
        ),
        MockPaper(
            "Reinforcement Learning for Robotics",
            "This research applies reinforcement learning algorithms to robotic control and navigation problems."
        ),
        MockPaper(
            "Transformer Models in Computer Vision",
            "We investigate the use of transformer architectures for computer vision tasks previously dominated by CNNs."
        ),
        MockPaper(
            "Federated Learning for Privacy-Preserving ML",
            "A survey of federated learning techniques that enable machine learning while preserving data privacy."
        ),
    ]
    
    service = TopicService()
    result = service.analyze_live_results(test_papers)
    
    if result["success"]:
        print(f"Success! Generated {result['num_topics']} topics")
        print("\nTopics:")
        print(result["topics_df"].to_string())
    else:
        print(f"Failed: {result['error']}")
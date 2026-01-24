# src/services/topic_service.py
"""
BUSINESS LOGIC: Topic analysis service
USES: src/models/topics.py (existing model)
"""
import pandas as pd
from typing import List, Dict, Any, Optional
import re

# Import existing model
from src.preprocessing.topic_modeller import build_topics, load_topics, load_corpus_with_topics
from src.services.search_service import Paper  # Your Paper class


class TopicService:
    """Service that USES the existing topics model"""
    
    def __init__(self):
        # Custom stop words list to filter out generic terms
        self.custom_stop_words = {
            # Generic ML/AI terms
            'data', 'model', 'models', 'learning', 'algorithm', 'algorithms',
            'method', 'methods', 'approach', 'approaches', 'framework',
            'problem', 'problems', 'task', 'tasks',
            
            # Generic verbs/process words
            'using', 'use', 'used', 'propose', 'proposed', 'proposes',
            'show', 'shows', 'shown', 'demonstrate', 'demonstrates',
            'introduce', 'introduces', 'introduced', 'present', 'presents',
            'propose', 'proposes', 'proposed', 'develop', 'develops',
            'propose', 'proposes', 'proposed', 'evaluate', 'evaluates',
            
            # Generic adjectives/quality words
            'novel', 'new', 'effective', 'efficient', 'efficiency',
            'better', 'improved', 'high', 'low', 'large', 'small',
            'different', 'various', 'multiple', 'several', 'recent',
            'state', 'art', 'existing', 'current', 'previous',
            
            # Generic nouns
            'time', 'performance', 'result', 'results', 'accuracy',
            'analysis', 'experiment', 'experiments', 'evaluation',
            'application', 'applications', 'work', 'study', 'paper',
            
            # Math/CS generic terms
            'function', 'functions', 'parameter', 'parameters',
            'value', 'values', 'set', 'sets', 'number', 'numbers',
            
            # Common ML terms that are too generic
            'network', 'networks', 'feature', 'features', 'layer', 'layers',
            'training', 'train', 'test', 'testing', 'validation',
            'dataset', 'datasets', 'sample', 'samples',
            
            # Common academic words
            'research', 'researchers', 'field', 'fields', 'area', 'areas',
            'work', 'works', 'study', 'studies', 'paper', 'papers',
        }
    
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
            
            # Custom tokenizer with better filtering
            def custom_tokenizer(text):
                # Convert to lowercase
                text = text.lower()
                # Remove special characters but keep hyphens in words like "pre-trained"
                text = re.sub(r'[^\w\s\-]', ' ', text)
                # Tokenize
                tokens = text.split()
                # Filter tokens
                filtered_tokens = []
                for token in tokens:
                    # Remove generic words
                    if token in self.custom_stop_words:
                        continue
                    # Remove short words (less than 3 chars) unless they're acronyms
                    if len(token) < 3 and not token.isupper():
                        continue
                    # Remove numbers and single characters
                    if token.isdigit() or len(token) == 1:
                        continue
                    # Remove common verb endings
                    if token.endswith(('ing', 'ed', 's', 'es', 'ly')):
                        stem = token.rstrip('ingedsly')
                        if stem in self.custom_stop_words:
                            continue
                    filtered_tokens.append(token)
                return filtered_tokens
            
            # Improved TF-IDF with custom stop words
            vectorizer = TfidfVectorizer(
                stop_words='english',  # Start with English stop words
                max_features=800,      # Reduced to get more meaningful terms
                ngram_range=(1, 3),    # Include up to 3-word phrases
                min_df=2,              # Term must appear in at least 2 documents
                max_df=0.8,            # Filter out terms that appear in >80% of docs
                tokenizer=custom_tokenizer
            )
            
            X = vectorizer.fit_transform(texts)
            
            # Check if we have enough features after filtering
            if X.shape[1] < 10:  # Less than 10 unique terms
                return {
                    "success": False,
                    "error": "Not enough meaningful terms after filtering generic words.",
                    "topics_df": pd.DataFrame(),
                    "paper_labels": [],
                    "paper_titles": []
                }
            
            # Cluster
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(X)
            
            # Extract topic keywords
            terms = vectorizer.get_feature_names_out()
            
            # Filter and rank keywords better
            topics = []
            for topic_id in range(n_topics):
                # Get raw scores for this topic
                scores = kmeans.cluster_centers_[topic_id]
                
                # Get indices sorted by score (descending)
                top_indices = scores.argsort()[::-1]
                
                # Filter keywords
                filtered_keywords = []
                for idx in top_indices:
                    keyword = terms[idx]
                    score = scores[idx]
                    
                    # Skip if score is too low
                    if score < 0.01:
                        continue
                    
                    # Additional filtering
                    # Skip terms that are too generic even after initial filtering
                    words = keyword.split()
                    if all(w in self.custom_stop_words or len(w) < 3 for w in words):
                        continue
                    
                    # Check if keyword is a subphrase of already included keyword
                    if any(keyword in k or k in keyword for k in filtered_keywords):
                        continue
                    
                    filtered_keywords.append(keyword)
                    
                    # Stop when we have enough good keywords
                    if len(filtered_keywords) >= 10:
                        break
                
                # If we don't have enough keywords, add some from the original list
                if len(filtered_keywords) < 5:
                    for idx in top_indices:
                        if terms[idx] not in filtered_keywords:
                            filtered_keywords.append(terms[idx])
                        if len(filtered_keywords) >= 5:
                            break
                
                count = int((labels == topic_id).sum())
                
                # Give topic a descriptive name based on top keywords
                if filtered_keywords:
                    # Take first 2-3 keywords for a name
                    name_keywords = filtered_keywords[:min(3, len(filtered_keywords))]
                    topic_name = " & ".join(name_keywords)
                else:
                    topic_name = f"Topic {topic_id}"
                
                topics.append({
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "doc_count": count,
                    "keywords": ", ".join(filtered_keywords[:8]),  # Show only top 8
                    "all_keywords": filtered_keywords
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
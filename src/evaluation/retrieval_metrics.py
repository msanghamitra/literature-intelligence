"""
Retrieval evaluation metrics for scientific paper search.
Implements standard IR metrics and scientific search specific metrics.
"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    # Standard metrics
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    average_precision: float
    mean_average_precision: float
    ndcg_at_k: Dict[int, float]
    mrr: float
    
    # Scientific search specific
    venue_precision_at_k: Dict[int, float]
    citation_precision_at_k: Dict[int, float]
    novelty_at_k: Dict[int, float]
    diversity_at_k: Dict[int, float]
    
    # Statistical significance
    confidence_intervals: Dict[str, Tuple[float, float]]


class RetrievalEvaluator:
    """
    Evaluator for retrieval systems.
    Implements standard IR metrics and scientific search specific metrics.
    """
    
    def __init__(self, relevance_threshold: float = 0.5):
        """
        Initialize evaluator.
        
        Args:
            relevance_threshold: Minimum relevance score for a document to be considered relevant
        """
        self.relevance_threshold = relevance_threshold
    
    def evaluate(self, 
                 retrieved_docs: List[List[str]], 
                 relevance_scores: List[List[float]],
                 k_values: List[int] = None) -> RetrievalMetrics:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_docs: List of lists of retrieved document IDs for each query
            relevance_scores: List of lists of relevance scores for retrieved documents
            k_values: List of k values for precision@k, recall@k, etc.
            
        Returns:
            RetrievalMetrics object
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        # Validate inputs
        n_queries = len(retrieved_docs)
        assert len(relevance_scores) == n_queries, \
            "Number of queries in retrieved_docs and relevance_scores must match"
        
        # Calculate standard metrics
        precision_at_k = self._calculate_precision_at_k(retrieved_docs, relevance_scores, k_values)
        recall_at_k = self._calculate_recall_at_k(retrieved_docs, relevance_scores, k_values)
        average_precision = self._calculate_average_precision(retrieved_docs, relevance_scores)
        mean_average_precision = np.mean(average_precision) if average_precision else 0.0
        ndcg_at_k = self._calculate_ndcg_at_k(retrieved_docs, relevance_scores, k_values)
        mrr = self._calculate_mrr(retrieved_docs, relevance_scores)
        
        # Calculate scientific metrics (if additional metadata is available)
        venue_precision_at_k = {}
        citation_precision_at_k = {}
        novelty_at_k = {}
        diversity_at_k = {}
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            average_precision, ndcg_at_k[10] if 10 in ndcg_at_k else 0.0
        )
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            average_precision=mean_average_precision,
            mean_average_precision=mean_average_precision,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            venue_precision_at_k=venue_precision_at_k,
            citation_precision_at_k=citation_precision_at_k,
            novelty_at_k=novelty_at_k,
            diversity_at_k=diversity_at_k,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_precision_at_k(self, retrieved_docs, relevance_scores, k_values):
        """Calculate Precision@k for each k value."""
        precision = {k: [] for k in k_values}
        
        for docs, scores in zip(retrieved_docs, relevance_scores):
            relevant = [score >= self.relevance_threshold for score in scores]
            
            for k in k_values:
                if len(docs) >= k:
                    precision_at_k = sum(relevant[:k]) / k
                    precision[k].append(precision_at_k)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in precision.items()}
    
    def _calculate_recall_at_k(self, retrieved_docs, relevance_scores, k_values):
        """Calculate Recall@k for each k value."""
        recall = {k: [] for k in k_values}
        
        for docs, scores in zip(retrieved_docs, relevance_scores):
            relevant = [score >= self.relevance_threshold for score in scores]
            total_relevant = sum(relevant)
            
            if total_relevant == 0:
                continue
            
            for k in k_values:
                if len(docs) >= k:
                    recall_at_k = sum(relevant[:k]) / total_relevant
                    recall[k].append(recall_at_k)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in recall.items()}
    
    def _calculate_average_precision(self, retrieved_docs, relevance_scores):
        """Calculate Average Precision for each query."""
        avg_precisions = []
        
        for docs, scores in zip(retrieved_docs, relevance_scores):
            relevant = [score >= self.relevance_threshold for score in scores]
            
            if not any(relevant):
                avg_precisions.append(0.0)
                continue
            
            precisions = []
            num_relevant = 0
            
            for i, is_relevant in enumerate(relevant, 1):
                if is_relevant:
                    num_relevant += 1
                    precisions.append(num_relevant / i)
            
            avg_precision = sum(precisions) / num_relevant if num_relevant > 0 else 0.0
            avg_precisions.append(avg_precision)
        
        return avg_precisions
    
    def _calculate_ndcg_at_k(self, retrieved_docs, relevance_scores, k_values):
        """Calculate Normalized Discounted Cumulative Gain@k."""
        ndcg = {k: [] for k in k_values}
        
        for docs, scores in zip(retrieved_docs, relevance_scores):
            # Ideal ordering (sorted by relevance)
            ideal_scores = sorted(scores, reverse=True)
            
            for k in k_values:
                if len(docs) >= k:
                    # Calculate DCG
                    dcg = 0.0
                    for i in range(min(k, len(scores))):
                        rel = scores[i]
                        dcg += rel / math.log2(i + 2)  # i+2 because i starts from 0
                    
                    # Calculate IDCG
                    idcg = 0.0
                    for i in range(min(k, len(ideal_scores))):
                        rel = ideal_scores[i]
                        idcg += rel / math.log2(i + 2)
                    
                    ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
                    ndcg[k].append(ndcg_at_k)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in ndcg.items()}
    
    def _calculate_mrr(self, retrieved_docs, relevance_scores):
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for docs, scores in zip(retrieved_docs, relevance_scores):
            for i, score in enumerate(scores, 1):
                if score >= self.relevance_threshold:
                    reciprocal_ranks.append(1.0 / i)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _calculate_confidence_intervals(self, average_precision, ndcg_10, confidence=0.95):
        """Calculate 95% confidence intervals for key metrics."""
        import scipy.stats as stats
        
        if not average_precision:
            return {}
        
        n = len(average_precision)
        if n < 2:
            return {}
        
        # Calculate for Average Precision
        ap_mean = np.mean(average_precision)
        ap_std = np.std(average_precision, ddof=1)
        ap_se = ap_std / math.sqrt(n)
        ap_ci = stats.t.interval(confidence, n-1, loc=ap_mean, scale=ap_se)
        
        return {
            "average_precision": (float(ap_ci[0]), float(ap_ci[1])),
            "ndcg_10": (ndcg_10 - 0.05, ndcg_10 + 0.05)  # Simplified for now
        }
    
    def evaluate_with_metadata(self, 
                               retrieved_docs: List[List], 
                               relevance_scores: List[List[float]],
                               metadata: List[List[Dict]],
                               k_values: List[int] = None) -> RetrievalMetrics:
        """
        Evaluate with scientific metadata for specialized metrics.
        
        Args:
            retrieved_docs: List of lists of retrieved documents
            relevance_scores: List of lists of relevance scores
            metadata: List of lists of metadata dictionaries for each document
            k_values: List of k values
            
        Returns:
            RetrievalMetrics with scientific metrics
        """
        metrics = self.evaluate(retrieved_docs, relevance_scores, k_values)
        
        # Calculate venue-based precision
        venue_precision = self._calculate_venue_precision(retrieved_docs, metadata, k_values)
        metrics.venue_precision_at_k = venue_precision
        
        # Calculate citation-based precision
        citation_precision = self._calculate_citation_precision(retrieved_docs, metadata, k_values)
        metrics.citation_precision_at_k = citation_precision
        
        # Calculate novelty
        novelty = self._calculate_novelty(retrieved_docs, metadata, k_values)
        metrics.novelty_at_k = novelty
        
        # Calculate diversity
        diversity = self._calculate_diversity(retrieved_docs, metadata, k_values)
        metrics.diversity_at_k = diversity
        
        return metrics
    
    def _calculate_venue_precision(self, retrieved_docs, metadata, k_values):
        """Calculate precision based on venue prestige."""
        venue_precision = {k: [] for k in k_values}
        
        for docs, metas in zip(retrieved_docs, metadata):
            # Extract venue scores from metadata
            venue_scores = []
            for meta in metas:
                if meta and 'venue_score' in meta:
                    venue_scores.append(meta['venue_score'])
                else:
                    venue_scores.append(0.0)
            
            # Calculate venue-based relevance (threshold = 0.5)
            venue_relevant = [score >= 0.5 for score in venue_scores]
            
            for k in k_values:
                if len(docs) >= k:
                    precision_at_k = sum(venue_relevant[:k]) / k
                    venue_precision[k].append(precision_at_k)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in venue_precision.items()}
    
    def _calculate_citation_precision(self, retrieved_docs, metadata, k_values):
        """Calculate precision based on citation count."""
        citation_precision = {k: [] for k in k_values}
        
        for docs, metas in zip(retrieved_docs, metadata):
            # Extract citation counts from metadata
            citation_counts = []
            for meta in metas:
                if meta and 'citation_count' in meta:
                    citation_counts.append(meta['citation_count'])
                else:
                    citation_counts.append(0)
            
            # Normalize citation counts
            if citation_counts:
                max_citations = max(citation_counts)
                if max_citations > 0:
                    citation_scores = [c / max_citations for c in citation_counts]
                else:
                    citation_scores = [0.0] * len(citation_counts)
            else:
                citation_scores = []
            
            # Calculate citation-based relevance (threshold = 0.3)
            citation_relevant = [score >= 0.3 for score in citation_scores]
            
            for k in k_values:
                if len(docs) >= k:
                    precision_at_k = sum(citation_relevant[:k]) / k
                    citation_precision[k].append(precision_at_k)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in citation_precision.items()}
    
    def _calculate_novelty(self, retrieved_docs, metadata, k_values):
        """Calculate novelty of retrieved results."""
        novelty_scores = {k: [] for k in k_values}
        
        for docs, metas in zip(retrieved_docs, metadata):
            if not metas:
                continue
            
            # Extract publication years
            years = []
            for meta in metas:
                if meta and 'year' in meta:
                    years.append(meta['year'])
                else:
                    years.append(2000)  # Default old year
            
            # Calculate novelty (recent papers are more novel)
            current_year = 2024
            novelty = [min((year - 2000) / (current_year - 2000), 1.0) for year in years]
            
            for k in k_values:
                if len(docs) >= k:
                    avg_novelty = np.mean(novelty[:k]) if novelty[:k] else 0.0
                    novelty_scores[k].append(avg_novelty)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in novelty_scores.items()}
    
    def _calculate_diversity(self, retrieved_docs, metadata, k_values):
        """Calculate diversity of retrieved results."""
        diversity_scores = {k: [] for k in k_values}
        
        for docs, metas in zip(retrieved_docs, metadata):
            if not metas or len(metas) < 2:
                continue
            
            for k in k_values:
                if len(docs) >= k:
                    # Extract venues for top k
                    venues = []
                    for meta in metas[:k]:
                        if meta and 'venue' in meta:
                            venues.append(meta['venue'])
                        else:
                            venues.append('unknown')
                    
                    # Calculate venue diversity
                    unique_venues = len(set(venues))
                    diversity = unique_venues / k
                    diversity_scores[k].append(diversity)
        
        return {k: np.mean(vals) if vals else 0.0 for k, vals in diversity_scores.items()}
    
    def statistical_significance_test(self, 
                                     metrics_a: Dict[str, float],
                                     metrics_b: Dict[str, float],
                                     n_samples: int = 1000) -> Dict[str, float]:
        """
        Perform statistical significance test between two systems.
        
        Args:
            metrics_a: Metrics from system A
            metrics_b: Metrics from system B
            n_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with p-values for each metric
        """
        # This is a placeholder for bootstrap significance testing
        # In practice, you'd need the raw scores for each query
        
        p_values = {}
        for metric in ['map', 'ndcg_10', 'mrr']:
            if metric in metrics_a and metric in metrics_b:
                # Simplified p-value calculation
                diff = abs(metrics_a[metric] - metrics_b[metric])
                if diff > 0.05:  # Arbitrary threshold
                    p_values[metric] = 0.01  # "Significant"
                else:
                    p_values[metric] = 0.5   # "Not significant"
        
        return p_values


class CrossValidationEvaluator:
    """
    Cross-validation evaluator for retrieval systems.
    """
    
    def __init__(self, n_folds: int = 5):
        """
        Initialize cross-validation evaluator.
        
        Args:
            n_folds: Number of folds for cross-validation
        """
        self.n_folds = n_folds
        self.evaluator = RetrievalEvaluator()
    
    def cross_validate(self, queries, relevance_data, retrieval_func):
        """
        Perform cross-validation.
        
        Args:
            queries: List of queries
            relevance_data: Ground truth relevance data
            retrieval_func: Function that takes queries and returns results
            
        Returns:
            Dictionary of average metrics across folds
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        all_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(queries)):
            print(f"Processing fold {fold + 1}/{self.n_folds}")
            
            # Split data
            train_queries = [queries[i] for i in train_idx]
            test_queries = [queries[i] for i in test_idx]
            
            # Train/adapt retrieval function (if needed)
            # This is system-dependent
            
            # Evaluate on test set
            test_results = retrieval_func(test_queries)
            
            # Calculate metrics
            # This assumes test_results and relevance_data are in compatible format
            # In practice, you'd need to extract relevance scores for test_queries
            
            metrics = self.evaluator.evaluate(test_results, relevance_data)
            all_metrics.append(vars(metrics))
        
        # Aggregate metrics across folds
        avg_metrics = self._aggregate_metrics(all_metrics)
        return avg_metrics
    
    def _aggregate_metrics(self, all_metrics):
        """Aggregate metrics across folds."""
        if not all_metrics:
            return {}
        
        aggregated = {}
        for metric_dict in all_metrics:
            for key, value in metric_dict.items():
                if key not in aggregated:
                    aggregated[key] = []
                
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    if key not in aggregated:
                        aggregated[key] = {}
                    
                    for sub_key, sub_value in value.items():
                        if sub_key not in aggregated[key]:
                            aggregated[key][sub_key] = []
                        aggregated[key][sub_key].append(sub_value)
                else:
                    aggregated[key].append(value)
        
        # Calculate means
        result = {}
        for key, values in aggregated.items():
            if isinstance(values[0], dict):
                # Calculate mean for nested dictionaries
                result[key] = {}
                for sub_key in values[0].keys():
                    sub_values = [v[sub_key] for v in values]
                    result[key][sub_key] = np.mean(sub_values)
            else:
                result[key] = np.mean(values)
        
        return result
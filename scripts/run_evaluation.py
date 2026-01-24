"""
Batch script to run comprehensive evaluation of retrieval and RAG systems.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.retrieval_metrics import RetrievalEvaluator
from src.evaluation.rag_metrics import RAGEvaluator, ScientificRAGEvaluator
from src.evaluation.ablation_runner import AblationRunner, GridSearchRunner
import yaml
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation of retrieval and RAG systems")
    
    subparsers = parser.add_subparsers(dest="command", help="Evaluation type")
    
    # Retrieval evaluation parser
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval evaluation")
    retrieval_parser.add_argument("--queries", type=str, required=True,
                                help="JSON file with test queries")
    retrieval_parser.add_argument("--results", type=str, required=True,
                                help="JSON file with retrieval results")
    retrieval_parser.add_argument("--output", type=str, default="evaluation_results",
                                help="Output directory for results")
    retrieval_parser.add_argument("--config", type=str, default="config/retriever.yaml",
                                help="Path to configuration file")
    
    # RAG evaluation parser
    rag_parser = subparsers.add_parser("rag", help="Run RAG evaluation")
    rag_parser.add_argument("--dataset", type=str, required=True,
                          help="JSON file with QA dataset")
    rag_parser.add_argument("--responses", type=str, required=True,
                          help="JSON file with RAG responses")
    rag_parser.add_argument("--output", type=str, default="evaluation_results",
                          help="Output directory for results")
    rag_parser.add_argument("--config", type=str, default="config/rag.yaml",
                          help="Path to configuration file")
    
    # Ablation study parser
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_parser.add_argument("--config", type=str, required=True,
                               help="YAML file with ablation configuration")
    ablation_parser.add_argument("--output", type=str, default="experiments",
                               help="Output directory for results")
    
    # Grid search parser
    grid_parser = subparsers.add_parser("grid", help="Run grid search")
    grid_parser.add_argument("--param-grid", type=str, required=True,
                           help="JSON file with parameter grid")
    grid_parser.add_argument("--base-config", type=str, required=True,
                           help="JSON file with base configuration")
    grid_parser.add_argument("--output", type=str, default="experiments",
                           help="Output directory for results")
    
    # Common arguments
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def load_json_file(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results, output_dir, filename):
    """Save results to JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    filepath = output_dir / filename
    
    # Convert to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    return filepath


def run_retrieval_evaluation(args, logger):
    """Run retrieval evaluation."""
    logger.info("Running retrieval evaluation")
    
    # Load data
    queries = load_json_file(args.queries)
    results = load_json_file(args.results)
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator()
    
    # Extract data for evaluation
    retrieved_docs = []
    relevance_scores = []
    
    for query_id, query_data in queries.items():
        if query_id in results:
            result_data = results[query_id]
            
            # Extract retrieved document IDs
            docs = [doc["id"] for doc in result_data.get("retrieved_docs", [])]
            retrieved_docs.append(docs)
            
            # Extract relevance scores (0/1 or graded)
            scores = [doc.get("relevance_score", 0.0) for doc in result_data.get("retrieved_docs", [])]
            relevance_scores.append(scores)
    
    # Run evaluation
    metrics = evaluator.evaluate(retrieved_docs, relevance_scores)
    
    # Save results
    results_dict = vars(metrics)
    output_file = save_results(results_dict, args.output, "retrieval_metrics.json")
    
    logger.info(f"Retrieval evaluation complete")
    logger.info(f"Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("RETRIEVAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of queries: {len(queries)}")
    print(f"\nKey Metrics:")
    print(f"  Mean Average Precision (MAP): {metrics.mean_average_precision:.4f}")
    print(f"  nDCG@10: {metrics.ndcg_at_k.get(10, 0):.4f}")
    print(f"  MRR: {metrics.mrr:.4f}")
    print(f"\nPrecision@k:")
    for k, score in metrics.precision_at_k.items():
        print(f"  P@{k}: {score:.4f}")
    
    return metrics


def run_rag_evaluation(args, logger):
    """Run RAG evaluation."""
    logger.info("Running RAG evaluation")
    
    # Load data
    dataset = load_json_file(args.dataset)
    responses = load_json_file(args.responses)
    
    # Initialize evaluator
    evaluator = ScientificRAGEvaluator()
    
    all_metrics = []
    
    for item_id, item_data in dataset.items():
        if item_id in responses:
            response_data = responses[item_id]
            
            # Extract data for evaluation
            question = item_data.get("question", "")
            reference_answer = item_data.get("answer", "")
            generated_answer = response_data.get("generated_answer", "")
            cited_documents = response_data.get("cited_documents", [])
            ground_truth_citations = item_data.get("citations", [])
            source_documents = response_data.get("source_documents", [])
            
            # Run comprehensive evaluation
            metrics = evaluator.comprehensive_evaluate(
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                cited_documents=cited_documents,
                ground_truth_citations=ground_truth_citations,
                source_documents=source_documents,
                question=question,
                retrieval_time=response_data.get("retrieval_time"),
                generation_time=response_data.get("generation_time"),
                token_count=response_data.get("token_count")
            )
            
            all_metrics.append(vars(metrics))
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                avg_metrics[key] = np.mean(values)
        
        # Save results
        output_file = save_results(avg_metrics, args.output, "rag_metrics.json")
        
        logger.info(f"RAG evaluation complete")
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("RAG EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Number of QA pairs: {len(all_metrics)}")
        print(f"\nAnswer Quality:")
        print(f"  Answer Relevance: {avg_metrics.get('answer_relevance', 0):.4f}")
        print(f"  Answer Accuracy (ROUGE-L): {avg_metrics.get('answer_accuracy', 0):.4f}")
        print(f"  BLEU Score: {avg_metrics.get('bleu_score', 0):.4f}")
        
        print(f"\nCitation Metrics:")
        print(f"  Citation Precision: {avg_metrics.get('citation_precision', 0):.4f}")
        print(f"  Citation Recall: {avg_metrics.get('citation_recall', 0):.4f}")
        print(f"  Citation F1: {avg_metrics.get('citation_f1', 0):.4f}")
        
        print(f"\nFaithfulness:")
        print(f"  Faithfulness Score: {avg_metrics.get('faithfulness', 0):.4f}")
        print(f"  Hallucination Rate: {avg_metrics.get('hallucination_rate', 0):.4f}")
        
        return avg_metrics
    else:
        logger.error("No metrics calculated - check input data")
        return {}


def run_ablation_study(args, logger):
    """Run ablation study."""
    logger.info("Running ablation study")
    
    # Load ablation configuration
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Define evaluation function
    def dummy_eval_function(config):
        """Dummy evaluation function - replace with actual implementation."""
        # In practice, this would run your system with the given config
        # and return metrics
        import random
        
        # Simulated metrics
        return {
            "accuracy": random.uniform(0.5, 0.9),
            "precision": random.uniform(0.6, 0.95),
            "recall": random.uniform(0.4, 0.85),
            "f1": random.uniform(0.5, 0.9),
            "latency": random.uniform(0.1, 2.0)
        }
    
    # Create ablation config
    ablation_config = AblationConfig(
        name=config_data.get("name", "ablation_study"),
        description=config_data.get("description", ""),
        components=config_data.get("components", {}),
        baseline=config_data.get("baseline", {}),
        metrics=config_data.get("metrics", ["accuracy", "f1"]),
        dataset=config_data.get("dataset", "test"),
        num_runs=config_data.get("num_runs", 3),
        random_seed=config_data.get("random_seed", 42)
    )
    
    # Run ablation
    runner = AblationRunner(experiment_dir=args.output)
    results = runner.run_ablation(ablation_config, dummy_eval_function)
    
    logger.info(f"Ablation study complete")
    logger.info(f"Results saved to: {args.output}")
    
    return results


def run_grid_search(args, logger):
    """Run grid search."""
    logger.info("Running grid search")
    
    # Load parameter grid and base config
    param_grid = load_json_file(args.param_grid)
    base_config = load_json_file(args.base_config)
    
    # Define evaluation function
    def dummy_eval_function(config):
        """Dummy evaluation function - replace with actual implementation."""
        import random
        
        # Simulated metrics based on configuration
        accuracy = 0.7
        
        # Simulate effect of hyperparameters
        if "learning_rate" in config:
            lr = config["learning_rate"]
            accuracy += (0.1 if 0.001 <= lr <= 0.01 else -0.1)
        
        if "batch_size" in config:
            bs = config["batch_size"]
            accuracy += (0.05 if 32 <= bs <= 128 else -0.05)
        
        # Add noise
        accuracy += random.uniform(-0.05, 0.05)
        
        return {
            "accuracy": max(0.1, min(0.99, accuracy)),
            "loss": 1.0 - accuracy,
            "training_time": random.uniform(10, 100)
        }
    
    # Run grid search
    runner = GridSearchRunner(experiment_dir=args.output)
    results_df = runner.run_grid_search(
        param_grid=param_grid,
        base_config=base_config,
        eval_function=dummy_eval_function,
        num_runs=2,
        random_seed=42
    )
    
    # Save results
    output_file = Path(args.output) / "grid_search_results.csv"
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"Grid search complete")
    logger.info(f"Results saved to: {output_file}")
    
    return results_df


def main():
    """Main function to run evaluation."""
    args = parse_arguments()
    logger = setup_logging(getattr(logging, args.log_level))
    
    if args.command == "retrieval":
        results = run_retrieval_evaluation(args, logger)
    elif args.command == "rag":
        results = run_rag_evaluation(args, logger)
    elif args.command == "ablation":
        results = run_ablation_study(args, logger)
    elif args.command == "grid":
        results = run_grid_search(args, logger)
    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)
    
    logger.info("Evaluation complete!")
    return results


if __name__ == "__main__":
    main()
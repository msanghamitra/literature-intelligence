"""
Ablation study runner for systematic evaluation of components.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime
import hashlib


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    name: str
    description: str
    components: Dict[str, List[Any]]  # Component name -> list of variants
    baseline: Dict[str, Any]  # Baseline configuration
    metrics: List[str]  # Metrics to track
    dataset: str  # Dataset to use
    num_runs: int  # Number of runs per configuration
    random_seed: int  # Random seed for reproducibility


@dataclass
class AblationResult:
    """Result of ablation study."""
    config_hash: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    std_errors: Dict[str, float]
    run_times: List[float]
    metadata: Dict[str, Any]


class AblationRunner:
    """
    Runner for ablation studies to evaluate component contributions.
    """
    
    def __init__(self, 
                 experiment_dir: str = "experiments",
                 cache_results: bool = True):
        """
        Initialize ablation runner.
        
        Args:
            experiment_dir: Directory to store experiment results
            cache_results: Whether to cache results to avoid recomputation
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        self.cache_results = cache_results
        self.results_cache = {}
        
        # Create subdirectories
        self.config_dir = self.experiment_dir / "configs"
        self.results_dir = self.experiment_dir / "results"
        self.plots_dir = self.experiment_dir / "plots"
        
        for dir_path in [self.config_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def run_ablation(self, 
                     config: AblationConfig,
                     eval_function: callable) -> List[AblationResult]:
        """
        Run ablation study.
        
        Args:
            config: Ablation study configuration
            eval_function: Function that takes config dict and returns metrics
            
        Returns:
            List of ablation results
        """
        print(f"Starting ablation study: {config.name}")
        print(f"Components: {list(config.components.keys())}")
        print(f"Number of configurations: {self._count_configurations(config)}")
        
        # Generate all configurations
        all_configs = self._generate_configurations(config)
        
        # Run evaluations
        results = []
        for i, config_dict in enumerate(all_configs, 1):
            print(f"\nEvaluating configuration {i}/{len(all_configs)}")
            print(f"Config: {self._config_to_string(config_dict)}")
            
            result = self._evaluate_configuration(config_dict, config, eval_function)
            results.append(result)
            
            # Save intermediate results
            if i % 5 == 0 or i == len(all_configs):
                self._save_results(results, config.name)
        
        # Analyze results
        self._analyze_results(results, config)
        
        return results
    
    def run_component_ablation(self,
                              base_config: Dict[str, Any],
                              components_to_ablate: List[str],
                              eval_function: callable,
                              component_variants: Dict[str, List[Any]] = None) -> Dict[str, AblationResult]:
        """
        Run ablation study for specific components.
        
        Args:
            base_config: Base configuration
            components_to_ablate: List of component names to ablate
            eval_function: Evaluation function
            component_variants: Optional specific variants for each component
            
        Returns:
            Dictionary mapping component names to ablation results
        """
        if component_variants is None:
            component_variants = {}
        
        results = {}
        
        for component in components_to_ablate:
            print(f"\nAblating component: {component}")
            
            # Get variants for this component
            if component in component_variants:
                variants = component_variants[component]
            else:
                # Default variants: remove, baseline, enhanced
                variants = [None, "baseline", "enhanced"]
            
            component_results = []
            
            for variant in variants:
                config = base_config.copy()
                if variant is None:
                    # Remove component
                    if component in config:
                        del config[component]
                    config_name = f"{component}_removed"
                else:
                    # Set component variant
                    config[component] = variant
                    config_name = f"{component}_{variant}"
                
                print(f"  Testing: {config_name}")
                
                # Evaluate
                try:
                    metrics = eval_function(config)
                    
                    result = AblationResult(
                        config_hash=hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8],
                        config=config,
                        metrics=metrics,
                        std_errors={k: 0.0 for k in metrics.keys()},  # Single run
                        run_times=[0.0],  # Placeholder
                        metadata={"component": component, "variant": variant}
                    )
                    
                    component_results.append(result)
                except Exception as e:
                    print(f"  Error evaluating {config_name}: {e}")
            
            # Store best result for this component
            if component_results:
                # Sort by primary metric (assume first metric is primary)
                primary_metric = list(component_results[0].metrics.keys())[0]
                component_results.sort(key=lambda x: x.metrics[primary_metric], reverse=True)
                results[component] = component_results[0]
        
        return results
    
    def _generate_configurations(self, config: AblationConfig) -> List[Dict[str, Any]]:
        """Generate all configurations for ablation study."""
        from itertools import product
        
        # Start with baseline
        all_configs = [config.baseline.copy()]
        
        # Generate all combinations for components being ablated
        component_names = list(config.components.keys())
        component_values = list(config.components.values())
        
        for combination in product(*component_values):
            new_config = config.baseline.copy()
            for name, value in zip(component_names, combination):
                # Handle nested components
                if '.' in name:
                    # Nested component (e.g., "retriever.type")
                    parts = name.split('.')
                    current = new_config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    new_config[name] = value
            
            # Only add if different from baseline
            if new_config != config.baseline:
                all_configs.append(new_config)
        
        return all_configs
    
    def _evaluate_configuration(self, 
                               config_dict: Dict[str, Any],
                               ablation_config: AblationConfig,
                               eval_function: callable) -> AblationResult:
        """Evaluate a single configuration."""
        config_hash = self._hash_config(config_dict)
        
        # Check cache
        if self.cache_results and config_hash in self.results_cache:
            print("  Using cached result")
            return self.results_cache[config_hash]
        
        # Run evaluation multiple times
        all_metrics = []
        run_times = []
        
        for run in range(ablation_config.num_runs):
            print(f"  Run {run + 1}/{ablation_config.num_runs}")
            
            # Set random seed for reproducibility
            seed = ablation_config.random_seed + run
            if 'random_seed' in config_dict:
                config_dict['random_seed'] = seed
            
            start_time = datetime.now()
            
            try:
                metrics = eval_function(config_dict)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  Error in run {run + 1}: {e}")
                # Use zeros for failed runs
                all_metrics.append({k: 0.0 for k in ablation_config.metrics})
            
            run_time = (datetime.now() - start_time).total_seconds()
            run_times.append(run_time)
        
        # Aggregate metrics across runs
        aggregated_metrics = {}
        std_errors = {}
        
        if all_metrics:
            for metric in ablation_config.metrics:
                values = [m.get(metric, 0.0) for m in all_metrics]
                aggregated_metrics[metric] = np.mean(values)
                std_errors[metric] = np.std(values) / np.sqrt(len(values))
        
        # Create result
        result = AblationResult(
            config_hash=config_hash,
            config=config_dict.copy(),
            metrics=aggregated_metrics,
            std_errors=std_errors,
            run_times=run_times,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "num_runs": ablation_config.num_runs,
                "avg_run_time": np.mean(run_times) if run_times else 0.0,
            }
        )
        
        # Cache result
        if self.cache_results:
            self.results_cache[config_hash] = result
        
        return result
    
    def _analyze_results(self, results: List[AblationResult], config: AblationConfig):
        """Analyze and visualize ablation results."""
        print(f"\n{'='*60}")
        print(f"Ablation Analysis: {config.name}")
        print(f"{'='*60}")
        
        # Convert to DataFrame for analysis
        df = self._results_to_dataframe(results, config)
        
        # Find baseline
        baseline_hash = self._hash_config(config.baseline)
        baseline_result = next((r for r in results if r.config_hash == baseline_hash), None)
        
        if baseline_result:
            print(f"\nBaseline Configuration:")
            print(f"  Config: {self._config_to_string(config.baseline)}")
            print(f"  Metrics: {baseline_result.metrics}")
            
            # Compare each configuration to baseline
            print(f"\nComparison to Baseline:")
            for result in results:
                if result.config_hash == baseline_hash:
                    continue
                
                print(f"\n  Configuration: {self._config_to_string(result.config)}")
                for metric in config.metrics:
                    baseline_value = baseline_result.metrics.get(metric, 0.0)
                    result_value = result.metrics.get(metric, 0.0)
                    improvement = result_value - baseline_value
                    improvement_pct = (improvement / baseline_value * 100) if baseline_value != 0 else 0
                    
                    print(f"    {metric}: {result_value:.4f} vs {baseline_value:.4f} "
                          f"(Δ={improvement:+.4f}, {improvement_pct:+.1f}%)")
        
        # Save analysis
        self._save_analysis(df, config)
        
        # Create visualizations
        self._create_visualizations(df, config)
    
    def _results_to_dataframe(self, results: List[AblationResult], config: AblationConfig) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        
        for result in results:
            row = {}
            
            # Add configuration parameters
            for component, variants in config.components.items():
                if '.' in component:
                    # Nested component
                    parts = component.split('.')
                    value = result.config
                    for part in parts:
                        value = value.get(part, {})
                    row[component] = str(value)
                else:
                    row[component] = str(result.config.get(component, "default"))
            
            # Add metrics
            for metric, value in result.metrics.items():
                row[metric] = value
            
            # Add metadata
            row['config_hash'] = result.config_hash
            row['avg_run_time'] = np.mean(result.run_times) if result.run_times else 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _save_results(self, results: List[AblationResult], experiment_name: str):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                "config_hash": result.config_hash,
                "config": result.config,
                "metrics": result.metrics,
                "std_errors": result.std_errors,
                "run_times": result.run_times,
                "metadata": result.metadata,
            })
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def _save_analysis(self, df: pd.DataFrame, config: AblationConfig):
        """Save analysis to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.results_dir / f"{config.name}_analysis_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary statistics
        summary = df.describe()
        summary_path = self.results_dir / f"{config.name}_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Ablation Study: {config.name}\n")
            f.write(f"Date: {timestamp}\n")
            f.write(f"Number of configurations: {len(df)}\n\n")
            f.write("Summary Statistics:\n")
            f.write(summary.to_string())
        
        print(f"Analysis saved to {csv_path}")
        print(f"Summary saved to {summary_path}")
    
    def _create_visualizations(self, df: pd.DataFrame, config: AblationConfig):
        """Create visualizations of ablation results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            sns.set_style("whitegrid")
            plt.figure(figsize=(12, 8))
            
            # Create one plot per metric
            for i, metric in enumerate(config.metrics[:4]):  # Limit to first 4 metrics
                plt.subplot(2, 2, i + 1)
                
                # Group by each component and plot
                for component in list(config.components.keys())[:3]:  # Limit to first 3 components
                    if component in df.columns:
                        component_df = df.groupby(component)[metric].mean().reset_index()
                        plt.plot(component_df[component], component_df[metric], 
                                marker='o', label=component)
                
                plt.title(f"{metric} by Component")
                plt.xlabel("Component Value")
                plt.ylabel(metric)
                plt.legend()
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.plots_dir / f"{config.name}_ablation_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization saved to {plot_path}")
            
        except ImportError:
            print("Matplotlib/seaborn not available. Skipping visualizations.")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def _hash_config(self, config_dict: Dict[str, Any]) -> str:
        """Create hash for configuration."""
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _config_to_string(self, config_dict: Dict[str, Any]) -> str:
        """Convert configuration to readable string."""
        parts = []
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                for subkey, subvalue in value.items():
                    parts.append(f"{key}.{subkey}={subvalue}")
            else:
                parts.append(f"{key}={value}")
        
        return ", ".join(parts[:5]) + ("..." if len(parts) > 5 else "")
    
    def _count_configurations(self, config: AblationConfig) -> int:
        """Count total number of configurations."""
        count = 1  # Baseline
        
        for variants in config.components.values():
            count *= len(variants)
        
        return count


class GridSearchRunner(AblationRunner):
    """
    Specialized runner for grid search over hyperparameters.
    """
    
    def run_grid_search(self,
                       param_grid: Dict[str, List[Any]],
                       base_config: Dict[str, Any],
                       eval_function: callable,
                       num_runs: int = 3,
                       random_seed: int = 42) -> pd.DataFrame:
        """
        Run grid search over hyperparameters.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
            base_config: Base configuration
            eval_function: Evaluation function
            num_runs: Number of runs per configuration
            random_seed: Random seed
            
        Returns:
            DataFrame with results
        """
        from itertools import product
        
        print(f"Starting grid search over {len(param_grid)} parameters")
        print(f"Total configurations: {self._count_param_combinations(param_grid)}")
        
        all_results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for i, combination in enumerate(product(*param_values), 1):
            config = base_config.copy()
            for name, value in zip(param_names, combination):
                # Set parameter value
                if '.' in name:
                    # Nested parameter
                    parts = name.split('.')
                    current = config
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    config[name] = value
            
            print(f"\nConfiguration {i}: {self._config_to_string(config)}")
            
            # Evaluate
            metrics_list = []
            run_times = []
            
            for run in range(num_runs):
                seed = random_seed + run
                config['random_seed'] = seed
                
                start_time = datetime.now()
                try:
                    metrics = eval_function(config)
                    metrics_list.append(metrics)
                except Exception as e:
                    print(f"  Error in run {run + 1}: {e}")
                    metrics_list.append({})
                
                run_times.append((datetime.now() - start_time).total_seconds())
            
            # Aggregate metrics
            if metrics_list:
                aggregated = {}
                for key in metrics_list[0].keys():
                    values = [m.get(key, 0.0) for m in metrics_list]
                    aggregated[key] = np.mean(values)
                    aggregated[f"{key}_std"] = np.std(values)
                
                # Create result row
                row = config.copy()
                row.update(aggregated)
                row['avg_run_time'] = np.mean(run_times)
                row['num_runs'] = num_runs
                
                all_results.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Find best configuration for each metric
        print(f"\n{'='*60}")
        print("Grid Search Results")
        print(f"{'='*60}")
        
        for metric in param_grid.keys():
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_value = df.loc[best_idx, metric]
                best_config = df.loc[best_idx, param_names].to_dict()
                
                print(f"\nBest {metric}: {best_value:.4f}")
                print(f"Configuration: {best_config}")
        
        return df
    
    def _count_param_combinations(self, param_grid: Dict[str, List[Any]]) -> int:
        """Count total number of parameter combinations."""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
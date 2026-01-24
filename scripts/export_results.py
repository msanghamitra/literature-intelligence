"""
Batch script to export evaluation results in various formats.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import yaml
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'export_results_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export evaluation results in various formats")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSON file with results")
    parser.add_argument("--output-dir", type=str, default="exports",
                       help="Output directory for exported files")
    parser.add_argument("--format", type=str, nargs="+", 
                       default=["json", "csv", "markdown"],
                       choices=["json", "csv", "markdown", "html", "latex", "yaml"],
                       help="Output formats")
    parser.add_argument("--include-raw", action="store_true",
                       help="Include raw data in exports")
    parser.add_argument("--aggregate", action="store_true",
                       help="Aggregate multiple result files")
    parser.add_argument("--compare", type=str, nargs="+",
                       help="Multiple result files to compare")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    return parser.parse_args()


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def export_json(results: Dict, output_path: Path, filename: str):
    """Export results as JSON."""
    filepath = output_path / f"{filename}.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return filepath


def export_csv(results: Dict, output_path: Path, filename: str):
    """Export results as CSV."""
    # Flatten nested dictionaries for CSV
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Handle different result structures
    if isinstance(results, list):
        # List of results
        flat_results = [flatten_dict(r) for r in results]
        df = pd.DataFrame(flat_results)
    elif isinstance(results, dict):
        # Single result dictionary
        flat_result = flatten_dict(results)
        df = pd.DataFrame([flat_result])
    else:
        raise ValueError(f"Unsupported results type: {type(results)}")
    
    filepath = output_path / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    return filepath


def export_markdown(results: Dict, output_path: Path, filename: str):
    """Export results as Markdown."""
    filepath = output_path / f"{filename}.md"
    
    with open(filepath, 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        if isinstance(results, dict):
            # Single result
            f.write("## Summary\n\n")
            
            # Create a table for key metrics
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for key, value in results.items():
                if isinstance(value, (int, float, str, bool)):
                    f.write(f"| {key} | {value} |\n")
                elif isinstance(value, dict):
                    # Nested dictionary - handle specially
                    f.write(f"| **{key}** | |\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, str, bool)):
                            f.write(f"|   {sub_key} | {sub_value} |\n")
        
        elif isinstance(results, list):
            # Multiple results
            f.write(f"## Results ({len(results)} entries)\n\n")
            
            # Create a table for all results
            if results:
                # Get all keys from first result
                first_result = results[0]
                if isinstance(first_result, dict):
                    headers = list(first_result.keys())
                    
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                    
                    for result in results:
                        row = []
                        for header in headers:
                            value = result.get(header, "")
                            if isinstance(value, (dict, list)):
                                value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                            row.append(str(value))
                        f.write("| " + " | ".join(row) + " |\n")
    
    return filepath


def export_html(results: Dict, output_path: Path, filename: str):
    """Export results as HTML."""
    filepath = output_path / f"{filename}.html"
    
    # Convert to pandas DataFrame for easy HTML export
    if isinstance(results, list):
        df = pd.DataFrame(results)
    elif isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame()
    
    html_content = df.to_html(index=False, classes='table table-striped', border=0)
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Results</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            .timestamp {{ color: #666; font-style: italic; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Results</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total results: {len(df)}</p>
        </div>
        
        {html_content}
    </body>
    </html>
    """
    
    with open(filepath, 'w') as f:
        f.write(full_html)
    
    return filepath


def export_latex(results: Dict, output_path: Path, filename: str):
    """Export results as LaTeX table."""
    filepath = output_path / f"{filename}.tex"
    
    # Convert to pandas DataFrame
    if isinstance(results, list):
        df = pd.DataFrame(results)
    elif isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = pd.DataFrame()
    
    latex_content = df.to_latex(index=False, caption="Evaluation Results", label="tab:results")
    
    with open(filepath, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\begin{document}\n\n")
        f.write(latex_content)
        f.write("\n\\end{document}\n")
    
    return filepath


def export_yaml(results: Dict, output_path: Path, filename: str):
    """Export results as YAML."""
    filepath = output_path / f"{filename}.yaml"
    
    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    return filepath


def compare_results(result_files: List[str]) -> Dict:
    """Compare multiple result files."""
    comparisons = {}
    
    for filepath in result_files:
        results = load_results(filepath)
        filename = Path(filepath).stem
        
        # Extract key metrics (simplified)
        if isinstance(results, dict):
            comparisons[filename] = {
                "file": filepath,
                "metrics": {k: v for k, v in results.items() 
                          if isinstance(v, (int, float))}
            }
        elif isinstance(results, list):
            # Average metrics if it's a list of results
            avg_metrics = {}
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in avg_metrics:
                            avg_metrics[key] = []
                        avg_metrics[key].append(value)
            
            comparisons[filename] = {
                "file": filepath,
                "metrics": {k: np.mean(v) for k, v in avg_metrics.items()}
            }
    
    return comparisons


def main():
    """Main function to export results."""
    args = parse_arguments()
    logger = setup_logging(getattr(logging, args.log_level))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate filename based on input
    input_stem = Path(args.input).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{input_stem}_{timestamp}"
    
    try:
        # Load results
        logger.info(f"Loading results from: {args.input}")
        results = load_results(args.input)
        
        # Compare results if requested
        if args.compare:
            logger.info(f"Comparing {len(args.compare)} result files")
            comparisons = compare_results([args.input] + args.compare)
            
            # Add comparisons to results
            results = {
                "primary_results": results,
                "comparisons": comparisons
            }
        
        # Export in requested formats
        exported_files = []
        
        for format_type in args.format:
            logger.info(f"Exporting as {format_type.upper()}")
            
            try:
                if format_type == "json":
                    filepath = export_json(results, output_dir, base_filename)
                elif format_type == "csv":
                    filepath = export_csv(results, output_dir, base_filename)
                elif format_type == "markdown":
                    filepath = export_markdown(results, output_dir, base_filename)
                elif format_type == "html":
                    filepath = export_html(results, output_dir, base_filename)
                elif format_type == "latex":
                    filepath = export_latex(results, output_dir, base_filename)
                elif format_type == "yaml":
                    filepath = export_yaml(results, output_dir, base_filename)
                else:
                    logger.warning(f"Unknown format: {format_type}")
                    continue
                
                exported_files.append(filepath)
                logger.info(f"  Exported to: {filepath}")
                
            except Exception as e:
                logger.error(f"  Failed to export as {format_type}: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Input file: {args.input}")
        print(f"Output directory: {output_dir}")
        print(f"Formats exported: {', '.join(args.format)}")
        print(f"Total files created: {len(exported_files)}")
        
        if exported_files:
            print(f"\nExported files:")
            for filepath in exported_files:
                print(f"  • {filepath}")
        
        logger.info("Export complete!")
        
    except Exception as e:
        logger.error(f"Error during export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
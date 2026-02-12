#!/usr/bin/env python3
"""
Data Collector for LADy Experiment Results

A tool to collect and consolidate agg.ad.pred.eval.mean.csv files from LADy 
experiment output directories into a single CSV file for analysis.
"""

import argparse
import os
import sys
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging


class ExperimentDataCollector:
    """Collects and consolidates LADy experiment CSV files."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save consolidated CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def find_csv_files(self, root_path: str) -> List[Tuple[str, Dict[str, str]]]:
        """
        Find all agg.ad.pred.eval.mean.csv files and extract metadata from paths.
        
        Args:
            root_path: Root directory to search
            
        Returns:
            List of tuples containing (file_path, metadata_dict)
        """
        csv_files = []
        root = Path(root_path)
        
        if not root.exists():
            self.logger.error(f"Directory does not exist: {root_path}")
            return csv_files
        
        # Search for CSV files using glob pattern
        csv_pattern = "**/agg.ad.pred.eval.mean.csv"
        found_files = list(root.glob(csv_pattern))
        
        self.logger.info(f"Found {len(found_files)} CSV files")
        
        for csv_file in found_files:
            metadata = self._extract_metadata_from_path(csv_file, root)
            if metadata:
                csv_files.append((str(csv_file), metadata))
                self.logger.debug(f"Found: {csv_file} -> {metadata}")
        
        return csv_files
    
    def _extract_metadata_from_path(self, csv_path: Path, root: Path) -> Optional[Dict[str, str]]:
        """
        Extract metadata from the file path structure.
        
        Expected structure: root/model_name/llm_model_name/agg.ad.pred.eval.mean.csv
        
        Args:
            csv_path: Path to the CSV file
            root: Root directory path
            
        Returns:
            Dictionary with extracted metadata or None if extraction fails
        """
        try:
            # Get relative path from root
            rel_path = csv_path.relative_to(root)
            parts = rel_path.parts
            
            if len(parts) < 3:
                self.logger.warning(f"Unexpected path structure: {rel_path}")
                return None
            
            # Extract components
            # parts[-3] = model_name (bert, ctm, lda, etc.)
            # parts[-2] = llm_model_name (anthropic-haiku3.5-25, etc.)
            # parts[-1] = filename (agg.ad.pred.eval.mean.csv)
            
            model_name = parts[-3]
            llm_model = parts[-2]
            
            # Extract experiment name from root directory name
            experiment_name = root.name
            
            metadata = {
                'experiment': experiment_name,
                'model_name': model_name,
                'llm_model': llm_model,
                'file_path': str(csv_path)
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {csv_path}: {e}")
            return None
    
    def consolidate_csv_files(self, csv_files: List[Tuple[str, Dict[str, str]]], 
                            output_filename: str = None, exp_type: Optional[int] = None) -> str:
        """
        Consolidate all CSV files into a single DataFrame and save.
        
        Args:
            csv_files: List of (file_path, metadata) tuples
            output_filename: Custom output filename (optional)
            exp_type: Experiment type for specialized layouts (1, 2, 3, etc.)
            
        Returns:
            Path to the consolidated CSV file
        """
        if not csv_files:
            self.logger.error("No CSV files to consolidate")
            return None
        
        consolidated_data = []
        
        for file_path, metadata in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Add metadata columns
                for key, value in metadata.items():
                    df[key] = value
                
                consolidated_data.append(df)
                self.logger.debug(f"Processed: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue
        
        if not consolidated_data:
            self.logger.error("No valid CSV files were processed")
            return None
        
        # Combine all DataFrames
        final_df = pd.concat(consolidated_data, ignore_index=True)
        
        # Apply experiment-specific formatting if requested
        if exp_type == 1:
            final_df = self._format_exp1_layout(final_df)
        elif exp_type == 2:
            final_df = self._format_exp2_layout(final_df)
        elif exp_type == 3:
            final_df = self._format_exp3_layout(final_df)
        else:
            # Default layout - reorder columns to put metadata first
            metadata_cols = ['experiment', 'model_name', 'llm_model']
            other_cols = [col for col in final_df.columns if col not in metadata_cols and col != 'file_path']
            final_cols = metadata_cols + other_cols + ['file_path']
            final_df = final_df[final_cols]
        
        # Generate output filename
        if not output_filename:
            experiment_name = metadata.get('experiment', 'unknown')
            output_filename = f"{experiment_name}_consolidated_results.csv"
        
        output_path = self.output_dir / output_filename
        
        # Save to CSV
        if exp_type == 1 or exp_type == 2 or exp_type == 3:
            # For exp1, exp2, and exp3 layouts, save without column headers since we have custom structure
            final_df.to_csv(output_path, index=False, header=False)
        else:
            # For default layout, save with headers
            final_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Consolidated {len(csv_files)} files into {output_path}")
        self.logger.info(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
        
        return str(output_path)
    
    def _format_exp1_layout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format data for Experiment Type 1 layout.
        
        Creates a pivot table structure with:
        - Header rows with experiment name and type
        - Model name rows with metrics as column headers
        - LLM model rows with actual metric values
        
        Args:
            df: Input DataFrame with standard format
            
        Returns:
            Formatted DataFrame for exp1 layout
        """
        # Drop fold columns and file_path, keep only mean values
        df_clean = df[['experiment', 'model_name', 'llm_model', 'metric', 'mean']].copy()
        
        # Create pivot table with metrics as columns
        pivot_df = df_clean.pivot_table(
            index=['experiment', 'model_name', 'llm_model'],
            columns='metric',
            values='mean',
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        # Get metric columns in a consistent order (dynamically from the data)
        metric_cols = [col for col in pivot_df.columns if col not in ['experiment', 'model_name', 'llm_model']]
        
        # Sort metrics with natural/numeric ordering (P_1, P_5, P_10, P_100)
        # This ensures logical ordering regardless of which metrics are present
        def natural_sort_key(metric):
            """Sort metrics naturally: P_1, P_5, P_10, P_100, recall_1, recall_5, etc."""
            import re
            # Split metric into base name and number
            parts = re.split(r'(\d+)', metric)
            # Convert numeric parts to integers for proper sorting
            return [int(part) if part.isdigit() else part for part in parts]
        
        metric_cols = sorted(metric_cols, key=natural_sort_key)
        
        # Log the metrics found for debugging
        self.logger.debug(f"Metrics found in data: {metric_cols}")
        
        # Create the exp1 format structure
        result_rows = []
        
        # Get experiment name for header
        experiment_name = pivot_df['experiment'].iloc[0] if not pivot_df.empty else 'unknown'
        
        # Calculate total columns needed
        total_cols = len(metric_cols) + 1  # +1 for model name column
        
        # Header row: experiment name in first column, experiment value in second
        header_row = ['experiment', experiment_name] + [''] * (total_cols - 2)
        result_rows.append(header_row)
        
        # Type row: type in first column, 1 in second
        type_row = ['type', '1'] + [''] * (total_cols - 2)
        result_rows.append(type_row)
        
        # Empty row
        empty_row = [''] * total_cols
        result_rows.append(empty_row)
        
        # Group by model_name
        for model_name in sorted(pivot_df['model_name'].unique()):
            model_data = pivot_df[pivot_df['model_name'] == model_name]
            
            # Model name row with metrics as headers
            model_header_row = [model_name] + metric_cols
            result_rows.append(model_header_row)
            
            # LLM model rows with actual values
            for _, row in model_data.iterrows():
                llm_model = row['llm_model']
                metric_values = [str(row[col]) if pd.notna(row[col]) else '0' for col in metric_cols]
                llm_row = [llm_model] + metric_values
                result_rows.append(llm_row)
            
            # Add empty row between model groups
            result_rows.append(empty_row)
        
        # Convert to DataFrame with generic column names
        columns = ['col_' + str(i) for i in range(total_cols)]
        result_df = pd.DataFrame(result_rows, columns=columns)
        
        return result_df
    
    def _format_exp2_layout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format data for Experiment Type 2 layout (Top Models Analysis).
        
        Creates a custom layout with:
        - Header rows with experiment metadata
        - Model names as main column headers
        - LLM model names as sub-headers
        - Dataset sizes as row indices with P@5 values
        
        Args:
            df: Input DataFrame with standard format
            
        Returns:
            Formatted DataFrame for exp2 layout
        """
        # Drop fold columns and file_path, keep only mean values
        df_clean = df[['experiment', 'model_name', 'llm_model', 'metric', 'mean']].copy()
        
        # For exp2, extract dataset size from LLM model name (e.g., "anthropic-haiku3.5-1300" -> 1300)
        import re
        
        def extract_dataset_size(llm_model_name):
            """Extract dataset size from LLM model name."""
            # Look for dataset size at the end of the name
            size_match = re.search(r'-(\d+)$', str(llm_model_name))
            if size_match:
                return int(size_match.group(1))
            return None
        
        df_clean['dataset_size'] = df_clean['llm_model'].apply(extract_dataset_size)
        
        # Filter out rows where we couldn't determine dataset size
        df_clean = df_clean[df_clean['dataset_size'].notna()]
        
        if df_clean.empty:
            self.logger.warning("Could not extract dataset sizes from LLM model names. Using basic layout.")
            return self._format_basic_layout(df)
        
        # Extract only P_5 metric (k=5) for exp2
        df_p5 = df_clean[df_clean['metric'] == 'P_5'].copy()
        
        if df_p5.empty:
            self.logger.warning("No P_5 metric found. Using first available metric.")
            first_metric = df_clean['metric'].iloc[0]
            df_p5 = df_clean[df_clean['metric'] == first_metric].copy()
        
        # Extract just the model name from LLM model strings
        # e.g., "anthropic-haiku3.5-1300" -> "haiku3.5"
        def extract_llm_model_name(full_name):
            """Extract simplified LLM model name."""
            # Split by "-" and take the middle part(s)
            parts = full_name.split('-')
            
            # Expected format: [provider]-[model_name]-[size]
            if len(parts) >= 3 and parts[-1].isdigit():
                # Join all middle parts (in case model name has hyphens)
                return '-'.join(parts[1:-1])
            elif len(parts) >= 2:
                # If no size suffix, just remove provider
                return '-'.join(parts[1:])
            
            return full_name
        
        df_p5['llm_model_clean'] = df_p5['llm_model'].apply(extract_llm_model_name)
        
        # Create pivot table
        pivot_df = df_p5.pivot_table(
            index='dataset_size',
            columns=['model_name', 'llm_model_clean'],
            values='mean',
            fill_value=0
        )
        
        # Get available models from data
        available_models = sorted(df_p5['model_name'].unique())
        
        # Dynamically build model-to-LLM mapping from the data
        model_llm_mapping = {}
        for model in available_models:
            # Get unique LLM models for this base model
            llm_models = df_p5[df_p5['model_name'] == model]['llm_model_clean'].unique()
            model_llm_mapping[model] = sorted(llm_models)
        
        # Create the result structure
        result_rows = []
        
        # Calculate total columns needed
        total_cols = 1  # First column for dataset size
        for model in available_models:
            total_cols += len(model_llm_mapping[model])
        
        # Header rows
        result_rows.append(['experiment', 'exp2_top_models'] + [''] * (total_cols - 2))
        result_rows.append(['type', '2'] + [''] * (total_cols - 2))
        result_rows.append(['k_value', '5'] + [''] * (total_cols - 2))
        result_rows.append([''] * total_cols)  # Empty row
        
        # Model header row
        model_header = ['']
        for model in available_models:
            llm_count = len(model_llm_mapping[model])
            model_header.append(model)
            if llm_count > 1:
                model_header.extend([''] * (llm_count - 1))
        result_rows.append(model_header)
        
        # LLM model header row
        llm_header = ['']
        for model in available_models:
            for llm in model_llm_mapping[model]:
                llm_header.append(llm)
        result_rows.append(llm_header)
        
        # Data rows for each dataset size
        for size in sorted(pivot_df.index):
            row = [str(size)]
            for model in available_models:
                for llm in model_llm_mapping[model]:
                    # Try to find the value in pivot table
                    if (model, llm) in pivot_df.columns:
                        value = pivot_df.loc[size, (model, llm)]
                        # Keep original precision
                        row.append(str(value))
                    else:
                        # If data not available, use 0 as placeholder
                        row.append("0")
            result_rows.append(row)
        
        # Convert to DataFrame with generic column names
        num_cols = len(result_rows[0])
        columns = ['col_' + str(i) for i in range(num_cols)]
        result_df = pd.DataFrame(result_rows, columns=columns)
        
        return result_df
    
    def _format_exp3_layout(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format data for Experiment Type 3 layout (Baseline Comparison).
        
        Uses the same layout structure as exp1 but with different header values:
        - experiment: "exp3_baseline_comparison"
        - type: "3"
        
        Excludes 700-size datasets since SemEval baselines only exist at 1300 and 2000.
        
        Args:
            df: Input DataFrame with standard format
            
        Returns:
            Formatted DataFrame for exp3 layout
        """
        # Drop fold columns and file_path, keep only mean values
        df_clean = df[['experiment', 'model_name', 'llm_model', 'metric', 'mean']].copy()
        
        # Extract dataset size from LLM model name and filter out 700 sizes
        import re
        
        def extract_dataset_size(llm_model_name):
            """Extract dataset size from LLM model name."""
            size_match = re.search(r'-(\d+)$', str(llm_model_name))
            if size_match:
                return int(size_match.group(1))
            return None
        
        df_clean['dataset_size'] = df_clean['llm_model'].apply(extract_dataset_size)
        
        # Filter out 700-size datasets
        df_clean = df_clean[df_clean['dataset_size'] != 700]
        
        # Drop the temporary dataset_size column
        df_clean = df_clean.drop('dataset_size', axis=1)
        
        # Create pivot table with metrics as columns
        pivot_df = df_clean.pivot_table(
            index=['experiment', 'model_name', 'llm_model'],
            columns='metric',
            values='mean',
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns.name = None
        
        # Get metric columns in a consistent order (dynamically from the data)
        metric_cols = [col for col in pivot_df.columns if col not in ['experiment', 'model_name', 'llm_model']]
        
        # Sort metrics with natural/numeric ordering (P_1, P_5, P_10, P_100)
        # This ensures logical ordering regardless of which metrics are present
        def natural_sort_key(metric):
            """Sort metrics naturally: P_1, P_5, P_10, P_100, recall_1, recall_5, etc."""
            import re
            # Split metric into base name and number
            parts = re.split(r'(\d+)', metric)
            # Convert numeric parts to integers for proper sorting
            return [int(part) if part.isdigit() else part for part in parts]
        
        metric_cols = sorted(metric_cols, key=natural_sort_key)
        
        # Helper function to identify baseline models
        def is_baseline_model(llm_model_name):
            """Check if a model is a baseline model based on name patterns."""
            llm_lower = str(llm_model_name).lower()
            baseline_patterns = ['baseline', 'random', 'rnd', 'majority', 'default']
            return any(pattern in llm_lower for pattern in baseline_patterns)
        
        # Log the metrics found for debugging
        self.logger.debug(f"Metrics found in data: {metric_cols}")
        
        # Create the exp3 format structure
        result_rows = []
        
        # Calculate total columns needed
        total_cols = len(metric_cols) + 1  # +1 for model name column
        
        # Header row: experiment name in first column, exp3_baseline_comparison in second
        header_row = ['experiment', 'exp3_baseline_comparison'] + [''] * (total_cols - 2)
        result_rows.append(header_row)
        
        # Type row: type in first column, 3 in second
        type_row = ['type', '3'] + [''] * (total_cols - 2)
        result_rows.append(type_row)
        
        # Empty row
        empty_row = [''] * total_cols
        result_rows.append(empty_row)
        
        # Group by model_name
        for model_name in sorted(pivot_df['model_name'].unique()):
            model_data = pivot_df[pivot_df['model_name'] == model_name]
            
            # Model name row with metrics as headers
            model_header_row = [model_name] + metric_cols
            result_rows.append(model_header_row)
            
            # Sort LLM models: non-baselines first, then baselines
            # First separate into baseline and non-baseline groups
            baseline_models = []
            non_baseline_models = []
            
            for _, row in model_data.iterrows():
                llm_model = row['llm_model']
                if is_baseline_model(llm_model):
                    baseline_models.append(row)
                else:
                    non_baseline_models.append(row)
            
            # Add non-baseline models first (sorted alphabetically)
            non_baseline_models.sort(key=lambda x: x['llm_model'])
            for row in non_baseline_models:
                llm_model = row['llm_model']
                metric_values = [str(row[col]) if pd.notna(row[col]) else '0' for col in metric_cols]
                llm_row = [llm_model] + metric_values
                result_rows.append(llm_row)
            
            # Add baseline models at the bottom
            # Sort with RND models first, then other baselines alphabetically
            def baseline_sort_key(row):
                llm_name = str(row['llm_model']).lower()
                if 'rnd' in llm_name or 'random' in llm_name:
                    return (0, row['llm_model'])  # Priority 0 for RND
                else:
                    return (1, row['llm_model'])  # Priority 1 for others
            
            baseline_models.sort(key=baseline_sort_key)
            for row in baseline_models:
                llm_model = row['llm_model']
                metric_values = [str(row[col]) if pd.notna(row[col]) else '0' for col in metric_cols]
                llm_row = [llm_model] + metric_values
                result_rows.append(llm_row)
            
            # Add empty row between model groups
            result_rows.append(empty_row)
        
        # Convert to DataFrame with generic column names
        columns = ['col_' + str(i) for i in range(total_cols)]
        result_df = pd.DataFrame(result_rows, columns=columns)
        
        return result_df
    
    def _format_basic_layout(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic layout for when specialized formatting fails."""
        metadata_cols = ['experiment', 'model_name', 'llm_model']
        other_cols = [col for col in df.columns if col not in metadata_cols and col != 'file_path']
        final_cols = metadata_cols + other_cols + ['file_path']
        return df[final_cols]
    
    def generate_summary_report(self, csv_files: List[Tuple[str, Dict[str, str]]]) -> str:
        """
        Generate a summary report of the collected data.
        
        Args:
            csv_files: List of (file_path, metadata) tuples
            
        Returns:
            Summary report as string
        """
        if not csv_files:
            return "No CSV files found for analysis."
        
        # Extract metadata for analysis
        experiments = set()
        models = set()
        llm_models = set()
        
        for _, metadata in csv_files:
            experiments.add(metadata.get('experiment', 'unknown'))
            models.add(metadata.get('model_name', 'unknown'))
            llm_models.add(metadata.get('llm_model', 'unknown'))
        
        report = f"""
=== LADy Experiment Data Collection Summary ===

📊 Collection Results:
   Total CSV files found: {len(csv_files)}
   
🧪 Experiments: {len(experiments)}
   {', '.join(sorted(experiments))}
   
🤖 Models: {len(models)}
   {', '.join(sorted(models))}
   
🦾 LLM Models: {len(llm_models)}
   {', '.join(sorted(llm_models))}

📁 File Distribution:
"""
        
        # Create distribution table
        distribution = {}
        for _, metadata in csv_files:
            exp = metadata.get('experiment', 'unknown')
            model = metadata.get('model_name', 'unknown')
            if exp not in distribution:
                distribution[exp] = {}
            if model not in distribution[exp]:
                distribution[exp][model] = 0
            distribution[exp][model] += 1
        
        for exp in sorted(distribution.keys()):
            report += f"\n   {exp}:\n"
            for model, count in sorted(distribution[exp].items()):
                report += f"     - {model}: {count} files\n"
        
        return report


def main():
    """Main entry point for the data collector."""
    parser = argparse.ArgumentParser(
        description='Collect and consolidate LADy experiment CSV results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/experiment_output/toys_run3
  %(prog)s -i /path/to/experiment_output/toys_run3 -e 1 -o exp1_results.csv
  %(prog)s -i /path/to/experiment_output/toys_run3 --summary-only
  %(prog)s -i /path/to/experiment_output/toys_run3 --output-dir ./analysis
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input directory containing experiment results'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output CSV filename (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for consolidated files (default: output)'
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Only display summary, do not create consolidated CSV'
    )
    
    parser.add_argument(
        '-e', '--exp-type',
        type=int,
        choices=[1, 2, 3],
        help='Experiment type for specialized layouts (1=LLM ranking, 2=scaling, 3=baseline)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize collector
    collector = ExperimentDataCollector(args.output_dir)
    
    # Find CSV files
    print(f"🔍 Searching for CSV files in: {args.input}")
    csv_files = collector.find_csv_files(args.input)
    
    if not csv_files:
        print("❌ No CSV files found!")
        sys.exit(1)
    
    # Generate and display summary
    summary = collector.generate_summary_report(csv_files)
    print(summary)
    
    # Consolidate files unless summary-only
    if not args.summary_only:
        layout_info = ""
        if args.exp_type:
            layout_info = f" (Experiment Type {args.exp_type} layout)"
        
        print(f"\n📊 Consolidating CSV files{layout_info}...")
        output_path = collector.consolidate_csv_files(csv_files, args.output, args.exp_type)
        
        if output_path:
            print(f"✅ Success! Consolidated data saved to: {output_path}")
            if args.exp_type == 1:
                print(f"\n💡 Exp1 layout: Excel-ready pivot format with metrics as columns!")
            elif args.exp_type == 2:
                print(f"\n💡 Exp2 layout: Top models comparison with P@5 values across dataset sizes!")
            elif args.exp_type == 3:
                print(f"\n💡 Exp3 layout: Baseline comparison with same format as Exp1!")
            else:
                print(f"\n💡 You can now open '{output_path}' in Excel for analysis!")
        else:
            print("❌ Failed to consolidate CSV files")
            sys.exit(1)
    else:
        print("\n📋 Summary-only mode - no CSV file created")


if __name__ == "__main__":
    main()
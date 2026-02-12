# LADy Experiment Data Collector

A powerful tool for collecting and consolidating evaluation results from LADy (Large Language Model Aspect Discovery) experiments into Excel-ready CSV files for comprehensive analysis.

**Designed specifically for LADy experiment output analysis**, this tool automatically discovers, extracts, and consolidates `agg.ad.pred.eval.mean.csv` files from complex experiment directory structures, adding rich metadata for comparative analysis across models, LLMs, and experiments.

## 🚀 Features

### Core Functionality
- **Recursive File Discovery**: Automatically finds all `agg.ad.pred.eval.mean.csv` files in experiment directories
- **Metadata Extraction**: Extracts experiment name, model type, and LLM model from directory structure
- **Smart Consolidation**: Combines multiple CSV files into a single analysis-ready dataset
- **Excel-Compatible Output**: Generates clean CSV files optimized for Excel analysis

### Advanced Features
- **Flexible Directory Support**: Works with any LADy experiment output structure
- **Rich Summary Reports**: Detailed statistics about collected data
- **Error Resilience**: Continues processing even if some files are corrupted
- **Path Validation**: Ensures all found files match expected structure patterns
- **Configurable Output**: Custom output directories and filenames

### ✨ Key Benefits
- **Time Saving**: Eliminates manual CSV collection and consolidation
- **Analysis Ready**: Output files are immediately usable in Excel, R, or Python
- **Comprehensive Metadata**: Each row includes experiment context for easy filtering
- **Research Reproducibility**: Clear tracking of data sources and experiment parameters

## 📁 Project Structure

```
data-collector/
├── collect_data.py         # Main collection script
├── output/                 # Default output directory
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pandas library

### Dependencies
```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install manually
pip install pandas
```

**Note**: The tool uses minimal dependencies to ensure compatibility across different research environments.

## 🔧 Usage

### Basic Usage
```bash
# Collect all CSV files from an experiment directory
python3 collect_data.py -i /path/to/experiment_output/toys_run3

# Custom output filename
python3 collect_data.py -i /path/to/experiment_output/toys_run3 -o my_analysis.csv

# Custom output directory
python3 collect_data.py -i /path/to/experiment_output/toys_run3 --output-dir ./analysis
```

### Experiment-Specific Layouts

Different experiments may require specialized output formats for optimal analysis:

```bash
# Experiment Type 1: LLM Model Ranking (Pivot table format)
python3 collect_data.py -i /path/to/experiment_output/exp1_llm_ranking -e 1

# Experiment Type 2: Scaling Analysis (Coming soon)
python3 collect_data.py -i /path/to/experiment_output/exp2_scaling -e 2

# Experiment Type 3: Baseline Comparison (Coming soon)
python3 collect_data.py -i /path/to/experiment_output/exp3_baselines -e 3
```

### Analysis Mode
```bash
# Get summary without creating consolidated file
python3 collect_data.py -i /path/to/experiment_output/toys_run3 --summary-only

# Verbose mode for debugging
python3 collect_data.py -i /path/to/experiment_output/toys_run3 -v
```

### Expected Directory Structure

The tool expects LADy experiment output with this structure:
```
experiment_output/toys_run3/          ← Experiment name
├── bert/                             ← Model name
│   └── anthropic-haiku3.5-25/        ← LLM model
│       └── agg.ad.pred.eval.mean.csv ← Target file
├── ctm/                              ← Model name
│   └── anthropic-haiku3.5-25/        ← LLM model
│       └── agg.ad.pred.eval.mean.csv ← Target file
└── lda/                              ← Model name
    └── anthropic-haiku3.5-25/        ← LLM model
        └── agg.ad.pred.eval.mean.csv ← Target file
```

### Command Line Options

```bash
python3 collect_data.py [OPTIONS]

Required:
  -i, --input PATH         Input directory containing experiment results

Optional:
  -o, --output FILENAME    Output CSV filename (auto-generated if not specified)
  --output-dir PATH        Output directory for consolidated files (default: output)
  -e, --exp-type {1,2,3}   Experiment type for specialized layouts
  --summary-only           Only display summary, do not create consolidated CSV
  -v, --verbose            Enable verbose logging
  -h, --help               Show help message and exit

Experiment Types:
  1 = LLM Model Ranking (Pivot table format with metrics as columns)
  2 = Scaling Analysis (Coming soon)
  3 = Baseline Comparison (Coming soon)
```

## 📊 Output Format

### Default CSV Structure (Standard Layout)

The tool generates a consolidated CSV with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `experiment` | Experiment directory name | `toys_run3` |
| `model_name` | LADy model type | `bert`, `ctm`, `lda` |
| `llm_model` | LLM model identifier | `anthropic-haiku3.5-25` |
| `metric` | Evaluation metric name | `P_1`, `recall_5`, `ndcg_cut_10` |
| `fold0` | Fold-specific value | `0.1333333333333333` |
| `mean` | Mean value across folds | `0.1333333333333333` |
| `file_path` | Source file path | `/path/to/original/file.csv` |

### Example Output

```csv
experiment,model_name,llm_model,metric,fold0,mean,file_path
toys_run3,bert,anthropic-haiku3.5-25,P_1,0.0,0.0,/path/to/bert/anthropic-haiku3.5-25/agg.ad.pred.eval.mean.csv
toys_run3,bert,anthropic-haiku3.5-25,P_5,0.1333,0.1333,/path/to/bert/anthropic-haiku3.5-25/agg.ad.pred.eval.mean.csv
toys_run3,ctm,anthropic-haiku3.5-25,P_1,0.05,0.05,/path/to/ctm/anthropic-haiku3.5-25/agg.ad.pred.eval.mean.csv
```

### Experiment Type 1 Layout (LLM Model Ranking)

When using `-e 1`, the tool creates a pivot table format optimized for LLM model comparison:

**Structure:**
- **Header rows**: Experiment name and type identifier
- **Column A**: Model names (bert, ctm, lda, etc.) with LLM models as sub-rows
- **Columns B+**: Metrics as columns (P_1, P_5, P_10, recall_1, etc.)
- **Values**: Only mean values (fold columns and file paths removed)

**Example Output:**
```csv
experiment,toys_run3,,,,,,,,,,,,,,,,,,,
type,1,,,,,,,,,,,,,,,,,,,
,,,,,,,,,,,,,,,,,,,,
bert,P_1,P_5,P_10,P_100,map_cut_1,map_cut_5,map_cut_10,map_cut_100,ndcg_cut_1,ndcg_cut_5,ndcg_cut_10,ndcg_cut_100,recall_1,recall_5,recall_10,recall_100,success_1,success_5,success_10,success_100
anthropic-haiku3.5-25,0,0.1333,0.0667,0.0067,0,0.1694,0.1694,0.1694,0,0.2780,0.2780,0.2780,0,0.5833,0.5833,0.5833,0,0.6667,0.6667,0.6667
,,,,,,,,,,,,,,,,,,,,
ctm,P_1,P_5,P_10,P_100,map_cut_1,map_cut_5,map_cut_10,map_cut_100,ndcg_cut_1,ndcg_cut_5,ndcg_cut_10,ndcg_cut_100,recall_1,recall_5,recall_10,recall_100,success_1,success_5,success_10,success_100
anthropic-haiku3.5-25,0,0.1667,0.0833,0.0083,0,0.2222,0.2222,0.2222,0,0.3522,0.3522,0.3522,0,0.6667,0.6667,0.6667,0,0.8333,0.8333,0.8333
```

**Benefits:**
- **Excel PivotTable Ready**: Perfect for creating charts and comparisons
- **Model Grouping**: Clear visual separation between different aspect detection models
- **Natural Metric Ordering**: Metrics sorted logically (P_1, P_5, P_10, P_100, not P_1, P_10, P_100, P_5)
- **Dynamic Metrics**: Automatically adapts to whatever metrics are present in your data
- **No Generic Headers**: Clean CSV without unnecessary column headers
- **Analysis Friendly**: Ideal for statistical analysis and visualization

## 📈 Data Analysis Workflows

### Excel Analysis with Exp1 Layout
1. **Open in Excel**: The CSV opens directly with proper structure - no preprocessing needed
2. **Instant Comparison**: Each model group clearly shows metric performance across LLM models
3. **Easy Charting**: Select metric columns to create comparison charts
4. **Conditional Formatting**: Highlight best/worst performers across models
5. **PivotTable Ready**: Structure is optimized for Excel PivotTable creation

### Python Analysis with Exp1 Layout
```python
import pandas as pd

# Load exp1 layout data (no headers)
df = pd.read_csv('toys_run3_exp1_layout.csv', header=None)

# Extract experiment info from header rows
experiment_name = df.iloc[0, 1]  # toys_run3
experiment_type = df.iloc[1, 1]  # 1

# Parse model sections for analysis
# Skip header rows (0,1,2) and find model name rows
model_rows = df[df.iloc[:, 0].str.match(r'^[a-zA-Z]+$', na=False)].index[1:]

# Extract P_5 performance for comparison
p5_col_idx = 2  # P_5 is typically column 2 after model name
for model_idx in model_rows:
    model_name = df.iloc[model_idx, 0]
    print(f"{model_name} P_5 performance:")
    # Get LLM model rows following this model
    llm_rows = model_idx + 1
    while llm_rows < len(df) and df.iloc[llm_rows, 0] and df.iloc[llm_rows, 0] != '':
        llm_model = df.iloc[llm_rows, 0]
        p5_value = df.iloc[llm_rows, p5_col_idx]
        print(f"  {llm_model}: {p5_value}")
        llm_rows += 1
```

### R Analysis with Exp1 Layout
```r
library(readr)
library(dplyr)
library(ggplot2)

# Load exp1 layout data (no headers)
data <- read_csv('toys_run3_exp1_layout.csv', col_names = FALSE)

# Extract experiment metadata
experiment_name <- data[1, 2][[1]]
experiment_type <- data[2, 2][[1]]

# For detailed analysis, you may prefer to use the default layout:
# python3 collect_data.py -i /path/to/data (without -e 1)
# Then use standard R analysis on the structured format

# Or process the exp1 layout by identifying model sections
# and extracting metric columns for visualization
```

## 🎯 Choosing the Right Layout

### When to Use Exp1 Layout (`-e 1`)
✅ **Perfect for:**
- **Excel Analysis**: Direct import into Excel for visual comparison
- **LLM Model Ranking**: Comparing multiple LLM models across evaluation metrics
- **Quick Visual Assessment**: Easy to spot patterns and performance differences
- **Presentation-Ready**: Clean format for reports and presentations
- **Single Experiment Focus**: When analyzing one experiment in detail

### When to Use Default Layout (no `-e` flag)
✅ **Perfect for:**
- **Statistical Analysis**: Structured data for pandas, R, or statistical software
- **Cross-Experiment Comparison**: Combining multiple experiments for meta-analysis
- **Automated Processing**: Programmatic analysis with consistent column structure
- **Database Import**: Normalized format for database storage
- **Time Series Analysis**: Tracking performance changes over multiple experiments

### Example Usage Decision Tree
```bash
# For Excel visualization and model comparison
python3 collect_data.py -i /path/to/exp1_llm_ranking -e 1

# For statistical analysis and programming
python3 collect_data.py -i /path/to/exp1_llm_ranking

# For combining multiple experiments
python3 collect_data.py -i /path/to/exp1_llm_ranking -o exp1.csv
python3 collect_data.py -i /path/to/exp2_scaling -o exp2.csv
python3 collect_data.py -i /path/to/exp3_baselines -o exp3.csv
# Then combine exp1.csv, exp2.csv, exp3.csv for meta-analysis
```

## 🎯 Common Use Cases

### 1. **LLM Model Ranking (Exp1 Layout)**
```bash
# Excel-ready format for direct visual comparison
python3 collect_data.py -i ../experiment_output/exp1_llm_model_ranking -e 1

# Instantly see which LLM models perform best for each aspect detection method
# Open in Excel for conditional formatting and charts
```

### 2. **Statistical Analysis (Default Layout)**
```bash
# Structured format for pandas/R analysis
python3 collect_data.py -i ../experiment_output/exp1_llm_model_ranking

# Use for regression analysis, statistical tests, and automated processing
```

### 3. **Cross-Experiment Meta-Analysis**
```bash
# Combine multiple experiments for comprehensive analysis
python3 collect_data.py -i ../experiment_output/exp1_llm_model_ranking -o exp1.csv
python3 collect_data.py -i ../experiment_output/exp2_scaling_analysis -o exp2.csv
python3 collect_data.py -i ../experiment_output/exp3_baseline_comparison -o exp3.csv

# Then merge datasets for meta-analysis across all experiments
```

### 4. **Quality Assurance**
```bash
# Verify all expected files are present before analysis
python3 collect_data.py -i ../experiment_output/toys_run3 --summary-only

# Check for missing experiments or incomplete model runs
```

### 5. **Presentation and Reporting**
```bash
# Generate clean, presentation-ready output
python3 collect_data.py -i ../experiment_output/exp1_llm_model_ranking -e 1 -o llm_ranking_report.csv

# Perfect for including in research papers and presentations
```

## 🏗️ Architecture & Design

### Data Collection Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Directory       │───▶│ Metadata        │───▶│ CSV             │
│ Scanning        │    │ Extraction      │    │ Consolidation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Find all        │    │ Extract:        │    │ Combine all     │
│ agg.*.csv files │    │ - experiment    │    │ CSVs with       │
│ recursively     │    │ - model_name    │    │ metadata cols   │
│                 │    │ - llm_model     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Metadata Extraction Logic

The tool intelligently extracts metadata from directory paths:

- **Experiment Name**: Taken from the root directory name (e.g., `toys_run3`)
- **Model Name**: Extracted from parent directory (e.g., `bert`, `ctm`, `lda`)
- **LLM Model**: Extracted from immediate parent directory (e.g., `anthropic-haiku3.5-25`)

### Error Handling

- **Missing Files**: Continues processing other files, reports missing ones
- **Malformed Paths**: Logs warnings but continues collection
- **CSV Reading Errors**: Skips problematic files, reports errors in verbose mode
- **Permission Issues**: Clear error messages with suggested solutions

## 🔄 Integration with LADy Pipeline

This tool integrates seamlessly with LADy research workflows:

### Typical Research Workflow
```bash
# 1. Run LADy experiments
cd experiment_scripts
./run_toys.sh

# 2. Collect and consolidate results
cd ../data-collector
python3 collect_data.py -i ../experiment_output/toys_run3

# 3. Analyze in Excel/R/Python
# Open output CSV in your preferred analysis tool
```

### Batch Analysis Workflow
```bash
# Collect multiple experiments
for exp in toys_run2 toys_run3 exp1_llm_model_ranking; do
    python3 collect_data.py -i ../experiment_output/$exp -o ${exp}_results.csv
done

# All results are now ready for comparative analysis
```

## 📊 Performance & Scalability

- **Speed**: Processes hundreds of CSV files in seconds
- **Memory**: Low memory footprint using pandas streaming
- **Scalability**: Handles large experiment datasets (10,000+ files)
- **Robustness**: Graceful handling of corrupted or missing files

## 🧪 Experiment Structure Support

### Currently Supported
- ✅ **Standard LADy Structure**: `experiment/model/llm_model/agg.ad.pred.eval.mean.csv`
- ✅ **Multi-fold Experiments**: Handles various fold structures
- ✅ **Mixed Model Types**: bert, ctm, lda, rnd, btm, etc.
- ✅ **Different LLM Models**: Anthropic, OpenAI, Google, xAI models

### Future Extensions
- 🔄 **Custom Path Patterns**: Configurable path extraction patterns
- 🔄 **Multi-metric Files**: Support for additional evaluation files
- 🔄 **Cross-dataset Analysis**: Compare across different base datasets

## 🛠️ Customization & Extension

### Adding Support for Different Experiments

For experiments with different directory structures, you can modify the `_extract_metadata_from_path` method:

```python
def _extract_metadata_from_path(self, csv_path: Path, root: Path) -> Optional[Dict[str, str]]:
    # Custom path extraction logic here
    # Adapt to your specific experiment structure
    pass
```

### Custom Output Formats

The tool can be extended to support additional output formats:

```python
# Add to ExperimentDataCollector class
def export_to_json(self, csv_files, output_path):
    # JSON export logic
    pass

def export_to_excel(self, csv_files, output_path):
    # Excel export with multiple sheets
    pass
```

## 🤝 Contributing

When modifying the data collector:

1. **Test with Multiple Experiments**: Ensure compatibility across different experiment types
2. **Maintain Metadata Consistency**: Keep metadata column names standardized
3. **Update Documentation**: Reflect changes in this README
4. **Add Error Handling**: Consider edge cases in directory structures
5. **Preserve Excel Compatibility**: Ensure output remains Excel-friendly

## 🐛 Troubleshooting

### Common Issues

**Q: "No CSV files found" but files exist**
A: Check that you're pointing to the correct experiment output directory and that files are named `agg.ad.pred.eval.mean.csv`

**Q: Missing metadata columns in output**
A: Verify that your directory structure matches the expected pattern: `experiment/model/llm_model/file.csv`

**Q: Excel shows scientific notation for small numbers**
A: This is normal Excel behavior. Select columns and format as "Number" with desired decimal places.

**Q: Some files not included in output**
A: Run with `-v` (verbose) flag to see detailed processing logs and identify problematic files.

### Getting Help

1. Run with `--summary-only` to see what files are detected
2. Use `-v` flag to see detailed processing information
3. Check that your directory structure matches expected patterns
4. Verify CSV files are valid and readable

## 📝 Example Session

```bash
$ python3 collect_data.py -i /home/user/LADy-kap/experiment_output/toys_run3

🔍 Searching for CSV files in: /home/user/LADy-kap/experiment_output/toys_run3

=== LADy Experiment Data Collection Summary ===

📊 Collection Results:
   Total CSV files found: 4

🧪 Experiments: 1
   toys_run3

🤖 Models: 4
   bert, btm, ctm, rnd

🦾 LLM Models: 1
   anthropic-haiku3.5-25

📁 File Distribution:

   toys_run3:
     - bert: 1 files
     - btm: 1 files
     - ctm: 1 files
     - rnd: 1 files

📊 Consolidating CSV files...
✅ Success! Consolidated data saved to: output/toys_run3_consolidated_results.csv

💡 You can now open 'output/toys_run3_consolidated_results.csv' in Excel for analysis!
```

## 📅 Version History

### v1.0.0 - Initial Release
- **Core Functionality**: Recursive CSV file discovery and metadata extraction
- **Default Layout**: Standard long-format output with full metadata
- **Summary Reporting**: Detailed analysis of collected data
- **Command-line Interface**: Comprehensive options for flexible usage

### v1.1.0 - Experiment-Specific Layouts  
- **Exp1 Layout (`-e 1`)**: Pivot table format optimized for LLM model ranking
  - Model grouping with metrics as columns
  - Excel-ready format without generic headers
  - Natural metric ordering (P_1, P_5, P_10, P_100)
  - Dynamic metric detection from CSV files
- **Dual Layout Support**: Both default and exp1 layouts available
- **Enhanced Documentation**: Comprehensive usage guidelines and examples

---

*This tool is part of the LADy (Large Language Model Aspect Discovery) research project. Generated datasets and analysis results should be used in accordance with your institution's research guidelines.*
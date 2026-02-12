# 🔬 Benchmarking Datasets for Implicit Aspect Detection

**A Comprehensive Evaluation of LLM-Generated Datasets for Implicit Aspect Detection Using the LADy Framework**

---

## 📋 Research Overview

This repository contains the implementation and experimental setup for evaluating **8 different Large Language Model (LLM) generated datasets** for implicit aspect detection across **4 architecture models** (BERT, CTM, BTM, Random). The research aims to establish the first comprehensive benchmark for implicit aspect detection and determine which LLM models generate the most effective training datasets.

### Research Questions

**RQ1**: How do different LLM-generated datasets perform across architecture models (BERT vs CTM vs BTM vs Random)?  
**RQ2**: Which LLM models generate the most effective implicit aspect datasets?  
**RQ3**: How do LLM-generated implicit datasets compare to explicit SemEval baselines?

### Key Contributions

- **First systematic LLM dataset evaluation** for implicit aspect detection
- **8 LLM models benchmarked** across multiple dataset sizes (700, 1300, 2000 sentences)
- **Cross-architecture evaluation** using transformer-based, neural topic, biterm topic, and random baseline models
- **Baseline establishment** against SemEval explicit aspect datasets
- **Scaling analysis** to determine optimal dataset sizes

---

## 🗂️ Repository Structure

### Core Directories

```
├── experiment_datasets/              # All experimental datasets
│   ├── semeval_implitcits/          # LLM-generated implicit datasets
│   ├── semeval_implitcits_toys/    # Toy datasets for testing
│   └── semeval_baselines/           # SemEval explicit baselines
├── experiment_scripts/              # Automated experiment scripts
│   ├── run_baselines.sh            # Run and cache baseline experiments
│   ├── run_exp1.sh                 # LLM ranking across architectures
│   ├── run_exp2.sh                 # Top model selection (analyzes exp1)
│   ├── run_exp3.sh                 # Baseline comparison (collects results)
│   └── stop_exp.sh                 # Stop running experiments
├── experiment_output/               # All experiment results and logs
├── experiment_docs/                 # Experiment documentation
│   ├── experiments_planning.md     # 3-phase experimental design
│   ├── experiment_settings.md      # Hyperparameters and settings
│   ├── run_baselines.md           # Baseline experiment guide
│   └── llm_model_selection.md      # LLM model selection criteria
├── src/                            # LADy framework source code
│   ├── params.py                   # Model parameters (uniform settings)
│   ├── main.py                     # Main experiment runner
│   ├── aml/                        # Architecture models (BERT, CTM, BTM, etc.)
│   └── cmn/                        # Common utilities
├── data-collector/                 # Submodule for result consolidation
├── datasets-generator/             # Submodule for LLM dataset generation
└── datasets-repairer/             # Submodule for XML corruption repair
```

---

## 🎯 Experimental Datasets

### LLM-Generated Datasets (8 Models × 3 Sizes)

Located in `experiment_datasets/semeval_implitcits/`:

| LLM Model | Provider | Sizes (sentences) | Files |
|-----------|----------|-------------------|-------|
| **GPT-4o** | OpenAI | 700, 1300, 2000 | `openai-gpt4o-{size}.xml` |
| **GPT-3.5-turbo** | OpenAI | 700, 1300, 2000 | `openai-gpt3.5turbo-{size}.xml` |
| **Claude Sonnet 4** | Anthropic | 700, 1300, 2000 | `anthropic-sonnet4-{size}.xml` |
| **Claude Haiku 3.5** | Anthropic | 700, 1300, 2000 | `anthropic-haiku3.5-{size}.xml` |
| **Gemini 2.5 Pro** | Google | 700, 1300, 2000 | `google-gemini2.5pro-{size}.xml` |
| **Gemini 2.5 Flash** | Google | 700, 1300, 2000 | `google-gemini2.5flash-{size}.xml` |
| **Grok 4** | xAI | 700, 1300, 2000 | `xai-grok4-{size}.xml` |
| **Grok 3** | xAI | 700, 1300, 2000 | `xai-grok3-{size}.xml` |

**Toy Datasets** (for testing): 15 and 25 sentences in `experiment_datasets/semeval_implitcits_toys/`

### SemEval Baseline Datasets

Located in `experiment_datasets/semeval_baselines/`:

| Dataset | Sentences | Purpose |
|---------|-----------|---------|
| **SemEval-15-res-1300.xml** | 1569 | Baseline comparison for 1300-size |
| **SemEval-16-res-2000.xml** | 2350 | Baseline comparison for 2000-size |

---

## 🚀 Running Experiments

### Prerequisites

1. **Environment Setup**:
   ```bash
   cd /home/thangk/msc/LADy-kap
   conda activate lady
   ```

2. **GPU Check**:
   ```bash
   nvidia-smi  # Ensure GPU is available
   ```

### Complete Experimental Pipeline

The experiments follow a 4-step pipeline:

#### Step 0: Run Baseline Experiments (One-time, Cached)

```bash
# Run SemEval baselines on all architectures
./experiment_scripts/run_baselines.sh

# Monitor progress
tail -f experiment_datasets/semeval_baselines/output/baselines.log

# Force re-run if needed
./experiment_scripts/run_baselines.sh --force
```

**What it does:**
- Runs explicit aspect detection on SemEval datasets
- Tests on all 4 architectures (BERT, CTM, BTM, Random)
- Caches results for reuse by exp3
- Takes 2-4 hours (first run only)

#### Step 1: LLM Model Ranking (`run_exp1.sh`)

```bash
# Test all LLM models at all sizes
./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/

# Or test with toy datasets (fast, ~30 min)
./experiment_scripts/run_exp1.sh --toy -i experiment_datasets/semeval_implitcits_toys/

# Monitor progress
tail -f experiment_output/exp1_*/experiments.log
```

**What it does:**
- Tests every LLM × size combination (24 datasets)
- Runs on all 4 architectures
- Generates `experiment_summary.txt` with P@5 rankings
- Takes 24-48 hours for full datasets
- Output: `experiment_output/exp1_llm_model_ranking/{arch}/{llm-size}/`

#### Step 2: Top Model Selection (`run_exp2.sh`)

```bash
# Analyze exp1 results and select top 3 LLMs per architecture
./experiment_scripts/run_exp2.sh -i experiment_output/exp1_llm_model_ranking

# Or for toy results
./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking

# Check selections
cat experiment_output/exp2_*/top_models_summary.txt
```

**What it does:**
- Parses exp1's `experiment_summary.txt`
- Calculates average P@5 scores across all sizes
- Selects top 3 LLMs per architecture
- Copies results to exp2 output (no experiments run)
- Takes <1 minute
- Output: `experiment_output/exp2_top_models/{arch}/{llm-size}/`

#### Step 3: Baseline Comparison (`run_exp3.sh`)

```bash
# Collect baseline and top model results for comparison
./experiment_scripts/run_exp3.sh -i experiment_output/exp2_top_models

# Or for toy results
./experiment_scripts/run_exp3.sh --toy -i experiment_output/toy_exp2_top_models

# View comparison summary
cat experiment_output/exp3_*/baseline_comparison_summary.txt
```

**What it does:**
- Checks if baselines are cached (runs if needed)
- Copies all baseline results to exp3 output
- Copies all exp2 results (top models) to exp3 output
- Organizes results for data collection
- Takes <5 minutes (if baselines cached)
- Output: `experiment_output/exp3_baseline_comparison/{arch}/`

### Quick Test with Toy Datasets

Test the complete pipeline with small datasets (15-25 sentences):

```bash
# Complete pipeline test (~45 minutes total)
./experiment_scripts/run_exp1.sh --toy -i experiment_datasets/semeval_implitcits_toys/
./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking
./experiment_scripts/run_exp3.sh --toy -i experiment_output/toy_exp2_top_models
```

### Monitoring and Control

```bash
# Check running processes
ps aux | grep -E "run_exp|run_baselines"

# Monitor experiment logs
tail -f experiment_output/exp1_*/experiments.log
tail -f experiment_datasets/semeval_baselines/output/baselines.log

# Check status files
cat experiment_output/exp1_*/status.log
cat experiment_datasets/semeval_baselines/output/status.log

# Stop experiments if needed
kill <PID>  # Use PID from status.log
```

---

## 📊 Experimental Configuration

### Uniform Parameters Across All Experiments

All experiments use identical parameters from `src/params.py`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Number of Aspects** | 5 | Consistent evaluation |
| **Cross-validation** | 5-fold | Robust results |
| **Train/Test Split** | 85/15 | Standard ratio |
| **GPU** | Index 1 | Configurable |

### Model-Specific Settings

**BERT**: 3 epochs, batch_size=8, learning_rate=2e-5  
**CTM**: 20 epochs, batch_size=16, bert-base-uncased  
**BTM**: 1000 iterations, all CPU cores  
**Random**: Uniform probability baseline

### Why Uniform Parameters?

Using the same parameters across all dataset sizes (700, 1300, 2000) enables:
- **Controlled comparison** of dataset size effects
- **Fair LLM evaluation** without parameter tuning bias
- **Scaling analysis** of pure data quantity effects
- **Research validity** by isolating variables

### 🎯 Unified Ground Truth System

The LADy framework now uses a **category-based ground truth system** that enables fair evaluation across both implicit and explicit aspect detection:

**Key Improvements**:
- **Unified Evaluation**: All models (BERT, CTM, BTM, Random) use aspect categories as ground truth
- **Implicit/Explicit Support**: Same evaluation framework works for both dataset types
- **Fair Comparison**: Models predicting different formats (words vs categories) are evaluated consistently

**How It Works**:
1. **Ground Truth**: Extracted from SemEval category annotations (e.g., "FOOD#QUALITY", "SERVICE#GENERAL")
2. **Model Predictions**:
   - BERT: Predicts categories directly
   - CTM/BTM: Predict words, mapped to categories via semantic similarity
   - Random: Generates random categories
3. **Evaluation**: All predictions compared against category ground truth using pytrec_eval

**Benefits**:
- Enables meaningful comparison between implicit datasets (no explicit terms) and explicit datasets
- Ensures all architecture models are evaluated on the same semantic categories
- Provides consistent metrics across different prediction formats

For detailed technical documentation, see [.kap/ground_truth_explanation.md](.kap/ground_truth_explanation.md)

---

## 📈 Data Collection and Analysis

### Using the Data Collector

After all experiments complete:

```bash
cd data-collector
python collect_results.py --exp3-dir ../experiment_output/exp3_baseline_comparison

# Or for specific experiments
python collect_individual_exp.py --input ../experiment_output/exp1_llm_model_ranking
```

**Output Features**:
- Consolidated CSV with all metrics
- Metadata for traceability
- Ready for analysis in Excel, R, or Python

### Output Structure

```
experiment_output/
├── exp1_llm_model_ranking/
│   ├── experiment_summary.txt          # P@5 rankings for all models
│   ├── bert/
│   │   ├── openai-gpt4o-700/
│   │   │   └── agg.ad.pred.eval.mean.csv
│   │   └── ... (all LLM-size combinations)
│   ├── ctm/
│   ├── btm/
│   └── rnd/
├── exp2_top_models/
│   ├── top_models_summary.txt          # Selected models with averages
│   └── {arch}/{top-llm-size}/         # Copied from exp1
└── exp3_baseline_comparison/
    ├── baseline_comparison_summary.txt  # Final comparison
    └── {arch}/
        ├── baseline-SemEval-*/         # Baseline results
        └── {llm-size}/                 # Top model results
```

### Key Metrics

- **P@k**: Precision at k (primary ranking metric)
- **Recall@k**: Coverage of relevant aspects
- **NDCG@k**: Ranking quality measure
- **MAP@k**: Mean Average Precision

---

## 🔧 Advanced Usage

### Custom Dataset Sizes

To test specific dataset combinations:

```bash
# Single file
./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/openai-gpt4o-1300.xml

# Multiple specific files
for file in openai-gpt4o-1300.xml anthropic-sonnet4-1300.xml; do
    ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/$file
done
```

### Modifying Parameters

Edit configuration section in scripts:

```bash
# In run_exp1.sh, run_baselines.sh, etc.
export GPU_ID="2"        # Change GPU
export NUM_ASPECTS="10"  # More aspects
export NUM_FOLDS="3"     # Fewer folds for speed
```

Or modify `src/params.py` for model-specific changes.

### Resuming Failed Experiments

Scripts automatically skip completed experiments:

```bash
# Just re-run the same command
./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/

# Check what's completed
find experiment_output/exp1_* -name "agg.ad.pred.eval.mean.csv" | wc -l
```

---

## 📚 Documentation

- **[Experiment Planning](experiment_docs/experiments_planning.md)**: Detailed 3-phase design
- **[Experiment Settings](experiment_docs/experiment_settings.md)**: All hyperparameters
- **[Baseline Guide](experiment_docs/run_baselines.md)**: Running baseline experiments
- **[LLM Selection](experiment_docs/llm_model_selection.md)**: How LLMs were chosen

---

## 🤝 Contributing

For questions or issues:
1. Check the documentation in `experiment_docs/`
2. Review the experiment logs
3. Ensure environment is properly activated

---

## 📄 License

See LICENSE for details.
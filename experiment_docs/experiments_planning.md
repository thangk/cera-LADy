# Experimental Testing Plan for LLM Dataset Benchmarking

## Research Overview

This document outlines the comprehensive experimental design for evaluating **LLM-generated implicit aspect datasets** using the LADy framework. The experiments benchmark dataset effectiveness across different architecture models (BERT, CTM, BTM, Random) to determine which LLM models generate the most effective datasets for implicit aspect detection.

## Research Questions

**RQ1**: How do different LLM-generated datasets perform across architecture models (BERT vs CTM vs BTM vs Random)?  
**RQ2**: Which LLM models generate the most effective implicit aspect datasets?  
**RQ3**: How do LLM-generated implicit datasets compare to explicit SemEval baselines?

## Available Datasets

### LLM-Generated Implicit Datasets
Located in `experiment_datasets/semeval_implitcits/`:
- `anthropic-haiku3.5-{700,1300,2000}.xml`
- `anthropic-sonnet4-{700,1300,2000}.xml`
- `google-gemini2.5flash-{700,1300,2000}.xml`
- `google-gemini2.5pro-{700,1300,2000}.xml`
- `openai-gpt3.5turbo-{700,1300,2000}.xml`
- `openai-gpt4o-{700,1300,2000}.xml`
- `xai-grok3-{700,1300,2000}.xml`
- `xai-grok4-{700,1300,2000}.xml`

**Toy Datasets** (for testing): Located in `experiment_datasets/semeval_implitcits_toys/`
- Same LLM models but with sizes 15 and 25 sentences

### SemEval Baseline Datasets
Located in `experiment_datasets/semeval_baselines/`:
- `SemEval-15-res-1300.xml` (1569 sentences, explicit aspects)
- `SemEval-16-res-2000.xml` (2350 sentences, explicit aspects)

## Three-Phase Experimental Design

### Prerequisites: Running Baselines (`run_baselines.sh`)

**Purpose**: Compute and cache baseline results for all architecture models

**Process**:
1. Run SemEval explicit baselines on all architecture models
2. Cache results in `experiment_datasets/semeval_baselines/output/`
3. These cached results will be used by exp3 for comparison

**Script Usage**:
```bash
# Run baselines (will skip if already cached)
./experiment_scripts/run_baselines.sh

# Force re-run all baselines
./experiment_scripts/run_baselines.sh --force
```

### Phase 1: LLM Model Ranking (`run_exp1.sh`)

**Purpose**: Test all LLM models at all sizes to identify top performers per architecture

**Process**:
1. Run every LLM model × size combination on all 4 architecture models
2. Generate P@5 performance scores for each combination
3. Create ranking summary in `experiment_summary.txt`

**Technical Details**:
- **Auto-detects**: All XML files in input directory
- **Architectures**: BERT, CTM, BTM, Random 
- **Metrics**: P@k, Recall@k, NDCG@k, MAP@k (k=1,5,10,100)
- **Output**: `experiment_output/exp1_llm_model_ranking/{arch}/{llm-size}/`

**Script Usage**:
```bash
# Run all LLM datasets
./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/

# Run toy datasets for testing
./experiment_scripts/run_exp1.sh --toy -i experiment_datasets/semeval_implitcits_toys/
```

### Phase 2: Top Model Selection (`run_exp2.sh`)

**Purpose**: Analyze exp1 results and copy top 3 LLM models per architecture

**Process**:
1. Parse `experiment_summary.txt` from exp1
2. Calculate average P@5 scores across all sizes for each LLM
3. Select top 3 LLMs per architecture (with alphabetical tiebreaking)
4. Copy selected model folders to exp2 output

**Technical Details**:
- **Input**: Exp1 output directory containing `experiment_summary.txt`
- **Selection**: Based on average P@5 across all dataset sizes
- **No experiments run**: Only analyzes and copies results
- **Output**: `experiment_output/exp2_top_models/{arch}/{llm-size}/`

**Script Usage**:
```bash
# Analyze exp1 results and copy top models
./experiment_scripts/run_exp2.sh -i experiment_output/exp1_llm_model_ranking

# Toy mode
./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking
```

### Phase 3: Baseline Comparison (`run_exp3.sh`)

**Purpose**: Collect baseline and exp2 results for final comparison

**Process**:
1. Check if baselines exist in cache, if not, call `run_baselines.sh`
2. Wait for baselines to complete if needed
3. Copy all baseline results to exp3 output
4. Copy all exp2 results (top models) to exp3 output
5. Generate comparison summary

**Technical Details**:
- **Baseline cache**: `experiment_datasets/semeval_baselines/output/`
- **No experiments run**: Only collects and organizes results
- **Output structure**: Baselines as `baseline-{dataset}`, LLMs as `{llm-size}`
- **Output**: `experiment_output/exp3_baseline_comparison/{arch}/`

**Script Usage**:
```bash
# Collect results from exp2
./experiment_scripts/run_exp3.sh -i experiment_output/exp2_top_models

# Toy mode
./experiment_scripts/run_exp3.sh --toy -i experiment_output/toy_exp2_top_models
```

## Experiment Parameters (Uniform Across All Experiments)

All experiments use the same parameters from `src/params.py`:

### Core Settings
- **Number of aspects**: 5 (`-naspects 5`)
- **Cross-validation folds**: 5 (`-nfolds 5`)
- **Train/test split**: 85/15 (`ratio: 0.85`)
- **GPU**: Index 1 (configurable)

### Model-Specific Parameters
- **BERT**: 3 epochs, batch_size=8, learning_rate=2e-5, max_seq_length=64
- **CTM**: 20 epochs, batch_size=16, num_samples=10, bert-base-uncased
- **BTM**: 1000 iterations, alpha=1/naspects, beta=0.01
- **LDA**: 1000 passes, multicore processing
- **RND**: Random baseline (no training)

### Why Uniform Parameters?
Using the same parameters across all dataset sizes (700, 1300, 2000) allows:
- **Controlled comparison** of dataset size effects
- **Fair evaluation** of LLM quality differences
- **Analysis of model robustness** to data quantity
- **Identification of scaling patterns** without confounding factors

## Overall Project Timeline

### Phase-by-Phase Duration
- **Baselines**: 2-4 hours (8 experiments, cached for reuse)
- **Phase 1**: 24-48 hours (all LLMs × all sizes × 4 architectures)
- **Phase 2**: <1 minute (analysis and copying only)
- **Phase 3**: <5 minutes (copying results) + baseline time if not cached

### Resource Requirements
- **GPU**: NVIDIA GPU (index 1) for BERT and CTM
- **CPU**: All cores for BTM and LDA
- **Memory**: 8-16GB GPU, 16-32GB system RAM
- **Storage**: ~50-100GB for all outputs

## Expected Outcomes and Analysis

### Performance Metrics
Each experiment produces metrics saved in `agg.ad.pred.eval.mean.csv`:
- **P@k**: Precision at k (primary metric for ranking)
- **Recall@k**: Recall at k
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision

### Analysis Dimensions

1. **LLM Creativity Analysis**
   - Which LLMs generate most learnable implicit aspects?
   - Do creative LLMs (GPT-4, Claude) outperform conservative ones?
   - Is there consistency across architecture models?

2. **Dataset Size Scaling**
   - How does performance scale from 700→1300→2000?
   - Which models benefit most from larger datasets?
   - At what size do we see diminishing returns?

3. **Architecture Robustness**
   - Which models handle implicit aspects best?
   - How large is the explicit→implicit performance gap?
   - Do neural models (BERT, CTM) outperform topic models?

4. **Baseline Comparison**
   - How do LLM-generated datasets compare to human annotations?
   - Is the implicit detection challenge realistic?
   - Which architecture minimizes the implicit detection penalty?

## Running the Complete Pipeline

### Step-by-Step Execution

1. **Setup Environment**:
   ```bash
   cd /home/thangk/msc/LADy-kap
   conda activate lady
   ```

2. **Run Baselines** (one-time, cached):
   ```bash
   ./experiment_scripts/run_baselines.sh
   # Monitor: tail -f experiment_datasets/semeval_baselines/output/baselines.log
   ```

3. **Run Phase 1** (LLM ranking):
   ```bash
   ./experiment_scripts/run_exp1.sh -i experiment_datasets/semeval_implitcits/
   # Monitor: tail -f experiment_output/exp1_*/experiments.log
   ```

4. **Run Phase 2** (top model selection):
   ```bash
   ./experiment_scripts/run_exp2.sh -i experiment_output/exp1_llm_model_ranking
   # Check: cat experiment_output/exp2_*/top_models_summary.txt
   ```

5. **Run Phase 3** (final comparison):
   ```bash
   ./experiment_scripts/run_exp3.sh -i experiment_output/exp2_top_models
   # Results: experiment_output/exp3_*/baseline_comparison_summary.txt
   ```

### Quick Testing with Toy Datasets

For testing the pipeline with smaller datasets:
```bash
# Test complete pipeline
./experiment_scripts/run_exp1.sh --toy -i experiment_datasets/semeval_implitcits_toys/
./experiment_scripts/run_exp2.sh --toy -i experiment_output/toy_exp1_llm_model_ranking
./experiment_scripts/run_exp3.sh --toy -i experiment_output/toy_exp2_top_models
```

## Data Collection and Analysis

After all experiments complete:

1. **Collect all CSV results**:
   ```bash
   # Use data-collector scripts or manually aggregate
   find experiment_output -name "agg.ad.pred.eval.mean.csv" -exec cp {} results/ \;
   ```

2. **Generate performance matrices**:
   - Rows: LLM models
   - Columns: Architecture models
   - Cells: P@5 scores
   - Separate matrices for each dataset size

3. **Create visualizations**:
   - Scaling curves (performance vs dataset size)
   - Architecture comparison bars
   - LLM ranking heatmaps

## Research Contributions

This experimental framework provides:

1. **First comprehensive LLM benchmark** for implicit aspect dataset generation
2. **Cross-architecture evaluation** revealing model-specific strengths
3. **Scaling analysis** for practical dataset size recommendations
4. **Baseline establishment** for future implicit aspect detection research
5. **Evidence-based LLM selection guidelines** for dataset generation tasks

---

**Note**: This plan reflects the updated 3-phase structure where exp2 only analyzes and copies results, with all baseline experiments handled by a dedicated script for efficiency and reusability.
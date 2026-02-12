# Running SemEval Baseline Experiments

This document describes how to run and manage the SemEval baseline experiments for comparison with LLM-generated implicit datasets.

## Overview

The `run_baselines.sh` script executes SemEval explicit aspect baseline experiments across all architecture models. Results are cached for reuse by exp3 and other experiments.

## Script Location

```bash
./experiment_scripts/run_baselines.sh
```

## Purpose

1. **Establish baselines**: Run explicit aspect detection on human-annotated SemEval datasets
2. **Cache results**: Store results for efficient reuse across experiments
3. **Enable comparison**: Provide baseline metrics for exp3 comparison with implicit datasets

## Usage

### Basic Usage

```bash
# Run baselines (will skip if already cached)
./experiment_scripts/run_baselines.sh

# Force re-run all baselines (ignore cache)
./experiment_scripts/run_baselines.sh --force

# Get help
./experiment_scripts/run_baselines.sh --help
```

### Prerequisites

1. Navigate to project root:
   ```bash
   cd /home/thangk/msc/LADy-kap
   ```

2. Activate conda environment:
   ```bash
   conda activate lady
   ```

3. Ensure baseline datasets exist:
   ```bash
   ls experiment_datasets/semeval_baselines/*.xml
   ```

## What It Does

### 1. Dataset Detection
- Finds all XML files in `experiment_datasets/semeval_baselines/`
- Currently processes:
  - `SemEval-15-res-1300.xml` (1569 sentences)
  - `SemEval-16-res-2000.xml` (2350 sentences)

### 2. Cache Checking
- Checks for existing results in `experiment_datasets/semeval_baselines/output/`
- Structure: `output/{dataset}/{architecture}/agg.ad.pred.eval.mean.csv`
- Skips experiments if results already exist (unless `--force`)

### 3. Experiment Execution
For each baseline dataset × architecture model:
- Runs with consistent parameters:
  - `-naspects 5`
  - `-nfolds 5`
  - `-gpu 1`
- Uses same params.py as all other experiments

### 4. Result Caching
- Saves results to cache directory
- Creates structure:
  ```
  experiment_datasets/semeval_baselines/output/
  ├── SemEval-15-res-1300/
  │   ├── bert/
  │   ├── ctm/
  │   ├── btm/
  │   └── rnd/
  └── SemEval-16-res-2000/
      ├── bert/
      ├── ctm/
      ├── btm/
      └── rnd/
  ```

## Monitoring Progress

### Real-time Monitoring
```bash
# Check if running
ps aux | grep run_baselines.sh

# Monitor progress log
tail -f experiment_datasets/semeval_baselines/output/baselines.log

# Check status file
cat experiment_datasets/semeval_baselines/output/status.log
```

### Status File
The script creates a live status file showing:
- Current status (RUNNING/COMPLETED)
- Process ID
- Runtime counter
- Progress (X/Y baselines completed)
- Success/failure counts

### Log Files
- Main log: `baselines.log`
- Individual experiments: `logs/{arch}_{dataset}.log`
- Summary: `baselines_summary.txt`

## Output Structure

```
experiment_datasets/semeval_baselines/output/
├── status.log                    # Live status
├── baselines.log                # Main execution log
├── baselines_summary.txt        # Results summary with P@5 scores
├── logs/                        # Individual experiment logs
│   ├── bert_SemEval-15-res-1300.log
│   ├── bert_SemEval-16-res-2000.log
│   └── ...
├── SemEval-15-res-1300/        # Cached results
│   ├── bert/
│   │   ├── agg.ad.pred.eval.mean.csv
│   │   ├── reviews.pkl
│   │   └── ...
│   └── ...
└── SemEval-16-res-2000/        # Cached results
    └── ...
```

## Configuration

### Hardcoded Settings
```bash
# GPU Configuration
export GPU_ID="1"

# Experiment Settings
export NUM_ASPECTS="5"
export NUM_FOLDS="5"

# Architecture Models
ARCH_MODELS=("bert" "ctm" "btm" "rnd")
```

### Model Parameters
Uses the same `src/params.py` as all experiments:
- BERT: 3 epochs, batch_size=8
- CTM: 20 epochs, batch_size=16
- BTM: 1000 iterations
- LDA: 1000 passes

## Expected Runtime

- **Per experiment**: 10-90 minutes (depends on model)
- **Total (first run)**: 2-4 hours for 8 experiments
- **Subsequent runs**: <1 minute (uses cache)

### Approximate Times per Model
- BERT: 45-90 minutes
- CTM: 25-45 minutes
- BTM: 10-20 minutes
- RND: <1 minute

## Integration with Exp3

The exp3 script automatically:
1. Checks if baselines exist in cache
2. Calls `run_baselines.sh` if needed
3. Waits for completion
4. Copies cached results to exp3 output

This ensures baselines are computed only once and reused efficiently.

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Solution: Reduce batch sizes in params.py
   - Or: Set different GPU_ID in script

2. **Missing Datasets**
   - Check: `ls experiment_datasets/semeval_baselines/*.xml`
   - Ensure XML files are properly formatted

3. **Permission Denied**
   - Make executable: `chmod +x experiment_scripts/run_baselines.sh`

4. **Process Already Running**
   - Check: `ps aux | grep run_baselines.sh`
   - Kill if needed: `kill <PID>`

### Clearing Cache

To force complete re-run:
```bash
# Remove all cached results
rm -rf experiment_datasets/semeval_baselines/output/

# Then run with --force
./experiment_scripts/run_baselines.sh --force
```

## Results Interpretation

The `baselines_summary.txt` shows P@5 scores:
```
Dataset: SemEval-15-res-1300
----------------------------------------
  bert: P@5 = 0.65
  ctm: P@5 = 0.58
  btm: P@5 = 0.52
  rnd: P@5 = 0.20
```

These scores represent performance on **explicit** aspect detection and serve as upper bounds for comparison with implicit datasets.

## Best Practices

1. **Run baselines first**: Before starting exp1/exp2/exp3
2. **Check cache**: Verify results exist before experiments
3. **Don't modify cache**: Let scripts manage the cache
4. **Monitor first run**: Ensure successful completion
5. **Keep logs**: Useful for debugging issues

---

**Note**: The baseline experiments are crucial for establishing performance bounds and validating that LLM-generated implicit datasets create meaningful challenges compared to explicit aspect detection.
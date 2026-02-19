# cera-LADy

<p align="center">
  <b>Latent Aspect Detection evaluation framework for benchmarking synthetic review datasets.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20development-yellow" alt="Status">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.1-green" alt="CUDA">
  <img src="https://img.shields.io/badge/license-research%20only-lightgrey" alt="License">
</p>

---

> **Note:** This repository is under active development as part of an MSc thesis at the University of Windsor. A fork of the [LADy](https://github.com/fani-lab/LADy) framework, adapted for domain-agnostic category-based evaluation of [CERA](https://github.com/thangk/cera)-generated datasets.

---

## Overview

cera-LADy evaluates synthetic review datasets for **implicit aspect detection** using four architecture models. It measures how well generated datasets (from CERA, heuristic baselines, or real data) enable models to detect latent aspects in reviews.

**Architecture Models:**

| Model | Type | Description |
|-------|------|-------------|
| **BERT** | Transformer | Fine-tuned BERT-E2E-ABSA for sequence labeling |
| **CTM** | Neural Topic Model | Contextualized Topic Model with BERT embeddings |
| **BTM** | Biterm Topic Model | Word co-occurrence patterns for short texts |
| **RND** | Random Baseline | Uniform probability baseline for comparison |

**Primary Metric:** P@5 (Precision at 5) across 5-fold cross-validation with 85/15 train/test split.

---

## Quick Start (Docker)

### Prerequisites
- Docker with NVIDIA Container Toolkit (GPU required for BERT/CTM)

### 1. Build and start the container

```bash
docker-compose up -d --build
```

### 2. Enter the container

```bash
docker exec -it cera-lady-cli bash
```

### 3. Run a benchmark job

```bash
cd /app/scripts
./run_job.sh -d ../datasets/real-baselines-full/laptop.xml \
             -o ../output/laptop-real \
             -c ../datasets/categories/laptops.csv \
             -a 86
```

### 4. Check results

```bash
# P@5 scores per model
cat output/laptop-real/agg.ad.pred.eval.mean.csv
```

---

## Running Individual Experiments

```bash
cd /app/src

# Single model on a single dataset
python -u main.py \
  -am bert \
  -data ../datasets/cera/reviews-run1-explicit.xml \
  -output ../output/cera \
  -naspects 5 \
  -categories ../datasets/categories/laptops.csv

# Available models: rnd, btm, ctm, bert
# Available flags:
#   -am          Architecture model (rnd|btm|ctm|bert)
#   -data        Path to SemEval XML dataset
#   -output      Output directory
#   -naspects    Number of aspects (default: 25)
#   -categories  Path to categories CSV (required for evaluation)
#   -gpu         GPU index (default: auto-detect)
#   -nfolds      Number of CV folds (default: 5)
#   -skip-agg    Skip result aggregation step
```

---

## Category-Based Evaluation

cera-LADy uses a **domain-agnostic category mapping** system that enables fair evaluation across both implicit and explicit aspect datasets:

1. **Ground truth** is extracted from SemEval category annotations (e.g., `FOOD#QUALITY`, `DISPLAY#DESIGN_FEATURES`)
2. **Model predictions** (words or topics) are mapped to categories via semantic similarity using sentence-transformers
3. **Evaluation** compares mapped predictions against category ground truth using pytrec_eval

Category files are provided for three domains:

| Domain | Categories | File |
|--------|-----------|------|
| Laptop | 86 | `datasets/categories/laptops.csv` |
| Restaurant | 13 | `datasets/categories/restaurants.csv` |
| Hotel | 36 | `datasets/categories/hotels.csv` |

---

## Repository Structure

```
cera-LADy/
├── datasets/
│   ├── categories/          # Domain category files (laptops, restaurants, hotels)
│   ├── real-baselines/      # Trimmed real SemEval datasets
│   └── real-baselines-full/ # Full SemEval datasets (laptop, restaurant, hotel)
├── scripts/
│   ├── run_job.sh           # Run all 4 models on a single dataset
│   └── predownload_models.py # Pre-download BERT/CTM model weights
├── src/
│   ├── main.py              # Main experiment runner
│   ├── params.py            # Model hyperparameters (dynamic batch sizing)
│   ├── aml/                 # Architecture models (BERT, CTM, BTM, RND)
│   ├── cmn/                 # Common utilities (review loading, category mapping)
│   └── octis/               # OCTIS topic modeling framework
├── Dockerfile               # PyTorch 2.1.0 + CUDA 12.1
├── docker-compose.yml       # GPU-enabled container config
└── pyproject.toml           # Python dependencies
```

Synthetic datasets (CERA-generated and heuristic) are provided via CERA job output directories rather than checked into this repository.

---

## Model Hyperparameters

All experiments use uniform parameters from `src/params.py`:

| Parameter | Value |
|-----------|-------|
| Aspects | 5 |
| Cross-validation | 5-fold |
| Train/Test Split | 85/15 |

| Model | Key Settings |
|-------|-------------|
| **BERT** | 3 epochs, lr=2e-5, bert-base-uncased, batch size auto-scaled by GPU VRAM |
| **CTM** | 20 epochs, bert-base-uncased embeddings, batch size auto-scaled by GPU VRAM |
| **BTM** | 1000 iterations, all CPU cores |
| **RND** | Uniform probability baseline |

> **Note:** BERT and CTM batch sizes are dynamically determined based on available GPU VRAM (see `src/params.py`).

---

## Output Structure

```
output/{dataset_name}/
├── reviews.pkl                              # Preprocessed reviews
├── splits.json                              # Train/test split indices
├── {naspects}/{model}/                      # Per-model results (e.g., 86/bert/)
│   ├── f{0..4}.model.*                      # Trained model per fold
│   ├── f{0..4}.model.ad.pred.*              # Predictions per fold
│   └── f{0..4}.model.ad.pred.*.eval.*       # Per-fold evaluation
├── agg.ad.pred.eval.mean.csv                # Aggregated metrics (P@5, NDCG, MAP, etc.)
└── {model}_log.txt                          # Training log
```

The `{naspects}` directory corresponds to the `-naspects` flag (e.g., `86` for laptop with 86 categories).

---

## Related Projects

- [CERA](https://github.com/thangk/cera) — Context-Engineered Reviews Architecture (synthetic dataset generator)
- [LADy](https://github.com/fani-lab/LADy) — Original LADy framework by fani-lab

---

## Citation

```bibtex
@mastersthesis{thang2026cera,
  title     = {CERA: Context-Engineered Reviews Architecture for
               Synthetic ABSA Dataset Generation},
  author    = {Thang, Kap},
  school    = {University of Windsor},
  year      = {2026},
  type      = {Master's Thesis}
}
```

---

## License

This project is for **research purposes only**.

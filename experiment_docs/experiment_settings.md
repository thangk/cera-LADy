# Experimental Settings and Hyperparameters

This document provides the experimental settings and hyperparameters used in the LLM dataset benchmarking study for implicit aspect detection. All settings use **uniform parameters** across dataset sizes to ensure controlled comparison and analysis of scaling effects.

## Dataset Configuration

### LLM-Generated Implicit Datasets
| Parameter | Value |
|-----------|-------|
| **Dataset Sizes** | 15, 25 (toy), 700, 1300, 2000 sentences |
| **LLM Models** | 8 models: GPT-4o, GPT-3.5-turbo, Claude Haiku/Sonnet, Gemini Flash/Pro, Grok 3/4 |
| **Domain** | Restaurant reviews (implicit aspects) |
| **Format** | SemEval XML structure |

### SemEval Baseline Datasets (Explicit)
| Dataset | Sentences | Reviews | Purpose |
|---------|-----------|---------|---------|
| **SemEval-15-res-1300** | 1569 | ~285 | Baseline for 1300-size comparison |
| **SemEval-16-res-2000** | 2350 | ~428 | Baseline for 2000-size comparison |

### Experimental Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Train/Test Split** | 85% / 15% | Standard split ratio |
| **Cross-Validation** | 5-fold | Robust evaluation |
| **Number of Aspects** | 5 | Uniform across all experiments |
| **Domain** | Restaurant reviews | Consistent domain |

## Model Hyperparameters (from src/params.py)

All models use **identical parameters** across all dataset sizes (700, 1300, 2000) to isolate dataset size effects.

### BERT (bert-e2e-absa)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Base Model** | bert-base-uncased | Standard pre-trained model |
| **Learning Rate** | 2e-5 | Standard BERT fine-tuning rate |
| **Epochs** | 3.0 | Balanced for all sizes |
| **Train Batch Size** | 8 | Dynamic reduction on OOM |
| **Eval Batch Size** | 4 | Memory-efficient |
| **Max Sequence Length** | 64 | Reduced from 128 for memory |
| **Max Steps** | 500 | Prevents overtraining on small data |
| **Gradient Accumulation** | 1 | No accumulation |
| **Warmup Steps** | 0 | No warmup |
| **Logging Steps** | 10 | Frequent monitoring |
| **Save Steps** | 50 | Regular checkpointing |
| **Tagging Schema** | BIEOS | Standard aspect tagging |
| **Random Seed** | 42 | Reproducibility |

### CTM (Contextualized Topic Model)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Base Model** | bert-base-uncased | Contextual embeddings |
| **Contextual Size** | 768 | BERT embedding dimension |
| **Epochs** | 20 | May be low for larger datasets |
| **Batch Size** | 16 | Auto-capped to dataset size |
| **Num Samples** | 10 | Sampling during training |
| **Inference Type** | combined | Best performance mode |
| **Verbose** | True | Detailed logging |
| **Random Seed** | 0 | Reproducibility |

### BTM (Biterm Topic Model)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Iterations** | 1000 | Good for convergence |
| **Alpha** | 1.0/naspects | Symmetric prior |
| **Beta** | 0.01 | Standard BTM beta |
| **CPU Cores** | All available | Multicore processing |
| **Random Seed** | 0 | Reproducibility |

### LDA (Latent Dirichlet Allocation)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Passes** | 1000 | High for convergence |
| **Workers** | All CPU cores | Parallel processing |
| **Per Word Topics** | True | Detailed topic info |
| **Random State** | 0 | Reproducibility |

### Random (Baseline)
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Mode** | Uniform random | Equal probability |
| **Seed** | 0 | Reproducible randomness |

## Evaluation Configuration

### Metrics
| Metric Type | Cut-off Values (k) | Primary Metric |
|-------------|-------------------|----------------|
| **Precision@k** | 1, 5, 10, 100 | P@5 for ranking |
| **Recall@k** | 1, 5, 10, 100 | Secondary |
| **NDCG@k** | 1, 5, 10, 100 | Ranking quality |
| **MAP@k** | 1, 5, 10, 100 | Overall performance |
| **Success@k** | 1, 5, 10, 100 | Binary success |

### Evaluation Framework
| Component | Implementation |
|-----------|----------------|
| **Evaluation Library** | pytrec_eval |
| **Ground Truth** | Aspect categories |
| **Prediction Mapping** | Category mapper |
| **Synonym Expansion** | Disabled (syn=False) |
| **Hidden Aspect Ratio** | 0.0 |

## Computational Environment

### Resource Allocation
| Component | Configuration |
|-----------|--------------|
| **GPU** | NVIDIA GPU index 1 (CUDA_VISIBLE_DEVICES=1) |
| **GPU Models** | BERT, CTM |
| **CPU Models** | BTM, LDA, Random |
| **Memory** | 8-16GB GPU, 16-32GB RAM |
| **Parallelism** | All CPU cores for topic models |

### Software Environment
| Component | Version/Configuration |
|-----------|---------------------|
| **Python** | 3.8+ |
| **Framework** | LADy (Latent Aspect Detection) |
| **BERT Library** | Hugging Face Transformers |
| **Topic Modeling** | Gensim (LDA), custom implementations |
| **Conda Environment** | lady |

## Pipeline Configuration

### Processing Steps (from settings['cmd'])
1. **prep**: Load and preprocess data
2. **train**: Train models with cross-validation
3. **test**: Run inference on test sets
4. **eval**: Calculate evaluation metrics
5. **agg**: Aggregate results across folds

### Data Processing
| Parameter | Value | Notes |
|-----------|-------|-------|
| **Document Type** | 'snt' | Sentence-level processing |
| **Language Augmentation** | Disabled (['']) | No backtranslation |
| **Max Length** | 1500 | Maximum text length |
| **Batch Processing** | True | Efficient loading |

## Experimental Design Rationale

### Why Uniform Parameters?
1. **Controlled Comparison**: Isolates dataset size as the only variable
2. **Fair LLM Evaluation**: No parameter tuning bias favoring specific LLMs
3. **Scaling Analysis**: Pure observation of how models handle data quantity
4. **Research Validity**: Prevents confounding factors from parameter changes

### Expected Outcomes by Dataset Size

**700 sentences (small)**:
- Risk of overfitting with 1000 iterations (LDA/BTM)
- BERT max_steps=500 may dominate over epochs
- Good for rapid prototyping

**1300 sentences (medium)**:
- Balanced for current parameters
- Expected optimal performance
- Good representation of practical use cases

**2000 sentences (large)**:
- Potential undertraining for CTM (only 20 epochs)
- Memory pressure for BERT
- Best for evaluating scalability

## Reproducibility Checklist

- [ ] Use conda environment: `lady`
- [ ] Set GPU: `export CUDA_VISIBLE_DEVICES=1`
- [ ] Use consistent seeds (0 for most models, 42 for BERT)
- [ ] Run from project root: `/home/thangk/msc/LADy-kap`
- [ ] Use provided experiment scripts for consistency
- [ ] Verify parameters match `src/params.py`

## Notes on Baseline Comparison

The SemEval baselines use:
- **Same parameters** as implicit datasets
- **Same evaluation framework** (category-based)
- **Same number of aspects** (5)
- **Same cross-validation** (5-fold)

This ensures the explicit→implicit performance gap measurement is valid and unbiased.

---

**Last Updated**: Reflects current experimental setup with uniform parameters across all dataset sizes for controlled comparison and meaningful scaling analysis.
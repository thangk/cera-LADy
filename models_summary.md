# LADy Framework Model Architectures Summary

This document provides an overview of the four model architectures used in the LADy framework for Aspect-Based Sentiment Analysis (ABSA).

## 📊 Model Comparison Table

| Model | Type | Architecture | Training Required | Key Characteristics |
|-------|------|-------------|-------------------|---------------------|
| **BERT** | Neural/Transformer | Deep Learning | Yes | End-to-end ABSA with contextual embeddings |
| **CTM** | Neural Topic Model | VAE + BERT | Yes | Combines neural networks with topic modeling |
| **BTM** | Probabilistic | Biterm Topic Model | Yes | Designed for short text analysis |
| **RND** | Baseline | Random Assignment | No | Control baseline for comparison |

---

## 🤖 BERT (Bidirectional Encoder Representations from Transformers)

### Overview
- **Type**: Neural Network / Transformer-based Model
- **Architecture**: Deep bidirectional transformer with attention mechanisms
- **Implementation**: Uses `bert_e2e_absa` library for end-to-end aspect-based sentiment analysis

### Key Features
- **Neural Architecture**: Yes - Deep neural network with multiple transformer layers
- **Pre-trained**: Uses pre-trained BERT models (e.g., BERT-base, BERT-large)
- **Context-Aware**: Captures bidirectional context through self-attention mechanisms
- **End-to-End**: Performs both aspect extraction and sentiment classification jointly

### Technical Details
- Processes text at the token level with positional embeddings
- Uses attention mechanisms to understand relationships between words
- Maximum sequence length: 511 tokens (implementation constraint)
- Outputs aspect terms with sentiment labels (POS/NEG/NEU)
- Dynamic batch sizing for GPU optimization

### Strengths
- Excellent at understanding context and word relationships
- State-of-the-art performance on many NLP tasks
- Can capture implicit semantic relationships

---

## 🎯 CTM (Contextualized Topic Model)

### Overview
- **Type**: Neural Topic Model
- **Architecture**: Variational Autoencoder (VAE) combined with BERT embeddings
- **Implementation**: Uses `contextualized_topic_models` library

### Key Features
- **Neural Architecture**: Yes - Combines neural networks (VAE) with contextualized embeddings
- **Hybrid Approach**: Merges traditional topic modeling with modern neural embeddings
- **Context-Aware**: Uses BERT embeddings for contextual understanding
- **Unsupervised**: Learns topics without labeled data

### Technical Details
- **Components**:
  - BERT encoder for contextual embeddings (default size: 768 dimensions)
  - VAE for topic distribution learning
  - Bag-of-Words (BoW) representation combined with contextual features
- **Parameters**:
  - Number of topics (n_components)
  - Contextual embedding size
  - Training epochs and batch size
- **Output**: Topic distributions over documents and word distributions over topics

### Strengths
- Combines the interpretability of topic models with neural network power
- Better handles polysemy (words with multiple meanings) than traditional topic models
- Produces coherent topics with contextual understanding

---

## 📝 BTM (Biterm Topic Model)

### Overview
- **Type**: Probabilistic Topic Model
- **Architecture**: Statistical model based on word co-occurrence patterns
- **Implementation**: Uses `bitermplus` library

### Key Features
- **Neural Architecture**: No - Pure probabilistic/statistical model
- **Short Text Optimized**: Specifically designed for short texts (tweets, reviews)
- **Biterm-based**: Models topics based on word pairs (biterms) rather than documents
- **Unsupervised**: No labeled data required

### Technical Details
- **Core Concept**: Models topics through word co-occurrence patterns in short texts
- **Algorithm**: 
  - Extracts biterms (word pairs) from documents
  - Models topics as distributions over biterms
  - Uses Gibbs sampling for inference
- **Parameters**:
  - Alpha (α): Document-topic distribution prior (default: 1.0/n_topics)
  - Beta (β): Topic-word distribution prior (default: 0.01)
  - Number of iterations for Gibbs sampling
- **Output**: Topic distributions and top words per topic

### Strengths
- Excellent for short texts where traditional topic models fail
- Addresses data sparsity issues in short documents
- More stable than LDA on short texts
- Computationally efficient

---

## 🎲 RND (Random Baseline)

### Overview
- **Type**: Baseline Model
- **Architecture**: Random assignment algorithm
- **Implementation**: Custom implementation for experimental control

### Key Features
- **Neural Architecture**: No - Pure algorithmic baseline
- **No Training**: Does not learn from data
- **Deterministic Random**: Uses seeded random generation for reproducibility
- **Control Baseline**: Provides lower bound for performance comparison

### Technical Details
- Assigns random aspect predictions to test documents
- Generates dummy aspect words (e.g., "rnd_aspect_0", "rnd_aspect_1")
- Uses consistent random seeds for reproducibility
- No model parameters to learn or tune

### Purpose
- Establishes performance floor for evaluation
- Helps quantify the actual learning achieved by other models
- Ensures that trained models perform better than chance
- Useful for statistical significance testing

---

## 🔄 Model Selection Guidelines

### When to Use Each Model:

1. **BERT**: 
   - When you need high accuracy for explicit aspect extraction
   - Have sufficient labeled training data
   - Can afford computational resources (GPU required)
   - Need joint aspect-sentiment analysis

2. **CTM**:
   - When you want interpretable topics with neural performance
   - Working with unlabeled or partially labeled data
   - Need to understand document-level themes
   - Want balance between interpretability and performance

3. **BTM**:
   - Analyzing short texts (< 20 words average)
   - Working with social media data, reviews, or comments
   - Need fast, interpretable results
   - Limited computational resources

4. **RND**:
   - Establishing baseline performance metrics
   - Validating that models are actually learning
   - Testing evaluation pipeline
   - Statistical significance testing

---

## 💡 Key Insights

- **Neural vs. Probabilistic**: BERT and CTM use neural architectures for better semantic understanding, while BTM uses pure probabilistic modeling for interpretability
- **Supervision**: BERT requires labeled data, while CTM and BTM can work unsupervised
- **Text Length**: BTM is optimized for short texts, BERT handles medium-length texts well, CTM works across various lengths
- **Computational Cost**: BERT > CTM > BTM > RND (in terms of training time and resources)
- **Interpretability**: BTM > CTM > BERT (topic models are more interpretable than transformers)
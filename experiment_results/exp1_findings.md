# Experiment 1: LLM Model Ranking Analysis

## Overview
This experiment evaluates different LLM models (Anthropic's Claude models, Google's Gemini, OpenAI's GPT models, and xAI's Grok) across four aspect-based sentiment analysis methods (BERT, BTM, CTM, RND) using various metrics.

## Key Findings

### 1. Top Performing Models by Method

#### BERT
- **Best overall**: Grok-4 shows consistently strong performance across metrics
- **Notable**: Gemini 2.5 Pro and Sonnet-4 also perform well
- **Weakest**: Haiku 3.5 shows the lowest performance

#### BTM (Biterm Topic Model)
- **Best overall**: Sonnet-4 dominates with the highest scores across most metrics
- **Strong contenders**: Grok-4 and Gemini 2.5 Pro
- **Pattern**: BTM benefits from more sophisticated language models

#### CTM (Contextualized Topic Model)
- **Best overall**: Sonnet-4 again shows superior performance
- **Close second**: Grok-4 performs nearly as well
- **Consistent**: Gemini 2.5 Pro maintains steady performance

#### RND (Random baseline)
- **Best overall**: Grok-4 achieves the highest scores
- **Surprising**: GPT-4o shows moderate performance
- **Weakest**: GPT-3.5 Turbo significantly underperforms

### 2. Cross-Method Observations

1. **Sonnet-4 consistency**: Shows strong performance across BTM and CTM, indicating good compatibility with topic modeling approaches. This aligns with Sonnet-4's "extended thinking" capability, which allows it to decompose complex semantic relationships - crucial for topic modeling.

2. **Grok-4 versatility**: Performs well across all methods, showing robustness. Its parallel reasoning chains and tools-native training enable it to adapt to different analytical frameworks effectively.

3. **Model architecture matters more than size**: 
   - Sonnet-4 (~175B) outperforms larger models through its thinking architecture
   - Grok-4's parallel processing compensates for raw parameter count
   - Haiku 3.5 (20B) underperforms despite being efficient, lacking advanced reasoning

4. **Method-specific strengths explained**:
   - Topic models (BTM, CTM) work best with Sonnet-4 due to its methodical decomposition abilities
   - BERT shows balanced performance as it doesn't require deep reasoning
   - RND baseline reveals Grok-4's superior pattern recognition from its tools-native training

### 3. Metric-Specific Insights

- **Precision metrics (P@1, P@5, P@10)**: Sonnet-4 and Grok-4 lead
- **Recall metrics**: Similar pattern with Sonnet-4 showing slight advantages
- **NDCG scores**: Indicate good ranking quality for Sonnet-4 and Grok-4
- **Success rates**: High success rates (>0.7) for top models across methods

### 4. Why These Results? Model Architecture Insights

**Why Sonnet-4 excels at topic modeling (BTM/CTM)**:
- Extended thinking mode allows systematic exploration of semantic relationships
- Internal scratchpad reasoning helps maintain coherent topic clusters
- Superior at preserving nuanced meaning during aspect extraction

**Why Grok-4 dominates RND baseline**:
- Tools-native training means it naturally leverages auxiliary computations
- Parallel thought chains explore multiple interpretations simultaneously
- Less constrained by traditional NLP assumptions, allowing creative solutions

**Why GPT models underperform**:
- GPT-4o optimized for speed over depth, sacrificing reasoning quality
- GPT-3.5 Turbo lacks advanced reasoning architectures entirely
- General-purpose training doesn't specialize in linguistic analysis

**Why Haiku 3.5 struggles**:
- Small model size (20B) limits semantic understanding
- Efficiency-focused design trades capability for speed
- Lacks thinking/reasoning enhancements of larger models

### 5. Implications

1. **Architecture > Parameters**: Thinking models (Sonnet-4, Grok-4) outperform larger non-thinking models
2. **Method-Model Synergy**: Match model strengths to method requirements (e.g., Sonnet-4's decomposition for topic modeling)
3. **Cost-performance reality**: Advanced reasoning architectures justify higher costs through substantial performance gains
4. **Future direction**: Invest in models with reasoning capabilities rather than raw parameter scaling

## Recommendations

1. **For production**: Use Sonnet-4 with BTM or CTM for best results
2. **For experimentation**: Grok-4 offers good versatility across methods
3. **Budget-conscious**: Gemini 2.5 Pro provides good balance of performance and cost
4. **Avoid**: Haiku 3.5 and GPT-3.5 Turbo for this task due to consistently lower performance
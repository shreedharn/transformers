# Transformers for GenAI: From First Principles to Practice

## TL;DR

Transformers revolutionized ML by solving sequence modeling with parallelizable attention mechanisms instead of sequential RNNs. The core insight: **attention allows every position to directly communicate with every other position**, eliminating bottlenecks. Key components are scaled dot-product attention (Q/K/V), multi-head self-attention, position-wise FFNs, and residual connections. Three architectures dominate: encoder-only (BERT) for understanding, decoder-only (GPT) for generation, and encoder-decoder (T5) for translation. Modern optimizations include KV caching for inference, efficient attention variants for long sequences, quantization for deployment, and parameter-efficient fine-tuning (LoRA). The attention mechanism's O(n²) complexity drives most current research into efficient alternatives.

## Table of Contents

1. [Why Transformers? A Short History](#why-transformers-a-short-history)
2. [Transformer Fundamentals](#transformer-fundamentals)
3. [Architectural Variants and When to Use Them](#architectural-variants-and-when-to-use-them)
4. [Training Objectives and Data Curriculum](#training-objectives-and-data-curriculum)
5. [How Attention Actually Runs at Scale](#how-attention-actually-runs-at-scale)
6. [The MLP (FFN) Block in Practice](#the-mlp-ffn-block-in-practice)
7. [Quantization for Practical Deployment](#quantization-for-practical-deployment)
8. [Fine-Tuning and Parameter-Efficient Methods](#fine-tuning-and-parameter-efficient-methods)
9. [Evaluation and Diagnostics](#evaluation-and-diagnostics)
10. [Putting It All Together](#putting-it-all-together)
11. [Further Reading](#further-reading)

## Why Transformers? A Short History

### The Sequential Processing Bottleneck

Traditional MLPs process fixed-size inputs, making them unsuitable for variable-length sequences. Early approaches like bag-of-words discarded sequence order entirely, losing critical information. The fundamental challenge: **how do we model dependencies between positions while preserving order information?**

### RNNs, LSTMs, and Their Limitations

Recurrent Neural Networks addressed sequence modeling by maintaining hidden state across time steps. LSTMs and GRUs improved on vanilla RNNs by solving vanishing gradients through gating mechanisms. However, three critical problems remained:

1. **Sequential bottleneck**: Each step depends on the previous, preventing parallelization
2. **Exposure bias**: Training uses ground truth, inference uses model predictions
3. **Limited long-range dependencies**: Information must flow through all intermediate states

### Seq2Seq + Attention: The Bridge

The seq2seq architecture with attention mechanisms ([Bahdanau et al., 2014](https://arxiv.org/abs/1409.0473)) introduced a crucial insight: **decoders can directly attend to any encoder position**. This eliminated the fixed-size bottleneck but still required sequential processing within encoder/decoder.

### The Key Leap: "Attention Is All You Need"

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) asked: what if we remove recurrence entirely and rely purely on attention? The breakthrough was realizing that:

- **Self-attention** allows positions to communicate directly
- **Parallel computation** becomes possible across all positions
- **Long-range dependencies** are captured without degradation
- **Inductive biases** are minimal, letting data drive learned patterns

## Transformer Fundamentals

### Tokenization and Embeddings

Modern transformers operate on **subword tokens** rather than characters or words. Two dominant approaches:

- **BPE (Byte-Pair Encoding)**: Merges frequent character pairs iteratively
- **SentencePiece**: Treats text as Unicode sequences, handles multilingual better

Token embeddings map discrete tokens to dense vectors. The embedding dimension $d_{model}$ typically ranges from 512 to 8192, balancing expressiveness with computational cost.

### Positional Information

Since attention is permutation-invariant, position information must be injected explicitly. Four main approaches:

**Sinusoidal Embeddings**: Original transformer design using sin/cos waves:
```
PE(pos, 2i) = sin(pos/10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
```

**Learned Embeddings**: Trainable position vectors, limited to fixed maximum length.

**RoPE (Rotary Position Embedding)**: Rotates query/key vectors by position-dependent angles, enabling length extrapolation.

**ALiBi (Attention with Linear Biases)**: Adds position-based linear bias to attention scores, very simple and effective.

### Scaled Dot-Product Attention

The core mechanism computing attention between queries, keys, and values:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix (what each position is looking for)
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix (what each position offers)  
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix (actual information to be aggregated)
- $\sqrt{d_k}$ scaling prevents softmax saturation for large dimensions

**Intuition**: Each query "asks" which keys are relevant, softmax creates a weighted average over values.

### Multi-Head Self-Attention (MHSA)

Instead of single attention, use $h$ parallel heads with different learned projections:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**Why multiple heads?** Each head can specialize in different attention patterns (local vs global, syntactic vs semantic).

### Residual Connections and LayerNorm

Critical for training stability:
- **Residual connections**: $\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$
- **Pre-LN vs Post-LN**: Pre-LN is more stable, enabling deeper models

### Position-wise Feed-Forward Networks (FFNs)

After attention mixes information between positions, FFNs process each position independently:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2  # Original ReLU version
FFN(x) = GELU(xW_1 + b_1)W_2 + b_2    # Modern GELU/SwiGLU variants
```

**Why FFNs matter**: Attention routes information, FFNs transform it. The intermediate dimension is typically 4× the model dimension.

### Minimal Self-Attention Implementation

```python
import torch
import torch.nn.functional as F

def simple_attention(x, d_k=64):
    # x: [seq_len, d_model] - single sequence, no batching
    seq_len, d_model = x.shape
    
    # Single-head projections - normally we'd have multiple heads
    W_q = torch.randn(d_model, d_k) / (d_model ** 0.5)
    W_k = torch.randn(d_model, d_k) / (d_model ** 0.5) 
    W_v = torch.randn(d_model, d_k) / (d_model ** 0.5)
    
    Q = x @ W_q  # [seq_len, d_k] - what each position seeks
    K = x @ W_k  # [seq_len, d_k] - what each position offers  
    V = x @ W_v  # [seq_len, d_k] - actual content to mix
    
    # Attention scores: how much does each query attend to each key?
    scores = Q @ K.T / (d_k ** 0.5)  # [seq_len, seq_len]
    weights = F.softmax(scores, dim=-1)  # Row-wise softmax
    
    return weights @ V  # [seq_len, d_k] - weighted combination of values
```

## Architectural Variants and When to Use Them

### Encoder-Only: BERT Family

**Structure**: Bidirectional self-attention across all positions
**Training**: Masked Language Modeling (MLM) - predict randomly masked tokens
**Use cases**: Classification, entity recognition, semantic similarity

**Key advantage**: Full bidirectional context
**Limitation**: Cannot generate sequences autoregressively

### Decoder-Only: GPT Family  

**Structure**: Causal self-attention - each position only attends to previous positions
**Training**: Causal Language Modeling (CLM) - predict next token
**Use cases**: Text generation, completion, few-shot learning

**Key advantage**: Natural generation capability
**Trade-off**: No future context during training

### Encoder-Decoder: T5 Family

**Structure**: Encoder (bidirectional) + Decoder (causal) with cross-attention
**Training**: Various objectives (span corruption, translation, etc.)
**Use cases**: Translation, summarization, structured tasks

**Key advantage**: Combines bidirectional understanding with generation
**Cost**: Higher parameter count and computational complexity

### Modern Architectural Innovations

**Multi-Query Attention (MQA)**: Single key/value head shared across queries, reducing KV cache size

**Grouped-Query Attention (GQA)**: Compromise between MHA and MQA - groups of queries share K/V

**Mixture of Experts (MoE)**: Replace some FFNs with expert networks, activated sparsely based on routing

**Vision Integration**: Vision Transformer (ViT) patches images into sequences. Cross-modal attention connects vision and language tokens.

## Training Objectives and Data Curriculum

### Core Pre-training Objectives

**Causal Language Modeling (CLM)**: Predict $p(x_{t+1} | x_1, ..., x_t)$
- Simple, scales well with data
- Emergent capabilities appear with scale

**Masked Language Modeling (MLM)**: Predict masked tokens using bidirectional context
- Better representations for understanding tasks
- Doesn't naturally generate sequences

**Span Corruption**: Mask contiguous spans, predict them autoregressively
- Bridges understanding and generation
- Used in T5, UL2

### Supervised Fine-Tuning and Instruction Tuning

After pre-training, models learn to follow instructions through supervised fine-tuning on (instruction, response) pairs. Key considerations:

- **Data quality over quantity**: Better curation improves performance
- **Format consistency**: Standardized prompt templates help generalization  
- **Task diversity**: Broad instruction coverage improves zero-shot capabilities

### Alignment: RLHF and Beyond

**Reinforcement Learning from Human Feedback (RLHF)**:
1. Train reward model on human preferences
2. Use PPO to optimize policy against reward model
3. Balances helpfulness, harmlessness, honesty

**Direct Preference Optimization (DPO)**: Alternative approach optimizing preferences directly without explicit reward model, simpler and more stable.

**Critical considerations**:
- **Data contamination**: Evaluation data leaking into training
- **Distribution mismatch**: Training vs deployment contexts
- **Reward hacking**: Models exploiting reward model weaknesses

## How Attention Actually Runs at Scale

### The O(n²) Problem

Self-attention computes pairwise relationships between all positions:
- **Memory**: O(n²) to store attention matrix
- **Compute**: O(n²d) for score computation
- At 2M tokens, attention matrix alone requires 16GB memory

### KV Caching for Autoregressive Generation

During generation, past key/value pairs are reused:

```python
def update_kv_cache(cache, new_k, new_v, position):
    # cache: dict with 'keys' [batch, heads, seq_len, d_k] and 'values'
    # new_k, new_v: [batch, heads, 1, d_k] - only the new position
    
    if cache is None:
        # First token - initialize cache
        cache = {'keys': new_k, 'values': new_v}
    else:
        # Append new token to existing cache
        cache['keys'] = torch.cat([cache['keys'], new_k], dim=2)
        cache['values'] = torch.cat([cache['values'], new_v], dim=2)
    
    return cache  # Return updated cache for next iteration
```

**Trade-off**: Memory grows linearly with sequence length, but compute per step stays constant.

### Efficient Attention Variants

**Sliding Window Attention**: Each position only attends to local neighborhood
- Longformer, BigBird use this pattern
- O(n) complexity but limited long-range modeling

**Block-Sparse Attention**: Attention follows structured sparse patterns
- Global + local + strided attention patterns
- Balances efficiency and expressiveness

**Linear Attention**: Reformulate attention to avoid explicit n² computation
- Performer uses random feature approximation
- Often sacrifices quality for efficiency

**FlashAttention**: Algorithm-level optimization using memory hierarchy
- Tiles attention computation to reduce memory traffic
- Achieves full O(n²) attention with better wall-clock time
- No approximation - mathematically equivalent to standard attention

### Serving at Scale

**Dynamic Batching**: Combine requests of different lengths efficiently

**Paged Attention**: Virtual memory for KV cache, enabling better memory utilization

**Speculative Decoding**: Use small model to predict multiple tokens, verify with large model

## The MLP (FFN) Block in Practice

### Role in the Architecture

While attention **routes** information between positions, MLPs **transform** the representations:

- **Attention**: "Which information should flow where?"
- **MLP**: "How should this information be processed?"

This creates a two-stage pattern: communicate, then compute.

### Width and Activation Choices

**Standard dimensions**: $d_{ffn} = 4 \times d_{model}$ is common, but varies by model family.

**Activation functions**:
- **ReLU**: Original choice, simple but can cause dead neurons
- **GELU**: Smooth approximation to ReLU, better gradients
- **SwiGLU**: Gated variant showing strong empirical performance

### SwiGLU Implementation

```python
import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        # SwiGLU needs 2 projections for gating - wider intermediate dim
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)  # Output projection  
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)  # Value projection
    
    def forward(self, x):
        # Split into gate and value, apply SwiGLU gating
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
```

### Mixture of Experts (MoE)

Replace dense FFNs with sparse expert routing:

**Benefits**: Increased capacity without proportional compute increase
**Challenges**: Load balancing, communication overhead, training stability

**When to use**: Very large models where parameter count matters more than per-token compute.

## Quantization for Practical Deployment

### Post-Training vs Quantization-Aware Training

**Post-Training Quantization (PTQ)**: Convert pre-trained weights without retraining
- Fast deployment, some accuracy loss
- Works well for 8-bit, struggles at 4-bit

**Quantization-Aware Training (QAT)**: Include quantization in training loop
- Better accuracy preservation, requires more resources
- Critical for extreme quantization (2-bit, binary)

### Common Quantization Schemes

**8-bit (INT8)**: Minimal accuracy loss, 2× memory reduction
- Per-tensor or per-channel scaling
- Activation quantization often more challenging than weights

**4-bit (INT4)**: Aggressive compression, requires careful calibration
- GPTQ: Gradual quantization minimizing reconstruction error
- AWQ: Protect activation-sensitive weights from quantization

### Where Quantization Helps Most

1. **Weight storage**: Linear layer parameters (majority of model size)
2. **KV cache**: Memory bound during long sequence generation
3. **Embedding tables**: Large vocabulary models

**Where it's risky**: Attention scores, layer norm statistics, outlier activations

### Quantization Example

```python
# Pseudo-code for 8-bit linear layer loading
def load_quantized_linear(fp16_weights):
    # Find scale factor for symmetric quantization
    scale = fp16_weights.abs().max() / 127
    
    # Quantize weights to int8 range [-128, 127]
    quantized = torch.round(fp16_weights / scale).clamp(-128, 127).to(torch.int8)
    
    return quantized, scale  # Store both for dequantization during forward pass

def quantized_forward(x, quantized_weights, scale):
    # Dequantize weights just-in-time for computation
    weights_fp16 = quantized_weights.float() * scale
    return torch.nn.functional.linear(x, weights_fp16)
```

### QLoRA: Quantized Base + Low-Rank Adapters

Combines aggressive quantization with parameter-efficient fine-tuning:
- Base model in 4-bit precision (frozen)
- LoRA adapters in full precision (trainable)
- Enables fine-tuning 70B models on single GPU

## Fine-Tuning and Parameter-Efficient Methods

### Full Fine-Tuning vs Parameter Efficiency

**Full fine-tuning** updates all parameters, providing maximum flexibility but requiring substantial compute and risking catastrophic forgetting.

**Parameter-efficient methods** update only small subsets, preserving pre-trained knowledge while adapting to new tasks.

### Low-Rank Adaptation (LoRA)

**Core insight**: Fine-tuning updates have low intrinsic dimensionality. Approximate weight updates with low-rank decomposition:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, and $r \ll d$.

```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=16, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        
        # Initialize A with small random, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # Frozen base computation + low-rank adaptation
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * (self.alpha / self.r)
        return base_out + lora_out
```

**Benefits**: ~0.1% parameters updated, maintains base model performance
**Limitations**: Limited expressiveness for very different target domains

### Other Parameter-Efficient Methods

**Prefix Tuning**: Prepend trainable "soft prompts" to input sequences
**P-Tuning v2**: Trainable prompts at multiple layers
**IA³**: Scale intermediate activations with learned vectors

### Instruction Tuning Best Practices

1. **Data diversity**: Cover wide range of instruction types
2. **Format consistency**: Standardize instruction/response templates
3. **Quality control**: Filter low-quality examples aggressively
4. **Evaluation alignment**: Test on representative downstream tasks

**Common pitfalls**:
- **Overfitting**: Small datasets lead to degraded general capabilities  
- **Distribution shift**: Training vs deployment instruction formats
- **Safety regression**: Fine-tuning can reduce safety alignment

## Evaluation and Diagnostics

### Intrinsic vs Extrinsic Metrics

**Perplexity**: $\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i | x_{<i})\right)$
- Good for comparing models on same domain
- Doesn't correlate perfectly with downstream performance

**Capability benchmarks**: MMLU, HumanEval, GSM8K, etc.
- More aligned with user utility
- But can be gamed or contaminated

### Long-Context Evaluation

**Needle-in-a-haystack**: Place specific information in long context, test retrieval
**Synthetic tasks**: Sorting, counting, simple reasoning over long sequences
**Real-world tasks**: Document QA, code completion with large context

**Caveat**: Models may memorize evaluation patterns rather than truly understanding long context.

### Performance Metrics

**Latency**: Time to first token (TTFT) vs time per token (TBT)
**Throughput**: Tokens per second, requests per second  
**Memory**: Peak usage, KV cache growth
**Quality**: BLEU, ROUGE, human evaluation

### Common Failure Modes

1. **Attention collapse**: All positions attend uniformly (attention entropy too low)
2. **Gradient vanishing**: Very deep models fail to train effectively
3. **Position interpolation failure**: Poor performance beyond training sequence length
4. **Calibration issues**: Overconfident predictions on uncertain queries

**Quick debugging checklist**:
- Check attention weight distributions
- Monitor gradient norms across layers  
- Validate position embedding behavior
- Test on known good examples

## Putting It All Together

### Information Flow in a Transformer Block

1. **Input**: Token embeddings + positional information
2. **Self-Attention**: Each position gathers information from relevant positions
3. **Residual + LayerNorm**: Stabilize and combine with input
4. **Feed-Forward**: Transform gathered information independently per position
5. **Residual + LayerNorm**: Final stabilization before next block

### Complete Forward Pass

```
Tokenization: "Hello world" → [1024, 2043]
↓
Embeddings: [2, 512] (seq_len=2, d_model=512)
↓
+ Positional: Add position information
↓
[Attention + FFN] × L: L transformer blocks
↓
Final LayerNorm: Stabilize representations  
↓
Output projection: [2, 512] → [2, vocab_size]
↓
Softmax: Convert logits to probabilities
```

### Layer Cooperation

**Early layers**: Learn basic patterns, syntax, local dependencies
**Middle layers**: Develop complex representations, long-range dependencies
**Late layers**: Task-specific processing, output formatting

**Attention patterns evolve**: Early layers show local attention, deeper layers learn semantic relationships.

**FFN specialization**: Some neurons become feature detectors, others perform specific computations.

### Training Dynamics

1. **Embedding learning**: Tokens learn semantic representations
2. **Attention development**: Heads specialize in different relationship types
3. **FFN optimization**: Neurons learn to process attended information
4. **Layer norm stabilization**: Maintains activation scales throughout training

**Key insight**: The architecture provides inductive biases, but most knowledge comes from data scale and training dynamics.

## Further Reading

### Foundation Papers
- **"Attention Is All You Need"** (Vaswani et al., 2017): [Original transformer paper](https://arxiv.org/abs/1706.03762)
- **BERT** (Devlin et al., 2018): [Bidirectional encoder representations](https://arxiv.org/abs/1810.04805)
- **GPT** (Radford et al., 2018): [Generative pre-training approach](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- **T5** (Raffel et al., 2019): [Text-to-text transfer transformer](https://arxiv.org/abs/1910.10683)

### Efficiency and Scaling
- **FlashAttention** (Dao et al., 2022): [Memory-efficient attention](https://arxiv.org/abs/2205.14135)
- **RoPE** (Su et al., 2021): [Rotary position embedding](https://arxiv.org/abs/2104.09864)
- **ALiBi** (Press et al., 2021): [Train short, test long attention bias](https://arxiv.org/abs/2108.12409)
- **MoE** (Shazeer et al., 2017): [Sparsely-gated mixture of experts](https://arxiv.org/abs/1701.06538)

### Fine-tuning and Adaptation
- **LoRA** (Hu et al., 2021): [Low-rank adaptation](https://arxiv.org/abs/2106.09685)
- **QLoRA** (Dettmers et al., 2023): [Efficient finetuning of quantized models](https://arxiv.org/abs/2305.14314)
- **RLHF** (Ouyang et al., 2022): [Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

### Quantization and Deployment  
- **GPTQ** (Frantar et al., 2022): [Accurate post-training quantization](https://arxiv.org/abs/2210.17323)
- **AWQ** (Lin et al., 2023): [Activation-aware weight quantization](https://arxiv.org/abs/2306.00978)
- **8-bit Inference** (Dettmers et al., 2022): [LLM.int8() breakthrough](https://arxiv.org/abs/2208.07339)

---

*This tutorial provides a comprehensive foundation for understanding and implementing transformer architectures. The field moves rapidly - always verify current best practices and emerging techniques.*
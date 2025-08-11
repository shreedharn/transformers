# Transformer Flow: From Text Input to Output Generation

## Overview

Transformer architectures represent a fundamental paradigm shift in sequence modeling, replacing recurrent and convolutional approaches with attention-based mechanisms for parallel processing of sequential data. The architecture's core innovation lies in the self-attention mechanism, which enables direct modeling of dependencies between any two positions in a sequence, regardless of their distance.

**Architectural Significance:** Transformers solve the fundamental bottleneck of sequential processing inherent in RNNs while capturing long-range dependencies more effectively than CNNs. The attention mechanism provides explicit, learnable routing of information between sequence positions, enabling the model to dynamically focus on relevant context.

**Key Innovations:**
- **Parallelizable attention computation:** Unlike RNNs, all positions can be processed simultaneously
- **Direct dependency modeling:** Attention weights explicitly model relationships between any pair of sequence positions
- **Position-invariant processing:** The base attention mechanism is permutation-equivariant, requiring explicit positional encoding

## Prerequisites

**Mathematical Foundations:**
- **Linear Algebra**: Matrix operations, eigenvalues, vector spaces (📚 See [Linear Algebra Essentials](./transformer_math.md#21-linear-algebra-essentials))
- **Probability Theory**: Distributions, information theory, maximum likelihood estimation (📚 See [Probability & Information Theory](./transformer_math.md#23-probability--information-theory))
- **Calculus**: Gradients, chain rule, optimization theory (📚 See [Matrix Calculus](./transformer_math.md#22-matrix-calculus-essentials))
- **Machine Learning**: Backpropagation, gradient descent, regularization techniques

**This document assumes familiarity with deep learning fundamentals and focuses on the mathematical rigor and computational details specific to transformer architectures.**

## Abstract

This document provides a comprehensive analysis of transformer architecture data flow, from tokenization through output generation. We examine the mathematical formulations, computational complexity, and implementation details of each component, with emphasis on the attention mechanism, layer composition, and training dynamics. The treatment assumes undergraduate-level mathematical sophistication and focuses on the rigorous understanding necessary for research and advanced implementation.

**📚 Mathematical Foundations:** Core mathematical concepts and derivations are detailed in [transformer_math.md](./transformer_math.md), providing the theoretical foundation for the computational procedures described here.

---

## Table of Contents

1. [Overview: The Complete Pipeline](#1-overview-the-complete-pipeline)
2. [Stage 1: Text to Tokens](#2-stage-1-text-to-tokens)
3. [Stage 2: Tokens to Embeddings](#3-stage-2-tokens-to-embeddings)
4. [Stage 3: Through the Transformer Stack](#4-stage-3-through-the-transformer-stack)
5. [Stage 4: Self-Attention Deep Dive](#5-stage-4-self-attention-deep-dive)
6. [Stage 5: KV Cache Operations](#6-stage-5-kv-cache-operations)
7. [Stage 6: Feed-Forward Networks](#7-stage-6-feed-forward-networks)
8. [Stage 7: Output Generation](#8-stage-7-output-generation)
9. [Training: Backpropagation Flow](#9-training-backpropagation-flow)
10. [Weight Updates and Optimization](#10-weight-updates-and-optimization)
11. [Complete Mathematical Summary](#11-complete-mathematical-summary)

---

## 1. Overview: The Complete Pipeline

**Let's trace through what happens when you type "The cat sat on" and the AI predicts the next word.**

Think of this process like a sophisticated translation system - but instead of translating between languages, we're translating from "human text" to "AI understanding" and back to "human text".

### Computational Pipeline Overview

**Core Processing Stages:**

1. **Tokenization:** Subword segmentation and vocabulary mapping
2. **Embedding:** Dense vector representation learning
3. **Position Encoding:** Sequence order information injection
4. **Transformer Layers:** Attention-based representation refinement
5. **Output Projection:** Vocabulary distribution computation
6. **Decoding:** Next-token selection strategies

### Detailed Process Flow

```
User Input: "The cat sat on"
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                             │
│              (How AI Understands Text)                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Tokenization:     "The cat sat on" → [464, 2415, 3332, 319] │
│    (Break into computer-friendly pieces)                   │
│                                                             │
│ 2. Embedding:        tokens → vectors [4, 768]             │
│    (Convert numbers to meaning representations)            │
│                                                             │
│ 3. Position Encoding: add positional info [4, 768]        │
│    (Tell the model WHERE each word appears)                │
│                                                             │
│ 4. Transformer Stack: 12 layers of attention + processing │
│    (Deep understanding - like reading comprehension)       │
│                                                             │
│ 5. Output Projection: → probabilities for all 50,000 words│
│    (Consider every possible next word)                     │
│                                                             │
│ 6. Sampling:         choose from top candidates            │
│    (Make the final decision)                               │
└─────────────────────────────────────────────────────────────┘
       ↓
Output: "the" (most likely next word)
```

**Tensor Dimensions and Semantic Interpretation:**
- **[464, 2415, 3332, 319]**: Discrete token indices mapping to vocabulary entries
- **[4, 768]**: Sequence length × model dimension tensor representing learned embeddings
- **12 layers**: Hierarchical representation learning through stacked transformer blocks
- **50,000 words**: Vocabulary size determining output distribution dimensionality

### Training: How Models Learn (Optional Detail)

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING: BACKWARD PASS                   │
│                (How AI Learns from Mistakes)               │
├─────────────────────────────────────────────────────────────┤
│ 1. Compute Loss:     Compare prediction with correct answer│
│    (Like grading a test - how wrong was the guess?)       │
│                                                             │
│ 2. Backpropagation: Find what caused the mistake          │
│    (Trace back through all the steps to find errors)      │
│                                                             │
│ 3. Weight Updates:   Adjust internal parameters            │
│    (Fine-tune the model to do better next time)           │
└─────────────────────────────────────────────────────────────┘
```

### Model Hyperparameters and Complexity

**Architectural Dimensions:**
- **Vocabulary size (V)**: Discrete token space cardinality, typically 32K-100K
- **Model dimension (d_model)**: Hidden state dimensionality determining representational capacity
- **Context length (n)**: Maximum sequence length for attention computation
- **Layer count (N)**: Depth of hierarchical representation learning
- **Attention heads (H)**: Parallel attention subspaces for diverse relationship modeling

**GPT-2 Specification:**
- Vocabulary: 50,257 BPE tokens
- Hidden dimension: 768
- Context window: 1,024 tokens
- Transformer blocks: 12 layers
- Multi-head attention: 12 heads per layer

---

## 2. Stage 1: Text to Tokens

### 🎯 Intuition: Breaking Text into Computer-Friendly Pieces

**Think of tokenization like chopping vegetables for a recipe.** You can't feed raw text directly to a computer - you need to break it into standardized pieces (tokens) that the computer can work with, just like chopping vegetables into uniform sizes for cooking.

**Why not just use individual letters or whole words?**
- **Letters**: Too many combinations, loses meaning ("c-a-t" tells us less than "cat")
- **Whole words**: Millions of possible words, can't handle new/misspelled words
- **Subwords** (what we use): Perfect balance - captures meaning while handling new words

### Process Flow
```
"The cat sat on" 
       ↓
┌─────────────────┐
│   Tokenizer     │  ← BPE/SentencePiece algorithm
│   - Split text  │    (Like a smart word chopper)
│   - Map to IDs  │
└─────────────────┘
       ↓
[464, 2415, 3332, 319]
```

### Simple Example with Real Numbers

Let's trace through tokenizing "The cat sat on":

**Step 1: Smart Splitting**
```
"The cat sat on" → ["The", " cat", " sat", " on"]
```
Notice: Spaces are included with words (except the first)

**Step 2: Look Up Numbers**
```
"The"   → 464   (most common word, low number)
" cat"  → 2415  (common word)  
" sat"  → 3332  (less common)
" on"   → 319   (very common, low number)
```

**Final Result:**
```
Input: "The cat sat on"      (18 characters)
Output: [464, 2415, 3332, 319]  (4 tokens)
Shape: [4] (sequence length = 4)
```

### Mathematical Formulation

**Tokenization Mapping**: 
$$T: \Sigma^* \to \{0, 1, \ldots, V-1\}^*$$

where:
- $\Sigma^*$ represents the space of all possible text strings
- $V$ is the vocabulary size 
- Output is a variable-length sequence of discrete token indices

**BPE Algorithm**: Given text $s$ and learned merge operations $M$:
$$\text{tokenize}(s) = [\text{vocab}[\tau_i] \mid \tau_i \in \text{BPE\_segment}(s, M)]$$

where $\text{BPE\_segment}$ applies the learned merge rules to produce subword tokens $\tau_i$.

**📖 Detailed Algorithm:** See [Tokenization Mathematics](./transformer_math.md#102-embedding-mathematics) for BPE training and inference procedures.

### Tokenization Challenges and Considerations

**Out-of-Vocabulary Handling:**
- **Unknown words**: Long or rare words decompose into character-level tokens, increasing sequence length
- **Domain mismatch**: Models trained on general text may poorly tokenize specialized domains (code, scientific notation)
- **Multilingual complexity**: Subword boundaries vary across languages, affecting cross-lingual transfer

**Performance Implications:**
- **Vocabulary size vs. sequence length trade-off**: Larger vocabularies reduce sequence length but increase computational cost
- **Compression efficiency**: Good tokenization minimizes information loss while maximizing compression

### Implementation Considerations:
- **Subword algorithms**: BPE, WordPiece, SentencePiece each have different inductive biases
- **Special tokens**: Sequence delimiters (`<|endoftext|>`, `<|pad|>`, `<|unk|>`) require careful handling
- **Context limitations**: Finite attention window constrains sequence length (typically 512-8192 tokens)
- **Batching requirements**: Variable-length sequences necessitate padding strategies

---

## 3. Stage 2: Tokens to Embeddings

### Dense Vector Representation Learning

Embedding layers transform discrete token indices into continuous vector representations within a learned semantic space. This mapping enables the model to capture distributional semantics and enables gradient-based optimization over the discrete vocabulary.

**Representational Advantages:**
- **Continuous optimization**: Dense vectors enable gradient-based learning over discrete token spaces
- **Semantic similarity**: Geometrically related vectors capture semantic relationships through distance metrics
- **Compositional structure**: Linear operations in embedding space can capture meaningful transformations

**Position encoding** addresses the permutation-invariance of attention mechanisms by injecting sequential order information into the representation space.

### Embedding Computation Pipeline

**Token Embedding Lookup:**
1. **Matrix indexing**: $E \in \mathbb{R}^{V \times d_{\text{model}}}$ provides learnable token representations
2. **Batch lookup**: $X_{\text{tok}} = E[T]$ where $T$ contains token indices
3. **Result**: Dense matrix $X_{\text{tok}} \in \mathbb{R}^{n \times d_{\text{model}}}$

**Position Encoding Addition:**
1. **Position information**: $PE \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$ encodes sequential order
2. **Sequence slicing**: $X_{\text{pos}} = PE[0:n]$ for sequence length $n$
3. **Element-wise addition**: $X = X_{\text{tok}} + X_{\text{pos}}$

**Output**: Combined semantic and positional representation $X \in \mathbb{R}^{n \times d_{\text{model}}}$

### Mathematical Formulation

**Token Embedding Transformation:**
$$X_{\text{tok}} = E[T] \quad \text{where } E \in \mathbb{R}^{V \times d_{\text{model}}}$$
$$X_{\text{tok}}[i] = E[t_i] \in \mathbb{R}^{d_{\text{model}}} \quad \text{for token index } t_i$$

**Position Encoding Variants:**

**Learned Positional Embeddings:**
$$X_{\text{pos}}[i] = PE[i] \quad \text{where } PE \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$$

**Sinusoidal Position Encoding:**
$$PE[\text{pos}, 2i] = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE[\text{pos}, 2i+1] = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**Combined Representation:**
$$X = X_{\text{tok}} + X_{\text{pos}} \in \mathbb{R}^{n \times d_{\text{model}}}$$

where $n$ is the sequence length and $d_{\text{model}}$ is the model dimension.

**📖 Theoretical Foundation:** See [Embedding Mathematics](./transformer_math.md#102-embedding-mathematics) and [Positional Encodings](./transformer_math.md#82-positional-encodings) for detailed analysis of learned vs. fixed position encodings.

### Concrete Example with Actual Numbers

Let's see what happens with our tokens [464, 2415, 3332, 319] ("The cat sat on"):

**Step 1: Token Embedding Lookup**
```
Token 464 ("The") → [0.1, -0.3, 0.7, 0.2, ...] (768 numbers total)
Token 2415 ("cat") → [-0.2, 0.8, -0.1, 0.5, ...]
Token 3332 ("sat") → [0.4, 0.1, -0.6, 0.3, ...]  
Token 319 ("on") → [0.2, -0.4, 0.3, -0.1, ...]
Result: [4, 768] array (4 words × 768 features each)
```

**Step 2: Position Embeddings**  
```
Position 0 → [0.01, 0.02, -0.01, 0.03, ...] (for "The")
Position 1 → [0.02, -0.01, 0.02, -0.01, ...] (for "cat")  
Position 2 → [-0.01, 0.03, 0.01, 0.02, ...] (for "sat")
Position 3 → [0.03, 0.01, -0.02, 0.01, ...] (for "on")
Result: [4, 768] array (position info for each word)
```

**Step 3: Add Together**
```
Final embedding for "The" = [0.1, -0.3, 0.7, 0.2, ...] + [0.01, 0.02, -0.01, 0.03, ...]
                           = [0.11, -0.28, 0.69, 0.23, ...]
```

### Tensor Shapes Example:
```
seq_len = 4, d_model = 768, vocab_size = 50257

Token IDs:     [4]         (4 tokens: "The", "cat", "sat", "on")
Embeddings:    [4, 768]    (lookup from [50257, 768] - each token → 768 numbers)
Pos Embed:     [4, 768]    (slice from [2048, 768] - position info for each token)
Final X:       [4, 768]    (combine token + position info)
```

**🎯 Shape Intuition:** Think of shapes like `[4, 768]` as a table with 4 rows (tokens) and 768 columns (features). Each row represents one word, each column represents one aspect of meaning.

### 🛠️ What Could Go Wrong?

**Common Misconceptions:**
- **"Why 768 dimensions?"** It's like having 768 different personality traits - enough to capture complex meaning relationships
- **"Why add position info?"** Without it, "cat sat on mat" and "mat on sat cat" would look identical to the model
- **"What if we run out of positions?"** Models have maximum sequence lengths (like 2048) - longer texts get truncated

---

## 4. Stage 3: Through the Transformer Stack

### Hierarchical Representation Learning

The transformer stack implements hierarchical feature extraction through stacked self-attention and feed-forward blocks. Each layer refines the input representations by modeling increasingly complex relationships and patterns within the sequence.

**Layer-wise Functionality:**
1. **Self-attention sublayer**: Models pairwise interactions between sequence positions
2. **Feed-forward sublayer**: Applies position-wise transformations to individual token representations
3. **Residual connections**: Preserve gradient flow and enable identity mappings
4. **Layer normalization**: Stabilizes training dynamics and accelerates convergence

**Representational Hierarchy:** Empirical analysis suggests progressive abstraction:
- **Early layers**: Syntactic patterns, local dependencies, surface-level features
- **Middle layers**: Semantic relationships, discourse structure, compositional meaning
- **Late layers**: Task-specific abstractions, high-level reasoning patterns

### Architecture Overview
```
X [seq_len, d_model] ← Our embedded words: [4, 768]
       ↓
┌─────────────────────────────────┐
│        Layer 1                  │
│  ┌─────────────────────────────┐│
│  │    Multi-Head Attention     ││ ← "What words relate to what?"
│  └─────────────────────────────┘│
│               ↓                 │
│  ┌─────────────────────────────┐│
│  │     Feed Forward            ││ ← "What does this combination mean?"
│  └─────────────────────────────┘│
└─────────────────────────────────┘
       ↓ (Better understanding)
┌─────────────────────────────────┐
│        Layer 2                  │
│           ...                   │ ← Even deeper analysis
└─────────────────────────────────┘
       ↓
       ... (10 more layers)
       ↓
┌─────────────────────────────────┐
│        Layer 12                 │ ← Final, sophisticated understanding
└─────────────────────────────────┘
       ↓
Final Hidden States [4, 768] ← Deeply processed word meanings
```

### Single Layer Mathematical Flow

**Transformer Block Architecture (Pre-LayerNorm):**

Given layer input $X^{(l-1)} \in \mathbb{R}^{n \times d_{\text{model}}}$:

**Self-Attention Sublayer:**
$$\tilde{X}^{(l-1)} = \text{LayerNorm}(X^{(l-1)})$$
$$A^{(l)} = \text{MultiHeadAttention}(\tilde{X}^{(l-1)}, \tilde{X}^{(l-1)}, \tilde{X}^{(l-1)})$$
$$X'^{(l)} = X^{(l-1)} + A^{(l)} \quad \text{(residual connection)}$$

**Feed-Forward Sublayer:**
$$\tilde{X}'^{(l)} = \text{LayerNorm}(X'^{(l)})$$
$$F^{(l)} = \text{FFN}(\tilde{X}'^{(l)})$$
$$X^{(l)} = X'^{(l)} + F^{(l)} \quad \text{(residual connection)}$$

**Design Rationale:**
- **Pre-normalization**: Stabilizes gradients and enables deeper networks compared to post-norm
- **Residual connections**: Address vanishing gradient problem via identity shortcuts
- **Two-sublayer structure**: Separates relationship modeling (attention) from feature transformation (FFN)

**📖 Theoretical Analysis:** See [Transformer Block Mathematics](./transformer_math.md#91-complete-block-equations) and [Residual Connections as Dynamical Systems](./transformer_math.md#32-residual-connections-as-discretized-dynamics) for detailed mathematical foundations.

---

## 5. Stage 4: Self-Attention Deep Dive

### Attention as Differentiable Key-Value Retrieval

Self-attention implements a differentiable associative memory mechanism where each sequence position (query) retrieves information from all positions (keys) based on learned compatibility functions. This enables modeling arbitrary dependencies without the sequential constraints of RNNs or the locality constraints of CNNs.

**Attention Mechanism Properties:**
- **Permutation equivariance**: Output permutes consistently with input permutation (before positional encoding)
- **Dynamic routing**: Information flow adapts based on input content rather than fixed connectivity
- **Parallel computation**: All pairwise interactions computed simultaneously
- **Global receptive field**: Each position can directly attend to any other position

**Self-attention vs. Cross-attention**: In self-attention, queries, keys, and values all derive from the same input sequence, enabling the model to relate different positions within a single sequence.

### Attention Weight Interpretation

**Attention Weights as Soft Alignment:**
Given query position $i$ and key positions $\{j\}$, attention weights $\alpha_{ij}$ represent the proportion of position $j$'s information incorporated into position $i$'s updated representation.

**Example: Coreference Resolution**
For sequence "The cat sat on it" processing token "it":
- $\alpha_{\text{it},\text{The}} = 0.05$ (minimal syntactic relevance)
- $\alpha_{\text{it},\text{cat}} = 0.15$ (potential antecedent)
- $\alpha_{\text{it},\text{sat}} = 0.10$ (predicate information)
- $\alpha_{\text{it},\text{on}} = 0.15$ (spatial relationship)
- $\alpha_{\text{it},\text{it}} = 0.55$ (self-attention for local processing)

**Constraint**: $\sum_j \alpha_{ij} = 1$ (probability simplex constraint from softmax normalization)

### Attention Computation Steps

**Step 1: Linear Projections**
- **Input**: $X \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Projections**: 
  - $Q = XW^Q \in \mathbb{R}^{n \times d_{\text{model}}}$
  - $K = XW^K \in \mathbb{R}^{n \times d_{\text{model}}}$
  - $V = XW^V \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Purpose**: Transform input into query, key, and value representations

**Step 2: Multi-Head Reshaping**
- **Reshape**: Split each matrix into $H$ heads of dimension $d_k = d_{\text{model}}/H$
- **Result shapes**: $Q, K, V \in \mathbb{R}^{n \times H \times d_k}$
- **Computational advantage**: Enables parallel processing of different attention patterns

**Step 3: Scaled Dot-Product Attention**
- **Compatibility scores**: $S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times H \times n}$
- **Scaling rationale**: Prevents softmax saturation as dimensionality increases
- **Complexity**: $O(n^2 d_k)$ per head, $O(H \cdot n^2 \cdot d_k) = O(n^2 d_{\text{model}})$ total

**Step 4: Causal Masking (for autoregressive models)**
- **Mask application**: $S_{\text{masked}} = S + M$ where $M_{ij} = -\infty$ for $j > i$
- **Effect**: Ensures position $i$ cannot attend to future positions
- **Implementation**: Applied before softmax to produce zero attention weights

**Step 5: Attention Weight Computation**
- **Normalization**: $A = \text{softmax}(S_{\text{masked}}, \text{dim}=-1) \in \mathbb{R}^{n \times H \times n}$
- **Properties**: Each row sums to 1, forming probability distributions
- **Interpretation**: $A_{ijh}$ represents how much position $j$ contributes to position $i$ in head $h$

**Step 6: Value Aggregation**
- **Weighted sum**: $O = AV \in \mathbb{R}^{n \times H \times d_k}$
- **Information flow**: Each position receives information from all positions weighted by attention

**Step 7: Head Concatenation and Output Projection**
- **Concatenation**: Reshape $O$ to $\mathbb{R}^{n \times d_{\text{model}}}$
- **Final projection**: $\text{Output} = OW^O \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Purpose**: Integrate information from all attention heads

### Multi-Head Attention Mathematical Formulation

**Core Attention Mechanism (Scaled Dot-Product):**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix (what information to retrieve)
- $K \in \mathbb{R}^{n \times d_k}$: Key matrix (what information is available)
- $V \in \mathbb{R}^{n \times d_v}$: Value matrix (actual information content)
- $d_k = d_v = d_{\text{model}}/H$: Head dimension
- $\sqrt{d_k}$: Temperature scaling to prevent saturation

**Multi-Head Attention Implementation:**
$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)W^O$$

where each head computes:
$$\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

**Linear Projection Matrices** (not MLPs):
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$: Per-head projection matrices
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$: Output projection matrix

**Implementation Details:**
1. **Parallel computation**: All heads computed simultaneously via reshaped tensor operations
2. **Linear projections**: Simple matrix multiplications, not multi-layer perceptrons
3. **Concatenation**: Head outputs concatenated along feature dimension
4. **Output projection**: Single linear transformation of concatenated heads

**📖 Derivation and Analysis:** See [Multi-Head Attention Theory](./transformer_math.md#81-multi-head-as-subspace-projections) and [Scaling Analysis](./transformer_math.md#72-why-the-sqrt_dk-scaling) for mathematical foundations.

**Causal Masking for Autoregressive Models:**
$$\text{mask}[i, j] = \begin{cases} 
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i 
\end{cases}$$

This lower-triangular mask ensures that position $i$ cannot attend to future positions $j > i$, maintaining the autoregressive property necessary for language modeling.

**Computational Implementation**: Typically applied as additive bias before softmax:
$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} + \text{mask}$$

**Computational Complexity Analysis:**

**Tensor Shapes with Standard Convention:**

Use $Q,K\in\mathbb{R}^{H\times n\times d_k}$. Then $S = Q K^T/\sqrt{d_k}\in\mathbb{R}^{H\times n\times n}$ (row = query position, col = key position). The example with $n{=}4, H{=}12, d_k{=}64$ gives $S\in[12,4,4]$.

**Complete Shape Analysis:**
- Initial projections: $[4, 768] \to [4, 768]$ each for Q, K, V
- Multi-head reshape: $[4, 768] \to [12, 4, 64]$ (heads first)
- Attention scores: $[12, 4, 64] \times [12, 4, 64]^T \to [12, 4, 4]$
- Attention weights: $[12, 4, 4]$ (normalized scores)
- Attention output: $[12, 4, 4] \times [12, 4, 64] \to [12, 4, 64]$
- Final output: $[12, 4, 64] \to [4, 768]$ (concatenate heads)

**Time Complexity:**
- Linear projections: $O(n \cdot d_{\text{model}}^2)$
- Attention computation: $O(H \cdot n^2 \cdot d_k) = O(n^2 \cdot d_{\text{model}})$
- Total per layer: $O(n^2 \cdot d_{\text{model}} + n \cdot d_{\text{model}}^2)$

**Space Complexity:**
- Attention matrices: $O(H \cdot n^2)$ 
- Activations: $O(n \cdot d_{\text{model}})$
- Quadratic scaling with sequence length motivates efficient attention variants

---

## 6. Stage 5: KV Cache Operations

### Autoregressive Generation Optimization

KV caching addresses the computational inefficiency of autoregressive generation by storing and reusing previously computed key-value pairs. This optimization reduces the time complexity of each generation step from quadratic to linear in the context length.

**Computational Motivation:**
During autoregressive generation, attention computations for previous tokens remain unchanged when processing new tokens. Caching eliminates redundant recomputation of these static attention components.

**Performance Impact:**
- **Without caching**: $O(t^2)$ computation per step for context length $t$
- **With caching**: $O(t)$ compute per step; **memory**: $O(t \cdot d_k)$ per head per layer $\Rightarrow O(L \cdot H \cdot t \cdot d_k)$ for the model (no $t^2$ term unless you store full attention matrices)
- **Trade-off**: Memory consumption for computational efficiency

### Computational Efficiency Analysis

**Problem Formulation:**
In standard autoregressive generation at step $t$, computing attention for the new token requires:
1. Recomputing K, V matrices for all previous tokens: $O(t \cdot d_{\text{model}}^2)$
2. Computing attention scores: $O(t^2 \cdot d_{\text{model}})$
3. Total complexity per step: $O(t^2 \cdot d_{\text{model}})$

**KV Cache Solution:**

**Without Cache (Step $t$):**
- Compute $K, V \in \mathbb{R}^{t \times d_{\text{model}}}$ for entire sequence
- Attention computation: $O(t^2 \cdot d_{\text{model}})$
- Memory requirement: $O(t \cdot d_{\text{model}})$ per step

**Note:** This assumes a full re-forward over the length-$t$ context per new token; frameworks differ, but the asymptotic bottleneck is the $t \times t$ score matrix either way.

**With Cache (Step $t$):**
- Compute only $k_{\text{new}}, v_{\text{new}} \in \mathbb{R}^{1 \times d_{\text{model}}}$
- Concatenate with cached $K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{(t-1) \times d_{\text{model}}}$
- Attention computation: $O(t \cdot d_{\text{model}})$
- Total memory requirement: $O(t \cdot d_{\text{model}})$ accumulated

### KV Cache Implementation

**Cache Architecture:**
```python
class KVCache:
    def __init__(self, num_layers, num_heads, d_head, max_seq_len):
        # Per-layer cache storage
        self.cache = {
            layer_idx: {
                'keys': torch.zeros(1, num_heads, max_seq_len, d_head),
                'values': torch.zeros(1, num_heads, max_seq_len, d_head),
                'current_length': 0
            }
            for layer_idx in range(num_layers)
        }
    
    def update(self, layer_idx, new_keys, new_values):
        """Append new key-value pairs to cache"""
        cache_entry = self.cache[layer_idx]
        curr_len = cache_entry['current_length']
        
        # Insert new keys and values
        cache_entry['keys'][:, :, curr_len:curr_len+1] = new_keys
        cache_entry['values'][:, :, curr_len:curr_len+1] = new_values
        cache_entry['current_length'] += 1
        
        return (
            cache_entry['keys'][:, :, :cache_entry['current_length']], 
            cache_entry['values'][:, :, :cache_entry['current_length']]
        )
```

**Mathematical Formulation:**

**Cache Update (Step $t$):**
$$q_{\text{new}} = x_{\text{new}} W^Q \in \mathbb{R}^{1 \times d_k}$$
$$k_{\text{new}} = x_{\text{new}} W^K \in \mathbb{R}^{1 \times d_k}$$
$$v_{\text{new}} = x_{\text{new}} W^V \in \mathbb{R}^{1 \times d_v}$$

**Cache Concatenation:**
$$K_{\text{full}} = [K_{\text{cache}}; k_{\text{new}}] \in \mathbb{R}^{t \times d_k}$$
$$V_{\text{full}} = [V_{\text{cache}}; v_{\text{new}}] \in \mathbb{R}^{t \times d_v}$$

**Cached Attention Computation:**
$$\text{scores} = \frac{q_{\text{new}} K_{\text{full}}^T}{\sqrt{d_k}} \in \mathbb{R}^{1 \times t}$$
$$\text{weights} = \text{softmax}(\text{scores}) \in \mathbb{R}^{1 \times t}$$
$$\text{output} = \text{weights} \cdot V_{\text{full}} \in \mathbb{R}^{1 \times d_v}$$

**Complexity Summary:**

**Time Complexity per Generation Step:**
- Without cache: $O(t^2 \cdot d_{\text{model}})$ where $t$ is current sequence length
- With cache: $O(t \cdot d_{\text{model}})$ per step

**Space Complexity:**
- Cache storage: $O(L \cdot H \cdot n_{\max} \cdot d_k)$ where $L$ is layers, $H$ is heads
- Trade-off: $O(t)$ memory overhead for $O(t)$ speedup per step

**Practical Considerations:**
- Memory bandwidth becomes bottleneck for large models
- Cache pre-allocation avoids dynamic memory allocation overhead
- Quantization (FP16, INT8) can reduce cache memory requirements

---

## 7. Stage 6: Feed-Forward Networks

### Position-wise Nonlinear Transformations

Feed-forward networks in transformers implement position-wise MLP layers that process each token representation independently. This component provides the model's primary source of nonlinear transformation capacity and parametric memory.

**Architectural Role:**
- **Nonlinearity injection**: Introduces essential nonlinear transformations between attention layers
- **Representation expansion**: Temporarily expands representation space for complex computations
- **Parameter concentration**: Contains majority of model parameters (~2/3 in standard architectures)
- **Position independence**: Applies identical transformations to each sequence position

**Design Philosophy:**
FFNs serve as "computational bottlenecks" that force the model to compress and process information efficiently, similar to autoencoder architectures.

### FFN Architecture and Computation

**Standard Two-Layer MLP:**
1. **Expansion layer**: $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{ffn}}}$ where $d_{\text{ffn}} = 4 \cdot d_{\text{model}}$
2. **Activation function**: Element-wise nonlinearity (GELU, ReLU, or SwiGLU)
3. **Contraction layer**: $\mathbb{R}^{d_{\text{ffn}}} \to \mathbb{R}^{d_{\text{model}}}$

**Computational Flow:**
- **Input**: $X \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Expansion**: $H_1 = XW_1 + b_1 \in \mathbb{R}^{n \times d_{\text{ffn}}}$
- **Activation**: $H_2 = \sigma(H_1) \in \mathbb{R}^{n \times d_{\text{ffn}}}$
- **Contraction**: $Y = H_2 W_2 + b_2 \in \mathbb{R}^{n \times d_{\text{model}}}$

**Design Rationale:**
- **4× expansion**: Provides sufficient representational capacity for complex transformations
- **Position-wise**: Each token processed independently, enabling parallelization
- **Bottleneck structure**: Forces efficient information compression and processing

### Mathematical Formulation

**Standard FFN Transformation:**
$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$: Expansion matrix
- $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$: Contraction matrix
- $b_1 \in \mathbb{R}^{d_{\text{ffn}}}, b_2 \in \mathbb{R}^{d_{\text{model}}}$: Bias vectors
- $\sigma$: Activation function (GELU, ReLU, SwiGLU)
- $d_{\text{ffn}} = 4 \cdot d_{\text{model}}$ (standard expansion ratio)

**GELU Activation Function:**
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the standard normal CDF. GELU provides smooth, differentiable activation with improved gradient flow compared to ReLU.

**Approximation**: $\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$

**📖 Activation Function Analysis:** See [GELU vs ReLU](./transformer_math.md#92-why-gelu-over-relu) and [SwiGLU Variants](./transformer_math.md#91-complete-block-equations) for detailed comparisons.

**SwiGLU Variant (Gated FFN):**
$$\text{SwiGLU}(x) = (W_1 x + b_1) \odot \text{SiLU}(W_2 x + b_2)$$

where:
- $\odot$ denotes element-wise (Hadamard) product
- $\text{SiLU}(x) = x \cdot \sigma(x)$ where $\sigma$ is sigmoid
- Requires two parallel linear transformations instead of sequential
- Often uses $d_{\text{ffn}} = \frac{8}{3} d_{\text{model}}$ to maintain parameter count

**Parameter Analysis:**

**Standard FFN:**
- Expansion layer: $d_{\text{model}} \times d_{\text{ffn}} = d_{\text{model}} \times 4d_{\text{model}} = 4d_{\text{model}}^2$
- Contraction layer: $d_{\text{ffn}} \times d_{\text{model}} = 4d_{\text{model}} \times d_{\text{model}} = 4d_{\text{model}}^2$
- Total: $8d_{\text{model}}^2$ parameters (plus biases)

**Example (GPT-2 scale, $d_{\text{model}} = 768$):** ~4.7M parameters per FFN layer

**Computational Complexity:**
- Forward pass: $O(n \cdot d_{\text{model}} \cdot d_{\text{ffn}}) = O(n \cdot d_{\text{model}}^2)$
- Dominates transformer computation cost due to large hidden dimension

---

## 8. Stage 7: Output Generation

### 🎯 Intuition: Making the Final Decision

**Think of output generation like a multiple-choice test with 50,000 possible answers.** After all the deep processing, the model has to decide: "What's the most likely next word?"

**The process:**
1. **Focus on the last word**: Only the final word's understanding matters for prediction
2. **Consider all possibilities**: Calculate how likely each of the 50,000 vocabulary words is to come next
3. **Make a choice**: Use various strategies to pick the final word

**Real-world analogy:** Like a game show where you've analyzed all the clues, and now you must choose your final answer from all possible options.

**Why only the last word?** In "The cat sat on ___", only the understanding after "on" matters for predicting the next word. Previous words have already influenced this final representation through attention.

### Output Generation Pipeline
**Step 1: Hidden State Extraction**
- **Input**: Final hidden states $H \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Selection**: Extract last position $h_{\text{last}} = H[-1, :] \in \mathbb{R}^{d_{\text{model}}}$
- **Rationale**: Only final position contains complete contextual information for next-token prediction

**Step 2: Language Model Head**
- **Linear transformation**: $\text{logits} = h_{\text{last}} W_{\text{lm}} + b_{\text{lm}}$
- **Shape transformation**: $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{|V|}$
- **Weight tying**: Often $W_{\text{lm}} = E^T$ (transpose of embedding matrix)

**Step 3: Temperature Scaling**
- **Scaled logits**: $\text{logits}_{\text{scaled}} = \frac{\text{logits}}{\tau}$
- **Temperature effects**: $\tau > 1$ increases randomness, $\tau < 1$ increases determinism

**Step 4: Probability Computation**
- **Softmax normalization**: $p_i = \frac{\exp(\text{logits}_i / \tau)}{\sum_{j=1}^{|V|} \exp(\text{logits}_j / \tau)}$
- **Result**: Valid probability distribution over vocabulary

**Step 5: Token Sampling**
- **Sampling strategy**: Select next token according to chosen decoding algorithm
- **Options**: Greedy, top-k, nucleus (top-p), beam search

### Mathematical Formulation

**Language Model Head:**
$$\text{logits} = h_{\text{final}} W_{\text{lm}} + b_{\text{lm}}$$

where:
- $h_{\text{final}} \in \mathbb{R}^{d_{\text{model}}}$: Final position hidden state
- $W_{\text{lm}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$: Language model projection matrix
- $b_{\text{lm}} \in \mathbb{R}^{|V|}$: Output bias (often omitted)
- $\text{logits} \in \mathbb{R}^{|V|}$: Unnormalized vocabulary scores

**Weight Tying**: Commonly $W_{\text{lm}} = E^T$ where $E$ is the token embedding matrix, reducing parameters and improving performance.

**Temperature-Scaled Softmax:**
$$p_i = \frac{\exp(\text{logits}_i / \tau)}{\sum_{j=1}^{|V|} \exp(\text{logits}_j / \tau)}$$

**Temperature Parameter $\tau$:**
- $\tau \to 0$: Approaches deterministic (argmax) selection
- $\tau = 1$: Standard softmax distribution
- $\tau \to \infty$: Approaches uniform distribution
- **Calibration**: $\tau$ can be tuned to match desired confidence levels

**Decoding Strategies:**

**Greedy Decoding:**
$$\text{next\_token} = \arg\max_{i} p_i$$
- **Advantages**: Deterministic, computationally efficient
- **Disadvantages**: Can produce repetitive or low-quality text

**Top-k Sampling:**
$$\text{next\_token} \sim \text{Categorical}(\text{top-k}(p, k))$$
- **Process**: Select from k most probable tokens, renormalize, then sample
- **Effect**: Truncates long tail of distribution while preserving diversity

**Nucleus (Top-p) Sampling:**
$$\text{next\_token} \sim \text{Categorical}(\{i : \sum_{j \in \text{top}(p)} p_j \leq p\})$$
- **Dynamic selection**: Includes smallest set of tokens with cumulative probability $\geq p$
- **Adaptive**: Adjusts vocabulary size based on confidence distribution
- **Typical values**: $p \in [0.9, 0.95]$ balances quality and diversity

**Beam Search**: For deterministic high-quality generation, maintains top-$b$ hypotheses at each step

---

## 9. Training: Backpropagation Flow

### 🎯 Intuition: How AI Models Learn from Mistakes

**Think of training like teaching a student to complete sentences.** You show them "The cat sat on the ___" and the correct answer "mat". If they guess "tree", you help them understand why "mat" was better and adjust their thinking process.

**The Learning Process:**
1. **Show examples**: Give the model text with known answers
2. **Let it guess**: Model predicts what comes next
3. **Grade the answer**: Compare prediction with the correct word
4. **Learn from mistakes**: Adjust internal "thought processes" to do better next time

**Why is this called "backpropagation"?** The error information flows backward through all the layers, helping each layer learn what it should have done differently.

**Real-world analogy:** Like a teacher reviewing a student's essay, marking errors, and explaining how each paragraph could be improved - but the "student" is a mathematical network with millions of parameters.

### Loss Computation

**Training Setup:**
```
Input sequence:  [t_1, t_2, t_3, ..., t_n]
Target sequence: [t_2, t_3, t_4, ..., t_{n+1}]  (shifted by 1)

Forward pass produces logits for each position:
logits[i] = prediction for position i+1
```

**Cross-Entropy Loss:**
```
For each position i:
  L_i = -log(P(t_{i+1} | context))
  
Total loss:
  L = (1/n) × Σ L_i = -(1/n) × Σ log(softmax(logits[i])[t_{i+1}])
```

**📖 Mathematical Details:** See [Cross-Entropy Loss](./transformer_math.md#23-probability--information-theory) in transformer_math.md for detailed intuitive explanation

### Backward Pass Flow

```
Loss: L (scalar)
       ↓
┌─────────────────────────────────────┐
│     Gradient w.r.t. Logits          │
│   ∂L/∂logits = probs - targets     │
│   [seq_len, vocab_size]             │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│   Gradient w.r.t. Final Hidden     │
│   ∂L/∂h_final = ∂L/∂logits @ W_lm^T│
│   [seq_len, d_model]                │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│      Through Layer N                │
│   ∂L/∂X^(N-1) = backward_layer_N() │
└─────────────────────────────────────┘
       ↓
       ...
       ↓
┌─────────────────────────────────────┐
│      Through Layer 1                │
│   ∂L/∂X^(0) = backward_layer_1()   │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│   Gradient w.r.t. Embeddings       │
│   ∂L/∂E = scatter_add(∂L/∂X^(0))   │
└─────────────────────────────────────┘
```

### Transformer Layer Backward Pass

**FFN Backward:**

```
# Forward: y = W2 @ φ(W1 @ x + b1) + b2
# Backward (batch-wise):
dY = ∂L/∂y
dH2 = dY @ W2^T
dH1 = dH2 ⊙ φ'(W1 @ x + b1)
∂L/∂x  = dH1 @ W1^T
∂L/∂W2 = φ(W1 @ x + b1)^T @ dY
∂L/∂W1 = x^T @ dH1
∂L/∂b2 = sum(dY, dim=0)
∂L/∂b1 = sum(dH1, dim=0)
```

**Attention Backward:**

Rewrite using $S,P,O$ factoring and the row-wise softmax Jacobian; match the equations in **transformer_math.md** (Section D above).

Let $S = QK^T/\sqrt{d_k}$, $P=\mathrm{softmax}(S)$ (row-wise), $O = PV$. Given $G_O=\partial \mathcal{L}/\partial O$:

$$
\begin{aligned}
G_V &= P^T G_O,\quad &&\text{(V same shape as }V\text{)}\\
G_P &= G_O V^T,\\
G_{S,r} &= \big(\mathrm{diag}(P_r) - P_r P_r^T\big)\, G_{P,r}\quad &&\text{(row }r\text{; softmax Jacobian)}\\
G_Q &= G_S K/\sqrt{d_k},\quad G_K = G_S^T Q/\sqrt{d_k}.
\end{aligned}
$$

**LayerNorm Backward:**
```
# Forward: y = γ ⊙ (x - μ)/σ + β  
# Backward:
∂L/∂x = (∂L/∂y ⊙ γ - mean(∂L/∂y ⊙ γ) - (x-μ) ⊙ mean(∂L/∂y ⊙ γ ⊙ (x-μ))/σ²) / σ
∂L/∂γ = sum(∂L/∂y ⊙ (x-μ)/σ, dim=0)
∂L/∂β = sum(∂L/∂y, dim=0)
```

**📖 Mathematical Details:** See [LayerNorm Mathematics](./transformer_math.md#53-normalization-techniques) in transformer_math.md for intuitive explanation of normalization

---

## 10. Weight Updates and Optimization

### Adam Optimizer Mathematics

**Adam maintains moving averages of gradients and squared gradients:**

```python
# Hyperparameters
β₁ = 0.9        # momentum decay
β₂ = 0.999      # RMSprop decay  
ε = 1e-8        # numerical stability
α = 1e-4        # learning rate

# For each parameter θ with gradient g:
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # momentum
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       # RMSprop

# Bias correction
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Parameter update
θ_{t+1} = θ_t - α × m̂_t / (√v̂_t + ε)
```

**📖 Mathematical Details:** See [Adam Optimizer](./transformer_math.md#41-from-sgd-to-adam) in transformer_math.md for intuitive explanations

### Learning Rate Scheduling

**Warmup + Cosine Decay:**
```python
def learning_rate_schedule(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return max_lr * 0.5 * (1 + cos(π * progress))
```

**📖 Mathematical Details:** See [Learning Rate Schedules](./transformer_math.md#42-learning-rate-schedules) in transformer_math.md for detailed explanations

### Gradient Clipping

```python
# Global gradient norm clipping
total_norm = sqrt(sum(||grad_i||² for all parameters))
clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
for param in parameters:
    param.grad *= clip_coef
```

**📖 Mathematical Details:** See [Gradient Clipping](./transformer_math.md#43-gradient-clipping) in transformer_math.md for intuitive explanations

### Parameter Update Flow

```
Computed Gradients: {∂L/∂θ_i} for all parameters
                   ↓
┌─────────────────────────────────────┐
│        Gradient Clipping            │
│   Clip by global norm if needed     │
└─────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────┐
│         Adam Optimizer              │
│   Update m_t, v_t for each param    │  
│   Compute bias-corrected estimates  │
│   Apply parameter updates           │
└─────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────┐
│      Update Model Parameters        │
│   θ_new = θ_old - lr * update      │
└─────────────────────────────────────┘
                   ↓
Updated model ready for next forward pass
```

---

## 11. Complete Mathematical Summary

### Forward Pass Equations

**Input Processing:**
```
X₀ = TokenEmbedding(tokens) + PositionalEmbedding(positions)
```

**Transformer Layer (l = 1, ..., N):**
```
# Attention sub-layer
X̃ₗ = LayerNorm(Xₗ₋₁)
Aₗ = MultiHeadAttention(X̃ₗ, X̃ₗ, X̃ₗ)
X'ₗ = Xₗ₋₁ + Aₗ

# FFN sub-layer  
X̃'ₗ = LayerNorm(X'ₗ)
Fₗ = FFN(X̃'ₗ) = W₂ₗ × GELU(W₁ₗ × X̃'ₗ + b₁ₗ) + b₂ₗ
Xₗ = X'ₗ + Fₗ
```

**Output Generation:**
```
logits = X_N[-1, :] @ W_lm
probs = softmax(logits / temperature)
next_token = sample(probs)
```

### Training Equations

**Loss Function:**
```
L = -1/T × Σₜ log P(tₜ₊₁ | t₁, ..., tₜ)
where P(tₜ₊₁ | context) = softmax(f(t₁, ..., tₜ))[tₜ₊₁]
```

**Parameter Updates:**
```
For each parameter θ with gradient g:

# Adam optimizer
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_{t+1} = θ_t - α × m̂_t/(√v̂_t + ε)

where m̂_t = m_t/(1-β₁ᵗ), v̂_t = v_t/(1-β₂ᵗ)
```

### Key Computational Complexities

**Per Layer:**
- Attention: O(seq_len² × d_model + seq_len × d_model²)
- FFN: O(seq_len × d_model²)
- Total per layer: O(seq_len² × d_model + seq_len × d_model²)

**Full Model:**
- Forward pass: O(N × (seq_len² × d_model + seq_len × d_model²))
- Backward pass: Same as forward (roughly)
- Memory: O(N × seq_len × d_model) for activations + O(N × d_model²) for parameters

**With KV Cache (generation):**
- First token: O(N × d_model²) 
- Subsequent tokens: O(N × seq_len × d_model) per token

---

## Summary

This document traced the complete flow of information through a GPT-style transformer:

1. **Text → Tokens**: Subword tokenization maps text to integer sequences
2. **Tokens → Embeddings**: Lookup tables convert tokens to dense vectors  
3. **Positional Encoding**: Adds position information to embeddings
4. **Transformer Stack**: N layers of attention + FFN with residual connections
5. **Self-Attention**: Computes contextualized representations via query-key-value
6. **KV Caching**: Optimizes generation by caching previous computations
7. **Feed-Forward**: Position-wise transformations after attention
8. **Output Generation**: Language model head + sampling for next token prediction
9. **Training**: Backpropagation computes gradients for all parameters
10. **Optimization**: Adam optimizer updates parameters using computed gradients

Each stage involves specific mathematical transformations, with careful attention to tensor shapes and computational complexity. The architecture's key innovations—attention mechanisms, residual connections, and layer normalization—enable effective learning and generation across long sequences.
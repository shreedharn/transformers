# Transformer Fundamentals: Architecture and Core Concepts

Building on the RNN journey: In the [RNN Tutorial](./rnn_intro.md), you learned how neural networks gained memory and why this revolutionized sequence processing. You also discovered RNN's fundamental limitations—vanishing gradients, sequential bottlenecks, and the inability to process long sequences efficiently. The Transformer architecture solved all these problems while preserving RNN's core insights about sequential understanding.

What you'll learn in this foundational guide: How the "Attention Is All You Need" breakthrough created the architecture powering ChatGPT, GPT-4, and modern AI. We'll cover the complete technical flow from input text to output generation, with mathematical rigor and implementation details for each component.

Part of a two-part series: This guide covers the foundational transformer architecture and core concepts (sections 1-12). For advanced topics including training, optimization, fine-tuning, and deployment, see [Transformer Advanced Topics](./transformers_advanced.md).

Prerequisites: Completed [RNN Tutorial](./rnn_intro.md) and the mathematical foundations listed below.

## 1. Prerequisites

Mathematical Foundations:

- Linear Algebra: Matrix operations, eigenvalues, vector spaces (📚 See [Linear Algebra Essentials](./transformers_math1.md#221-vectors-as-word-meanings))
- Probability Theory: Distributions, information theory, maximum likelihood estimation (📚 See [Probability & Information Theory](./transformers_math1.md#224-softmax-and-cross-entropy-from-scores-to-decisions))
- Calculus: Gradients, chain rule, optimization theory (📚 See [Matrix Calculus Essentials](./transformers_math1.md#223-gradients-as-learning-signals))
- Machine Learning: Backpropagation, gradient descent, regularization techniques

## 2. Overview

Transformer architectures represent a fundamental paradigm shift in sequence modeling, replacing recurrent and convolutional approaches with attention-based mechanisms for parallel processing of sequential data. The architecture's core innovation lies in the self-attention mechanism, which enables direct modeling of dependencies between any two positions in a sequence, regardless of their distance.

Architectural Significance: Transformers solve the fundamental bottleneck of sequential processing inherent in RNNs while capturing long-range dependencies more effectively than CNNs. The attention mechanism provides explicit, learnable routing of information between sequence positions, enabling the model to dynamically focus on relevant context.

Key Innovations:

- Parallelizable attention computation: Unlike RNNs, all positions can be processed simultaneously
- Direct dependency modeling: Attention weights explicitly model relationships between any pair of sequence positions
- Position-invariant processing: The base attention mechanism is permutation-equivariant, requiring explicit positional encoding

## 3. Historical Context: The Transformer Breakthrough

The transformer architecture represents the culmination of decades of research in sequence modeling. While earlier architectures like MLPs struggled with variable-length inputs, RNNs suffered from vanishing gradients, and LSTMs remained sequential bottlenecks, the transformer solved all these problems with a single revolutionary insight.

### "Attention Is All You Need"

The Revolutionary Question: Vaswani et al. (2017) asked a simple but profound question: *What if we remove recurrence entirely and rely purely on attention?*

Key Insight: If attention can help RNNs access any part of the input, why not use attention as the primary mechanism for processing sequences, rather than just an auxiliary tool?

Previous Evolution:

- MLPs: Established neural foundations but couldn't handle variable-length sequences
- RNNs: Introduced sequential processing but suffered from vanishing gradient problems
- LSTMs/GRUs: Solved vanishing gradients through gating mechanisms but remained sequential
- Seq2Seq + Attention: Eliminated information bottlenecks but still relied on recurrence

### Removing Sequential Processing

RNN Limitation: Even with attention, RNNs must process sequences step-by-step:
```
h₁ → h₂ → h₃ → h₄ → h₅  (sequential dependency)
```

Transformer Innovation: Process all positions simultaneously using self-attention:
```
All positions computed in parallel using attention
```

### Self-Attention: The Core Mechanism

Self-Attention Concept: Instead of attending from decoder to encoder, have each position in a sequence attend to all positions in the same sequence (including itself). This is like a "reading comprehension" task where, for each word, the model asks: "Which other words in this sentence are most relevant to understanding this specific word?"

To do this, the model creates three special versions of each word's embedding:

- **Query (Q):** A representation of the current word, used to "ask" for relevant information. Think of it as a question: "What parts of the sentence are important for me?"
- **Key (K):** A representation of a word used for being "looked up" by other words. It acts like a label or a signpost, saying "Here is the information I contain."
- **Value (V):** The actual information or "meaning" of a word. Once a word is identified as important (via its Key), the Value is what gets passed along.

This Q, K, V system is inspired by information retrieval systems. Imagine searching for a video online: your search term is the **Query**, the video titles are the **Keys**, and the video content itself is the **Value**. The search engine matches your Query to the most relevant Keys to show you the best Values.

Mathematical Foundation:
Given input sequence:

$$
\begin{aligned}
X &= [x\_1, x\_2, \ldots, x\_n] \newline
\text{Step 1: Create Q, K, V} \quad Q &= XW^Q, \quad K = XW^K, \quad V = XW^V \newline
\text{Step 2: Calculate Attention} \quad \text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$

Key Properties:

- Parallel Computation: All positions processed simultaneously
- Full Connectivity: Every position can attend to every other position
- No Recurrence: No sequential dependencies in computation

### Multi-Head Attention

Motivation: Different attention heads can capture different types of relationships (syntactic, semantic, positional, etc.).

Implementation:

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}\_1, \ldots, \text{head}\_h)W^O \newline
\text{head}\_i &= \text{Attention}(QW\_i^Q, KW\_i^K, VW\_i^V)
\end{aligned}
$$

### Position Encoding Solution

Problem: Attention is permutation-equivariant—"cat sat on mat" and "mat on sat cat" would be processed identically.

Solution: Add positional information to input embeddings:

$$
\begin{aligned}
\text{input} = \text{token\_embedding} + \text{positional\_encoding}
\end{aligned}
$$

Each token's embedding is combined with a positional encoding vector, ensuring the model can distinguish between different positions in the sequence.

### Complete Transformer Architecture

Encoder Block (Pre-LayerNorm):

1. LayerNorm → Multi-head self-attention → Add residual
2. LayerNorm → Feed-forward network → Add residual

Decoder Block (Pre-LayerNorm):

1. LayerNorm → Masked multi-head self-attention → Add residual
2. LayerNorm → Multi-head cross-attention → Add residual
3. LayerNorm → Feed-forward network → Add residual

### Why Transformers Work So Well

1. Parallelization: All sequence positions processed simultaneously, enabling efficient GPU utilization
2. Long-Range Dependencies: Direct connections between any two positions eliminate information bottlenecks
3. Computational Efficiency: Can leverage modern parallel hardware effectively
4. Modeling Flexibility: Minimal inductive biases allow learning patterns from data
5. Transfer Learning: Pre-trained transformers transfer exceptionally well to new tasks
6. Direct Information Flow: No information bottlenecks—every position can directly access information from every other position

The sections that follow will dive deep into the technical implementation of these concepts, showing exactly how transformers process text from input to output generation.

## 4. Pipeline

Let's trace through what happens when you type "The cat sat on" and the AI predicts the next word.

Think of this process like a sophisticated translation system - but instead of translating between languages, we're translating from "human text" to "AI understanding" and back to "human text".

### Computational Pipeline Overview

Core Processing Stages:

1. Tokenization: Subword segmentation and vocabulary mapping
2. Embedding: Dense vector representation learning
3. Position Encoding: Sequence order information injection
4. Transformer Layers: Attention-based representation refinement
5. Output Projection: Vocabulary distribution computation
6. Decoding: Next-token selection strategies

### Detailed Process Flow

```
User Input: "The cat sat on"
       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                                 │
│              (How AI Understands Text)                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Tokenization:     "The cat sat on" → [464, 2415, 3332, 319]  │
│    (Break into computer-friendly pieces)                        │
│                                                                 │
│ 2. Embedding:        tokens → vectors [4, 768]                  │
│    (Convert numbers to meaning representations)                 │
│                                                                 │
│ 3. Position Encoding: add positional info [4, 768]              │
│    (Tell the model WHERE each word appears)                     │
│                                                                 │
│ 4. Transformer Stack: 12 layers of attention + processing       │
│    (Deep understanding - like reading comprehension)            │
│                                                                 │
│ 5. Output Projection: → probabilities for all 50,000 words.     │
│    (Consider every possible next word)                          │
│                                                                 │
│ 6. Sampling:         choose from top candidates                 │
│    (Make the final decision)                                    │
└─────────────────────────────────────────────────────────────────┘
       ↓
Output: "the" (most likely next word)
```

Tensor Dimensions and Semantic Interpretation:

- [464, 2415, 3332, 319]: Discrete token indices mapping to vocabulary entries
Tensor Dimensions:

$$
\begin{aligned}
\text{Shape [4, 768]:} \quad &\text{Sequence length × model dimension tensor} \newline
&\text{representing learned embeddings}
\end{aligned}
$$
- 12 layers: Hierarchical representation learning through stacked transformer blocks
- 50,000 words: Vocabulary size determining output distribution dimensionality

### Training: How Models Learn (Optional Detail)

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING: BACKWARD PASS                   │
│                (How AI Learns from Mistakes)                │
├─────────────────────────────────────────────────────────────┤
│ 1. Compute Loss:     Compare prediction with correct answer │
│    (Like grading a test - how wrong was the guess?)         │
│                                                             │
│ 2. Backpropagation: Find what caused the mistake            │
│    (Trace back through all the steps to find errors)        │
│                                                             │
│ 3. Weight Updates:   Adjust internal parameters             │
│    (Fine-tune the model to do better next time)             │
└─────────────────────────────────────────────────────────────┘
```

### Model Hyperparameters and Complexity

Architectural Dimensions:

- Vocabulary size (V): Discrete token space cardinality, typically 32K-100K
- Model dimension (d_model): Hidden state dimensionality determining representational capacity
- Context length (n): Maximum sequence length for attention computation
- Layer count (N): Depth of hierarchical representation learning
- Attention heads (H): Parallel attention subspaces for diverse relationship modeling

GPT-2 Specification:

- Vocabulary: 50,257 BPE tokens
- Hidden dimension: 768
- Context window: 1,024 tokens
- Transformer blocks: 12 layers
- Multi-head attention: 12 heads per layer

---

## 5. Stage 1: Text to Tokens

### 🎯 Intuition: Breaking Text into Computer-Friendly Pieces

Think of tokenization like chopping vegetables for a recipe. You can't feed raw text directly to a computer - you need to break it into standardized pieces (tokens) that the computer can work with, just like chopping vegetables into uniform sizes for cooking.

Why not just use individual letters or whole words?

- Letters: Too many combinations, loses meaning ("c-a-t" tells us less than "cat")
- Whole words: Millions of possible words, can't handle new/misspelled words
- Subwords (what we use): Perfect balance - captures meaning while handling new words

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

Step 1: Smart Splitting
```
"The cat sat on" → ["The", " cat", " sat", " on"]
```
Notice: Spaces are included with words (except the first)

Step 2: Look Up Numbers
```
"The"   → 464   (most common word, low number)
" cat"  → 2415  (common word)
" sat"  → 3332  (less common)
" on"   → 319   (very common, low number)
```

Final Result:
```
Input: "The cat sat on"      (18 characters)
Output: [464, 2415, 3332, 319]  (4 tokens)
Shape: [4] (sequence length = 4)
```

### Mathematical Formulation

Tokenization Mapping:

$$
\begin{aligned}
T: \Sigma^* &\to \{0, 1, \ldots, V-1\}^* \newline
\Sigma^* &: \text{Space of all possible text strings} \newline
V &: \text{Vocabulary size} \newline
\text{Output} &: \text{Variable-length sequence of discrete token indices}
\end{aligned}
$$

BPE Algorithm: Given text and learned merge operations:

$$
\begin{aligned}
\text{tokenize}(s) &= [\text{vocab}[\tau\_i] \mid \tau\_i \in \mathrm{BPE\_segment}(s, M)] \newline
s &: \text{Input text} \newline
M &: \text{Learned merge operations} \newline
\mathrm{BPE\_segment} &: \text{Applies learned merge rules} \newline
\tau\_i &: \text{Subword tokens from segmentation}
\end{aligned}
$$

📖 Detailed Algorithm: See [Tokenization Mathematics](./transformers_math1.md#82-embedding-mathematics) for BPE training and inference procedures.

### Tokenization Challenges and Considerations

Out-of-Vocabulary Handling:

- Unknown words: Long or rare words decompose into character-level tokens, increasing sequence length
- Domain mismatch: Models trained on general text may poorly tokenize specialized domains (code, scientific notation)
- Multilingual complexity: Subword boundaries vary across languages, affecting cross-lingual transfer

Performance Implications:

- Vocabulary size vs. sequence length trade-off: Larger vocabularies reduce sequence length but increase computational cost
- Compression efficiency: Good tokenization minimizes information loss while maximizing compression

### Implementation Considerations:

- Subword algorithms: BPE, WordPiece, SentencePiece each have different inductive biases
- Special tokens: Sequence delimiters (`<|endoftext|>`, `<|pad|>`, `<|unk|>`) require careful handling
- Context limitations: Finite attention window constrains sequence length (typically 512-8192 tokens)
- Batching requirements: Variable-length sequences necessitate padding strategies

---

## 6. Stage 2: Tokens to Embeddings

### Dense Vector Representation Learning

Embedding layers transform discrete token indices into continuous vector representations within a learned semantic space. This mapping enables the model to capture distributional semantics and enables gradient-based optimization over the discrete vocabulary.

Representational Advantages:

- Continuous optimization: Dense vectors enable gradient-based learning over discrete token spaces
- Semantic similarity: Geometrically related vectors capture semantic relationships through distance metrics
- Compositional structure: Linear operations in embedding space can capture meaningful transformations

Position encoding addresses the permutation-invariance of attention mechanisms by injecting sequential order information into the representation space.

### Embedding Computation Pipeline

Token Embedding Lookup:

$$
\begin{aligned}
\text{Step 1: Matrix indexing} \quad E &\in \mathbb{R}^{V \times d\_{\text{model}}} : \text{Token embedding matrix} \newline
\text{Step 2: Batch lookup} \quad X\_{\text{tok}} &= E[T] : \text{Lookup embeddings for tokens T} \newline
\text{Step 3: Dense representation} \quad X\_{\text{tok}} &\in \mathbb{R}^{n \times d\_{\text{model}}} : \text{Dense token representations} \newline
\text{Step 4: Position encoding} \quad PE &\in \mathbb{R}^{n\_{\max} \times d\_{\text{model}}} : \text{Position encoding matrix} \newline
\text{Step 5: Sequence slicing} \quad X\_{\text{pos}} &= PE[0:n] : \text{Position encodings for sequence length n} \newline
\text{Step 6: Element-wise addition} \quad X &= X\_{\text{tok}} + X\_{\text{pos}} : \text{Combined representation}
\end{aligned}
$$

Output: Combined semantic and positional representation

$$
\begin{aligned}
X \in \mathbb{R}^{n \times d\_{\text{model}}}
\end{aligned}
$$

### Mathematical Formulation

Token Embedding Transformation:

$$
\begin{aligned}
X\_{\text{tok}} &= E[T] \text{ where } E \in \mathbb{R}^{V \times d\_{\text{model}}} \newline
X\_{\text{tok}}[i] &= E[t\_i] \in \mathbb{R}^{d\_{\text{model}}} \text{ for token index } t\_i
\end{aligned}
$$

Position Encoding Variants:

Learned Positional Embeddings:

$$
\begin{aligned}
X\_{\text{pos}}[i] = PE[i] \text{ where } PE \in \mathbb{R}^{n\_{\max} \times d\_{\text{model}}}
\end{aligned}
$$

Sinusoidal Position Encoding:

$$
\begin{aligned}
PE[\text{pos}, 2i] &= \sin\left(\frac{\text{pos}}{10000^{2i/d\_{\text{model}}}}\right) \newline
PE[\text{pos}, 2i+1] &= \cos\left(\frac{\text{pos}}{10000^{2i/d\_{\text{model}}}}\right)
\end{aligned}
$$

**Why this formula?** This clever design uses sine and cosine functions at different frequencies for each dimension of the embedding. This has a crucial property: for any fixed offset `k`, the positional encoding at `pos+k` can be represented as a linear transformation of the positional encoding at `pos`. This means the model can easily learn to understand relative positions, which is vital for language. It also has the advantage of being able to generalize to sequence lengths longer than the ones seen during training.

Combined Representation:

$$
\begin{aligned}
X &= X\_{\text{tok}} + X\_{\text{pos}} \in \mathbb{R}^{n \times d\_{\text{model}}} \newline
n &: \text{Sequence length} \newline
d\_{\text{model}} &: \text{Model dimension}
\end{aligned}
$$

📖 Theoretical Foundation: See [Embedding Mathematics](./transformers_math1.md#82-embedding-mathematics) and [Positional Encodings](./transformers_math1.md#62-advanced-positional-encodings) for detailed analysis of learned vs. fixed position encodings.

### Concrete Example with Actual Numbers

Let's see what happens with our tokens [464, 2415, 3332, 319] ("The cat sat on"):

Step 1: Token Embedding Lookup
```
Token 464 ("The") → [0.1, -0.3, 0.7, 0.2, ...] (768 numbers total)
Token 2415 ("cat") → [-0.2, 0.8, -0.1, 0.5, ...]
Token 3332 ("sat") → [0.4, 0.1, -0.6, 0.3, ...]
Token 319 ("on") → [0.2, -0.4, 0.3, -0.1, ...]
Result: [4, 768] array (4 words × 768 features each)
```

Step 2: Position Embeddings
```
Position 0 → [0.01, 0.02, -0.01, 0.03, ...] (for "The")
Position 1 → [0.02, -0.01, 0.02, -0.01, ...] (for "cat")
Position 2 → [-0.01, 0.03, 0.01, 0.02, ...] (for "sat")
Position 3 → [0.03, 0.01, -0.02, 0.01, ...] (for "on")
Result: [4, 768] array (position info for each word)
```

Step 3: Add Together
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

🎯 Shape Intuition: Think of shapes like `[4, 768]` as a table with 4 rows (tokens) and 768 columns (features). Each row represents one word, each column represents one aspect of meaning.

### 🛠️ What Could Go Wrong?

Common Misconceptions:

- "Why 768 dimensions?" It's like having 768 different personality traits - enough to capture complex meaning relationships
- "Why add position info?" Without it, "cat sat on mat" and "mat on sat cat" would look identical to the model
- "What if we run out of positions?" Models have maximum sequence lengths (like 2048) - longer texts get truncated

---

## 7. Stage 3: Through the Transformer Stack

### Hierarchical Representation Learning

The transformer stack implements hierarchical feature extraction through stacked self-attention and feed-forward blocks. Each layer refines the input representations by modeling increasingly complex relationships and patterns within the sequence.

Layer-wise Functionality:

1. Self-attention sublayer: Models pairwise interactions between sequence positions
2. Feed-forward sublayer: Applies position-wise transformations to individual token representations
3. Residual connections: Preserve gradient flow and enable identity mappings
4. Layer normalization: Stabilizes training dynamics and accelerates convergence

Representational Hierarchy: Empirical analysis suggests progressive abstraction:

- Early layers: Syntactic patterns, local dependencies, surface-level features
- Middle layers: Semantic relationships, discourse structure, compositional meaning
- Late layers: Task-specific abstractions, high-level reasoning patterns

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

Transformer Block Architecture (Pre-LayerNorm):

*Note: The architecture described here uses "Pre-LayerNorm," where normalization is applied before the attention and feed-forward sublayers. This is a common and effective variant found in many modern transformers (like GPT-2, BERT, and LLaMA). The original "Attention Is All You Need" paper used "Post-LayerNorm," where normalization is applied after the residual connection. Pre-LayerNorm generally leads to more stable training.*

Given layer input:

$$
\begin{aligned}
X^{(l-1)} \in \mathbb{R}^{n \times d\_{\text{model}}}
\end{aligned}
$$

Self-Attention Sublayer:

$$
\begin{aligned}
\tilde{X}^{(l-1)} &= \text{LayerNorm}(X^{(l-1)}) \newline
A^{(l)} &= \text{MultiHeadAttention}(\tilde{X}^{(l-1)}, \tilde{X}^{(l-1)}, \tilde{X}^{(l-1)}) \newline
X'^{(l)} &= X^{(l-1)} + A^{(l)} \quad \text{(residual connection)}
\end{aligned}
$$

Feed-Forward Sublayer:

$$
\begin{aligned}
\tilde{X}'^{(l)} &= \text{LayerNorm}(X'^{(l)}) \newline
F^{(l)} &= \text{FFN}(\tilde{X}'^{(l)}) \newline
X^{(l)} &= X'^{(l)} + F^{(l)} \quad \text{(residual connection)}
\end{aligned}
$$

Design Rationale:

- Pre-normalization: Stabilizes gradients and enables deeper networks compared to post-norm
- Residual connections: Address vanishing gradient problem via identity shortcuts
- Two-sublayer structure: Separates relationship modeling (attention) from feature transformation (FFN)

📖 Theoretical Analysis: See [Transformer Block Mathematics](./transformers_math1.md#71-complete-block-equations) and [Residual Connections as Discretized Dynamics](./transformers_math1.md#213-residual-connections-as-discretized-dynamics) for detailed mathematical foundations.

---

## 8. Architectural Variants: Encoder, Decoder, and Encoder-Decoder

### Understanding Different Transformer Architectures

While the core transformer architecture provides a flexible foundation, different variants optimize for specific use cases through architectural modifications. The choice between encoder-only, decoder-only, and encoder-decoder architectures fundamentally affects the model's capabilities and training dynamics.

### Encoder-Only: BERT Family

Structure: Bidirectional self-attention across all positions
Training: Masked Language Modeling (MLM) - predict randomly masked tokens
Use cases: Classification, entity recognition, semantic similarity

Mathematical Formulation:

$$
\begin{aligned}
\text{Input} &: [x\_1, [\text{MASK}], x\_3, \ldots, x\_n] \newline
\text{Objective} &: \mathcal{L} = -\sum\_{i \in \text{masked}} \log P(x\_i | x\_{\setminus i}) \newline
\text{Attention} &: \text{Full bidirectional attention matrix (no causal masking)}
\end{aligned}
$$

Key advantage: Full bidirectional context enables deep understanding
Limitation: Cannot generate sequences autoregressively

Example Implementation Pattern:
```python
# BERT-style encoder block
def encoder_attention(x, mask=None):
    # No causal masking - full bidirectional attention
    Q, K, V = project_qkv(x)
    scores = Q @ K.T / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = softmax(scores, dim=-1)
    return weights @ V
```

### Decoder-Only: GPT Family

Structure: Causal self-attention - each position only attends to previous positions
Training: Causal Language Modeling (CLM) - predict next token
Use cases: Text generation, completion, few-shot learning

Mathematical Formulation:

$$
\begin{aligned}
\text{Input} &: [x\_1, x\_2, \ldots, x\_t] \newline
\text{Objective} &: \mathcal{L} = -\sum\_{t=1}^{n-1} \log P(x\_{t+1} | x\_1, \ldots, x\_t) \newline
\text{Attention mask} &: A\_{ij} = 0 \text{ for } j > i \text{ (causal mask)}
\end{aligned}
$$

Key advantage: Natural autoregressive generation capability
Trade-off: No future context during training

Causal Masking Implementation:
```python
def causal_mask(seq_len):
    # Create lower triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.masked_fill(mask == 0, -float('inf'))
```

### Encoder-Decoder: T5 Family

Structure: Encoder (bidirectional) + Decoder (causal) with cross-attention

Training: Various objectives (span corruption, translation, etc.)

Use cases: Translation, summarization, structured tasks

Mathematical Components:

1. Encoder self-attention: Bidirectional processing of input sequence
2. Decoder self-attention: Causal attention over generated tokens
3. Cross-attention: Decoder attends to encoder outputs

Cross-Attention Formulation:

$$
\begin{aligned}
\text{CrossAttention}(Q\_{\text{dec}}, K\_{\text{enc}}, V\_{\text{enc}}) = \text{softmax}\left(\frac{Q\_{\text{dec}}K\_{\text{enc}}^T}{\sqrt{d\_k}}\right)V\_{\text{enc}}
\end{aligned}
$$

Key advantage: Combines bidirectional understanding with generation
Cost: Higher parameter count and computational complexity

### Modern Architectural Innovations

Multi-Query Attention (MQA): Single key/value head shared across queries, reduces KV cache size for generation:

$$
\begin{aligned}
\text{Standard MHA:} \quad K, V &\in \mathbb{R}^{n \times H \times d\_k} \newline
\text{MQA variant:} \quad K, V &\in \mathbb{R}^{n \times d\_k} \text{ (shared across queries)}
\end{aligned}
$$

Grouped-Query Attention (GQA): Compromise between MHA and MQA, where groups of queries share K/V heads, balancing quality and efficiency.

Mixture of Experts (MoE): Replace some FFNs with expert networks using sparse activation based on learned routing, increasing capacity without proportional compute increase.

Position Encoding Variants:

- RoPE (Rotary Position Embedding): Rotates query/key vectors by position-dependent angles
- ALiBi (Attention with Linear Biases): Adds position-based linear bias to attention scores
- Learned vs. Sinusoidal: Trainable position vectors vs. fixed mathematical functions

### Choosing the Right Architecture

For Understanding Tasks: Use encoder-only (BERT-style)

- Classification, named entity recognition, semantic similarity
- Full bidirectional context is crucial

For Generation Tasks: Use decoder-only (GPT-style)

- Text completion, creative writing, code generation
- Autoregressive capability is primary requirement

For Sequence-to-Sequence Tasks: Use encoder-decoder (T5-style)

- Translation, summarization, question answering
- Clear input/output distinction with different processing needs

Performance Considerations:

- Parameter efficiency: Decoder-only reuses weights for both understanding and generation
- Training efficiency: Encoder-only can process full sequences in parallel
- Inference patterns: Consider whether bidirectional context or autoregressive generation is needed

---

## 9. Stage 4: Self-Attention Deep Dive

### Attention as Differentiable Key-Value Retrieval

Self-attention implements a differentiable associative memory mechanism where each sequence position (query) retrieves information from all positions (keys) based on learned compatibility functions. This enables modeling arbitrary dependencies without the sequential constraints of RNNs or the locality constraints of CNNs.

Attention Mechanism Properties:

- Permutation equivariance: Output permutes consistently with input permutation (before positional encoding)
- Dynamic routing: Information flow adapts based on input content rather than fixed connectivity
- Parallel computation: All pairwise interactions computed simultaneously
- Global receptive field: Each position can directly attend to any other position

Self-attention vs. Cross-attention: In self-attention, queries, keys, and values all derive from the same input sequence, enabling the model to relate different positions within a single sequence.

### Attention Weight Interpretation

Attention Weights as Soft Alignment:

Given query position and key positions, attention weights represent the proportion of information flow between sequence positions:

$$
\begin{aligned}
\text{Query position:} \quad &i \newline
\text{Key positions:} \quad &\{j\} \newline
\text{Attention weights:} \quad &\alpha\_{ij} \text{ (proportion of position } j \text{'s information incorporated into position } i \text{)}
\end{aligned}
$$

Example: Coreference Resolution
For sequence "The cat sat on it" processing token "it":

$$
\begin{aligned}
\alpha\_{\text{it},\text{The}} &= 0.05 \text{ (minimal syntactic relevance)} \newline
\alpha\_{\text{it},\text{cat}} &= 0.15 \text{ (potential antecedent)} \newline
\alpha\_{\text{it},\text{sat}} &= 0.10 \text{ (predicate information)} \newline
\alpha\_{\text{it},\text{on}} &= 0.15 \text{ (spatial relationship)} \newline
\alpha\_{\text{it},\text{it}} &= 0.55 \text{ (self-attention for local processing)}
\end{aligned}
$$

Constraint: Probability simplex constraint from softmax normalization:

$$
\begin{aligned}
\sum\_j \alpha\_{ij} = 1
\end{aligned}
$$

### Attention Computation Steps

Step 1: Linear Projections

$$
\begin{aligned}
\text{Input:} \quad X &\in \mathbb{R}^{n \times d\_{\text{model}}} : \text{Input sequence representations} \newline
\text{Query projections:} \quad Q &= XW^Q \in \mathbb{R}^{n \times d\_{\text{model}}} \newline
\text{Key projections:} \quad K &= XW^K \in \mathbb{R}^{n \times d\_{\text{model}}} \newline
\text{Value projections:} \quad V &= XW^V \in \mathbb{R}^{n \times d\_{\text{model}}}
\end{aligned}
$$

Step 2: Multi-Head Reshaping

$$
\begin{aligned}
\text{Number of heads:} \quad H &: \text{Attention heads for parallel processing} \newline
\text{Head dimension:} \quad d\_k &= d\_{\text{model}}/H \newline
\text{Reshaped tensors:} \quad Q, K, V &\in \mathbb{R}^{H \times n \times d\_k} \text{ (batch dimension omitted)}
\end{aligned}
$$

Step 3: Scaled Dot-Product Attention

$$
\begin{aligned}
\text{Compatibility scores:} \quad S &= \frac{QK^T}{\sqrt{d\_k}} \in \mathbb{R}^{H \times n \times n} \newline
\text{Scaling rationale:} \quad &\text{Prevents softmax saturation as dimensionality increases} \newline
\text{Complexity:} \quad &O(n^2 d\_k) \text{ per head, } O(n^2 d\_{\text{model}}) \text{ total}
\end{aligned}
$$

Step 4: Causal Masking (for autoregressive models)

$$
\begin{aligned}
\text{Mask application:} \quad S\_{\text{masked}} &= S + M \newline
\text{Future position mask:} \quad M\_{ij} &= -\infty \text{ for } j > i \newline
\text{Effect:} \quad &\text{Ensures causal ordering in autoregressive generation}
\end{aligned}
$$

Step 5: Attention Weight Computation

$$
\begin{aligned}
\text{Normalization:} \quad A &= \text{softmax}(S\_{\text{masked}}, \text{dim}=-1) \in \mathbb{R}^{H \times n \times n} \newline
\text{Interpretation:} \quad A\_{ijh} &: \text{Position } j \text{ contribution to position } i \text{ in head } h \newline
\text{Properties:} \quad &\text{Each row sums to 1, forming probability distributions}
\end{aligned}
$$

Step 6: Value Aggregation

$$
\begin{aligned}
\text{Weighted sum:} \quad O &= AV \in \mathbb{R}^{H \times n \times d\_k} \newline
\text{Information flow:} \quad &\text{Each position receives weighted information from all positions}
\end{aligned}
$$

Step 7: Head Concatenation and Output Projection

$$
\begin{aligned}
\text{Concatenation:} \quad O\_{\text{concat}} &: \text{Reshape } O \text{ to } \mathbb{R}^{n \times d\_{\text{model}}} \newline
\text{Final projection:} \quad \text{Output} &= O\_{\text{concat}}W^O \in \mathbb{R}^{n \times d\_{\text{model}}} \newline
\text{Purpose:} \quad &\text{Integrate information from all attention heads}
\end{aligned}
$$

### Multi-Head Attention Mathematical Formulation

Core Attention Mechanism (Scaled Dot-Product):

$$
\begin{aligned}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$
\begin{aligned}
Q &\in \mathbb{R}^{n \times d\_k} : \text{Query matrix (what information to retrieve)} \newline
K &\in \mathbb{R}^{n \times d\_k} : \text{Key matrix (what information is available)} \newline
V &\in \mathbb{R}^{n \times d\_v} : \text{Value matrix (actual information content)} \newline
d\_k &= d\_v = d\_{\text{model}}/H : \text{Head dimension} \newline
\sqrt{d\_k} &: \text{Temperature scaling to prevent saturation}
\end{aligned}
$$

Multi-Head Attention Implementation:

$$
\begin{aligned}
\text{MultiHead}(X) &= \text{Concat}(\text{head}\_1, \ldots, \text{head}\_H)W^O \newline
\text{head}\_i &= \text{Attention}(XW\_i^Q, XW\_i^K, XW\_i^V)
\end{aligned}
$$

Linear Projection Matrices (not MLPs):

$$
\begin{aligned}
W\_i^Q, W\_i^K, W\_i^V &\in \mathbb{R}^{d\_{\text{model}} \times d\_k} : \text{Per-head projections} \newline
W^O &\in \mathbb{R}^{d\_{\text{model}} \times d\_{\text{model}}} : \text{Output projection}
\end{aligned}
$$

Implementation Details:

1. Parallel computation: All heads computed simultaneously via reshaped tensor operations
2. Linear projections: Simple matrix multiplications, not multi-layer perceptrons
3. Concatenation: Head outputs concatenated along feature dimension
4. Output projection: Single linear transformation of concatenated heads

📖 Derivation and Analysis: See [Multi-Head Attention Theory](./transformers\_\1ath1.md#61-multi-head-as-subspace-projections) and [Scaling Analysis](./transformers\_\1ath1.md#52-why-the-sqrtd\_\1-scaling) for mathematical foundations.

Causal Masking for Autoregressive Models:

$$
\begin{aligned}
\text{mask}[i, j] = \begin{cases}
0 & \text{if } j \leq i \newline
-\infty & \text{if } j > i
\end{cases}
\end{aligned}
$$

This lower-triangular mask ensures that position cannot attend to future positions, maintaining the autoregressive property necessary for language modeling.

Computational Implementation: Typically applied as additive bias before softmax:

$$
\begin{aligned}
\text{scores} &= \frac{QK^T}{\sqrt{d\_k}} + \text{mask} \newline
i &: \text{Current position} \newline
j &: \text{Attended position}
\end{aligned}
$$

Computational Complexity Analysis:

Tensor Shapes with Standard Convention:
A common convention is to use shapes like `[batch\_\1ize, num\_\1eads, seq\_\1en, head\_\1im]`. For simplicity, we omit the batch dimension.

$$



$$

Time Complexity:

- Linear projections: The creation of Q, K, and V involves matrix multiplications of shape `[n, d_model]` with `[d_model, d_model]`, resulting in a complexity of $O(n \cdot d_{\text{model}}^2)$.
- Attention computation: The multiplication of Q and K matrices of shape `[n, d_k]` has a complexity of $O(n^2 \cdot d_k)$ per head. For all H heads, this is $O(H \cdot n^2 \cdot d_k) = O(n^2 \cdot d_{\text{model}})$.
- Total per layer: For a typical transformer where `d_model` > `n`, the linear projections are often the most computationally expensive part. The combined complexity is $O(n \cdot d_{\text{model}}^2 + n^2 \cdot d_{\text{model}})$.

  $$
\begin{aligned}
\text{Linear projections (Q, K, V)} &: O(n \cdot d_{\text{model}}^2) \newline
  \text{Attention scores (QK^T)} &: O(n^2 \cdot d_{\text{model}}) \newline
  \text{Value aggregation (AV)} &: O(n^2 \cdot d_{\text{model}}) \newline
  \text{Total per layer} &: O(n \cdot d_{\text{model}}^2 + n^2 \cdot d_{\text{model}})
\end{aligned}
$$

Space Complexity:

- Attention matrices: Quadratic in sequence length per head
- Activations: Linear in sequence length and model dimension
- Quadratic scaling with sequence length motivates efficient attention variants

  $$
\begin{aligned}
\text{Attention matrices} &: O(H \cdot n^2) \newline
  \text{Activations} &: O(n \cdot d\_{\text{model}})
\end{aligned}
$$

---

## 10. Stage 5: KV Cache Operations

### Autoregressive Generation Optimization

KV caching addresses the computational inefficiency of autoregressive generation by storing and reusing previously computed key-value pairs. This optimization reduces the time complexity of each generation step from quadratic to linear in the context length.

Computational Motivation:
During autoregressive generation, attention computations for previous tokens remain unchanged when processing new tokens. Caching eliminates redundant recomputation of these static attention components.

Performance Impact:

- Without caching: Quadratic computation per step
- With caching: Linear compute per step with memory overhead
- Trade-off: Memory consumption for computational efficiency

  $$
\begin{aligned}
\text{Without caching} &: O(t^2) \text{ computation per step} \newline
  \text{With caching} &: O(t) \text{ compute per step} \newline
  \text{Memory overhead} &: O(L \cdot H \cdot t \cdot d\_k) \text{ for model} \newline
  t &: \text{Context length} \newline
  L &: \text{Number of layers} \newline
  H &: \text{Number of heads}
\end{aligned}
$$

### Computational Efficiency Analysis

Problem Formulation:
In standard autoregressive generation at step $$
t
$$, computing attention for the new token requires:

$$
\begin{aligned}
\text{K, V recomputation:} \quad &O(t \cdot d\_{\text{model}}^2) \newline
\text{Attention scores:} \quad &O(t^2 \cdot d\_{\text{model}}) \newline
\text{Total per step:} \quad &O(t^2 \cdot d\_{\text{model}}) \newline
t &: \text{Current step}
\end{aligned}
$$

KV Cache Solution:

Without Cache (Step t):

$$
\begin{aligned}
\text{Full sequence matrices:} \quad K, V &\in \mathbb{R}^{t \times d\_{\text{model}}} \newline
\text{Attention computation:} \quad &O(t^2 \cdot d\_{\text{model}}) \text{ (quadratic complexity)} \newline
\text{Memory requirement:} \quad &O(t \cdot d\_{\text{model}}) \text{ per step} \newline
t &: \text{Current step}
\end{aligned}
$$

Note: This assumes a full re-forward over the context per new token; frameworks differ, but the asymptotic bottleneck is the score matrix computation.

With Cache (Step t):

$$
\begin{aligned}
\text{New key-value pairs:} \quad k\_{\text{new}}, v\_{\text{new}} &\in \mathbb{R}^{1 \times d\_{\text{model}}} \newline
\text{Cached matrices:} \quad K\_{\text{cache}}, V\_{\text{cache}} &\in \mathbb{R}^{(t-1) \times d\_{\text{model}}} \newline
\text{Attention computation:} \quad &O(t \cdot d\_{\text{model}}) \text{ (linear complexity)} \newline
\text{Memory requirement:} \quad &O(t \cdot d\_{\text{model}}) \text{ accumulated over time}
\end{aligned}
$$

### KV Cache Implementation

Cache Architecture:
```python
class KVCache:
    def _\_\1nit__(self, num\_\1ayers, num\_\1eads, d\_\1ead, max\_\1eq\_\1en):
        # Per-layer cache storage
        self.cache = {
            layer\_\1dx: {
                'keys': torch.zeros(1, num\_\1eads, max\_\1eq\_\1en, d\_\1ead),
                'values': torch.zeros(1, num\_\1eads, max\_\1eq\_\1en, d\_\1ead),
                'current\_\1ength': 0
            }
            for layer\_\1dx in range(num\_\1ayers)
        }

    def update(self, layer\_\1dx, new\_\1eys, new\_\1alues):
        """Append new key-value pairs to cache"""
        cache\_\1ntry = self.cache[layer\_\1dx]
        curr\_\1en = cache\_\1ntry['current\_\1ength']

        # Insert new keys and values
        cache\_\1ntry['keys'][:, :, curr\_\1en:curr\_\1en+1] = new\_\1eys
        cache\_\1ntry['values'][:, :, curr\_\1en:curr\_\1en+1] = new\_\1alues
        cache\_\1ntry['current\_\1ength'] += 1

        return (
            cache\_\1ntry['keys'][:, :, :cache\_\1ntry['current\_\1ength']],
            cache\_\1ntry['values'][:, :, :cache\_\1ntry['current\_\1ength']]
        )
```

Mathematical Formulation:

Cache Update (Step):

$$
\begin{aligned}
q\_{\text{new}} &= x\_{\text{new}} W^Q \in \mathbb{R}^{1 \times d\_k} \newline
k\_{\text{new}} &= x\_{\text{new}} W^K \in \mathbb{R}^{1 \times d\_k} \newline
v\_{\text{new}} &= x\_{\text{new}} W^V \in \mathbb{R}^{1 \times d\_v}
\end{aligned}
$$

Cache Concatenation:

$$
\begin{aligned}
K\_{\text{full}} &= [K\_{\text{cache}}; k\_{\text{new}}] \in \mathbb{R}^{t \times d\_k} \newline
V\_{\text{full}} &= [V\_{\text{cache}}; v\_{\text{new}}] \in \mathbb{R}^{t \times d\_v}
\end{aligned}
$$

Cached Attention Computation:

$$
\begin{aligned}
\text{scores} &= \frac{q\_{\text{new}} K\_{\text{full}}^T}{\sqrt{d\_k}} \in \mathbb{R}^{1 \times t} \newline
\text{weights} &= \text{softmax}(\text{scores}) \in \mathbb{R}^{1 \times t} \newline
\text{output} &= \text{weights} \cdot V\_{\text{full}} \in \mathbb{R}^{1 \times d\_v}
\end{aligned}
$$

Complexity Summary:

Time Complexity per Generation Step:

$$
\begin{aligned}
\text{First token:} \quad &O(d\_{\text{model}}^2) \text{ (no cache)} \newline
\text{Subsequent tokens:} \quad &O(d\_{\text{model}}^2 + H \times \text{seq\_len} \times d\_k) \text{ per token}
\end{aligned}
$$

Space Complexity:

$$
\begin{aligned}
\text{Cache storage:} \quad &O(L \cdot H \cdot n\_{\max} \cdot d\_k) \newline
\text{Trade-off:} \quad &O(t) \text{ memory overhead for } O(t) \text{ speedup per step} \newline
L &: \text{Number of layers} \newline
H &: \text{Number of heads}
\end{aligned}
$$

Practical Considerations:

- Memory bandwidth becomes bottleneck for large models
- Cache pre-allocation avoids dynamic memory allocation overhead
- Quantization (FP16, INT8) can reduce cache memory requirements

---

## 11. Stage 6: Feed-Forward Networks

### Position-wise Nonlinear Transformations

Feed-forward networks in transformers implement position-wise MLP layers that process each token representation independently. This component provides the model's primary source of nonlinear transformation capacity and parametric memory.

Architectural Role:

- Nonlinearity injection: Introduces essential nonlinear transformations between attention layers
- Representation expansion: Temporarily expands representation space for complex computations
- Parameter concentration: Contains majority of model parameters (~2/3 in standard architectures)
- Position independence: Applies identical transformations to each sequence position

Design Philosophy:
FFNs serve as "computational bottlenecks" that force the model to compress and process information efficiently, similar to autoencoder architectures.

### FFN Architecture and Computation

Standard Two-Layer MLP:

Expansion layer:

$$
\begin{aligned}
\mathbb{R}^{d\_{\text{model}}} \to \mathbb{R}^{d\_{\text{ffn}}} \text{ where } d\_{\text{ffn}} = 4 \cdot d\_{\text{model}}
\end{aligned}
$$

2. Activation function: Element-wise nonlinearity (GELU, ReLU, or SwiGLU)

Contraction layer:

$$
\begin{aligned}
\mathbb{R}^{d\_{\text{ffn}}} \to \mathbb{R}^{d\_{\text{model}}}
\end{aligned}
$$

Computational Flow:

$$
\begin{aligned}
\text{Input:} \quad X &\in \mathbb{R}^{n \times d\_{\text{model}}} : \text{Sequence representations} \newline
\text{Expansion:} \quad H\_1 &= XW\_1 + b\_1 \in \mathbb{R}^{n \times d\_{\text{ffn}}} \newline
\text{Activation:} \quad H\_2 &= \sigma(H\_1) \in \mathbb{R}^{n \times d\_{\text{ffn}}} \newline
\text{Contraction:} \quad Y &= H\_2 W\_2 + b\_2 \in \mathbb{R}^{n \times d\_{\text{model}}}
\end{aligned}
$$

Design Rationale:

- 4× expansion: Provides sufficient representational capacity for complex transformations
- Position-wise: Each token processed independently, enabling parallelization
- Bottleneck structure: Forces efficient information compression and processing

### Mathematical Formulation

Standard FFN Transformation:

$$
\begin{aligned}
\text{FFN}(x) = W\_2 \cdot \sigma(W\_1 x + b\_1) + b\_2
\end{aligned}
$$
\begin{aligned}
W\_1 &\in \mathbb{R}^{d\_{\text{model}} \times d\_{\text{ffn}}} : \text{Expansion matrix} \newline
W\_2 &\in \mathbb{R}^{d\_{\text{ffn}} \times d\_{\text{model}}} : \text{Contraction matrix} \newline
b\_1 &\in \mathbb{R}^{d\_{\text{ffn}}}, b\_2 \in \mathbb{R}^{d\_{\text{model}}} : \text{Bias vectors} \newline
\sigma &: \text{Activation function (GELU, ReLU, SwiGLU)} \newline
d\_{\text{ffn}} &= 4 \cdot d\_{\text{model}} : \text{Standard expansion ratio}
\end{aligned}
$$

GELU Activation Function:

$$
\begin{aligned}
\text{GELU}(x) &= x \cdot \Phi(x) \newline
&= x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] \newline
\Phi(x) &: \text{Standard normal CDF}
\end{aligned}
$$

GELU provides smooth, differentiable activation with improved gradient flow compared to ReLU.

Approximation:

$$
\begin{aligned}
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
\end{aligned}
$$

📖 Activation Function Analysis: See [GELU vs ReLU](./transformers_math1.md#72-why-gelu-over-relu) and [SwiGLU Variants](./transformers_math1.md#71-complete-block-equations) for detailed comparisons.

SwiGLU Variant (Gated FFN):

$$
\begin{aligned}
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)
\end{aligned}
$$
\begin{aligned}
\odot &: \text{Element-wise (Hadamard) product} \newline
\text{Swish}(x) &= x \cdot \text{sigmoid}(x) \quad \text{(also known as SiLU)} \newline
W, V &: \text{Two separate weight matrices}
\end{aligned}
$$

Modern implementations like in the LLaMA architecture often use a more efficient formulation:
$$
\begin{aligned}
\text{FFN}_{\text{SwiGLU}}(x) = ( \text{Swish}(x W_1) \odot (x W_2) ) W_3
\end{aligned}
$$

- $W_1$ and $W_2$ project the input $x$ into an expanded intermediate dimension.
- The $\odot$ operation acts as a "gate," controlling which information flows through.
- $W_3$ projects the gated representation back to the model dimension.

To maintain a similar parameter count to the standard FFN, the intermediate dimension is typically set to $\frac{2}{3}$ of the standard FFN's `d_ffn`, which is why you often see an intermediate size of $\frac{2}{3} \times 4 d_{\text{model}} = \frac{8}{3} d_{\text{model}}$. This gated design has been shown to improve performance in many modern LLMs.

Parameter Analysis:

Standard FFN:

$$
\begin{aligned}
\text{Expansion layer:} \quad &d\_{\text{model}} \times d\_{\text{ffn}} = 4d\_{\text{model}}^2 \text{ parameters} \newline
\text{Contraction layer:} \quad &d\_{\text{ffn}} \times d\_{\text{model}} = 4d\_{\text{model}}^2 \text{ parameters} \newline
\text{Total:} \quad &8d\_{\text{model}}^2 \text{ parameters (plus biases)}
\end{aligned}
$$

Example: For GPT-2 scale, approximately 4.7M parameters per FFN layer:

$$
\begin{aligned}
d\_{\text{model}} = 768 \Rightarrow \text{~4.7M parameters per FFN layer}
\end{aligned}
$$

Computational Complexity:

$$
\begin{aligned}
\text{Forward pass:} \quad &O(n \cdot d\_{\text{model}} \cdot d\_{\text{ffn}}) = O(n \cdot d\_{\text{model}}^2) \newline
\text{Note:} \quad &\text{Dominates transformer computation cost due to large hidden dimension}
\end{aligned}
$$

---

## 12. Stage 7: Output Generation

### 🎯 Intuition: Making the Final Decision

Think of output generation like a multiple-choice test with 50,000 possible answers. After all the deep processing, the model has to decide: "What's the most likely next word?"

The process:

1. Focus on the last word: Only the final word's understanding matters for prediction
2. Consider all possibilities: Calculate how likely each of the 50,000 vocabulary words is to come next
3. Make a choice: Use various strategies to pick the final word

Real-world analogy: Like a game show where you've analyzed all the clues, and now you must choose your final answer from all possible options.

Why only the last word? In "The cat sat on ___", only the understanding after "on" matters for predicting the next word. Previous words have already influenced this final representation through attention.

### Output Generation Pipeline
Step 1: Hidden State Extraction

$$
\begin{aligned}
\text{Input:} \quad H &\in \mathbb{R}^{n \times d\_{\text{model}}} : \text{Final hidden states from transformer stack} \newline
\text{Selection:} \quad h\_{\text{last}} &= H[-1, :] \in \mathbb{R}^{d\_{\text{model}}} : \text{Last position} \newline
\text{Rationale:} \quad &\text{Only final position contains complete contextual information}
\end{aligned}
$$

Step 2: Language Model Head

$$
\begin{aligned}
\text{Linear transformation:} \quad \text{logits} &= h\_{\text{last}} W\_{\text{lm}} + b\_{\text{lm}} \newline
\text{Shape transformation:} \quad &\mathbb{R}^{d\_{\text{model}}} \to \mathbb{R}^{|V|} \newline
\text{Weight tying:} \quad W\_{\text{lm}} &= E^T \text{ (often use transpose of embedding matrix)}
\end{aligned}
$$

Step 3: Temperature Scaling

$$
\begin{aligned}
\text{Scaled logits:} \quad \text{logits}\_{\text{scaled}} &= \frac{\text{logits}}{\tau} \newline
\text{Higher temperature:} \quad \tau > 1 &: \text{Increases randomness} \newline
\text{Lower temperature:} \quad \tau < 1 &: \text{Increases determinism}
\end{aligned}
$$

Step 4: Probability Computation

$$
\begin{aligned}
\text{Softmax normalization:} \quad p\_i &= \frac{\exp(\text{logits}\_i / \tau)}{\sum\_{j=1}^{|V|} \exp(\text{logits}\_j / \tau)} \newline
\text{Result:} \quad &\text{Valid probability distribution over vocabulary}
\end{aligned}
$$

Step 5: Token Sampling

- Sampling strategy: Select next token according to chosen decoding algorithm
- Options: Greedy, top-k, nucleus (top-p), beam search

### Mathematical Formulation

Language Model Head:

$$
\begin{aligned}
\text{logits} = h\_{\text{final}} W\_{\text{lm}} + b\_{\text{lm}}
\end{aligned}
$$
\begin{aligned}
h\_{\text{final}} &\in \mathbb{R}^{d\_{\text{model}}} : \text{Final position hidden state} \newline
W\_{\text{lm}} &\in \mathbb{R}^{d\_{\text{model}} \times |V|} : \text{Language model projection matrix} \newline
b\_{\text{lm}} &\in \mathbb{R}^{|V|} : \text{Output bias (often omitted)} \newline
\text{logits} &\in \mathbb{R}^{|V|} : \text{Unnormalized vocabulary scores}
\end{aligned}
$$

Weight Tying: Commonly use transpose of token embedding matrix, reducing parameters and improving performance.

  $$
\begin{aligned}
W\_{\text{lm}} = E^T \text{ where } E \text{ is the token embedding matrix}
\end{aligned}
$$

Temperature-Scaled Softmax:

$$
\begin{aligned}
p\_i = \frac{\exp(\text{logits}\_i / \tau)}{\sum\_{j=1}^{|V|} \exp(\text{logits}\_j / \tau)}
\end{aligned}
$$

Temperature Parameter:

$$
\begin{aligned}
\tau \to 0 &: \text{Approaches deterministic (argmax) selection} \newline
\tau = 1 &: \text{Standard softmax distribution} \newline
\tau \to \infty &: \text{Approaches uniform distribution} \newline
\tau &: \text{Calibration parameter for desired confidence levels}
\end{aligned}
$$

Decoding Strategies:

Greedy Decoding

$$
\begin{aligned}
\text{next\_token} = \arg\max\_{i} p\_i
\end{aligned}
$$

- Advantages: Deterministic, fast, and simple to implement
- Disadvantages: Often produces repetitive or generic text; lacks diversity

Top-k Sampling

$$
\begin{aligned}
\text{next\_token} \sim \text{Categorical}(\text{top-k}(p, k))
\end{aligned}
$$
\begin{aligned}
\text{Process:} \quad &\text{Select tokens with highest probabilities, renormalize and sample} \newline
\text{Effect:} \quad &\text{Limits sampling to most likely options, balancing diversity and quality} \newline
k &: \text{Number of top tokens to consider}
\end{aligned}
$$

Nucleus (Top-p) Sampling:

$$
\begin{aligned}
\text{next\_token} \sim \text{Categorical}(\{i : \sum\_{j \in \text{top}(p)} p\_j \leq p\})
\end{aligned}
$$
\begin{aligned}
\text{Dynamic selection:} \quad &\text{Includes smallest set of tokens with cumulative probability threshold} \newline
\text{Adaptive:} \quad &\text{Adjusts vocabulary size based on confidence distribution} \newline
p &\geq \text{cumulative probability threshold} \newline
p &\in [0.9, 0.95] : \text{Typical range for balancing quality and diversity}
\end{aligned}
$$

Beam Search: For deterministic high-quality generation, maintains top-$$b$$ hypotheses at each step

---

## Next Steps: Advanced Topics

This completes our journey through the foundational transformer architecture. You now understand how transformers process text from raw input to output generation, including:

- Tokenization and embedding: Converting text to dense vectors
- Self-attention mechanism: How models relate different parts of sequences
- Transformer blocks: The core building blocks that process representations
- Architectural variants: Encoder-only, decoder-only, and encoder-decoder designs
- Output generation: How models produce final predictions

Continue your learning with advanced topics: For deeper understanding of how these models are trained, optimized, and deployed in practice, see [Transformer Advanced Topics](./transformers_advanced.md), which covers:

- Training objectives and data curriculum
- Backpropagation and optimization
- Parameter-efficient fine-tuning methods
- Quantization for deployment
- Evaluation and diagnostics
- Complete mathematical summaries

Together, these guides provide comprehensive coverage of transformer architectures from theoretical foundations to practical implementation.
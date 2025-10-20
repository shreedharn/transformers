# The Mathematics of Transformers: From First Principles to Practice
## Part 2: Advanced Concepts and Scaling

## Overview

This second part builds upon the foundational concepts from [Part 1: Building Intuition and Core Concepts](./transformers_math1.md) to cover advanced topics essential for implementing and scaling Transformer models in practice. Here we focus on optimization techniques, training stability, efficient attention implementations, and the mathematical considerations needed for real-world large models.

Prerequisites: We assume you've completed Part 1, which covers mathematical preliminaries, basic neural networks, attention mechanisms, multi-head attention, and Transformer blocks. If you haven't read Part 1 yet, please start there for the foundational understanding.

What You'll Learn:

- Advanced optimization algorithms (SGD momentum, Adam, AdamW) and their mathematical foundations
- Learning rate schedules and gradient clipping techniques
- Efficient attention implementations for scaling to long sequences
- Regularization and calibration techniques for better generalization
- Common pitfalls and how to avoid them
- Implementation best practices for numerical stability

Appendices:

- [A. Symbol/Shape Reference](#appendix-a-symbolshape-reference)
- [B. Key Derivations](#appendix-b-key-derivations)

Additional Resources:

- [Part 1: Building Intuition and Core Concepts](./transformers_math1.md)
- [Glossary](./glossary.md) - Comprehensive terms and definitions

## 8. Practical Numerics & Implementation Notes

### 8.1 Initialization Strategies

Xavier/Glorot for Linear Layers:

$$
\begin{aligned}
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad (49)
\end{aligned}
$$

Attention-Specific: Initialize query/key projections with smaller variance to prevent attention collapse (overly peaked attention distributions).

### 8.2 Mixed Precision Training

FP16 Forward, FP32 Gradients: Use half precision for speed, full precision for numerical stability:
💻 Implementation Example: For Automatic Mixed Precision implementation, see [Optimization Notebook](./pynb/math_ref/optimization.ipynb)

### 8.3 Gradient Clipping

Global Norm Clipping: As detailed in equation (11), we clip gradients to prevent explosive updates.

## 9. Optimization for Deep Networks

### 9.1 From SGD to Adam

📚 Quick Reference: See [Adam Optimizer](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) and [Gradient Descent](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

SGD with Momentum:

$$
\begin{aligned}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + (1-\beta) \nabla_\theta \mathcal{L} \quad (5) \newline
\theta_t &= \theta_{t-1} - \eta \mathbf{v}_t \quad (6)
\end{aligned}
$$

What momentum does: Like a ball rolling down a hill. Instead of just following the current slope (gradient), momentum keeps some memory of where you were going before. This helps you:

- Roll through small bumps (escape local minima)
- Speed up in consistent directions (valleys)  
- Slow down when direction changes (near the bottom)

Bowling ball analogy: A heavy bowling ball doesn't stop immediately when it hits a small bump - it uses its momentum to keep rolling toward the pins (optimal solution).

Understanding the formula:

$$
{\textstyle
\begin{aligned}
\mathbf{v}_t \quad & : \text{ Current "velocity" (combination of current gradient + previous velocity)} \newline
\beta \approx 0.9 \quad & : \text{ How much previous velocity to keep 90% } \newline
(1 - \beta) = 0.1 \quad & : \text{ How much current gradient to use 10% } \newline
\eta \quad & : \text{ Learning rate (step size)}
\end{aligned}
}
$$

Adam Optimizer: Combines momentum with adaptive learning rates:

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L} \quad (7)\\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2 \quad (8)\\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \quad (9)
\end{aligned}
$$

with bias-corrected estimates defined as:

$$
\begin{aligned}
\hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \quad \text{(bias-corrected first and second moment estimates)}
\end{aligned}
$$

What Adam does - explained simply:

Adam is like having a smart GPS that adjusts your driving based on two things:

$$
{\textstyle
\begin{aligned}
\text{1. } \mathbf{m}_t \text{ (momentum):} \quad &\text{"Which direction have we been going lately?" - Like momentum, but with exponential averaging} \newline
\text{2. } \mathbf{v}_t \text{ (second moment):} \quad &\text{"How bumpy has the road been?" - Tracks how much the gradients have been changing}
\end{aligned}
}
$$

The key insight: If the road has been very bumpy (high variance in gradients), take smaller steps. If it's been smooth and consistent, you can take bigger steps.

Breaking down the symbols:

$$
{\textstyle
\begin{aligned}
\beta_1 \approx 0.9 \quad &: \text{ How much to remember from previous direction (90\%)} \newline
\beta_2 \approx 0.999 \quad &: \text{ How much to remember from previous bumpiness (99.9\%)} \newline
\epsilon \approx 10^{-8} \quad &: \text{ Tiny number to prevent division by zero} \newline
\hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t \quad &: \text{ Bias-corrected estimates (explained below)}
\end{aligned}
}
$$

Bias correction intuition: At the beginning, the moment estimates are initialized to zero, creating a bias. The correction mechanism addresses this:

$$
\begin{aligned}
\mathbf{m}_0 = \mathbf{v}_0 &= 0 \quad \text{(initial bias toward zero)} \newline
\text{Correction factor:} \quad &(1-\beta^t) \quad \text{(starts small and approaches 1)}
\end{aligned}
$$

Car analogy: Adam is like cruise control that:

- Remembers which direction you've been driving (momentum)
- Adjusts speed based on road conditions (adaptive learning rate)
- Starts cautiously but gets more confident over time (bias correction)

### 9.2 Advanced Optimizers

AdamW vs Adam: AdamW decouples weight decay from gradient-based updates:

Adam with L2 regularization:

$$
{\textstyle
\begin{aligned}
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
\end{aligned}
}
$$

AdamW (decoupled weight decay):
$$
{\textstyle
\begin{aligned}
\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
}
$$

Why AdamW is better: Weight decay is applied regardless of gradient magnitude, leading to better generalization.

Beta-2 Warmup: Gradually adjust the second moment decay parameter for improved training stability:

$$
\begin{aligned}
\beta_2^{\text{initial}} &\approx 0.99 \quad \text{(high initial value)} \newline
\beta_2^{\text{final}} &\approx 0.999 \quad \text{(gradually decrease over first few thousand steps)}
\end{aligned}
$$

Gradient Accumulation: Simulate larger batch sizes:
💻 Implementation Example: For gradient accumulation implementation, see [Optimization Notebook](./pynb/math_ref/optimization.ipynb)

### 9.3 Learning Rate Schedules

Why do we need schedules? Think of learning to drive: you start slow in the parking lot (warmup), drive at normal speed on the highway (main training), then slow down carefully when approaching your destination (decay).

Warmup: Gradually increase learning rate to avoid early instability:
$$
{\textstyle
\begin{aligned}
\eta_t = \eta_{\text{max}} \cdot \min\left(\frac{t}{T_{\text{warmup}}}, 1\right) \quad (10)
\end{aligned}
}
$$

Why warmup works:

- Early training is chaotic: Random initial weights create wild gradients
- Start gentle: Small learning rate prevents the model from making terrible early decisions
- Build confidence gradually: As the model learns basic patterns, we can be more aggressive

Driving analogy: You don't floor the gas pedal the moment you start your car in winter - you let it warm up first.

Cosine Decay: Smooth reduction following cosine curve prevents abrupt changes.

Why cosine decay? 

- Smooth slowdown: Like gradually applying brakes instead of slamming them
- Fine-tuning phase: Later in training, we want to make small adjustments, not big jumps
- Mathematical smoothness: Cosine provides a natural, smooth curve from 1 to 0

Formula:
$$
{\textstyle
\begin{aligned}
\eta_t = \eta_{\text{max}} \cdot 0.5 \left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)
\end{aligned}
}
$$

Real-world analogy: Like landing an airplane - you approach fast, then gradually slow down for a smooth landing, not a crash.

Original Transformer Schedule: Combines warmup with inverse square root decay:
$$
{\textstyle
\begin{aligned}
\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot T_{\text{warmup}}^{-1.5})
\end{aligned}
}
$$

When to use cosine vs original: Cosine for fine-tuning and shorter training; original schedule for training from scratch with very large models.

### 9.4 Gradient Clipping

The Problem: Sometimes gradients become extremely large (exploding gradients), causing the model to make huge, destructive updates.

The Solution: Clip (limit) the gradients to a maximum norm.

Global Norm Clipping:
$$
{\textstyle
\begin{aligned}
\tilde{g} = \min\left(1, \frac{c}{\|\mathbf{g}\|_2}\right) \mathbf{g} \quad (11)
\end{aligned}
}
$$

What this does intuitively:

$$
{\textstyle
\begin{aligned}
&\text{Calculate the total "size" of all gradients combined: } \|\mathbf{g}\|_2 \newline
&\text{If this size exceeds our limit } c\text{, scale all gradients down proportionally} \newline
&\text{If it's within the limit, leave gradients unchanged}
\end{aligned}
}
$$

Speedometer analogy: Like a speed limiter in a car. If you try to go 120 mph but the limit is 65 mph, it scales your speed down to 65 mph while keeping you in the same direction.

Why proportional scaling? We want to keep the relative direction of updates the same, just make them smaller. It's like turning down the volume on music - all frequencies get reduced equally.

Example:

$$
{\textstyle
\begin{aligned}
&\text{Your gradients total to norm 50, but your clip value is 5} \newline
&\text{Scaling factor: } \min(1, 5/50) = 0.1 \newline
&\text{All gradients get multiplied by 0.1 reduced to 10% of original size}
\end{aligned}
}
$$

### 9.5 Numerical Stability

Log-Sum-Exp Trick: For numerical stability in softmax:
$$
{\textstyle
\begin{aligned}
\log\left(\sum_{i=1}^n e^{x_i}\right) = c + \log\left(\sum_{i=1}^n e^{x_i - c}\right) \quad (12)
\end{aligned}
}
$$

with the stabilization parameter preventing numerical overflow:

$$
\begin{aligned}
c = \max_i x_i \quad \text{(maximum value for numerical stability)}
\end{aligned}
$$

## 10. Efficient Attention & Scaling

### 10.1 Complexity Analysis

Standard Attention Complexity:

$$
{\textstyle
\begin{aligned}
\text{Time:} \quad &O(n^2 d) \text{ for sequence length } n, \text{ model dimension } d \newline
\text{Space:} \quad &O(n^2 + nd) \text{ for attention matrix and activations}
\end{aligned}
}
$$

Memory Bottleneck: The attention matrix dominates memory usage for long sequences:

$$
\begin{aligned}
A \in \mathbb{R}^{n \times n} \quad \text{(quadratic memory requirement)}
\end{aligned}
$$

Detailed Complexity Breakdown:

$$
{\textstyle
\begin{aligned}
\text{1. QK}^T \text{ computation:} \quad &O(n^2 d) \text{ time, } O(n^2) \text{ space} \newline
\text{2. Softmax normalization:} \quad &O(n^2) \text{ time and space} \newline
\text{3. Attention-Value multiplication:} \quad &O(n^2 d) \text{ time, } O(nd) \text{ space} \newline
\text{4. Total:} \quad &O(n^2 d) \text{ time, } O(n^2 + nd) \text{ space}
\end{aligned}
}
$$

Scaling Challenges:

- Quadratic scaling limits practical sequence lengths
- Memory requirements grow quadratically with sequence length
- Computational cost increases quadratically even with parallelization

### 10.2 FlashAttention: Memory-Efficient Attention

Core Idea: Compute attention without materializing the full attention matrix:

$$
\begin{aligned}
n \times n \text{ attention matrix} \quad \text{(avoided through tiling)}
\end{aligned}
$$

Tiling Strategy:

$$
{\textstyle
\begin{aligned}
\text{1. Divide } Q, K, V \text{ into blocks} \newline
\text{2. Compute attention scores block by block} \newline
\text{3. Use online softmax to maintain numerical stability} \newline
\text{4. Accumulate results without storing intermediate attention weights}
\end{aligned}
}
$$

Memory Reduction: FlashAttention achieves significant memory savings:

$$
\begin{aligned}
\text{Standard:} \quad &O(n^2) \text{ memory} \newline
\text{FlashAttention:} \quad &O(n) \text{ memory}
\end{aligned}
$$

Speed Improvement: Better GPU utilization through reduced memory bandwidth requirements.

Key Insight: Trade computational redundancy for memory efficiency - recompute rather than store.

### 10.3 Multi-Query and Grouped-Query Attention

Multi-Query Attention (MQA): Share key and value projections across heads:

$$
{\textstyle
\begin{aligned}
\text{Queries:} \quad &Q \in \mathbb{R}^{B \times H \times n \times d_k} \text{ (per-head)} \newline
\text{Keys/Values:} \quad &K, V \in \mathbb{R}^{B \times 1 \times n \times d_k} \text{ (shared)}
\end{aligned}
}
$$

Grouped-Query Attention (GQA): Intermediate approach - group heads:

$$
{\textstyle
\begin{aligned}
&\text{Divide } H \text{ heads into } G \text{ groups} \newline
&\text{Each group shares K, V projections} \newline
&\text{Reduces KV cache size by factor } H/G
\end{aligned}
}
$$

KV Cache Memory Analysis:

$$
{\textstyle
\begin{aligned}
\text{Standard MHA:} \quad &2 \cdot B \cdot H \cdot n \cdot d_k \text{ parameters} \newline
\text{MQA:} \quad &2 \cdot B \cdot 1 \cdot n \cdot d_k \text{ parameters (H× reduction)} \newline
\text{GQA:} \quad &2 \cdot B \cdot G \cdot n \cdot d_k \text{ parameters}
\end{aligned}
}
$$

Quantization: Reduce memory further with int8/fp16 KV cache storage.

### 10.4 KV Caching for Autoregressive Generation

Key Insight: During generation, keys and values for previous tokens don't change.

Cache Update:

$$
\begin{aligned}
K_{\text{cache}} \gets \mathrm{concat}(K_{\text{cache}},\ k_{\text{new}}) \quad (42)
\end{aligned}
$$

$$
{\textstyle
\begin{aligned}
K_{\text{cache}} \quad &: \text{ Cached keys from previous tokens.} \newline
V_{\text{cache}} \quad &: \text{ Cached values from previous tokens.} \newline
k_{\text{new}}, v_{\text{new}} \quad &: \text{ Key and value for the new token.} \newline
q_{\text{new}} \quad &: \text{ Query for the new token.}
\end{aligned}
}
$$

At each generation step, append the new key and value to the cache, then compute attention using the full cache.

Memory Trade-off: KV caching balances memory and computation:

$$
\begin{aligned}
\text{Cache growth:} \quad &O(nd) \text{ memory} \newline
\text{Recomputation saved:} \quad &O(n^2) \text{ operations eliminated}
\end{aligned}
$$

💻 Implementation Example: For KV Cache implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

### 10.5 Linear Attention Approximations

Kernel Method View: Use feature maps to approximate softmax attention:

$$
\begin{aligned}
\text{Original:} \quad &\text{softmax}(\mathbf{q}^T\mathbf{k}) \newline
\text{Approximation:} \quad &\phi(\mathbf{q})^T \phi(\mathbf{k}) \quad \text{(feature map } \phi\text{)}
\end{aligned}
$$

Linear Attention:
$$
{\textstyle
\begin{aligned}
\text{LinAttn}(Q,K,V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})} \quad (45)
\end{aligned}
}
$$

Complexity Reduction: Linear attention improves computational complexity:

$$
\begin{aligned}
\text{Standard:} \quad &O(n^2 d) \newline
\text{Linear:} \quad &O(nd^2) \quad \text{(beneficial when } d < n\text{)}
\end{aligned}
$$

## 11. Regularization, Generalization, and Calibration

### 11.1 Dropout in Transformers

Attention Dropout: Applied to attention weights:
$$
{\textstyle
\begin{aligned}
A_{\text{dropped}} = \text{Dropout}(\text{softmax}(QK^T/\sqrt{d_k})) \quad (46)
\end{aligned}
}
$$

FFN Dropout: Applied after first linear transformation:
$$
{\textstyle
\begin{aligned}
\text{FFN}(\mathbf{x}) = W_2 \cdot \text{Dropout}(\text{GELU}(W_1 \mathbf{x})) \quad (47)
\end{aligned}
}
$$

### 11.2 Evaluation and Calibration

Expected Calibration Error (ECE): Measures how well predicted probabilities match actual outcomes:
$$
{\textstyle
\begin{aligned}
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|
\end{aligned}
}
$$

with the following components:

$$
\begin{aligned}
B_m &: \text{Probability bins} \newline
\text{acc}(B_m) &: \text{Accuracy in bin } m \newline
\text{conf}(B_m) &: \text{Confidence in bin } m
\end{aligned}
$$

Temperature Scaling: Post-training calibration method:
$$
{\textstyle
\begin{aligned}
P_{\text{cal}}(y|x) = \text{softmax}(\mathbf{z}/T)
\end{aligned}
}
$$

with temperature parameter controlling confidence:

$$
\begin{aligned}
T > 1 &: \text{Less confident predictions (smoother distribution)} \newline
T < 1 &: \text{More confident predictions (sharper distribution)}
\end{aligned}
$$

Perplexity Dependence on Tokenizer: PPL comparisons only valid with same tokenizer. Different tokenizers create different sequence lengths and vocabulary sizes.

Example: "hello world" might be:

- GPT tokenizer: ["hel", "lo", " wor", "ld"] (4 tokens)
- Character-level: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] (11 tokens)

### 11.3 Advanced Tokenization

Byte-Level BPE vs Unigram:

- BPE: Greedily merges frequent character pairs, handles any Unicode
- Unigram: Probabilistic model, often better for morphologically rich languages

Special Token Handling:

- BOS (Beginning of Sequence): Often used for unconditional generation
- EOS (End of Sequence): Signals completion, crucial for proper training
- PAD: For batching variable-length sequences

Embedding/LM-Head Tying Caveats:
When sharing weights, ensure shape compatibility:

$$
{\textstyle
\begin{aligned}
\text{Embedding:} \quad &E \in \mathbb{R}^{V \times d_{\text{model}}} \newline
\text{LM head: needs} \quad &\mathbb{R}^{d_{\text{model}} \times V} \newline
\text{Solution: Use } E^T \text{ for output projection (as shown in equation 40)}
\end{aligned}
}
$$

### 11.4 Label Smoothing

Smooth Labels: Replace one-hot targets with:
$$
{\textstyle
\begin{aligned}
y_{\text{smooth}} = (1-\alpha) y_{\text{true}} + \frac{\alpha}{V} \mathbf{1} \quad (48)
\end{aligned}
}
$$

Effect on Gradients: Prevents overconfident predictions and improves calibration.

## 14. Common Pitfalls & Misconceptions

### 14.1 High-Dimensional Distance Misconceptions

Pitfall: Using Euclidean distance instead of cosine similarity in high dimensions.

Fix: In high-dimensional spaces, cosine similarity is more discriminative:

$$
\begin{aligned}
d > 100 \quad \text{(most vectors are approximately orthogonal)}
\end{aligned}
$$

### 14.2 Attention Scaling Mistakes

Pitfall: Forgetting the proper scaling factor or using wrong dimension.

Symptom: Attention weights become too peaked, leading to poor gradients.

Fix: Always apply the correct scaling:

$$
\begin{aligned}
\text{Scaling factor:} \quad &1/\sqrt{d_k} \newline
\text{Key dimension:} \quad &d_k = d_{\text{model}}/h \quad \text{(common implementation)}
\end{aligned}
$$

### 14.3 LayerNorm Placement

Pitfall: Using post-LayerNorm (original) instead of pre-LayerNorm (modern).
Issue: Post-LN can lead to training instability in deep models.
Modern Practice: Apply LayerNorm before attention and FFN blocks.

### 14.4 Softmax Temperature Misuse

Pitfall: Applying temperature scaling inconsistently.

Correct Usage: Use temperature parameter to control distribution sharpness:

$$
\begin{aligned}
\text{softmax}(\mathbf{z}/\tau) \quad \text{where } \tau \text{ controls sharpness}
\end{aligned}
$$

$$
{\textstyle
\begin{aligned}
\tau > 1 \quad &: \text{ Smoother distribution} \newline
\tau < 1 \quad &: \text{ Sharper distribution}
\end{aligned}
}
$$

## 15. Summary & What to Learn Next

### 15.1 Key Mathematical Insights

$$
{\textstyle
\begin{aligned}
\text{1. Attention as Similarity Search:} \quad &\text{Q/K/V framework emerges naturally from maximum inner product search} \newline
\text{2. Scaling Laws:} \quad &1/\sqrt{d_k} \text{ scaling prevents attention collapse (overly peaked distributions) in high dimensions} \newline
\text{3. Residual Connections:} \quad &\text{Enable gradient flow through deep networks via skip connections} \newline
\text{4. Multi-Head Architecture:} \quad &\text{Parallel subspace projections enable diverse attention patterns}
\end{aligned}
}
$$

### 15.2 Advanced Techniques Covered

1. Optimization: SGD momentum, Adam, AdamW with proper learning rate schedules
2. Efficiency: FlashAttention, Multi-Query/Grouped-Query Attention, KV caching
3. Regularization: Dropout, label smoothing, calibration techniques
4. Numerical Stability: Gradient clipping, mixed precision, proper initialization

### 15.3 Next Steps

Scaling Laws: Study how performance scales with model size, data, and compute (Kaplan et al., 2020)

Parameter-Efficient Fine-Tuning: LoRA, adapters, and other methods for efficient adaptation

Retrieval-Augmented Models: Combining parametric knowledge with external memory

Advanced Architectures: Mixture of Experts, sparse attention patterns, and alternative architectures

---

## Connection to Part 1

This tutorial builds directly on the foundations established in [Part 1: Building Intuition and Core Concepts](./transformers_math1.md). Together, these two parts provide a complete mathematical understanding of Transformer architectures, from basic principles through advanced implementation considerations.

If you haven't already, we highly recommend reading Part 1 first to build the necessary intuition before diving into these advanced topics.

---

## Further Reading

For a comprehensive collection of all papers referenced in this tutorial and additional resources, see **[Further Reading](./further.md)**.

Key papers referenced in this Part 2:

**Optimization:**
- Loshchilov & Hutter (2019) - AdamW optimizer
- Kaplan et al. (2020) - Scaling laws

**Efficiency:**
- Dao et al. (2022) - FlashAttention
- Shazeer (2019) - Multi-query attention

**Architecture:**
- Vaswani et al. (2017) - Original transformer
- Xiong et al. (2020) - LayerNorm placement
- Press & Wolf (2017) - Weight tying

---

## Appendix A: Symbol/Shape Reference

### Single-Head Attention Shapes
| Symbol | Meaning | Typical Shape |
|:--------|:---------|:---------------|
| \(Q, K, V\) | Query, Key, Value matrices | \([n \times d_k], [n \times d_k], [n \times d_v]\) |
| \(n\) | Sequence length | Scalar |
| \(d_{\text{model}}\) | Model dimension | Scalar (512, 768, 1024, etc.) |
| \(d_k, d_v\) | Key, value dimensions | Usually \(d_{\text{model}}/h\) |
| \(h\) | Number of attention heads | Scalar (8, 12, 16, etc.) |



### Multi-Head & Batched Shapes

| Symbol | Meaning | Batched Multi-Head Shape |
|:-------|:---------|:------------------------|
| \(Q, K, V\) | Projected queries, keys, values | \([B, H, n, d_k], [B, H, n, d_k], [B, H, n, d_v]\) |
| \(\text{Attn}\) | Attention weights matrix | \([B, H, n, n]\) |
| \(\text{Output}_{\text{pre}}\) | Attention output (pre-concat) | \([B, H, n, d_v]\) |
| \(\text{Output}_{\text{proj}}\) | Final output (post-concat) | \([B, n, d_{\text{model}}]\) |
| \(W^Q, W^K, W^V, W^{\text{proj}}\) | Attention projection matrices | \([d_{\text{model}} \times d_k]\) per head |
| \(W^O\) | Output projection | \([d_{\text{model}} \times d_{\text{model}}]\) |


Convention:

$$
\begin{aligned}
B &: \text{Batch size} \newline
H &: \text{Number of heads} \newline
n &: \text{Sequence length} \newline
d_k = d_v &= d_{\text{model}}/H
\end{aligned}
$$

## Appendix B: Key Derivations

### B.1 Softmax Gradient

For the softmax probability:

$$
\begin{aligned}
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\end{aligned}
$$

The gradient is:

$$
{\textstyle
\begin{aligned}
\frac{\partial p_i}{\partial z_j} = \begin{cases}
p_i(1 - p_i) & \text{if } i = j \\
-p_i p_j & \text{if } i \neq j
\end{cases} = p_i(\delta_{ij} - p_j)
\end{aligned}
}
$$

### B.2 Matrix Calculus Identities

$$
\begin{aligned}
\text{Trace-Vec Identity:} \quad &\text{tr}(AB) = \text{vec}(A^T)^T \text{vec}(B) \newline
\text{Kronecker Product:} \quad &\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X) \newline
\text{Chain Rule for Matrices:} \quad &\frac{\partial f}{\partial X} = \sum_Y \frac{\partial f}{\partial Y} \frac{\partial Y}{\partial X}
\end{aligned}
$$
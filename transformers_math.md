# The Mathematics of Transformers: From First Principles to Practice

## Abstract

This tutorial builds the mathematical foundations of Transformer architectures from first principles, targeting motivated high school students with basic algebra and geometry background. We progress systematically from optimization theory and high-dimensional geometry through attention mechanisms to complete Transformer blocks, emphasizing mathematical intuition, worked derivations, and practical implementation considerations. Every mathematical concept is explained with real-world analogies and intuitive reasoning before diving into the formal mathematics.

## Assumptions & Conventions

**Mathematical Notation:**
- Vectors are row-major; matrices multiply on the right
- Shapes annotated as [seq, dim] or [batch, seq, heads, dim]
- Masking uses additive large negative values (-∞) before softmax
- Row-wise softmax normalization
- Token/position numbering starts from 0
- Default dtype is fp32 unless specified
- Equations numbered sequentially throughout document

## Table of Contents

1. [Roadmap](#1-roadmap)
2. [Mathematical Preliminaries](#2-mathematical-preliminaries)
   - [2.1.0 From Line Slopes to Neural Network Training](#210-from-line-slopes-to-neural-network-training)
3. [Multilayer Perceptrons as a Warm-Up](#3-multilayer-perceptrons-as-a-warm-up)
4. [High-Dimensional Geometry & Similarity](#4-high-dimensional-geometry--similarity)
5. [From Similarity to Attention](#5-from-similarity-to-attention)
6. [Multi-Head Attention & Positional Information](#6-multi-head-attention--positional-information)
7. [Transformer Block Mathematics](#7-transformer-block-mathematics)
8. [Practical Numerics & Implementation Notes](#8-practical-numerics--implementation-notes)
9. [Optimization for Deep Networks](#9-optimization-for-deep-networks)
10. [Efficient Attention & Scaling](#10-efficient-attention--scaling)
11. [Regularization, Generalization, and Calibration](#11-regularization-generalization-and-calibration)
12. [Training Objective & Tokenization/Embeddings](#12-training-objective--tokenizationembeddings)
13. [Worked Mini-Examples](#13-worked-mini-examples)
14. [Common Pitfalls & Misconceptions](#14-common-pitfalls--misconceptions)
15. [Summary & What to Learn Next](#15-summary--what-to-learn-next)

**Appendices:**
- [A. Symbol/Shape Reference](#appendix-a-symbolshape-reference)
- [B. Key Derivations](#appendix-b-key-derivations)
- [C. Glossary](#appendix-c-glossary)

## 1. Roadmap

We begin with optimization fundamentals and high-dimensional geometry, then build attention as a principled similarity search mechanism. The journey: **gradients → similarity metrics → attention → multi-head attention → full Transformer blocks → efficient inference**. Each step connects mathematical theory to practical implementation, culminating in a complete understanding of how Transformers process sequences through learned representations and attention-based information routing.

## 2. Mathematical Preliminaries

📚 **Quick Reference**: For a comprehensive table of all mathematical concepts used in neural networks, see [Mathematical Quick Reference](./math_quick_ref.md).

### 2.1 Linear Algebra Essentials

**What is a vector?** Think of it as an arrow in space that has both direction and length. In transformers, vectors represent word meanings - words with similar meanings point in similar directions.

**Vectors and Norms:** For $\mathbf{v} \in \mathbb{R}^d$ (a vector with d numbers):
- L2 norm: 
```math
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^d v_i^2}
```
  - **What this means:** The "length" of the vector, like measuring a stick with a ruler
  - **Why it matters:** Longer vectors represent "stronger" or "more confident" representations

- Inner product: 
```math
\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^d u_i v_i
```
  - **What this measures:** How much two vectors point in the same direction
  - **Intuition:** Like measuring how similar two things are. If vectors represent word meanings, a high inner product means the words are related
  - **Real example:** "king" and "queen" vectors would have a high inner product because they're conceptually similar

**Matrix Operations:** For matrices $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$:
- Matrix multiplication: 
```math
(AB)_{ij} = \sum_{k=1}^n A_{ik}B_{kj}
```
  - **What this does:** Combines information from two tables of numbers in a specific way
  - **Think of it as:** Matrix A transforms space, then matrix B transforms it again - like applying two filters in sequence

- Transpose: 
```math
(A^T)_{ij} = A_{ji}
```
  - **What transpose ($A^T$) means:** Flip the matrix over its diagonal - rows become columns, columns become rows
  - **Why we use it:** Often we need to "reverse" a transformation or change the direction of information flow

- Block matrices enable efficient computation of attention over sequences
  - **Practical point:** Instead of processing one word at a time, we can process entire sentences at once using matrices

### 2.2 Matrix Calculus Essentials

**What is a gradient?** Think of hiking on a mountain. The gradient at any point tells you the direction of steepest ascent. In machine learning, we want to find the valley (minimum loss), so we go in the opposite direction of the gradient - downhill.

**Gradient Shapes:** If $f: \mathbb{R}^{m \times n} \to \mathbb{R}$, then $\nabla_X f \in \mathbb{R}^{m \times n}$

**What this means:** The gradient has the same shape as what you're taking the gradient with respect to. If X is a 3×4 matrix, its gradient is also 3×4. This makes sense - you need to know how much to adjust each individual number in X.

**Chain Rule:** For $f(g(x))$: 
```math
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \frac{\partial g}{\partial x}
```

**Chain rule intuition:** Like a chain reaction. If changing x affects g, and changing g affects f, then the total effect of x on f is the product of both effects. It's like asking "if I turn this knob (x) by a little bit, how much does the final output (f) change?" You multiply the sensitivity at each step.

**Useful Identities:**
- If $Y = AXB$ and $f$ depends on $Y$, then: 
```math
\frac{\partial f}{\partial X} = A^T \left(\frac{\partial f}{\partial Y}\right) B^T
```
  **Parameters:** $A\in\mathbb{R}^{m\times r}, X\in\mathbb{R}^{r\times s}, B\in\mathbb{R}^{s\times n}$, so $\partial f/\partial X\in\mathbb{R}^{r\times s}$.
  **Intuition:** Backward through linear maps flips them with transposes.
- 
```math
\nabla_X \text{tr}(AX) = A^T
```
- 
```math
\nabla_X \|X\|_F^2 = 2X
```

**What these mean:**
- The first says "when matrices multiply, gradients flow backward through transposes"
- The second: "the trace (sum of diagonal elements) has a simple gradient"
- The third: "the gradient of squared magnitude is just twice the original" (like how d/dx(x²) = 2x)

### 2.3 Probability & Information Theory

**Softmax as Gibbs Distribution:**
```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \quad (1)
```

**What softmax does intuitively:** Imagine you're choosing which restaurant to go to. Each restaurant has a "score" (z_i). Softmax converts these raw scores into probabilities that sum to 1, like saying "there's a 30% chance I'll pick restaurant A, 50% chance for B, 20% chance for C." The exponential function (e^x) ensures that higher scores get disproportionately higher probabilities - if one restaurant is much better, it gets most of the probability mass.

**Why use exponentials?** Because they're always positive (probabilities can't be negative) and they amplify differences - a score of 5 vs 4 creates a much bigger probability difference than 2 vs 1.

This represents a Gibbs distribution with "energy" $-z_i$ and temperature $T=1$.

**Cross-Entropy Loss:**
```math
\mathcal{L} = -\sum_{i=1}^n y_i \log p_i \quad (2)
```

**What cross-entropy loss does intuitively:** Think of it as a "wrongness penalty." If the model predicts the right answer with high confidence (p_i close to 1), the loss is small. If it predicts the wrong answer or is uncertain about the right answer, the loss is large. The logarithm harshly penalizes being confidently wrong - predicting 0.01% chance for the correct answer gives a huge penalty.

**Why cross-entropy instead of just counting wrong answers?** Because it rewards confidence in correct predictions and punishes overconfidence in wrong ones. It's like the difference between "I'm pretty sure this is right" vs "I'm absolutely certain this is wrong."

**Deeper reasons why cross-entropy is perfect for transformers:**

1. **Information-theoretic foundation:** Cross-entropy measures the "surprise" in the prediction. If the model is confident in the right answer (low surprise), loss is low. If it's surprised by the correct answer (assigned low probability), loss is high.

2. **Matches the softmax output:** Since transformers output probability distributions via softmax, we need a loss function that works with probabilities. Cross-entropy is the natural choice for probability distributions.

3. **Encourages proper calibration:** Unlike squared error, cross-entropy pushes the model to output probabilities that reflect its true confidence. A well-trained model should be right 90% of the time when it says it's 90% confident.

4. **Smooth gradients everywhere:** The gradient of cross-entropy with respect to the final layer outputs is simply (predicted probability - true probability). This creates clean, well-behaved gradients for training.

5. **Class imbalance considerations:** Unweighted cross-entropy does **not** handle class imbalance by itself; use class weights or resampling when imbalance is a concern (language modeling often leaves it unweighted because frequencies reflect the data distribution).

**Mathematical intuition:** The logarithm grows very quickly as probability approaches 0. This means being slightly wrong about a high-confidence prediction gets heavily penalized, encouraging the model to be humble about its uncertainty.

where $y_i$ are true labels (what the answer actually is) and $p_i$ are predicted probabilities (what the model thinks the answer is).

## 2.1 Calculus to Differential Equations

### 2.1.0 From Line Slopes to Neural Network Training

**Building Intuition: Slope of a Line**

Let's start with something familiar - the equation of a straight line:
```math
y = mx + b
```

where:
- $m$ is the **slope** - tells us how steep the line is
- $b$ is the **y-intercept** - where the line crosses the y-axis

**What does slope mean intuitively?** Slope tells us "for every step I move right, how much do I move up or down?" If the slope is 2, then moving 1 step right means moving 2 steps up. If the slope is -0.5, then moving 1 step right means moving 0.5 steps down.

**Connecting to Optimization:** In machine learning, we want to find the "bottom of a valley" - the point where our error is smallest. To do this, we need to know which direction is "downhill."

#### From 2D Slopes to Gradient Descent

**The Derivative as Slope:** For any function $f(x)$, the derivative $\frac{df}{dx}$ tells us the slope at any point:

```math
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
```

**What this equation means:** "If I move a tiny amount h to the right, how much does f change?" The derivative is the limit as that tiny amount approaches zero.

**Gradient Descent in One Dimension:** If we want to minimize $f(x)$, we update $x$ using:

```math
x_{\text{new}} = x_{\text{old}} - \alpha \frac{df}{dx}
```

where $\alpha$ (alpha) is the **learning rate** - how big steps we take.

**The key insight:** The negative sign makes us go *opposite* to the slope direction. If the slope is positive (going uphill to the right), we move left. If the slope is negative (going downhill to the right), we move right.

#### Worked Example: f(x) = x²

Let's minimize $f(x) = x^2$ starting from $x = 3$:

**Step 1:** Compute the derivative: $\frac{df}{dx} = 2x$

**Step 2:** Choose learning rate: $\alpha = 0.1$

**Step 3:** Update step by step:

```
Iteration 0: x = 3.0, f(x) = 9.0, df/dx = 6.0
             x_new = 3.0 - 0.1 × 6.0 = 2.4

Iteration 1: x = 2.4, f(x) = 5.76, df/dx = 4.8  
             x_new = 2.4 - 0.1 × 4.8 = 1.92

Iteration 2: x = 1.92, f(x) = 3.69, df/dx = 3.84
             x_new = 1.92 - 0.1 × 3.84 = 1.54

...continuing...

Iteration 10: x ≈ 0.27, f(x) ≈ 0.07
```

**What's happening:** We're taking steps toward x = 0 (the minimum of x²), and each step gets smaller as we approach the bottom.

#### Extending to Multiple Variables: Gradients as Vectors

**From Derivatives to Gradients:** When we have multiple variables, like $f(x, y) = x^2 + y^2$, we need **partial derivatives**:

- $\frac{\partial f}{\partial x} = 2x$ (rate of change with respect to x, holding y fixed)
- $\frac{\partial f}{\partial y} = 2y$ (rate of change with respect to y, holding x fixed)

**The Gradient Vector:** We combine these into a **gradient vector**:

```math
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
```

**Vector Gradient Descent:** Now our update rule becomes:

```math
\begin{bmatrix} x_{\text{new}} \\ y_{\text{new}} \end{bmatrix} = \begin{bmatrix} x_{\text{old}} \\ y_{\text{old}} \end{bmatrix} - \alpha \begin{bmatrix} 2x_{\text{old}} \\ 2y_{\text{old}} \end{bmatrix}
```

**Intuition:** The gradient vector points in the direction of steepest *ascent*. By moving in the *opposite* direction (negative gradient), we go downhill most quickly.

#### Matrix Form for Machine Learning

**Setting up the Problem:** In machine learning, we have:
- **X**: Input matrix with shape (samples × features) - each row is one data point
- **W**: Weight matrix with shape (features × outputs) - the parameters we want to learn  
- **b**: Bias vector with shape (outputs,) - additional adjustable parameters
- **Y_hat**: Predictions with shape (samples × outputs), computed as Y_hat = XW + b

**Loss Function:** We measure how wrong our predictions are using mean squared error:

```math
L = \frac{1}{N} \sum_{i=1}^N \|Y_{\text{true}}^{(i)} - Y_{\text{hat}}^{(i)}\|^2
```

**Matrix Gradients:** To minimize the loss, we need gradients with respect to W and b:

```math
\frac{\partial L}{\partial W} = \frac{2}{N} X^T (Y_{\text{hat}} - Y_{\text{true}})
```

```math
\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^N (Y_{\text{hat}}^{(i)} - Y_{\text{true}}^{(i)})
```

**Matrix Gradient Descent Updates:**

```math
W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W}
```

```math
b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial L}{\partial b}
```

**Key Insight:** The same learning rate $\alpha$ controls the step size for all parameters, just like in our simple 1D case.

#### Single Hidden Layer MLP: Putting It All Together

**Forward Pass Equations:**

```math
Z_1 = X W_1 + b_1 \quad \text{(shape: samples $\times$ hidden units)}
```
```math
A_1 = \sigma(Z_1) \quad \text{(apply activation function element-wise)}
```
```math
Z_2 = A_1 W_2 + b_2 \quad \text{(shape: samples $\times$ outputs)}
```
```math
Y_{\text{hat}} = Z_2 \quad \text{(final predictions)}
```

where $\sigma$ is an activation function like ReLU or sigmoid.

**Backward Pass (Backpropagation):**

**Step 1:** Compute the error at the output:
```math
\delta_2 = Y_{\text{hat}} - Y_{\text{true}} \quad \text{(shape: samples × outputs)}
```

**Step 2:** Compute gradients for output layer:
```math
\frac{\partial L}{\partial W_2} = \frac{1}{N} A_1^T \delta_2
```
```math
\frac{\partial L}{\partial b_2} = \frac{1}{N} \sum_{i=1}^N \delta_2^{(i)}
```

**Step 3:** Backpropagate error to hidden layer:
```math
\delta_1 = (\delta_2 W_2^T) \odot \sigma'(Z_1) \quad \text{(element-wise multiplication)}
```

**Step 4:** Compute gradients for hidden layer:
```math
\frac{\partial L}{\partial W_1} = \frac{1}{N} X^T \delta_1
```
```math
\frac{\partial L}{\partial b_1} = \frac{1}{N} \sum_{i=1}^N \delta_1^{(i)}
```

**Step 5:** Update all parameters using the same learning rate:
```math
W_1 \leftarrow W_1 - \alpha \frac{\partial L}{\partial W_1}
```
```math
b_1 \leftarrow b_1 - \alpha \frac{\partial L}{\partial b_1}
```
```math
W_2 \leftarrow W_2 - \alpha \frac{\partial L}{\partial W_2}
```
```math
b_2 \leftarrow b_2 - \alpha \frac{\partial L}{\partial b_2}
```

**Understanding the δ terms:**
- $\delta_2$: "How much does changing each output neuron's value affect the loss?"
- $\delta_1$: "How much does changing each hidden neuron's value affect the loss?"

The $\delta$ terms flow backwards through the network, carrying error information from the output back to earlier layers.

#### The Learning Rate α: Universal Step Size Controller

**Same Role Everywhere:** Notice that $\alpha$ plays the identical role in:
- 1D gradient descent: $x \leftarrow x - \alpha \frac{df}{dx}$
- Vector gradient descent: $\mathbf{x} \leftarrow \mathbf{x} - \alpha \nabla f$
- Matrix gradient descent: $W \leftarrow W - \alpha \frac{\partial L}{\partial W}$
- Neural network training: All parameters use the same $\alpha$

**Choosing α:**
- **Too large:** Updates overshoot the minimum, causing oscillation or divergence
- **Too small:** Updates are tiny, causing very slow convergence
- **Just right:** Steady progress toward the minimum without overshooting

#### Visual Connection: From Slopes to Networks

```
1D Slope:          2D Gradient:           Matrix Gradients:        MLP Training:
                                                                         
    f(x)             f(x,y)                    L(W,b)                Forward:
     /|                 /|\                      /|\                X→Z₁→A₁→Z₂→Ŷ
    / |                / | \                    / | \                   
   /  |               /  |  \                  /  |  \               Backward:
slope |             ∇f   |  ∇f                ∇L  |   ∇L           δ₂←δ₁←∇W₁,∇b₁
   \  |               \  |  /                  \  |  /               
    \ |                \ | /                    \ | /                Update:
     \|                 \|/                      \|/               W,b ← W,b-α∇
      x                  x                        θ                 (same α!)
                                                                     
                                                                         
Update: x₁ = x₀ - α(df/dx)   [x,y]₁ = [x,y]₀ - α∇f    θ₁ = θ₀ - α∇L    All params use α
```

**The Big Picture:** Whether we're finding the bottom of a simple parabola or training a neural network with millions of parameters, we're doing the same fundamental thing:

1. **Measure the slope** (derivative, gradient, or backpropagated error)
2. **Take a step in the opposite direction** (negative sign)
3. **Control step size** (learning rate α)
4. **Repeat until we reach the bottom**

This is why understanding the simple case of line slopes gives us insight into the most sophisticated neural network training algorithms.

### 2.1.1 Gradient Fields and Optimization

**Gradient Descent as Continuous Flow:** Parameter updates $\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$ approximate the ODE:
```math
\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta) \quad (3)
```

**Understanding the symbols:**
- $\theta$ (theta): The parameters we want to learn - think of these as the "knobs" we can adjust
- $\eta$ (eta): Learning rate - how big steps we take. Like deciding whether to take baby steps or giant leaps when hiking downhill
- $\nabla$ (nabla): The gradient symbol - points in the direction of steepest ascent (we go opposite direction to descend)
- $\mathcal{L}$ (script L): The loss function - measures how "wrong" our current parameters are

**What the equation means:** "Change the parameters in the opposite direction of the gradient, scaled by the learning rate."

This connects discrete optimization to continuous dynamical systems.

**Why This Matters:** Understanding optimization as flow helps explain momentum methods, learning rate schedules, and convergence behavior.

### 2.1.2 Residual Connections as Discretized Dynamics

**Residual Block:** $\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)$ approximates:
```math
\frac{d\mathbf{h}}{dt} = F(\mathbf{h}) \quad (4)
```

**What residual connections do intuitively:** Think of them as "safety nets" for information. Without residual connections, information would have to successfully pass through every layer to reach the output. With residual connections, information can "skip over" layers that might be learning slowly or poorly.

**Highway analogy:** Imagine driving from city A to city B. Without residual connections, you MUST go through every small town along the way. With residual connections, there's a highway that bypasses some towns - you still visit some towns (transformation), but you're guaranteed to make progress toward your destination even if some towns are roadblocked.

**Why this enables deep networks:** In very deep networks (50+ layers), gradients tend to vanish as they backpropagate. Residual connections provide a "gradient highway" - gradients can flow directly backward through the skip connections, ensuring that even early layers receive useful training signals.

This enables training very deep networks by maintaining gradient flow.

**Stability Consideration:** The transformation $F$ should be well-conditioned to avoid exploding/vanishing gradients.

💻 **Implementation Example**: For a practical implementation of residual connections, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

## 9. Optimization for Deep Networks

### 9.1 From SGD to Adam

📚 **Quick Reference**: See [Adam Optimizer](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) and [Gradient Descent](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

**SGD with Momentum:**
```math
\begin{align}
\mathbf{v}_t &= \beta \mathbf{v}_{t-1} + (1-\beta) \nabla_\theta \mathcal{L} \quad (5)\\
\theta_t &= \theta_{t-1} - \eta \mathbf{v}_t \quad (6)
\end{align}
```

**What momentum does:** Like a ball rolling down a hill. Instead of just following the current slope (gradient), momentum keeps some memory of where you were going before. This helps you:
- **Roll through small bumps** (escape local minima)
- **Speed up in consistent directions** (valleys)  
- **Slow down when direction changes** (near the bottom)

**Bowling ball analogy:** A heavy bowling ball doesn't stop immediately when it hits a small bump - it uses its momentum to keep rolling toward the pins (optimal solution).

**Understanding the formula:**
- $\mathbf{v}_t$: Current \"velocity\" (combination of current gradient + previous velocity)
- $\beta \approx 0.9$: How much previous velocity to keep (90%)  
- $(1-\beta) = 0.1$: How much current gradient to use (10%)
- $\eta$: Learning rate (step size)

**Adam Optimizer:** Combines momentum with adaptive learning rates:
```math
\begin{align}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_\theta \mathcal{L} \quad (7)\\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_\theta \mathcal{L})^2 \quad (8)\\
\theta_t &= \theta_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \quad (9)
\end{align}
```

where $\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$ are bias-corrected estimates.

**What Adam does - explained simply:**

Adam is like having a smart GPS that adjusts your driving based on two things:

1. **$\mathbf{m}_t$ (momentum):** "Which direction have we been going lately?" - Like momentum, but with exponential averaging
2. **$\mathbf{v}_t$ (second moment):** "How bumpy has the road been?" - Tracks how much the gradients have been changing

**The key insight:** If the road has been very bumpy (high variance in gradients), take smaller steps. If it's been smooth and consistent, you can take bigger steps.

**Breaking down the symbols:**
- $\beta_1 \approx 0.9$: How much to remember from previous direction (90%)
- $\beta_2 \approx 0.999$: How much to remember from previous bumpiness (99.9%) 
- $\epsilon \approx 10^{-8}$: Tiny number to prevent division by zero
- $\hat{\mathbf{m}}_t$, $\hat{\mathbf{v}}_t$: Bias-corrected estimates (explained below)

**Bias correction intuition:** At the beginning, $\mathbf{m}_0 = \mathbf{v}_0 = 0$, so the averages are biased toward zero. We correct for this by dividing by $(1-\beta^t)$, which starts small and approaches 1.

**Car analogy:** Adam is like cruise control that:
- Remembers which direction you've been driving (momentum)
- Adjusts speed based on road conditions (adaptive learning rate)
- Starts cautiously but gets more confident over time (bias correction)

### 9.2 Advanced Optimizers

**AdamW vs Adam:** AdamW decouples weight decay from gradient-based updates:

**Adam with L2 regularization:**
```math
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
```

**AdamW (decoupled weight decay):**
```math
\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

**Why AdamW is better:** Weight decay is applied regardless of gradient magnitude, leading to better generalization.

**$\beta_2$ Warmup:** Start with high $\beta_2$ (e.g., 0.99) and gradually decrease to final value (e.g., 0.999) over first few thousand steps. Helps with training stability.

**Gradient Accumulation:** Simulate larger batch sizes:
💻 **Implementation Example**: For gradient accumulation implementation, see [Optimization Notebook](./pynb/math_ref/optimization.ipynb)

### 9.3 Learning Rate Schedules

**Why do we need schedules?** Think of learning to drive: you start slow in the parking lot (warmup), drive at normal speed on the highway (main training), then slow down carefully when approaching your destination (decay).

**Warmup:** Gradually increase learning rate to avoid early instability:
```math
\eta_t = \eta_{\text{max}} \cdot \min\left(\frac{t}{T_{\text{warmup}}}, 1\right) \quad (10)
```

**Why warmup works:**
- **Early training is chaotic:** Random initial weights create wild gradients
- **Start gentle:** Small learning rate prevents the model from making terrible early decisions
- **Build confidence gradually:** As the model learns basic patterns, we can be more aggressive

**Driving analogy:** You don't floor the gas pedal the moment you start your car in winter - you let it warm up first.

**Cosine Decay:** Smooth reduction following cosine curve prevents abrupt changes.

**Why cosine decay?** 
- **Smooth slowdown:** Like gradually applying brakes instead of slamming them
- **Fine-tuning phase:** Later in training, we want to make small adjustments, not big jumps
- **Mathematical smoothness:** Cosine provides a natural, smooth curve from 1 to 0

**Formula:**
```math
\eta_t = \eta_{\text{max}} \cdot 0.5 \left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)
```

**Real-world analogy:** Like landing an airplane - you approach fast, then gradually slow down for a smooth landing, not a crash.

**Original Transformer Schedule:** Combines warmup with inverse square root decay:
```math
\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot T_{\text{warmup}}^{-1.5})
```

**When to use cosine vs original:** Cosine for fine-tuning and shorter training; original schedule for training from scratch with very large models.

### 4.3 Gradient Clipping

**The Problem:** Sometimes gradients become extremely large (exploding gradients), causing the model to make huge, destructive updates.

**The Solution:** Clip (limit) the gradients to a maximum norm.

**Global Norm Clipping:**
```math
\tilde{g} = \min\left(1, \frac{c}{\|\mathbf{g}\|_2}\right) \mathbf{g} \quad (11)
```

**What this does intuitively:**
- Calculate the total "size" of all gradients combined: $\|\mathbf{g}\|_2$
- If this size exceeds our limit $c$, scale all gradients down proportionally
- If it's within the limit, leave gradients unchanged

**Speedometer analogy:** Like a speed limiter in a car. If you try to go 120 mph but the limit is 65 mph, it scales your speed down to 65 mph while keeping you in the same direction.

**Why proportional scaling?** We want to keep the relative direction of updates the same, just make them smaller. It's like turning down the volume on music - all frequencies get reduced equally.

**Example:**
- Your gradients total to norm 50, but your clip value is 5
- Scaling factor: $\min(1, 5/50) = 0.1$  
- All gradients get multiplied by 0.1 (reduced to 10% of original size)

### 9.4 Numerical Stability

**Log-Sum-Exp Trick:** For numerical stability in softmax:
```math
\log\left(\sum_{i=1}^n e^{x_i}\right) = c + \log\left(\sum_{i=1}^n e^{x_i - c}\right) \quad (12)
```

where $c = \max_i x_i$ prevents overflow.

## 3. Multilayer Perceptrons as a Warm-Up

### 3.1 Forward Pass

**Two-layer MLP:**

**Shape Analysis:** If input $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{in}}}$:
- $W^{(1)} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{hidden}}}$
- $W^{(2)} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{out}}}$

### 3.2 Backpropagation Derivation

**Loss Gradient w.r.t. Output:**
```math
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \quad (16)
```

**Weight Gradients:**

```math
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{(2)}} &= (\mathbf{h}^{(1)})^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \\
\frac{\partial \mathcal{L}}{\partial W^{(1)}} &= \mathbf{x}^T \left[ \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} W^{(2)T} \right) \odot \sigma'(\mathbf{z}^{(1)}) \right]
\end{aligned}
```




- Where $\odot$ denotes element-wise multiplication.
- $\sigma'(\mathbf{z}^{(1)})$ is the derivative of the activation function applied elementwise.
- $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}}$ is the gradient of the loss with respect to the pre-activation output of the second layer.


where $\odot$ denotes element-wise multiplication.

### 3.3 Advanced Normalization Techniques

**LayerNorm:** Normalizes across features within each sample:
```math
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \quad (19)
```

**Shape Analysis:** For $\mathbf{x} \in \mathbb{R}^{n \times d}$: $\gamma, \beta \in \mathbb{R}^{d}$ (learnable per-feature parameters)

**Understanding the Greek letters:**
- $\mu$ (mu): The mean (average) of all the numbers
- $\sigma$ (sigma): The standard deviation - how spread out the numbers are from the average
- $\gamma$ (gamma): A learnable scale parameter - lets the model decide how much to amplify the result
- $\beta$ (beta): A learnable shift parameter - lets the model decide how much to shift the result
- $\epsilon$ (epsilon): A tiny number to prevent division by zero

**What LayerNorm does:** "Make all the numbers have zero average and unit variance, then let the model scale and shift them as needed." It's like standardizing test scores so they're all on the same scale.

where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$.

**Why LayerNorm for Sequences:** Unlike BatchNorm, it doesn't depend on batch statistics, making it suitable for variable-length sequences.

### 3.4 Alternative Normalization Methods

**RMSNorm (Root Mean Square Norm):** Simplifies LayerNorm by removing the mean:
```math
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
```

**Benefits:** Faster computation, similar performance to LayerNorm.

**Scaled Residuals:** In very deep networks, scale residual connections:
```math
\mathbf{h}_{l+1} = \mathbf{h}_l + \alpha \cdot F(\mathbf{h}_l)
```
where $\alpha < 1$ prevents residual explosion.

**Pre-LN vs Post-LN:**
- **Pre-LN (modern):** $\mathbf{h}_{l+1} = \mathbf{h}_l + F(\text{LN}(\mathbf{h}_l))$
- **Post-LN (original):** $\mathbf{h}_{l+1} = \text{LN}(\mathbf{h}_l + F(\mathbf{h}_l))$

**Pre-LN advantages:** Better gradient flow, more stable training, enables training deeper models.

## 4. High-Dimensional Geometry & Similarity

### 4.1 Distance Metrics in High Dimensions

**Euclidean Distance:** 
```math
d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2
```

**Cosine Similarity:** 
```math
\cos(\theta) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}
```

**Concentration of Measure:** In high dimensions, most random vectors are approximately orthogonal, making cosine similarity more discriminative than Euclidean distance.

### 4.2 Distance Calculations with Concrete Examples

Using simple example vectors to illustrate the concepts:
- "cat" vector: [0.8, 0.2, 0.1]
- "dog" vector: [0.7, 0.3, 0.2]

#### 4.2.1 Cosine Similarity

**Formula:** 
```math
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}
```

**Step-by-step calculation:**

1. **Dot product:** A·B = (0.8×0.7) + (0.2×0.3) + (0.1×0.2) = 0.56 + 0.06 + 0.02 = 0.64

2. **Magnitudes:**
   - ||A|| = √(0.8² + 0.2² + 0.1²) = √(0.64 + 0.04 + 0.01) = √0.69 ≈ 0.83
   - ||B|| = √(0.7² + 0.3² + 0.2²) = √(0.49 + 0.09 + 0.04) = √0.62 ≈ 0.79

3. **Cosine similarity:** 0.64 / (0.83 × 0.79) ≈ 0.64 / 0.66 ≈ **0.97**

**Interpretation:** Values range from -1 (opposite) to 1 (identical). 0.97 indicates high similarity - "cat" and "dog" point in nearly the same direction in semantic space.

#### 4.2.2 Euclidean Distance

**Formula:** 
```math
d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}
```

**Step-by-step calculation:**

1. **Differences:** (0.8-0.7)² + (0.2-0.3)² + (0.1-0.2)² = 0.01 + 0.01 + 0.01 = 0.03

2. **Distance:** √0.03 ≈ **0.17**

**Interpretation:** Lower values indicate closer proximity. 0.17 is small, confirming "cat" and "dog" are close in space.

**When to use each:**
- **Cosine:** When direction matters more than magnitude (text similarity, semantic relationships)
- **Euclidean:** When absolute distance matters (image features, exact matching)

### 4.3 Maximum Inner Product Search (MIPS)

**Problem:** Find 
```math
\mathbf{v}^* = \arg\max_{\mathbf{v} \in \mathcal{V}} \mathbf{q}^T \mathbf{v}
```

This is exactly what attention computes when finding relevant keys for a given query!

**Connection to Attention:** Query-key similarity in attention is inner product search over learned embeddings.

💻 **Implementation Example**: For high-dimensional similarity comparisons, see [Vectors & Geometry Notebook](./pynb/math_ref/vectors_geometry.ipynb)

## 5. From Similarity to Attention

📚 **Quick Reference**: See [Scaled Dot-Product Attention](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

### 5.1 Deriving Scaled Dot-Product Attention

**Step 1:** Start with similarity search between query $\mathbf{q}$ and keys $\{\mathbf{k}_i\}$:
```math
s_i = \mathbf{q}^T \mathbf{k}_i \quad (20)
```

**Step 2:** Convert similarities to weights via softmax:
```math
\alpha_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}} \quad (21)
```

**Understanding $\alpha$ (alpha):** These are the attention weights - they tell us "how much should I pay attention to each word?" The $\alpha_i$ values all add up to 1, like percentages.

**Why use softmax instead of just raw similarities?**
1. **Probabilities must sum to 1:** We want a weighted average, so weights must be between 0 and 1 and sum to 1. Raw similarities could be negative or not sum to 1.
2. **Differentiable selection:** Softmax provides a "soft" way to pick the most relevant items. Instead of hard selection (pick the best, ignore the rest), it gives more weight to better matches while still considering others.
3. **Handles different scales:** Raw similarity scores might vary wildly in range. Softmax normalizes them into a consistent 0-1 probability scale.
4. **Amplifies differences:** The exponential in softmax amplifies differences between scores. If one word is much more relevant, it gets much more attention.
5. **Smooth gradients:** Unlike hard max (which would create step functions), softmax is smooth everywhere, enabling gradient-based learning.

**Real-world analogy:** Like deciding how much attention to give each person in a room - you don't ignore everyone except one person (hard max), but you do focus more on the most interesting people while still being somewhat aware of others.

**Step 3:** Aggregate values using weights:
```math
\mathbf{o} = \sum_{i=1}^n \alpha_i \mathbf{v}_i \quad (22)
```

**What this step does:** Take a weighted average of all the value vectors. It's like saying "give me 30% of word 1's information, 50% of word 2's information, and 20% of word 3's information."

**Matrix Form:** For sequences, this becomes:
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad (23)
```

**Shape Analysis:** $Q \in \mathbb{R}^{n \times d_k}, K \in \mathbb{R}^{n \times d_k}, V \in \mathbb{R}^{n \times d_v} \Rightarrow \text{Output} \in \mathbb{R}^{n \times d_v}$

**What this equation accomplishes step-by-step:**
1. $QK^T$ creates a "compatibility matrix" - every query checks against every key
2. $\frac{1}{\sqrt{d_k}}$ scales the scores to prevent them from getting too large
3. $\text{softmax}$ converts raw compatibility scores into probability-like weights
4. Multiply by $V$ to get a weighted sum of value vectors

**Library analogy:** Think of attention like a smart librarian. Q is your question ("I need books about neural networks"), K represents the subject tags of all books, V contains the actual book contents. The librarian (attention mechanism) looks at your question, checks which books are most relevant (QK^T), decides how much attention to give each book (softmax), and gives you a summary that's mostly from the most relevant books but includes a little from others (weighted sum with V).

### 5.4 Masked Attention

**Causal Masking:** For autoregressive models, prevent attention to future tokens:
```math
P = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right) \quad (24)
```

where mask $M_{ij} \in \{0, -\infty\}$ with $M_{ij} = -\infty$ if $i < j$ (future positions).

**Numerical Stability:** Instead of $-\infty$, use large negative values (e.g., $-10^9$) to prevent NaN gradients:

💻 **Implementation Example**: For causal mask implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

**Padding Masks:** Mask out padding tokens by setting their attention scores to $-\infty$ before softmax. This ensures padding tokens receive zero attention weight.

**Key Insight:** Additive masking (adding $-\infty$) is preferred over multiplicative masking because it works naturally with softmax normalization.

### 5.2 Why the $\sqrt{d_k}$ Scaling?

**Variance Analysis:** If $Q, K$ have i.i.d. entries with variance $\sigma^2$, then:
```math
\text{Var}(QK^T) = d_k \sigma^4 \quad (24)
```

**Understanding $\sigma$ (sigma):** This represents the variance - how spread out the numbers are. Think of it as measuring "how noisy" or "how varied" the data is.

**What "i.i.d." means:** Independent and identically distributed - each number is random and doesn't depend on the others, like rolling dice multiple times.

Without scaling, attention weights become too peaked (sharp) as $d_k$ increases, leading to poor gradients and attention collapse.

**Pitfall:** Forgetting this scaling leads to attention collapse - weights become too concentrated rather than appropriately distributed.

### 5.3 Backpropagation Through Attention

**Softmax Gradient:** For $\mathbf{p} = \text{softmax}(\mathbf{z})$:
```math
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) \quad (25)
```

where $\delta_{ij}$ is the Kronecker delta.

**What is $\delta_{ij}$ (delta)?** It's a simple function that equals 1 when i=j (same position) and 0 otherwise. Think of it as asking "are these the same thing?" - if yes, return 1; if no, return 0.

**Attention Gradients:**

Let $S = QK^T/\sqrt{d_k}$, $P=\mathrm{softmax}(S)$ (row-wise), $O = PV$. Given $G_O=\partial \mathcal{L}/\partial O$:

```math
\begin{aligned}
G_V &= P^T G_O,\quad &&\text{(V same shape as }V\text{)}\\
G_P &= G_O V^T,\\
G_{S,r} &= \big(\mathrm{diag}(P_r) - P_r P_r^T\big)\, G_{P,r}\quad &&\text{(row }r\text{; softmax Jacobian)}\\
G_Q &= G_S K/\sqrt{d_k},\quad G_K = G_S^T Q/\sqrt{d_k}.
\end{aligned}
```

**Parameters:** $Q,K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$.
**Intuition:** Backprop splits into (i) linear parts, (ii) softmax Jacobian per row.

## 6. Multi-Head Attention & Positional Information

📚 **Quick Reference**: See [Multi-Head Attention](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) and [Positional Encoding](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

### 6.1 Multi-Head as Subspace Projections

**Single Head:** Projects to subspace of dimension $d_k = d_{\text{model}}/h$:
```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \quad (26)
```

**Multi-Head Combination:**
```math
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \quad (27)
```

**Why multiple heads instead of one big head?** Different heads can specialize in different types of relationships:
- **Head 1 might focus on syntax:** "What words are grammatically related?"
- **Head 2 might focus on semantics:** "What words are conceptually similar?"
- **Head 3 might focus on position:** "What words are nearby?"
- **Head 4 might focus on long-range dependencies:** "What words are related despite being far apart?"

**Meeting analogy:** Like having different experts in a meeting. Instead of one generalist trying to understand everything, you have specialists: a grammar expert, a meaning expert, a structure expert, etc. Each contributes their perspective, then all insights are combined (concatenated and projected through $W^O$).

**Why split the dimension?** Rather than having 8 heads each looking at 512-dimensional vectors, we have 8 heads each looking at 64-dimensional projections (512/8=64). This forces each head to focus on a specific subset of features, encouraging specialization.

**Implementation Details:**

**Linear Projections (Not MLPs):**
Each head uses simple linear transformations:
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ where $d_k = d_{\text{model}}/h$
- These are **single linear layers**, not multi-layer perceptrons
- Total projection parameters per layer: $3 \times h \times d_{\text{model}} \times d_k = 3d_{\text{model}}^2$

**Efficient Implementation:**
Instead of computing each head separately, implementations often:
1. Concatenate all head projections: $W^Q = [W_1^Q, W_2^Q, ..., W_h^Q] \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
2. Compute $Q = XW^Q, K = XW^K, V = XW^V$ in parallel
3. Reshape tensors to separate heads: $(n, d_{\text{model}}) \to (n, h, d_k)$
4. Apply attention computation across all heads simultaneously

**Parameter Analysis:**
- **Per-head projections**: $3 \times d_{\text{model}} \times d_k = 3 \times d_{\text{model}} \times (d_{\text{model}}/h)$
- **Total projections**: $3d_{\text{model}}^2$ (same as single-head with full dimension)
- **Output projection**: $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ adds $d_{\text{model}}^2$ parameters
- **Total multi-head parameters**: $4d_{\text{model}}^2$

### 6.2 Advanced Positional Encodings

**Sinusoidal Encoding:** Provides absolute position information:
```math
\begin{align}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d_{\text{model}}}) \quad (28)\\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d_{\text{model}}}) \quad (29)
\end{align}
```

**Mathematical Properties of Sinusoidal Encoding:**
- **Linearity**: For any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$
- **Relative position encoding**: The dot product $PE_{pos_i} \cdot PE_{pos_j}$ depends only on $|pos_i - pos_j|$
- **Extrapolation**: Can handle sequences longer than seen during training

**RoPE (Rotary Position Embedding):** Rotates query-key pairs by position-dependent angles:
```math
\begin{align}
\mathbf{q}_m^{(i)} &= R_{\Theta,m}^{(i)} \mathbf{q}^{(i)} \quad (30)\\
\mathbf{k}_n^{(i)} &= R_{\Theta,n}^{(i)} \mathbf{k}^{(i)} \quad (31)
\end{align}
```

**RoPE Rotation Matrix:**
```math
R_{\Theta,m}^{(i)} = \begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
```

where $\theta_i = 10000^{-2i/d_{\text{model}}}$ for dimension pairs.

**Key RoPE Properties:**
- **Relative position dependency**: $\mathbf{q}_m^T \mathbf{k}_n$ depends only on $(m-n)$, not absolute positions
- **Length preservation**: Rotation matrices preserve vector norms
- **Computational efficiency**: Can be implemented without explicit matrix multiplication
- **Long sequence generalization**: Better extrapolation to longer sequences than learned embeddings

**RoPE Implementation Insight:** Instead of rotating the full vectors, RoPE applies rotations to consecutive dimension pairs, enabling efficient implementation via element-wise operations.

💻 **Implementation Example**: For RoPE (Rotary Position Embedding) implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)
    

### 6.3 Alternative Position Encodings

**ALiBi (Attention with Linear Biases):** Adds position-dependent bias to attention scores:
```math
\text{ALiBi-Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{bias}_{ij}\right)V
```

where $\text{bias}_{ij} = -m \cdot |i - j|$ and $m$ is a head-specific slope.

**Benefits of ALiBi:**
- No position embeddings needed
- Excellent extrapolation to longer sequences
- Linear relationship between position distance and attention bias

**T5 Relative Position Bias:** Learns relative position embeddings:
```math
A_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}} + b_{\text{rel}(i,j)}
```

where $b_{\text{rel}(i,j)}$ is a learned bias based on relative distance $\text{rel}(i,j)$.

**RoPE Scaling Variants:**
- **Base scaling:** Increase base frequency: $\theta_i = \alpha^{-2i/d} \cdot 10000^{-2i/d}$
- **NTK scaling:** Interpolate frequencies for better long-context performance
- **When to use:** Base scaling for modest extensions (2-4x), NTK for extreme length

## 7. Transformer Block Mathematics

### 7.1 Complete Block Equations

**Pre-LayerNorm Architecture:**
```math
\begin{align}
\mathbf{h}_1 &= \text{LayerNorm}(\mathbf{x}) \quad (32)\\
\mathbf{h}_2 &= \mathbf{x} + \text{MultiHeadAttn}(\mathbf{h}_1, \mathbf{h}_1, \mathbf{h}_1) \quad (33)\\
\mathbf{h}_3 &= \text{LayerNorm}(\mathbf{h}_2) \quad (34)\\
\mathbf{y} &= \mathbf{h}_2 + \text{FFN}(\mathbf{h}_3) \quad (35)
\end{align}
```

**Feed-Forward Network:**
```math
\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2 \quad (36)
```

**Shape Analysis:** $\mathbf{x} \in \mathbb{R}^{n \times d_{\text{model}}}, W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}, W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$

**What the FFN does intuitively:** Think of it as a "thinking step" for each word individually. After attention has mixed information between words, the FFN lets each word "process" and transform its representation. It's like giving each word some individual processing time to digest the information it received from other words.

**Why expand then contract?** The FFN first expands the representation to a higher dimension (usually 4× larger), applies a nonlinearity, then contracts back down. This is like having a "working space" where the model can perform more complex computations before producing the final result.

**Shape Tracking:** For input $\mathbf{x} \in \mathbb{R}^{n \times d_{\text{model}}}$:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$ (typically $d_{\text{ffn}} = 4d_{\text{model}}$) - "expansion layer"
- $W_2 \in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}}$ - "contraction layer"

### 7.2 Why GELU over ReLU?

**GELU Definition:**
```math
\text{GELU}(x) = x \cdot \Phi(x) \quad (37)
```

**What GELU does intuitively:** GELU is like a "smooth switch." Unlike ReLU which harshly cuts off negative values to zero, GELU gradually transitions from "mostly off" to "mostly on." It asks "how much should I activate this neuron?" and gives a smooth answer between 0 and the input value.

**Why smoother is better:**
- **Better gradients:** ReLU has a sharp corner at zero (gradient jumps from 0 to 1). GELU is smooth everywhere, so gradients flow better during training.
- **Probabilistic interpretation:** GELU can be seen as randomly dropping out inputs based on their value - inputs closer to the mean of a normal distribution are more likely to "survive."
- **More expressive:** The smooth transition lets the model make more nuanced decisions about what information to keep.

**Real-world analogy:** ReLU is like a light switch (on/off), while GELU is like a dimmer switch (gradual control).

where $\Phi(x)$ is the standard normal CDF (cumulative distribution function - tells you the probability that a normal random variable is less than x). GELU provides smoother gradients than ReLU, improving optimization.

## 12. Training Objective & Tokenization/Embeddings

### 12.1 Next-Token Prediction

**Autoregressive Objective:**
```math
\mathcal{L} = -\sum_{t=1}^T \log P(x_t | x_1, ..., x_{t-1}) \quad (38)
```

**Shape Analysis:** For batch size $B$, sequence length $T$: loss computed per sequence, averaged over batch

**Implementation:** Use causal mask in attention to prevent information leakage from future tokens.

### 12.2 Embedding Mathematics

**Token Embeddings:** Map discrete tokens to continuous vectors:
```math
\mathbf{e}_i = E[i] \in \mathbb{R}^{d_{\text{model}}} \quad (39)
```

where $E \in \mathbb{R}^{V \times d_{\text{model}}}$ is the embedding matrix.

**Weight Tying:** Share embedding matrix $E$ with output projection to reduce parameters:
```math
P(w_t | \text{context}) = \text{softmax}(\mathbf{h}_t E^T) \quad (40)
```

**Shape Analysis:** For $\mathbf{h}_t \in \mathbb{R}^{1 \times d_{\text{model}}}$ and $E \in \mathbb{R}^{V \times d_{\text{model}}}$:
- $\mathbf{h}_t E^T \in \mathbb{R}^{1 \times V}$ (logits over vocabulary)
- Consistent with row-vector convention

**Perplexity:** Measures model uncertainty:
```math
\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})\right) \quad (41)
```

## 10. Efficient Attention & Scaling

### 10.1 Complexity Analysis

**Standard Attention Complexity:**
- Time: $O(n^2 d)$ for sequence length $n$, model dimension $d$
- Space: $O(n^2 + nd)$ for attention matrix and activations

**Memory Bottleneck:** Attention matrix $A \in \mathbb{R}^{n \times n}$ dominates memory usage for long sequences.

**Detailed Complexity Breakdown:**
1. **QK^T computation**: $O(n^2 d)$ time, $O(n^2)$ space
2. **Softmax normalization**: $O(n^2)$ time and space
3. **Attention-Value multiplication**: $O(n^2 d)$ time, $O(nd)$ space
4. **Total**: $O(n^2 d)$ time, $O(n^2 + nd)$ space

**Scaling Challenges:**
- Quadratic scaling limits practical sequence lengths
- Memory requirements grow quadratically with sequence length
- Computational cost increases quadratically even with parallelization

### 10.2 FlashAttention: Memory-Efficient Attention

**Core Idea:** Compute attention without materializing the full $n \times n$ attention matrix.

**Tiling Strategy:**
1. Divide $Q$, $K$, $V$ into blocks
2. Compute attention scores block by block
3. Use online softmax to maintain numerical stability
4. Accumulate results without storing intermediate attention weights

**Memory Reduction:** From $O(n^2)$ to $O(n)$ memory complexity for the attention computation.

**Speed Improvement:** Better GPU utilization through reduced memory bandwidth requirements.

**Key Insight:** Trade computational redundancy for memory efficiency - recompute rather than store.

### 10.3 Multi-Query and Grouped-Query Attention

**Multi-Query Attention (MQA):** Share key and value projections across heads:
- Queries: $Q \in \mathbb{R}^{B \times H \times n \times d_k}$ (per-head)
- Keys/Values: $K, V \in \mathbb{R}^{B \times 1 \times n \times d_k}$ (shared)

**Grouped-Query Attention (GQA):** Intermediate approach - group heads:
- Divide $H$ heads into $G$ groups
- Each group shares K, V projections
- Reduces KV cache size by factor $H/G$

**KV Cache Memory Analysis:**
- **Standard MHA:** $2 \cdot B \cdot H \cdot n \cdot d_k$ parameters
- **MQA:** $2 \cdot B \cdot 1 \cdot n \cdot d_k$ parameters (H× reduction)
- **GQA:** $2 \cdot B \cdot G \cdot n \cdot d_k$ parameters

**Quantization:** Reduce memory further with int8/fp16 KV cache storage.

### 10.4 KV Caching for Autoregressive Generation

**Key Insight:** During generation, keys and values for previous tokens don't change.

**Cache Update:**

```math
K_{\text{cache}} \gets \mathrm{concat}(K_{\text{cache}},\ k_{\text{new}}) \tag{42}
```

- **$K_{\text{cache}}$**: Cached keys from previous tokens.
- **$V_{\text{cache}}$**: Cached values from previous tokens.
- **$k_{\text{new}}, v_{\text{new}}$**: Key and value for the new token.
- **$q_{\text{new}}$**: Query for the new token.

At each generation step, append the new key and value to the cache, then compute attention using the full cache.

**Memory Trade-off:** Cache size grows as $O(nd)$ but eliminates $O(n^2)$ recomputation.

💻 **Implementation Example**: For KV Cache implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

### 10.5 Linear Attention Approximations

**Kernel Method View:** Approximate $\text{softmax}(\mathbf{q}^T\mathbf{k})$ with $\phi(\mathbf{q})^T \phi(\mathbf{k})$ for feature map $\phi$.

**Linear Attention:**
```math
\text{LinAttn}(Q,K,V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})} \quad (45)
```

**Complexity Reduction:** Reduces from $O(n^2 d)$ to $O(nd^2)$ when $d < n$.

## 11. Regularization, Generalization, and Calibration

### 11.1 Dropout in Transformers

**Attention Dropout:** Applied to attention weights:
```math
A_{\text{dropped}} = \text{Dropout}(\text{softmax}(QK^T/\sqrt{d_k})) \quad (46)
```

**FFN Dropout:** Applied after first linear transformation:
```math
\text{FFN}(\mathbf{x}) = W_2 \cdot \text{Dropout}(\text{GELU}(W_1 \mathbf{x})) \quad (47)
```

### 11.2 Evaluation and Calibration

**Expected Calibration Error (ECE):** Measures how well predicted probabilities match actual outcomes:
```math
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|
```
where $B_m$ are probability bins, $\text{acc}$ is accuracy, $\text{conf}$ is confidence.

**Temperature Scaling:** Post-training calibration method:
```math
P_{\text{cal}}(y|x) = \text{softmax}(\mathbf{z}/T)
```
where $T > 1$ makes predictions less confident, $T < 1$ more confident.

**Perplexity Dependence on Tokenizer:** PPL comparisons only valid with same tokenizer. Different tokenizers create different sequence lengths and vocabulary sizes.

**Example:** "hello world" might be:
- GPT tokenizer: ["hel", "lo", " wor", "ld"] (4 tokens)
- Character-level: ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"] (11 tokens)

### 11.3 Advanced Tokenization

**Byte-Level BPE vs Unigram:**
- **BPE:** Greedily merges frequent character pairs, handles any Unicode
- **Unigram:** Probabilistic model, often better for morphologically rich languages

**Special Token Handling:**
- **BOS (Beginning of Sequence):** Often used for unconditional generation
- **EOS (End of Sequence):** Signals completion, crucial for proper training
- **PAD:** For batching variable-length sequences

**Embedding/LM-Head Tying Caveats:**
When sharing weights, ensure shape compatibility:
- Embedding: $E \in \mathbb{R}^{V \times d_{\text{model}}}$
- LM head: needs $\mathbb{R}^{d_{\text{model}} \times V}$
- Solution: Use $E^T$ for output projection (as shown in equation 40)

### 11.4 Label Smoothing

**Smooth Labels:** Replace one-hot targets with:
```math
y_{\text{smooth}} = (1-\alpha) y_{\text{true}} + \frac{\alpha}{V} \mathbf{1} \quad (48)
```

**Effect on Gradients:** Prevents overconfident predictions and improves calibration.

## 8. Practical Numerics & Implementation Notes

### 8.1 Initialization Strategies

**Xavier/Glorot for Linear Layers:**
```math
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad (49)
```

**Attention-Specific:** Initialize query/key projections with smaller variance to prevent attention collapse (overly peaked attention distributions).

### 8.2 Mixed Precision Training

**FP16 Forward, FP32 Gradients:** Use half precision for speed, full precision for numerical stability:
💻 **Implementation Example**: For Automatic Mixed Precision implementation, see [Optimization Notebook](./pynb/math_ref/optimization.ipynb)

### 8.3 Gradient Clipping

**Global Norm Clipping:** As detailed in equation (11), we clip gradients to prevent explosive updates.

## 13. Worked Mini-Examples

### 13.1 Tiny Attention Forward Pass

**Setup:** $n=2$ tokens, $d_k=d_v=3$, single head.

**Input:**
```
Q = [[1, 0, 1],    K = [[1, 1, 0],    V = [[2, 0, 1],
     [0, 1, 1]]         [1, 0, 1]]         [1, 1, 0]]
```

**Step 1:** Compute raw scores $QK^T$:
```math
QK^T = \begin{bmatrix}1 & 0 & 1\\0 & 1 & 1\end{bmatrix} \begin{bmatrix}1 & 1\\1 & 0\\0 & 1\end{bmatrix} = \begin{bmatrix}1 & 2\\1 & 1\end{bmatrix}
```

**Step 2:** Scale by $1/\sqrt{d_k} = 1/\sqrt{3} \approx 0.577$:
```math
S = \frac{QK^T}{\sqrt{3}} = \begin{bmatrix}0.577 & 1.155\\0.577 & 0.577\end{bmatrix}
```

**Step 3:** Apply softmax (row-wise, rounded to 3 d.p.):
- Row 1: $e^{0.577} = 1.781, e^{1.155} = 3.173$, sum $= 4.954$
- Row 2: $e^{0.577} = 1.781, e^{0.577} = 1.781$, sum $= 3.562$

```math
A = \begin{bmatrix}
0.359 & 0.641 \\
0.500 & 0.500
\end{bmatrix}
```

**Step 4:** Compute output $O = AV$:
```math
O = \begin{bmatrix}0.359 & 0.641\\0.500 & 0.500\end{bmatrix} \begin{bmatrix}2 & 0 & 1\\1 & 1 & 0\end{bmatrix} = \begin{bmatrix}1.359 & 0.641 & 0.359\\1.500 & 0.500 & 0.500\end{bmatrix}
```

💻 **Implementation Example**: For attention computation verification, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

### 13.2 Backprop Through Simple Attention

**Given:** 
```math
\frac{\partial \mathcal{L}}{\partial O} = \begin{bmatrix}1 & 0 & 1\\0 & 1 & 0\end{bmatrix}
```

**Gradient w.r.t. Values:**
```math
\frac{\partial \mathcal{L}}{\partial V} = A^T \frac{\partial \mathcal{L}}{\partial O} = \begin{bmatrix}0.359 & 0.500\\0.641 & 0.500\end{bmatrix}\begin{bmatrix}1 & 0 & 1\\0 & 1 & 0\end{bmatrix} = \begin{bmatrix}0.359 & 0.500 & 0.359\\0.641 & 0.500 & 0.641\end{bmatrix}
```

💻 **Implementation Example**: For gradient verification using finite differences, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

**Check for Understanding:** Verify that gradient shapes match parameter shapes and that the chain rule is applied correctly.

## 14. Common Pitfalls & Misconceptions

### 14.1 High-Dimensional Distance Misconceptions

**Pitfall:** Using Euclidean distance instead of cosine similarity in high dimensions.
**Fix:** In $d > 100$, most vectors are approximately orthogonal, making cosine similarity more discriminative.

### 14.2 Attention Scaling Mistakes

**Pitfall:** Forgetting $1/\sqrt{d_k}$ scaling or using wrong dimension.
**Symptom:** Attention weights become too peaked, leading to poor gradients.
**Fix:** Always scale by $\sqrt{d_k}$ where $d_k$ is the key dimension. Note that $d_k=d_{\text{model}}/h$ under common implementations.

### 14.3 LayerNorm Placement

**Pitfall:** Using post-LayerNorm (original) instead of pre-LayerNorm (modern).
**Issue:** Post-LN can lead to training instability in deep models.
**Modern Practice:** Apply LayerNorm before attention and FFN blocks.

### 14.4 Softmax Temperature Misuse

**Pitfall:** Applying temperature scaling inconsistently.
**Correct Usage:** Temperature $\tau$ in $\text{softmax}(\mathbf{z}/\tau)$ controls sharpness:
- $\tau > 1$: Smoother distribution
- $\tau < 1$: Sharper distribution

## 15. Summary & What to Learn Next

### 15.1 Key Mathematical Insights

1. **Attention as Similarity Search:** Q/K/V framework emerges naturally from maximum inner product search
2. **Scaling Laws:** $1/\sqrt{d_k}$ scaling prevents attention collapse (overly peaked distributions) in high dimensions  
3. **Residual Connections:** Enable gradient flow through deep networks via skip connections
4. **Multi-Head Architecture:** Parallel subspace projections enable diverse attention patterns

### 15.2 Next Steps

**Scaling Laws:** Study how performance scales with model size, data, and compute (Kaplan et al., 2020)

**Parameter-Efficient Fine-Tuning:** LoRA, adapters, and other methods for efficient adaptation

**Retrieval-Augmented Models:** Combining parametric knowledge with external memory

**Advanced Architectures:** Mixture of Experts, sparse attention patterns, and alternative architectures

---

## Further Reading

**Core Papers:**
[1] Vaswani, A., et al. "Attention is all you need." *Advances in Neural Information Processing Systems*, 2017.
[2] Devlin, J., et al. "BERT: Pre-training of deep bidirectional transformers for language understanding." *NAACL-HLT*, 2019.
[3] Brown, T., et al. "Language models are few-shot learners." *Advances in Neural Information Processing Systems*, 2020.

**Mathematical Foundations:**
[4] Kaplan, J., et al. "Scaling laws for neural language models." *arXiv preprint arXiv:2001.08361*, 2020.
[5] Su, J., et al. "RoFormer: Enhanced transformer with rotary position embedding." *arXiv preprint arXiv:2104.09864*, 2021.

**Efficiency & Scaling:**
[6] Dao, T., et al. "FlashAttention: Fast and memory-efficient exact attention with IO-awareness." *Advances in Neural Information Processing Systems*, 2022.
[7] Shazeer, N. "Fast transformer decoding: One write-head is all you need." *arXiv preprint arXiv:1911.02150*, 2019.

**Training & Optimization:**
[8] Loshchilov, I., & Hutter, F. "Decoupled weight decay regularization." *ICLR*, 2019.
[9] Xiong, R., et al. "On layer normalization in the transformer architecture." *ICML*, 2020.
[10] Press, O., & Wolf, L. "Using the output embedding to improve language models." *EACL*, 2017.

---

## Appendix A: Symbol/Shape Reference

### Single-Head Attention Shapes
| Symbol | Meaning | Typical Shape |
|--------|---------|---------------|
| $Q, K, V$ | Query, Key, Value matrices | $[n \times d_k], [n \times d_k], [n \times d_v]$ |
| $n$ | Sequence length | Scalar |
| $d_{\text{model}}$ | Model dimension | Scalar (512, 768, 1024, etc.) |
| $d_k, d_v$ | Key, value dimensions | Usually $d_{\text{model}}/h$ |
| $h$ | Number of attention heads | Scalar (8, 12, 16, etc.) |

### Multi-Head & Batched Shapes
| Symbol | Meaning | Batched Multi-Head Shape |
|--------|---------|-------------------------|
| $Q, K, V$ | Projected queries, keys, values | $[B, H, n, d_k], [B, H, n, d_k], [B, H, n, d_v]$ |
| $A$ | Attention weights matrix | $[B, H, n, n]$ |
| $O$ | Attention output (pre-concat) | $[B, H, n, d_v]$ |
| $O_{\text{proj}}$ | Final output (post-concat) | $[B, n, d_{\text{model}}]$ |
| $W^Q, W^K, W^V$ | Attention projection matrices | $[d_{\text{model}} \times d_k]$ per head |
| $W^O$ | Output projection | $[d_{\text{model}} \times d_{\text{model}}]$ |

**Convention:** $B$ = batch size, $H$ = number of heads, $n$ = sequence length, $d_k = d_v = d_{\text{model}}/H$

## Appendix B: Key Derivations

### B.1 Softmax Gradient

For $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$:

```math
\frac{\partial p_i}{\partial z_j} = \begin{cases}
p_i(1 - p_i) & \text{if } i = j \\
-p_i p_j & \text{if } i \neq j
\end{cases} = p_i(\delta_{ij} - p_j)
```

### B.2 Matrix Calculus Identities

**Trace-Vec Identity:** $\text{tr}(AB) = \text{vec}(A^T)^T \text{vec}(B)$

**Kronecker Product:** $\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$

**Chain Rule for Matrices:** $\frac{\partial f}{\partial X} = \sum_Y \frac{\partial f}{\partial Y} \frac{\partial Y}{\partial X}$

## Appendix C: Glossary

**Attention Collapse:** Phenomenon where attention weights become too peaked (concentrated on few tokens) rather than uniform, leading to poor gradient flow and reduced model expressiveness.

**Causal Mask:** Lower-triangular mask preventing attention to future tokens in autoregressive models.

**KV Cache:** Stored key-value pairs from previous tokens to accelerate autoregressive generation.

**Multi-Head Attention:** Parallel attention mechanisms operating on different learned subspaces.

**Position Encoding:** Method to inject sequential order information into permutation-equivariant attention.

**Scaled Dot-Product Attention:** Core attention mechanism using $\text{softmax}(QK^T/\sqrt{d_k})V$.

**Teacher Forcing:** Training technique using ground truth tokens as inputs instead of model predictions.
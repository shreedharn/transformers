# The Mathematics of Transformers: From First Principles to Practice
## Part 1: Building Intuition and Core Concepts

## Abstract

This tutorial builds the mathematical foundations of Transformer architectures from first principles, targeting motivated high school students with basic algebra and geometry background. This first part focuses on building intuition, covering linear algebra basics, simple networks, and the step-by-step path to attention and the Transformer block. We progress systematically from optimization theory and high-dimensional geometry through attention mechanisms to complete Transformer blocks, emphasizing mathematical intuition, worked derivations, and practical implementation considerations. Every mathematical concept is explained with real-world analogies and intuitive reasoning before diving into the formal mathematics.

For advanced topics including optimization, training stability, scaling laws, and implementation details for real-world large models, see [Part 2: Advanced Concepts and Scaling](./transformers_math2.md).

## Assumptions & Conventions

Mathematical Notation:

- Vectors are row-major; matrices multiply on the right
- Shapes annotated as [seq, dim] or [batch, seq, heads, dim]
- Masking uses additive large negative values (-∞) before softmax
- Row-wise softmax normalization
- Token/position numbering starts from 0
- Default dtype is fp32 unless specified
- Equations numbered sequentially throughout document

For Advanced Topics:

- [Part 2: Advanced Concepts and Scaling](./transformers_math2.md) covers optimization, efficient attention, regularization, and implementation details

Additional Resources:

- [Glossary](./glossary.md) - Comprehensive terms and definitions

## 1. Roadmap

We begin with optimization fundamentals and high-dimensional geometry, then build attention as a principled similarity search mechanism. The journey: **gradients → similarity metrics → attention → multi-head attention → full Transformer blocks**. Each step connects mathematical theory to practical implementation, culminating in a complete understanding of how Transformers process sequences through learned representations and attention-based information routing.

For the complete journey including **efficient inference**, **scaling laws**, and **advanced optimization techniques**, continue to [Part 2](./transformers_math2.md).

## 2. Neural Network Training: Mathematical Foundations

📚 Quick Reference: For pure mathematical concepts and formulas, see [Mathematical Quick Reference](./math_quick_ref.md). This section focuses on mathematical concepts **in the context of deep learning**.

### 2.1 From Training to Inference: The Complete Journey

#### 2.1.1 From Line Slopes to Neural Network Training

Building Intuition: Slope of a Line

Let's start with something familiar - the equation of a straight line:
$$
{\textstyle
\begin{aligned}
y = mx + b
\end{aligned}
}
$$

where:

- m is the slope - tells us how steep the line is
- b is the y-intercept - where the line crosses the y-axis

What does slope mean intuitively? Slope tells us "for every step I move right, how much do I move up or down?" If the slope is 2, then moving 1 step right means moving 2 steps up. If the slope is -0.5, then moving 1 step right means moving 0.5 steps down.

Connecting to Optimization: In machine learning, we want to find the "bottom of a valley" - the point where our error is smallest. To do this, we need to know which direction is "downhill."

#### From 2D Slopes to Gradient Descent

The Derivative as Slope: 

$$
{\textstyle
\begin{aligned}
\text{For any function } f(x), \text{ the derivative } \frac{df}{dx} \text{ tells us the slope at any point:} \newline
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
\end{aligned}
}
$$

What this equation means: "If I move a tiny amount h to the right, how much does f change?" The derivative is the limit as that tiny amount approaches zero.

Gradient Descent in One Dimension: To minimize a function, we update our position using:

$$
{\textstyle
\begin{aligned}
f(x) &\quad \text{(function to minimize)} \newline
x_{\text{new}} &= x_{\text{old}} - \alpha \frac{df}{dx} \newline
\alpha &\quad \text{(learning rate - controls step size)}
\end{aligned}
}
$$

The key insight: The negative sign makes us go *opposite* to the slope direction. If the slope is positive (going uphill to the right), we move left. If the slope is negative (going downhill to the right), we move right.

#### Worked Example: f(x) = x²

Let's minimize this function with the following setup:

$$
\begin{aligned}
f(x) &= x^2 \quad \text{(function to minimize)} \newline
x_0 &= 3 \quad \text{(starting point)} \newline
\frac{df}{dx} &= 2x \quad \text{(derivative)} \newline
\alpha &= 0.1 \quad \text{(learning rate)}
\end{aligned}
$$

Step-by-step updates:

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

What's happening: We're taking steps toward x = 0 (the minimum of x²), and each step gets smaller as we approach the bottom.

#### Extending to Multiple Variables: Gradients as Vectors

From Derivatives to Gradients:

When we have multiple variables, we need partial derivatives:

$$
{\textstyle
\begin{aligned}
\frac{\partial f}{\partial x} = 2x \quad &\text{(rate of change with respect to x, holding y fixed)} \newline
\frac{\partial f}{\partial y} = 2y \quad &\text{(rate of change with respect to y, holding x fixed)}
\end{aligned}
}
$$

The Gradient Vector: We combine these into a gradient vector:

$$
{\textstyle
\begin{aligned}
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
\end{aligned}
}
$$

Vector Gradient Descent: Now our update rule becomes:

$$
{\textstyle
\begin{aligned}
\begin{bmatrix} x_{\text{new}} \\ y_{\text{new}} \end{bmatrix} = \begin{bmatrix} x_{\text{old}} \\ y_{\text{old}} \end{bmatrix} - \alpha \begin{bmatrix} 2x_{\text{old}} \\ 2y_{\text{old}} \end{bmatrix}
\end{aligned}
}
$$

Intuition: The gradient vector points in the direction of steepest *ascent*. By moving in the *opposite* direction (negative gradient), we go downhill most quickly.

#### Matrix Form for Machine Learning

Setting up the Problem: In machine learning, we have:

- X: Input matrix with shape (samples × features) - each row is one data point
- W: Weight matrix with shape (features × outputs) - the parameters we want to learn  
- b: Bias vector with shape (outputs,) - additional adjustable parameters
- Y_hat: Predictions with shape (samples × outputs), computed as Y_hat = XW + b

Loss Function: We measure how wrong our predictions are using mean squared error:

$$
{\textstyle
\begin{aligned}
L = \frac{1}{N} \sum_{i=1}^N \|Y_{\text{true}}^{(i)} - Y_{\text{hat}}^{(i)}\|^2
\end{aligned}
}
$$

Matrix Gradients: To minimize the loss, we need gradients with respect to W and b:

$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial W} = \frac{2}{N} X^T (Y_{\text{hat}} - Y_{\text{true}})
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i=1}^N (Y_{\text{hat}}^{(i)} - Y_{\text{true}}^{(i)})
\end{aligned}
}
$$

Matrix Gradient Descent Updates:

$$
{\textstyle
\begin{aligned}
W_{\text{new}} = W_{\text{old}} - \alpha \frac{\partial L}{\partial W}
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
b_{\text{new}} = b_{\text{old}} - \alpha \frac{\partial L}{\partial b}
\end{aligned}
}
$$

Key Insight: The learning rate plays the same role across all parameters:

$$
\begin{aligned}
\alpha \quad \text{(controls step size for all parameters, just like in 1D case)}
\end{aligned}
$$

#### Single Hidden Layer MLP: Putting It All Together

Forward Pass Equations:

$$
{\textstyle
\begin{aligned}
Z_1 = X W_1 + b_1 \quad \text{(shape: samples x hidden units)}
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
A_1 = \sigma(Z_1) \quad \text{(apply activation function element-wise)}
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
Z_2 = A_1 W_2 + b_2 \quad \text{(shape: samples x outputs)}
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
Y_{\text{hat}} = Z_2 \quad \text{(final predictions)}
\end{aligned}
}
$$

with the activation function defined as:

$$
{\textstyle
\begin{aligned}
\sigma \quad \text{(activation function like ReLU or sigmoid)}
\end{aligned}
}
$$

Backward Pass (Backpropagation):

Step 1: Compute the error at the output:
$$
{\textstyle
\begin{aligned}
\delta_2 = Y_{\text{hat}} - Y_{\text{true}} \quad \text{(shape: samples × outputs)}
\end{aligned}
}
$$

Step 2: Compute gradients for output layer:
$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial W_2} = \frac{1}{N} A_1^T \delta_2
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial b_2} = \frac{1}{N} \sum_{i=1}^N \delta_2^{(i)}
\end{aligned}
}
$$

Step 3: Backpropagate error to hidden layer:
$$
{\textstyle
\begin{aligned}
\delta_1 = (\delta_2 W_2^T) \odot \sigma'(Z_1) \quad \text{(element-wise multiplication)}
\end{aligned}
}
$$

Step 4: Compute gradients for hidden layer:
$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial W_1} = \frac{1}{N} X^T \delta_1
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
\frac{\partial L}{\partial b_1} = \frac{1}{N} \sum_{i=1}^N \delta_1^{(i)}
\end{aligned}
}
$$

Step 5: Update all parameters using the same learning rate:
$$
{\textstyle
\begin{aligned}
W_1 \leftarrow W_1 - \alpha \frac{\partial L}{\partial W_1}
\end{aligned}
}
$$

$$
{\textstyle
\begin{aligned}
b_1 \leftarrow b_1 - \alpha \frac{\partial L}{\partial b_1}
\end{aligned}
}
$$
{\textstyle
\begin{aligned}
W_2 \leftarrow W_2 - \alpha \frac{\partial L}{\partial W_2}
\end{aligned}
}
$$
{\textstyle
\begin{aligned}
b_2 \leftarrow b_2 - \alpha \frac{\partial L}{\partial b_2}
\end{aligned}
}
$$

Understanding the δ terms:

$$
{\textstyle
\begin{aligned}
\delta_2 \quad &: \text{"How much does changing each output neuron's value affect the loss?"} \newline
\delta_1 \quad &: \text{"How much does changing each hidden neuron's value affect the loss?"}
\end{aligned}
}
$$

The error terms flow backwards through the network, carrying error information from the output back to earlier layers:

$$
\begin{aligned}
\delta \quad \text{(error gradient terms)}
\end{aligned}
$$

#### The Learning Rate α: Universal Step Size Controller

Same Role Everywhere:

Notice that α plays the identical role in:

$$
{\textstyle
\begin{aligned}
\text{1D gradient descent:} \quad &x \leftarrow x - \alpha \frac{df}{dx} \newline
\text{Vector gradient descent:} \quad &\mathbf{x} \leftarrow \mathbf{x} - \alpha \nabla f \newline
\text{Matrix gradient descent:} \quad &W \leftarrow W - \alpha \frac{\partial L}{\partial W} \newline
\text{Neural network training:} \quad &\text{All parameters use the same } \alpha
\end{aligned}
}
$$

Choosing α:

- Too large: Updates overshoot the minimum, causing oscillation or divergence
- Too small: Updates are tiny, causing very slow convergence
- Just right: Steady progress toward the minimum without overshooting

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

The Big Picture: Whether we're finding the bottom of a simple parabola or training a neural network with millions of parameters, we're doing the same fundamental thing:

1. Measure the slope (derivative, gradient, or backpropagated error)
2. Take a step in the opposite direction (negative sign)
3. Control step size (learning rate α)
4. Repeat until we reach the bottom

This is why understanding the simple case of line slopes gives us insight into the most sophisticated neural network training algorithms.

#### 2.1.2 Gradient Fields and Optimization

Gradient Descent as Continuous Flow: Parameter updates approximate the ODE:

$$
\begin{aligned}
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L} \quad \text{(discrete update rule)}
\end{aligned}
$$

This approximates:
$$
{\textstyle
\begin{aligned}
\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta) \quad (3)
\end{aligned}
}
$$

Understanding the symbols:

$$
{\textstyle
\begin{aligned}
\theta \quad &\text{(theta): The parameters we want to learn - think of these as the "knobs" we can adjust} \newline
\eta \quad &\text{(eta): Learning rate - how big steps we take. Like deciding whether to take baby steps or giant leaps when hiking downhill} \newline
\nabla \quad &\text{(nabla): The gradient symbol - points in the direction of steepest ascent (we go opposite direction to descend)} \newline
\mathcal{L} \quad &\text{(script L): The loss function - measures how "wrong" our current parameters are}
\end{aligned}
}
$$

What the equation means: "Change the parameters in the opposite direction of the gradient, scaled by the learning rate."

This connects discrete optimization to continuous dynamical systems.

Why This Matters: Understanding optimization as flow helps explain momentum methods, learning rate schedules, and convergence behavior.

#### 2.1.3 Residual Connections as Discretized Dynamics

Residual Block approximates the continuous dynamics:

$$
\begin{aligned}
\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l) \quad \text{(discrete residual connection)}
\end{aligned}
$$

This approximates:
$$
{\textstyle
\begin{aligned}
\frac{d\mathbf{h}}{dt} = F(\mathbf{h}) \quad (4)
\end{aligned}
}
$$

What residual connections do intuitively: Think of them as "safety nets" for information. Without residual connections, information would have to successfully pass through every layer to reach the output. With residual connections, information can "skip over" layers that might be learning slowly or poorly.

Highway analogy: Imagine driving from city A to city B. Without residual connections, you MUST go through every small town along the way. With residual connections, there's a highway that bypasses some towns - you still visit some towns (transformation), but you're guaranteed to make progress toward your destination even if some towns are roadblocked.

Why this enables deep networks: In very deep networks (50+ layers), gradients tend to vanish as they backpropagate. Residual connections provide a "gradient highway" - gradients can flow directly backward through the skip connections, ensuring that even early layers receive useful training signals.

This enables training very deep networks by maintaining gradient flow.

Stability Consideration: The transformation should be well-conditioned to avoid exploding/vanishing gradients:

$$
\begin{aligned}
F \quad \text{(transformation function with bounded gradients)}
\end{aligned}
$$

💻 Implementation Example: For a practical implementation of residual connections, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

### 2.2 Deep Learning Mathematics in Context

#### 2.2.1 Vectors as Word Meanings

What vectors represent in transformers: Vectors are not just mathematical objects—they encode semantic meaning. Each word becomes a point in high-dimensional space where:

- Similar words cluster together: "king" and "queen" vectors point in similar directions
- Vector arithmetic captures relationships: "king" - "man" + "woman" ≈ "queen"
- Distance measures semantic similarity: Cosine similarity between "cat" and "dog" is higher than between "cat" and "airplane"

Why this matters for attention: When transformers compute attention, they're asking "which word meanings are most relevant to understanding this context?" This is fundamentally a similarity search in semantic space.

#### 2.2.2 Matrices as Transformations of Meaning

Linear transformations in neural networks:

$$
{\textstyle
\begin{aligned}
\text{Weight matrices } W \text{ transform input meanings:} \quad &\mathbf{h}_{\text{new}} = \mathbf{h}_{\text{old}} W \newline
\text{Multiple transformations compose:} \quad &\mathbf{h}_3 = \mathbf{h}_1 W_1 W_2 \text{ applies two sequential meaning transformations} \newline
\text{Transpose operations } W^T \text{ reverse transformations during backpropagation}
\end{aligned}
}
$$

Block matrix operations enable parallel processing:
$$
{\textstyle
\begin{aligned}
\begin{bmatrix} \mathbf{h}_1 \\ \mathbf{h}_2 \\ \vdots \\ \mathbf{h}_n \end{bmatrix} W = \begin{bmatrix} \mathbf{h}_1 W \\ \mathbf{h}_2 W \\ \vdots \\ \mathbf{h}_n W \end{bmatrix}
\end{aligned}
}
$$
This processes entire sequences simultaneously instead of word-by-word.

#### 2.2.3 Gradients as Learning Signals

What gradients mean in neural networks: Gradients tell us "if I adjust this parameter slightly, how much will my prediction error change?" This guides learning:

- Large gradients: Parameter strongly affects error → make bigger adjustments
- Small gradients: Parameter weakly affects error → make smaller adjustments
- Zero gradients: Parameter doesn't affect error → don't change it

Chain rule enables credit assignment: In deep networks, we need to know how output errors relate to early layer parameters. The chain rule flows error signals backward through the network:
$$
{\textstyle
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_3} \frac{\partial \mathbf{h}_3}{\partial \mathbf{h}_2} \frac{\partial \mathbf{h}_2}{\partial W_1}
\end{aligned}
}
$$

#### 2.2.4 Softmax and Cross-Entropy: From Scores to Decisions

Softmax converts neural network outputs to probabilities:
$$
{\textstyle
\begin{aligned}
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} \quad (1)
\end{aligned}
}
$$

Why transformers use this combination:

1. Neural networks output raw scores (logits) that can be any real number
2. Softmax normalizes these into probabilities that sum to 1
3. Cross-entropy loss measures prediction quality using these probabilities

Cross-Entropy Loss:
$$
{\textstyle
\begin{aligned}
\mathcal{L} = -\sum_{i=1}^n y_i \log p_i \quad (2)
\end{aligned}
}
$$

Why cross-entropy is perfect for language modeling:

- Encourages confident correct predictions: Low loss when p_i ≈ 1 for correct answer
- Harshly penalizes confident wrong predictions: High loss when p_i ≈ 0 for correct answer
- Matches softmax naturally: Both work with probability distributions

Provides clean gradients:

$$
\begin{aligned}
\nabla \mathcal{L} = \mathbf{p} - \mathbf{y} \quad \text{(predicted - true)}
\end{aligned}
$$

Example: If the model predicts 90% probability for the correct next word, loss is low. If it predicts 1% probability for the correct next word, loss is very high.

## 3. Multilayer Perceptrons as a Warm-Up

### 3.1 Forward Pass

Two-layer MLP:

Shape Analysis:

For input dimensions:

$$
\begin{aligned}
\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{in}}}
\end{aligned}
$$

The weight matrices have shapes:

$$
{\textstyle
\begin{aligned}
W^{(1)} &\in \mathbb{R}^{d_{\text{in}} \times d_{\text{hidden}}} \newline
W^{(2)} &\in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{out}}}
\end{aligned}
}
$$

### 3.2 Backpropagation Derivation

Loss Gradient w.r.t. Output:
$$
{\textstyle
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \quad (16)
\end{aligned}
}
$$

Weight Gradients:

$$
{\textstyle
\begin{aligned}
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{(2)}} &= (\mathbf{h}^{(1)})^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \\
\frac{\partial \mathcal{L}}{\partial W^{(1)}} &= \mathbf{x}^T \left[ \left( \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} W^{(2)T} \right) \odot \sigma'(\mathbf{z}^{(1)}) \right]
\end{aligned}
\end{aligned}
}
$$

with the operator defined as:

$$
\begin{aligned}
\odot \quad \text{(element-wise multiplication)}
\end{aligned}
$$

Additional notation:

$$
{\textstyle
\begin{aligned}
\sigma'(\mathbf{z}^{(1)}) \quad &\text{is the derivative of the activation function applied elementwise} \newline
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(2)}} \quad &\text{is the gradient of the loss with respect to the pre-activation output of the second layer}
\end{aligned}
}
$$

### 3.3 Advanced Normalization Techniques

LayerNorm: Normalizes across features within each sample:
$$
{\textstyle
\begin{aligned}
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \quad (19)
\end{aligned}
}
$$

Shape Analysis: For input and learnable parameters:

$$
\begin{aligned}
\mathbf{x} &\in \mathbb{R}^{n \times d} \quad \text{(input)} \newline
\gamma, \beta &\in \mathbb{R}^{d} \quad \text{(learnable per-feature parameters)}
\end{aligned}
$$

Understanding the Greek letters:

$$
{\textstyle
\begin{aligned}
\mu \quad &\text{(mu): The mean (average) of all the numbers} \newline
\sigma \quad &\text{(sigma): The standard deviation - how spread out the numbers are from the average} \newline
\gamma \quad &\text{(gamma): A learnable scale parameter - lets the model decide how much to amplify the result} \newline
\beta \quad &\text{(beta): A learnable shift parameter - lets the model decide how much to shift the result} \newline
\epsilon \quad &\text{(epsilon): A tiny number to prevent division by zero}
\end{aligned}
}
$$

What LayerNorm does: "Make all the numbers have zero average and unit variance, then let the model scale and shift them as needed." It's like standardizing test scores so they're all on the same scale.

where the statistics are computed as:

$$
\begin{aligned}
\mu &= \frac{1}{d}\sum_{i=1}^d x_i \quad \text{(mean)} \newline
\sigma^2 &= \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 \quad \text{(variance)}
\end{aligned}
$$

Why LayerNorm for Sequences: Unlike BatchNorm, it doesn't depend on batch statistics, making it suitable for variable-length sequences.

### 3.4 Alternative Normalization Methods

RMSNorm (Root Mean Square Norm): Simplifies LayerNorm by removing the mean:
$$
{\textstyle
\begin{aligned}
\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
\end{aligned}
}
$$

Benefits: Faster computation, similar performance to LayerNorm.

Scaled Residuals: In very deep networks, scale residual connections:
$$
{\textstyle
\begin{aligned}
\mathbf{h}_{l+1} = \mathbf{h}_l + \alpha \cdot F(\mathbf{h}_l)
\end{aligned}
}
$$
where the scaling factor prevents residual explosion:

$$
\begin{aligned}
\alpha < 1 \quad \text{(scaling factor to prevent explosion)}
\end{aligned}
$$

Pre-LN vs Post-LN:

$$
{\textstyle
\begin{aligned}
\text{Pre-LN (modern):} \quad &\mathbf{h}_{l+1} = \mathbf{h}_l + F(\text{LN}(\mathbf{h}_l)) \newline
\text{Post-LN (original):} \quad &\mathbf{h}_{l+1} = \text{LN}(\mathbf{h}_l + F(\mathbf{h}_l))
\end{aligned}
}
$$

Pre-LN advantages: Better gradient flow, more stable training, enables training deeper models.

## 4. High-Dimensional Geometry & Similarity

### 4.1 Distance Metrics in High Dimensions

Euclidean Distance: 
$$
{\textstyle
\begin{aligned}
d_2(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2
\end{aligned}
}
$$

Cosine Similarity: 
$$
{\textstyle
\begin{aligned}
\cos(\theta) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}
\end{aligned}
}
$$

Concentration of Measure: In high dimensions, most random vectors are approximately orthogonal, making cosine similarity more discriminative than Euclidean distance.

### 4.2 Distance Calculations with Concrete Examples

Using simple example vectors to illustrate the concepts:

- "cat" vector: [0.8, 0.2, 0.1]
- "dog" vector: [0.7, 0.3, 0.2]

#### 4.2.1 Cosine Similarity

Formula: 
$$
{\textstyle
\begin{aligned}
\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}
\end{aligned}
}
$$

Step-by-step calculation:

1. Dot product: A·B = (0.8×0.7) + (0.2×0.3) + (0.1×0.2) = 0.56 + 0.06 + 0.02 = 0.64
2. Magnitudes:
   - ||A|| = √(0.8² + 0.2² + 0.1²) = √(0.64 + 0.04 + 0.01) = √0.69 ≈ 0.83
   - ||B|| = √(0.7² + 0.3² + 0.2²) = √(0.49 + 0.09 + 0.04) = √0.62 ≈ 0.79

3. Cosine similarity: 0.64 / (0.83 × 0.79) ≈ 0.64 / 0.66 ≈ 0.97

Interpretation: Values range from -1 (opposite) to 1 (identical). 0.97 indicates high similarity - "cat" and "dog" point in nearly the same direction in semantic space.

#### 4.2.2 Euclidean Distance

Formula: 
$$
{\textstyle
\begin{aligned}
d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2 + (z_1-z_2)^2}
\end{aligned}
}
$$

Step-by-step calculation:

1. Differences: (0.8-0.7)² + (0.2-0.3)² + (0.1-0.2)² = 0.01 + 0.01 + 0.01 = 0.03
2. Distance: √0.03 ≈ 0.17

Interpretation: Lower values indicate closer proximity. 0.17 is small, confirming "cat" and "dog" are close in space.

When to use each:

- Cosine: When direction matters more than magnitude (text similarity, semantic relationships)
- Euclidean: When absolute distance matters (image features, exact matching)

### 4.3 Maximum Inner Product Search (MIPS)

Problem: Find 
$$
{\textstyle
\begin{aligned}
\mathbf{v}^* = \arg\max_{\mathbf{v} \in \mathcal{V}} \mathbf{q}^T \mathbf{v}
\end{aligned}
}
$$

This is exactly what attention computes when finding relevant keys for a given query!

Connection to Attention: Query-key similarity in attention is inner product search over learned embeddings.

💻 Implementation Example: For high-dimensional similarity comparisons, see [Vectors & Geometry Notebook](./pynb/math_ref/vectors_geometry.ipynb)

## 5. From Similarity to Attention

📚 Quick Reference: See [Scaled Dot-Product Attention](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

### 5.1 Deriving Scaled Dot-Product Attention

Step 1: Start with similarity search between query and keys:

$$
\begin{aligned}
\mathbf{q} &\quad \text{(query vector)} \newline
\{\mathbf{k}_i\} &\quad \text{(set of key vectors)}
\end{aligned}
$$

Compute similarities:
$$
{\textstyle
\begin{aligned}
s_i = \mathbf{q}^T \mathbf{k}_i \quad (20)
\end{aligned}
}
$$

Step 2: Convert similarities to weights via softmax:
$$
{\textstyle
\begin{aligned}
\alpha_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}} \quad (21)
\end{aligned}
}
$$

Understanding the attention weights: These tell us "how much should I pay attention to each word?" The weights are defined as:

$$
\begin{aligned}
\alpha &\quad \text{(attention weight vector)} \newline
\alpha_i &\quad \text{(individual weights that sum to 1, like percentages)}
\end{aligned}
$$

Why use softmax instead of just raw similarities?

1. Probabilities must sum to 1: We want a weighted average, so weights must be between 0 and 1 and sum to 1. Raw similarities could be negative or not sum to 1.
2. Differentiable selection: Softmax provides a "soft" way to pick the most relevant items. Instead of hard selection (pick the best, ignore the rest), it gives more weight to better matches while still considering others.
3. Handles different scales: Raw similarity scores might vary wildly in range. Softmax normalizes them into a consistent 0-1 probability scale.
4. Amplifies differences: The exponential in softmax amplifies differences between scores. If one word is much more relevant, it gets much more attention.
5. Smooth gradients: Unlike hard max (which would create step functions), softmax is smooth everywhere, enabling gradient-based learning.

Real-world analogy: Like deciding how much attention to give each person in a room - you don't ignore everyone except one person (hard max), but you do focus more on the most interesting people while still being somewhat aware of others.

Step 3: Aggregate values using weights:
$$
{\textstyle
\begin{aligned}
\mathbf{o} = \sum_{i=1}^n \alpha_i \mathbf{v}_i \quad (22)
\end{aligned}
}
$$

What this step does: Take a weighted average of all the value vectors. It's like saying "give me 30% of word 1's information, 50% of word 2's information, and 20% of word 3's information."

Matrix Form: For sequences, this becomes:
$$
{\textstyle
\begin{aligned}
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \quad (23)
\end{aligned}
}
$$

Shape Analysis:

$$
\begin{aligned}
Q &\in \mathbb{R}^{n \times d_k} \quad \text{(queries)} \newline
K &\in \mathbb{R}^{n \times d_k} \quad \text{(keys)} \newline
V &\in \mathbb{R}^{n \times d_v} \quad \text{(values)} \newline
\text{Output} &\in \mathbb{R}^{n \times d_v}
\end{aligned}
$$

What this equation accomplishes step-by-step:

$$
{\textstyle
\begin{aligned}
\text{1. } QK^T \quad &\text{creates a "compatibility matrix" - every query checks against every key} \newline
\text{2. } \frac{1}{\sqrt{d_k}} \quad &\text{scales the scores to prevent them from getting too large} \newline
\text{3. } \text{softmax} \quad &\text{converts raw compatibility scores into probability-like weights} \newline
\text{4. Multiply by } V \quad &\text{to get a weighted sum of value vectors}
\end{aligned}
}
$$

Library analogy: Think of attention like a smart librarian. Q is your question ("I need books about neural networks"), K represents the subject tags of all books, V contains the actual book contents. The librarian (attention mechanism) looks at your question, checks which books are most relevant (QK^T), decides how much attention to give each book (softmax), and gives you a summary that's mostly from the most relevant books but includes a little from others (weighted sum with V).

### 5.4 Masked Attention

Causal Masking: For autoregressive models, prevent attention to future tokens:
$$
{\textstyle
\begin{aligned}
P = \text{softmax}\left(\frac{QK^T + M}{\sqrt{d_k}}\right) \quad (24)
\end{aligned}
}
$$

where the mask is defined as:

$$
\begin{aligned}
M_{ij} &\in \{0, -\infty\} \newline
M_{ij} &= -\infty \quad \text{if } i < j \text{ (future positions)}
\end{aligned}
$$

Numerical Stability: Use large negative values to prevent NaN gradients:

$$
\begin{aligned}
-\infty \rightarrow -10^9 \quad \text{(practical approximation)}
\end{aligned}
$$

Implementation:

💻 Implementation Example: For causal mask implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

Padding Masks: Mask out padding tokens by setting their attention scores before softmax:

$$
\begin{aligned}
\text{score}_{\text{padding}} \rightarrow -\infty \quad \text{(ensures zero attention weight)}
\end{aligned}
$$

This ensures padding tokens receive zero attention weight.

Key Insight: Additive masking is preferred over multiplicative masking:

$$
\begin{aligned}
\text{additive: } \quad &\text{score} + (-\infty) \quad \text{(works naturally with softmax)}
\end{aligned}
$$

### 5.2 Why the $$\sqrt{d_k}$$ Scaling?

Variance Analysis: For query and key matrices with independent identically distributed entries:

$$
\begin{aligned}
Q, K &\quad \text{(matrices with i.i.d. entries)} \newline
\text{variance} &= \sigma^2
\end{aligned}
$$

Then:
$$
{\textstyle
\begin{aligned}
\text{Var}(QK^T) = d_k \sigma^4 \quad (24)
\end{aligned}
}
$$

Understanding the variance parameter: This measures how spread out the numbers are - think of it as measuring "how noisy" or "how varied" the data is:

$$
\begin{aligned}
\sigma \quad \text{(standard deviation - measure of spread)}
\end{aligned}
$$

What "i.i.d." means: Independent and identically distributed - each number is random and doesn't depend on the others, like rolling dice multiple times.

Without scaling, attention weights become too peaked (sharp), leading to poor gradients and attention collapse. The relationship between key dimension and attention sharpness:

$$
{\textstyle
\begin{aligned}
d_k \uparrow \quad &\Rightarrow \quad \text{attention weights become sharper} \\
&\Rightarrow \quad \text{gradient flow degrades}
\end{aligned}
}
$$

Pitfall: Forgetting this scaling leads to attention collapse - weights become too concentrated rather than appropriately distributed.

### 5.3 Backpropagation Through Attention

Softmax Gradient: The derivative of softmax output with respect to input logits is:
$$
{\textstyle
\begin{aligned}
\mathbf{p} &= \text{softmax}(\mathbf{z}) \quad \text{(softmax function)} \\
\frac{\partial p_i}{\partial z_j} &= p_i(\delta_{ij} - p_j) \quad (25)
\end{aligned}
}
$$

The Kronecker delta function:

$$
{\textstyle
\begin{aligned}
\delta_{ij} = \begin{cases}
1 & \text{if } i = j \text{ (same position)} \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
}
$$

What is the Kronecker delta? It's a simple function that equals 1 when i=j (same position) and 0 otherwise. Think of it as asking "are these the same thing?" - if yes, return 1; if no, return 0.

Attention Gradients:

Define the intermediate variables for gradient computation:

$$
{\textstyle
\begin{aligned}
S &= QK^T/\sqrt{d_k} \quad \text{(scaled similarity scores)} \\
P &= \mathrm{softmax}(S) \quad \text{(attention weights, row-wise)} \\
O &= PV \quad \text{(attention output)} \\
G_O &= \partial \mathcal{L}/\partial O \quad \text{(gradient from next layer)}
\end{aligned}
}
$$

The gradients flow backward through the attention mechanism as follows:

$$
{\textstyle
\begin{aligned}
\begin{aligned}
G_V &= P^T G_O,\quad &&\text{(V same shape as }V\text{)}\\
G_P &= G_O V^T,\\
G_{S,r} &= \big(\mathrm{diag}(P_r) - P_r P_r^T\big)\, G_{P,r}\quad &&\text{(row }r\text{; softmax Jacobian)}\\
G_Q &= G_S K/\sqrt{d_k},\quad G_K = G_S^T Q/\sqrt{d_k}.
\end{aligned}
\end{aligned}
}
$$

Parameter shapes for attention mechanism:

$$
{\textstyle
\begin{aligned}
Q, K &\in \mathbb{R}^{n\times d_k} \quad \text{(queries and keys)} \\
V &\in \mathbb{R}^{n\times d_v} \quad \text{(values)}
\end{aligned}
}
$$

Intuition: Backprop splits into (i) linear parts, (ii) softmax Jacobian per row.

## 6. Multi-Head Attention & Positional Information

📚 Quick Reference: See [Multi-Head Attention](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) and [Positional Encoding](./math_quick_ref.md#mathematical-quick-reference-for-neural-networks) in the mathematical reference table.

### 6.1 Multi-Head as Subspace Projections

Single Head: Each head projects to a lower-dimensional subspace:

$$
{\textstyle
\begin{aligned}
d_k &= d_{\text{model}}/h \quad \text{(subspace dimension)} \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \quad (26)
\end{aligned}
}
$$

Multi-Head Combination:
$$
{\textstyle
\begin{aligned}
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \quad (27)
\end{aligned}
}
$$

Why multiple heads instead of one big head? Different heads can specialize in different types of relationships:

- Head 1 might focus on syntax: "What words are grammatically related?"
- Head 2 might focus on semantics: "What words are conceptually similar?"
- Head 3 might focus on position: "What words are nearby?"
- Head 4 might focus on long-range dependencies: "What words are related despite being far apart?"

Meeting analogy: Like having different experts in a meeting. Instead of one generalist trying to understand everything, you have specialists: a grammar expert, a meaning expert, a structure expert, etc. Each contributes their perspective, then all insights are combined through concatenation and output projection.

The output projection matrix:

$$
{\textstyle
\begin{aligned}
W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}} \quad \text{(combines all head outputs)}
\end{aligned}
}
$$

Why split the dimension? Rather than having 8 heads each looking at 512-dimensional vectors, we have 8 heads each looking at 64-dimensional projections (512/8=64). This forces each head to focus on a specific subset of features, encouraging specialization.

Implementation Details:

Linear Projections (Not MLPs):

Each head uses simple linear transformations:

$$
{\textstyle
\begin{aligned}
W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_k} \quad &\text{where } d_k = d_{\text{model}}/h
\end{aligned}
}
$$

- These are single linear layers, not multi-layer perceptrons

Total projection parameters per layer:

$$
\begin{aligned}
3 \times h \times d_{\text{model}} \times d_k = 3d_{\text{model}}^2
\end{aligned}
$$

Efficient Implementation:

Instead of computing each head separately, implementations often:

$$
{\textstyle
\begin{aligned}
\text{1. Concatenate all head projections:} \quad &W^Q = [W_1^Q, W_2^Q, ..., W_h^Q] \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \newline
\text{2. Compute in parallel:} \quad &Q = XW^Q, K = XW^K, V = XW^V \newline
\text{3. Reshape tensors to separate heads:} \quad &(n, d_{\text{model}}) \to (n, h, d_k) \newline
\text{4. Apply attention computation across all heads simultaneously}
\end{aligned}
}
$$

Parameter Analysis:

$$
{\textstyle
\begin{aligned}
\text{Per-head projections:} \quad &3 \times d_{\text{model}} \times d_k = 3 \times d_{\text{model}} \times (d_{\text{model}}/h) \newline
\text{Total projections:} \quad &3d_{\text{model}}^2 \text{ (same as single-head with full dimension)} \newline
\text{Output projection:} \quad &W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}} \text{ adds } d_{\text{model}}^2 \text{ parameters} \newline
\text{Total multi-head parameters:} \quad &4d_{\text{model}}^2
\end{aligned}
}
$$

### 6.2 Advanced Positional Encodings

Sinusoidal Encoding: Provides absolute position information:
$$
{\textstyle
\begin{aligned}
\begin{align}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d_{\text{model}}}) \quad (28)\\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d_{\text{model}}}) \quad (29)
\end{align}
\end{aligned}
}
$$

Mathematical Properties of Sinusoidal Encoding:

$$
{\textstyle
\begin{aligned}
\text{Linearity:} \quad &\text{For any fixed offset } k, PE_{pos+k} \text{ can be expressed as a linear function of } PE_{pos} \newline
\text{Relative position encoding:} \quad &\text{The dot product } PE_{pos_i} \cdot PE_{pos_j} \text{ depends only on } |pos_i - pos_j| \newline
\text{Extrapolation:} \quad &\text{Can handle sequences longer than seen during training}
\end{aligned}
}
$$

RoPE (Rotary Position Embedding): Rotates query-key pairs by position-dependent angles:
$$
{\textstyle
\begin{aligned}
\begin{align}
\mathbf{q}_m^{(i)} &= R_{\Theta,m}^{(i)} \mathbf{q}^{(i)} \quad (30)\\
\mathbf{k}_n^{(i)} &= R_{\Theta,n}^{(i)} \mathbf{k}^{(i)} \quad (31)
\end{align}
\end{aligned}
}
$$

RoPE Rotation Matrix:
$$
{\textstyle
\begin{aligned}
R_{\Theta,m}^{(i)} = \begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\end{aligned}
}
$$

The rotation frequency for each dimension pair:

$$
{\textstyle
\begin{aligned}
\theta_i = 10000^{-2i/d_{\text{model}}} \quad \text{(frequency for dimension pair } i\text{)}
\end{aligned}
}
$$

Key RoPE Properties:

$$
{\textstyle
\begin{aligned}
\text{Relative position dependency:} \quad &\mathbf{q}_m^T \mathbf{k}_n \text{ depends only on } (m-n), \text{ not absolute positions} \newline
\text{Length preservation:} \quad &\text{Rotation matrices preserve vector norms} \newline
\text{Computational efficiency:} \quad &\text{Can be implemented without explicit matrix multiplication} \newline
\text{Long sequence generalization:} \quad &\text{Better extrapolation to longer sequences than learned embeddings}
\end{aligned}
}
$$

RoPE Implementation Insight: Instead of rotating the full vectors, RoPE applies rotations to consecutive dimension pairs, enabling efficient implementation via element-wise operations.

💻 Implementation Example: For RoPE (Rotary Position Embedding) implementation, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)
    

### 6.3 Alternative Position Encodings

ALiBi (Attention with Linear Biases): Adds position-dependent bias to attention scores:
$$
{\textstyle
\begin{aligned}
\text{ALiBi-Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{bias}_{ij}\right)V
\end{aligned}
}
$$

The bias term depends on position distance:

$$
{\textstyle
\begin{aligned}
\text{bias}_{ij} &= -m \cdot |i - j| \quad \text{(position-distance penalty)} \\
m &\quad \text{(head-specific slope parameter)}
\end{aligned}
}
$$

Benefits of ALiBi:

$$
{\textstyle
\begin{aligned}
&\text{No position embeddings needed} \newline
&\text{Excellent extrapolation to longer sequences} \newline
&\text{Linear relationship between position distance and attention bias}
\end{aligned}
}
$$

T5 Relative Position Bias: Learns relative position embeddings:
$$
{\textstyle
\begin{aligned}
A_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}} + b_{\text{rel}(i,j)}
\end{aligned}
}
$$

The learned bias term:

$$
{\textstyle
\begin{aligned}
b_{\text{rel}(i,j)} &\quad \text{(learned bias for relative position)} \\
\text{rel}(i,j) &\quad \text{(relative distance function)}
\end{aligned}
}
$$

RoPE Scaling Variants:

$$
{\textstyle
\begin{aligned}
\text{Base scaling:} \quad &\text{Increase base frequency: } \theta_i = \alpha^{-2i/d} \cdot 10000^{-2i/d} \newline
\text{NTK scaling:} \quad &\text{Interpolate frequencies for better long-context performance} \newline
\text{When to use:} \quad &\text{Base scaling for modest extensions (2-4x), NTK for extreme length}
\end{aligned}
}
$$

## 7. Transformer Block Mathematics

### 7.1 Complete Block Equations

Pre-LayerNorm Architecture:
$$
{\textstyle
\begin{aligned}
\begin{align}
\mathbf{h}_1 &= \text{LayerNorm}(\mathbf{x}) \quad (32)\\
\mathbf{h}_2 &= \mathbf{x} + \text{MultiHeadAttn}(\mathbf{h}_1, \mathbf{h}_1, \mathbf{h}_1) \quad (33)\\
\mathbf{h}_3 &= \text{LayerNorm}(\mathbf{h}_2) \quad (34)\\
\mathbf{y} &= \mathbf{h}_2 + \text{FFN}(\mathbf{h}_3) \quad (35)
\end{align}
\end{aligned}
}
$$

Feed-Forward Network:
$$
{\textstyle
\begin{aligned}
\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2 \quad (36)
\end{aligned}
}
$$

Shape Analysis:

$$
{\textstyle
\begin{aligned}
\mathbf{x} &\in \mathbb{R}^{n \times d_{\text{model}}} \quad \text{(input)} \\
W_1 &\in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}} \quad \text{(first layer)} \\
W_2 &\in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}} \quad \text{(second layer)}
\end{aligned}
}
$$

What the FFN does intuitively: Think of it as a "thinking step" for each word individually. After attention has mixed information between words, the FFN lets each word "process" and transform its representation. It's like giving each word some individual processing time to digest the information it received from other words.

Why expand then contract? The FFN first expands the representation to a higher dimension (usually 4× larger), applies a nonlinearity, then contracts back down. This is like having a "working space" where the model can perform more complex computations before producing the final result.

Shape Tracking through the FFN layers:

$$
{\textstyle
\begin{aligned}
\mathbf{x} &\in \mathbb{R}^{n \times d_{\text{model}}} \quad \text{(input)} \\
W_1 &\in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}} \quad \text{(typically } d_{\text{ffn}} = 4d_{\text{model}}\text{) - "expansion layer"} \\
W_2 &\in \mathbb{R}^{d_{\text{ffn}} \times d_{\text{model}}} \quad \text{- "contraction layer"}
\end{aligned}
}
$$

### 7.2 Why GELU over ReLU?

GELU Definition:
$$
{\textstyle
\begin{aligned}
\text{GELU}(x) = x \cdot \Phi(x) \quad (37)
\end{aligned}
}
$$

What GELU does intuitively: GELU is like a "smooth switch." Unlike ReLU which harshly cuts off negative values to zero, GELU gradually transitions from "mostly off" to "mostly on." It asks "how much should I activate this neuron?" and gives a smooth answer between 0 and the input value.

Why smoother is better:

- Better gradients: ReLU has a sharp corner at zero (gradient jumps from 0 to 1). GELU is smooth everywhere, so gradients flow better during training.
- Probabilistic interpretation: GELU can be seen as randomly dropping out inputs based on their value - inputs closer to the mean of a normal distribution are more likely to "survive."
- More expressive: The smooth transition lets the model make more nuanced decisions about what information to keep.

Real-world analogy: ReLU is like a light switch (on/off), while GELU is like a dimmer switch (gradual control).

The standard normal CDF component:

$$
{\textstyle
\begin{aligned}
\Phi(x) \quad \text{(standard normal cumulative distribution function)}
\end{aligned}
}
$$

What is the CDF? It tells you the probability that a normal random variable is less than x. GELU provides smoother gradients than ReLU, improving optimization.

## 8. Training Objective & Tokenization/Embeddings

### 8.1 Next-Token Prediction

Autoregressive Objective:
$$
{\textstyle
\begin{aligned}
\mathcal{L} = -\sum_{t=1}^T \log P(x_t | x_1, ..., x_{t-1}) \quad (38)
\end{aligned}
}
$$

Shape Analysis:

$$
{\textstyle
\begin{aligned}
B &\quad \text{(batch size)} \\
T &\quad \text{(sequence length)} \\
&\text{Loss computed per sequence, averaged over batch}
\end{aligned}
}
$$

Implementation: Use causal mask in attention to prevent information leakage from future tokens.

### 8.2 Embedding Mathematics

Token Embeddings: Map discrete tokens to continuous vectors:
$$
{\textstyle
\begin{aligned}
\mathbf{e}_i = E[i] \in \mathbb{R}^{d_{\text{model}}} \quad (39)
\end{aligned}
}
$$

The embedding matrix:

$$
{\textstyle
\begin{aligned}
E \in \mathbb{R}^{V \times d_{\text{model}}} \quad \text{(vocabulary size × model dimension)}
\end{aligned}
}
$$

Weight Tying: Share the embedding matrix with output projection to reduce parameters:
$$
{\textstyle
\begin{aligned}
P(w_t | \text{context}) = \text{softmax}(\mathbf{h}_t E^T) \quad (40)
\end{aligned}
}
$$

Shape Analysis:

$$
{\textstyle
\begin{aligned}
\mathbf{h}_t &\in \mathbb{R}^{1 \times d_{\text{model}}} \quad \text{(hidden state)} \\
E &\in \mathbb{R}^{V \times d_{\text{model}}} \quad \text{(embedding matrix)} \\
\mathbf{h}_t E^T &\in \mathbb{R}^{1 \times V} \quad \text{(logits over vocabulary)} \\
&\text{Consistent with row-vector convention}
\end{aligned}
}
$$

Perplexity: Measures model uncertainty:

$$
{\textstyle
\begin{aligned}
\mathrm{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P\left(x_t \mid x_{<t}\right) \right) \quad (41)
\end{aligned}
}
$$

> Note: This equation may not render correctly in GitHub. Use a Markdown viewer!

## 9. Worked Mini-Examples

### 9.1 Tiny Attention Forward Pass

Setup parameters:

$$
{\textstyle
\begin{aligned}
n &= 2 \quad \text{(number of tokens)} \\
d_k &= d_v = 3 \quad \text{(key/value dimension)} \\
&\text{(single head)}
\end{aligned}
}
$$

Input:
```
Q = [[1, 0, 1],    K = [[1, 1, 0],    V = [[2, 0, 1],
     [0, 1, 1]]         [1, 0, 1]]         [1, 1, 0]]
```

Step 1: Compute raw similarity scores:

$$
{\textstyle
\begin{aligned}
QK^T &= \begin{bmatrix}1 & 0 & 1\\0 & 1 & 1\end{bmatrix} \begin{bmatrix}1 & 1\\1 & 0\\0 & 1\end{bmatrix} \\
&= \begin{bmatrix}1 & 2\\1 & 1\end{bmatrix}
\end{aligned}
}
$$

Step 2: Apply scaling factor:

$$
{\textstyle
\begin{aligned}
\frac{1}{\sqrt{d_k}} &= \frac{1}{\sqrt{3}} \approx 0.577 \quad \text{(scaling factor)} \\
S &= \frac{QK^T}{\sqrt{3}} = \begin{bmatrix}0.577 & 1.155\\0.577 & 0.577\end{bmatrix}
\end{aligned}
}
$$

Step 3: Apply softmax (row-wise, rounded to 3 d.p.):

$$
{\textstyle
\begin{aligned}
\text{Row 1:} \quad &e^{0.577} = 1.781, e^{1.155} = 3.173, \text{ sum } = 4.954 \newline
\text{Row 2:} \quad &e^{0.577} = 1.781, e^{0.577} = 1.781, \text{ sum } = 3.562
\end{aligned}
}
$$
{\textstyle
\begin{aligned}
A = \begin{bmatrix}
0.359 & 0.641 \\
0.500 & 0.500
\end{bmatrix}
\end{aligned}
}
$$

Step 4: Multiply attention weights by values:

$$
{\textstyle
\begin{aligned}
O &= AV \\
&= \begin{bmatrix}0.359 & 0.641\\0.500 & 0.500\end{bmatrix} \begin{bmatrix}2 & 0 & 1\\1 & 1 & 0\end{bmatrix} \\
&= \begin{bmatrix}1.359 & 0.641 & 0.359\\1.500 & 0.500 & 0.500\end{bmatrix}
\end{aligned}
}
$$

💻 Implementation Example: For attention computation verification, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

### 9.2 Backprop Through Simple Attention

Given: 
$$
{\textstyle
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial O} = \begin{bmatrix}1 & 0 & 1\\0 & 1 & 0\end{bmatrix}
\end{aligned}
}
$$

Gradient w.r.t. Values:
$$
{\textstyle
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial V} = A^T \frac{\partial \mathcal{L}}{\partial O} = \begin{bmatrix}0.359 & 0.500\\0.641 & 0.500\end{bmatrix}\begin{bmatrix}1 & 0 & 1\\0 & 1 & 0\end{bmatrix} = \begin{bmatrix}0.359 & 0.500 & 0.359\\0.641 & 0.500 & 0.641\end{bmatrix}
\end{aligned}
}
$$

💻 Implementation Example: For gradient verification using finite differences, see [Advanced Concepts Notebook](./pynb/math_ref/advanced_concepts.ipynb)

Check for Understanding: Verify that gradient shapes match parameter shapes and that the chain rule is applied correctly.

---

## Continuing to Advanced Topics

This concludes Part 1 of the mathematics tutorial, covering the foundational concepts needed to understand how Transformers work. You now understand:

1. Mathematical foundations - from basic calculus to gradient descent
2. Attention as similarity search - how Q/K/V naturally emerge
3. Multi-head attention - parallel specialized attention patterns
4. Transformer blocks - combining attention with feed-forward networks
5. Training objectives - next-token prediction and embeddings

Continue to [Part 2: Advanced Concepts and Scaling](./transformers_math2.md) to learn about:

- Advanced optimization techniques (Adam, learning rate schedules)
- Efficient attention implementations (FlashAttention, KV caching)
- Regularization and generalization techniques
- Implementation best practices and common pitfalls
- Scaling laws and practical considerations

The mathematical foundation you've built here will serve you well as we explore more sophisticated training techniques and efficiency optimizations in Part 2.
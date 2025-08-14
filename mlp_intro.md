# Multi-Layer Perceptrons (MLPs): A Step-by-Step Tutorial

**What you'll learn:** How MLPs process fixed-size inputs through layers of neurons, transform data with weights and activations, and serve as the foundation for all modern neural networks. We'll work through math and intuition together, with tiny examples you can follow by hand.

**Prerequisites:** Basic linear algebra (vectors, matrices, dot products). No prior neural network experience needed.

---

## 1. What is an MLP?

Imagine you want to predict whether an email is spam based on features like "number of exclamation marks," "contains word 'free'," and "sender reputation score." An MLP takes these features and combines them through multiple layers of simple computations to make a prediction.

### MLP vs Simple Linear Model

**Simple Linear Model:**
```
Input Features → Single Computation → Output
[3, 1, 0.2]   →   w₁×3 + w₂×1 + w₃×0.2 + b   →   spam probability
```
- Direct mapping from features to output
- Can only learn linear relationships
- Limited expressiveness

**MLP (Multi-Layer Perceptron):**
```
Layer 1: [3, 1, 0.2] → [h₁, h₂, h₃] (hidden features)
Layer 2: [h₁, h₂, h₃] → [h₄, h₅] (more hidden features)  
Layer 3: [h₄, h₅] → [spam probability] (output)
```
- Multiple layers of transformations
- Can learn complex, non-linear patterns  
- Much more expressive

**Key Insight:** An MLP is like having multiple simple models stacked on top of each other, where each layer learns increasingly complex features from the layer below.

---

## 2. The Core MLP Equations

The heart of every MLP layer is this transformation:

$$h = \sigma(Wx + b)$$

Let's break this down term by term:

| Term | Size | Meaning |
|------|------|---------|
| $x$ | $(D_{in},)$ | **Input vector** - features coming into this layer |
| $W$ | $(D_{in}, D_{out})$ | **Weight matrix** - learned transformation |
| $b$ | $(D_{out},)$ | **Bias vector** - learned offset |
| $Wx + b$ | $(D_{out},)$ | **Linear combination** - weighted sum of inputs |
| $\sigma(\cdot)$ | - | **Activation function** - introduces non-linearity |
| $h$ | $(D_{out},)$ | **Output vector** - transformed features |

### Visual Breakdown

```
Input Features     Weight Matrix     Bias      Activation
     x        ×         W         +   b    →    σ(·)    →   h
   [x₁]        [w₁₁ w₁₂]         [b₁]                    [h₁]
   [x₂]    ×   [w₂₁ w₂₂]     +   [b₂]   →   σ(·)   →   [h₂]
              [w₃₁ w₃₂]
```

**Why this structure?**
- **$Wx$:** Each output neuron gets a weighted combination of all inputs
- **$+ b$:** Bias allows shifting the activation threshold
- **$\sigma(\cdot)$:** Non-linearity enables learning complex patterns

**Common Activation Functions:**
- **ReLU**: $\sigma(z) = \max(0, z)$ - most popular, simple and effective
- **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$ - outputs between 0 and 1
- **Tanh**: $\sigma(z) = \tanh(z)$ - outputs between -1 and 1

---

## 3. Multi-Layer Architecture

### Stacking Layers

A complete MLP chains multiple layers together:

$$h^{(1)} = \sigma^{(1)}(W^{(1)}x + b^{(1)})$$
$$h^{(2)} = \sigma^{(2)}(W^{(2)}h^{(1)} + b^{(2)})$$
$$\vdots$$
$$y = W^{(L)}h^{(L-1)} + b^{(L)}$$

**Layer Naming Convention:**
- **Input Layer**: The original features $x$
- **Hidden Layers**: Intermediate layers $h^{(1)}, h^{(2)}, \ldots$
- **Output Layer**: Final predictions $y$ (often no activation for regression)

### Information Flow

```
Input → Hidden 1 → Hidden 2 → ... → Output
  x   →   h¹     →   h²      →     →   y

Each layer transforms its input into increasingly abstract representations
```

**What Each Layer Learns:**
- **Layer 1**: Basic feature combinations (edges, simple patterns)
- **Layer 2**: More complex features (shapes, motifs)  
- **Layer 3+**: High-level concepts (objects, semantic meaning)

---

## 4. Hidden Size and Layer Depth

### Hidden Size: Width of Each Layer

Hidden size controls how many features each layer can learn:

```
Small hidden size (2 neurons):  h = [h₁, h₂]
Large hidden size (100 neurons): h = [h₁, h₂, ..., h₁₀₀]
```

- **Larger hidden size**: Can learn more complex patterns, but more parameters
- **Smaller hidden size**: Simpler model, less prone to overfitting

**Analogy:** Like having 2 vs 100 "detectors" in each layer to find patterns.

### Layer Depth: How Many Layers

Depth controls the complexity of patterns the network can learn:

```
Shallow (2 layers): Input → Hidden → Output
Deep (5 layers):    Input → H1 → H2 → H3 → H4 → Output
```

- **Deeper networks**: Can learn more hierarchical, abstract representations
- **Shallow networks**: Simpler, faster, easier to train

**Rule of Thumb:** Start shallow, go deeper only if needed.

### Parameter Counting

For a layer with input size $D_{in}$ and output size $D_{out}$:
- **Weights**: $D_{in} \times D_{out}$ parameters
- **Biases**: $D_{out}$ parameters
- **Total**: $D_{in} \times D_{out} + D_{out} = D_{out}(D_{in} + 1)$

**Example Network:**
- Input: 10 features
- Hidden 1: 20 neurons → $20 \times (10 + 1) = 220$ parameters
- Hidden 2: 15 neurons → $15 \times (20 + 1) = 315$ parameters  
- Output: 1 neuron → $1 \times (15 + 1) = 16$ parameters
- **Total**: 551 parameters

---

## 5. Worked Example: Email Spam Detection

Let's trace through a tiny example step by step. We'll use:
- **Input features**: [num_exclamations, has_word_free, sender_reputation]
- **Hidden layer size**: 2 neurons
- **Output**: spam probability

### Step 0: Initialize

**Input email features:**
```
Email: "Free money!!!"
x = [3, 1, 0.8]  # [3 exclamations, has "free", reputation 0.8]
```

**Learned weights (after training):**
```
# Hidden Layer 1 (2 neurons)
W¹ = [[0.5, -0.2],    # 3×2 matrix: input-to-hidden
      [0.8,  0.6],
      [-0.3, 0.4]]

b¹ = [0.1, -0.5]      # 2-element bias vector

# Output Layer (1 neuron)  
W² = [[0.7],          # 2×1 matrix: hidden-to-output
      [1.2]]

b² = [-0.3]           # 1-element bias vector
```

### Step 1: Forward Pass Through Hidden Layer

**Input:** $x = [3, 1, 0.8]$

**Compute linear combination:**
```
W¹x + b¹ = [[0.5, -0.2],   [3,    [0.1,     [0.5×3 + 0.8×1 + (-0.3)×0.8,     [2.06,
            [0.8,  0.6], × [1, + [-0.5] = [-0.2×3 + 0.6×1 + 0.4×0.8]   = [0.72]
            [-0.3, 0.4]]   [0.8]

Calculation:
Neuron 1: 0.5×3 + 0.8×1 + (-0.3)×0.8 + 0.1 = 1.5 + 0.8 - 0.24 + 0.1 = 2.16
Neuron 2: (-0.2)×3 + 0.6×1 + 0.4×0.8 + (-0.5) = -0.6 + 0.6 + 0.32 - 0.5 = -0.18
```

**Apply ReLU activation:**
```
h¹ = ReLU([2.16, -0.18]) = [max(0, 2.16), max(0, -0.18)] = [2.16, 0]
```

### Step 2: Forward Pass Through Output Layer

**Input:** $h^{(1)} = [2.16, 0]$

**Compute linear combination:**
```
W²h¹ + b² = [[0.7],   [2.16,    [-0.3] = [0.7×2.16 + 1.2×0] + [-0.3]
             [1.2]] × [0]   +            = [1.512] + [-0.3]
                                         = [1.212]
```

**Apply sigmoid for probability:**
```
y = sigmoid(1.212) = 1/(1 + e^(-1.212)) = 1/(1 + 0.297) = 0.771
```

**Result:** 77.1% probability this email is spam!

### Step 3: Understanding What Happened

```
Original Features: [3 exclamations, has "free", good reputation]
                         ↓
Hidden Layer:      [2.16, 0]  # Learned feature combinations
                         ↓  
Output:           0.771       # 77% spam probability
```

**Hidden Neuron Analysis:**
- **Neuron 1 (activated)**: Detected pattern of many exclamations + "free" word
- **Neuron 2 (silent)**: Its pattern wasn't triggered by this input

**Key Insight:** The MLP learned that the combination of exclamations and "free" is a strong spam indicator, even with good sender reputation.

---

## 6. Training: How MLPs Learn

### The Learning Process

MLPs learn through **supervised learning**:
1. **Forward Pass**: Compute predictions using current weights
2. **Loss Calculation**: Measure how wrong the predictions are
3. **Backward Pass**: Compute gradients using backpropagation
4. **Weight Update**: Adjust weights to reduce the loss

> **📚 Mathematical Deep Dive**: For a complete step-by-step mathematical explanation of how gradient descent works from line slopes to neural network training, see **[transformers_math.md Section 2.1.0](./transformers_math.md#210-from-line-slopes-to-neural-network-training)** - includes worked examples and the connection between simple derivatives and MLP backpropagation.

### Loss Functions

**For Binary Classification (like spam detection):**
```
Binary Cross-Entropy: L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

Where:
- y = true label (0 or 1)
- ŷ = predicted probability
```

**For Regression (predicting numbers):**
```
Mean Squared Error: L = (y - ŷ)²

Where:
- y = true value  
- ŷ = predicted value
```

### Backpropagation: The Learning Algorithm

**Forward Pass** (what we just did):
```
x → h¹ → y → Loss
```

**Backward Pass** (compute gradients):
```
∂L/∂W² ← computed from output
∂L/∂W¹ ← flows back through hidden layer
```

**Weight Updates:**
```
W² = W² - α × ∂L/∂W²  # α is learning rate
W¹ = W¹ - α × ∂L/∂W¹
b² = b² - α × ∂L/∂b²
b¹ = b¹ - α × ∂L/∂b¹
```

> **🔗 Mathematical Connection**: The backpropagation equations above are derived step-by-step in **[transformers_math.md Section 2.1.0](./transformers_math.md#210-from-line-slopes-to-neural-network-training)**. See the "Single Hidden Layer MLP" subsection for the complete mathematical derivation including the δ terms and chain rule applications.

### Training Loop Example

```
For each batch of training examples:
    1. Forward pass: compute predictions
    2. Compute loss: how wrong are we?
    3. Backward pass: compute gradients
    4. Update weights: move in direction to reduce loss
    5. Repeat until loss is small enough
```

---

## 7. MLP vs Other Models

### MLP vs Linear Regression

| **Aspect** | **Linear Regression** | **MLP** |
|------------|---------------------|---------|
| **Equation** | $y = Wx + b$ | $y = W^L \sigma(W^{L-1} \sigma(\ldots))$ |
| **Decision Boundary** | Straight line/plane | Curved, complex shapes |
| **Expressiveness** | Limited to linear patterns | Can learn any continuous function |
| **Training** | Closed-form solution | Iterative optimization |

### MLP Advantages

✅ **Universal Approximation**: Can learn any continuous function with enough neurons
✅ **Non-linear Patterns**: Captures complex relationships in data
✅ **Automatic Features**: Learns useful feature combinations
✅ **Scalable**: Works with large datasets and many features

### MLP Limitations

❌ **No Sequential Memory**: Processes each input independently
❌ **Fixed Input Size**: Can't handle variable-length inputs
❌ **No Spatial Structure**: Doesn't understand image/text structure
❌ **Many Parameters**: Can overfit with small datasets

### When to Use MLPs

**Good for:**
- Tabular data (rows and columns)
- Classification and regression tasks
- Fixed-size feature vectors
- When you need a simple, interpretable baseline

**Not ideal for:**
- Sequential data (text, time series) → Use RNNs/Transformers
- Images → Use CNNs  
- Variable-length inputs → Use sequence models
- When you need to understand spatial relationships

---

## 8. Common Challenges and Solutions

### Overfitting: When MLPs Memorize

**Problem**: Network performs well on training data but poorly on new data.

**Signs:**
- Training accuracy: 99%
- Test accuracy: 60%
- Large gap indicates overfitting

**Solutions:**
```
1. Reduce model size (fewer layers/neurons)
2. Add regularization:
   - Dropout: Randomly turn off neurons during training
   - L2 penalty: Penalize large weights
3. More training data
4. Early stopping: Stop when validation loss increases
```

### Underfitting: When MLPs Are Too Simple

**Problem**: Network can't learn the underlying patterns.

**Signs:**
- Both training and test accuracy are low
- Loss plateaus at a high value

**Solutions:**
```
1. Increase model size (more layers/neurons)
2. Train for more epochs
3. Lower learning rate for finer adjustments
4. Check for bugs in data preprocessing
```

### Gradient Problems

**Vanishing Gradients**: Gradients become too small in deep networks
```
Solutions:
- Use ReLU activation (not sigmoid/tanh)
- Better weight initialization (Xavier/He)
- Batch normalization
- Skip connections
```

**Exploding Gradients**: Gradients become too large
```
Solutions:  
- Gradient clipping: Cap gradient magnitude
- Lower learning rate
- Better weight initialization
```

> **🎯 Gradient Flow Mathematics**: To understand the mathematical foundations of why gradients vanish or explode, and how gradient descent fundamentally works, see **[transformers_math.md Section 2.1.0](./transformers_math.md#210-from-line-slopes-to-neural-network-training)**. The section builds intuition from simple 1D slopes to complex neural network training.

---

## 9. Activation Functions Deep Dive

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

```
Input:  [-2, -1, 0, 1, 2]
Output: [ 0,  0, 0, 1, 2]
```

**Advantages:**
- Simple and fast to compute
- Doesn't saturate for positive values
- Sparse activation (many neurons output 0)

**Disadvantages:**
- "Dead neurons" - can output 0 forever if weights become negative

### Sigmoid

$$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

```
Input:  [-2, -1, 0, 1, 2]
Output: [0.12, 0.27, 0.5, 0.73, 0.88]
```

**Advantages:**
- Smooth, differentiable everywhere
- Outputs between 0 and 1 (good for probabilities)

**Disadvantages:**
- Saturates for large |x| (gradients → 0)
- Not zero-centered (can slow learning)

### Tanh (Hyperbolic Tangent)

$$\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```
Input:  [-2, -1, 0, 1, 2]
Output: [-0.96, -0.76, 0, 0.76, 0.96]
```

**Advantages:**
- Zero-centered (better than sigmoid)
- Smooth and differentiable

**Disadvantages:**
- Still suffers from saturation
- More expensive to compute than ReLU

### Choosing Activations

```
Hidden Layers: Use ReLU (default choice)
- Fast, simple, works well in practice
- Use Leaky ReLU if you see many dead neurons

Output Layer:
- Binary classification: Sigmoid
- Multi-class classification: Softmax  
- Regression: Linear (no activation)
```

---

## 10. Practical Implementation Tips

### Network Architecture Design

**Start Simple:**
```
1. Begin with 1-2 hidden layers
2. Hidden size = 2-4× input size (rule of thumb)
3. Add layers only if underfitting
```

**Common Patterns:**
```
Small dataset (< 10K samples):
Input → Hidden(64) → Hidden(32) → Output

Medium dataset (10K-100K samples):  
Input → Hidden(128) → Hidden(64) → Hidden(32) → Output

Large dataset (> 100K samples):
Input → Hidden(256) → Hidden(128) → Hidden(64) → Output
```

### Hyperparameter Tuning

**Learning Rate:**
```
Too high: Loss oscillates or explodes
Too low:  Very slow convergence
Sweet spot: Usually 0.001 - 0.01
```

> **📊 Learning Rate Intuition**: For a visual and mathematical explanation of why learning rate choice matters, see the worked example with f(x) = x² in **[transformers_math.md Section 2.1.0](./transformers_math.md#210-from-line-slopes-to-neural-network-training)** - shows exactly how different learning rates affect convergence behavior.

**Batch Size:**
```
Small (32): Noisy gradients, more exploration
Large (512): Stable gradients, faster per epoch
Common choice: 64-128
```

**Training Tips:**
```
1. Normalize input features: zero mean, unit variance
2. Initialize weights properly (Xavier/He initialization)
3. Monitor both training and validation loss
4. Use learning rate scheduling (reduce over time)
```

### Debugging Checklist

**If training isn't working:**
```
1. Check data: Are labels correct? Proper preprocessing?
2. Start tiny: Can model overfit a single batch?
3. Verify gradients: Are they flowing properly?
4. Learning rate: Try 10× higher and 10× lower
5. Model capacity: Too big (overfit) or too small (underfit)?
```

---

## 11. Summary: The MLP Foundation

### How MLPs Work

1. **Layer-wise Processing**: Transform inputs through multiple layers
2. **Non-linear Combinations**: Each layer learns complex feature combinations  
3. **Universal Approximation**: Can learn any continuous function with enough neurons
4. **Supervised Learning**: Learn from input-output examples through backpropagation

### Key Components Working Together

```
Input Features → Layer 1 → Layer 2 → ... → Output
     x       → σ(W¹x+b¹) → σ(W²h¹+b²) →  → W^L h^(L-1)+b^L

Where each layer applies: Linear Transformation → Non-linear Activation
```

**Weight Matrices ($W$):** "How should features be combined?"
**Bias Vectors ($b$):** "What are the default activation thresholds?"  
**Activations ($\sigma$):** "How should we introduce non-linearity?"

### Capacity Control

- **Width (hidden size)**: How many patterns each layer can detect
- **Depth (num layers)**: How complex/hierarchical patterns can be
- **Regularization**: Controls overfitting vs underfitting balance

### MLP's Role in Modern AI

**Foundation for Everything:**
- **CNNs**: MLPs + spatial structure for images
- **RNNs**: MLPs + memory for sequences  
- **Transformers**: MLPs + attention mechanisms
- **Modern architectures**: All use MLP components

**Key Insight:** MLPs are the "universal building block" - understanding them deeply helps with all neural network architectures.

---

## 12. Final Visualization: Email Spam Detection

```
Email: "Free money!!!"
Features: [3, 1, 0.8]  # [exclamations, has_free, reputation]
                ↓
Hidden Layer 1: W¹x + b¹ = [2.16, -0.18]
                ↓  
After ReLU:     h¹ = [2.16, 0]
                ↓
Output Layer:   W²h¹ + b² = [1.212] 
                ↓
After Sigmoid:  y = 0.771  # 77% spam probability
```

**The Journey:** From raw email features to spam probability through learned transformations. The MLP discovered that "many exclamations + free word" is a strong spam signal.

**What It Learned:**
- Hidden neuron 1: Detects "suspicious promotional patterns"  
- Hidden neuron 2: Detects some other pattern (inactive for this email)
- Output combination: Weighs these patterns to make final decision

---

## Next Steps

Now that you understand MLPs:

1. **Limitations**: MLPs can't handle sequences well (no memory)
2. **Next Architecture**: RNNs add memory for sequential data
3. **Modern Context**: MLPs are components in Transformers and other architectures
4. **Implementation**: Try building an MLP in PyTorch or TensorFlow

> **Continue Learning**: Ready for sequences? See **[rnn_intro.md](./rnn_intro.md)** to learn how RNNs add memory to the MLP foundation.

**Remember:** MLPs taught us that neural networks could learn complex, non-linear patterns through simple transformations. Every modern architecture builds on these core principles - making MLPs essential foundational knowledge.
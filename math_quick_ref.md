# Mathematical Quick Reference for Neural Networks

A comprehensive reference of core mathematical concepts used in neural networks and deep learning, with detailed intuitions and practical PyTorch examples.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
   - [Matrix Multiplication](#matrix-multiplication)
   - [Matrix Transpose](#matrix-transpose)
   - [Matrix Inverse](#matrix-inverse)
   - [Eigenvalues & Eigenvectors](#eigenvalues--eigenvectors)
   - [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)

2. [Vectors & Geometry](#vectors--geometry)
   - [Dot Product](#dot-product)
   - [Cosine Similarity](#cosine-similarity)
   - [Euclidean Distance](#euclidean-distance)
   - [Lp Norms](#lp-norms)

3. [Calculus](#calculus)
   - [Derivatives](#derivatives)
   - [Partial Derivatives](#partial-derivatives)
   - [Chain Rule](#chain-rule)
   - [Gradient](#gradient)
   - [Hessian](#hessian)
   - [Jacobian](#jacobian)

4. [Differential Equations](#differential-equations)
   - [Ordinary Differential Equations (ODEs)](#ordinary-differential-equations-odes)
   - [Partial Differential Equations (PDEs)](#partial-differential-equations-pdes)

5. [Nonlinear Functions](#nonlinear-functions)
   - [Hyperbolic Tangent (tanh)](#hyperbolic-tangent-tanh)
   - [Sigmoid](#sigmoid)
   - [ReLU](#relu)

6. [Probability & Statistics](#probability--statistics)
   - [Expectation](#expectation)
   - [Variance](#variance)
   - [Softmax](#softmax)
   - [Cross-Entropy Loss](#cross-entropy-loss)

7. [Optimization](#optimization)
   - [Gradient Descent](#gradient-descent)
   - [Adam Optimizer](#adam-optimizer)

8. [Attention Mechanisms](#attention-mechanisms)
   - [Scaled Dot-Product Attention](#scaled-dot-product-attention)

9. [Transformer Components](#transformer-components)
   - [Layer Normalization](#layer-normalization)
   - [Residual Connections](#residual-connections)

10. [Advanced Concepts](#advanced-concepts)
    - [Temperature Scaling](#temperature-scaling)

---

## Linear Algebra

> 📓 **Jupyter Notebook:** [pynb/math_ref/linear_algebra.ipynb](pynb/math_ref/linear_algebra.ipynb)

### Matrix Multiplication

**Formula:** $\mathbf{C} = \mathbf{A}\mathbf{B}$ where $C_{ij} = \sum_k A_{ik}B_{kj}$

**Intuition:** The fundamental operation of neural networks. Each row of $\mathbf{A}$ represents neuron weights, each column of $\mathbf{B}$ represents input vectors. The result computes weighted sums for all neurons simultaneously, enabling parallel computation. This is why a single matrix multiplication can represent an entire layer's forward pass.

**How it affects output:** Matrix multiplication transforms input vectors from one feature space to another. Small changes in weights create proportional changes in outputs, making gradient-based learning possible.

### Matrix Transpose

**Formula:** $(\mathbf{A})^T_{ij} = \mathbf{A}_{ji}$

**Intuition:** Essential for backpropagation. When gradients flow backward through a layer with weights $\mathbf{W}$, we need $\mathbf{W}^T$ to properly route the gradient signals back to the previous layer. The transpose "reverses" the forward direction of information flow.

**How it affects output:** Transpose changes how information flows. In forward pass, weights map inputs to outputs. In backward pass, transpose maps output gradients back to input gradients.

### Matrix Inverse

**Formula:** $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$

**Intuition:** Used in analytical solutions for least squares (normal equations) and understanding linear transformations. In neural networks, helps analyze layer transformations and appears in second-order optimization methods like Newton's method.

**How it affects output:** Matrix inverse provides exact solutions when they exist. However, neural networks rarely use inverses directly due to computational cost and numerical instability.

### Eigenvalues & Eigenvectors

**Formula:** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$

**Intuition:** Reveals principal directions of data variation (PCA), helps analyze gradient flow and conditioning of weight matrices. Large eigenvalues can indicate exploding gradients, while small ones suggest vanishing gradients. Critical for understanding optimization landscapes.

**How it affects output:** Eigenvalues indicate how much each direction gets amplified. Large eigenvalues can cause exploding gradients, small ones cause vanishing gradients.

### Singular Value Decomposition (SVD)

**Formula:** $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$

**Intuition:** Decomposes any matrix into orthogonal transformations and scaling. Used in dimensionality reduction, weight initialization, and analyzing the effective rank of learned representations. Helps understand what transformations neural network layers are actually learning.

**How it affects output:** SVD reveals the intrinsic dimensionality and structure of transformations. It's used for initialization to maintain gradient flow and for analyzing what neural networks learn.

---

## Vectors & Geometry

> 📓 **Jupyter Notebook:** [pynb/math_ref/vectors_geometry.ipynb](pynb/math_ref/vectors_geometry.ipynb)

### Dot Product

**Formula:** $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos(\theta)$

**Intuition:** Measures how "aligned" two vectors are. In neurons, the dot product between input $\mathbf{x}$ and weights $\mathbf{w}$ gives the raw activation strength—high when input pattern matches what the neuron is looking for. This is the core operation that determines neuron firing.

**How it affects output:** High dot product means input and weight vectors point in similar directions, creating strong positive activation. Orthogonal vectors produce zero activation.

### Cosine Similarity

**Formula:** $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$

**Intuition:** Measures similarity independent of magnitude. Used in attention mechanisms, word embeddings, and similarity-based learning. Helps neural networks focus on directional patterns rather than absolute magnitudes, making them more robust to scaling variations.

**How it affects output:** Cosine similarity ranges from -1 to 1, making it scale-invariant. Similar directions get high scores regardless of vector magnitude.

### Euclidean Distance

**Formula:** $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_i (a_i - b_i)^2}$

**Intuition:** Measures how "far apart" two points are in feature space. Used in loss functions (MSE), clustering, and nearest neighbor methods. In neural networks, helps measure prediction errors and similarity between representations.

**How it affects output:** Smaller distances indicate more similar points. MSE loss penalizes large errors quadratically, making the model focus on reducing big mistakes first.

### Lp Norms

**Formula:** $\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$

**Intuition:** Measures vector "size" in different ways. $L_1$ promotes sparsity (many weights become zero), $L_2$ promotes smoothness (weights stay small). Used in regularization to control model complexity and prevent overfitting by penalizing large weights.

**How it affects output:** L1 norm creates sparse solutions (many zeros), L2 norm creates smooth solutions (small values). Higher p values focus more on the largest elements.

---

## Calculus

> 📓 **Jupyter Notebook:** [pynb/math_ref/calculus.ipynb](pynb/math_ref/calculus.ipynb)

### Derivatives

**Formula:** $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$

**Intuition:** Measures how fast a function changes. In neural networks, tells us how much the loss changes when we tweak a parameter. This is the foundation of gradient-based learning—we follow the derivative to find better parameter values.

**How it affects output:** Derivatives tell us sensitivity - how much output changes for small input changes. Essential for parameter updates.

### Partial Derivatives

**Formula:** $\frac{\partial f}{\partial x_i}$

**Intuition:** Derivative with respect to one variable while holding others constant. Neural networks have millions of parameters, so we need partial derivatives to see how the loss changes with respect to each individual weight or bias.

**How it affects output:** Each parameter gets its own gradient, allowing independent optimization of millions of parameters simultaneously.

### Chain Rule

**Formula:** $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$

**Intuition:** The mathematical foundation of backpropagation. Neural networks are compositions of functions (layer after layer), so to compute gradients we multiply derivatives along the chain from output back to input. This is why it's called "backpropagation"—propagating derivatives backward through the composition.

**How it affects output:** Chain rule enables automatic differentiation through arbitrarily deep networks by systematically applying the composition rule.

### Gradient

**Formula:** $\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]$

**Intuition:** Points in the direction of steepest increase. In neural network training, we move parameters in the negative gradient direction (steepest decrease) to minimize the loss function. The gradient tells us both direction and magnitude of the best parameter update.

**How it affects output:** Gradient provides both direction (sign) and magnitude for optimal parameter updates. Larger gradients indicate more sensitive parameters.

### Hessian

**Formula:** $\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

**Intuition:** Matrix of second derivatives that describes the curvature of the loss surface. Helps understand convergence behavior and is used in advanced optimization methods like Newton's method and natural gradients. High curvature areas require smaller learning rates.

**How it affects output:** Hessian reveals optimization landscape curvature. High condition numbers indicate difficult optimization requiring careful learning rates.

### Jacobian

**Formula:** $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$

**Intuition:** Matrix of first derivatives for vector-valued functions. Essential for backpropagation through layers that output vectors (like hidden layers). Each element shows how one output component changes with respect to one input component, enabling gradient flow through complex architectures.

**How it affects output:** Jacobian enables gradient flow through vector outputs, crucial for multi-output layers and complex architectures.

---

## Differential Equations

> 📓 **Jupyter Notebook:** [pynb/math_ref/differential_equations.ipynb](pynb/math_ref/differential_equations.ipynb)

### Ordinary Differential Equations (ODEs)

**Formula:** $\frac{dy}{dt} = f(y, t)$

**Intuition:** Models how quantities change over time. Neural ODEs treat layer depth as continuous time, allowing adaptive depth and memory-efficient training. ResNets approximate the solution to ODEs, explaining why skip connections work so well for deep networks.

**How it affects output:** ODEs provide continuous depth, memory efficiency, and theoretical foundations for understanding deep networks as dynamical systems.

### Partial Differential Equations (PDEs)

**Formula:** $\frac{\partial u}{\partial t} = f\left(u, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \ldots\right)$

**Intuition:** Models complex spatiotemporal phenomena. Physics-informed neural networks (PINNs) embed PDE constraints directly into the loss function, allowing neural networks to solve scientific computing problems while respecting physical laws.

**How it affects output:** PINNs ensure solutions satisfy physical laws, enabling neural networks to solve scientific problems with built-in physical constraints.

---

## Nonlinear Functions

> 📓 **Jupyter Notebook:** [pynb/math_ref/nonlinear_functions.ipynb](pynb/math_ref/nonlinear_functions.ipynb)

### Hyperbolic Tangent (tanh)

**Formula:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Intuition:** Activation function that squashes inputs to $(-1, 1)$. Provides nonlinearity needed for complex patterns while keeping outputs bounded. The S-shape introduces smooth nonlinear decision boundaries, and its zero-centered output helps with gradient flow compared to sigmoid.

**How it affects output:** Tanh produces zero-centered outputs helping gradient flow, but can suffer from vanishing gradients when inputs are large.

### Sigmoid

**Formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$

**Intuition:** Squashes inputs to $(0, 1)$, naturally interpreted as probabilities. Used in binary classification and gating mechanisms (LSTM gates). However, suffers from vanishing gradients at extremes, which is why ReLU became more popular in deep networks.

**How it affects output:** Sigmoid outputs natural probabilities but suffers from vanishing gradients, making it problematic for deep networks but perfect for gates.

### ReLU

**Formula:** $\text{ReLU}(x) = \max(0, x)$

**Intuition:** Simple nonlinearity that sets negative values to zero. Solves vanishing gradient problem because gradient is either 0 or 1. Promotes sparsity (many neurons inactive) which makes networks more interpretable and efficient. Biologically inspired by neuron firing thresholds.

**How it affects output:** ReLU enables deep networks by preventing vanishing gradients and promoting sparsity, but can suffer from dying neurons.

---

## Probability & Statistics

> 📓 **Jupyter Notebook:** [pynb/math_ref/probability_statistics.ipynb](pynb/math_ref/probability_statistics.ipynb)

### Expectation

**Formula:** $\mathbb{E}[X] = \sum_x x \cdot P(X = x)$

**Intuition:** Average value of a random variable. In neural networks, we often work with expected loss over data distributions. Batch statistics, dropout, and stochastic optimization all rely on expectation to handle randomness in training and make models robust to unseen data.

**How it affects output:** Expectation provides theoretical foundation for loss functions, batch statistics, and stochastic optimization in neural networks.

### Variance

**Formula:** $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$

**Intuition:** Measures spread of a distribution. Critical for weight initialization (to prevent vanishing/exploding gradients) and batch normalization (to stabilize training). Understanding variance helps design networks that maintain good signal propagation through many layers.

**How it affects output:** Proper variance control through initialization and normalization prevents vanishing/exploding gradients and stabilizes training.

### Softmax

**Formula:** $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$

**Intuition:** Converts real values to probability distribution. Essential for multi-class classification and attention mechanisms. The exponential amplifies differences while ensuring outputs sum to 1, creating a "soft" version of selecting the maximum value that's differentiable for gradient-based learning.

**How it affects output:** Softmax creates valid probability distributions and enables differentiable "argmax" operations for classification and attention.

### Cross-Entropy Loss

**Formula:** $\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)$

**Intuition:** Measures difference between predicted and true probability distributions. Natural loss function for classification because it heavily penalizes confident wrong predictions. Mathematically connected to maximum likelihood estimation and information theory—minimizing cross-entropy maximizes the likelihood of correct predictions.

**How it affects output:** Cross-entropy encourages confident correct predictions while heavily penalizing confident mistakes, leading to well-calibrated classifiers.

---

## Optimization

> 📓 **Jupyter Notebook:** [pynb/math_ref/optimization.ipynb](pynb/math_ref/optimization.ipynb)

### Gradient Descent

**Formula:** $\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$

**Intuition:** Iteratively moves parameters in direction of steepest loss decrease. The fundamental learning algorithm for neural networks. The learning rate $\alpha$ controls step size—too large causes instability, too small causes slow convergence. Modern variants (Adam, RMSprop) adapt the learning rate automatically.

**How it affects output:** Gradient descent provides the fundamental mechanism for learning by iteratively improving parameters based on loss gradients.

### Adam Optimizer

**Formula:** $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon}\hat{m}_t$

**Intuition:** Adaptive learning rate optimizer that maintains running averages of gradients (momentum) and squared gradients (variance). Automatically adjusts learning rates per parameter based on historical gradients. Like cruise control for optimization - speeds up in flat areas, slows down in steep areas.

**How it affects output:** Adam adapts learning rates per parameter, leading to faster convergence and better handling of sparse gradients compared to SGD.

---

## Attention Mechanisms

> 📓 **Jupyter Notebook:** [pynb/math_ref/attention_mechanisms.ipynb](pynb/math_ref/attention_mechanisms.ipynb)

### Scaled Dot-Product Attention

**Formula:** $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

**Intuition:** The core operation of transformers. Queries (Q) search through keys (K) to find relevant information, then retrieve corresponding values (V). The scaling by $\sqrt{d_k}$ prevents attention from becoming too sharp/peaked as dimensions grow, maintaining good gradient flow and distributed attention weights.

**How it affects output:** Attention allows models to dynamically focus on relevant parts of the input, enabling long-range dependencies and context-aware representations.

---

## Transformer Components

> 📓 **Jupyter Notebook:** [pynb/math_ref/transformer_components.ipynb](pynb/math_ref/transformer_components.ipynb)

### Layer Normalization

**Formula:** $\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$ where $\mu, \sigma$ computed per sample

**Intuition:** Normalizes activations within each sample to have zero mean and unit variance. Unlike batch normalization, works independently for each sample, making it stable for variable sequence lengths and small batch sizes. Helps with gradient flow and training stability.

**How it affects output:** Layer normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift and enabling higher learning rates.

### Residual Connections

**Formula:** $\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l)$

**Intuition:** Creates "gradient highways" that allow information to flow directly through the network. Essential for training very deep networks (like transformers with many layers) by preventing vanishing gradients. Acts like a safety net - even if some layers learn poorly, information can still reach the output.

**How it affects output:** Residual connections enable training of much deeper networks by providing gradient highways and allowing layers to learn incremental changes.

---

## Advanced Concepts

> 📓 **Jupyter Notebook:** [pynb/math_ref/advanced_concepts.ipynb](pynb/math_ref/advanced_concepts.ipynb)

### Temperature Scaling

**Formula:** $\text{softmax}(z/\tau)$ where $\tau$ is temperature

**Intuition:** Controls the "sharpness" of probability distributions. Lower temperature makes the distribution more peaked (confident), higher temperature makes it more uniform (uncertain). Used in generation for creativity control and in calibration to match predicted confidence with actual accuracy.

**How it affects output:** Temperature scaling controls the confidence/uncertainty trade-off in model outputs, enabling calibrated predictions and controllable generation creativity.
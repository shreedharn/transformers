# Mathematical Quick Reference for Neural Networks

A comprehensive reference of core mathematical concepts used in neural networks and deep learning, with detailed intuitions and practical PyTorch examples.

## Linear Algebra

> 📓 Jupyter Notebook: [pynb/math_ref/linear_algebra.ipynb](pynb/math_ref/linear_algebra.ipynb)

### Matrix Multiplication
$$
\begin{aligned}
\mathbf{C} &= \mathbf{A}\mathbf{B} \newline
C\_{ij} &= \sum\_k A\_{ik}B\_{kj}
\end{aligned}
$$
Intuition: The fundamental operation of neural networks. Each row of the weight matrix represents neuron weights, each column of the input matrix represents input vectors. The result computes weighted sums for all neurons simultaneously, enabling parallel computation. This is why a single matrix multiplication can represent an entire layer's forward pass.

How it affects output: Matrix multiplication transforms input vectors from one feature space to another. Small changes in weights create proportional changes in outputs, making gradient-based learning possible.

### Matrix Transpose
$$
\begin{aligned}
(\mathbf{A})^T\_{ij} = \mathbf{A}\_{ji}
\end{aligned}
$$
Intuition: Essential for backpropagation. When gradients flow backward through a layer with weights, we need the transpose to properly route the gradient signals back to the previous layer. The transpose "reverses" the forward direction of information flow.

How it affects output: Transpose changes how information flows. In forward pass, weights map inputs to outputs. In backward pass, transpose maps output gradients back to input gradients.

### Matrix Inverse
$$
\begin{aligned}
\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}
\end{aligned}
$$
Intuition: Used in analytical solutions for least squares (normal equations) and understanding linear transformations. In neural networks, helps analyze layer transformations and appears in second-order optimization methods like Newton's method.

How it affects output: Matrix inverse provides exact solutions when they exist. However, neural networks rarely use inverses directly due to computational cost and numerical instability.

### Eigenvalues & Eigenvectors
$$
\begin{aligned}
\mathbf{A}\mathbf{v} = \lambda\mathbf{v}
\end{aligned}
$$
Intuition: Reveals principal directions of data variation (PCA), helps analyze gradient flow and conditioning of weight matrices. Large eigenvalues can indicate exploding gradients, while small ones suggest vanishing gradients. Critical for understanding optimization landscapes.

How it affects output: Eigenvalues indicate how much each direction gets amplified. Large eigenvalues can cause exploding gradients, small ones cause vanishing gradients.

### Singular Value Decomposition (SVD)
$$
\begin{aligned}
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
\end{aligned}
$$
Intuition: Decomposes any matrix into orthogonal transformations and scaling. Used in dimensionality reduction, weight initialization, and analyzing the effective rank of learned representations. Helps understand what transformations neural network layers are actually learning.

How it affects output: SVD reveals the intrinsic dimensionality and structure of transformations. It's used for initialization to maintain gradient flow and for analyzing what neural networks learn.

---

## Vectors & Geometry

> 📓 Jupyter Notebook: [pynb/math_ref/vectors_geometry.ipynb](pynb/math_ref/vectors_geometry.ipynb)

### Dot Product
$$
\begin{aligned}
\mathbf{a} \cdot \mathbf{b} = \sum\_i a\_i b\_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos(\theta)
\end{aligned}
$$
Intuition: Measures how "aligned" two vectors are. In neurons, the dot product between input vectors and weight vectors gives the raw activation strength—high when input pattern matches what the neuron is looking for. This is the core operation that determines neuron firing.

How it affects output: High dot product means input and weight vectors point in similar directions, creating strong positive activation. Orthogonal vectors produce zero activation.

### Cosine Similarity
$$
\begin{aligned}
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}
\end{aligned}
$$
Intuition: Measures similarity independent of magnitude. Used in attention mechanisms, word embeddings, and similarity-based learning. Helps neural networks focus on directional patterns rather than absolute magnitudes, making them more robust to scaling variations.

How it affects output: Cosine similarity ranges from -1 to 1, making it scale-invariant. Similar directions get high scores regardless of vector magnitude.

### Euclidean Distance
$$
\begin{aligned}
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|\_2 = \sqrt{\sum\_i (a\_i - b\_i)^2}
\end{aligned}
$$
Intuition: Measures how "far apart" two points are in feature space. Used in loss functions (MSE), clustering, and nearest neighbor methods. In neural networks, helps measure prediction errors and similarity between representations.

How it affects output: Smaller distances indicate more similar points. MSE loss penalizes large errors quadratically, making the model focus on reducing big mistakes first.

### Lp Norms
$$
\begin{aligned}
\|\mathbf{x}\|\_p = \left(\sum\_i |x\_i|^p\right)^{1/p}
\end{aligned}
$$
Intuition: Measures vector "size" in different ways. L1 norm promotes sparsity (many weights become zero), L2 norm promotes smoothness (weights stay small). Used in regularization to control model complexity and prevent overfitting by penalizing large weights.

How it affects output: L1 norm creates sparse solutions (many zeros), L2 norm creates smooth solutions (small values). Higher p values focus more on the largest elements.

---

## Calculus

> 📓 Jupyter Notebook: [pynb/math_ref/calculus.ipynb](pynb/math_ref/calculus.ipynb)

### Derivatives
$$
\begin{aligned}
f'(x) = \lim\_{h \to 0} \frac{f(x+h) - f(x)}{h}
\end{aligned}
$$
Intuition: Measures how fast a function changes. In neural networks, tells us how much the loss changes when we tweak a parameter. This is the foundation of gradient-based learning—we follow the derivative to find better parameter values.

How it affects output: Derivatives tell us sensitivity - how much output changes for small input changes. Essential for parameter updates.

### Partial Derivatives
$$
\begin{aligned}
\frac{\partial f}{\partial x\_i}
\end{aligned}
$$
Intuition: Derivative with respect to one variable while holding others constant. Neural networks have millions of parameters, so we need partial derivatives to see how the loss changes with respect to each individual weight or bias.

How it affects output: Each parameter gets its own gradient, allowing independent optimization of millions of parameters simultaneously.

### Chain Rule
$$
\begin{aligned}
\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)
\end{aligned}
$$
Intuition: The mathematical foundation of backpropagation. Neural networks are compositions of functions (layer after layer), so to compute gradients we multiply derivatives along the chain from output back to input. This is why it's called "backpropagation"—propagating derivatives backward through the composition.

How it affects output: Chain rule enables automatic differentiation through arbitrarily deep networks by systematically applying the composition rule.

### Gradient
$$
\begin{aligned}
\nabla f = \left[\frac{\partial f}{\partial x\_1}, \frac{\partial f}{\partial x\_2}, \ldots, \frac{\partial f}{\partial x\_n}\right]
\end{aligned}
$$
Intuition: Points in the direction of steepest increase. In neural network training, we move parameters in the negative gradient direction (steepest decrease) to minimize the loss function. The gradient tells us both direction and magnitude of the best parameter update.

How it affects output: Gradient provides both direction (sign) and magnitude for optimal parameter updates. Larger gradients indicate more sensitive parameters.

### Hessian
$$
\begin{aligned}
\mathbf{H}\_{ij} = \frac{\partial^2 f}{\partial x\_i \partial x\_j}
\end{aligned}
$$
Intuition: Matrix of second derivatives that describes the curvature of the loss surface. Helps understand convergence behavior and is used in advanced optimization methods like Newton's method and natural gradients. High curvature areas require smaller learning rates.

How it affects output: Hessian reveals optimization landscape curvature. High condition numbers indicate difficult optimization requiring careful learning rates.

### Jacobian
$$
\begin{aligned}
\mathbf{J}\_{ij} = \frac{\partial f\_i}{\partial x\_j}
\end{aligned}
$$
Intuition: Matrix of first derivatives for vector-valued functions. Essential for backpropagation through layers that output vectors (like hidden layers). Each element shows how one output component changes with respect to one input component, enabling gradient flow through complex architectures.

How it affects output: Jacobian enables gradient flow through vector outputs, crucial for multi-output layers and complex architectures.

---

## Differential Equations

> 📓 Jupyter Notebook: [pynb/math_ref/differential_equations.ipynb](pynb/math_ref/differential_equations.ipynb)

### Ordinary Differential Equations (ODEs)
$$
\begin{aligned}
\frac{dy}{dt} = f(y, t)
\end{aligned}
$$
Intuition: Models how quantities change over time. Neural ODEs treat layer depth as continuous time, allowing adaptive depth and memory-efficient training. ResNets approximate the solution to ODEs, explaining why skip connections work so well for deep networks.

How it affects output: ODEs provide continuous depth, memory efficiency, and theoretical foundations for understanding deep networks as dynamical systems.

### Partial Differential Equations (PDEs)
$$
\begin{aligned}
\frac{\partial u}{\partial t} = f\left(u, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \ldots\right)
\end{aligned}
$$
Intuition: Models complex spatiotemporal phenomena. Physics-informed neural networks (PINNs) embed PDE constraints directly into the loss function, allowing neural networks to solve scientific computing problems while respecting physical laws.

How it affects output: PINNs ensure solutions satisfy physical laws, enabling neural networks to solve scientific problems with built-in physical constraints.

---

## Nonlinear Functions

> 📓 Jupyter Notebook: [pynb/math_ref/nonlinear_functions.ipynb](pynb/math_ref/nonlinear_functions.ipynb)

### Hyperbolic Tangent (tanh)
$$
\begin{aligned}
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\end{aligned}
$$
Intuition: Activation function that squashes inputs to (-1, 1). Provides nonlinearity needed for complex patterns while keeping outputs bounded. The S-shape introduces smooth nonlinear decision boundaries, and its zero-centered output helps with gradient flow compared to sigmoid.

How it affects output: Tanh produces zero-centered outputs helping gradient flow, but can suffer from vanishing gradients when inputs are large.

### Sigmoid
$$
\begin{aligned}
\sigma(x) = \frac{1}{1 + e^{-x}}
\end{aligned}
$$
Intuition: Squashes inputs to (0, 1), naturally interpreted as probabilities. Used in binary classification and gating mechanisms (LSTM gates). However, suffers from vanishing gradients at extremes, which is why ReLU became more popular in deep networks.

How it affects output: Sigmoid outputs natural probabilities but suffers from vanishing gradients, making it problematic for deep networks but perfect for gates.

### ReLU
$$
\begin{aligned}
\text{ReLU}(x) = \max(0, x)
\end{aligned}
$$
Intuition: Simple nonlinearity that sets negative values to zero. Solves vanishing gradient problem because gradient is either 0 or 1. Promotes sparsity (many neurons inactive) which makes networks more interpretable and efficient. Biologically inspired by neuron firing thresholds.

How it affects output: ReLU enables deep networks by preventing vanishing gradients and promoting sparsity, but can suffer from dying neurons.

---

## Probability & Statistics

> 📓 Jupyter Notebook: [pynb/math_ref/probability_statistics.ipynb](pynb/math_ref/probability_statistics.ipynb)

### Expectation
$$
\begin{aligned}
\mathbb{E}[X] = \sum\_x x \cdot P(X = x)
\end{aligned}
$$
Intuition: Average value of a random variable. In neural networks, we often work with expected loss over data distributions. Batch statistics, dropout, and stochastic optimization all rely on expectation to handle randomness in training and make models robust to unseen data.

How it affects output: Expectation provides theoretical foundation for loss functions, batch statistics, and stochastic optimization in neural networks.

### Variance
$$
\begin{aligned}
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
\end{aligned}
$$
Intuition: Measures spread of a distribution. Critical for weight initialization (to prevent vanishing/exploding gradients) and batch normalization (to stabilize training). Understanding variance helps design networks that maintain good signal propagation through many layers.

How it affects output: Proper variance control through initialization and normalization prevents vanishing/exploding gradients and stabilizes training.

### Softmax
$$
\begin{aligned}
\text{softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_j e^{x\_j}}
\end{aligned}
$$
Intuition: Converts real values to probability distribution. Essential for multi-class classification and attention mechanisms. The exponential amplifies differences while ensuring outputs sum to 1, creating a "soft" version of selecting the maximum value that's differentiable for gradient-based learning.

How it affects output: Softmax creates valid probability distributions and enables differentiable "argmax" operations for classification and attention.

### Cross-Entropy Loss
$$
\begin{aligned}
\mathcal{L} = -\sum\_i y\_i \log(\hat{y}\_i)
\end{aligned}
$$
Intuition: Measures difference between predicted and true probability distributions. Natural loss function for classification because it heavily penalizes confident wrong predictions. Mathematically connected to maximum likelihood estimation and information theory—minimizing cross-entropy maximizes the likelihood of correct predictions.

How it affects output: Cross-entropy encourages confident correct predictions while heavily penalizing confident mistakes, leading to well-calibrated classifiers.

---

## Optimization

> 📓 Jupyter Notebook: [pynb/math_ref/optimization.ipynb](pynb/math_ref/optimization.ipynb)

### Gradient Descent
$$
\begin{aligned}
\theta\_{t+1} = \theta\_t - \alpha \nabla\_\theta \mathcal{L}(\theta\_t)
\end{aligned}
$$
Intuition: Iteratively moves parameters in direction of steepest loss decrease. The fundamental learning algorithm for neural networks. The learning rate controls step size—too large causes instability, too small causes slow convergence. Modern variants (Adam, RMSprop) adapt the learning rate automatically.

How it affects output: Gradient descent provides the fundamental mechanism for learning by iteratively improving parameters based on loss gradients.

### Adam Optimizer
$$
\begin{aligned}
m\_t &= \beta\_1 m\_{t-1} + (1-\beta\_1)g\_t \newline
v\_t &= \beta\_2 v\_{t-1} + (1-\beta\_2)g\_t^2 \newline
\theta\_t &= \theta\_{t-1} - \frac{\alpha}{\sqrt{v\_t} + \epsilon}\hat{m}\_t
\end{aligned}
$$
Intuition: Adaptive learning rate optimizer that maintains running averages of gradients (momentum) and squared gradients (variance). Automatically adjusts learning rates per parameter based on historical gradients. Like cruise control for optimization - speeds up in flat areas, slows down in steep areas.

How it affects output: Adam adapts learning rates per parameter, leading to faster convergence and better handling of sparse gradients compared to SGD.

---

## Attention Mechanisms

> 📓 Jupyter Notebook: [pynb/math_ref/attention_mechanisms.ipynb](pynb/math_ref/attention_mechanisms.ipynb)

### Scaled Dot-Product Attention
$$
\begin{aligned}
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$
Intuition: The core operation of transformers. Queries (Q) search through keys (K) to find relevant information, then retrieve corresponding values (V). The scaling prevents attention from becoming too sharp/peaked as dimensions grow, maintaining good gradient flow and distributed attention weights.

How it affects output: Attention allows models to dynamically focus on relevant parts of the input, enabling long-range dependencies and context-aware representations.

---

## Transformer Components

> 📓 Jupyter Notebook: [pynb/math_ref/transformer_components.ipynb](pynb/math_ref/transformer_components.ipynb)

### Layer Normalization
$$
\begin{aligned}
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta \quad \text{where } \mu, \sigma \text{ computed per sample}
\end{aligned}
$$
Intuition: Normalizes activations within each sample to have zero mean and unit variance. Unlike batch normalization, works independently for each sample, making it stable for variable sequence lengths and small batch sizes. Helps with gradient flow and training stability.

How it affects output: Layer normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift and enabling higher learning rates.

### Residual Connections
$$
\begin{aligned}
\mathbf{h}\_{l+1} = \mathbf{h}\_l + F(\mathbf{h}\_l)
\end{aligned}
$$
Intuition: Creates "gradient highways" that allow information to flow directly through the network. Essential for training very deep networks (like transformers with many layers) by preventing vanishing gradients. Acts like a safety net - even if some layers learn poorly, information can still reach the output.

How it affects output: Residual connections enable training of much deeper networks by providing gradient highways and allowing layers to learn incremental changes.

---

## Advanced Concepts

> 📓 Jupyter Notebook: [pynb/math_ref/advanced_concepts.ipynb](pynb/math_ref/advanced_concepts.ipynb)

### Temperature Scaling
$$
\begin{aligned}
\text{softmax}(z/\tau) \quad \text{where } \tau \text{ is temperature}
\end{aligned}
$$
Intuition: Controls the "sharpness" of probability distributions. Lower temperature makes the distribution more peaked (confident), higher temperature makes it more uniform (uncertain). Used in generation for creativity control and in calibration to match predicted confidence with actual accuracy.

How it affects output: Temperature scaling controls the confidence/uncertainty trade-off in model outputs, enabling calibrated predictions and controllable generation creativity.
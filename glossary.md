# Glossary: Neural Networks and Transformers

A comprehensive dictionary of key terms, concepts, and technical vocabulary used throughout the neural networks and transformers learning materials. Each term includes cross-references to detailed explanations in the repository documents.

## A

Activation Function: A mathematical function applied to the output of a neuron to introduce non-linearity. Without activation functions, neural networks would only be able to learn linear relationships.

- Common types:

  $$
\begin{aligned}
\text{ReLU:} \quad &f(x) = \max(0, x) \newline
  \text{Sigmoid:} \quad &\sigma(x) = \frac{1}{1+e^{-x}} \newline
  \text{Tanh:} \quad &\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\end{aligned}
$$
- Deep dive: [nn_intro.md Section 3](./nn_intro.md#the-role-of-activation-functions-space-warping) for geometric intuition, [mlp_intro.md Section 9](./mlp_intro.md#9-activation-functions-deep-dive) for detailed comparison

Attention Collapse: Phenomenon where attention weights become too peaked (concentrated on few tokens) rather than uniform, leading to poor gradient flow and reduced model expressiveness.

- Mathematical foundation: [transformers_math1.md Section 5.2](./transformers_math1.md#52-why-the-sqrtd_k-scaling) for scaling analysis

Adam Optimizer: An adaptive optimization algorithm that combines momentum with per-parameter learning rate adaptation.

- Formula:

  $$
\begin{aligned}
\theta\_t = \theta\_{t-1} - \alpha \frac{\hat{m}\_t}{\sqrt{\hat{v}\_t} + \epsilon} \quad \text{where } \hat{m}\_t, \hat{v}\_t \text{ are bias-corrected moment estimates}
\end{aligned}
$$
- Details: [nn_intro.md Section 5](./nn_intro.md#adam-adaptive-moments) for intuition, [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for implementation

Artificial Intelligence (AI): Computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and natural language understanding.

- Context: [nn_intro.md Section 1](./nn_intro.md#1-what-is-ai-ml-and-deep-learning) for AI/ML/DL hierarchy

Attention Head: A specialized component of multi-head attention that focuses on specific types of relationships (e.g., grammar, semantics, position).

- Implementation: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for complete technical details
- Code: [pytorch_ref.md Section 10](./pytorch_ref.md#self-attention-from-scratch) for from-scratch implementation

Attention Mechanism: A technique that allows models to focus on relevant parts of the input sequence when making predictions.

- Formula:

  $$
\begin{aligned}
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$
- Explanation: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for detailed mechanics

---

## B

Backpropagation: The algorithm used to train neural networks by calculating gradients and propagating errors backward through the network layers.

- Mathematical foundation: Uses chain rule to compute gradients:

  $$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial h^{(l+1)}} \cdot \frac{\partial h^{(l+1)}}{\partial W^{(l)}}
\end{aligned}
$$
- Step-by-step: [mlp_intro.md Section 6](./mlp_intro.md#backpropagation-the-learning-algorithm) for detailed derivation
- Implementation: [pytorch_ref.md Section 3](./pytorch_ref.md#3-autograd-finding-gradients) for automatic differentiation

Batch Normalization: A technique that normalizes layer inputs to stabilize training and accelerate convergence.

- Formula:

  $$
\begin{aligned}
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad \text{followed by learned scaling and shifting}
\end{aligned}
$$
- Details: [pytorch_ref.md Section 6](./pytorch_ref.md#6-vanishingexploding-gradients) for gradient stabilization context

BERT (Bidirectional Encoder Representations from Transformers): A transformer-based model that reads text bidirectionally for better context understanding.

- Architecture: [transformers_fundamentals.md Section 8](./transformers_fundamentals.md#encoder-only-bert-family) for encoder-only design
- Training: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)

Bias: An additional parameter in a neuron that allows the activation function to shift, providing flexibility in the decision boundary.

- Mathematical role: Shifts hyperplane:

  $$
\begin{aligned}
z = Wx + b
\end{aligned}
$$
- Geometric intuition: [nn_intro.md Section 3](./nn_intro.md#the-role-of-bias-flexible-positioning) for spatial understanding
- Examples: [mlp_intro.md Section 5](./mlp_intro.md#5-worked-example-advanced-spam-detection) for worked calculations

---

## C

Causal Mask: Lower-triangular mask preventing attention to future tokens in autoregressive models.

- Mathematical implementation: [transformers_math1.md Section 5.4](./transformers_math1.md#54-masked-attention) for masking details

Centroid: The center point of a cluster in vector space, representing the average position of all vectors in that cluster.

- Context: [knowledge_store.md](./knowledge_store.md) for vector search applications

Cross-Entropy Loss: The standard loss function for classification tasks that measures the difference between predicted and true probability distributions.

- Formula:

  $$
\begin{aligned}
\mathcal{L} = -\sum\_{i=1}^{C} y\_i \log(p\_i) \quad \text{where } y\_i \text{ is true label and } p\_i \text{ is predicted probability}
\end{aligned}
$$
- Intuition: [nn_intro.md Section 5](./nn_intro.md#for-classification-problems) for geometric interpretation
- Implementation: [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for PyTorch usage

---

## D

Deep Learning: A subset of machine learning using neural networks with multiple hidden layers (typically 3 or more) to learn complex patterns in data.

- Foundations: [nn_intro.md Section 1](./nn_intro.md#deep-learning-dl) for definition and examples
- Why it works: [nn_intro.md Section 2](./nn_intro.md#2-why-deep-learning-for-nlp) for NLP advantages

Dropout: A regularization technique that randomly sets some neurons to zero during training to prevent overfitting.

- Implementation: [mlp_intro.md Section 8](./mlp_intro.md#overfitting-when-mlps-memorize) for overfitting solutions
- Code: [pytorch_ref.md Section 8](./pytorch_ref.md#8-mlps-in-pytorch) for practical usage

---

## E

Embedding: A dense numerical vector representation of text, images, or other data that captures semantic meaning in high-dimensional space.

- Mathematical foundation:

  $$
\begin{aligned}
\mathbf{e}\_i = E[i] \in \mathbb{R}^{d\_{\text{model}}} \quad \text{where } E \text{ is the embedding matrix}
\end{aligned}
$$
- Geometric intuition: [nn_intro.md Section 5](./nn_intro.md#text-embeddings-bridging-language-and-mathematics) for complete explanation
- Applications: [knowledge_store.md](./knowledge_store.md) for semantic search and knowledge storage

Encoder: In transformer architecture, the component that processes input sequences to create contextualized representations.

- Architecture: [transformers_fundamentals.md Section 8](./transformers_fundamentals.md#encoder-only-bert-family) for encoder-only models
- vs Decoder: [transformers_fundamentals.md Section 8](./transformers_fundamentals.md#8-architectural-variants-encoder-decoder-and-encoder-decoder) for comparison

Epoch: One complete pass through the entire training dataset during neural network training.

- Training context: [nn_intro.md Section 5](./nn_intro.md#1-epochs) for training concepts
- Implementation: [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for training loops

---

## F

Feed-Forward Network: A neural network component where information flows in one direction, typically used within transformer blocks.

- In transformers: [transformers_fundamentals.md Section 11](./transformers_fundamentals.md#11-stage-6-feed-forward-networks) for detailed explanation
- Implementation: Two linear transformations with activation:

  $$
\begin{aligned}
\text{FFN}(x) = \max(0, xW\_1 + b\_1)W\_2 + b\_2
\end{aligned}
$$

---

## G

GPT (Generative Pre-trained Transformer): A family of decoder-only transformer models designed for text generation.

- Architecture: [transformers_fundamentals.md Section 8](./transformers_fundamentals.md#decoder-only-gpt-family) for decoder-only design
- Training: Causal Language Modeling (CLM) for next-token prediction

Gradient Descent: An optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent.

- Mathematical foundation:

  $$
\begin{aligned}
\theta\_{\text{new}} = \theta\_{\text{old}} - \alpha \nabla\_{\theta} \mathcal{L}
\end{aligned}
$$
- Intuition: [nn_intro.md Section 5](./nn_intro.md#gradient-descent-the-universal-learning-algorithm) for complete explanation
- Variants: [nn_intro.md Section 5](./nn_intro.md#from-simple-to-sophisticated-the-evolution-of-optimizers) for SGD, momentum, Adam

GRU (Gated Recurrent Unit): A type of RNN with gating mechanisms that help handle long-term dependencies, simpler than LSTM.

- Implementation: [pytorch_ref.md Section 9](./pytorch_ref.md#rnn-vs-lstm-vs-gru-comparison) for comparison with RNN/LSTM
- Evolution: [rnn_intro.md Section 10](./rnn_intro.md#10-evolution-beyond-vanilla-rnns) for historical context

---

## H

Hidden Layer: Layers in a neural network between the input and output layers that process and transform the data.

- vs Hidden State: [rnn_intro.md Section 4](./rnn_intro.md#4-understanding-hidden-states-vs-hidden-layers) for crucial distinction
- In RNNs: [rnn_intro.md Section 4](./rnn_intro.md#4-understanding-hidden-states-vs-hidden-layers) for memory vs architecture

Hidden State: The internal representation vector that flows through a neural network at a specific processing step.

- Mathematical definition:

  $$
\begin{aligned}
h^{(l)} = f(W^{(l)}h^{(l-1)} + b^{(l)}) \quad \text{for layer } l
\end{aligned}
$$
- In RNNs: [rnn_intro.md Section 3](./rnn_intro.md#3-the-core-rnn-equation) for memory across time steps
- Worked example: [rnn_intro.md Section 7](./rnn_intro.md#7-worked-example-cat-sat-here) for "cat sat here" processing

HNSW (Hierarchical Navigable Small World): A graph-based indexing method that creates multiple navigation layers for fast approximate nearest neighbor search.

- Context: [knowledge_store.md](./knowledge_store.md) for vector database indexing

Hyperparameter: Configuration settings for machine learning models that are set before training begins (learning rate, batch size, etc.).

- Tuning guide: [mlp_intro.md Section 10](./mlp_intro.md#hyperparameter-tuning) for practical advice

---

## I

IVF (Inverted File): A clustering-based indexing method that groups similar vectors together and searches only within relevant clusters.

- Vector search: [knowledge_store.md](./knowledge_store.md) for database optimization techniques

---

## K

KV Cache: Stored key-value pairs from previous tokens to accelerate autoregressive generation.

- Mathematical foundation: [transformers_math2.md Section 10.4](./transformers_math2.md#104-kv-caching-for-autoregressive-generation) for efficiency details

---

## L

Layer Normalization: A normalization technique applied within transformer layers to stabilize training.

- Formula: Applied to each position independently across the feature dimension
- In transformers: [transformers_fundamentals.md Section 7](./transformers_fundamentals.md#7-stage-3-through-the-transformer-stack) for detailed explanation

Learning Rate: A hyperparameter that controls the step size in gradient descent optimization.

- Importance: [nn_intro.md Section 5](./nn_intro.md#the-learning-rate-α-speed-vs-accuracy-trade-off) for intuitive explanation
- Schedules: [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for dynamic adjustment

Loss Function: A function that measures how well the neural network's predictions match the actual target values.

- Types: Cross-entropy for classification, MSE for regression
- Deep dive: [nn_intro.md Section 5](./nn_intro.md#loss-functions-the-networks-report-card) for complete explanation
- Implementation: [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for PyTorch examples

LSTM (Long Short-Term Memory): A type of RNN with gating mechanisms designed to handle long-term dependencies and mitigate vanishing gradients.

- Architecture: [pytorch_ref.md Section 9](./pytorch_ref.md#why-gating-mechanisms) for gating explanation
- vs RNN: [rnn_intro.md Section 10](./rnn_intro.md#10-evolution-beyond-vanilla-rnns) for improvements over vanilla RNNs

---

## M

Machine Learning (ML): A subset of AI where systems learn patterns from data without being explicitly programmed for every scenario.

- vs AI vs DL: [nn_intro.md Section 1](./nn_intro.md#1-what-is-ai-ml-and-deep-learning) for clear hierarchy
- Traditional methods: [nn_intro.md Section 2](./nn_intro.md#challenges-with-traditional-ml-for-text) for comparison with deep learning

Masked Language Modeling (MLM): A training objective where some tokens are masked and the model learns to predict them.

- In BERT: [transformers_advanced.md](./transformers_advanced.md) for bidirectional training context

Multi-Head Attention: An extension of attention that runs multiple attention mechanisms in parallel to capture different types of relationships.

- Formula:

  $$
\begin{aligned}
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}\_1, ..., \text{head}\_h)W^O
\end{aligned}
$$
- Complete explanation: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for mathematical details
- Mathematical foundation: [transformers_math1.md Section 6.1](./transformers_math1.md#61-multi-head-as-subspace-projections) for subspace projections
- Implementation: [pytorch_ref.md Section 10](./pytorch_ref.md#10-transformers-in-pytorch) for code examples

Multi-Layer Perceptron (MLP): A neural network with one or more hidden layers between input and output layers.

- Complete tutorial: [mlp_intro.md](./mlp_intro.md) for step-by-step explanation
- vs single perceptron: [mlp_intro.md Section 1](./mlp_intro.md#mlp-vs-single-perceptron) for capability comparison
- Worked example: [mlp_intro.md Section 5](./mlp_intro.md#5-worked-example-advanced-spam-detection) for detailed calculations

---

## N

Natural Language Processing (NLP): A field of AI focused on enabling computers to understand, interpret, and generate human language.

- Why deep learning: [nn_intro.md Section 2](./nn_intro.md#2-why-deep-learning-for-nlp) for advantages over traditional methods
- Applications: [nn_intro.md Section 6](./nn_intro.md#6-where-neural-networks-shine-in-nlp) for practical uses

Neural Network: A computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information.

- Foundation: [nn_intro.md Section 3](./nn_intro.md#3-the-neuron-and-the-perceptron) for basic building blocks
- Geometric view: [nn_intro.md Section 4](./nn_intro.md#geometric-intuition-from-1d-to-n-d) for spatial understanding

---

## O

Optimizer: An algorithm that updates neural network parameters to minimize the loss function.

- Types: SGD, Adam, RMSprop
- Comparison: [nn_intro.md Section 5](./nn_intro.md#from-simple-to-sophisticated-the-evolution-of-optimizers) for evolution
- PyTorch: [pytorch_ref.md Section 5](./pytorch_ref.md#5-optimization-loop--losses) for practical implementation

Overfitting: When a model performs well on training data but poorly on new, unseen data because it has memorized rather than learned generalizable patterns.

- Solutions: [mlp_intro.md Section 8](./mlp_intro.md#overfitting-when-mlps-memorize) for regularization techniques
- vs Underfitting: [nn_intro.md Section 5](./nn_intro.md#4-overfitting-vs-underfitting) for comparison

---

## P

Perceptron: The basic building block of neural networks, consisting of inputs, weights, a bias, and an activation function.

- Formula:

  $$
\begin{aligned}
y = f\left(\sum\_{i=1}^{n} w\_i x\_i + b\right)
\end{aligned}
$$
- Complete explanation: [nn_intro.md Section 3](./nn_intro.md#3-the-neuron-and-the-perceptron) for biological inspiration and mathematics
- Limitations: [nn_intro.md Section 4](./nn_intro.md#limitations-of-single-perceptrons) for XOR problem

Position Encoding: Method to inject sequential order information into permutation-equivariant attention.

- Mathematical foundation: [transformers_fundamentals.md Section 6](./transformers_fundamentals.md#6-stage-2-tokens-to-embeddings) for sinusoidal encoding
- Advanced techniques: [transformers_math1.md Section 6.2](./transformers_math1.md#62-advanced-positional-encodings) for RoPE and ALiBi
- Implementation: [pytorch_ref.md Section 10](./pytorch_ref.md#10-transformers-in-pytorch) for code examples

Product Quantization (PQ): A compression technique that splits vectors into chunks and replaces each chunk with a representative centroid ID.

- Vector databases: [knowledge_store.md](./knowledge_store.md) for storage optimization

---

## Q

Query Vector: In attention mechanisms, the vector representing "what information is being sought" from other positions.

- Q, K, V mechanism: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for complete attention explanation

---

## R

RAG (Retrieval-Augmented Generation): A system that combines vector store retrieval with LLM generation to provide informed, grounded responses.

- Architecture: [knowledge_store.md](./knowledge_store.md) for implementation patterns

ReLU (Rectified Linear Unit): An activation function that outputs the input if positive, zero otherwise.

- Formula:

  $$
\begin{aligned}
f(x) = \max(0, x)
\end{aligned}
$$
- Advantages: [nn_intro.md Section 3](./nn_intro.md#common-activation-functions) for vanishing gradient prevention
- Geometric effect: [nn_intro.md Section 4](./nn_intro.md#step-2-relu-activation-bends-space) for space folding in XOR example

RNN (Recurrent Neural Network): A neural network designed for sequential data that maintains hidden states across time steps.

- Core equation:

  $$
\begin{aligned}
h\_t = f(W\_x x\_t + W\_h h\_{t-1} + b)
\end{aligned}
$$
- Complete tutorial: [rnn_intro.md](./rnn_intro.md) for step-by-step explanation
- Limitations: [rnn_intro.md Section 9](./rnn_intro.md#9-the-vanishing-gradient-problem-rnns-fatal-flaw) for vanishing gradients

Regularization: Techniques to prevent overfitting by constraining model complexity.

- Methods: Dropout, L1/L2 regularization, early stopping
- Practical guide: [mlp_intro.md Section 8](./mlp_intro.md#8-common-challenges-and-solutions) for implementation

---

## S

Scaled Dot-Product Attention: Core attention mechanism with the following formula:

$$
\begin{aligned}
\text{softmax}(QK^T/\sqrt{d\_k})V
\end{aligned}
$$
- Mathematical derivation: [transformers_math1.md Section 5.1](./transformers_math1.md#51-deriving-scaled-dot-product-attention) for complete derivation
- Implementation: [pytorch_ref.md Section 10](./pytorch_ref.md#self-attention-from-scratch) for from-scratch code

Self-Attention: An attention mechanism where queries, keys, and values all come from the same sequence, allowing positions to attend to each other.

- Formula:

  $$
\begin{aligned}
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_k}}\right)V
\end{aligned}
$$
- Implementation: [pytorch_ref.md Section 10](./pytorch_ref.md#self-attention-from-scratch) for from-scratch code
- Intuition: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for detailed mechanics

Sigmoid: An activation function that maps any input to a value between 0 and 1.

- Formula:

  $$
\begin{aligned}
\sigma(x) = \frac{1}{1 + e^{-x}}
\end{aligned}
$$
- Properties: [nn_intro.md Section 3](./nn_intro.md#common-activation-functions) for squashing behavior
- Uses: Binary classification, historical neural networks

Similarity Threshold: A minimum similarity score required for a document to be considered relevant in vector search.

- Vector search: [knowledge_store.md](./knowledge_store.md) for retrieval systems

Softmax: A function that converts a vector of real numbers into a probability distribution.

- Formula:

  $$
\begin{aligned}
\text{softmax}(x\_i) = \frac{e^{x\_i}}{\sum\_{j=1}^{K} e^{x\_j}}
\end{aligned}
$$
- Usage: Output layer for multi-class classification, attention weights

Stochastic Gradient Descent (SGD): A variant of gradient descent that uses random mini-batches instead of the full dataset.

- Benefits: [nn_intro.md Section 5](./nn_intro.md#from-simple-to-sophisticated-the-evolution-of-optimizers) for comparison with other optimizers

---

## T

Tanh (Hyperbolic Tangent): An activation function that maps inputs to values between -1 and 1.

- Formula:

  $$
\begin{aligned}
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\end{aligned}
$$
- Advantages: [nn_intro.md Section 3](./nn_intro.md#common-activation-functions) for zero-centered output
- Comparison: [mlp_intro.md Section 9](./mlp_intro.md#tanh-hyperbolic-tangent) for detailed analysis

Teacher Forcing: Training technique using ground truth tokens as inputs instead of model predictions.

- Autoregressive training: [transformers_math1.md Section 8.1](./transformers_math1.md#81-next-token-prediction) for implementation details

Temperature: A parameter controlling randomness in text generation; lower values make outputs more focused, higher values more creative.

- Text generation: Used in softmax:

  $$
\begin{aligned}
p\_i = \frac{e^{x\_i/T}}{\sum\_j e^{x\_j/T}}
\end{aligned}
$$

Token: The basic unit of text processing in NLP models (words, subwords, or characters).

- Tokenization: [transformers_fundamentals.md Section 5](./transformers_fundamentals.md#5-stage-1-text-to-tokens) for detailed explanation
- Embeddings: [nn_intro.md Section 5](./nn_intro.md#text-embeddings-bridging-language-and-mathematics) for vector representation

Top-K: A parameter limiting selection to the K most likely tokens (in LLMs) or K most similar documents (in vector stores).

- Sampling: Controls generation diversity in language models

Top-P (Nucleus Sampling): A parameter that dynamically selects tokens based on cumulative probability until reaching threshold P.

- vs Top-K: More adaptive selection for text generation

Transformer: A neural network architecture that uses self-attention mechanisms to process sequential data efficiently.

- Complete reference: [transformers_fundamentals.md](./transformers_fundamentals.md) for comprehensive technical details
- Key innovation: [transformers_fundamentals.md Section 9](./transformers_fundamentals.md#9-stage-4-self-attention-deep-dive) for attention mechanism explanation
- Implementation: [pytorch_ref.md Section 10](./pytorch_ref.md#10-transformers-in-pytorch) for code examples

---

## U

Universal Approximation Theorem: A mathematical theorem stating that neural networks with sufficient neurons can approximate any continuous function.

- Implication: [nn_intro.md Section 4](./nn_intro.md#multi-layer-perceptrons-mlps-high-dimensional-sculptors) for theoretical foundation
- Practical meaning: Justifies why neural networks are so powerful for complex pattern learning

Underfitting: When a model is too simple to capture the underlying patterns in the data.

- vs Overfitting: [nn_intro.md Section 5](./nn_intro.md#4-overfitting-vs-underfitting) for comparison
- Solutions: [mlp_intro.md Section 8](./mlp_intro.md#underfitting-when-mlps-are-too-simple) for model complexity increase

---

## V

Vanishing Gradient Problem: A fundamental issue in deep networks where gradients become exponentially small in earlier layers, preventing effective learning.

- Mathematical analysis: [rnn_intro.md Section 9](./rnn_intro.md#9-the-vanishing-gradient-problem-rnns-fatal-flaw) for RNN-specific issues
- Solutions: [pytorch_ref.md Section 6](./pytorch_ref.md#6-vanishingexploding-gradients) for practical fixes
- Why ReLU helps: [nn_intro.md Section 3](./nn_intro.md#common-activation-functions) for gradient preservation

Vector Store: A database optimized for storing and searching high-dimensional numerical vectors representing semantic content.

- Complete guide: [knowledge_store.md](./knowledge_store.md) for implementation, indexing, and comparison with LLM weights
- Applications: Semantic search, RAG systems, recommendation engines

Vocabulary: The complete set of unique tokens (words, subwords, characters) that a model can process.

- Size considerations: [transformers_fundamentals.md Section 5](./transformers_fundamentals.md#5-stage-1-text-to-tokens) for tokenization trade-offs

---

## W

Weight: Parameters in a neural network that determine the strength of connections between neurons and are learned during training.

- Mathematical role:

  $$
\begin{aligned}
z = w\_1x\_1 + w\_2x\_2 + ... + w\_nx\_n + b
\end{aligned}
$$
- Geometric interpretation: [nn_intro.md Section 3](./nn_intro.md#the-role-of-weights-feature-importance-and-direction) for spatial understanding
- Training: [nn_intro.md Section 5](./nn_intro.md#gradient-descent-the-universal-learning-algorithm) for optimization process

Word Embedding: A dense vector representation of words that captures semantic relationships.

- Mathematical foundation: Words mapped to high-dimensional space:

  $$
\begin{aligned}
\text{Words} \rightarrow \mathbb{R}^{d} \quad \text{where similar words have similar vectors}
\end{aligned}
$$
- Properties: [nn_intro.md Section 5](./nn_intro.md#why-embeddings-work) for distributional hypothesis
- Applications: [knowledge_store.md](./knowledge_store.md) for semantic search and knowledge storage

---

## 📚 Additional Resources

- Mathematical Foundations: [math_quick_ref.md](./math_quick_ref.md) for formulas and derivations
- Implementation Patterns: [pytorch_ref.md](./pytorch_ref.md) for practical coding examples  
- Historical Context: [history_quick_ref.md](./history_quick_ref.md) for evolution timeline
- Hands-on Examples: [mlp_intro.md](./mlp_intro.md) and [rnn_intro.md](./rnn_intro.md) for worked calculations

---

*This glossary covers fundamental terms from neural networks through transformer architectures. Each term includes cross-references to detailed explanations in the repository documents for deeper understanding.*
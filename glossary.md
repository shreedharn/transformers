# Glossary: Neural Networks and Transformers

A comprehensive dictionary of key terms, concepts, and technical vocabulary used throughout the neural networks and transformers learning materials.

## Table of Contents

- [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [G](#g) | [H](#h) | [I](#i) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [Q](#q) | [R](#r) | [S](#s) | [T](#t) | [U](#u) | [V](#v) | [W](#w)

---

## A

**Activation Function**: A mathematical function applied to the output of a neuron to introduce non-linearity. Common examples include ReLU, sigmoid, and tanh. Without activation functions, neural networks would only be able to learn linear relationships.

**Artificial Intelligence (AI)**: Computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and natural language understanding.

**Attention Head**: A specialized component of multi-head attention that focuses on specific types of relationships (e.g., grammar, semantics, position).

## B

**Backpropagation**: The algorithm used to train neural networks by calculating gradients and propagating errors backward through the network layers. It determines how much each weight contributed to the final error and adjusts them accordingly.

**Bias**: An additional parameter in a neuron that allows the activation function to shift, helping the network fit data better. It acts like an intercept term in linear regression, providing flexibility in the decision boundary.

## C

**Centroid**: The center point of a cluster in vector space, representing the average position of all vectors in that cluster.

## D

**Deep Learning**: A subset of machine learning using neural networks with multiple hidden layers (typically 3 or more) to learn complex patterns in data. The "deep" refers to the many layers that enable hierarchical feature learning.

## E

**Embedding**: A dense numerical vector representation of text, images, or other data that captures semantic meaning in high-dimensional space. Word embeddings allow neural networks to understand that similar words have similar meanings.

**Epoch**: One complete pass through the entire training dataset during neural network training. Multiple epochs are usually needed for the model to converge to optimal weights.

## G

**Gradient Descent**: An optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent. It's the core method for updating neural network weights during training.

## H

**Hidden Layer**: Layers in a neural network between the input and output layers that process and transform the data. These layers learn increasingly abstract representations of the input.

**HNSW (Hierarchical Navigable Small World)**: A graph-based indexing method that creates multiple navigation layers for fast approximate nearest neighbor search.

## I

**IVF (Inverted File)**: A clustering-based indexing method that groups similar vectors together and searches only within relevant clusters.

## L

**Loss Function**: A function that measures how well the neural network's predictions match the actual target values. Different tasks (classification, regression) use different loss functions.

## M

**Machine Learning (ML)**: A subset of AI where systems learn patterns from data without being explicitly programmed for every scenario. It includes both traditional methods (decision trees, SVM) and deep learning.

**Multi-Layer Perceptron (MLP)**: A neural network with one or more hidden layers between input and output layers. MLPs can learn non-linear patterns and are the foundation of deeper architectures.

## N

**Natural Language Processing (NLP)**: A field of AI focused on enabling computers to understand, interpret, and generate human language. It includes tasks like translation, sentiment analysis, and text generation.

**Neural Network**: A computing system inspired by biological neural networks, consisting of interconnected nodes (neurons) that process information. Each connection has a weight that determines the strength of the signal.

## O

**Overfitting**: When a model performs well on training data but poorly on new, unseen data because it has memorized rather than learned generalizable patterns. It's a common problem that requires regularization techniques to address.

## P

**Perceptron**: The basic building block of neural networks, consisting of inputs, weights, a bias, and an activation function. It performs a weighted sum of inputs and applies an activation function to produce an output.

**Product Quantization (PQ)**: A compression technique that splits vectors into chunks and replaces each chunk with a representative centroid ID.

## Q

**Query Vector**: In attention mechanisms, the vector representing "what information is being sought" from other positions.

## R

**RAG (Retrieval-Augmented Generation)**: A system that combines vector store retrieval with LLM generation to provide informed, grounded responses.

**ReLU (Rectified Linear Unit)**: An activation function that outputs the input if positive, zero otherwise: f(x) = max(0, x). It's popular because it's simple to compute and helps address the vanishing gradient problem.

## S

**Sigmoid**: An activation function that maps any input to a value between 0 and 1: f(x) = 1/(1 + e^(-x)). It's often used in binary classification tasks and historically was very common in neural networks.

**Similarity Threshold**: A minimum similarity score required for a document to be considered relevant in vector search.

## T

**Temperature**: A parameter controlling randomness in text generation; lower values make outputs more focused, higher values more creative.

**Top-K**: A parameter limiting selection to the K most likely tokens (in LLMs) or K most similar documents (in vector stores).

**Top-P (Nucleus Sampling)**: A parameter that dynamically selects tokens based on cumulative probability until reaching threshold P.

**Transformer**: A neural network architecture that uses self-attention mechanisms to process sequential data efficiently, powering models like GPT and BERT. It revolutionized NLP by processing all positions in a sequence simultaneously.

## U

**Universal Approximation Theorem**: A mathematical theorem stating that neural networks with sufficient neurons can approximate any continuous function. This provides theoretical justification for why neural networks are so powerful.

## V

**Vector Store**: A database optimized for storing and searching high-dimensional numerical vectors representing semantic content.

## W

**Weight**: Parameters in a neural network that determine the strength of connections between neurons and are learned during training. Weights are adjusted through backpropagation to minimize the loss function.

---

*This glossary covers fundamental terms from neural networks through transformer architectures. Terms are continuously updated as new concepts are introduced in the learning materials.*
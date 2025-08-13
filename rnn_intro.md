# Recurrent Neural Networks (RNNs): A Step-by-Step Tutorial

**What you'll learn:** How RNNs process sequences step-by-step, carry memory through hidden states, and differ from regular feedforward networks. We'll work through math and intuition together, with tiny examples you can follow by hand.

**Prerequisites:** Basic linear algebra (vectors, matrices, dot products). No prior RNN experience needed.

---

## 1. What is an RNN?

Imagine you're reading a sentence word by word. As you read each word, you remember what came before it. This memory helps you understand the current word in context. RNNs work similarly—they process sequences step-by-step while maintaining a "memory" of what they've seen.

### RNN vs Regular Neural Network (MLP)

**Regular MLP (Multi-Layer Perceptron):**
```
Input → Hidden Layer → Output
  x   →      h       →   y
```
- Processes fixed-size inputs all at once
- No memory between different inputs
- Each layer has different weights

**RNN (Recurrent Neural Network):**
```
Time step 1: x₁ → RNN → h₁ → y₁
Time step 2: x₂ → RNN → h₂ → y₂  (uses h₁ as memory)
Time step 3: x₃ → RNN → h₃ → y₃  (uses h₂ as memory)
```
- Processes sequences one element at a time
- Carries "hidden state" (memory) between time steps
- Same weights reused at every time step

**Key Insight:** An RNN is like having a single neural network that processes a sequence by applying itself repeatedly, each time using both the current input and its memory of the past.

---

## 2. The Core RNN Equation

The heart of every RNN is this update rule:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

Let's break this down term by term:

| Term | Size | Meaning |
|------|------|---------|
| $x_t$ | $(E,)$ | **Current input** - word embedding at time $t$ |
| $h_{t-1}$ | $(H,)$ | **Past memory** - hidden state from previous step |
| $W_{xh}$ | $(E, H)$ | **Input weights** - transform current input |
| $W_{hh}$ | $(H, H)$ | **Hidden weights** - transform past memory |
| $b_h$ | $(H,)$ | **Bias** - learned offset |
| $h_t$ | $(H,)$ | **New memory** - updated hidden state |

### Visual Breakdown

```
Past Memory    Current Input
    h_{t-1}  +      x_t
       ↓              ↓
   W_{hh} * h_{t-1} + W_{xh} * x_t + b_h
                      ↓
                   tanh(...)
                      ↓
                New Memory h_t
```

**Why `tanh`?**
- **Non-linearity:** Without it, the RNN would just be linear algebra (boring!)
- **Bounded output:** `tanh` keeps values between -1 and +1, preventing explosion
- **Zero-centered:** Helps with gradient flow during training

**The Magic:** At each step, the RNN combines:
1. **What's happening now** ($W_{xh} \cdot x_t$) 
2. **What it remembers** ($W_{hh} \cdot h_{t-1}$)
3. **Its learned bias** ($b_h$)

---

## 3. Where These Weights Come From

### Weight Initialization
Initially, $W_{xh}$, $W_{hh}$, and $b_h$ are **random numbers**. The RNN learns by adjusting these weights through training.

### Training Process: Backpropagation Through Time (BPTT)

```
Forward Pass (compute predictions):
Step 1: h₁ = tanh(W_xh·x₁ + W_hh·h₀ + b_h)
Step 2: h₂ = tanh(W_xh·x₂ + W_hh·h₁ + b_h)
Step 3: h₃ = tanh(W_xh·x₃ + W_hh·h₂ + b_h)

Compute Loss:
loss = compare(predictions, true_labels)

Backward Pass (compute gradients):
loss/W_xh flows back through ALL time steps
loss/W_hh flows back through ALL time steps
loss/b_h  flows back through ALL time steps
```

**Key Point:** The same weights ($W_{xh}$, $W_{hh}$) are used at every time step, but gradients flow back through the entire sequence.

### Weight Sharing vs MLPs

| **MLP** | **RNN** |
|---------|---------|
| Layer 1 has weights $W_1$ | Time step 1 uses weights $W_{xh}, W_{hh}$ |
| Layer 2 has weights $W_2$ | Time step 2 uses **same** weights $W_{xh}, W_{hh}$ |
| Layer 3 has weights $W_3$ | Time step 3 uses **same** weights $W_{xh}, W_{hh}$ |
| Each layer learns different transformations | All time steps share the same transformation |

---

## 4. Hidden Size and Embedding Size

### Embedding Size (E): Input Detail

Think of embedding size as the "resolution" of your input:

```
Low resolution (E=2):  "cat" → [0.1, 0.8]
High resolution (E=100): "cat" → [0.1, 0.8, -0.3, 0.5, ..., 0.2]
```

- **Larger E:** More detailed representation, captures more nuances
- **Smaller E:** Simpler representation, less detail but faster computation

**Analogy:** Like describing a photo with 2 words vs 100 words.

### Hidden Size (H): Memory Capacity

Hidden size controls how much "memory" the RNN can maintain:

```
Small memory (H=2):  h_t = [0.3, -0.7]  # Like a small notebook
Large memory (H=100): h_t = [0.3, -0.7, 0.1, ..., 0.9]  # Like a large notebook
```

- **Larger H:** Can remember more complex patterns, longer dependencies
- **Smaller H:** Limited memory, but faster and less prone to overfitting

**Analogy:** Like having a small backpack vs a large backpack for carrying memories.

### Weight Matrix Shapes

The dimensions determine the weight matrix shapes:

| Weight | Shape | Purpose |
|--------|-------|---------|
| $W_{xh}$ | $(E \times H)$ | Maps input dimension to hidden dimension |
| $W_{hh}$ | $(H \times H)$ | Maps hidden dimension to itself (recurrence) |
| $b_h$ | $(H,)$ | Bias for each hidden unit |

**Example:** If $E=50$ (word embeddings) and $H=128$ (hidden size):
- $W_{xh}$: $(50 \times 128)$ matrix with 6,400 parameters
- $W_{hh}$: $(128 \times 128)$ matrix with 16,384 parameters  
- $b_h$: $(128,)$ vector with 128 parameters
- **Total:** 22,912 parameters

---

## 5. Worked Example: "cat sat here"

Let's trace through a tiny example step by step. We'll use:
- **Vocabulary:** {"cat": 0, "sat": 1, "here": 2}
- **Embedding size:** $E = 2$ 
- **Hidden size:** $H = 2$
- **Sequence:** "cat sat here"

### Step 0: Initialize

**Embeddings (learned lookup table):**
```
"cat"  (id=0) → x₁ = [0.5, 0.2]
"sat"  (id=1) → x₂ = [0.1, 0.9]  
"here" (id=2) → x₃ = [0.8, 0.3]
```

**Initial hidden state:**
```
h₀ = [0.0, 0.0]  # Start with no memory
```

**Learned weights (after training):**
```
W_xh = [[0.3, 0.7],    # 2×2 matrix: input-to-hidden
        [0.4, 0.2]]

W_hh = [[0.1, 0.5],    # 2×2 matrix: hidden-to-hidden  
        [0.6, 0.3]]

b_h = [0.1, 0.2]       # 2-element bias vector
```

### Step 1: Process "cat"

**Input:** $x_1 = [0.5, 0.2]$, **Memory:** $h_0 = [0.0, 0.0]$

**Compute contributions:**
```
W_xh · x₁ = [[0.3, 0.7],  · [0.5,  = [0.3·0.5 + 0.7·0.2,  = [0.29,
             [0.4, 0.2]]     0.2]     0.4·0.5 + 0.2·0.2]     0.24]

W_hh · h₀ = [[0.1, 0.5],  · [0.0,  = [0.0,
             [0.6, 0.3]]     0.0]     0.0]
```

**Combine and activate:**
```
W_xh·x₁ + W_hh·h₀ + b_h = [0.29, 0.24] + [0.0, 0.0] + [0.1, 0.2]
                        = [0.39, 0.44]

h₁ = tanh([0.39, 0.44]) = [0.37, 0.41]
```

### Step 2: Process "sat" 

**Input:** $x_2 = [0.1, 0.9]$, **Memory:** $h_1 = [0.37, 0.41]$

**Compute contributions:**
```
W_xh · x₂ = [[0.3, 0.7],  · [0.1,  = [0.66,
             [0.4, 0.2]]     0.9]     0.22]

W_hh · h₁ = [[0.1, 0.5],  · [0.37, = [0.24,
             [0.6, 0.3]]     0.41]    0.35]
```

**Combine and activate:**
```
W_xh·x₂ + W_hh·h₁ + b_h = [0.66, 0.22] + [0.24, 0.35] + [0.1, 0.2]
                        = [1.00, 0.77]

h₂ = tanh([1.00, 0.77]) = [0.76, 0.65]
```

### Step 3: Process "here"

**Input:** $x_3 = [0.8, 0.3]$, **Memory:** $h_2 = [0.76, 0.65]$

**Compute contributions:**
```
W_xh · x₃ = [[0.3, 0.7],  · [0.8,  = [0.45,
             [0.4, 0.2]]     0.3]     0.38]

W_hh · h₂ = [[0.1, 0.5],  · [0.76, = [0.40,
             [0.6, 0.3]]     0.65]    0.65]
```

**Combine and activate:**
```
W_xh·x₃ + W_hh·h₂ + b_h = [0.45, 0.38] + [0.40, 0.65] + [0.1, 0.2]
                        = [0.95, 1.23]

h₃ = tanh([0.95, 1.23]) = [0.74, 0.84]
```

### Summary: Memory Evolution

```
Start:    h₀ = [0.00, 0.00]  # No memory
"cat":    h₁ = [0.37, 0.41]  # Remembers "cat"  
"sat":    h₂ = [0.76, 0.65]  # Remembers "cat sat"
"here":   h₃ = [0.74, 0.84]  # Remembers "cat sat here"
```

**Key Insight:** Each hidden state $h_t$ encodes information about the entire sequence up to time $t$. The RNN builds up contextual understanding step by step.

---

## 6. RNN vs MLP Training

### MLP Training: Layer-by-Layer

```
Input → Layer 1 → Layer 2 → Layer 3 → Output
 x    →   W₁    →   W₂    →   W₃    →   y

Backprop:
∂loss/∂W₃ ← computed from output layer
∂loss/∂W₂ ← flows back one layer  
∂loss/∂W₁ ← flows back two layers
```

**Characteristics:**
- Each layer has **different weights**
- Gradients flow **backward through layers**
- Training is **straightforward** - standard backprop

### RNN Training: Backpropagation Through Time (BPTT)

```
x₁ → RNN → h₁ → y₁
x₂ → RNN → h₂ → y₂  (same weights!)
x₃ → RNN → h₃ → y₃  (same weights!)

Backprop Through Time:
∂loss/∂W_xh ← sum of gradients from ALL time steps  
∂loss/∂W_hh ← sum of gradients from ALL time steps
∂loss/∂b_h  ← sum of gradients from ALL time steps
```

**Characteristics:**
- **Same weights** used at every time step
- Gradients flow **backward through time AND layers**
- Training is **more complex** - gradients must be accumulated across time

### The Gradient Flow Challenge

In deep RNNs or long sequences, gradients can:

**Vanish (become too small):**
```
Step 50 → Step 49 → ... → Step 2 → Step 1
  →         →                →       →
0.001    0.0001           0.000...001  H0
```
- Early time steps receive almost no learning signal
- RNN forgets long-term dependencies

**Explode (become too large):**
```
Step 1 → Step 2 → ... → Step 49 → Step 50  
  →       →               →        →
 1.5     2.25            [overflow]        NaN
```
- Gradients grow exponentially
- Training becomes unstable

**Solutions:** Gradient clipping, LSTM/GRU architectures, careful initialization

---

## 7. Summary: The Big Picture

### How RNNs Work

1. **Sequential Processing:** Handle one input at a time, maintaining memory
2. **Weight Sharing:** Same transformation applied at every time step  
3. **Memory Accumulation:** Hidden state captures increasingly rich context
4. **Flexible Length:** Can process sequences of any length

### Key Components Working Together

```
Current Input (x_t)
                      → Combine → tanh → New Memory (h_t)
Past Memory (h_{t-1})

Where "Combine" means: W_xh·x_t + W_hh·h_{t-1} + b_h
```

**$W_{xh}$:** "How should I interpret the current input?"
**$W_{hh}$:** "How should I update my memory based on what I remember?"
**$b_h$:** "What's my default tendency when updating memory?"

### Capacity Control

- **Embedding Size (E):** How much detail each input word carries
- **Hidden Size (H):** How much memory the RNN can maintain
- **Sequence Length (T):** How far back the RNN needs to remember

### Training Differences from MLPs

| **Aspect** | **MLP** | **RNN** |
|------------|---------|---------|
| **Weights** | Different per layer | Shared across time |
| **Gradients** | Flow through layers | Flow through time and layers |  
| **Challenges** | Standard optimization | Vanishing/exploding gradients |
| **Memory** | None between samples | Carries context within sequence |

---

## 8. Final Visualization: "cat sat here" Through Time

```
Time Step 1: "cat"
Input: [0.5, 0.2]
                    → tanh([0.39, 0.44]) → h₁=[0.37, 0.41]
Memory: [0.0, 0.0]
                    
Time Step 2: "sat"  
Input: [0.1, 0.9]
                    → tanh([1.00, 0.77]) → h₂=[0.76, 0.65]
Memory: [0.37, 0.41]
                    
Time Step 3: "here"
Input: [0.8, 0.3]
                    → tanh([0.95, 1.23]) → h₃=[0.74, 0.84]
Memory: [0.76, 0.65]

Final Memory: [0.74, 0.84] encodes "cat sat here"
```

**The Journey:** From no memory to rich contextual understanding, one step at a time. The RNN learns to compress the entire sequence history into a fixed-size hidden state vector.

---

## Next Steps

Now that you understand the fundamentals:

1. **Limitations:** RNNs struggle with very long sequences (vanishing gradients)
2. **Solutions:** LSTM and GRU architectures address these issues  
3. **Modern Alternatives:** Transformers have largely replaced RNNs for many tasks
4. **Implementation:** Try building an RNN in PyTorch or TensorFlow

**Remember:** RNNs taught us that neural networks could have memory. This insight paved the way for all modern sequence models, including the Transformers that power today's language models.
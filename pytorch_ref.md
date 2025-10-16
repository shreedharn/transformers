# PyTorch Reference: From MLPs to Transformers

## 1. Title & Quickstart

### Run This First

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Check PyTorch version and device
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = torch.device('cpu')  # We'll use CPU for reproducibility
```

### What You'll Be Able to Do After This

After reading this guide, you'll be able to:

- Create and manipulate tensors with proper shapes for ML
- Build neural networks from scratch using `nn.Module`
- Implement training loops with automatic differentiation
- Handle variable-length sequences with padding and masking
- Build MLPs for classification with proper initialization
- Implement RNNs/LSTMs for sequence processing
- Create attention mechanisms and Transformer blocks
- Debug common shape mismatches and gradient issues
- Save/load models and handle device placement
- Understand when to use different optimizers and losses

## 2. Tensors: Vectors & Matrices in PyTorch

📓 Interactive Examples: [Tensor Basics Notebook](./pynb/basic/tensors_basics.ipynb)

### Core Mental Model

Tensors are n-dimensional arrays that carry data and gradients through neural networks. Think of them as generalized matrices that know how to compute derivatives.

The notebook covers:

- Basic tensor creation and manipulation
- Vector/matrix operations and broadcasting  
- One-hot vs embedding vectors
- Batch operations and shape handling

**Math** Cross-Reference to `./math_quick_ref.md`:
> Inner products and matrix shapes: When we compute `X @ W`, we're applying the matrix multiplication rule from **Mathematical Preliminaries**. The gradient `∂L/∂W = X^T ∂L/∂y` follows from the chain rule identities.

## 3. Autograd (Finding Gradients)

📓 Interactive Examples: [Autograd Notebook](./pynb/basic/autograd.ipynb)

This notebook covers:

- Basic gradient computation and the chain rule
- Optimizer integration patterns
- No-grad context and detach operations  
- Gradient clipping demonstrations
- Higher-order derivatives

**Math** Cross-Reference to `./transformers_math1.md`:
> The softmax gradient `∂p_i/∂z_j = p_i(δ_{ij} - p_j)` from **equation (25)** explains why PyTorch's `CrossEntropyLoss` yields the clean gradient `(p - y)` at the logits. The **log-sum-exp trick (12)** prevents overflow in softmax computation, which PyTorch handles automatically in `F.softmax()`.

## 4. Modules, Parameters, Initialization

📓 Interactive Examples: [Modules & Parameters Notebook](./pynb/basic/modules_parameters.ipynb)

This notebook covers:

- Basic module structure and inheritance from nn.Module
- Parameter counting formulas for different layer types
- Initialization strategies (Xavier, Kaiming, custom)
- Advanced module patterns and parameter sharing

### Parameter Counting Formulas

```python
def count_parameters(layer):
    """Count parameters in common layer types"""
    if isinstance(layer, nn.Linear):
        # Linear(in_features, out_features): in*out + out
        return layer.in_features * layer.out_features + layer.out_features
    elif isinstance(layer, nn.Embedding):
        # Embedding(num_embeddings, embedding_dim): num*dim
        return layer.num_embeddings * layer.embedding_dim
    elif isinstance(layer, nn.LSTM):
        # LSTM has 4 gates, each with input and hidden weights + bias
        input_size, hidden_size = layer.input_size, layer.hidden_size
        num_layers = layer.num_layers
        bidirectional = 2 if layer.bidirectional else 1
        
        # Per layer: 4 gates * (input_weights + hidden_weights + bias)
        per_layer = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        return per_layer * num_layers * bidirectional
    else:
        return sum(p.numel() for p in layer.parameters())

# Examples
linear = nn.Linear(100, 50)
embedding = nn.Embedding(1000, 128)
lstm = nn.LSTM(128, 64, num_layers=2)

print(f"Linear(100, 50) parameters: {count_parameters(linear)}")
print(f"Expected: {100 * 50 + 50} = {100 * 50 + 50}")

print(f"Embedding(1000, 128) parameters: {count_parameters(embedding)}")
print(f"Expected: {1000 * 128} = {1000 * 128}")

print(f"LSTM(128, 64, layers=2) parameters: {count_parameters(lstm)}")
```

### Initialization Strategies

```python
def init_weights(m):
    """Initialize weights based on layer type"""
    if isinstance(m, nn.Linear):
        # Xavier (Glorot) for tanh/sigmoid activations
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Small random values for embeddings
        nn.init.uniform_(m.weight, -0.1, 0.1)

# Alternative: Kaiming for ReLU activations
def kaiming_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply to model
model = SimpleNet(10, 20, 5)
model.apply(init_weights)

print("Before and after initialization:")
for name, param in model.named_parameters():
    print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
```

**Math** Cross-Reference to `./transformers_math1.md`:
> MLP Forward Pass (13-15): `z^(1) = xW^(1) + b^(1)`, `h^(1) = σ(z^(1))`, `z^(2) = h^(1)W^(2) + b^(2)`
> 
> LayerNorm (19): `LayerNorm(x) = γ ⊙ (x - μ)/√(σ² + ε) + β` where `γ, β` are learnable parameters

## 5. Optimization Loop & Losses

📓 Interactive Examples: [Optimization & Training Notebook](./pynb/basic/optimization_training.ipynb)

This notebook covers:

- Canonical training loop patterns
- Optimizer comparison (SGD vs Adam vs AdamW)
- Common loss functions for different tasks
- Train vs eval modes with practical examples
- Learning rate scheduling strategies

**Math** Cross-Reference to `./math_quick_ref.md`:
> Adam Updates: Adaptive learning rates with momentum. **Learning Rate Warmup** prevents early training instability in large models.

## 6. Vanishing/Exploding Gradients

### The Problem

In deep networks, gradients can vanish (become too small) or explode (become too large) as they backpropagate through layers. This is especially problematic for RNNs processing long sequences.

> See also: `./rnn_intro.md` discusses how vanishing gradients motivated the development of LSTM/GRU architectures with gating mechanisms.

### Practical Fixes in PyTorch

```python
# 1. Better Activations: ReLU/GELU instead of tanh/sigmoid
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))  # Sigmoid causes vanishing gradients
        return x

class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
    
    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))  # GELU has better gradient flow
        return x

# 2. Residual Connections: Let gradients flow directly
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
    
    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = F.gelu(layer(x))
            x = x + residual  # Skip connection
        return x

# 3. LayerNorm: Normalize activations
class NormalizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(10) for _ in range(10)])
    
    def forward(self, x):
        for layer, ln in zip(self.layers, self.layer_norms):
            x = ln(F.gelu(layer(x)))
        return x

# 4. Gradient Clipping: Prevent explosion
def train_with_clipping(model, data_loader, max_norm=1.0):
    optimizer = optim.Adam(model.parameters())
    for data, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
```

### Toy RNN Explosion Demo

```python
# Demonstrate exploding gradients in vanilla RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Bad initialization - causes explosion
        nn.init.constant_(self.W_hh.weight, 2.0)  # Eigenvalues > 1
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden = torch.zeros(batch_size, self.hidden_size)
        
        for t in range(seq_len):
            hidden = torch.tanh(self.W_ih(x[:, t]) + self.W_hh(hidden))
        
        return hidden

# Test explosion
rnn = SimpleRNN(5, 10)
x = torch.randn(2, 20, 5)  # batch=2, seq_len=20, input_size=5
y = torch.randn(2, 10)

loss = F.mse_loss(rnn(x), y)
loss.backward()

# Check gradient norm
total_norm = sum(p.grad.norm().item() ** 2 for p in rnn.parameters()) ** 0.5
print(f"Gradient norm without clipping: {total_norm:.2f}")

# Fixed version with clipping
rnn.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

clipped_norm = sum(p.grad.norm().item() ** 2 for p in rnn.parameters()) ** 0.5
print(f"Gradient norm with clipping: {clipped_norm:.2f}")
```

**Math** Cross-Reference to `./transformers_math1.md`:
> Residual as ODE (4): `h_{l+1} = h_l + F(h_l)` approximates the differential equation `dh/dt = F(h)`, enabling gradient highways through skip connections.
>
> Gradient Clipping (11): `g̃ = min(1, c/||g||₂) · g` scales gradients proportionally when norm exceeds threshold `c`.

## 7. Mapping Table: ML Concepts → PyTorch Objects

| **Concept** | **Math/Idea** | **PyTorch Construct** | **Equation Reference** |
|-------------|---------------|----------------------|----------------------|
| **MLPs** |
| Linear layer | `y = Wx + b` | `nn.Linear(in_features, out_features)` | (13-15) |
| Activation | `σ(z)` | `nn.ReLU()`, `nn.GELU()`, `F.relu()`, `F.gelu()` | |
| Layer normalization | `γ ⊙ (x-μ)/√(σ²+ε) + β` | `nn.LayerNorm(normalized_shape)` | (19) |
| Dropout regularization | Random zeroing | `nn.Dropout(p=0.1)`, `F.dropout()` | |
| **RNNs/LSTMs** |
| RNN cell | `h_t = tanh(W_ih x_t + W_hh h_{t-1})` | `nn.RNN()`, `nn.RNNCell()` | |
| LSTM cell | Gated updates | `nn.LSTM()`, `nn.LSTMCell()` | |
| GRU cell | Simplified gating | `nn.GRU()`, `nn.GRUCell()` | |
| Sequence packing | Variable lengths | `pack_padded_sequence()`, `pad_sequence()` | |
| **Transformers** |
| Scaled dot-product attention | `softmax(QK^T/√d_k)V` | `nn.MultiheadAttention()` | (23) |
| Self-attention | Q,K,V from same input | `nn.TransformerEncoderLayer()` | |
| Causal mask | Lower triangular | `torch.triu()`, `attn_mask` parameter | (24) |
| Position embeddings | Learnable positions | `nn.Embedding(max_len, d_model)` | |
| Sinusoidal positions | Fixed sin/cos | Custom implementation | (28-29) |
| Feed-forward network | `GELU(xW₁)W₂` | `nn.TransformerEncoderLayer.linear1/2` | (36) |
| **Embeddings & Tokens** |
| Token embeddings | Lookup table | `nn.Embedding(vocab_size, embed_dim)` | (39) |
| Positional encoding | Add position info | Manual or `nn.Embedding` | |
| **Data Handling** |
| Dataset wrapper | Data access | `torch.utils.data.Dataset` | |
| Batch loading | Mini-batches | `torch.utils.data.DataLoader` | |
| Padding sequences | Same length | `pad_sequence()` | |
| Collate function | Custom batching | `collate_fn` parameter | |
| **Training & Optimization** |
| Loss functions | Various objectives | `nn.CrossEntropyLoss()`, `nn.MSELoss()` | (2) |
| SGD optimizer | Gradient descent | `optim.SGD()` | (5-6) |
| Adam optimizer | Adaptive learning | `optim.Adam()`, `optim.AdamW()` | (7-9) |
| Learning rate scheduling | Dynamic LR | `lr_scheduler.StepLR()`, etc. | (10) |
| Gradient clipping | Norm limiting | `clip_grad_norm_()` | (11) |
| **Utilities** |
| No gradients | Inference mode | `torch.no_grad()` | |
| Detach from graph | Stop gradients | `tensor.detach()` | |
| Random seeding | Reproducibility | `torch.manual_seed()` | |
| Device placement | GPU/CPU | `tensor.to(device)`, `model.to(device)` | |
| Model saving | Persistence | `torch.save(model.state_dict())` | |

**Math** annotations reference equations from `./transformers_math1.md` and `./transformers_math2.md` where applicable.

## 8. MLPs in PyTorch

📓 Interactive Examples: [MLPs Notebook](./pynb/dl/mlps.ipynb)

This notebook demonstrates:

- From equations to PyTorch code
- Training MLPs on synthetic datasets
- Common gotchas and debugging tips

```python
# Method 1: Using nn.Sequential (simplest)
mlp_sequential = nn.Sequential(
    nn.Linear(784, 128),    # W₁x + b₁
    nn.ReLU(),              # σ₁ 
    nn.Linear(128, 10),     # W₂h + b₂
    # No final activation - we'll use CrossEntropyLoss
)

print(f"Sequential MLP: {mlp_sequential}")

# Method 2: Custom nn.Module (more control)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Flatten if needed: [batch, height, width] -> [batch, height*width]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass with shapes
        x = self.fc1(x)      # [batch, input] -> [batch, hidden]
        x = F.relu(x)        # Apply activation
        x = self.dropout(x)  # Regularization
        x = self.fc2(x)      # [batch, hidden] -> [batch, classes]
        return x

# Create model and check shapes
mlp = MLP(input_size=784, hidden_size=128, num_classes=10)
dummy_input = torch.randn(32, 784)  # Batch of 32 samples
output = mlp(dummy_input)
print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")

# Parameter counting
total_params = sum(p.numel() for p in mlp.parameters())
print(f"Total parameters: {total_params}")
print(f"Expected: {784*128 + 128 + 128*10 + 10} = {784*128 + 128 + 128*10 + 10}")
```

### Training MLP on Synthetic Data

```python
# Create synthetic classification dataset
def create_toy_dataset(n_samples=1000, n_features=20, n_classes=5):
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    # Create separable classes with some noise
    centers = torch.randn(n_classes, n_features) * 2
    y = torch.zeros(n_samples, dtype=torch.long)
    
    for i in range(n_samples):
        distances = torch.norm(X[i] - centers, dim=1)
        y[i] = torch.argmin(distances)
    
    return X, y

# Generate data
X, y = create_toy_dataset()
print(f"Dataset: X.shape={X.shape}, y.shape={y.shape}")
print(f"Classes: {torch.unique(y)}")

# Split data
n_train = 800
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Create DataLoader
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the MLP
mlp = MLP(input_size=20, hidden_size=50, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(mlp.parameters(), lr=0.001)

mlp.train()
for epoch in range(20):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = mlp(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Average Loss = {total_loss/len(train_loader):.4f}")

# Test accuracy
mlp.eval()
with torch.no_grad():
    test_outputs = mlp(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (test_predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.2%}")
```

### Common Gotchas

```python
# Gotcha 1: Wrong loss for classification
# DON'T: Apply softmax before CrossEntropyLoss
wrong_outputs = F.softmax(mlp(X_test), dim=1)  # Softmax applied
wrong_loss = criterion(wrong_outputs, y_test)   # CrossEntropyLoss expects logits!

# CORRECT: CrossEntropyLoss expects raw logits
correct_outputs = mlp(X_test)  # No softmax
correct_loss = criterion(correct_outputs, y_test)

# Gotcha 2: Device mismatch
if torch.cuda.is_available():
    device = torch.device('cuda')
    mlp = mlp.to(device)
    X_test = X_test.to(device)  # Both model AND data must be on same device
    y_test = y_test.to(device)

# Gotcha 3: Wrong target dtype for CrossEntropyLoss
# CrossEntropyLoss expects Long tensor targets, not Float
targets_wrong = torch.tensor([0.0, 1.0, 2.0])  # Float - will cause error
targets_correct = torch.tensor([0, 1, 2])      # Long - correct
```

**Math** Cross-Reference to `./transformers_math1.md`:
> MLP Forward/Backprop (13-18): The code above implements `z^(1) = xW^(1) + b^(1)`, `h^(1) = σ(z^(1))`, `z^(2) = h^(1)W^(2) + b^(2)` with automatic gradient computation for the backward pass.

## 9. RNNs, LSTMs, GRUs

📓 Interactive Examples: [RNNs, LSTMs, GRUs Notebook](./pynb/dl/rnns.ipynb)

This notebook covers:

- Why gating mechanisms are needed
- LSTM sequence classifier implementation
- Variable length sequence handling with packing
- RNN vs LSTM vs GRU comparisons

### Why Gating Mechanisms?

> See also: `./rnn_intro.md` explains how vanilla RNNs suffer from vanishing gradients over long sequences, motivating LSTM/GRU architectures with gates that control information flow.

### Minimal Sequence Classifier with LSTM

```python
# Many-to-one sequence classification
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, lengths=None):
        # x: [batch_size, seq_len] of token indices
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        if lengths is not None:
            # Pack for variable-length sequences (more efficient)
            embedded = pack_padded_sequence(embedded, lengths, 
                                          batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.lstm(embedded)
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            # Regular forward pass
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state for classification
        # hidden: [num_layers, batch, hidden_dim]
        last_hidden = hidden[-1]  # [batch, hidden_dim]
        output = self.classifier(self.dropout(last_hidden))
        return output

# Parameter count for LSTM
vocab_size, embed_dim, hidden_dim, num_classes = 1000, 64, 128, 5
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)

# Count LSTM parameters
lstm = model.lstm
input_size, hidden_size = lstm.input_size, lstm.hidden_size
num_layers = lstm.num_layers

# LSTM formula: 4 gates * (input_weights + hidden_weights + bias) * num_layers
lstm_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size) * num_layers
print(f"LSTM parameters: {lstm_params}")
print(f"Expected: {4 * (64 * 128 + 128 * 128 + 128) * 1} = {4 * (64 * 128 + 128 * 128 + 128) * 1}")
```

### Variable Length Sequence Handling

```python
# Create sequences of different lengths
def create_variable_sequences():
    sequences = [
        torch.tensor([1, 5, 3, 8, 2]),        # length 5
        torch.tensor([4, 7, 9]),              # length 3  
        torch.tensor([2, 6, 1, 4, 9, 3, 7]), # length 7
        torch.tensor([8, 2])                  # length 2
    ]
    lengths = torch.tensor([len(seq) for seq in sequences])
    return sequences, lengths

sequences, lengths = create_variable_sequences()
print(f"Sequence lengths: {lengths}")

# Method 1: Padding (simpler, but less efficient)
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
print(f"Padded shape: {padded.shape}")
print(f"Padded sequences:\n{padded}")

# Method 2: Packing (more efficient, variable computation)
# Sort by length (required for packing)
sorted_lengths, sorted_idx = lengths.sort(descending=True)
sorted_sequences = [sequences[i] for i in sorted_idx]
padded_sorted = pad_sequence(sorted_sequences, batch_first=True, padding_value=0)

# Pack the padded sequences
packed = pack_padded_sequence(padded_sorted, sorted_lengths, batch_first=True)
print(f"Packed data shape: {packed.data.shape}")
print(f"Batch sizes: {packed.batch_sizes}")

# Training with variable lengths
def train_lstm_with_variable_lengths():
    model = LSTMClassifier(vocab_size=100, embed_dim=32, hidden_dim=64, num_classes=3)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy data
    batch_sequences = [torch.randint(1, 100, (torch.randint(5, 15, (1,)).item(),)) for _ in range(8)]
    batch_lengths = torch.tensor([len(seq) for seq in batch_sequences])
    batch_labels = torch.randint(0, 3, (8,))
    
    # Pad and sort
    sorted_lengths, sorted_idx = batch_lengths.sort(descending=True)
    sorted_sequences = [batch_sequences[i] for i in sorted_idx]
    sorted_labels = batch_labels[sorted_idx]
    padded_batch = pad_sequence(sorted_sequences, batch_first=True, padding_value=0)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(padded_batch, sorted_lengths)
    loss = criterion(outputs, sorted_labels)
    loss.backward()
    
    # Gradient clipping for RNNs (important!)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    print(f"Training loss: {loss.item():.4f}")

train_lstm_with_variable_lengths()
```

### RNN vs LSTM vs GRU Comparison

```python
# Compare architectures on same task
def compare_rnn_architectures():
    input_size, hidden_size, num_layers = 50, 64, 2
    batch_size, seq_len = 4, 10
    
    # Create models
    rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    
    # Count parameters
    def count_rnn_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"RNN parameters: {count_rnn_params(rnn)}")
    print(f"LSTM parameters: {count_rnn_params(lstm)}")  
    print(f"GRU parameters: {count_rnn_params(gru)}")
    
    # Compare outputs
    x = torch.randn(batch_size, seq_len, input_size)
    
    rnn_out, rnn_hidden = rnn(x)
    lstm_out, (lstm_hidden, lstm_cell) = lstm(x)
    gru_out, gru_hidden = gru(x)
    
    print(f"Output shapes - RNN: {rnn_out.shape}, LSTM: {lstm_out.shape}, GRU: {gru_out.shape}")
    print(f"Hidden shapes - RNN: {rnn_hidden.shape}, LSTM: {lstm_hidden.shape}, GRU: {gru_hidden.shape}")
    
    # LSTM also has cell state
    print(f"LSTM cell state shape: {lstm_cell.shape}")

compare_rnn_architectures()
```

Gotcha: Always use gradient clipping with RNNs to prevent exploding gradients, especially for long sequences.

## 10. Transformers in PyTorch

📓 Interactive Examples: [Transformers Notebook](./pynb/dl/transformers.ipynb)

This notebook demonstrates:

- Self-attention mechanisms from scratch
- Using PyTorch's built-in Transformer layers
- Causal and padding masks
- Next-token prediction with character-level models

> See also: `./transformers_fundamentals.md` explains the complete Transformer block flow: multi-head self-attention → residual connection → layer norm → feed-forward network → residual connection → layer norm.

### Self-Attention from Scratch

```python
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_model = d_model
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)
        
        self.scale = 1 / (self.d_k ** 0.5)  # 1/√d_k scaling
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_k]
        K = self.W_k(x)  # [batch, seq_len, d_k]
        V = self.W_v(x)  # [batch, seq_len, d_k]
        
        # Compute attention scores: QK^T/√d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, seq_len, seq_len]
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch, seq_len, d_k]
        
        return output, attn_weights

# Test single-head attention
d_model, seq_len, batch_size = 64, 8, 2
x = torch.randn(batch_size, seq_len, d_model)

attention = SingleHeadSelfAttention(d_model)
output, weights = attention(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights sum (should be ~1.0): {weights.sum(dim=-1).mean():.4f}")
```

### Using PyTorch's Built-in Transformer

```python
# Complete Transformer-based classifier
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=4*d_model,  # Common choice: 4x model dimension
            dropout=0.1,
            activation='gelu',
            batch_first=True  # Input shape: [batch, seq, feature]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, src_key_padding_mask=None):
        # x: [batch_size, seq_len] of token indices
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)  # [batch, seq_len, d_model]
        pos_emb = self.position_embedding(pos)  # [batch, seq_len, d_model]
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling for classification
        if src_key_padding_mask is not None:
            # Mask out padding tokens before averaging
            mask = src_key_padding_mask.unsqueeze(-1).float()
            x = x * (1 - mask)  # Zero out padded positions
            lengths = (1 - src_key_padding_mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths  # [batch, d_model]
        else:
            x = x.mean(dim=1)  # [batch, d_model]
        
        # Classification
        output = self.classifier(x)  # [batch, num_classes]
        return output

# Create model
model = TransformerClassifier(
    vocab_size=5000, 
    d_model=128, 
    nhead=8, 
    num_layers=4, 
    num_classes=3
)

print(f"Model: {model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Masking: Causal and Padding

```python
def create_causal_mask(seq_len):
    """Create lower triangular mask for causal attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # Convert to boolean mask

def create_padding_mask(sequences, pad_idx=0):
    """Create mask for padding tokens"""
    return sequences == pad_idx

# Example usage
seq_len = 5
batch_size = 2

# Causal mask (for autoregressive models)
causal_mask = create_causal_mask(seq_len)
print(f"Causal mask:\n{causal_mask.int()}")

# Padding mask
sequences = torch.tensor([
    [1, 3, 5, 2, 0],  # Last token is padding
    [4, 2, 0, 0, 0],  # Last 3 tokens are padding
])
padding_mask = create_padding_mask(sequences, pad_idx=0)
print(f"Padding mask:\n{padding_mask}")

# Use in transformer
with torch.no_grad():
    # For classification (no causal mask needed)
    output = model(sequences, src_key_padding_mask=padding_mask)
    print(f"Classification output shape: {output.shape}")
```

### Next-Token Prediction Demo

```python
# Tiny language model for demonstration
class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1000, d_model)  # Support up to 1000 positions
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, 
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.embedding(x) + self.pos_embedding(pos)
        
        # Causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transform
        x = self.transformer(x, mask=causal_mask)
        
        # Language modeling head
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits

# Create tiny model and synthetic data
vocab_size = 20  # Very small vocab for demo
model = TinyLM(vocab_size=vocab_size, d_model=32, nhead=4, num_layers=2)

# Generate synthetic sequences
torch.manual_seed(42)
batch_size, seq_len = 4, 8
sequences = torch.randint(1, vocab_size-1, (batch_size, seq_len))

# Training step
optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(10):
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(sequences)  # [batch, seq_len, vocab]
    
    # Shift for next-token prediction
    # Input: [w1, w2, w3], Target: [w2, w3, w4] 
    input_logits = logits[:, :-1]  # [batch, seq_len-1, vocab]
    target_tokens = sequences[:, 1:]  # [batch, seq_len-1]
    
    # Flatten for CrossEntropyLoss
    loss = criterion(
        input_logits.reshape(-1, vocab_size), 
        target_tokens.reshape(-1)
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# Simple generation (greedy)
model.eval()
with torch.no_grad():
    start_tokens = torch.tensor([[1, 2]])  # Start sequence
    generated = start_tokens.clone()
    
    for _ in range(5):  # Generate 5 tokens
        logits = model(generated)
        next_token = torch.argmax(logits[0, -1])  # Last position, greedy
        generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    print(f"Generated sequence: {generated[0].tolist()}")
```

**Math** Cross-Reference to `./transformers_math1.md`:
> Scaled Dot-Product Attention (23): `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
> 
> Why 1/√d_k scaling: Prevents attention weights from becoming too peaked as dimensions increase, maintaining gradient flow.
>
> Complete Transformer Block (32-35): Pre-LayerNorm architecture with residual connections around attention and FFN.

## 11. Most-Used PyTorch APIs

### Tensor Operations Cheat Sheet

```python
# Creation
x = torch.tensor([1, 2, 3])                    # From list
zeros = torch.zeros(3, 4)                      # Zero tensor
ones = torch.ones(2, 3)                        # Ones tensor  
empty = torch.empty(2, 3)                      # Uninitialized
arange = torch.arange(0, 10, 2)               # Range: [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)            # 5 points from 0 to 1
rand = torch.rand(3, 4)                       # Uniform [0, 1)
randn = torch.randn(3, 4)                     # Standard normal

# Stacking and concatenation
a, b = torch.randn(3, 4), torch.randn(3, 4)
stacked = torch.stack([a, b], dim=0)          # [2, 3, 4] - new dimension
concatenated = torch.cat([a, b], dim=0)       # [6, 4] - along existing dim

# Reshaping
x = torch.randn(12)
reshaped = x.view(3, 4)                       # [3, 4] - must be contiguous
reshaped2 = x.reshape(2, 6)                   # [2, 6] - handles non-contiguous
x_t = reshaped.permute(1, 0)                  # [4, 3] - transpose dimensions

# Broadcasting and repetition
x = torch.tensor([[1], [2], [3]])             # [3, 1]
repeated = x.repeat(1, 4)                     # [3, 4] - copy data
expanded = x.expand(3, 4)                     # [3, 4] - no copy, uses broadcasting

# Einstein summation (powerful for complex operations)
a = torch.randn(3, 4)  
b = torch.randn(4, 5)
c = torch.einsum('ij,jk->ik', a, b)          # Matrix multiply: equivalent to a @ b
attention_scores = torch.einsum('bqd,bkd->bqk', Q, K)  # Batch attention: Q @ K.T
```

### Neural Network Layers

```python
# Basic layers
linear = nn.Linear(784, 128)                  # Fully connected
embedding = nn.Embedding(10000, 300, padding_idx=0)  # Word embeddings
dropout = nn.Dropout(0.1)                     # Regularization

# Activations
relu = nn.ReLU()                             # Or F.relu()
gelu = nn.GELU()                             # Or F.gelu() 
tanh = nn.Tanh()                             # Or torch.tanh()
sigmoid = nn.Sigmoid()                        # Or torch.sigmoid()

# Normalization  
layer_norm = nn.LayerNorm(512)               # Layer normalization
batch_norm = nn.BatchNorm1d(256)             # Batch normalization

# Sequence models
rnn = nn.RNN(100, 128, batch_first=True)
lstm = nn.LSTM(100, 128, num_layers=2, dropout=0.1, batch_first=True)
gru = nn.GRU(100, 128, batch_first=True)

# Attention
multihead_attn = nn.MultiheadAttention(512, 8, batch_first=True)
transformer_layer = nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True)
transformer = nn.TransformerEncoder(transformer_layer, 6)
```

### Loss Functions and Optimizers

```python
# Loss functions
ce_loss = nn.CrossEntropyLoss()              # Classification (expects logits)
mse_loss = nn.MSELoss()                      # Regression
bce_loss = nn.BCEWithLogitsLoss()            # Binary classification
nll_loss = nn.NLLLoss()                      # Negative log likelihood

# Optimizers
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Preferred

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Usage in training loop
for epoch in range(num_epochs):
    # ... training code ...
    scheduler.step()  # Update learning rate
```

### Data Handling

```python
from torch.utils.data import Dataset, DataLoader, random_split

# Custom dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# DataLoader
dataset = MyDataset(X, y)
train_set, val_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Sequence utilities
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

sequences = [torch.randn(5, 10), torch.randn(3, 10), torch.randn(8, 10)]
padded = pad_sequence(sequences, batch_first=True)  # Pad to same length
packed = pack_padded_sequence(padded, [5, 3, 8], batch_first=True, enforce_sorted=False)
```

### Utilities

```python
# Gradient control
with torch.no_grad():                         # Disable gradient computation
    predictions = model(x)

detached = tensor.detach()                    # Remove from computation graph
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

# Reproducibility
torch.manual_seed(42)                        # Set PyTorch seed
torch.backends.cudnn.deterministic = True    # For full reproducibility

# Model persistence
torch.save(model.state_dict(), 'model.pth')  # Save parameters only (recommended)
model.load_state_dict(torch.load('model.pth'))  # Load parameters

# Full model save (less portable)
torch.save(model, 'full_model.pth')
model = torch.load('full_model.pth')

# Device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
tensor = tensor.to(device)
```

**Math** Cross-Reference: `torch.einsum('bqd,bkd->bqk', Q, K)` directly implements the `QK^T` operation from attention equation (23) in `./transformers_math1.md`.

## 12. Common Gotchas & How to Avoid Them

📓 Interactive Examples: [Debugging & Gotchas Notebook](./pynb/basic/debugging_gotchas.ipynb)

This notebook covers:

- Training mode vs eval mode issues
- Autograd pitfalls and gradient accumulation
- Data type and shape mismatches
- Memory and device problems
- Effective debugging techniques

### Training Mode Gotchas

```python
# ❌ Wrong: Forgetting to set training mode
model.eval()  # Accidentally left in eval mode
for batch in train_loader:
    # Dropout and BatchNorm won't work as expected!
    pass

# ✅ Correct: Always set mode explicitly
model.train()  # Before training
for batch in train_loader:
    # Training code
    pass

model.eval()   # Before inference
with torch.no_grad():
    predictions = model(test_data)
```

### Autograd Gotchas

```python
# ❌ Wrong: In-place operations can break gradients
x = torch.randn(5, requires_grad=True)
x[0] = 0  # In-place modification - breaks autograd!

# ✅ Correct: Use non-in-place operations
x = torch.randn(5, requires_grad=True)
x_modified = x.clone()
x_modified[0] = 0

# ❌ Wrong: Forgetting zero_grad()
for epoch in range(10):
    loss = compute_loss()
    loss.backward()  # Gradients accumulate!
    optimizer.step()

# ✅ Correct: Clear gradients each step
for epoch in range(10):
    optimizer.zero_grad()  # Clear previous gradients
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

### Data Type Gotchas

```python
# ❌ Wrong: Mixed data types
logits = model(x).float()  # Float tensor
targets = torch.tensor([0, 1, 2])  # Long tensor - correct
# But mixed operations can cause issues

# ❌ Wrong: Wrong target type for CrossEntropyLoss
targets_wrong = torch.tensor([0., 1., 2.])  # Float targets
loss = nn.CrossEntropyLoss()(logits, targets_wrong)  # Error!

# ✅ Correct: Match expected types
targets_correct = torch.tensor([0, 1, 2])  # Long targets
loss = nn.CrossEntropyLoss()(logits, targets_correct)  # Works!

# ✅ Tip: Check dtypes when debugging
print(f"Logits dtype: {logits.dtype}")
print(f"Targets dtype: {targets_correct.dtype}")
```

### Shape Gotchas

```python
# ❌ Wrong: Batch dimension confusion
# Many PyTorch functions expect batch_first=True now, but some old code uses batch_first=False
rnn_old = nn.LSTM(input_size=10, hidden_size=20, batch_first=False)
x = torch.randn(32, 15, 10)  # [batch, seq, features]
# This will treat 32 as sequence length and 15 as batch size!

# ✅ Correct: Be explicit about batch_first
rnn_new = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
x = torch.randn(32, 15, 10)  # [batch, seq, features] - now correct

# ❌ Wrong: Unexpected dimension removal
x = torch.randn(1, 10)  # [1, 10]
x_squeezed = x.squeeze()  # [10] - removed batch dimension!
predictions = model(x_squeezed)  # Error if model expects 2D input

# ✅ Correct: Be specific about squeeze dimensions
x_squeezed_safe = x.squeeze(0) if x.size(0) == 1 else x
```

### Memory and Device Gotchas

```python
# ❌ Wrong: Model and data on different devices
model = model.to('cuda')
data = torch.randn(32, 10)  # Still on CPU
output = model(data)  # RuntimeError: Expected all tensors to be on the same device

# ✅ Correct: Keep model and data on same device
model = model.to(device)
data = data.to(device)
output = model(data)

# ❌ Wrong: Mixing contiguous and non-contiguous tensors
x = torch.randn(4, 5)
x_t = x.transpose(0, 1)  # Non-contiguous
x_reshaped = x_t.view(-1)  # Error! view() requires contiguous tensor

# ✅ Correct: Use .contiguous() or .reshape()
x_reshaped = x_t.contiguous().view(-1)  # Option 1
x_reshaped = x_t.reshape(-1)            # Option 2 (handles non-contiguous)
```

### DataLoader Gotchas

```python
# ❌ Potential issues with DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
# num_workers > 0 can cause issues on some systems
# pin_memory=True only helps if using GPU

# ✅ Safe defaults
loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=0,      # Start with 0, increase if needed
    pin_memory=False,   # Only set True if using GPU and helps
    drop_last=False,    # Be explicit about partial batches
)

# ❌ Wrong: Not handling variable batch sizes
for batch in loader:
    x, y = batch
    # Last batch might be smaller than batch_size!
    assert x.size(0) == 32  # This might fail

# ✅ Correct: Handle variable batch sizes
for batch in loader:
    x, y = batch
    batch_size = x.size(0)  # Actual batch size
    # Use batch_size in calculations
```

### Broadcasting Surprises

```python
# ❌ Unexpected broadcasting
a = torch.randn(3, 1)
b = torch.randn(4)
c = a + b  # Results in [3, 4] tensor - might not be intended!

# ✅ Be explicit about dimensions
a = torch.randn(3, 1)
b = torch.randn(1, 4)  # Make intent clear
c = a + b  # [3, 4] - now clearly intentional

# ✅ Check shapes when debugging
print(f"a: {a.shape}, b: {b.shape}, result: {c.shape}")
```

### Model Saving/Loading Gotchas

```python
# ❌ Wrong: Saving entire model (less portable)
torch.save(model, 'model.pth')
# Issues: depends on class definition, larger file size

# ✅ Correct: Save state_dict (recommended)
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {'lr': 0.001, 'hidden_dim': 128},
    'epoch': epoch,
    'loss': loss.item(),
}, 'checkpoint.pth')

# Loading
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

**Math** Cross-Reference to `./transformers_math1.md`:
> Numerical stability issues like log-sum-exp overflow (equation 12) are handled automatically by PyTorch functions like `F.softmax()` and `F.log_softmax()`, but be aware when implementing custom operations.

## 13. End-to-End Mini Example

### Character-Level Next-Token Prediction

We'll build a character-level language model that learns to predict the next character in a sequence. This demonstrates the complete pipeline from data preparation to training and generation, using an efficient parallel training approach.

```python
import string
import random

# Data preparation
def create_char_dataset(text_samples, seq_length=20):
    """Create character-level dataset from text samples"""
    # Create character vocabulary
    chars = sorted(list(set("".join(text_samples))))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    # Create input and target sequences
    sequences = []
    full_text = "".join(text_samples)
    for i in range(0, len(full_text) - seq_length, seq_length):
        # Input sequence
        input_seq = full_text[i : i + seq_length]
        # Target sequence (shifted by 1)
        target_seq = full_text[i + 1 : i + seq_length + 1]
        sequences.append((input_seq, target_seq))

    return sequences, char_to_idx, idx_to_char, vocab_size

# Generate synthetic text data (fairy tale style)
def generate_synthetic_text():
    """Generate simple synthetic text for training"""
    templates = [
        "once upon a time there was a brave knight who saved the kingdom",
        "the princess lived in a tall tower surrounded by a deep moat",
        "a dragon flew over the mountains breathing fire and smoke",
        "the wizard cast a spell to protect the village from danger",
        "knights rode horses through the forest searching for treasure",
        "the castle had many rooms filled with gold and silver",
        "magical creatures lived in the enchanted forest nearby",
        "the king ruled his kingdom with wisdom and kindness"
    ]
    
    # Create variations
    texts = []
    for template in templates:
        # Add some random variations
        for _ in range(5):
            text = template
            if random.random() > 0.5:
                text = text.replace("the", "a")
            if random.random() > 0.5:
                text = text.replace("and", "or")
            texts.append(text + " ")
    
    return texts

# Character-level Transformer model
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        x = self.char_embedding(x) + self.pos_embedding(pos)
        
        # Causal mask for autoregressive prediction
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        
        # Transformer forward
        x = self.transformer(x, mask=causal_mask)
        
        # Predict next character
        logits = self.lm_head(x)
        return logits

# Dataset class
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, char_to_idx):
        self.sequences = sequences
        self.char_to_idx = char_to_idx
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        # Convert to indices
        input_indices = torch.tensor([self.char_to_idx[ch] for ch in input_seq], dtype=torch.long)
        target_indices = torch.tensor([self.char_to_idx[ch] for ch in target_seq], dtype=torch.long)
        return input_indices, target_indices

# Training function
def train_char_model():
    """Complete training pipeline"""
    print("🚀 Starting Character-Level Language Model Training")
    
    # Generate data
    texts = generate_synthetic_text()
    print(f"📝 Generated {len(texts)} text samples")
    print(f"📝 Sample text: '{texts[0][:50]}...'")
    
    # Create dataset
    seq_length = 30 # Using a slightly longer sequence
    sequences, char_to_idx, idx_to_char, vocab_size = create_char_dataset(texts, seq_length)
    print(f"📊 Vocabulary size: {vocab_size}")
    print(f"📊 Number of training sequences: {len(sequences)}")
    print(f"📊 Characters: {list(char_to_idx.keys())}")
    
    # Create DataLoader
    dataset = CharDataset(sequences, char_to_idx)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model setup
    model = CharTransformer(
        vocab_size=vocab_size,
        d_model=64,     # Small for demo
        nhead=4,        # 4 attention heads
        num_layers=2,   # 2 transformer layers
        max_len=seq_length
    )
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    model.train()
    print("🏋️ Starting training...")
    
    for epoch in range(50):  # Small number for demo
        total_loss = 0
        num_batches = 0
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_batch)  # [batch, seq_len, vocab_size]
            
            # For efficient parallel training, we predict all tokens at once.
            # We need to reshape the logits and targets for CrossEntropyLoss.
            # Logits: [batch, seq_len, vocab] -> [batch * seq_len, vocab]
            # Target: [batch, seq_len] -> [batch * seq_len]
            loss = criterion(logits.reshape(-1, vocab_size), target_batch.reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"📈 Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}")
    
    print("✅ Training completed!")
    return model, char_to_idx, idx_to_char, vocab_size

# Generation function
def generate_text(model, char_to_idx, idx_to_char, start_text="once upon", max_length=50):
    """Generate text using the trained model"""
    model.eval()
    
    # Convert start text to indices
    current_seq = [char_to_idx.get(ch, 0) for ch in start_text]
    generated_text = start_text
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_tensor = torch.tensor(current_seq).unsqueeze(0)  # [1, current_len]
            
            # Generate next character
            logits = model(input_tensor)  # [1, current_len, vocab_size]
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # Sample next character (with some randomness)
            probs = F.softmax(last_logits / 0.8, dim=-1)  # Temperature = 0.8
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Convert back to character
            next_char = idx_to_char[next_char_idx]
            generated_text += next_char
            current_seq.append(next_char_idx)
            
            # Stop at natural ending
            if next_char in '.!?' or len(generated_text) > max_length:
                break
    
    return generated_text

# Run the complete example
if __name__ == "__main__":
    # Train model
    model, char_to_idx, idx_to_char, vocab_size = train_char_model()
    
    # Generate some text
    print("\n🎭 Generating text...")
    print("=" * 50)
    
    for start_prompt in ["once upon", "the king", "a dragon"]:
        generated = generate_text(model, char_to_idx, idx_to_char, start_prompt)
        print(f"Prompt: '{start_prompt}'")
        print(f"Generated: '{generated}'")
        print("-" * 30)
    
    # Inspect model behavior
    print("\n🔍 Model Analysis:")
    print(f"Vocabulary: {list(char_to_idx.keys())[:20]}...")  # Show first 20 chars
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test attention patterns (simplified)
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor([[char_to_idx.get(ch, 0) for ch in "once upon a time"]])
        logits = model(test_input)
        probs = F.softmax(logits[0, -1], dim=-1)
        
        # Show top predicted characters
        top_probs, top_indices = torch.topk(probs, k=5)
        print("\n📊 Top 5 next character predictions for 'once upon a time':")
        for prob, idx in zip(top_probs, top_indices):
            char = idx_to_char[idx.item()]
            print(f"  '{char}': {prob.item():.3f}")

# Run the example
train_char_model()
```

### Key Learning Points

```python
# Shape debugging - print shapes at critical points
def debug_shapes():
    batch_size, seq_len, vocab_size = 4, 10, 50
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    model = CharTransformer(vocab_size, d_model=32, nhead=4, num_layers=2)
    
    print(f"Input shape: {x.shape}")
    
    # Step through model
    char_emb = model.char_embedding(x)
    print(f"After char embedding: {char_emb.shape}")
    
    pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    pos_emb = model.pos_embedding(pos)
    print(f"Position embedding: {pos_emb.shape}")
    
    combined = char_emb + pos_emb
    print(f"Combined embeddings: {combined.shape}")
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    print(f"Causal mask shape: {causal_mask.shape}")
    
    # Forward through transformer
    transformed = model.transformer(combined, mask=causal_mask)
    print(f"After transformer: {transformed.shape}")
    
    # Language modeling head
    logits = model.lm_head(transformed)
    print(f"Final logits: {logits.shape}")

debug_shapes()
```

**Math** Cross-Reference to `./transformers_math1.md`:
> This example implements:
> - Attention (23): Self-attention within each transformer layer
> - FFN (36): Feed-forward networks in transformer layers  
> - Causal masking: Ensures autoregressive property for language modeling
> - Cross-entropy loss (2): Standard objective for next-token prediction

## 14. Appendix: Quick Mapping & Formula Cards

### Architecture → PyTorch Quick Reference

| **Architecture** | **Key Layers** | **Input Shape** | **Output Shape** | **Parameters** |
|------------------|----------------|-----------------|------------------|----------------|
| **MLP (2-layer)** | `nn.Linear` × 2 | `[B, D_in]` | `[B, D_out]` | `D_in×D_h + D_h + D_h×D_out + D_out` |
| **RNN** | `nn.RNN` | `[B, T, D_in]` | `[B, T, D_h]` | `D_in×D_h + D_h×D_h + D_h` |
| **LSTM** | `nn.LSTM` | `[B, T, D_in]` | `[B, T, D_h]` | `4×(D_in×D_h + D_h×D_h + D_h)` |
| **Transformer** | `nn.TransformerEncoder` | `[B, T, D_model]` | `[B, T, D_model]` | See Multi-Head Attention below |

### Multi-Head Attention Parameter Breakdown

```python
def count_transformer_params(d_model, nhead, num_layers):
    """Count parameters in transformer encoder"""
    
    # Per layer
    # Multi-head attention: Q, K, V projections + output projection
    mha_params = 4 * (d_model * d_model)  # W_q, W_k, W_v, W_o
    
    # Feed-forward network (usually 4x expansion)
    ffn_hidden = 4 * d_model
    ffn_params = d_model * ffn_hidden + ffn_hidden + ffn_hidden * d_model + d_model
    
    # Layer normalization (2 per layer: before MHA, before FFN)  
    ln_params = 2 * (2 * d_model)  # γ and β for each norm
    
    per_layer = mha_params + ffn_params + ln_params
    total = per_layer * num_layers
    
    return {
        'mha_per_layer': mha_params,
        'ffn_per_layer': ffn_params, 
        'ln_per_layer': ln_params,
        'per_layer': per_layer,
        'total': total
    }

# Example
params = count_transformer_params(d_model=512, nhead=8, num_layers=6)
print(f"6-layer Transformer (d_model=512): {params['total']:,} parameters")
```

### When to Use What: Decision Tree

```python
def choose_architecture():
    """
    Decision helper for architecture choice
    """
    decision_tree = """
    📊 Architecture Decision Tree:
    
    ├── Input Type?
    │   ├── Tabular/Fixed-size vectors
    │   │   └── Use: MLP (nn.Linear layers)
    │   │       └── Hidden size: 2-4x input size
    │   │
    │   ├── Sequences (text, time series)
    │   │   ├── Length < 100, need state?
    │   │   │   └── Use: LSTM/GRU (nn.LSTM, nn.GRU)
    │   │   │       └── Hidden size: 64-512
    │   │   │
    │   │   ├── Length > 100, need attention?
    │   │   │   └── Use: Transformer (nn.TransformerEncoder)
    │   │   │       └── d_model: 128-768, heads: 4-12
    │   │   │
    │   │   └── Very long sequences (>1000)?
    │   │       └── Consider: Efficient attention variants
    │   │
    │   └── Structured prediction?
    │       └── Use: Transformer with appropriate masking
    
    💡 Rules of thumb:

    - Start simple: MLP baseline for non-sequential
    - RNNs: Good for streaming/online processing
    - Transformers: Best for batch processing, parallel training
    - Always try smaller models first (faster iteration)
    """
    print(decision_tree)

choose_architecture()
```

### Essential Code → Math Equation Mapping

| **PyTorch Code** | **Math Equation** | **Reference** |
|------------------|-------------------|---------------|
| `F.softmax(scores, dim=-1)` | `softmax(z)_i = e^{z_i}/∑e^{z_j}` | transformers_math1.md (1) |
| `F.cross_entropy(logits, targets)` | `L = -∑ y_i log p_i` | transformers_math1.md (2) |
| `torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)` | `QK^T/√d_k` | transformers_math1.md (23) |
| `F.layer_norm(x, normalized_shape)` | `γ(x-μ)/√(σ²+ε) + β` | transformers_math1.md (19) |
| `F.gelu(linear(x))` | `GELU(xW + b)` | transformers_math1.md (36) |
| `clip_grad_norm_(params, max_norm)` | `g̃ = min(1, c/‖g‖)g` | transformers_math2.md (11) |
| `optimizer.step()` with AdamW | Adam update rules | transformers_math2.md (7-9) |

### Final Checklist for New PyTorch Users

```python
def pytorch_checklist():
    """
    Essential checklist for PyTorch development
    """
    checklist = """

    ✅ PyTorch Development Checklist:
    
    🔧 Setup:
    [ ] Set random seeds (torch.manual_seed, random.seed, np.random.seed)
    [ ] Choose device (CPU/GPU) and move model + data consistently
    [ ] Check PyTorch version compatibility
    
    📊 Data:
    [ ] Verify data shapes [batch, ...] throughout pipeline
    [ ] Handle variable-length sequences with padding/packing
    [ ] Check data types (Long for class indices, Float for inputs)
    [ ] Implement proper train/val/test splits
    
    🧠 Model:
    [ ] Inherit from nn.Module, implement forward()
    [ ] Initialize parameters appropriately (Xavier/Kaiming)
    [ ] Count parameters to verify model size
    [ ] Add dropout/regularization where appropriate
    
    🏋️ Training:
    [ ] Set model.train() before training, model.eval() before inference
    [ ] Clear gradients with optimizer.zero_grad()
    [ ] Use appropriate loss function (CrossEntropy for classification)
    [ ] Add gradient clipping for RNNs/deep networks
    [ ] Monitor loss convergence, not just final value
    
    🐛 Debugging:
    [ ] Print shapes at each step when debugging
    [ ] Check for NaN/inf in loss and gradients
    [ ] Verify data loading with small batches first
    [ ] Test model with dummy data before real training
    
    💾 Deployment:
    [ ] Save/load state_dict, not full model
    [ ] Store hyperparameters separately from model weights
    [ ] Use torch.no_grad() for inference
    [ ] Consider torch.jit.script for production
    """
    print(checklist)

pytorch_checklist()
```

---

**Congratulations!** 🎉 You now have a comprehensive guide to PyTorch for sequence modeling. This reference covers the essential patterns you'll use whether building MLPs, RNNs, or Transformers. Keep this handy as you build your own models!

> Remember: Start simple, debug with shapes, and always set your random seeds for reproducible results!
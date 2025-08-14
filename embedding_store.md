# LLM Weights vs Vector Stores: Knowledge Storage and Similarity Calculations

## Table of Contents

1. [Knowledge Storage Mechanisms](#part-i-knowledge-storage-mechanisms)
   - [LLM Weight-Based Knowledge Storage](#1-llm-weight-based-knowledge-storage)
   - [Vector Store Knowledge Storage](#2-vector-store-knowledge-storage)
   - [Are They the Same Concept?](#3-critical-question-are-they-the-same-concept)

2. [Mathematical Foundations](#part-ii-mathematical-foundations)
   - [Distance Calculations with Examples](#4-distance-calculations-with-concrete-examples)
   - [Cosine Similarity](#cosine-similarity)
   - [Euclidean Distance](#euclidean-distance)

3. [Relevance Across Contexts](#part-iii-relevance-across-contexts)
   - [During LLM Training](#5-during-llm-training)
   - [During LLM Inference](#6-during-llm-inference)
   - [In Vector Store Operations](#7-in-vector-store-operations)

4. [Practical Integration](#part-iv-practical-integration)
   - [Complementary Systems](#8-complementary-systems)
   - [Concrete Comparative Example](#9-concrete-comparative-example)

---

## PART I: Knowledge Storage Mechanisms

### 1. LLM Weight-Based Knowledge Storage

#### Introduction: What Are LLM Weights?

Imagine a library where instead of storing books on specific shelves, knowledge is encoded in the arrangement of millions of interconnected neurons. Large Language Models store knowledge through **distributed representations** across billions of parameters (weights). Unlike traditional databases where "cats are animals" might be stored as a discrete fact in row 1247, LLMs encode this relationship as patterns distributed across multiple layers of neural networks.

**Core Intuition**: Think of LLM weights like a vast neural web where:
- **Embedding weights** position words in semantic space (similar words cluster together)
- **Attention weights** learn which words should "pay attention" to each other
- **Feed-forward weights** transform and refine these relationships through non-linear computations

The magic happens when these three types of weights work together to generate contextual understanding.

#### Weight Computation Timeline and Lifecycle

**The Journey of a Token Through Transformer Layers**

To understand how weights store knowledge, let's follow the word "cat" through a transformer model step-by-step.

**Step 1: From Text to Numbers (Tokenization and Embedding)**

First, we need to understand how text becomes numbers that neural networks can process:

```
Text: "cat"
↓ Tokenization (convert text to numbers)
Token ID: 1247  (each word/subword gets a unique integer ID)
↓ Embedding Lookup (convert ID to vector)
Embedding vector: [0.8, 0.2, 0.1]  (learned position in semantic space)
```

**What is W_embed?**
- `W_embed` is the **embedding matrix** with shape `[vocab_size, embedding_dim]`
- For a vocabulary of 50,000 words and 768-dimensional embeddings: `W_embed` is `[50000, 768]`
- Each row represents one word's position in semantic space
- `W_embed[1247]` gives us the embedding vector for token ID 1247 ("cat")

**During Training Phase:**

1. **Embedding Weight Learning:**
   ```
   Input: Token "cat" (ID: 1247)
   Lookup: embedding_vector = W_embed[1247] = [0.8, 0.2, 0.1]
   
   How W_embed learns:
   - Initially random: W_embed[1247] = [0.1, -0.3, 0.9] (random values)
   - Through training: Gradually moves to [0.8, 0.2, 0.1] (learned optimal position)
   - Goal: Similar words (cat, dog, animal) get similar vectors
   ```
   
   **Why these specific numbers?** The exact values aren't meaningful individually. What matters is their **relationships**:
   - `cat=[0.8, 0.2, 0.1]` and `dog=[0.7, 0.3, 0.2]` are close (similar animals)
   - `cat=[0.8, 0.2, 0.1]` and `house=[0.1, 0.8, 0.3]` are distant (different concepts)

2. **Attention Weight Computation:**

   **What are Query, Key, Value?**
   Think of attention like a library search:
   - **Query (Q)**: "What am I looking for?" (current word's question)
   - **Key (K)**: "What do I offer?" (other words' advertisements)  
   - **Value (V)**: "Here's my information" (other words' actual content)

   ```
   Starting point: h_cat = [0.8, 0.2, 0.1] (from embedding layer)
   
   Query creation: Q_cat = h_cat · W_q
   - W_q is learned "question-forming" matrix [3×3 in our simple example]
   - Q_cat = [0.8, 0.2, 0.1] · W_q = [0.9, 0.3, 0.2] (cat's question vector)
   
   Key creation: K_animal = h_animal · W_k  
   - W_k is learned "advertisement" matrix [3×3]
   - K_animal = [0.7, 0.3, 0.2] · W_k = [0.8, 0.4, 0.1] (animal's key vector)
   ```

   **Why Softmax Here?**
   ```
   Attention Score: Q_cat · K_animal = [0.9, 0.3, 0.2] · [0.8, 0.4, 0.1] = 0.86
   
   With multiple words: [Q_cat · K_the, Q_cat · K_big, Q_cat · K_animal] = [0.1, 0.3, 0.86]
   
   Softmax converts to probabilities: softmax([0.1, 0.3, 0.86]) = [0.05, 0.13, 0.82]
   ```
   
   **Softmax Purpose**: Ensures attention weights sum to 1.0, creating a probability distribution over which words to focus on. Cat pays 82% attention to "animal", 13% to "big", 5% to "the".

3. **Feed-Forward Weight Computation:**

   **What happens after attention?**
   After attention, we have enriched representations. The feed-forward network (FFN) applies non-linear transformations to extract higher-level patterns.

   ```
   FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2
   
   Input: x = [0.75, 0.28, 0.15] (enriched cat representation after attention)
   
   First transformation: 
   z = W_1 · x + b_1 = [2048×768] · [0.75, 0.28, 0.15] + bias = [intermediate_vector]
   
   ReLU activation: 
   activated = ReLU(z) = max(0, z) (removes negative values, adds non-linearity)
   
   Final projection:
   output = W_2 · activated + b_2 = [768×2048] · activated + bias = [final_representation]
   ```

   **Why ReLU Here (Not Softmax)?**
   - **ReLU**: Used for **feature transformation** - decides which features to keep (positive) or discard (negative)
   - **Softmax**: Used for **probability distributions** - when we need weights that sum to 1
   - FFN uses ReLU because we're transforming features, not creating attention weights

**During Inference Phase:**
- **Weight Loading**: Pre-trained weights loaded into GPU memory as frozen parameters
- **Forward Computation**: Weights applied deterministically without updates
- **Hidden State Flow**: `token_id → embedding → layer_1 → ... → layer_n → output`

#### Hidden States and Weight Interaction

**What Are Hidden States?**

Think of hidden states as "evolving understanding" of each word as it passes through transformer layers. Like how your understanding of "bank" changes when you read "river bank" vs "savings bank", hidden states capture context-dependent meaning.

**The Complete Journey: From Static Embeddings to Rich Representations**

Let's trace how the word "cat" evolves through multiple transformer layers:

```
Input Sentence: "The big cat runs"

Layer 0 (Initial Embeddings - Static Word Meanings):
h_the = [0.1, 0.9, 0.2]    (generic "the" embedding)
h_big = [0.6, 0.1, 0.8]    (generic "big" embedding)  
h_cat = [0.8, 0.2, 0.1]    (generic "cat" embedding)
h_runs = [0.3, 0.7, 0.4]   (generic "runs" embedding)
```

At this stage, each word has its basic dictionary meaning, with no context.

```
Layer 1 (After Self-Attention - Words Start Talking):
h_cat' = Attention(h_cat, [h_the, h_big, h_cat, h_runs])

How attention weights are computed:
- Cat asks: "Who should I pay attention to?"
- Attention scores: [the: 0.05, big: 0.35, cat: 0.40, runs: 0.20]
- Weighted combination: 
  h_cat' = 0.05·h_the + 0.35·h_big + 0.40·h_cat + 0.20·h_runs
         = 0.05·[0.1,0.9,0.2] + 0.35·[0.6,0.1,0.8] + 0.40·[0.8,0.2,0.1] + 0.20·[0.3,0.7,0.4]
         = [0.005,0.045,0.01] + [0.21,0.035,0.28] + [0.32,0.08,0.04] + [0.06,0.14,0.08]
         = [0.595, 0.3, 0.41] → after normalization: [0.52, 0.37, 0.31]
```

Now "cat" understands it's a "big cat" (not just any cat).

```
Layer 6 (Deep Understanding - Rich Contextual Meaning):
h_cat'' = [0.75, 0.28, 0.15]

Through 6 layers of processing:
- "Cat" now knows it's the subject of the sentence
- It understands it's "big" (size attribute)
- It anticipates "runs" (the action it performs)
- It has absorbed semantic relationships from training data
```

**How Knowledge Emerges**

The relationship "cats are animals" emerges through this multi-layer process:

1. **Embedding Layer**: Positions "cat" and "animal" in similar regions of semantic space
2. **Attention Layers**: Learn that "cat" often appears with "animal" in training text
3. **Feed-Forward Layers**: Extract the hierarchical relationship (cat ⊂ pet ⊂ animal)
4. **Output Layer**: High probability for "animal" when predicting after "The cat is an ___"

**The Distributed Knowledge Pattern**:
- No single weight stores "cats are animals"
- The relationship emerges from the collective behavior of millions of weights
- Knowledge is implicit in the learned transformations, not explicit storage

#### Multi-Head Attention Weight Mechanics

**Weight Matrix Organization:**
```
d_model = 768, num_heads = 12, d_head = 64

W_q shape: [768, 768] organized as 12 blocks of [768, 64]
W_k shape: [768, 768] organized as 12 blocks of [768, 64]  
W_v shape: [768, 768] organized as 12 blocks of [768, 64]
```

**Head Specialization Example:**
```
Head 1: Subject-Verb Relationships
  Q_cat · K_runs = high attention score
  Learns: "cat runs", "dog sleeps", "pet plays"

Head 2: Adjective-Noun Relationships  
  Q_big · K_cat = high attention score
  Learns: "big cat", "small dog", "big house"

Head 3: Semantic Hierarchies
  Q_cat · K_animal = high attention score
  Learns: "cat is animal", "dog is pet"
```

**Knowledge Encoding Patterns:**
The relationship "cats are animals" is distributed across:
- **Embedding layer**: Positions "cat" and "animal" in similar semantic regions
- **Attention heads 3,7,11**: Learn hierarchical category relationships
- **FFN layers**: Refine the hierarchical mapping through non-linear transformations
- **Output layer**: High probability for "animal" given "cat is ___" contexts

### 2. Vector Store Knowledge Storage

#### Introduction: What Are Vector Stores?

Imagine a massive digital library where instead of organizing books by subject or author, each book is placed based on its semantic "fingerprint" - a unique numerical signature capturing its meaning. Vector databases work exactly this way: they store knowledge as **discrete embeddings** (high-dimensional numerical vectors) that can be quickly searched by similarity.

**Core Intuition**: Think of vector stores like:
- **A GPS system for meaning**: Every piece of text gets coordinates in "semantic space"
- **A similarity search engine**: Find documents that are "close" to your query in meaning
- **Explicit storage**: Unlike LLMs, knowledge is stored as retrievable vectors, not learned patterns

**Key Difference from LLMs**: While LLMs encode knowledge implicitly in weight patterns, vector stores maintain explicit, searchable representations of facts and documents.

#### How Vector Search Works: From Simple to Sophisticated

**The Fundamental Challenge**

Given a query like "What animals make good pets?", how do we find relevant documents from millions of stored vectors? This is the core challenge vector databases solve.

**Method 1: Brute Force Search (The Simple Approach)**

```
Problem: Find documents similar to query "What animals are pets?"

Step 1: Convert query to vector
query_vec = embedding_model("What animals are pets?") = [0.72, 0.31, 0.18]

Step 2: Compare against ALL stored documents
doc1: "Cats are popular pets" → vector: [0.82, 0.31, 0.15]
doc2: "Dogs make loyal companions" → vector: [0.79, 0.33, 0.18]  
doc3: "Houses need maintenance" → vector: [0.12, 0.85, 0.43]

Step 3: Calculate similarities
similarity(query, doc1) = cosine([0.72, 0.31, 0.18], [0.82, 0.31, 0.15]) = 0.94
similarity(query, doc2) = cosine([0.72, 0.31, 0.18], [0.79, 0.33, 0.18]) = 0.96
similarity(query, doc3) = cosine([0.72, 0.31, 0.18], [0.12, 0.85, 0.43]) = 0.32

Step 4: Rank results
1. doc2 (similarity: 0.96) - "Dogs make loyal companions"
2. doc1 (similarity: 0.94) - "Cats are popular pets"  
3. doc3 (similarity: 0.32) - "Houses need maintenance"

Complexity: O(n·d) where n=documents, d=dimensions
```

**Problem**: This works for small datasets, but imagine searching through 100 million documents with 768-dimensional vectors - it's too slow!

The solution: **Approximate Nearest Neighbor (ANN)** methods that trade a small amount of accuracy for massive speed improvements.

**Method 2: HNSW - The Highway System for Vectors**

Think of HNSW (Hierarchical Navigable Small World) like a highway system with multiple levels:

```
Analogy: Finding a restaurant in a city
- Level 2 (Highway): Connect major cities [NYC ←→ Chicago ←→ LA]
- Level 1 (Main roads): Connect neighborhoods within cities  
- Level 0 (Local streets): Connect every block

HNSW Vector Index:
Level 2: [doc1] ←→ [doc5] ←→ [doc12]  (sparse, long-distance connections)
Level 1: [doc1] ←→ [doc2] ←→ [doc5] ←→ [doc8] ←→ [doc12] (medium connections)
Level 0: [doc1] ←→ [doc2] ←→ [doc3] ←→ [doc4] ←→ [doc5] ←→ ... (dense, local connections)

Search Process:
1. Start at Level 2, find closest document to query
2. Navigate to similar documents at this level
3. Drop down to Level 1, continue navigation  
4. Drop to Level 0, find final closest matches

Why it's fast: O(log n) instead of O(n)
- Like using highways instead of local roads for long-distance travel
```

**Method 3: IVF - The Clustering Approach**

Think of IVF (Inverted File) like organizing a library by topic:

```
Traditional Library Organization:
Fiction → [Book1, Book2, Book3, ...]
Science → [Book10, Book11, Book12, ...]
History → [Book20, Book21, Book22, ...]

Vector Database Clustering:
Cluster 1 (Animal docs): centroid = [0.8, 0.2, 0.1]
├── "Cats are pets" → [0.82, 0.31, 0.15]
├── "Dogs love running" → [0.79, 0.33, 0.18]  
└── "Small pets need care" → [0.75, 0.35, 0.22]

Cluster 2 (House docs): centroid = [0.1, 0.8, 0.3]
├── "Big house in park" → [0.12, 0.85, 0.43]
└── "House needs cleaning" → [0.08, 0.78, 0.35]

Search Process:
1. Query: "What animals are pets?" → [0.72, 0.31, 0.18]
2. Find closest cluster centroid → Cluster 1 (animals)
3. Search only within Cluster 1 → much faster!
4. Reduced search: 3 documents instead of 5 total

Speedup: From O(n) to O(n/k) where k = number of clusters
```

**Method 4: Product Quantization - The Compression Master**

PQ is like creating a "summary" of each vector to save memory:

```
Problem: 768-dimensional vectors use lots of memory
Solution: Compress vectors while preserving similarity relationships

Original vector (768 dimensions): [0.82, 0.31, 0.15, 0.67, 0.23, ..., 0.44]
                                  │  96 dims  │  96 dims  │ ... │  96 dims  │
                                  └─ chunk 1 ──┴─ chunk 2 ──┴─────┴─ chunk 8 ─┘

Step 1: Split into 8 chunks of 96 dimensions each
chunk_1 = [0.82, 0.31, 0.15, ...]  (96 numbers)
chunk_2 = [0.67, 0.23, 0.44, ...]  (96 numbers)
...
chunk_8 = [0.33, 0.78, 0.12, ...]  (96 numbers)

Step 2: For each chunk, find the closest "prototype" from 256 learned centroids
chunk_1 closest to centroid_42 → store ID: 42
chunk_2 closest to centroid_156 → store ID: 156
...
chunk_8 closest to centroid_73 → store ID: 73

Step 3: Compressed representation
Original: 768 × 32 bits = 24,576 bits  
Compressed: 8 × 8 bits = 64 bits
Compression ratio: 384× smaller!

Trade-off: Slight accuracy loss for massive memory savings
```

#### Understanding Index Performance Trade-offs

Different indexes optimize for different priorities:

#### Vector Database Terminology and Index Implementations

**Index Performance Characteristics:**

| Index Type | Build Time | Query Time | Memory Usage | Recall |
|------------|------------|------------|--------------|--------|
| **IVF_FLAT** | O(n) | O(√n) | High (exact vectors) | High |
| **IVF_PQ** | O(n) | O(√n) | Low (compressed) | Medium |
| **HNSW** | O(n log n) | O(log n) | Medium (graph structure) | High |
| **IVF_SQ8** | O(n) | O(√n) | Medium (8-bit quantized) | Medium-High |

**Practical Example with Our Vocabulary:**
```
Document Collection:
doc1: "The cat sleeps in house" → vector_1: [0.82, 0.31, 0.15, ...]
doc2: "Big dog runs in park" → vector_2: [0.79, 0.25, 0.18, ...]
doc3: "Small pet loves plays" → vector_3: [0.75, 0.35, 0.22, ...]

IVF_FLAT Index:
- Exact distance computation, high memory usage
- Best for: High-accuracy requirements, smaller datasets

IVF_PQ Index:
- Compressed vectors, ~100x memory reduction
- Best for: Large-scale deployment, acceptable recall trade-off

HNSW Index:
- Graph navigation, logarithmic search complexity
- Best for: Real-time applications, balanced accuracy/speed
```

#### Vector Database Operations and Production Considerations

**Insertion Pipeline:**
```
1. Document Processing:
   "Cats are small animals" → Text preprocessing

2. Embedding Generation:
   Sentence transformer → [0.84, 0.29, 0.17, ..., 0.43] (768-dim)

3. Index Insertion:
   HNSW: Add node, connect to M nearest neighbors
   IVF: Assign to nearest cluster, append to inverted list
   
4. Metadata Association:
   vector_id → {text: "Cats are small animals", category: "animals", timestamp: "2024-01-15"}
```

**Query Processing with Filtering:**
```
Query: "What do pets do?" with filter: category="animals"

1. Generate query embedding: [0.72, 0.31, 0.18, ...]
2. Perform similarity search in vector index
3. Apply metadata filter to results
4. Rank by similarity score:
   - doc3: similarity=0.91, category="animals" ✓
   - doc1: similarity=0.87, category="animals" ✓
   - doc5: similarity=0.85, category="places" ✗ (filtered out)
```

**Scalability and Distributed Search:**
```
Horizontal Sharding Strategy:
Shard 1: documents 1-1M    → HNSW index on server 1
Shard 2: documents 1M-2M   → HNSW index on server 2
Shard 3: documents 2M-3M   → HNSW index on server 3

Query Distribution:
1. Send query to all shards in parallel
2. Each shard returns top-k results
3. Merge and re-rank globally
4. Return final top-k to client

Load Balancing:
- Read replicas for high-throughput search
- Write coordination for consistent updates
- Async replication for eventual consistency
```

**Dynamic Updates and Maintenance:**
```
Vector Addition:
- HNSW: Insert with probability-based level assignment
- IVF: Recompute cluster assignment, may trigger rebalancing

Index Optimization:
- Periodic rebuild for optimal performance
- Background compaction for deleted vectors
- Cluster rebalancing when data distribution changes

Version Control:
- Embedding model updates change vector dimensions
- Strategy: Maintain multiple indexes during transition
- Gradual migration from old to new embedding space
```

### 3. Critical Question: Are They the Same Concept?

**NO** - These are fundamentally different approaches:

| Aspect | LLM Weights | Vector Stores |
|--------|-------------|---------------|
| **Storage** | Distributed patterns | Discrete vectors |
| **Knowledge Access** | Generated through computation | Retrieved through search |
| **Updates** | Requires retraining | Add/remove vectors |
| **Capacity** | Limited by parameter count | Unlimited external storage |
| **Latency** | Fixed computation cost | Variable search cost |

---

## PART II: Mathematical Foundations

### 4. Distance Calculations with Concrete Examples

Using our vocabulary examples:
- "cat" vector: [0.8, 0.2, 0.1]
- "dog" vector: [0.7, 0.3, 0.2]

### Cosine Similarity

**Formula:** `cos(θ) = (A·B)/(||A||×||B||)`

**Step-by-step calculation:**

1. **Dot product:** A·B = (0.8×0.7) + (0.2×0.3) + (0.1×0.2) = 0.56 + 0.06 + 0.02 = 0.64

2. **Magnitudes:**
   - ||A|| = √(0.8² + 0.2² + 0.1²) = √(0.64 + 0.04 + 0.01) = √0.69 ≈ 0.83
   - ||B|| = √(0.7² + 0.3² + 0.2²) = √(0.49 + 0.09 + 0.04) = √0.62 ≈ 0.79

3. **Cosine similarity:** 0.64 / (0.83 × 0.79) ≈ 0.64 / 0.66 ≈ **0.97**

**Interpretation:** Values range from -1 (opposite) to 1 (identical). 0.97 indicates high similarity - "cat" and "dog" point in nearly the same direction in semantic space.

### Euclidean Distance

**Formula:** `d = √[(x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²]`

**Step-by-step calculation:**

1. **Differences:** (0.8-0.7)² + (0.2-0.3)² + (0.1-0.2)² = 0.01 + 0.01 + 0.01 = 0.03

2. **Distance:** √0.03 ≈ **0.17**

**Interpretation:** Lower values indicate closer proximity. 0.17 is small, confirming "cat" and "dog" are close in space.

**When to use each:**
- **Cosine:** When direction matters more than magnitude (text similarity, semantic relationships)
- **Euclidean:** When absolute distance matters (image features, exact matching)

---

## PART III: Relevance Across Contexts

### 5. During LLM Training

**Yes, transformers use similarity calculations internally:**

1. **Attention Mechanism:** Uses dot products (related to cosine similarity)
   ```
   Attention(Q,K,V) = softmax(QK^T/√d)V
   ```
   The QK^T operation computes similarity between query and key vectors.

2. **Gradient Flow:** Backpropagation updates weights based on similarity between predicted and actual tokens.

3. **Example during training:**
   When processing "The big cat ___", attention weights learn that "cat" tokens should attend strongly to "animal" tokens, encoded through weight updates.

### 6. During LLM Inference

**Knowledge retrieval happens through distributed computation:**

1. **Next-token prediction:** The model doesn't "search" for knowledge - it computes probability distributions over the vocabulary.

2. **Example:** For "The big cat ___"
   - Input embeddings flow through attention layers
   - Each layer refines representations using learned weight patterns
   - Final layer outputs probabilities: P("runs") = 0.3, P("sleeps") = 0.25, P("house") = 0.05

3. **No explicit similarity search** - knowledge emerges from the computational graph.

### 7. In Vector Store Operations

**Explicit similarity search for retrieval:**

1. **Query embedding:** "What do cats do?" → [0.75, 0.25, 0.15]

2. **Search process:**
   ```
   For each stored vector:
     similarity = cosine(query, stored_vector)
     if similarity > threshold: add to results
   ```

3. **RAG Pipeline:**
   - Retrieve: Find relevant documents using similarity search
   - Augment: Add retrieved content to LLM prompt
   - Generate: LLM produces response with external knowledge

---

## PART IV: Practical Integration

### 8. Complementary Systems

**How LLMs and vector stores work together:**

1. **LLM Weights Store:** General language patterns, reasoning capabilities, common knowledge
2. **Vector Stores Handle:** Specific facts, recent information, large knowledge bases

**Trade-offs:**

| Factor | LLM Weights | Vector Stores |
|--------|-------------|---------------|
| **Speed** | Fast (fixed computation) | Variable (depends on index size) |
| **Updates** | Slow (retraining required) | Fast (add/remove vectors) |
| **Accuracy** | High for trained knowledge | High for indexed content |
| **Cost** | High inference cost | Storage + search cost |

### 9. Concrete Comparative Example

**Sentence:** "Small pet loves park"

**In LLM Weights:**
- Training updates attention patterns between "small"→"pet", "pet"→"loves", "loves"→"park"
- Knowledge encoded as probability: P("loves"|"small pet") = 0.15
- Retrieved through forward pass computation

**In Vector Store:**
- Document: "Small pets love spending time in parks" → Vector: [0.45, 0.67, 0.22]
- Query: "pet activities" → Vector: [0.43, 0.65, 0.24]
- Cosine similarity: 0.98 → Document retrieved
- Knowledge accessed through explicit search

**Key Difference:** LLM generates the relationship through learned patterns, while vector store retrieves pre-existing representations.

---

## Conclusion

LLM weights and vector stores represent **complementary approaches** to knowledge storage:

- **LLM weights** excel at capturing general patterns and generating contextual responses
- **Vector stores** excel at providing specific, updatable information through retrieval

Modern AI systems often combine both: vector stores provide specific facts while LLM weights provide reasoning and language generation capabilities. The choice of similarity metric (cosine vs Euclidean) depends on whether you're measuring semantic similarity or exact proximity.